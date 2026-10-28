from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, insert, select, text

from db.config import engine
from db.models import CleanSeries, DailyTotal, DataVersion, SalesHistory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = Path("C:/Users/ThinamaWijesekara/OneDrive - ZILLIONe/Desktop/Advanced Time series/data/raw/train.csv")
DAILY_BASELINE = PROJECT_ROOT / "data/raw/daily_total_sales_initial.csv"
CLEAN_BASELINE = PROJECT_ROOT / "data/clean/train_cleaned_data/daily_total_sales_clean.csv"


def _parse_store_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"History file not found: {path}")
    frame = pd.read_csv(path, usecols=["date", "store_nbr", "family", "sales", "onpromotion"])
    frame["date"] = pd.to_datetime(frame["date"], format="mixed", errors="coerce").dt.normalize()
    for column in ("sales", "store_nbr", "onpromotion"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "sales", "store_nbr", "onpromotion"]].isna().any().any():
        raise ValueError("train.csv contains missing or unparseable required values.")
    frame["family"] = frame["family"].astype("string")
    if len(frame) < 3_000_000:
        raise ValueError(f"History has {len(frame):,} rows; at least 3,000,000 are required.")
    if frame["date"].max() < pd.Timestamp("2017-08-15"):
        raise ValueError(f"History ends on {frame['date'].max():%Y-%m-%d}; last date must be on or after 2017-08-15.")
    return frame


def _copy_dataframe(connection, table: str, columns: list[str], frame: pd.DataFrame, chunk_size: int = 100_000) -> None:
    raw = connection.connection
    column_sql = ", ".join(columns)
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start:start + chunk_size]
        buffer = io.StringIO(newline="")
        chunk.to_csv(buffer, index=False, header=False, columns=columns, quoting=csv.QUOTE_MINIMAL, na_rep="")
        buffer.seek(0)
        with raw.cursor() as cursor:
            with cursor.copy(f"COPY {table} ({column_sql}) FROM STDIN WITH (FORMAT CSV)") as copy:
                copy.write(buffer.read())


def _check_daily_totals(frame: pd.DataFrame) -> tuple[int, float]:
    baseline = pd.read_csv(DAILY_BASELINE, parse_dates=["timestamp_utc"])
    actual = frame.groupby("date", sort=True)["sales"].sum().rename("sales").reset_index()
    merged = baseline.rename(columns={"timestamp_utc": "date"}).merge(actual, on="date", suffixes=("_baseline", "_actual"))
    if len(merged) != 1_688 or len(actual) != 1_688:
        raise ValueError(f"Expected 1,688 daily dates; baseline has {len(baseline)}, database total has {len(actual)}.")
    if merged["sales_baseline"].isna().any() or merged["sales_actual"].isna().any():
        raise ValueError("Daily-total validation found missing dates.")
    differences = (merged["sales_actual"] - merged["sales_baseline"]).abs()
    allowed = differences <= 0.01
    relative = differences / merged["sales_baseline"].abs().clip(lower=1e-12)
    allowed |= relative <= 1e-9
    if not allowed.all():
        raise ValueError(f"Daily totals differ from baseline on {(~allowed).sum()} dates; maximum difference {differences.max():.9f}.")
    return len(actual), float(differences.max())


def _build_clean_series(engine, daily: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="store-sales-db-") as work:
        work = Path(work)
        data_dir = work / "data"
        master_dir = data_dir / "clean" / "train_cleaned_data"
        raw_dir = data_dir / "raw"
        reference_dir = data_dir / "reference"
        master_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        reference_dir.mkdir(parents=True)
        for name in ("holidays_events.csv", "stores.csv"):
            source = PROJECT_ROOT / "data" / "reference" / name
            (reference_dir / name).write_bytes(source.read_bytes())
        daily_file = master_dir / "daily_total_sales.csv"
        daily.rename(columns={"date": "timestamp_utc"}).to_csv(daily_file, index=False, date_format="%Y-%m-%d")
        script = "from pipeline.functions import preprocess; preprocess()"
        env = os.environ.copy()
        env["STORE_SALES_DATA_DIR"] = str(data_dir)
        subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env, check=True)
        # Copy the required reference files and seed in the temporary data layout.
        clean_file = master_dir / "daily_total_sales_clean.csv"
        if not clean_file.exists():
            raise RuntimeError("Existing preprocess() did not write its clean series.")
        clean = pd.read_csv(clean_file, parse_dates=["timestamp_utc"])
    clean = clean.rename(columns={"timestamp_utc": "date", "sales_raw": "sales_raw", "is_closure": "is_closure"})
    return clean[["date", "sales", "sales_raw", "is_closure"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-load store-sales history")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    started = time.monotonic()
    frame = _parse_store_history(args.path)
    eng = engine()
    daily = frame.groupby("date", as_index=False)["sales"].sum().rename(columns={"date": "date"})
    calendar = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(calendar, fill_value=0.0).rename_axis("date").reset_index()
    with eng.begin() as connection:
        connection.execute(delete(SalesHistory))
        connection.execute(delete(DailyTotal))
        connection.execute(delete(CleanSeries))
        history = frame.rename(columns={"date": "date"})[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
        history["store_nbr"] = history["store_nbr"].astype(int)
        history["onpromotion"] = history["onpromotion"].astype(int)
        history["sales"] = history["sales"].round(3)
        _copy_dataframe(connection, "sales_history", ["date", "store_nbr", "family", "sales", "onpromotion"], history)
        daily["sales"] = daily["sales"].round(9)
        _copy_dataframe(connection, "daily_totals", ["date", "sales"], daily)
    count, max_difference = _check_daily_totals(daily)
    print(f"Loaded {len(frame):,} history rows and {count:,} daily totals; maximum baseline difference {max_difference:.9f}.")

    clean = _build_clean_series(eng, daily)
    baseline = pd.read_csv(CLEAN_BASELINE, parse_dates=["timestamp_utc"])
    compare = clean.merge(baseline.rename(columns={"timestamp_utc": "date"}), on="date", suffixes=("", "_baseline"))
    if len(compare) != len(baseline):
        raise ValueError("Clean series and baseline do not contain the same dates.")
    clean_diff = (compare["sales"] - compare["sales_baseline"]).abs()
    closure_match = compare["is_closure"] == compare["is_closure_baseline"]
    if clean_diff.max() > 0.001 or not closure_match.all():
        raise ValueError("Clean series differs from the pipeline baseline.")
    with eng.begin() as connection:
        connection.execute(delete(DataVersion))
        connection.execute(delete(CleanSeries))
        _copy_dataframe(connection, "clean_series", ["date", "sales", "sales_raw", "is_closure"], clean)
        connection.execute(insert(DataVersion).values(data_version=1, last_date=daily["date"].max(), rows=len(daily), run_id=None))
    print(f"Clean series verified; baseline closure dates match. Load time: {time.monotonic() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
