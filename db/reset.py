from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from db.config import database_url, engine
from db.models import Batch, CleanSeries, DailyTotal, DataVersion, Forecast, Run, SalesNew
from db.load_history import CLEAN_BASELINE, DAILY_BASELINE, _copy_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset database runtime data to the baseline")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        answer = input("Reset all new data and forecasts? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Reset cancelled.")
            return 0
    backup_dir = PROJECT_ROOT / "runtime_backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"store_sales_{stamp}.dump"
    pg_dump = shutil.which("pg_dump")
    if pg_dump:
        subprocess.run(["pg_dump", "--format=custom", f"--dbname={database_url()}", "--file", str(backup_file)], check=True)
        print(f"Database backup created at {backup_file}.")
    else:
        print("pg_dump is not available; no database backup was created.")

    daily = pd.read_csv(DAILY_BASELINE)
    daily = daily.rename(columns={"timestamp_utc": "date"})
    clean = pd.read_csv(CLEAN_BASELINE).rename(columns={"timestamp_utc": "date"})
    clean["sales"] = pd.to_numeric(clean["sales"])
    clean["sales_raw"] = pd.to_numeric(clean["sales_raw"])
    clean["is_closure"] = clean["is_closure"].astype(bool)
    eng = engine()
    with eng.begin() as connection:
        connection.execute(delete(Forecast))
        connection.execute(delete(SalesNew))
        connection.execute(delete(Run))
        connection.execute(delete(Batch))
        connection.execute(delete(DataVersion).where(DataVersion.data_version != 1))
        connection.execute(delete(DailyTotal))
        connection.execute(delete(CleanSeries))
        _copy_dataframe(connection, "daily_totals", ["date", "sales"], daily[["date", "sales"]])
        _copy_dataframe(connection, "clean_series", ["date", "sales", "sales_raw", "is_closure"], clean[["date", "sales", "sales_raw", "is_closure"]])
    print("Reset complete; baseline daily_totals and clean_series restored.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
