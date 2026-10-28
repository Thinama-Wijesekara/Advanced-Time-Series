"""Store-sales ingestion, preprocessing, model refitting, and forecasting."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("STORE_SALES_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
MASTER_DIR = CLEAN_DIR / "train_cleaned_data"
REFERENCE_DIR = DATA_DIR / "reference"
BACKUPS_DIR = DATA_DIR / "backups"
FORECASTS_DIR = DATA_DIR / "forecasts"
UPLOADS_DIR = RAW_DIR / "uploads"
MASTER_RAW_PATH = MASTER_DIR / "daily_total_sales.csv"
SEED_RAW_PATH = RAW_DIR / "daily_total_sales_initial.csv"
CLEAN_PATH = MASTER_DIR / "daily_total_sales_clean.csv"
HOLIDAYS_PATH = REFERENCE_DIR / "holidays_events.csv"
STORES_PATH = REFERENCE_DIR / "stores.csv"
CONFIG_PATH = PROJECT_ROOT / "pipeline" / "config.json"
MODEL_ROOT = Path(
    os.environ.get(
        "STORE_SALES_MODEL_DIR",
        PROJECT_ROOT / "models" / "selected_models",
    )
)

LARGE_ZERO_SHOT = "chronos_t5_large_zero_shot_adjusted"
LARGE_ENSEMBLE = "ensemble_chronos_t5_large_gbm_w0.7_2"
LEVEL_CALENDAR = "level_calendar_no_model"
CHRONOS_MODEL_ID = "amazon/chronos-t5-large"
REFIT_MODEL_NAMES = (LARGE_ENSEMBLE, LEVEL_CALENDAR)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _ensure_data_directories() -> None:
    for directory in (
        RAW_DIR,
        UPLOADS_DIR,
        MASTER_DIR,
        REFERENCE_DIR,
        BACKUPS_DIR,
        FORECASTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not MASTER_RAW_PATH.exists():
        if not SEED_RAW_PATH.is_file():
            raise FileNotFoundError(
                f"Seed daily-total data not found: {SEED_RAW_PATH}."
            )
        shutil.copy2(SEED_RAW_PATH, MASTER_RAW_PATH)
    if not HOLIDAYS_PATH.exists():
        raise FileNotFoundError(f"Holiday reference file not found: {HOLIDAYS_PATH}.")


def _parse_dates(values: pd.Series, column_name: str) -> pd.Series:
    dates = pd.to_datetime(
        values.astype("string"),
        format="mixed",
        errors="coerce",
        utc=True,
    )
    if dates.isna().any():
        bad_rows = (np.flatnonzero(dates.isna().to_numpy()) + 2).tolist()
        raise ValueError(
            f"Upload has unparseable {column_name} values on CSV row(s): {bad_rows}."
        )
    return dates.dt.tz_convert(None).dt.normalize()


def _normalize_sales(upload_df: pd.DataFrame) -> pd.Series:
    sales = pd.to_numeric(upload_df["sales"], errors="coerce")
    invalid_sales = sales.isna() | ~np.isfinite(sales.to_numpy(dtype="float64"))
    if invalid_sales.any():
        bad_rows = (np.flatnonzero(invalid_sales.to_numpy()) + 2).tolist()
        raise ValueError(
            f"Upload has non-numeric or missing sales values on CSV row(s): {bad_rows}."
        )
    if (sales < 0).any():
        bad_rows = (np.flatnonzero((sales < 0).to_numpy()) + 2).tolist()
        raise ValueError(f"Upload has negative sales values on CSV row(s): {bad_rows}.")
    return sales.astype("float64")


def _normalize_daily_dataframe(upload_df: pd.DataFrame) -> pd.DataFrame:
    date_column = "timestamp_utc" if "timestamp_utc" in upload_df.columns else "date"
    missing = {date_column, "sales"}.difference(upload_df.columns)
    if missing:
        raise ValueError(
            "Daily-total upload must contain columns 'timestamp_utc' (or 'date') "
            "and 'sales'; "
            f"missing: {', '.join(sorted(missing))}."
        )
    if upload_df.empty:
        raise ValueError("Upload contains no data rows.")

    normalized = pd.DataFrame(
        {
            "timestamp_utc": _parse_dates(upload_df[date_column], date_column),
            "sales": _normalize_sales(upload_df),
        }
    )
    if normalized["timestamp_utc"].duplicated().any():
        duplicates = (
            normalized.loc[
                normalized["timestamp_utc"].duplicated(keep=False), "timestamp_utc"
            ]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Daily-total upload has duplicate dates: {', '.join(duplicates[:10])}."
        )
    return normalized


def _expected_store_count() -> int:
    if not STORES_PATH.is_file():
        raise FileNotFoundError(f"Store reference file not found: {STORES_PATH}.")
    stores = pd.read_csv(STORES_PATH)
    if "store_nbr" not in stores.columns or stores.empty:
        raise ValueError(
            f"Store reference file must contain store_nbr values: {STORES_PATH}."
        )
    count = int(stores["store_nbr"].nunique(dropna=True))
    if count != len(stores) or count == 0:
        raise ValueError(
            f"Store reference has missing or duplicate store_nbr values: {STORES_PATH}."
        )
    return count


def _validate_upload_dates(dates: pd.Series, history_end: pd.Timestamp) -> None:
    unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    if unique_dates.empty:
        raise ValueError("Upload has no complete dates to process.")
    latest_start = history_end + pd.Timedelta(days=1)
    if unique_dates[0] > latest_start:
        raise ValueError(
            f"Upload starts on {unique_dates[0]:%Y-%m-%d}, leaving a gap after "
            f"the history ending {history_end:%Y-%m-%d}; it must start no later "
            f"than {latest_start:%Y-%m-%d}."
        )
    expected_dates = pd.date_range(unique_dates[0], unique_dates[-1], freq="D")
    missing_dates = expected_dates.difference(unique_dates)
    missing_dates = missing_dates[
        ~((missing_dates.month == 12) & (missing_dates.day == 25))
    ]
    if not missing_dates.empty:
        missing_text = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:10])
        raise ValueError(f"Upload has missing calendar dates: {missing_text}.")


def _load_master_for_upload() -> tuple[pd.DataFrame, pd.Timestamp]:
    if not MASTER_RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Daily-total master file not found: {MASTER_RAW_PATH}."
        )
    try:
        master = pd.read_csv(MASTER_RAW_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not read master data '{MASTER_RAW_PATH}': {exc}"
        ) from exc
    normalized = _normalize_daily_dataframe(master)
    if normalized.empty:
        raise ValueError(f"Daily-total master is empty: {MASTER_RAW_PATH}.")
    normalized = normalized.sort_values("timestamp_utc").reset_index(drop=True)
    return normalized, pd.Timestamp(normalized["timestamp_utc"].max())


def _normalize_store_upload(upload_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"date", "store_nbr", "family", "sales"}
    missing = required_columns.difference(upload_df.columns)
    if missing:
        raise ValueError(
            "Store-level upload must contain columns 'date', 'store_nbr', "
            "'family', and 'sales'; "
            f"missing: {', '.join(sorted(missing))}."
        )
    if upload_df.empty:
        raise ValueError("Upload contains no data rows.")

    normalized = pd.DataFrame(
        {
            "date": _parse_dates(upload_df["date"], "date"),
            "store_nbr": upload_df["store_nbr"],
            "family": upload_df["family"],
            "sales": _normalize_sales(upload_df),
        }
    )
    if normalized["store_nbr"].isna().any():
        raise ValueError("Store-level upload has missing store_nbr values.")
    if normalized["family"].isna().any():
        raise ValueError("Store-level upload has missing family values.")

    _, history_end = _load_master_for_upload()
    expected_count = _expected_store_count()
    store_counts = normalized.groupby("date")["store_nbr"].nunique()
    last_date = pd.Timestamp(store_counts.index.max())

    if int(store_counts.loc[last_date]) < expected_count:
        print(
            f"WARNING: Dropping incomplete final upload date {last_date:%Y-%m-%d}: "
            f"{int(store_counts.loc[last_date])} stores found; expected {expected_count}."
        )
        normalized = normalized[normalized["date"] != last_date].copy()
        store_counts = store_counts.drop(last_date)

    invalid_counts = store_counts[store_counts != expected_count]
    if not invalid_counts.empty:
        details = ", ".join(
            f"{date:%Y-%m-%d}: {count} stores"
            for date, count in invalid_counts.head(10).items()
        )
        raise ValueError(
            f"Upload dates must each contain {expected_count} unique stores. "
            f"Invalid date(s): {details}."
        )

    _validate_upload_dates(normalized["date"], history_end)
    return normalized


def ingest_upload(path: str | Path) -> pd.DataFrame:
    """Validate and archive an untouched store-level or daily-total CSV."""
    upload_path = Path(path)
    if not upload_path.is_file():
        raise FileNotFoundError(f"Upload file not found: {upload_path}")
    try:
        source_df = pd.read_csv(upload_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read upload CSV '{upload_path}': {exc}") from exc

    if {"date", "store_nbr", "family", "sales"}.issubset(source_df.columns):
        upload_df = _normalize_store_upload(source_df)
    elif {"timestamp_utc", "sales"}.issubset(source_df.columns):
        upload_df = _normalize_daily_dataframe(source_df)
        _, history_end = _load_master_for_upload()
        _validate_upload_dates(upload_df["timestamp_utc"], history_end)
    else:
        raise ValueError(
            "Upload must contain either date, store_nbr, family, sales or "
            "timestamp_utc, sales columns."
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = UPLOADS_DIR / f"upload_{_timestamp()}_{upload_path.name}"
    shutil.copy2(upload_path, archive_path)
    return upload_df


def aggregate_upload(upload_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate store records and fill absent Christmas dates in either format."""
    if "store_nbr" not in upload_df.columns:
        daily = _normalize_daily_dataframe(upload_df)
        dates = pd.DatetimeIndex(daily["timestamp_utc"])
        calendar = pd.date_range(dates.min(), dates.max(), freq="D")
        missing_dates = calendar.difference(dates)
        christmas_dates = missing_dates[
            (missing_dates.month == 12) & (missing_dates.day == 25)
        ]
        if len(christmas_dates):
            print(
                "Filled absent upload date(s) with 0.0 sales: "
                + ", ".join(date.strftime("%Y-%m-%d") for date in christmas_dates)
            )
            daily = pd.concat(
                [
                    daily,
                    pd.DataFrame(
                        {"timestamp_utc": christmas_dates, "sales": 0.0}
                    ),
                ],
                ignore_index=True,
            )
        return daily.sort_values("timestamp_utc").reset_index(drop=True)

    daily = (
        upload_df.groupby("date", as_index=False, sort=True)["sales"]
        .sum()
        .rename(columns={"date": "timestamp_utc"})
    )
    dates = pd.DatetimeIndex(daily["timestamp_utc"])
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    missing_dates = calendar.difference(dates)
    if len(missing_dates):
        print(
            "Filled absent upload date(s) with 0.0 sales: "
            + ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates)
        )
    return (
        daily.set_index("timestamp_utc")
        .reindex(calendar, fill_value=0.0)
        .rename_axis("timestamp_utc")
        .reset_index()
    )


def update_master_raw(upload_df: pd.DataFrame) -> pd.DataFrame:
    """Back up and merge daily totals; uploaded values replace matching dates."""
    normalized_upload = _normalize_daily_dataframe(upload_df)
    _ensure_data_directories()
    normalized_master, _ = _load_master_for_upload()

    backup_path = BACKUPS_DIR / f"daily_total_sales_master_{_timestamp()}.csv"
    shutil.copy2(MASTER_RAW_PATH, backup_path)

    merged = pd.concat([normalized_master, normalized_upload], ignore_index=True)
    merged = (
        merged.drop_duplicates("timestamp_utc", keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    merged.to_csv(MASTER_RAW_PATH, index=False, date_format="%Y-%m-%d")
    return merged


def preprocess() -> pd.DataFrame:
    """Apply the existing full-series closure preprocessing and save the clean file."""
    _ensure_data_directories()
    df = pd.read_csv(MASTER_RAW_PATH, parse_dates=["timestamp_utc"])
    hol = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])

    # 2. full daily index, no duplicates
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").set_index("timestamp_utc")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.index.name = "timestamp_utc"
    df["sales"] = df["sales"].astype(float)

    # 3. closure days: far below the local 7-day median
    local_med = df["sales"].rolling(7, center=True, min_periods=3).median()
    df["is_closure"] = df["sales"] < 0.2 * local_med
    nat_days = set(hol[(hol.locale == "National") & (~hol.transferred)].date)

    # 4. impute closures with same weekday +/-1 and +/-2 weeks (non-closure days only)
    clean = df["sales"].copy()
    clean[df.is_closure] = np.nan
    for d in df.index[df.is_closure]:
        neigh = [d + pd.Timedelta(weeks=k) for k in (-2, -1, 1, 2)]
        vals = [clean.get(n) for n in neigh if n in clean.index and not pd.isna(clean.get(n))]
        clean[d] = np.mean(vals)
    df["sales_raw"] = df["sales"]
    df["sales"] = clean

    # 5. flags for error analysis only
    df["is_national_holiday"] = df.index.isin(nat_days)
    df["dow"] = df.index.dayofweek

    # 6. check + save
    assert df["sales"].isna().sum() == 0 and (df["sales"] > 0).all()
    clean_df = df.reset_index()

    if CLEAN_PATH.exists():
        backup_path = BACKUPS_DIR / f"daily_total_sales_clean_{_timestamp()}.csv"
        shutil.copy2(CLEAN_PATH, backup_path)
    clean_df.to_csv(CLEAN_PATH, index=False)
    return clean_df


def _pipeline_settings() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _read_model_info(model_name: str) -> dict[str, Any]:
    info_path = MODEL_ROOT / model_name / "model_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {info_path}")
    with info_path.open(encoding="utf-8-sig") as info_file:
        return json.load(info_file)


def Xcal(dts: pd.DatetimeIndex) -> np.ndarray:
    dow, dom, mend = dts.dayofweek.to_numpy(), dts.day.to_numpy(), np.asarray(dts.is_month_end, dtype=float)
    return np.column_stack([(dow == d) for d in range(7)] + [(dom == d) for d in range(1, 32)] + [mend]).astype(float)


def _fit_effect_model(
    ly: np.ndarray,
    xcal: np.ndarray,
    upto: int,
    alpha: float,
    rolling_window: int,
    min_periods: int,
) -> Ridge:
    s = pd.Series(ly[:upto])
    dev = (
        s - s.rolling(rolling_window, center=True, min_periods=min_periods).mean()
    ).to_numpy()
    rows = np.where(~np.isnan(dev))[0]
    return Ridge(alpha=alpha).fit(xcal[rows], dev[rows])


def level_calendar(
    ly: np.ndarray,
    eff: np.ndarray,
    origins: range | list[int],
    horizon: int,
    lookback: int,
) -> np.ndarray:
    return np.stack(
        [
            np.expm1(
                (ly[s - lookback : s] - eff[s - lookback : s]).mean()
                + eff[s : s + horizon]
            )
            for s in origins
        ]
    )


def make_rows(
    orgs: range | list[int],
    ly: np.ndarray,
    dow: np.ndarray,
    dom: np.ndarray,
    mend: np.ndarray,
    horizon: int,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notebook feature construction for each forecast start and horizon day."""
    F, T, L = [], [], []
    for s in orgs:
        lvl, lvl7 = (
            ly[s - lookback : s].mean(),
            ly[s - 7 : s].mean() - ly[s - lookback : s].mean(),
        )
        for h in range(horizon):
            t, m = s + h, h // 7 + 1
            F.append([dow[t], dom[t], mend[t], h, lvl7, ly[t - 7 * m] - lvl, ly[t - 7 * (m + 1)] - lvl])
            T.append(ly[t] - lvl)
            L.append(lvl)
    return np.array(F, float), np.array(T), np.array(L)


def ctx_windows(
    series: np.ndarray,
    split: str,
    origins: dict[str, list[int]],
    context: int,
) -> np.ndarray:
    return np.stack([series[s - context : s] for s in origins[split]]).astype("float32")


def back(P: np.ndarray, eff_w: np.ndarray) -> np.ndarray:
    return np.expm1(np.log1p(P) + eff_w)


@torch.no_grad()
def predict_all(
    pipe: Any,
    X: np.ndarray,
    horizon: int,
    num_samples: int,
    seed: int,
    batch_size: int,
    desc: str = "",
) -> np.ndarray:
    torch.manual_seed(seed)
    out = []
    for i in tqdm(range(0, len(X), batch_size), desc=desc):
        samples = pipe.predict(
            torch.as_tensor(X[i : i + batch_size]),
            horizon,
            num_samples=num_samples,
        )
        out.append(samples.float().median(dim=1).values.cpu().numpy())
    return np.concatenate(out)


def refit() -> dict[str, Path]:
    """Refit and version Ridge and GBM models without overwriting prior versions."""
    settings = _pipeline_settings()
    _ensure_data_directories()
    df = pd.read_csv(CLEAN_PATH, parse_dates=["timestamp_utc"])
    dts = pd.DatetimeIndex(df["timestamp_utc"])
    y = df["sales"].to_numpy("float64")
    ly = np.log1p(y)
    n = len(y)
    train_end, valid_end = int(n * settings["train_frac"]), int(n * (settings["train_frac"] + settings["valid_frac"]))
    xcal = Xcal(dts)

    dow, dom, mend = dts.dayofweek.to_numpy(), dts.day.to_numpy(), np.asarray(dts.is_month_end, dtype=float)

    ridge_train = _fit_effect_model(
        ly,
        xcal,
        train_end,
        settings["ridge_alpha"],
        settings["calendar_rolling_window"],
        settings["calendar_min_periods"],
    )
    ridge_full = _fit_effect_model(
        ly,
        xcal,
        valid_end,
        settings["ridge_alpha"],
        settings["calendar_rolling_window"],
        settings["calendar_min_periods"],
    )
    Ftr, Ttr, _ = make_rows(
        range(35, train_end - settings["horizon"] + 1),
        ly,
        dow,
        dom,
        mend,
        settings["horizon"],
        settings["level_lookback"],
    )
    gbm = HistGradientBoostingRegressor(
        max_iter=settings["gbm_max_iter"],
        learning_rate=settings["gbm_learning_rate"],
        max_depth=settings["gbm_max_depth"],
        l2_regularization=settings["gbm_l2_regularization"],
        categorical_features=[0, 1],
        random_state=settings["gbm_random_state"],
    ).fit(Ftr, Ttr)

    version_name = _timestamp()
    version_paths: dict[str, Path] = {}
    staging_paths: dict[str, Path] = {}
    for model_name in REFIT_MODEL_NAMES:
        info = _read_model_info(model_name)
        expected_context = (
            settings["context"] if model_name == LARGE_ENSEMBLE else settings["level_lookback"]
        )
        if int(info["context"]) != expected_context or int(info["horizon"]) != settings["horizon"]:
            raise ValueError(
                f"Notebook forecast settings do not match {model_name}/model_info.json."
            )
        version_dir = MODEL_ROOT / model_name / "versions" / version_name
        staging_dir = version_dir.with_name(f".tmp_{version_name}")
        staging_dir.mkdir(parents=True, exist_ok=False)
        if model_name == LARGE_ENSEMBLE:
            joblib.dump(ridge_train, staging_dir / "calendar_ridge_train.joblib")
            joblib.dump(gbm, staging_dir / "gbm.joblib")
        joblib.dump(ridge_full, staging_dir / "calendar_ridge_train_valid.joblib")
        staging_paths[model_name] = staging_dir

    for model_name, staging_dir in staging_paths.items():
        version_dir = staging_dir.with_name(version_name)
        staging_dir.rename(version_dir)
        version_paths[model_name] = version_dir

    return version_paths


def _newest_model_dir(model_name: str) -> Path:
    model_dir = MODEL_ROOT / model_name
    versions_dir = model_dir / "versions"
    if versions_dir.is_dir():
        versions = sorted(
            path
            for path in versions_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".tmp_")
        )
        if versions:
            return versions[-1]
    return model_dir


def _future_gbm_rows(
    y: np.ndarray,
    future_dates: pd.DatetimeIndex,
    horizon: int,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    ly = np.log1p(y.astype("float64"))
    dow = future_dates.dayofweek.to_numpy()
    dom = future_dates.day.to_numpy()
    mend = np.asarray(future_dates.is_month_end, dtype=float)
    origin = len(y)
    lvl = ly[origin - lookback : origin].mean()
    lvl7 = ly[origin - 7 : origin].mean() - lvl
    F, L = [], []
    for h in range(horizon):
        t, m = origin + h, h // 7 + 1
        F.append(
            [
                dow[h],
                dom[h],
                mend[h],
                h,
                lvl7,
                ly[t - 7 * m] - lvl,
                ly[t - 7 * (m + 1)] - lvl,
            ]
        )
        L.append(lvl)
    return np.array(F, float), np.array(L)


def _load_chronos_pipelines(
    model_infos: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str, Any]:
    from chronos import ChronosPipeline
    from transformers import set_seed

    settings = _pipeline_settings()
    set_seed(settings["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    pipelines: dict[str, Any] = {}
    for info in model_infos.values():
        if "huggingface_name" not in info:
            continue
        model_id = info["huggingface_name"]
        if model_id != CHRONOS_MODEL_ID:
            raise ValueError(
                f"Only {CHRONOS_MODEL_ID} is allowed; found model ID {model_id!r}."
            )
        if model_id not in pipelines:
            pipelines[model_id] = ChronosPipeline.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
            )
    return pipelines, device, dtype


def forecast(
    pipelines: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> Path:
    """Forecast all three models, or only the requested selected-model identifier."""
    settings = _pipeline_settings()
    _ensure_data_directories()
    df = pd.read_csv(CLEAN_PATH, parse_dates=["timestamp_utc"])
    dates = pd.DatetimeIndex(df["timestamp_utc"])
    y = df["sales"].to_numpy("float64")
    if model_id == LEVEL_CALENDAR:
        future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=settings["horizon"], freq="D")
        xcal_history = Xcal(dates)
        xcal_future = Xcal(future_dates)
        baseline_model_dir = _newest_model_dir(LEVEL_CALENDAR)
        baseline_ridge = joblib.load(baseline_model_dir / "calendar_ridge_train_valid.joblib")
        lookback = int(settings["level_lookback"])
        ly = np.log1p(y)
        baseline_level = (ly[-lookback:] - baseline_ridge.predict(xcal_history)[-lookback:]).mean()
        result = pd.DataFrame({"date": future_dates, "model": LEVEL_CALENDAR, "forecast": np.expm1(baseline_level + baseline_ridge.predict(xcal_future))})
        if not np.isfinite(result["forecast"].to_numpy(dtype="float64")).all():
            raise ValueError("Forecast output contains NaN or infinite values.")
        forecast_path = FORECASTS_DIR / f"forecast_{_timestamp()}.csv"
        result.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")
        return forecast_path

    if len(y) < settings["context"]:
        raise ValueError(
            f"Need at least {settings['context']} clean daily rows; found {len(y)}."
        )

    model_infos = {
        LARGE_ZERO_SHOT: _read_model_info(LARGE_ZERO_SHOT),
        LARGE_ENSEMBLE: _read_model_info(LARGE_ENSEMBLE),
        LEVEL_CALENDAR: _read_model_info(LEVEL_CALENDAR),
    }
    selected_names = {LARGE_ZERO_SHOT, LARGE_ENSEMBLE, LEVEL_CALENDAR}
    if model_id is not None and model_id not in selected_names:
        raise ValueError(f"Unknown selected model: {model_id}.")
    for model_name, info in model_infos.items():
        expected_context = (
            settings["context"]
            if model_name in (LARGE_ZERO_SHOT, LARGE_ENSEMBLE)
            else settings["level_lookback"]
        )
        if int(info["context"]) != expected_context:
            raise ValueError(
                f"{model_name}/model_info.json context must be {expected_context}."
            )
        if int(info["horizon"]) != int(settings["horizon"]):
            raise ValueError(
                f"{model_name}/model_info.json horizon must match pipeline/config.json."
            )
        if int(info.get("num_samples", -1)) != int(settings["num_samples"]):
            raise ValueError(
                f"{model_name}/model_info.json num_samples must match pipeline/config.json."
            )
        if int(info.get("seed", -1)) != int(settings["seed"]):
            raise ValueError(
                f"{model_name}/model_info.json seed must match pipeline/config.json."
            )
        if info.get("sample_aggregation") != settings["sample_aggregation"]:
            raise ValueError(
                f"{model_name}/model_info.json sample_aggregation must match "
                "pipeline/config.json."
            )
        if model_name in (LARGE_ZERO_SHOT, LARGE_ENSEMBLE):
            if info.get("huggingface_name") != CHRONOS_MODEL_ID:
                raise ValueError(
                    f"{model_name}/model_info.json must reference only {CHRONOS_MODEL_ID}."
                )
        elif "huggingface_name" in info:
            raise ValueError(
                f"{model_name}/model_info.json must not reference a Chronos model."
            )
    needs_chronos = bool(set(model_infos) & {LARGE_ZERO_SHOT, LARGE_ENSEMBLE})
    if needs_chronos and pipelines is None:
        pipelines, _, _ = _load_chronos_pipelines(model_infos)

    zero_info = model_infos[LARGE_ZERO_SHOT] if LARGE_ZERO_SHOT in model_infos else model_infos[LARGE_ENSEMBLE] if LARGE_ENSEMBLE in model_infos else None
    zero_model_id = zero_info["huggingface_name"]
    zero_context = int(zero_info["context"])
    zero_horizon = int(zero_info["horizon"])
    zero_samples = int(zero_info.get("num_samples", settings["num_samples"]))
    zero_seed = int(zero_info.get("seed", settings["seed"]))

    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=zero_horizon, freq="D")
    xcal_history = Xcal(dates)
    xcal_future = Xcal(pd.DatetimeIndex(future_dates))
    ly = np.log1p(y)
    zero_effect_estimator = joblib.load(
        _newest_model_dir(LARGE_ENSEMBLE) / "calendar_ridge_train.joblib"
    )
    cal_history = zero_effect_estimator.predict(xcal_history)
    cal_future = zero_effect_estimator.predict(xcal_future)
    y_adjusted = np.expm1(ly - cal_history)
    context = ctx_windows(
        y_adjusted,
        "future",
        {"future": [len(y_adjusted)]},
        zero_context,
    )
    large_adjusted = back(
        predict_all(
            pipelines[zero_model_id],
            context,
            zero_horizon,
            zero_samples,
            zero_seed,
            settings["prediction_batch_size"],
        ),
        cal_future.reshape(1, -1),
    )

    output_frames = [
        pd.DataFrame(
            {
                "date": future_dates,
                "model": LARGE_ZERO_SHOT,
                "forecast": large_adjusted.reshape(-1),
            }
        )
    ]
    info = model_infos[LARGE_ENSEMBLE]
    context_length = int(info["context"])
    horizon = int(info["horizon"])
    num_samples = int(info.get("num_samples", settings["num_samples"]))
    seed = int(info.get("seed", settings["seed"]))
    if context_length != zero_context or horizon != zero_horizon:
        raise ValueError("Large ensemble settings do not match the large zero-shot model settings.")

    model_dir = _newest_model_dir(LARGE_ENSEMBLE)
    gbm = joblib.load(model_dir / "gbm.joblib")
    ridge = joblib.load(model_dir / "calendar_ridge_train.joblib")
    ensemble_cal_future = ridge.predict(xcal_future)
    ensemble_cal_history = ridge.predict(xcal_history)
    adjusted_history = np.expm1(ly - ensemble_cal_history)
    ensemble_context = ctx_windows(
        adjusted_history,
        "future",
        {"future": [len(adjusted_history)]},
        context_length,
    )
    chronos_adjusted = large_adjusted.copy()
    F, L = _future_gbm_rows(
        y,
        pd.DatetimeIndex(future_dates),
        horizon,
        settings["level_lookback"],
    )
    gbm_forecast = np.expm1(gbm.predict(F) + L).reshape(1, -1)
    weights = info["ensemble_weights"]
    ensemble_forecast = (
        float(weights["chronos"]) * chronos_adjusted
        + float(weights["gbm"]) * gbm_forecast
    )
    output_frames.append(
        pd.DataFrame(
            {
                "date": future_dates,
                "model": LARGE_ENSEMBLE,
                "forecast": ensemble_forecast.reshape(-1),
            }
        )
    )

    baseline_model_dir = _newest_model_dir(LEVEL_CALENDAR)
    baseline_ridge = joblib.load(baseline_model_dir / "calendar_ridge_train_valid.joblib")
    baseline_history_effect = baseline_ridge.predict(xcal_history)
    baseline_future_effect = baseline_ridge.predict(xcal_future)
    lookback = int(settings["level_lookback"])
    baseline_level = (ly[-lookback:] - baseline_history_effect[-lookback:]).mean()
    baseline_forecast = np.expm1(baseline_level + baseline_future_effect)
    output_frames.append(
        pd.DataFrame(
            {
                "date": future_dates,
                "model": LEVEL_CALENDAR,
                "forecast": baseline_forecast,
            }
        )
    )

    forecast_df = pd.concat(output_frames, ignore_index=True)
    expected_models = {LARGE_ZERO_SHOT, LARGE_ENSEMBLE, LEVEL_CALENDAR}
    if model_id is not None:
        forecast_df = forecast_df.loc[forecast_df["model"] == model_id].copy()
        expected_models = {model_id}
    if set(forecast_df["model"]) != expected_models:
        raise ValueError("Forecast output does not contain exactly the three selected models.")
    model_counts = forecast_df.groupby("model").size()
    if not model_counts.eq(int(settings["horizon"])).all():
        raise ValueError(
            f"Each selected model must produce {settings['horizon']} forecast rows."
        )
    if not np.isfinite(forecast_df["forecast"].to_numpy(dtype="float64")).all():
        raise ValueError("Forecast output contains NaN or infinite values.")

    timestamp = _timestamp()
    forecast_path = FORECASTS_DIR / f"forecast_{timestamp}.csv"
    forecast_df.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")
    forecast_df.to_csv(FORECASTS_DIR / "latest_forecast.csv", index=False, date_format="%Y-%m-%d")
    return forecast_path
