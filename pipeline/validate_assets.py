"""Validate the lightweight inputs and model bundles needed by the pipeline."""

from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTED_MODELS = {
    "chronos_t5_large_zero_shot_adjusted": ("zero-shot", 512, True),
    "ensemble_chronos_t5_large_gbm_w0.7_2": ("ensemble", 512, True),
    "level_calendar_no_model": ("calendar baseline", 28, False),
}


def _csv_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as data_file:
        header = next(csv.reader(data_file), None)
    if header is None:
        raise ValueError(f"CSV is empty: {path}")
    return set(header)


def _validate_daily_history(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as data_file:
        rows = csv.DictReader(data_file)
        previous_date = None
        count = 0
        for row in rows:
            try:
                value = row["timestamp_utc"]
                try:
                    current_date = date.fromisoformat(value)
                except ValueError:
                    current_date = datetime.strptime(value, "%m/%d/%Y").date()
                sales = float(row["sales"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid date or sales value in {path}: {row}") from exc
            if not math.isfinite(sales) or sales < 0:
                raise ValueError(f"Sales must be finite and non-negative in {path}: {row}")
            if previous_date is not None and current_date != previous_date + timedelta(days=1):
                raise ValueError(f"Daily seed history has a missing or unordered date before {current_date}.")
            previous_date = current_date
            count += 1
    if count < 512:
        raise ValueError(f"Daily seed history has {count} rows; at least 512 are required.")


def validate_assets() -> None:
    config_path = ROOT / "pipeline" / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if int(config.get("horizon", -1)) != 16 or int(config.get("context", -1)) != 512:
        raise ValueError("Pipeline config must use the selected 512-day context and 16-day horizon.")

    train_path = ROOT / "data" / "raw" / "daily_total_sales_initial.csv"
    master_path = ROOT / "data" / "clean" / "train_cleaned_data" / "daily_total_sales.csv"
    clean_path = ROOT / "data" / "clean" / "train_cleaned_data" / "daily_total_sales_clean.csv"
    holidays_path = ROOT / "data" / "reference" / "holidays_events.csv"
    stores_path = ROOT / "data" / "reference" / "stores.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing initial daily-total CSV: {train_path}")
    if not {"timestamp_utc", "sales"}.issubset(_csv_columns(train_path)):
        raise ValueError("Initial daily-total CSV must contain timestamp_utc and sales columns.")
    _validate_daily_history(train_path)
    if not master_path.is_file():
        raise FileNotFoundError(f"Missing tracked daily-total master CSV: {master_path}")
    _validate_daily_history(master_path)
    if not clean_path.is_file():
        raise FileNotFoundError(f"Missing tracked cleaned daily-total CSV: {clean_path}")
    clean_columns = _csv_columns(clean_path)
    if not {
        "timestamp_utc",
        "sales",
        "sales_raw",
        "is_closure",
        "is_national_holiday",
        "dow",
    }.issubset(clean_columns):
        raise ValueError(f"Cleaned history is missing required preprocessing columns: {clean_path}")
    _validate_daily_history(clean_path)
    if not holidays_path.is_file():
        raise FileNotFoundError(f"Missing holiday reference CSV: {holidays_path}")
    if not {"date", "type", "locale", "transferred"}.issubset(_csv_columns(holidays_path)):
        raise ValueError("Holiday CSV is missing required calendar feature columns.")
    if not stores_path.is_file():
        raise FileNotFoundError(f"Missing store reference CSV: {stores_path}")
    store_columns = _csv_columns(stores_path)
    if "store_nbr" not in store_columns:
        raise ValueError("Store reference CSV must contain store_nbr.")
    with stores_path.open(newline="", encoding="utf-8-sig") as stores_file:
        stores = [row["store_nbr"] for row in csv.DictReader(stores_file)]
    if not stores or len(stores) != len(set(stores)):
        raise ValueError("Store reference CSV must contain unique store_nbr values.")

    model_root = ROOT / "models" / "selected_models"
    actual_models = {
        path.name for path in model_root.iterdir() if path.is_dir()
    }
    if actual_models != set(SELECTED_MODELS):
        raise ValueError(
            f"Expected exactly the selected model folders {sorted(SELECTED_MODELS)}; "
            f"found {sorted(actual_models)}."
        )

    required_sample_count = int(config.get("num_samples", -1))
    required_seed = int(config.get("seed", -1))
    sample_aggregation = config.get("sample_aggregation")
    if required_sample_count != 20 or required_seed != 42 or sample_aggregation != "median":
        raise ValueError(
            "Pipeline config must use 20 samples, seed 42, and median sample aggregation."
        )

    for folder, (method, context, uses_chronos) in SELECTED_MODELS.items():
        model_dir = model_root / folder
        info_path = model_dir / "model_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing model metadata: {info_path}")
        with info_path.open(encoding="utf-8-sig") as info_file:
            info = json.load(info_file)
        if info.get("method") != method or int(info.get("context", -1)) != context:
            raise ValueError(f"Unexpected method or context in {info_path}.")
        if (
            int(info.get("horizon", -1)) != int(config["horizon"])
            or int(info.get("num_samples", -1)) != required_sample_count
            or int(info.get("seed", -1)) != required_seed
            or info.get("sample_aggregation") != sample_aggregation
        ):
            raise ValueError(f"Forecast settings in {info_path} do not match pipeline/config.json.")
        if uses_chronos:
            if info.get("huggingface_name") != "amazon/chronos-t5-large":
                raise ValueError(
                    f"Only amazon/chronos-t5-large may be loaded by {info_path}."
                )
        elif "huggingface_name" in info:
            raise ValueError(f"Calendar baseline must not reference a Chronos model: {info_path}.")

    ensemble_dir = model_root / "ensemble_chronos_t5_large_gbm_w0.7_2"
    for filename in ("calendar_ridge_train.joblib", "calendar_ridge_train_valid.joblib", "gbm.joblib"):
        if not (ensemble_dir / filename).is_file():
            raise FileNotFoundError(f"Missing ensemble artifact: {ensemble_dir / filename}")

    baseline_dir = model_root / "level_calendar_no_model"
    for filename in ("calendar_ridge_train.joblib", "calendar_ridge_train_valid.joblib"):
        if not (baseline_dir / filename).is_file():
            raise FileNotFoundError(f"Missing baseline artifact: {baseline_dir / filename}")


if __name__ == "__main__":
    validate_assets()
    print("Training data, holiday reference, and three selected model bundles are valid.")
