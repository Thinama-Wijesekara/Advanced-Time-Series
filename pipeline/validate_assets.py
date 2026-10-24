"""Validate the lightweight inputs and model bundles needed by the pipeline."""

from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTED_MODELS = {
    "chronos_t5_large_zero_shot_adjusted": ("zero-shot", 512),
    "ensemble_chronos_t5_large_gbm_w0.7_2": ("ensemble", 512),
    "level_calendar_no_model": ("calendar baseline", 28),
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
    holidays_path = ROOT / "data" / "reference" / "holidays_events.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing initial daily-total CSV: {train_path}")
    if not {"timestamp_utc", "sales"}.issubset(_csv_columns(train_path)):
        raise ValueError("Initial daily-total CSV must contain timestamp_utc and sales columns.")
    _validate_daily_history(train_path)
    if not holidays_path.is_file():
        raise FileNotFoundError(f"Missing holiday reference CSV: {holidays_path}")
    if not {"date", "type", "locale", "transferred"}.issubset(_csv_columns(holidays_path)):
        raise ValueError("Holiday CSV is missing required calendar feature columns.")

    for folder, (method, context) in SELECTED_MODELS.items():
        model_dir = ROOT / "models" / "selected_models" / folder
        info_path = model_dir / "model_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing model metadata: {info_path}")
        with info_path.open(encoding="utf-8-sig") as info_file:
            info = json.load(info_file)
        if info.get("method") != method or int(info.get("context", -1)) != context:
            raise ValueError(f"Unexpected method or context in {info_path}.")
        if int(info.get("horizon", -1)) != 16:
            raise ValueError(f"Expected a 16-day horizon in {info_path}.")

    ensemble_dir = ROOT / "models" / "selected_models" / "ensemble_chronos_t5_large_gbm_w0.7_2"
    for filename in ("calendar_ridge_train.joblib", "calendar_ridge_train_valid.joblib", "gbm.joblib"):
        if not (ensemble_dir / filename).is_file():
            raise FileNotFoundError(f"Missing ensemble artifact: {ensemble_dir / filename}")

    baseline_dir = ROOT / "models" / "selected_models" / "level_calendar_no_model"
    for filename in ("calendar_ridge_train.joblib", "calendar_ridge_train_valid.joblib"):
        if not (baseline_dir / filename).is_file():
            raise FileNotFoundError(f"Missing baseline artifact: {baseline_dir / filename}")


if __name__ == "__main__":
    validate_assets()
    print("Training data, holiday reference, and three selected model bundles are valid.")
