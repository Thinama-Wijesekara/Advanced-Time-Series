"""Reset local working data to the repository baseline after making a backup."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from api import jobs


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def reset_runtime(*, clear_forecast: bool, confirmed: bool) -> Path:
    if not confirmed:
        raise ValueError("Reset requires --confirm; no runtime files were changed.")
    jobs.ensure_runtime()
    data_dir = jobs.RUNTIME_DATA_DIR.resolve()
    runtime_dir = jobs.RUNTIME_DIR.resolve()
    if not _is_within(data_dir, runtime_dir) or data_dir == runtime_dir:
        raise ValueError(
            "Reset is limited to a data directory inside runtime/. "
            "The configured external data directory was left unchanged."
        )

    baseline = jobs.REPO_ROOT / "data"
    required = (
        baseline / "raw" / "daily_total_sales_initial.csv",
        baseline / "clean" / "train_cleaned_data" / "daily_total_sales.csv",
        baseline / "clean" / "train_cleaned_data" / "daily_total_sales_clean.csv",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot reset; required tracked baseline files are missing: "
            + ", ".join(str(path) for path in missing)
        )

    backup_root = runtime_dir / "runtime_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = backup_root / f"data_before_reset_{stamp}"
    if data_dir.exists():
        shutil.copytree(data_dir, backup)
    else:
        backup.mkdir()

    previous_forecasts = backup / "forecasts"
    staging = runtime_dir / f".data_reset_{stamp}"
    displaced = runtime_dir / f".data_before_reset_{stamp}"
    jobs._copy_source_tree(baseline, staging, models=False)
    forecast_dir = staging / "forecasts"
    if clear_forecast:
        if forecast_dir.exists():
            shutil.rmtree(forecast_dir)
    elif previous_forecasts.is_dir():
        if forecast_dir.exists():
            shutil.rmtree(forecast_dir)
        shutil.copytree(previous_forecasts, forecast_dir)

    os.replace(data_dir, displaced)
    try:
        os.replace(staging, data_dir)
    except Exception:
        os.replace(displaced, data_dir)
        raise
    shutil.rmtree(displaced)

    jobs.STATE_DIR.mkdir(parents=True, exist_ok=True)
    jobs.atomic_json(jobs.MANIFEST_PATH, {"uploads": []})
    jobs.atomic_json(jobs.JOBS_PATH, {"jobs": []})
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back up and reset runtime data to the tracked repository baseline."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="confirm replacing runtime/data (a backup is created first)",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="also clear the existing runtime forecast",
    )
    arguments = parser.parse_args()
    try:
        backup = reset_runtime(
            clear_forecast=arguments.forecast,
            confirmed=arguments.confirm,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Runtime data reset to the tracked baseline. Backup: {backup}")


if __name__ == "__main__":
    main()
