"""Command-line runner for the store-sales data pipeline."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import functions


def _run_step(number: int, name: str, action: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    try:
        result = action()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(
            f"Step {number}: Failed ({name}, {elapsed:.2f}s): {exc}",
            file=sys.stderr,
        )
        raise
    elapsed = time.perf_counter() - started
    print(f"Step {number}: Done ({name}, {elapsed:.2f}s)")
    return result


def run_pipeline(upload_path: str | Path | None = None) -> Path:
    print(f"Pipeline started: {datetime.now().isoformat(timespec='seconds')}")
    if upload_path is None:
        print("Step 1: Skipped (no upload provided)")
        print("Step 2: Skipped (no upload provided)")
        print("Step 3: Skipped (no upload provided)")
    else:
        uploaded = _run_step(
            1,
            "validate and archive upload",
            lambda: functions.ingest_upload(upload_path),
        )
        daily_totals = _run_step(
            2,
            "aggregate upload by date",
            lambda: functions.aggregate_upload(uploaded),
        )
        _run_step(
            3,
            "merge upload into master history",
            lambda: functions.update_master_raw(daily_totals),
        )

    _run_step(4, "clean full daily series", functions.preprocess)
    _run_step(5, "refit and version models", functions.refit)
    return _run_step(6, "forecast", functions.forecast)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print("Usage: python data_pipeline.py [path/to/upload.csv]", file=sys.stderr)
        return 2
    upload_path = Path(args[0]) if args else None
    try:
        run_pipeline(upload_path)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
