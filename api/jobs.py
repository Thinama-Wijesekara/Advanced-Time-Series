"""Runtime isolation and serialized background jobs for the local API."""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api import publishing


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "runtime"
RUNTIME_DATA_DIR = RUNTIME_DIR / "data"
RUNTIME_MODEL_DIR = RUNTIME_DIR / "models" / "selected_models"
INCOMING_DIR = RUNTIME_DIR / "incoming"
JOBS_PATH = RUNTIME_DIR / "jobs.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
STEP_NAMES = (
    "Validate and archive upload",
    "Aggregate sales by date",
    "Merge into full history",
    "Clean the full daily series",
    "Refit calendar and GBM models",
    "Forecast the next 16 days",
)

_runtime_lock = threading.Lock()
_functions: Any | None = None
_chronos_lock = threading.Lock()
_chronos_pipelines: dict[str, Any] | None = None


def _copy_source_tree(source: Path, destination: Path, *, models: bool) -> None:
    if destination.exists():
        return

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        if models:
            ignored.update(
                name
                for name in names
                if name == "versions" or name.startswith(".tmp_")
            )
        else:
            ignored.update(
                name
                for name in names
                if name in {"train.csv", "uploads", "backups"}
                or (name.startswith("forecast_") and name.endswith(".csv"))
            )
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def ensure_runtime() -> None:
    """Create ignored runtime copies without copying the excluded store-level train file."""
    with _runtime_lock:
        _copy_source_tree(REPO_ROOT / "data", RUNTIME_DATA_DIR, models=False)
        _copy_source_tree(
            REPO_ROOT / "models" / "selected_models",
            RUNTIME_MODEL_DIR,
            models=True,
        )
        INCOMING_DIR.mkdir(parents=True, exist_ok=True)
        (RUNTIME_DATA_DIR / "raw" / "uploads").mkdir(parents=True, exist_ok=True)
        (RUNTIME_DATA_DIR / "backups").mkdir(parents=True, exist_ok=True)
        (RUNTIME_DATA_DIR / "forecasts").mkdir(parents=True, exist_ok=True)
        (RUNTIME_DIR / "screenshots").mkdir(parents=True, exist_ok=True)
        os.environ["STORE_SALES_DATA_DIR"] = str(RUNTIME_DATA_DIR)
        os.environ["STORE_SALES_MODEL_DIR"] = str(RUNTIME_MODEL_DIR)


def pipeline_functions() -> Any:
    global _functions
    ensure_runtime()
    if _functions is None:
        pipeline_path = str(REPO_ROOT / "pipeline")
        if pipeline_path not in sys.path:
            sys.path.insert(0, pipeline_path)
        _functions = importlib.import_module("functions")
    return _functions


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_job_id: str | None = None
        self._reservation = False
        self._state: dict[str, Any] = {
            "jobs": [],
            "latest_successful_upload": None,
            "last_forecast_created_at": None,
        }

    def load(self) -> None:
        ensure_runtime()
        with self._lock:
            if JOBS_PATH.is_file():
                try:
                    self._state = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Could not read runtime job state: {exc}") from exc
            self._state.setdefault("jobs", [])
            self._state.setdefault("latest_successful_upload", None)
            self._state.setdefault("last_forecast_created_at", None)
            for job in self._state["jobs"]:
                if job.get("status") in {"queued", "running"}:
                    job.update(
                        status="failed",
                        error="The server restarted before this job finished.",
                        current_step_name=job.get("current_step_name") or "Interrupted",
                    )
            self._persist_locked()

    def _persist_locked(self) -> None:
        temporary = JOBS_PATH.with_name(f".jobs-{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            for attempt in range(6):
                try:
                    os.replace(temporary, JOBS_PATH)
                    return
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (2**attempt))
        finally:
            temporary.unlink(missing_ok=True)

    def reserve_upload(self) -> bool:
        with self._lock:
            if self._reservation or self._active_job_id is not None:
                return False
            self._reservation = True
            return True

    def release_reservation(self) -> None:
        with self._lock:
            self._reservation = False

    def is_running(self) -> bool:
        with self._lock:
            return self._reservation or self._active_job_id is not None

    def enqueue(self, job: dict[str, Any], upload_path: Path) -> str:
        with self._lock:
            if not self._reservation or self._active_job_id is not None:
                raise RuntimeError("Upload reservation was lost before job submission.")
            self._reservation = False
            job_id = str(job["job_id"])
            self._active_job_id = job_id
            self._state["jobs"].insert(0, job)
            self._state["jobs"] = self._state["jobs"][:100]
            try:
                self._persist_locked()
            except Exception:
                self._active_job_id = None
                self._state["jobs"] = [
                    saved
                    for saved in self._state["jobs"]
                    if saved.get("job_id") != job_id
                ]
                raise
            worker = threading.Thread(
                target=self._run_job,
                args=(job_id, upload_path),
                name=f"store-sales-job-{job_id[:8]}",
                daemon=True,
            )
            try:
                worker.start()
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = f"Could not start background job: {exc}"
                job["finished_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                self._active_job_id = None
                self._persist_locked()
                raise
            return job_id

    def _find_job_locked(self, job_id: str) -> dict[str, Any] | None:
        return next(
            (job for job in self._state["jobs"] if job.get("job_id") == job_id),
            None,
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._find_job_locked(job_id)
            return json.loads(json.dumps(job)) if job is not None else None

    def list_uploads(self) -> list[dict[str, Any]]:
        fields = (
            "job_id",
            "created_at",
            "original_file_name",
            "rows",
            "days_added",
            "first_date_added",
            "last_date_added",
            "status",
            "publish_status",
            "publish_message",
            "error",
        )
        with self._lock:
            return [
                {field: job.get(field) for field in fields}
                for job in self._state["jobs"]
            ]

    def latest_new_range(self) -> dict[str, str] | None:
        with self._lock:
            value = self._state.get("latest_successful_upload")
            return dict(value) if value else None

    def last_forecast_created_at(self) -> str | None:
        with self._lock:
            return self._state.get("last_forecast_created_at")

    def _set_step(self, job_id: str, number: int, status: str) -> None:
        with self._lock:
            job = self._find_job_locked(job_id)
            if job is None:
                raise RuntimeError(f"Job state disappeared for {job_id}.")
            job["status"] = "running"
            job["current_step"] = number
            job["current_step_name"] = STEP_NAMES[number - 1]
            job["steps"][number - 1]["status"] = status
            if status == "running":
                job["steps"][number - 1]["started_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
            self._persist_locked()

    def _finish_step(self, job_id: str, number: int, started: float) -> None:
        with self._lock:
            job = self._find_job_locked(job_id)
            if job is None:
                raise RuntimeError(f"Job state disappeared for {job_id}.")
            job["steps"][number - 1]["status"] = "done"
            job["steps"][number - 1]["seconds"] = round(time.perf_counter() - started, 3)
            self._persist_locked()

    def _run_job(self, job_id: str, upload_path: Path) -> None:
        latest_path: Path | None = None
        previous_forecast: bytes | None = None
        try:
            functions = pipeline_functions()
            latest_path = functions.FORECASTS_DIR / "latest_forecast.csv"
            previous_forecast = (
                latest_path.read_bytes() if latest_path.is_file() else None
            )
            self._run_pipeline_steps(job_id, upload_path, functions)
            settings = publishing.publish_settings(REPO_ROOT)
            outcome = publishing.PublishOutcome("disabled")
            if settings.error:
                outcome = publishing.PublishOutcome("failed", settings.error)
            elif settings.enabled:
                with self._lock:
                    job = self._find_job_locked(job_id)
                    if job is not None:
                        job["publish_status"] = "publishing"
                        job["publish_message"] = (
                            "Publishing derived data to the public GitHub repository "
                            "and starting its pipeline."
                        )
                        self._persist_locked()
                try:
                    outcome = publishing.publish_runtime_outputs(
                        REPO_ROOT, RUNTIME_DATA_DIR
                    )
                except Exception as exc:
                    LOGGER.error(
                        "Optional GitHub publishing failed (%s); local results remain available.",
                        type(exc).__name__,
                    )
                    outcome = publishing.PublishOutcome(
                        "failed",
                        "GitHub publishing failed unexpectedly. Local results remain "
                        "available; inspect the app configuration and retry.",
                    )
            with self._lock:
                job = self._find_job_locked(job_id)
                if job is None:
                    raise RuntimeError(f"Job state disappeared for {job_id}.")
                job["status"] = "done"
                job["current_step"] = 6
                job["current_step_name"] = STEP_NAMES[5]
                job["publish_status"] = outcome.status
                job["publish_message"] = outcome.message
                job["finished_at"] = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                self._state["latest_successful_upload"] = (
                    {
                        "first_date": job["first_date_added"],
                        "last_date": job["last_date_added"],
                    }
                    if job["days_added"]
                    else None
                )
                self._state["last_forecast_created_at"] = job["forecast_created_at"]
                self._persist_locked()
        except Exception as exc:
            LOGGER.exception("Store-sales pipeline job %s failed", job_id)
            if latest_path is not None and previous_forecast is not None:
                temporary = latest_path.with_name(f".latest-{job_id}.tmp")
                try:
                    temporary.write_bytes(previous_forecast)
                    os.replace(temporary, latest_path)
                except OSError:
                    LOGGER.exception("Could not restore the previous latest forecast")
            with self._lock:
                job = self._find_job_locked(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["finished_at"] = datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    )
                    self._persist_locked()
        finally:
            upload_path.unlink(missing_ok=True)
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _run_pipeline_steps(self, job_id: str, upload_path: Path, functions: Any) -> None:
        def run_step(number: int, action: Any) -> Any:
            self._set_step(job_id, number, "running")
            started = time.perf_counter()
            try:
                result = action()
            except Exception:
                with self._lock:
                    job = self._find_job_locked(job_id)
                    if job is not None:
                        job["steps"][number - 1]["status"] = "failed"
                        job["steps"][number - 1]["seconds"] = round(
                            time.perf_counter() - started, 3
                        )
                        self._persist_locked()
                raise
            self._finish_step(job_id, number, started)
            return result

        uploaded = run_step(1, lambda: self._ingest(upload_path, job_id, functions))
        daily_totals = run_step(2, lambda: functions.aggregate_upload(uploaded))
        run_step(3, lambda: functions.update_master_raw(daily_totals))
        run_step(4, functions.preprocess)
        run_step(5, functions.refit)
        forecast_path = run_step(
            6,
            lambda: functions.forecast(pipelines=self._load_chronos_once(functions)),
        )
        forecast_stamp = datetime.strptime(
            forecast_path.stem.removeprefix("forecast_"), "%Y%m%d_%H%M%S_%f"
        ).astimezone().isoformat(timespec="seconds")
        with self._lock:
            job = self._find_job_locked(job_id)
            if job is not None:
                job["forecast_created_at"] = forecast_stamp

    @staticmethod
    def _ingest(upload_path: Path, job_id: str, functions: Any) -> pd.DataFrame:
        if upload_path.suffix.lower() == ".csv":
            return functions.ingest_upload(upload_path)

        source_df = pd.read_excel(upload_path, sheet_name=0, engine="openpyxl")
        temporary_csv = INCOMING_DIR / f"{job_id}.xlsx-converted.csv"
        archives_before = set(functions.UPLOADS_DIR.glob("upload_*"))
        original_archive: Path | None = None
        staging_archive: Path | None = None
        try:
            source_df.to_csv(temporary_csv, index=False)
            normalized = functions.ingest_upload(temporary_csv)
            new_archives = set(functions.UPLOADS_DIR.glob("upload_*")) - archives_before
            functions.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            original_archive = (
                functions.UPLOADS_DIR
                / f"upload_{functions._timestamp()}_{job_id}.xlsx"
            )
            staging_archive = original_archive.with_name(
                f".{original_archive.name}.tmp"
            )
            shutil.copyfile(upload_path, staging_archive)
            os.replace(staging_archive, original_archive)
            for archive in new_archives:
                archive.unlink(missing_ok=True)
            return normalized
        except Exception:
            if original_archive is not None:
                original_archive.unlink(missing_ok=True)
            if staging_archive is not None:
                staging_archive.unlink(missing_ok=True)
            for archive in set(functions.UPLOADS_DIR.glob("upload_*")) - archives_before:
                archive.unlink(missing_ok=True)
            raise
        finally:
            temporary_csv.unlink(missing_ok=True)

    def _load_chronos_once(self, functions: Any) -> dict[str, Any]:
        global _chronos_pipelines
        with _chronos_lock:
            if _chronos_pipelines is None:
                infos = {
                    model_name: functions._read_model_info(model_name)
                    for model_name in (
                        functions.LARGE_ZERO_SHOT,
                        functions.LARGE_ENSEMBLE,
                        functions.LEVEL_CALENDAR,
                    )
                }
                _chronos_pipelines, _, _ = functions._load_chronos_pipelines(infos)
            return _chronos_pipelines


job_manager = JobManager()


def initialize() -> None:
    ensure_runtime()
    pipeline_functions()
    job_manager.load()
    settings = publishing.publish_settings(REPO_ROOT)
    if settings.error:
        LOGGER.warning("Optional publishing configuration: %s", settings.error)


def validate_uploaded_file(path: Path, extension: str) -> tuple[pd.DataFrame, int]:
    functions = pipeline_functions()
    try:
        if extension == ".csv":
            source_df = pd.read_csv(path)
        else:
            source_df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except Exception as exc:
        if extension == ".csv":
            raise ValueError(f"Could not read upload CSV '{path}': {exc}") from exc
        raise ValueError(f"Could not read upload Excel file: {exc}") from exc

    if not {"date", "store_nbr", "family", "sales"}.issubset(source_df.columns):
        functions._normalize_store_upload(source_df)
    normalized = functions._normalize_store_upload(source_df)
    master_dates = pd.to_datetime(
        pd.read_csv(functions.MASTER_RAW_PATH, usecols=["timestamp_utc"])[
            "timestamp_utc"
        ],
        format="mixed",
    ).dt.normalize()
    if normalized.empty:
        new_dates = pd.DatetimeIndex([])
    else:
        upload_calendar = pd.date_range(
            normalized["date"].min(), normalized["date"].max(), freq="D"
        )
        added_dates = upload_calendar.difference(pd.DatetimeIndex(master_dates))
        new_dates = added_dates
    metadata = {
        "rows": len(source_df),
        "days_added": len(new_dates),
        "first_date_added": new_dates.min().strftime("%Y-%m-%d")
        if len(new_dates)
        else None,
        "last_date_added": new_dates.max().strftime("%Y-%m-%d")
        if len(new_dates)
        else None,
    }
    return normalized, metadata


def build_job(
    original_file_name: str,
    extension: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "job_id": uuid.uuid4().hex,
        "created_at": now,
        "original_file_name": original_file_name,
        "extension": extension,
        **metadata,
        "status": "queued",
        "current_step": 0,
        "current_step_name": "Waiting to start",
        "error": None,
        "publish_status": "not_started",
        "publish_message": None,
        "forecast_created_at": None,
        "steps": [
            {
                "number": index,
                "name": name,
                "status": "waiting",
                "seconds": None,
            }
            for index, name in enumerate(STEP_NAMES, 1)
        ],
    }
