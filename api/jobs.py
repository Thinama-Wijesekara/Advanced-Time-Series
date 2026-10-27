"""Runtime isolation, file inventory, and serialized pipeline jobs."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
_configured_data = os.environ.get("STORE_SALES_DATA_DIR")
RUNTIME_DIR = Path(
    os.environ.get(
        "STORE_SALES_RUNTIME_DIR",
        str(Path(_configured_data).parent if _configured_data else REPO_ROOT / "runtime"),
    )
).resolve()
RUNTIME_DATA_DIR = Path(
    _configured_data or RUNTIME_DIR / "data"
).resolve()
RUNTIME_MODEL_DIR = Path(
    os.environ.get("STORE_SALES_MODEL_DIR", RUNTIME_DIR / "models" / "selected_models")
).resolve()
INCOMING_DIR = RUNTIME_DIR / "incoming"
STATE_DIR = RUNTIME_DIR / "state"
JOBS_PATH = STATE_DIR / "jobs.json"
MANIFEST_PATH = STATE_DIR / "manifest.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
WATCH_INTERVAL_SECONDS = 5
WATCH_DEBOUNCE_SECONDS = 15
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

    def ignore(_: str, names: list[str]) -> set[str]:
        if models:
            return {
                name
                for name in names
                if name == "versions" or name.startswith(".tmp_")
            }
        return {
            name
            for name in names
            if name in {"train.csv", "test.csv", "uploads", "backups"}
            or (name.startswith("forecast_") and name.endswith(".csv"))
        }

    shutil.copytree(source, destination, ignore=ignore)


def ensure_runtime() -> None:
    """Create runtime copies unless caller-provided data/model paths are configured."""
    with _runtime_lock:
        if not _configured_data:
            _copy_source_tree(REPO_ROOT / "data", RUNTIME_DATA_DIR, models=False)
        if not os.environ.get("STORE_SALES_MODEL_DIR"):
            _copy_source_tree(
                REPO_ROOT / "models" / "selected_models",
                RUNTIME_MODEL_DIR,
                models=True,
            )
        for directory in (
            RUNTIME_DATA_DIR / "raw" / "uploads",
            RUNTIME_DATA_DIR / "clean" / "train_cleaned_data",
            RUNTIME_DATA_DIR / "backups",
            RUNTIME_DATA_DIR / "forecasts",
            INCOMING_DIR,
            STATE_DIR,
            RUNTIME_DIR / "screenshots",
            RUNTIME_MODEL_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("STORE_SALES_DATA_DIR", str(RUNTIME_DATA_DIR))
        os.environ.setdefault("STORE_SALES_MODEL_DIR", str(RUNTIME_MODEL_DIR))


def pipeline_functions() -> Any:
    global _functions
    ensure_runtime()
    if _functions is None:
        pipeline_path = str(REPO_ROOT / "pipeline")
        if pipeline_path not in sys.path:
            sys.path.insert(0, pipeline_path)
        _functions = importlib.import_module("functions")
    return _functions


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_original_name(name: str) -> str:
    leaf = name.replace("\\", "/").rsplit("/", 1)[-1]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
    return sanitized[:120] or "upload"


def _file_signature(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (path.name, stat.st_mtime_ns, stat.st_size)


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _upload_id() -> str:
    candidate = datetime.now().replace(microsecond=0)
    upload_dir = RUNTIME_DATA_DIR / "raw" / "uploads"
    while any(
        upload_dir.glob(f"upload_{candidate:%Y-%m-%d}_{candidate:%H%M%S}_*")
    ):
        candidate += timedelta(seconds=1)
    return candidate.strftime("%Y%m%d_%H%M%S")


def _dated_path(series: str, last_date: str, upload_date: str, upload_id: str) -> Path:
    return RUNTIME_DATA_DIR / "clean" / "train_cleaned_data" / (
        f"daily_total_sales{('_clean' if series == 'clean' else '')}"
        f"__through_{last_date}__uploaded_{upload_date}__{upload_id}.csv"
    )


def _read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {"uploads": []}
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read runtime manifest '{MANIFEST_PATH}': {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("uploads", []), list):
        raise ValueError(f"Runtime manifest has an invalid format: {MANIFEST_PATH}.")
    return value


def _read_store_file(path: Path, functions: Any) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        source = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    else:
        source = pd.read_csv(path)
    if {"date", "store_nbr", "family", "sales"}.issubset(source.columns):
        return functions._normalize_store_upload(source)
    if {"timestamp_utc", "sales"}.issubset(source.columns):
        normalized = functions._normalize_daily_dataframe(source)
        _, history_end = functions._load_master_for_upload()
        functions._validate_upload_dates(normalized["timestamp_utc"], history_end)
        return normalized
    return functions._normalize_store_upload(source)


def _normalize_upload_dataframe(source: pd.DataFrame, functions: Any) -> pd.DataFrame:
    if {"date", "store_nbr", "family", "sales"}.issubset(source.columns):
        return functions._normalize_store_upload(source)
    if {"timestamp_utc", "sales"}.issubset(source.columns):
        normalized = functions._normalize_daily_dataframe(source)
        _, history_end = functions._load_master_for_upload()
        functions._validate_upload_dates(normalized["timestamp_utc"], history_end)
        return normalized
    functions._normalize_store_upload(source)
    raise AssertionError("Upload validation returned without a recognized schema.")


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_job_id: str | None = None
        self._reservation = False
        self._queued_rebuild: dict[str, Any] | None = None
        self._state: dict[str, Any] = {"jobs": []}
        self._watcher: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._observed_live: tuple[Any, Any] | None = None
        self._pending_live_since: float | None = None
        self._pending_live_reason: str | None = None
        self._pending_missing_since: float | None = None
        self._pending_missing_ids: set[str] = set()

    @property
    def functions(self) -> Any:
        return pipeline_functions()

    def load(self) -> None:
        ensure_runtime()
        with self._lock:
            if JOBS_PATH.is_file():
                try:
                    self._state = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Could not read runtime job state: {exc}") from exc
            self._state.setdefault("jobs", [])
            for job in self._state["jobs"]:
                if job.get("status") in {"queued", "running"}:
                    job.update(
                        status="failed",
                        error="The server restarted before this job finished.",
                        current_step_name="Interrupted",
                    )
            self._persist_locked()
        self._refresh_watch_baseline()

    def _persist_locked(self) -> None:
        atomic_json(JOBS_PATH, self._state)

    def start_watcher(self) -> None:
        if self._watcher is not None and self._watcher.is_alive():
            return
        self._watcher_stop.clear()
        self._watcher = threading.Thread(
            target=self._watch_loop,
            name="store-sales-data-watcher",
            daemon=True,
        )
        self._watcher.start()

    def stop_watcher(self) -> None:
        self._watcher_stop.set()
        if self._watcher is not None:
            self._watcher.join(timeout=2)
        self._watcher = None

    def _live_paths(self) -> tuple[Path, Path, Path]:
        functions = self.functions
        return (
            functions.MASTER_RAW_PATH,
            functions.CLEAN_PATH,
            functions.FORECASTS_DIR / "latest_forecast.csv",
        )

    def _refresh_watch_baseline(self) -> None:
        master, clean, _ = self._live_paths()
        self._observed_live = (_file_signature(master), _file_signature(clean))
        try:
            _read_manifest()
        except ValueError as exc:
            LOGGER.error("%s", exc)

    def reserve_upload(self) -> bool:
        with self._lock:
            if self._reservation or self._active_job_id is not None:
                return False
            self._reservation = True
            return True

    def release_reservation(self) -> None:
        with self._lock:
            self._reservation = False
            queued = self._queued_rebuild if self._active_job_id is None else None
            if queued is not None:
                self._queued_rebuild = None
        if queued is not None:
            if queued.get("live_edit"):
                self._start_live_edit(queued["reason"])
            else:
                self._start_rebuild(queued["reason"], queued["upload_ids"])

    def is_running(self) -> bool:
        with self._lock:
            return (
                self._reservation
                or self._active_job_id is not None
                or self._queued_rebuild is not None
            )

    def enqueue(self, job: dict[str, Any], upload_path: Path) -> str:
        return self._enqueue_thread(job, upload_path)

    def _enqueue_thread(
        self,
        job: dict[str, Any],
        upload_path: Path | None,
    ) -> str:
        with self._lock:
            if not self._reservation or self._active_job_id is not None:
                raise RuntimeError("Upload reservation was lost before job submission.")
            job_id = str(job["job_id"])
            previous_jobs = list(self._state["jobs"])
            self._active_job_id = job_id
            self._state["jobs"].insert(0, job)
            self._state["jobs"] = self._state["jobs"][:100]
            try:
                self._persist_locked()
                worker = threading.Thread(
                    target=self._run_job,
                    args=(job_id, upload_path),
                    name=f"store-sales-job-{job_id[:8]}",
                    daemon=True,
                )
                worker.start()
            except Exception:
                self._active_job_id = None
                self._state["jobs"] = previous_jobs
                self._reservation = False
                self._persist_locked()
                raise
            self._reservation = False
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

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._find_job_locked(job_id)
            if job is None:
                raise RuntimeError(f"Job state disappeared for {job_id}.")
            job.update(fields)
            self._persist_locked()

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
                job["steps"][number - 1]["started_at"] = _iso_now()
            self._persist_locked()

    def _run_step(self, job_id: str, number: int, action: Any) -> Any:
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
        with self._lock:
            job = self._find_job_locked(job_id)
            if job is not None:
                job["steps"][number - 1]["status"] = "done"
                job["steps"][number - 1]["seconds"] = round(
                    time.perf_counter() - started, 3
                )
                self._persist_locked()
        return result

    def _run_job(self, job_id: str, upload_path: Path | None) -> None:
        functions = self.functions
        backup_paths = (
            functions.MASTER_RAW_PATH,
            functions.CLEAN_PATH,
            functions.FORECASTS_DIR / "latest_forecast.csv",
        )
        snapshots: dict[Path, bytes | None] = {}
        try:
            snapshots = {
                path: path.read_bytes() if path.is_file() else None
                for path in backup_paths
            }
            job = self.get_job(job_id)
            if job is None:
                raise RuntimeError(f"Job state disappeared for {job_id}.")
            if job.get("kind") == "upload":
                self._run_upload(job_id, upload_path, functions)
            elif job.get("kind") == "rebuild":
                self._run_rebuild(job_id, functions)
            elif job.get("kind") == "live_edit":
                self._run_live_edit(job_id, functions)
            else:
                self._run_rerun(job_id, functions)
            self._save_success_snapshots(job_id)
            if job.get("kind") == "rebuild":
                self._finalize_rebuild(job_id)
            self._update_job(
                job_id,
                status="done",
                finished_at=_iso_now(),
                current_step=6,
                current_step_name=STEP_NAMES[5],
            )
        except Exception as exc:
            LOGGER.exception("Store-sales job %s failed", job_id)
            saved_job = self.get_job(job_id) or {}
            archive_name = saved_job.get("archive_path")
            if archive_name:
                try:
                    Path(archive_name).unlink(missing_ok=True)
                except OSError:
                    LOGGER.exception("Could not remove failed upload archive %s", archive_name)
            try:
                self._cleanup_failed_upload(saved_job)
            except Exception:
                LOGGER.exception("Could not remove failed upload snapshot files")
            self._restore_files(snapshots)
            try:
                self._update_job(
                    job_id,
                    status="failed",
                    error=str(exc),
                    finished_at=_iso_now(),
                )
            except Exception:
                LOGGER.exception("Could not save failed job state")
        finally:
            if upload_path is not None:
                try:
                    upload_path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.exception("Could not remove staged upload %s", upload_path)
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                queued = self._queued_rebuild
                self._queued_rebuild = None
            self._refresh_watch_baseline()
            if queued is not None:
                if queued.get("live_edit"):
                    self._start_live_edit(queued["reason"])
                else:
                    self._start_rebuild(queued["reason"], queued["upload_ids"])

    @staticmethod
    def _restore_files(snapshots: dict[Path, bytes | None]) -> None:
        for path, content in snapshots.items():
            try:
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.restore")
                    temporary.write_bytes(content)
                    os.replace(temporary, path)
            except OSError:
                LOGGER.exception("Could not restore a runtime file after job failure")

    def _run_upload(
        self, job_id: str, upload_path: Path | None, functions: Any
    ) -> None:
        if upload_path is None:
            raise RuntimeError("The upload job has no staged upload file.")
        job = self.get_job(job_id)
        uploaded = self._run_step(
            job_id,
            1,
            lambda: self._ingest(upload_path, job, functions),
        )
        daily = self._run_step(
            job_id, 2, lambda: functions.aggregate_upload(uploaded)
        )
        self._run_step(job_id, 3, lambda: functions.update_master_raw(daily))
        self._run_step(job_id, 4, functions.preprocess)
        self._run_step(job_id, 5, functions.refit)
        forecast_path = self._run_step(
            job_id,
            6,
            lambda: functions.forecast(pipelines=self._load_chronos_once(functions)),
        )
        self._update_job(
            job_id,
            forecast_created_at=self._forecast_stamp(forecast_path),
            first_forecast_date=self._first_forecast_date(forecast_path),
        )

    def _run_rebuild(self, job_id: str, functions: Any) -> None:
        job = self.get_job(job_id)
        manifest = _read_manifest()
        removed_ids = set(job.get("upload_ids", []))
        kept_records = [
            record
            for record in manifest.get("uploads", [])
            if record.get("upload_id") not in removed_ids
        ]
        seed = functions.SEED_RAW_PATH
        if not seed.is_file():
            raise FileNotFoundError(f"Seed daily-total data not found: {seed}.")
        self._backup_live_files(functions)
        ignored: list[dict[str, str]] = []
        archives_dir = functions.UPLOADS_DIR
        ordered_records = sorted(
            kept_records,
            key=lambda record: str(record.get("created_at", "")),
        )
        self._run_step(
            job_id,
            1,
            lambda: self._restore_seed(functions, seed),
        )

        def replay_uploads() -> None:
            for record in ordered_records:
                archive = archives_dir / str(record.get("archive", ""))
                if not archive.is_file():
                    continue
                try:
                    uploaded = _read_store_file(archive, functions)
                except ValueError as exc:
                    message = str(exc)
                    if "leaving a gap after the history ending" in message:
                        ignored.append(
                            {
                                "upload_id": str(record.get("upload_id", "")),
                                "original_file": str(
                                    record.get("original_file_name", archive.name)
                                ),
                                "reason": message,
                            }
                        )
                        continue
                    raise
                daily = functions.aggregate_upload(uploaded)
                functions.update_master_raw(daily)

        self._run_step(job_id, 2, replay_uploads)
        self._run_step(job_id, 3, lambda: None)
        self._run_step(job_id, 4, functions.preprocess)
        self._run_step(job_id, 5, functions.refit)
        forecast_path = self._run_step(
            job_id,
            6,
            lambda: functions.forecast(pipelines=self._load_chronos_once(functions)),
        )
        self._update_job(
            job_id,
            ignored_uploads=ignored,
            removed_upload_ids=sorted(removed_ids),
            forecast_created_at=self._forecast_stamp(forecast_path),
            first_forecast_date=self._first_forecast_date(forecast_path),
        )
    def _finalize_rebuild(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job state disappeared for {job_id}.")
        manifest = _read_manifest()
        remaining_ids = set(job.get("upload_ids", []))
        kept_records = [
            record
            for record in manifest.get("uploads", [])
            if record.get("upload_id") not in remaining_ids
        ]
        self._remove_upload_files(remaining_ids, manifest, kept_records)

    @staticmethod
    def _restore_seed(functions: Any, seed: Path) -> None:
        functions._ensure_data_directories()
        shutil.copy2(seed, functions.MASTER_RAW_PATH)

    def _run_live_edit(self, job_id: str, functions: Any) -> None:
        job = self.get_job(job_id)
        master = functions.MASTER_RAW_PATH
        clean = functions.CLEAN_PATH
        self._backup_live_files(functions)
        if master.stat().st_mtime_ns > clean.stat().st_mtime_ns:
            self._run_step(job_id, 4, functions.preprocess)
        else:
            self._run_step(job_id, 4, lambda: None)
        self._run_step(job_id, 5, functions.refit)
        forecast_path = self._run_step(
            job_id,
            6,
            lambda: functions.forecast(pipelines=self._load_chronos_once(functions)),
        )
        self._update_job(
            job_id,
            detected_reason=job.get("detected_reason"),
            forecast_created_at=self._forecast_stamp(forecast_path),
            first_forecast_date=self._first_forecast_date(forecast_path),
        )

    def _run_rerun(self, job_id: str, functions: Any) -> None:
        master = functions.MASTER_RAW_PATH
        clean = functions.CLEAN_PATH
        self._backup_live_files(functions)
        if master.is_file() and (not clean.is_file() or master.stat().st_mtime_ns > clean.stat().st_mtime_ns):
            self._run_step(job_id, 4, functions.preprocess)
        else:
            self._run_step(job_id, 4, lambda: None)
        self._run_step(job_id, 5, functions.refit)
        forecast_path = self._run_step(
            job_id,
            6,
            lambda: functions.forecast(pipelines=self._load_chronos_once(functions)),
        )
        self._update_job(
            job_id,
            forecast_created_at=self._forecast_stamp(forecast_path),
            first_forecast_date=self._first_forecast_date(forecast_path),
        )

    def _backup_live_files(self, functions: Any) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for path in (
            functions.MASTER_RAW_PATH,
            functions.CLEAN_PATH,
            functions.FORECASTS_DIR / "latest_forecast.csv",
        ):
            if path.is_file():
                backup = functions.BACKUPS_DIR / f"{path.stem}_{stamp}{path.suffix}"
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)

    @staticmethod
    def _forecast_stamp(path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        )

    @staticmethod
    def _first_forecast_date(path: Path) -> str:
        forecast = pd.read_csv(path, usecols=["date"])
        if forecast.empty:
            raise ValueError(f"Forecast file is empty: {path}.")
        dates = pd.to_datetime(forecast["date"], errors="raise")
        return dates.min().strftime("%Y-%m-%d")

    def _ingest(
        self, upload_path: Path, job: dict[str, Any], functions: Any
    ) -> pd.DataFrame:
        extension = str(job["extension"]).lower()
        archive_dir = functions.UPLOADS_DIR
        archive_dir.mkdir(parents=True, exist_ok=True)
        if extension == ".xlsx":
            source = pd.read_excel(upload_path, sheet_name=0, engine="openpyxl")
            normalized = _normalize_upload_dataframe(source, functions)
            archive = archive_dir / (
                f"upload_{datetime.strptime(job['upload_id'], '%Y%m%d_%H%M%S'):%Y-%m-%d_%H%M%S}_"
                f"{_safe_original_name(job['original_file_name'])}"
            )
            if archive.exists():
                raise FileExistsError(f"Upload archive already exists: {archive.name}.")
            shutil.copy2(upload_path, archive)
        else:
            existing_archives = set(archive_dir.glob("upload_*"))
            normalized = functions.ingest_upload(upload_path)
            generated = set(archive_dir.glob("upload_*")) - existing_archives
            if len(generated) != 1:
                raise RuntimeError("Could not identify the single archived upload.")
            temporary_archive = next(iter(generated))
            upload_date = datetime.strptime(job["upload_id"], "%Y%m%d_%H%M%S")
            archive = archive_dir / (
                f"upload_{upload_date:%Y-%m-%d}_{upload_date:%H%M%S}_"
                f"{_safe_original_name(job['original_file_name'])}"
            )
            if archive.exists():
                raise FileExistsError(f"Upload archive already exists: {archive.name}.")
            os.replace(temporary_archive, archive)
        self._update_job(job["job_id"], archive_path=str(archive))
        return normalized

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        for suffix in range(2, 10000):
            candidate = path.with_name(f"{path.stem}__{suffix}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not create a unique dated copy for {path.name}.")

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

    def _watch_loop(self) -> None:
        while not self._watcher_stop.wait(WATCH_INTERVAL_SECONDS):
            try:
                self.watch_once()
            except Exception:
                LOGGER.exception("Runtime data watcher scan failed")

    def watch_once(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        functions = self.functions
        master, clean, _ = self._live_paths()
        current_live = (_file_signature(master), _file_signature(clean))
        try:
            manifest = _read_manifest()
        except ValueError as exc:
            LOGGER.error("%s", exc)
            manifest = {"uploads": []}
        records = manifest.get("uploads", [])
        missing_ids: set[str] = set()
        for record in records:
            upload_id = str(record.get("upload_id", ""))
            if not upload_id:
                continue
            required_names = [
                record.get("archive"),
                record.get("aggregated_file"),
                record.get("cleaned_file"),
            ]
            for key, prefix in (
                ("archive", functions.UPLOADS_DIR),
                ("aggregated_file", functions.MASTER_DIR),
                ("cleaned_file", functions.MASTER_DIR),
            ):
                if record.get(key):
                    if not (prefix / record[key]).is_file():
                        missing_ids.add(upload_id)
                        break
            if not any(required_names):
                continue

        if missing_ids:
            if self._pending_missing_since is None:
                self._pending_missing_since = now
                self._pending_missing_ids = set(missing_ids)
            else:
                self._pending_missing_ids.update(missing_ids)
            if now - self._pending_missing_since >= WATCH_DEBOUNCE_SECONDS:
                self._queue_or_start_rebuild(
                    "Upload removed", set(self._pending_missing_ids)
                )
                self._pending_missing_since = None
                self._pending_missing_ids.clear()
        else:
            self._pending_missing_since = None
            self._pending_missing_ids.clear()

        previous = self._observed_live
        self._observed_live = current_live
        if current_live[0] is None or current_live[1] is None:
            self._pending_live_since = None
            return
        if previous is None or current_live == previous:
            if self._pending_live_since is not None:
                if now - self._pending_live_since >= WATCH_DEBOUNCE_SECONDS:
                    reason = self._pending_live_reason or "Live file edited"
                    self._queue_or_start_rebuild(reason, set(), live_edit=True)
                    self._pending_live_since = None
                    self._pending_live_reason = None
            return
        if self.is_running():
            self._pending_live_since = None
            return
        self._pending_live_since = now
        self._pending_live_reason = (
            "Aggregated live file edited"
            if current_live[0] != previous[0]
            else "Cleaned live file edited"
        )

    def _queue_or_start_rebuild(
        self,
        reason: str,
        upload_ids: set[str],
        *,
        live_edit: bool = False,
    ) -> None:
        with self._lock:
            if self._active_job_id is not None or self._reservation:
                previous = self._queued_rebuild or {}
                queued_upload_ids = set(previous.get("upload_ids", [])) | upload_ids
                self._queued_rebuild = {
                    "reason": previous.get("reason", reason),
                    "upload_ids": sorted(queued_upload_ids),
                    "live_edit": not queued_upload_ids
                    and bool(previous.get("live_edit", False) or live_edit),
                }
                return
        if live_edit:
            self._start_live_edit(reason)
        else:
            self._start_rebuild(reason, sorted(upload_ids))

    def _start_rebuild(self, reason: str, upload_ids: list[str]) -> None:
        if not self.reserve_upload():
            with self._lock:
                previous = self._queued_rebuild or {}
                self._queued_rebuild = {
                    "reason": previous.get("reason", reason),
                    "upload_ids": sorted(
                        set(previous.get("upload_ids", [])) | set(upload_ids)
                    ),
                    "live_edit": bool(previous.get("live_edit", False)),
                }
            return
        job = build_job("Data removed: rebuilding", "", {}, kind="rebuild")
        job.update(
            status="queued",
            current_step_name="Data removed: rebuilding",
            detected_reason=reason,
            upload_ids=upload_ids,
        )
        self._enqueue_thread(job, None)

    def _start_live_edit(self, reason: str) -> None:
        if not self.reserve_upload():
            with self._lock:
                self._queued_rebuild = {
                    "reason": reason,
                    "upload_ids": [],
                    "live_edit": True,
                }
            return
        job = build_job("Live file edited: rebuilding forecast", "", {}, kind="live_edit")
        job.update(
            status="queued",
            current_step_name=reason,
            detected_reason=reason,
        )
        self._enqueue_thread(job, None)

    def _job_summary(self) -> list[dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._state.get("jobs", [])))

    def list_uploads(self) -> list[dict[str, Any]]:
        functions = self.functions
        try:
            manifest = _read_manifest()
        except ValueError:
            manifest = {"uploads": []}
        records = {
            record.get("archive"): record
            for record in manifest.get("uploads", [])
            if record.get("archive")
        }
        jobs = {job.get("upload_id"): job for job in self._job_summary()}
        result = []
        for archive in sorted(
            functions.UPLOADS_DIR.glob("upload_*"), reverse=True
        ):
            record = records.get(archive.name, {})
            upload_id = record.get("upload_id") or _upload_id_from_name(archive.name)
            job = jobs.get(upload_id, {})
            result.append(
                {
                    "job_id": record.get("job_id") or job.get("job_id"),
                    "upload_id": upload_id,
                    "created_at": record.get("created_at")
                    or job.get("created_at")
                    or _archive_created_at(archive),
                    "original_file_name": record.get("original_file_name")
                    or _archive_original_name(archive.name),
                    "rows": record.get("rows", job.get("rows", 0)),
                    "days_added": record.get("days_added", job.get("days_added", 0)),
                    "first_date_added": record.get("upload_first_date")
                    or job.get("first_date_added"),
                    "last_date_added": record.get("upload_last_date")
                    or job.get("last_date_added"),
                    "status": job.get("status", "done" if record else "archived"),
                    "ignored_reason": next(
                        (
                            ignored["reason"]
                            for saved_job in self._job_summary()
                            for ignored in saved_job.get("ignored_uploads", [])
                            if ignored.get("upload_id") == upload_id
                        ),
                        None,
                    ),
                }
            )
        return result

    def list_datasets(self) -> list[dict[str, Any]]:
        functions = self.functions
        entries = []
        pattern = re.compile(
            r"^daily_total_sales(?:_clean)?__through_(\d{4}-\d{2}-\d{2})__(uploaded|rebuilt)_(.+)\.csv$"
        )
        for path in functions.MASTER_DIR.glob("daily_total_sales__through_*.csv"):
            entries.append(self._dataset_info(path, pattern, "aggregated"))
        for path in functions.MASTER_DIR.glob("daily_total_sales_clean__through_*.csv"):
            entries.append(self._dataset_info(path, pattern, "cleaned"))
        return sorted(entries, key=lambda row: (row["through_date"], row["file"]), reverse=True)

    @staticmethod
    def _dataset_info(path: Path, pattern: re.Pattern[str], series: str) -> dict[str, Any]:
        match = pattern.match(path.name)
        try:
            frame = pd.read_csv(path, usecols=["timestamp_utc"])
            rows = len(frame)
        except Exception as exc:
            raise ValueError(f"Could not read dated dataset '{path.name}': {exc}") from exc
        return {
            "file": path.name,
            "series": series,
            "through_date": match.group(1) if match else None,
            "kind": match.group(2) if match else "dated",
            "upload_id": match.group(3).removesuffix(".csv") if match else None,
            "rows": rows,
            "download_url": f"/datasets/{path.name}",
        }

    def _remove_upload_files(
        self,
        upload_ids: set[str],
        manifest: dict[str, Any],
        remaining_records: list[dict[str, Any]],
    ) -> None:
        records = {
            str(record.get("upload_id")): record
            for record in manifest.get("uploads", [])
        }
        functions = self.functions
        removed_dir = functions.BACKUPS_DIR / "removed_uploads"
        moved: list[tuple[Path, Path]] = []
        for upload_id in upload_ids:
            record = records.get(upload_id, {})
            names = {
                str(record.get(key, ""))
                for key in ("archive", "aggregated_file", "cleaned_file")
            }
            names.discard("")
            for directory in (functions.UPLOADS_DIR, functions.MASTER_DIR):
                names.update(path.name for path in directory.glob("*") if upload_id in path.name)
            for name in names:
                for source in (functions.UPLOADS_DIR / name, functions.MASTER_DIR / name):
                    if source.is_file():
                        try:
                            removed_dir.mkdir(parents=True, exist_ok=True)
                            destination = self._unique_path(removed_dir / source.name)
                            os.replace(source, destination)
                            moved.append((source, destination))
                            companion = source.with_suffix(".json")
                            if companion.exists():
                                companion_destination = self._unique_path(
                                    removed_dir / companion.name
                                )
                                os.replace(companion, companion_destination)
                                moved.append((companion, companion_destination))
                        except Exception:
                            for original, backup in reversed(moved):
                                if backup.exists():
                                    original.parent.mkdir(parents=True, exist_ok=True)
                                    os.replace(backup, original)
                            raise
        try:
            atomic_json(MANIFEST_PATH, {"uploads": remaining_records})
        except Exception:
            for original, backup in reversed(moved):
                if backup.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, original)
            raise

    def _save_success_snapshots(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job state disappeared for {job_id}.")
        if job.get("kind") not in {"upload", "rebuild"}:
            return
        functions = self.functions
        master = pd.read_csv(functions.MASTER_RAW_PATH, parse_dates=["timestamp_utc"])
        clean = pd.read_csv(functions.CLEAN_PATH, parse_dates=["timestamp_utc"])
        if master.empty or clean.empty:
            raise ValueError("The live aggregated or cleaned series is empty.")
        last_date = pd.Timestamp(master["timestamp_utc"].max()).strftime("%Y-%m-%d")
        upload_id = str(job.get("upload_id") or job["job_id"])
        if job.get("kind") == "upload":
            upload_date = datetime.strptime(upload_id, "%Y%m%d_%H%M%S").strftime(
                "%Y-%m-%d"
            )
            master_copy = _dated_path("master", last_date, upload_date, upload_id)
            clean_copy = _dated_path("clean", last_date, upload_date, upload_id)
        else:
            directory = functions.MASTER_DIR
            master_copy = directory / (
                f"daily_total_sales__through_{last_date}__rebuilt_{upload_id}.csv"
            )
            clean_copy = directory / (
                f"daily_total_sales_clean__through_{last_date}__rebuilt_{upload_id}.csv"
            )
        master_copy = self._unique_path(master_copy)
        clean_copy = self._unique_path(clean_copy)
        metadata = {
            "job_id": job["job_id"],
            "upload_id": job.get("upload_id"),
            "original_file_name": job["original_file_name"],
            "upload_first_date": job.get("first_date_added"),
            "upload_last_date": job.get("last_date_added"),
            "first_date": job.get("first_date_added"),
            "last_date": job.get("last_date_added"),
            "days_added": job.get("days_added", 0),
            "rows": job.get("rows", 0),
            "created_at": job["created_at"],
        }
        manifest_path = MANIFEST_PATH
        previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
        created_files: list[Path] = []
        try:
            atomic_copy(functions.MASTER_RAW_PATH, master_copy)
            created_files.append(master_copy)
            atomic_copy(functions.CLEAN_PATH, clean_copy)
            created_files.append(clean_copy)
            for path in (master_copy, clean_copy):
                atomic_json(path.with_suffix(".json"), metadata)
                created_files.append(path.with_suffix(".json"))
        except Exception:
            for path in created_files:
                path.unlink(missing_ok=True)
            raise
        if job.get("kind") == "rebuild":
            return
        record = {
            **metadata,
            "archive": Path(job["archive_path"]).name,
            "aggregated_file": master_copy.name,
            "cleaned_file": clean_copy.name,
        }
        manifest = _read_manifest()
        records = [
            item
            for item in manifest.get("uploads", [])
            if item.get("upload_id") != job["upload_id"]
        ]
        records.append(record)
        try:
            atomic_json(MANIFEST_PATH, {"uploads": records})
        except Exception:
            for path in created_files:
                path.unlink(missing_ok=True)
            if previous_manifest is not None:
                temporary = manifest_path.with_name(
                    f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
                )
                temporary.write_bytes(previous_manifest)
                os.replace(temporary, manifest_path)
            else:
                manifest_path.unlink(missing_ok=True)
            raise

    def _cleanup_failed_upload(self, job: dict[str, Any]) -> None:
        upload_id = str(job.get("upload_id") or "")
        if not upload_id:
            return
        functions = self.functions
        for path in functions.MASTER_DIR.glob(f"*{upload_id}*"):
            if path.is_file():
                path.unlink(missing_ok=True)
        try:
            manifest = _read_manifest()
        except ValueError:
            return
        records = [
            item
            for item in manifest.get("uploads", [])
            if item.get("upload_id") != upload_id
        ]
        try:
            atomic_json(MANIFEST_PATH, {"uploads": records})
        except Exception:
            LOGGER.exception("Could not remove failed upload from the runtime manifest")

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        for suffix in range(2, 10000):
            candidate = path.with_name(f"{path.stem}__{suffix}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not create a unique dated copy for {path.name}.")

    def status(self) -> dict[str, Any]:
        functions = self.functions
        live_files = []
        for path in (functions.MASTER_RAW_PATH, functions.CLEAN_PATH):
            live_files.append(_describe_data_file(path, "timestamp_utc"))
        forecast = _describe_data_file(
            functions.FORECASTS_DIR / "latest_forecast.csv",
            "date",
            forecast=True,
        )
        with self._lock:
            active_job_id = self._active_job_id
            active_job = (
                self._find_job_locked(active_job_id) if active_job_id else None
            )
            queued = dict(self._queued_rebuild) if self._queued_rebuild else None
            has_pending_change = (
                self._pending_live_since is not None
                or self._pending_missing_since is not None
            )
            pending_reason = self._pending_live_reason or (
                "Upload removal detected; rebuild scheduled"
                if self._pending_missing_since is not None
                else "Runtime file change detected; rebuild scheduled"
            )
        working = None
        if active_job_id or queued or has_pending_change:
            if active_job:
                reason = active_job.get("detected_reason") or active_job.get(
                    "current_step_name"
                )
            elif queued:
                reason = queued.get("reason", "Queued rebuild")
            else:
                reason = pending_reason
            working = {
                "job_id": active_job_id,
                "reason": reason,
                "pending": active_job_id is None,
            }
        return {
            "runtime_path": str(RUNTIME_DIR),
            "files": live_files,
            "forecast": forecast,
            "working": working,
        }

    def data_version(self) -> str:
        return json.dumps(
            [
                signature
                for signature in (
                    _file_signature(path)
                    for path in self._live_paths()
                )
            ],
            separators=(",", ":"),
        )

    def latest_new_range(self) -> dict[str, str] | None:
        try:
            manifest = _read_manifest()
        except ValueError:
            return None
        newest = max(
            manifest.get("uploads", []),
            key=lambda record: str(record.get("created_at", "")),
            default=None,
        )
        if newest is None:
            return None
        metadata_path = self.functions.MASTER_DIR / str(newest.get("cleaned_file", ""))
        metadata_path = metadata_path.with_suffix(".json")
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def enqueue_manual_rerun(self) -> str | None:
        if not self.reserve_upload():
            return None
        job = build_job("Manual rerun", "", {}, kind="rerun")
        return self._enqueue_thread(job, None)


def _describe_data_file(
    path: Path,
    date_column: str,
    *,
    forecast: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": path.name, "path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["error"] = f"File not found: {path}."
        return result
    try:
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"File is empty: {path}.")
        if date_column not in frame.columns:
            raise ValueError(f"Required column '{date_column}' is missing from {path}.")
        dates = pd.to_datetime(frame[date_column], format="mixed", errors="raise")
        result.update(
            rows=len(frame),
            first_date=dates.min().strftime("%Y-%m-%d"),
            last_date=dates.max().strftime("%Y-%m-%d"),
            modified_at=datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        )
        if forecast:
            result["first_forecast_date"] = dates.min().strftime("%Y-%m-%d")
            result["forecast_created_at"] = result["modified_at"]
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _upload_id_from_name(name: str) -> str:
    match = re.match(r"upload_\d{4}-\d{2}-\d{2}_(\d{6})_", name)
    if not match:
        return ""
    date = name[7:17].replace("-", "")
    return f"{date}_{match.group(1)}"


def _archive_original_name(name: str) -> str:
    return re.sub(r"^upload_\d{4}-\d{2}-\d{2}_\d{6}_", "", name)


def _archive_created_at(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return ""


job_manager = JobManager()


def build_job(
    original_file_name: str,
    extension: str,
    metadata: dict[str, Any],
    *,
    kind: str = "upload",
) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    upload_id = _upload_id() if kind == "upload" else ""
    return {
        "job_id": uuid.uuid4().hex,
        "upload_id": upload_id,
        "created_at": now,
        "original_file_name": original_file_name,
        "extension": extension,
        **metadata,
        "kind": kind,
        "status": "queued",
        "current_step": 0,
        "current_step_name": "Waiting to start",
        "error": None,
        "steps": [
            {"number": index, "name": name, "status": "waiting", "seconds": None}
            for index, name in enumerate(STEP_NAMES, 1)
        ],
    }


def validate_uploaded_file(path: Path, extension: str) -> tuple[pd.DataFrame, dict[str, Any]]:
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

    normalized = _normalize_upload_dataframe(source_df, functions)
    dates = normalized["date"] if "date" in normalized.columns else normalized["timestamp_utc"]
    master_dates = pd.to_datetime(
        pd.read_csv(functions.MASTER_RAW_PATH, usecols=["timestamp_utc"])[
            "timestamp_utc"
        ],
        format="mixed",
    ).dt.normalize()
    added_dates = pd.date_range(dates.min(), dates.max(), freq="D").difference(
        pd.DatetimeIndex(master_dates)
    )
    return normalized, {
        "rows": len(source_df),
        "days_added": len(added_dates),
        "first_date_added": added_dates.min().strftime("%Y-%m-%d") if len(added_dates) else None,
        "last_date_added": added_dates.max().strftime("%Y-%m-%d") if len(added_dates) else None,
    }


def initialize() -> None:
    ensure_runtime()
    pipeline_functions()
    job_manager.load()
    LOGGER.info("Store-sales runtime data directory: %s", RUNTIME_DATA_DIR)
    LOGGER.info("Store-sales runtime state directory: %s", STATE_DIR)
    job_manager.start_watcher()


def shutdown() -> None:
    job_manager.stop_watcher()
