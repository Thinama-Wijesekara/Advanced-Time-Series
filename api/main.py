"""Loopback-only FastAPI app for local store-sales uploads and forecasts."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import jobs
from api import publishing


LOGGER = logging.getLogger(__name__)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
CHUNK_SIZE = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.initialize()
    yield


app = FastAPI(
    title="Advanced Time Series Sales Forecasting",
    description="Local store-sales upload, refit, and forecast dashboard.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    functions = jobs.pipeline_functions()
    publish_settings = publishing.publish_settings(jobs.REPO_ROOT)
    master_path = functions.MASTER_RAW_PATH
    last_date = None
    if master_path.is_file():
        master = pd.read_csv(master_path, usecols=["timestamp_utc"])
        if not master.empty:
            last_date = pd.to_datetime(
                master["timestamp_utc"], format="mixed"
            ).max().strftime("%Y-%m-%d")
    return {
        "status": "OK",
        "last_data_date": last_date,
        "job_running": jobs.job_manager.is_running(),
        "publishing_enabled": publish_settings.enabled,
        "publishing_config_error": publish_settings.error,
    }


@app.post("/upload", status_code=202)
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    if not jobs.job_manager.reserve_upload():
        raise HTTPException(
            status_code=409,
            detail="A pipeline job is already running.",
        )

    upload_path: Path | None = None
    enqueued = False
    try:
        original_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not original_name:
            original_name = "upload"
        extension = Path(original_name).suffix.lower()
        if extension not in {".csv", ".xlsx"}:
            raise HTTPException(
                status_code=400,
                detail="Upload must be a .xlsx or .csv file.",
            )

        jobs.ensure_runtime()
        upload_path = jobs.INCOMING_DIR / f"{uuid.uuid4().hex}{extension}"
        size = 0
        with upload_path.open("wb") as output:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > jobs.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Upload exceeds the 100 MB limit.",
                    )
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="Upload contains no data rows.")

        try:
            _, metadata = await asyncio.to_thread(
                jobs.validate_uploaded_file, upload_path, extension
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Could not validate uploaded file")
            raise HTTPException(
                status_code=422,
                detail=f"Could not validate upload: {exc}",
            ) from exc

        job = jobs.build_job(original_name, extension, metadata)
        try:
            job_id = jobs.job_manager.enqueue(job, upload_path)
        except Exception:
            LOGGER.exception("Could not start background pipeline job")
            raise HTTPException(
                status_code=500,
                detail="Could not start the pipeline job.",
            ) from None
        enqueued = True
        return {"job_id": job_id}
    finally:
        await file.close()
        if not enqueued:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)
            jobs.job_manager.release_reservation()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/uploads")
def recent_uploads() -> list[dict[str, Any]]:
    return jobs.job_manager.list_uploads()


@app.get("/history")
def history(days: int = Query(default=365, ge=1)) -> list[dict[str, Any]]:
    functions = jobs.pipeline_functions()
    if not functions.CLEAN_PATH.is_file():
        raise HTTPException(status_code=404, detail="Cleaned history is not available.")
    dataframe = pd.read_csv(
        functions.CLEAN_PATH,
        parse_dates=["timestamp_utc"],
    ).tail(days)
    new_range = jobs.job_manager.latest_new_range()
    first_new = (
        pd.Timestamp(new_range["first_date"])
        if new_range and new_range.get("first_date")
        else None
    )
    last_new = (
        pd.Timestamp(new_range["last_date"])
        if new_range and new_range.get("last_date")
        else None
    )
    result: list[dict[str, Any]] = []
    for row in dataframe.itertuples(index=False):
        current_date = pd.Timestamp(row.timestamp_utc)
        is_new = (
            first_new is not None
            and last_new is not None
            and first_new <= current_date <= last_new
        )
        result.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "sales": float(row.sales),
                "is_new": bool(is_new),
            }
        )
    return result


def _latest_forecast_path() -> Path:
    return jobs.pipeline_functions().FORECASTS_DIR / "latest_forecast.csv"


def _forecast_created_at(path: Path) -> str:
    stored = jobs.job_manager.last_forecast_created_at()
    if stored:
        return stored
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )


@app.get("/forecast/latest")
def latest_forecast() -> dict[str, Any]:
    path = _latest_forecast_path()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No forecast is available yet.")
    try:
        dataframe = pd.read_csv(path, parse_dates=["date"])
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read latest forecast: {exc}",
        ) from exc
    rows = [
        {
            "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            "model": str(row.model),
            "forecast": float(row.forecast),
        }
        for row in dataframe.itertuples(index=False)
    ]
    return {"created_at": _forecast_created_at(path), "rows": rows}


@app.get("/forecast/download")
def download_forecast() -> FileResponse:
    path = _latest_forecast_path()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No forecast is available yet.")
    return FileResponse(
        path,
        media_type="text/csv",
        filename="latest_forecast.csv",
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
