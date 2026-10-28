from __future__ import annotations

import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from api.services import ACTIVE_RUN, LOCK, MODEL_IDS, MODEL_INFO, batch_pending, data_workflow, db_error, latest_version
from db.config import database_url, engine
from db.models import Batch, CleanSeries, DailyTotal, Forecast, Run, SalesNew

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    import os
    expected = os.getenv("API_KEY", "")
    if not expected:
        raise RuntimeError("API_KEY is required in .env.")
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key.")
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return x_api_key


app = FastAPI(title="Advanced Time Series Store Sales")

@app.on_event("startup")
def startup_check():
    import os
    if not os.getenv("API_KEY", "").strip():
        raise RuntimeError("API_KEY is required in .env. Add it to the local environment file.")
    database_url()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        with Session(engine()) as session:
            last = session.scalar(select(func.max(CleanSeries.date)))
            return {"status": "OK", "database_ok": True, "database_error": None, "last_data_date": last.isoformat() if last else None, "active_run": ACTIVE_RUN}
    except Exception as exc:
        info = db_error(exc)
        return {"status": "ERROR", "database_ok": info["database_ok"], "database_error": info["database_error"], "last_data_date": None, "active_run": ACTIVE_RUN}


@app.get("/api/pending")
def pending() -> dict[str, Any]:
    try:
        with Session(engine()) as session:
            pending_rows = session.execute(select(SalesNew).where(SalesNew.status == "pending").order_by(SalesNew.id)).scalars().all()
            direct = session.scalar(select(func.count()).select_from(SalesNew).where(SalesNew.status == "pending", SalesNew.batch_id.is_(None))) or 0
            batches = []
            for batch in session.execute(select(Batch).where(Batch.status == "pending").order_by(Batch.received_at.desc())).scalars():
                batches.append({"batch_id": batch.batch_id, "source": batch.source, "original_file_name": batch.original_file_name, "first_date": batch.first_date.isoformat() if batch.first_date else None, "last_date": batch.last_date.isoformat() if batch.last_date else None, "rows": batch.rows, "status": batch.status, "message": batch.message})
            return {"pending_rows": len(pending_rows), "direct_rows": direct, "batches": batches}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {db_error(exc)['database_error']}") from exc


@app.post("/api/start")
def start_data() -> dict[str, int]:
    global ACTIVE_RUN
    if ACTIVE_RUN is not None:
        raise HTTPException(status_code=409, detail="A data or forecast run is already active.")
    try:
        with Session(engine()) as session:
            _, batch_ids = batch_pending(session)
            session.commit()
            count = session.scalar(select(func.count()).select_from(SalesNew).where(SalesNew.status == "pending")) or 0
            if not count:
                raise HTTPException(status_code=400, detail="There is no pending data to start.")
            run = Run(kind="data", status="queued", step="Read pending data", batch_ids=batch_ids)
            session.add(run)
            session.flush()
            run_id = run.run_id
            session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {db_error(exc)['database_error']}") from exc
    with LOCK:
        if ACTIVE_RUN is not None:
            raise HTTPException(status_code=409, detail="A data or forecast run is already active.")
        ACTIVE_RUN = run_id
    thread = threading.Thread(target=_run_data, args=(run_id,), daemon=True)
    thread.start()
    return {"run_id": run_id}


def _run_data(run_id: int) -> None:
    global ACTIVE_RUN
    from pipeline import functions as pipeline_functions
    try:
        with Session(engine()) as session:
            rows = session.execute(select(SalesNew).where(SalesNew.status == "pending").order_by(SalesNew.id)).scalars().all()
            batch_ids = sorted({row.batch_id for row in rows if row.batch_id is not None})
            records = [{"date": row.date.isoformat(), "store_nbr": row.store_nbr, "family": row.family, "sales": float(row.sales), "onpromotion": row.onpromotion} for row in rows]
            session.execute(update(Run).where(Run.run_id == run_id).values(status="running", step="Read pending data", started_at=datetime.now(timezone.utc)))
            session.commit()
        for step in ("Aggregate by date", "Merge into history", "Clean the series", "Refit calendar and gradient boosting"):
            with Session(engine()) as session:
                session.execute(update(Run).where(Run.run_id == run_id).values(step=step))
                session.commit()
        version = data_workflow(records)
        with Session(engine()) as session:
            session.execute(update(SalesNew).where(SalesNew.status == "pending").values(status="processed"))
            session.execute(update(Batch).where(Batch.batch_id.in_(batch_ids)).values(status="processed", message=None))
            session.execute(update(Run).where(Run.run_id == run_id).values(status="done", step="Completed", finished_at=datetime.now(timezone.utc), message=None))
            session.commit()
    except Exception as exc:
        message = str(exc)
        with Session(engine()) as session:
            session.execute(update(Run).where(Run.run_id == run_id).values(status="failed", finished_at=datetime.now(timezone.utc), message=message))
            session.execute(update(Batch).where(Batch.batch_id.in_(batch_ids if 'batch_ids' in locals() else [])).values(status="failed", message=message))
            session.execute(update(SalesNew).where(SalesNew.status == "pending", SalesNew.batch_id.in_(batch_ids if 'batch_ids' in locals() else [])).values(status="failed"))
            session.commit()
    finally:
        with LOCK:
            ACTIVE_RUN = None


@app.post("/model/data")
def model_data(records: list[dict[str, Any]], _key: str = Depends(require_api_key)) -> dict[str, Any]:
    try:
        return {"data_version": data_workflow(records)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data processing failed: {exc}") from exc


@app.post("/api/forecast/run")
def api_forecast_run(model: str = Query(...)) -> dict[str, int]:
    global ACTIVE_RUN
    if model not in MODEL_IDS:
        raise HTTPException(status_code=404, detail="Unknown model id.")
    if ACTIVE_RUN is not None:
        raise HTTPException(status_code=409, detail="A data or forecast run is already active.")
    try:
        with Session(engine()) as session:
            version = latest_version(session)
            if version is None:
                raise HTTPException(status_code=404, detail="No data version is available.")
            if session.scalar(select(func.count()).select_from(Forecast).where(Forecast.data_version == version.data_version, Forecast.model == model)):
                return {"run_id": 0}
            run = Run(kind="forecast", model=model, status="queued", step=f"Computing forecast for {MODEL_INFO[model][0]}", batch_ids=[])
            session.add(run); session.flush(); run_id = run.run_id; session.commit()
    except HTTPException:
        raise
    with LOCK:
        if ACTIVE_RUN is not None:
            raise HTTPException(status_code=409, detail="A data or forecast run is already active.")
        ACTIVE_RUN = run_id
    threading.Thread(target=_run_forecast, args=(run_id, model), daemon=True).start()
    return {"run_id": run_id}


def _run_forecast(run_id: int, model: str) -> None:
    global ACTIVE_RUN
    try:
        _forecast_model(model)
        with Session(engine()) as session:
            session.execute(update(Run).where(Run.run_id == run_id).values(status="done", step="Completed", finished_at=datetime.now(timezone.utc), message=None))
            session.commit()
    except Exception as exc:
        with Session(engine()) as session:
            session.execute(update(Run).where(Run.run_id == run_id).values(status="failed", finished_at=datetime.now(timezone.utc), message=str(exc)))
            session.commit()
    finally:
        with LOCK:
            ACTIVE_RUN = None


@app.post("/model/forecast")
def model_forecast(payload: dict[str, str], _key: str = Depends(require_api_key)) -> dict[str, Any]:
    model = payload.get("model")
    if model not in MODEL_IDS:
        raise HTTPException(status_code=404, detail="Unknown model id.")
    try:
        return {"rows": _forecast_model(model)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc


def _forecast_model(model: str) -> int:
    import importlib
    from pipeline import functions
    with Session(engine()) as session:
        version = latest_version(session)
        if version is None:
            raise ValueError("No data version is available.")
        clean = session.execute(select(CleanSeries).order_by(CleanSeries.date)).scalars().all()
    work = PROJECT_ROOT / "runtime" / "work" / "data"
    clean_dir = work / "clean" / "train_cleaned_data"
    clean_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"timestamp_utc": row.date, "sales": float(row.sales), "sales_raw": float(row.sales_raw), "is_closure": row.is_closure} for row in clean]).to_csv(clean_dir / "daily_total_sales_clean.csv", index=False, date_format="%Y-%m-%d")
    os_model = PROJECT_ROOT / "runtime" / "work" / "models"
    import os
    os.environ["STORE_SALES_DATA_DIR"] = str(work)
    os.environ["STORE_SALES_MODEL_DIR"] = str(os_model)
    functions = importlib.reload(functions)
    pipelines = None
    if model != "level_calendar_no_model":
        from api.services import get_chronos_pipeline
        pipelines = {"amazon/chronos-t5-large": get_chronos_pipeline()}
    result = functions.forecast(pipelines=pipelines, model_id=model)
    frame = pd.read_csv(result, parse_dates=["date"])
    if model != "level_calendar_no_model":
        frame = frame.loc[frame["model"] == model]
    if frame.empty or len(frame) != 16 or frame["forecast"].isna().any():
        raise ValueError("Forecast output is incomplete or contains invalid values.")
    with Session(engine()) as session:
        session.execute(__import__("sqlalchemy").delete(Forecast).where(Forecast.data_version == version.data_version, Forecast.model == model))
        session.execute(__import__("sqlalchemy").insert(Forecast), [{"data_version": version.data_version, "model": model, "date": row.date.date(), "forecast": round(float(row.forecast), 9)} for row in frame.itertuples(index=False)])
        session.commit()
    return len(frame)


@app.get("/api/runs/{run_id}")
def run_status(run_id: int) -> dict[str, Any]:
    try:
        with Session(engine()) as session:
            run = session.get(Run, run_id)
            if run is None: raise HTTPException(status_code=404, detail="Run not found.")
            elapsed = None
            if run.started_at and run.finished_at: elapsed = (run.finished_at - run.started_at).total_seconds()
            return {"run_id": run.run_id, "status": run.status, "step": run.step, "seconds_per_step": elapsed, "error": run.message}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=503, detail=f"Database error: {db_error(exc)['database_error']}") from exc


@app.get("/api/history")
def history(days: int = Query(default=365, ge=1)) -> dict[str, Any]:
    try:
        with Session(engine()) as session:
            rows = session.execute(select(CleanSeries).order_by(CleanSeries.date.desc()).limit(days)).scalars().all()
            latest = session.execute(select(Batch).where(Batch.status == "processed").order_by(Batch.received_at.desc()).limit(1)).scalar_one_or_none()
            first, last = (latest.first_date, latest.last_date) if latest else (None, None)
            return {"rows": [{"date": row.date.isoformat(), "sales": float(row.sales), "is_new": bool(first and last and first <= row.date <= last)} for row in reversed(rows)]}
    except Exception as exc: raise HTTPException(status_code=503, detail=f"Database error: {db_error(exc)['database_error']}") from exc


@app.get("/api/forecast")
def forecast_state(model: str = Query(...)) -> dict[str, Any]:
    if model not in MODEL_IDS: raise HTTPException(status_code=404, detail="Unknown model id.")
    try:
        with Session(engine()) as session:
            version = latest_version(session)
            if version is None: return {"state": "not_computed", "rows": [], "created_at": None, "first_forecast_date": None, "last_forecast_date": None, "data_version": None}
            rows = session.execute(select(Forecast).where(Forecast.data_version == version.data_version, Forecast.model == model).order_by(Forecast.date)).scalars().all()
            if rows:
                return {"state": "ready", "rows": [{"date": r.date.isoformat(), "forecast": float(r.forecast)} for r in rows], "created_at": rows[0].created_at.isoformat() if rows[0].created_at else None, "first_forecast_date": rows[0].date.isoformat(), "last_forecast_date": rows[-1].date.isoformat(), "data_version": version.data_version}
            active = session.execute(select(Run).where(Run.kind == "forecast", Run.model == model, Run.status.in_(["queued", "running"])).order_by(Run.run_id.desc()).limit(1)).scalar_one_or_none()
            return {"state": "computing" if active else "not_computed", "rows": [], "created_at": None, "first_forecast_date": None, "last_forecast_date": None, "data_version": version.data_version}
    except Exception as exc: raise HTTPException(status_code=503, detail=f"Database error: {db_error(exc)['database_error']}") from exc


@app.get("/api/forecast/download")
def forecast_download(model: str = Query(...)) -> FileResponse:
    state = forecast_state(model)
    rows = state["rows"]
    path = PROJECT_ROOT / "runtime" / "forecast_download.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return FileResponse(path, media_type="text/csv", filename=f"{model}_forecast.csv")


@app.get("/api/models")
def models() -> list[dict[str, str]]:
    return [{"id": key, "plain_name": value[0], "description": value[1], "kind": value[2]} for key, value in MODEL_INFO.items()]


@app.get("/api/batches")
def batches() -> dict[str, Any]:
    try:
        with Session(engine()) as session:
            result = []
            for b in session.execute(select(Batch).order_by(Batch.received_at.desc()).limit(50)).scalars():
                result.append({"batch_id": b.batch_id, "source": b.source, "original_file_name": b.original_file_name, "received_at": b.received_at.isoformat() if b.received_at else None, "first_date": b.first_date.isoformat() if b.first_date else None, "last_date": b.last_date.isoformat() if b.last_date else None, "rows": b.rows, "status": b.status, "message": b.message})
            return {"rows": result}
    except Exception as exc: raise HTTPException(status_code=503, detail=f"Database error: {db_error(exc)['database_error']}") from exc

app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
