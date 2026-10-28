from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.orm import Session

from db.config import engine
from db.models import Batch, CleanSeries, DailyTotal, DataVersion, Forecast, Run, SalesHistory, SalesNew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "chronos_t5_large_zero_shot_adjusted",
    "ensemble_chronos_t5_large_gbm_w0.7_2",
    "level_calendar_no_model",
)
MODEL_INFO = {
    MODEL_IDS[0]: ("Chronos-large zero-shot", "Calendar-adjusted Chronos-large point forecast", "slow"),
    MODEL_IDS[1]: ("Chronos-large + GBM ensemble", "Chronos-large combined with the refitted gradient booster", "slow"),
    MODEL_IDS[2]: ("Level + calendar", "Level baseline with calendar effects from Ridge", "instant"),
}
LOCK = threading.RLock()
ACTIVE_RUN: int | None = None
CHRONOS_PIPELINE: Any | None = None
CHRONOS_LOCK = threading.Lock()


def db_error(exc: Exception) -> dict[str, Any]:
    return {"database_ok": False, "database_error": str(exc.orig if hasattr(exc, "orig") else exc)}


def latest_version(session: Session):
    return session.execute(select(DataVersion).order_by(DataVersion.data_version.desc())).scalar_one_or_none()


def batch_pending(session: Session) -> tuple[list[dict[str, Any]], list[int]]:
    direct_count = session.scalar(select(func.count()).select_from(SalesNew).where(SalesNew.status == "pending", SalesNew.batch_id.is_(None))) or 0
    batch_ids = []
    if direct_count:
        direct = session.execute(select(func.min(SalesNew.date), func.max(SalesNew.date), func.count()).select_from(SalesNew).where(SalesNew.status == "pending", SalesNew.batch_id.is_(None))).one()
        result = session.execute(insert(Batch).values(source="database-direct", original_file_name="SQL or connector rows", received_at=datetime.now(timezone.utc),
            first_date=direct[0], last_date=direct[1], rows=direct_count, status="pending", message=None).returning(Batch.batch_id))
        batch_id = result.scalar_one()
        session.execute(update(SalesNew).where(SalesNew.status == "pending", SalesNew.batch_id.is_(None)).values(batch_id=batch_id))
        batch_ids.append(batch_id)
    rows = session.execute(select(SalesNew).where(SalesNew.status == "pending").order_by(SalesNew.id)).scalars().all()
    ids = sorted({row.batch_id for row in rows if row.batch_id is not None})
    batch_ids.extend(batch_id for batch_id in ids if batch_id not in batch_ids)
    result = [dict(id=row.id, batch_id=row.batch_id, date=row.date.isoformat(), store_nbr=row.store_nbr, family=row.family,
                   sales=float(row.sales), onpromotion=row.onpromotion) for row in rows]
    return result, batch_ids


def run_record(session: Session, kind: str, model: str | None = None, batch_ids: list[int] | None = None) -> Run:
    run = Run(kind=kind, model=model, status="queued", step="Queued", message=None, batch_ids=batch_ids or [])
    session.add(run)
    session.flush()
    return run


def data_workflow(records: list[dict[str, Any]]) -> int:
    from pipeline import functions as pipeline_functions

    work = PROJECT_ROOT / "runtime" / "work"
    data_dir = work / "data"
    model_dir = work / "models"
    shutil.rmtree(work, ignore_errors=True)
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "clean" / "train_cleaned_data").mkdir(parents=True)
    (data_dir / "reference").mkdir(parents=True)
    for name in ("holidays_events.csv", "stores.csv"):
        shutil.copy2(PROJECT_ROOT / "data" / "reference" / name, data_dir / "reference" / name)

    with Session(engine()) as session:
        totals = session.execute(select(DailyTotal).order_by(DailyTotal.date)).scalars().all()
        pd.DataFrame([{"timestamp_utc": row.date, "sales": float(row.sales)} for row in totals]).to_csv(
            data_dir / "clean" / "train_cleaned_data" / "daily_total_sales.csv", index=False, date_format="%Y-%m-%d")

    upload = pd.DataFrame(records)
    upload["date"] = pd.to_datetime(upload["date"])
    os.environ["STORE_SALES_DATA_DIR"] = str(data_dir)
    os.environ["STORE_SALES_MODEL_DIR"] = str(model_dir)
    # Prepare a model copy and version directory without modifying tracked models.
    shutil.copytree(PROJECT_ROOT / "models" / "selected_models", model_dir, ignore=shutil.ignore_patterns("versions", ".tmp_*"))

    normalized = pipeline_functions._normalize_store_upload(upload.rename(columns={"date": "date"}))
    pipeline_functions._validate_upload_dates(normalized["date"], pd.Timestamp(pd.read_csv(data_dir / "clean" / "train_cleaned_data" / "daily_total_sales.csv")["timestamp_utc"].max()))
    counts = normalized.groupby("date").size()
    expected = pipeline_functions._expected_store_count() * len(pd.read_csv(PROJECT_ROOT / "data" / "reference" / "stores.csv"))
    pipeline_functions._expected_store_count() if False else None
    daily = pipeline_functions.aggregate_upload(normalized)
    master = pd.DataFrame(pd.read_csv(data_dir / "clean" / "train_cleaned_data" / "daily_total_sales.csv")).rename(columns={"timestamp_utc": "date"})
    merged = pd.concat([master.assign(timestamp_utc=pd.to_datetime(master["date"])), daily[["timestamp_utc", "sales"]]], ignore_index=True)
    merged = merged.drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc")
    merged.rename(columns={"timestamp_utc": "timestamp_utc"}).to_csv(data_dir / "clean" / "train_cleaned_data" / "daily_total_sales.csv", index=False, date_format="%Y-%m-%d")
    pipeline_functions.preprocess()
    pipeline_functions.refit()

    master_out = pd.read_csv(data_dir / "clean" / "train_cleaned_data" / "daily_total_sales.csv", parse_dates=["timestamp_utc"])
    clean = pd.read_csv(data_dir / "clean" / "train_cleaned_data" / "daily_total_sales_clean.csv", parse_dates=["timestamp_utc"])
    version_dir = f"database-{int(time.time())}"
    with Session(engine()) as session:
        session.execute(delete(DailyTotal))
        session.execute(insert(DailyTotal), [{"date": row.timestamp_utc.date(), "sales": round(float(row.sales), 9)} for row in master_out.itertuples(index=False)])
        session.execute(delete(CleanSeries))
        session.execute(insert(CleanSeries), [{"date": row.timestamp_utc.date(), "sales": round(float(row.sales), 9), "sales_raw": round(float(row.sales_raw), 9), "is_closure": bool(row.is_closure)} for row in clean.itertuples(index=False)])
        previous = latest_version(session)
        new_number = int(previous.data_version) + 1 if previous else 1
        session.add(DataVersion(data_version=new_number, last_date=clean["timestamp_utc"].max().date(), rows=len(clean), run_id=None))
        session.commit()
        return new_number


def get_chronos_pipeline():
    global CHRONOS_PIPELINE
    with CHRONOS_LOCK:
        if CHRONOS_PIPELINE is not None:
            return CHRONOS_PIPELINE
        from pipeline.functions import _load_chronos_pipelines
        pipelines, _, _ = _load_chronos_pipelines({
            "chronos_t5_large_zero_shot_adjusted": {"huggingface_name": "amazon/chronos-t5-large"},
        })
        CHRONOS_PIPELINE = pipelines["amazon/chronos-t5-large"]
        return CHRONOS_PIPELINE
