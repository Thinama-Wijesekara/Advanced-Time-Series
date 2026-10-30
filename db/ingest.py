from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from db.config import engine
from db.models import Batch, DailyTotal, SalesNew
from pipeline import functions as pipeline_functions


def normalize_upload(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        source = pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        source = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    else:
        raise ValueError("Upload must be a CSV or Excel (.xlsx) file.")
    if not {"date", "store_nbr", "family", "sales"}.issubset(source.columns):
        raise ValueError("Upload must contain columns 'date', 'store_nbr', 'family', and 'sales'; missing columns are required.")
    normalized = pipeline_functions._normalize_store_upload(source)
    if "onpromotion" in source.columns:
        promotion = pd.to_numeric(source["onpromotion"], errors="coerce")
        if promotion.isna().any() or (promotion < 0).any():
            raise ValueError("Upload has invalid onpromotion values.")
        normalized["onpromotion"] = promotion.to_numpy(dtype=int)
    else:
        normalized["onpromotion"] = 0
    return normalized


def add_batch(dataframe: pd.DataFrame, source: str, original_file_name: str, replace: bool = False) -> int:
    if dataframe.empty:
        raise ValueError("Upload contains no data rows.")
    frame = dataframe.copy()
    frame["date"] = pd.to_datetime(frame["date"], format="mixed", errors="coerce")
    required = {"date", "store_nbr", "family", "sales", "onpromotion"}
    if not required.issubset(frame.columns):
        raise ValueError("Upload must contain date, store_nbr, family, sales and onpromotion columns.")
    eng = engine()
    with Session(eng) as session:
        db_dates = session.execute(select(DailyTotal.date).order_by(DailyTotal.date)).scalars().all()
        if not db_dates:
            raise ValueError("History has not been loaded; run db.load_history first.")
        history_end = pd.Timestamp(max(db_dates))
        pipeline_functions._validate_upload_dates(pd.Series(frame["date"].drop_duplicates()), history_end)
        first, last = frame["date"].min(), frame["date"].max()
        overlapping = set(frame["date"].dt.date).intersection(set(pd.Timestamp(d).date() for d in db_dates))
        if overlapping and not replace:
            raise ValueError("Upload overlaps existing history; use --replace to replace those dates.")
        result = session.execute(insert(Batch).values(source=source, original_file_name=original_file_name,
            first_date=first.date(), last_date=last.date(), rows=len(frame), status="pending", message=None).returning(Batch.batch_id))
        batch_id = result.scalar_one()
        session.execute(insert(SalesNew), [dict(batch_id=batch_id, date=row.date.to_pydatetime().date(),
            store_nbr=int(row.store_nbr), family=str(row.family), sales=round(float(row.sales), 3),
            onpromotion=int(row.onpromotion), status="pending") for row in frame.itertuples(index=False)])
        session.commit()
        return int(batch_id)
