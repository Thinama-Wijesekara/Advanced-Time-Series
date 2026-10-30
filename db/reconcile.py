from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from db.config import engine
from db.models import Batch, CleanSeries, DailyTotal, DataVersion, Forecast, SalesNew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_BASELINE = PROJECT_ROOT / "data" / "raw" / "daily_total_sales_initial.csv"
CLEAN_BASELINE = PROJECT_ROOT / "data" / "clean" / "train_cleaned_data" / "daily_total_sales_clean.csv"


def reconcile_removed_data(session: Session) -> bool:
    """Remove derived database data whose source rows no longer exist in sales_new."""
    processed_count = session.scalar(select(__import__('sqlalchemy').func.count()).select_from(SalesNew).where(SalesNew.status == "processed")) or 0
    if processed_count:
        return False

    latest_daily = session.scalar(select(__import__('sqlalchemy').func.max(DailyTotal.date)))
    latest_clean = session.scalar(select(__import__('sqlalchemy').func.max(CleanSeries.date)))
    if latest_daily is not None and latest_daily.isoformat() <= "2017-08-15" and latest_clean is not None and latest_clean.isoformat() <= "2017-08-15":
        return False

    daily = pd.read_csv(DAILY_BASELINE).rename(columns={"timestamp_utc": "date"})
    clean = pd.read_csv(CLEAN_BASELINE).rename(columns={"timestamp_utc": "date"})
    clean["sales"] = pd.to_numeric(clean["sales"])
    clean["sales_raw"] = pd.to_numeric(clean["sales_raw"])
    clean["is_closure"] = clean["is_closure"].astype(bool)

    with engine().begin() as connection:
        connection.execute(delete(Forecast))
        connection.execute(delete(DailyTotal))
        connection.execute(delete(CleanSeries))
        connection.execute(delete(DataVersion).where(DataVersion.data_version > 1))
        connection.execute(delete(Batch).where(Batch.status == "processed"))
        connection.execute(__import__('sqlalchemy').insert(DailyTotal), [
            {"date": pd.Timestamp(row.date).date(), "sales": round(float(row.sales), 9)}
            for row in daily.itertuples(index=False)
        ])
        connection.execute(__import__('sqlalchemy').insert(CleanSeries), [
            {"date": pd.Timestamp(row.date).date(), "sales": round(float(row.sales), 9), "sales_raw": round(float(row.sales_raw), 9), "is_closure": bool(row.is_closure)}
            for row in clean.itertuples(index=False)
        ])
    return True


def reconcile_now() -> bool:
    with Session(engine()) as session:
        changed = reconcile_removed_data(session)
        session.commit()
    return changed
