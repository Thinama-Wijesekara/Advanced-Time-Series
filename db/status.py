from __future__ import annotations

import argparse

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.config import engine
from db.models import Batch, CleanSeries, DailyTotal, DataVersion, Forecast, Run, SalesHistory, SalesNew


def main() -> int:
    parser = argparse.ArgumentParser(description="Show store-sales database status")
    parser.parse_args()
    tables = (SalesHistory, SalesNew, Batch, DailyTotal, CleanSeries, DataVersion, Forecast, Run)
    with Session(engine()) as session:
        for table in tables:
            count = session.scalar(select(func.count()).select_from(table))
            print(f"{table.__tablename__}: {count:,}")
        for table, column in ((SalesHistory, "date"), (SalesNew, "date"), (DailyTotal, "date"), (CleanSeries, "date")):
            bounds = session.execute(select(func.min(getattr(table, column)), func.max(getattr(table, column)))).one()
            print(f"{table.__tablename__} dates: {bounds[0]} to {bounds[1]}")
        pending = session.scalar(select(func.count()).select_from(SalesNew).where(SalesNew.status == "pending"))
        print(f"pending rows: {pending:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
