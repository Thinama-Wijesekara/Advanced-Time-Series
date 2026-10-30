from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.config import engine
from db.models import CleanSeries, DataVersion, DailyTotal


def main() -> None:
    with Session(engine()) as session:
        existing = session.scalar(select(DataVersion).where(DataVersion.data_version == 1))
        if existing:
            print("Baseline data_version 1 already exists.")
            return
        daily_count = session.scalar(select(__import__('sqlalchemy').func.count()).select_from(DailyTotal))
        clean_count = session.scalar(select(__import__('sqlalchemy').func.count()).select_from(CleanSeries))
        if not daily_count or not clean_count:
            raise SystemExit("Baseline tables are empty. Run db.init and db.load_history first.")
        last_date = session.scalar(select(__import__('sqlalchemy').func.max(CleanSeries.date)))
        session.add(DataVersion(data_version=1, last_date=last_date, rows=clean_count, run_id=None))
        session.commit()
        print("Created baseline data_version 1.")


if __name__ == "__main__":
    main()
