from __future__ import annotations

import argparse

from db.config import engine
from db.models import Base


def main() -> int:
    parser = argparse.ArgumentParser(description="Create store-sales database tables")
    parser.parse_args()
    eng = engine()
    with eng.connect() as connection:
        connection.exec_driver_sql("SELECT 1")
    Base.metadata.create_all(eng)
    print("Database tables created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
