from __future__ import annotations

import argparse

import sqlalchemy.exc
from db.config import engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the PostgreSQL database connection")
    parser.parse_args()
    try:
        with engine().connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except sqlalchemy.exc.OperationalError as exc:
        text = str(exc.orig).lower()
        if "password authentication failed" in text:
            print("Database authentication failed: wrong user or password.")
        elif "does not exist" in text and "database" in text:
            print("Database does not exist.")
        elif "could not connect to server" in text or "connection refused" in text:
            print("PostgreSQL server is not running or is not reachable.")
        else:
            print(f"Database connection failed: {exc.orig}")
        return 1
    print("Database connection OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
