"""Regression tests for upload handling around missing Christmas dates."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import functions


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE = REPO_ROOT / "data"
HISTORY_START = pd.Timestamp("2017-01-01")


class UploadCalendarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name) / "data"
        shutil.copytree(DATA_SOURCE, self.data_dir)
        self.upload_dir = self.data_dir / "raw" / "uploads"

        paths = {
            "DATA_DIR": self.data_dir,
            "RAW_DIR": self.data_dir / "raw",
            "CLEAN_DIR": self.data_dir / "clean",
            "MASTER_DIR": self.data_dir / "clean" / "train_cleaned_data",
            "REFERENCE_DIR": self.data_dir / "reference",
            "BACKUPS_DIR": self.data_dir / "backups",
            "FORECASTS_DIR": self.data_dir / "forecasts",
            "UPLOADS_DIR": self.upload_dir,
            "MASTER_RAW_PATH": self.data_dir
            / "clean"
            / "train_cleaned_data"
            / "daily_total_sales.csv",
            "SEED_RAW_PATH": self.data_dir / "raw" / "daily_total_sales_initial.csv",
            "CLEAN_PATH": self.data_dir
            / "clean"
            / "train_cleaned_data"
            / "daily_total_sales_clean.csv",
            "HOLIDAYS_PATH": self.data_dir / "reference" / "holidays_events.csv",
            "STORES_PATH": self.data_dir / "reference" / "stores.csv",
        }
        for name, value in paths.items():
            context = patch.object(functions, name, value)
            context.start()
            self.addCleanup(context.stop)

    def _set_history_end(self, end_date: str) -> None:
        master = pd.read_csv(functions.MASTER_RAW_PATH, parse_dates=["timestamp_utc"])
        cutoff = pd.Timestamp(end_date)
        master = master[master["timestamp_utc"] <= cutoff].copy()
        if cutoff > master["timestamp_utc"].max():
            extension_dates = pd.date_range(
                master["timestamp_utc"].max() + pd.Timedelta(days=1),
                cutoff,
                freq="D",
            )
            extension = pd.DataFrame(
                {"timestamp_utc": extension_dates, "sales": 500_000.0}
            )
            master = pd.concat([master, extension], ignore_index=True)
        master.to_csv(functions.MASTER_RAW_PATH, index=False)

    def _store_upload(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        stores = pd.read_csv(functions.STORES_PATH)["store_nbr"].tolist()
        return pd.DataFrame(
            [
                {
                    "date": date,
                    "store_nbr": store,
                    "family": "TEST",
                    "sales": 100.0,
                }
                for date in dates
                for store in stores
            ]
        )

    def _save_upload(self, dataframe: pd.DataFrame, name: str) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.upload_dir / name
        dataframe.to_csv(path, index=False)
        return path

    def test_store_upload_fills_christmas_and_preprocesses(self) -> None:
        self._set_history_end("2017-08-15")
        master = pd.read_csv(functions.MASTER_RAW_PATH, parse_dates=["timestamp_utc"])
        extension_dates = pd.date_range("2017-08-16", "2017-12-19", freq="D")
        extension = pd.DataFrame(
            {"timestamp_utc": extension_dates, "sales": 500_000.0}
        )
        pd.concat([master, extension], ignore_index=True).to_csv(
            functions.MASTER_RAW_PATH, index=False
        )

        upload_dates = pd.date_range("2017-12-20", "2017-12-31", freq="D")
        upload = self._store_upload(upload_dates.delete(5))
        path = self._save_upload(upload, "christmas_store.csv")

        daily = functions.aggregate_upload(functions.ingest_upload(path))
        christmas = daily.loc[
            daily["timestamp_utc"] == pd.Timestamp("2017-12-25"), "sales"
        ]
        self.assertEqual(christmas.tolist(), [0.0])
        functions.update_master_raw(daily)

        clean = functions.preprocess().set_index("timestamp_utc")
        self.assertTrue(bool(clean.loc["2017-12-25", "is_closure"]))
        self.assertEqual(clean.loc["2017-12-25", "sales_raw"], 0.0)
        self.assertGreater(clean.loc["2017-12-25", "sales"], 0.0)
        self.assertFalse(clean["sales"].isna().any())

    def test_daily_total_upload_fills_missing_christmas(self) -> None:
        self._set_history_end("2017-12-19")
        dates = pd.date_range("2017-12-20", "2017-12-31", freq="D")
        dates = dates[~((dates.month == 12) & (dates.day == 25))]
        upload = pd.DataFrame({"timestamp_utc": dates, "sales": 100.0})
        path = self._save_upload(upload, "christmas_daily.csv")

        daily = functions.aggregate_upload(functions.ingest_upload(path))
        christmas = daily.loc[
            daily["timestamp_utc"] == pd.Timestamp("2017-12-25"), "sales"
        ]
        self.assertEqual(christmas.tolist(), [0.0])

    def test_missing_non_christmas_date_is_rejected(self) -> None:
        self._set_history_end("2017-12-19")
        dates = pd.date_range("2017-12-20", "2017-12-31", freq="D")
        dates = dates[
            ~dates.isin(
                [pd.Timestamp("2017-12-25"), pd.Timestamp("2017-12-26")]
            )
        ]
        path = self._save_upload(self._store_upload(dates), "missing_dec26.csv")

        with self.assertRaisesRegex(
            ValueError, "Upload has missing calendar dates: 2017-12-26"
        ):
            functions.ingest_upload(path)
        self.assertFalse(list(self.upload_dir.glob("upload_*")))

        daily_path = self._save_upload(
            pd.DataFrame({"timestamp_utc": dates, "sales": 100.0}),
            "daily_missing_dec26.csv",
        )
        with self.assertRaisesRegex(
            ValueError, "Upload has missing calendar dates: 2017-12-26"
        ):
            functions.ingest_upload(daily_path)
        self.assertFalse(list(self.upload_dir.glob("upload_*")))

    def test_repeated_last_twenty_days_upload_has_no_duplicates(self) -> None:
        self._set_history_end("2017-07-26")
        dates = pd.date_range("2017-07-27", "2017-08-15", freq="D")
        path = self._save_upload(self._store_upload(dates), "last_twenty_days.csv")

        for _ in range(2):
            daily = functions.aggregate_upload(functions.ingest_upload(path))
            functions.update_master_raw(daily)

        master = pd.read_csv(functions.MASTER_RAW_PATH, parse_dates=["timestamp_utc"])
        self.assertEqual(len(master), 1688)
        self.assertFalse(master["timestamp_utc"].duplicated().any())

    def test_one_missing_store_is_rejected_without_archiving(self) -> None:
        self._set_history_end("2017-07-26")
        dates = pd.date_range("2017-07-27", "2017-08-15", freq="D")
        upload = self._store_upload(dates)
        upload = upload.drop(upload.index[0])
        path = self._save_upload(upload, "missing_store.csv")
        original_master = functions.MASTER_RAW_PATH.read_bytes()

        with self.assertRaisesRegex(ValueError, "Invalid date"):
            functions.ingest_upload(path)

        self.assertEqual(functions.MASTER_RAW_PATH.read_bytes(), original_master)
        self.assertFalse(list(self.upload_dir.glob("upload_*")))

    def test_three_day_gap_is_rejected_without_archiving(self) -> None:
        self._set_history_end("2017-07-26")
        dates = pd.date_range("2017-07-27", "2017-08-15", freq="D")
        dates = dates[~dates.isin(pd.date_range("2017-08-01", "2017-08-03"))]
        path = self._save_upload(self._store_upload(dates), "three_day_gap.csv")
        original_master = functions.MASTER_RAW_PATH.read_bytes()

        with self.assertRaisesRegex(ValueError, "Upload has missing calendar dates"):
            functions.ingest_upload(path)

        self.assertEqual(functions.MASTER_RAW_PATH.read_bytes(), original_master)
        self.assertFalse(list(self.upload_dir.glob("upload_*")))


if __name__ == "__main__":
    unittest.main()
