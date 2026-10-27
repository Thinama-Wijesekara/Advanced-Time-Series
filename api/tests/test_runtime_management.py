from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api import jobs, main


class RuntimeManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "runtime" / "data"
        self.master_dir = self.data / "clean" / "train_cleaned_data"
        self.uploads_dir = self.data / "raw" / "uploads"
        self.forecasts_dir = self.data / "forecasts"
        self.backups_dir = self.data / "backups"
        self.state_dir = self.root / "runtime" / "state"
        for directory in (
            self.master_dir,
            self.uploads_dir,
            self.forecasts_dir,
            self.backups_dir,
            self.state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.master_path = self.master_dir / "daily_total_sales.csv"
        self.clean_path = self.master_dir / "daily_total_sales_clean.csv"
        pd.DataFrame(
            {"timestamp_utc": ["2017-08-15"], "sales": [100.0]}
        ).to_csv(self.master_path, index=False)
        pd.DataFrame(
            {"timestamp_utc": ["2017-08-15"], "sales": [100.0]}
        ).to_csv(self.clean_path, index=False)
        self.fake_functions = SimpleNamespace(
            MASTER_DIR=self.master_dir,
            MASTER_RAW_PATH=self.master_path,
            CLEAN_PATH=self.clean_path,
            UPLOADS_DIR=self.uploads_dir,
            FORECASTS_DIR=self.forecasts_dir,
            BACKUPS_DIR=self.backups_dir,
        )
        self.manager = jobs.JobManager()

        values = {
            "RUNTIME_DIR": self.root / "runtime",
            "RUNTIME_DATA_DIR": self.data,
            "INCOMING_DIR": self.root / "runtime" / "incoming",
            "STATE_DIR": self.state_dir,
            "JOBS_PATH": self.state_dir / "jobs.json",
            "MANIFEST_PATH": self.state_dir / "manifest.json",
        }
        for name, value in values.items():
            context = patch.object(jobs, name, value)
            context.start()
            self.addCleanup(context.stop)
        functions_context = patch.object(
            jobs, "pipeline_functions", return_value=self.fake_functions
        )
        functions_context.start()
        self.addCleanup(functions_context.stop)

    def test_xlsx_archive_is_an_untouched_original_copy(self) -> None:
        import sys

        sys.path.insert(0, str(jobs.REPO_ROOT / "pipeline"))
        import functions

        function_paths = {
            "MASTER_RAW_PATH": self.master_path,
            "STORES_PATH": self.data / "reference" / "stores.csv",
            "UPLOADS_DIR": self.uploads_dir,
        }
        (self.data / "reference").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"store_nbr": [1, 2]}).to_csv(
            function_paths["STORES_PATH"], index=False
        )
        function_paths_contexts = [
            patch.object(functions, name, value)
            for name, value in function_paths.items()
        ]
        for context in function_paths_contexts:
            context.start()
            self.addCleanup(context.stop)

        original = self.root / "incoming" / "client_upload.xlsx"
        original.parent.mkdir(parents=True)
        source = pd.DataFrame(
            {
                "date": ["2017-08-16", "2017-08-16"],
                "store_nbr": [1, 2],
                "family": ["GROCERY", "GROCERY"],
                "sales": [11.0, 12.0],
                "client_note": ["keep", "original"],
            }
        )
        source.to_excel(original, index=False, engine="openpyxl")
        original_bytes = original.read_bytes()

        job = jobs.build_job("client_upload.xlsx", ".xlsx", {"rows": 2})
        self.manager._state = {"jobs": [job]}
        normalized = self.manager._ingest(original, job, functions)

        archive = Path(self.manager.get_job(job["job_id"])["archive_path"])
        self.assertEqual(archive.read_bytes(), original_bytes)
        self.assertEqual(len(normalized), 2)
        self.assertTrue(archive.name.endswith("_client_upload.xlsx"))

    def test_watcher_debounces_missing_upload_assets(self) -> None:
        (self.state_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "uploads": [
                        {
                            "upload_id": "20261027_010203",
                            "archive": "upload_2026-10-27_010203_client.csv",
                            "aggregated_file": "daily_total_sales__through_2026-10-27.csv",
                            "cleaned_file": "daily_total_sales_clean__through_2026-10-27.csv",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        calls: list[tuple[str, set[str]]] = []
        self.manager._queue_or_start_rebuild = (
            lambda reason, upload_ids, **kwargs: calls.append((reason, upload_ids))
        )

        self.manager.watch_once(now=100.0)
        self.assertEqual(calls, [])
        self.manager.watch_once(now=116.0)
        self.assertEqual(calls, [("Upload removed", {"20261027_010203"})])

    def test_queued_rebuild_starts_when_upload_reservation_is_released(self) -> None:
        self.manager._reservation = True
        self.manager._queued_rebuild = {
            "reason": "Upload removed",
            "upload_ids": ["20261027_010203"],
            "live_edit": False,
        }
        calls: list[tuple[str, list[str]]] = []
        self.manager._start_rebuild = (
            lambda reason, upload_ids: calls.append((reason, upload_ids))
        )

        self.manager.release_reservation()

        self.assertEqual(calls, [("Upload removed", ["20261027_010203"])])
        self.assertIsNone(self.manager._queued_rebuild)

    def test_data_version_changes_when_live_file_changes(self) -> None:
        first = self.manager.data_version()
        with self.master_path.open("a", encoding="utf-8") as file:
            file.write("\n")
        self.assertNotEqual(first, self.manager.data_version())

    def test_health_returns_ok_with_an_explicit_data_error(self) -> None:
        missing_functions = SimpleNamespace(
            MASTER_RAW_PATH=self.root / "missing.csv"
        )
        with (
            patch.object(main.jobs, "pipeline_functions", return_value=missing_functions),
            patch.object(main.jobs.job_manager, "is_running", return_value=False),
        ):
            response = TestClient(main.app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "OK")
        self.assertIn("File not found", response.json()["data_error"])

    def test_dashboard_routes_serve_temp_history_forecast_and_page(self) -> None:
        forecast_path = self.forecasts_dir / "latest_forecast.csv"
        pd.DataFrame(
            {
                "date": ["2017-08-16"],
                "model": ["level_calendar_no_model"],
                "forecast": [123.0],
            }
        ).to_csv(forecast_path, index=False)
        with (
            patch.object(jobs, "pipeline_functions", return_value=self.fake_functions),
            patch.object(jobs.job_manager, "latest_new_range", return_value=None),
        ):
            client = TestClient(main.app)
            page = client.get("/")
            history = client.get("/history?days=1")
            forecast = client.get("/forecast/latest")
            runtime = client.get("/status")

        self.assertEqual(page.status_code, 200)
        self.assertIn("Advanced Time Series", page.text)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()[0]["date"], "2017-08-15")
        self.assertEqual(forecast.status_code, 200)
        self.assertEqual(forecast.json()["rows"][0]["forecast"], 123.0)
        self.assertEqual(runtime.status_code, 200)
        self.assertTrue(runtime.json()["files"][0]["exists"])

    def test_unsupported_upload_type_is_rejected_without_staging(self) -> None:
        incoming = self.root / "runtime" / "incoming"
        with (
            patch.object(jobs, "INCOMING_DIR", incoming),
            patch.object(jobs, "ensure_runtime"),
        ):
            response = TestClient(main.app).post(
                "/upload",
                files={"file": ("notes.txt", b"not a sales upload", "text/plain")},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Upload must be a .xlsx or .csv file.")
        self.assertFalse(incoming.exists())

    def test_failed_job_restores_previous_live_data_and_forecast(self) -> None:
        forecast_path = self.forecasts_dir / "latest_forecast.csv"
        forecast_path.write_text("original forecast", encoding="utf-8")
        archive = self.uploads_dir / "upload_2026-10-27_010203_failed.csv"
        archive.write_text("original upload", encoding="utf-8")
        original_master = self.master_path.read_bytes()
        original_clean = self.clean_path.read_bytes()
        original_forecast = forecast_path.read_bytes()
        job = jobs.build_job("failed.csv", ".csv", {}, kind="upload")
        job["archive_path"] = str(archive)
        self.manager._state = {"jobs": [job]}
        self.manager._active_job_id = job["job_id"]

        def fail_after_writing(job_id, upload_path, functions):
            self.master_path.write_text("changed master", encoding="utf-8")
            forecast_path.write_text("changed forecast", encoding="utf-8")
            raise RuntimeError("simulated forecast failure")

        with patch.object(self.manager, "_run_upload", side_effect=fail_after_writing):
            self.manager._run_job(job["job_id"], None)

        self.assertEqual(self.master_path.read_bytes(), original_master)
        self.assertEqual(self.clean_path.read_bytes(), original_clean)
        self.assertEqual(forecast_path.read_bytes(), original_forecast)
        self.assertFalse(archive.exists())
        self.assertEqual(self.manager.get_job(job["job_id"])["status"], "failed")

    def test_reset_requires_confirmation_before_touching_runtime(self) -> None:
        from api import reset_data

        with self.assertRaisesRegex(ValueError, "--confirm"):
            reset_data.reset_runtime(clear_forecast=False, confirmed=False)

    def test_reset_backs_up_data_preserves_forecast_and_clears_uploads(self) -> None:
        from api import reset_data

        archive = self.uploads_dir / "upload_2026-10-27_010203_client.csv"
        archive.write_text("client data", encoding="utf-8")
        forecast = self.forecasts_dir / "latest_forecast.csv"
        forecast.write_text("date,model,forecast\n", encoding="utf-8")

        backup = reset_data.reset_runtime(clear_forecast=False, confirmed=True)

        self.assertTrue((backup / "raw" / "uploads" / archive.name).is_file())
        self.assertFalse((self.data / "raw" / "uploads" / archive.name).exists())
        self.assertTrue((self.data / "forecasts" / "latest_forecast.csv").is_file())
        self.assertTrue((self.data / "clean" / "train_cleaned_data" / "daily_total_sales.csv").is_file())
        self.assertEqual(json.loads((self.state_dir / "manifest.json").read_text()), {"uploads": []})
        self.assertEqual(json.loads((self.state_dir / "jobs.json").read_text()), {"jobs": []})


if __name__ == "__main__":
    unittest.main()
