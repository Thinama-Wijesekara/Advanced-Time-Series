from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from api import publishing


class PublishingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        os.environ.pop("PUBLISH_TO_GITHUB", None)
        os.environ.pop("GITHUB_TOKEN_PUBLISH", None)
        self.addCleanup(self.environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime_data = self.root / "runtime-data"
        for relative_path in publishing.PUBLISHABLE_FILES:
            path = self.runtime_data / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("local version\n", encoding="utf-8")

    def test_default_and_explicit_false_leave_outputs_local(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        result = publishing.publish_runtime_outputs(repo, self.runtime_data)
        self.assertEqual(result.status, "disabled")
        self.assertIsNone(result.message)
        os.environ["PUBLISH_TO_GITHUB"] = "false"
        self.assertFalse(publishing.publish_settings(repo).enabled)

    def test_enabled_without_token_keeps_local_outputs(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        (repo / ".env").write_text(
            "PUBLISH_TO_GITHUB=true\nGITHUB_TOKEN_PUBLISH=\n",
            encoding="utf-8",
        )
        result = publishing.publish_runtime_outputs(repo, self.runtime_data)
        self.assertEqual(result.status, "failed")
        self.assertIn("GITHUB_TOKEN_PUBLISH is missing", result.message or "")
        for relative_path in publishing.PUBLISHABLE_FILES:
            self.assertEqual(
                (self.runtime_data / relative_path).read_text(encoding="utf-8"),
                "local version\n",
            )

    def test_success_publishes_only_derived_outputs_then_dispatches(self) -> None:
        repo, remote = self._make_repository()
        (repo / ".env").write_text(
            "PUBLISH_TO_GITHUB=true\nGITHUB_TOKEN_PUBLISH=test-token-not-for-output\n",
            encoding="utf-8",
        )
        for relative_path in publishing.PUBLISHABLE_FILES:
            path = self.runtime_data / relative_path
            path.write_text("published version\n", encoding="utf-8")
        raw_upload = self.runtime_data / "raw" / "uploads" / "original.csv"
        raw_upload.parent.mkdir(parents=True)
        raw_upload.write_text("store-level upload\n", encoding="utf-8")

        with (
            mock.patch.object(
                publishing,
                "_github_repository",
                return_value=("owner", "repository"),
            ),
            mock.patch.object(publishing, "_dispatch_workflow") as dispatch,
        ):
            result = publishing.publish_runtime_outputs(repo, self.runtime_data)

        self.assertEqual(result.status, "published")
        self.assertNotIn("test-token-not-for-output", repr(result))
        dispatch.assert_called_once_with("owner", "repository", "test-token-not-for-output")
        changed = self._git(
            repo,
            ["--git-dir", str(remote), "diff-tree", "--no-commit-id", "--name-only", "-r", "main"],
        ).stdout.splitlines()
        self.assertEqual(set(changed), {path.as_posix() for path in publishing.PUBLISHABLE_FILES})
        tree = self._git(
            repo,
            ["--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"],
        ).stdout.splitlines()
        self.assertEqual(set(tree), {path.as_posix() for path in publishing.PUBLISHABLE_FILES})
        for relative_path in publishing.PUBLISHABLE_FILES:
            published = self._git(
                repo,
                ["--git-dir", str(remote), "show", f"main:{relative_path.as_posix()}"],
            ).stdout
            self.assertEqual(published, "published version\n")
        self.assertNotIn(".env", tree)

    def test_push_failure_keeps_local_results_and_does_not_dispatch(self) -> None:
        repo, remote = self._make_repository()
        (repo / ".env").write_text(
            "PUBLISH_TO_GITHUB=true\nGITHUB_TOKEN_PUBLISH=test-token\n",
            encoding="utf-8",
        )
        for relative_path in publishing.PUBLISHABLE_FILES:
            path = self.runtime_data / relative_path
            path.write_text("new local version\n", encoding="utf-8")

        original_run_git = publishing._run_git

        def reject_push(
            directory: Path,
            arguments: list[str],
            environment: dict[str, str],
            *,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            if arguments[0] == "push":
                return subprocess.CompletedProcess(["git", *arguments], 1, "", "rejected")
            return original_run_git(
                directory,
                arguments,
                environment,
                check=check,
            )

        with (
            mock.patch.object(
                publishing,
                "_github_repository",
                return_value=("owner", "repository"),
            ),
            mock.patch.object(publishing, "_run_git", side_effect=reject_push),
            mock.patch.object(publishing, "_dispatch_workflow") as dispatch,
        ):
            result = publishing.publish_runtime_outputs(repo, self.runtime_data)

        self.assertEqual(result.status, "failed")
        self.assertIn("Local runtime results are unchanged", result.message or "")
        dispatch.assert_not_called()
        for relative_path in publishing.PUBLISHABLE_FILES:
            self.assertEqual(
                (self.runtime_data / relative_path).read_text(encoding="utf-8"),
                "new local version\n",
            )
            remote_contents = self._git(
                repo,
                ["--git-dir", str(remote), "show", f"main:{relative_path.as_posix()}"],
            ).stdout
            self.assertEqual(remote_contents, "baseline\n")

    def test_dispatch_failure_reports_that_files_are_already_public(self) -> None:
        repo, remote = self._make_repository()
        (repo / ".env").write_text(
            "PUBLISH_TO_GITHUB=true\nGITHUB_TOKEN_PUBLISH=test-token\n",
            encoding="utf-8",
        )
        for relative_path in publishing.PUBLISHABLE_FILES:
            path = self.runtime_data / relative_path
            path.write_text("public version\n", encoding="utf-8")

        with (
            mock.patch.object(
                publishing,
                "_github_repository",
                return_value=("owner", "repository"),
            ),
            mock.patch.object(
                publishing,
                "_dispatch_workflow",
                side_effect=publishing.PublishError(
                    "GitHub rejected the publishing token."
                ),
            ),
        ):
            result = publishing.publish_runtime_outputs(repo, self.runtime_data)

        self.assertEqual(result.status, "dispatch_failed")
        self.assertIn("were published to the public repository", result.message or "")
        self.assertNotIn("test-token", repr(result))
        for relative_path in publishing.PUBLISHABLE_FILES:
            remote_contents = self._git(
                repo,
                ["--git-dir", str(remote), "show", f"main:{relative_path.as_posix()}"],
            ).stdout
            self.assertEqual(remote_contents, "public version\n")

    def test_origin_credentials_are_rejected_without_exposing_them(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        self._git(repo, ["init", "--initial-branch=main"])
        self._git(
            repo,
            [
                "remote",
                "add",
                "origin",
                "https://user:embedded-secret@github.com/owner/repository.git",
            ],
        )
        with self.assertRaises(publishing.PublishError) as raised:
            publishing._github_repository(repo, os.environ.copy())
        self.assertNotIn("embedded-secret", str(raised.exception))

    def _make_repository(self) -> tuple[Path, Path]:
        repo = self.root / "repo"
        remote = self.root / "remote.git"
        repo.mkdir()
        self._git(repo, ["init", "--initial-branch=main"])
        self._git(repo, ["config", "user.name", "Test"])
        self._git(repo, ["config", "user.email", "test@example.invalid"])
        for relative_path in publishing.PUBLISHABLE_FILES:
            path = repo / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("baseline\n", encoding="utf-8")
        self._git(repo, ["add", "-f", "--", *(str(path) for path in publishing.PUBLISHABLE_FILES)])
        self._git(repo, ["commit", "-m", "baseline"])
        self._git(repo, ["init", "--bare", str(remote)])
        self._git(repo, ["remote", "add", "origin", str(remote)])
        self._git(repo, ["push", "-u", "origin", "main"])
        return repo, remote

    @staticmethod
    def _git(directory: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
