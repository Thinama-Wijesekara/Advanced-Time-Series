"""Opt-in publication of derived outputs to the public GitHub repository."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PublishSettings:
    enabled: bool
    error: str | None = None


@dataclass(frozen=True)
class PublishOutcome:
    status: str
    message: str | None = None


PUBLISHABLE_FILES = (
    Path("data/clean/train_cleaned_data/daily_total_sales.csv"),
    Path("data/clean/train_cleaned_data/daily_total_sales_clean.csv"),
    Path("data/forecasts/latest_forecast.csv"),
)
ENV_KEYS = {"PUBLISH_TO_GITHUB", "GITHUB_TOKEN_PUBLISH"}
GITHUB_API = "https://api.github.com"


class PublishError(RuntimeError):
    """A safe-to-display publication error that never includes credentials."""


def _dotenv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_local_environment(repo_root: Path) -> str | None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return None
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "Could not read .env. Check its file permissions."

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in ENV_KEYS and not os.environ.get(key, "").strip():
            os.environ[key] = _dotenv_value(value)
    return None


def publish_settings(repo_root: Path) -> PublishSettings:
    load_error = _load_local_environment(repo_root)
    if load_error:
        return PublishSettings(enabled=False, error=load_error)

    configured = os.environ.get("PUBLISH_TO_GITHUB", "false").strip().lower()
    if configured not in {"true", "false"}:
        return PublishSettings(
            enabled=False,
            error="PUBLISH_TO_GITHUB must be set to true or false.",
        )
    return PublishSettings(enabled=configured == "true")


def publish_runtime_outputs(repo_root: Path, runtime_data_dir: Path) -> PublishOutcome:
    settings = publish_settings(repo_root)
    if settings.error:
        return PublishOutcome("failed", settings.error)
    if not settings.enabled:
        return PublishOutcome("disabled")

    token = os.environ.get("GITHUB_TOKEN_PUBLISH", "").strip()
    if not token:
        return PublishOutcome(
            "failed",
            "GitHub publishing is enabled, but GITHUB_TOKEN_PUBLISH is missing. "
            "Add it to the ignored .env file and restart the app. Local results "
            "are unchanged.",
        )

    try:
        for relative_path in PUBLISHABLE_FILES:
            if not (runtime_data_dir / relative_path).is_file():
                raise PublishError(
                    "A required derived output is missing. Local results are unchanged."
                )

        git_env = _git_environment(token)
        owner, repository = _github_repository(repo_root, git_env)
        _publish_commit(repo_root, runtime_data_dir, git_env)
    except PublishError as exc:
        return PublishOutcome("failed", str(exc))

    try:
        _dispatch_workflow(owner, repository, token)
    except PublishError as exc:
        return PublishOutcome(
            "dispatch_failed",
            "The derived files were published to the public repository, but the "
            f"GitHub pipeline could not be started: {exc}",
        )

    return PublishOutcome(
        "published",
        "Aggregated history, cleaned history, and latest forecast were published "
        "to the public repository; the GitHub pipeline was started.",
    )


def _git_environment(token: str) -> dict[str, str]:
    authorization = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {authorization}",
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
        }
    )
    return environment


def _run_git(
    directory: Path,
    arguments: list[str],
    environment: dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if check:
            raise PublishError(
                f"Git {arguments[0]} could not complete. Local results are unchanged."
            ) from exc
        return subprocess.CompletedProcess(["git", *arguments], 1, "", "")

    if check and result.returncode:
        message = {
            "fetch": "Could not fetch the latest public repository state.",
            "worktree": "Could not prepare an isolated publication checkout.",
            "add": "Could not stage the three derived output files.",
            "commit": "Could not commit the derived output files.",
            "pull": (
                "Could not rebase onto the latest public main branch. "
                "Check for concurrent changes or conflicts."
            ),
            "push": (
                "Could not push the derived outputs to the public main branch. "
                "Check token permissions and branch protection."
            ),
            "branch": "Could not clean up the temporary publication branch.",
        }.get(arguments[0], "A Git operation failed.")
        raise PublishError(f"{message} Local runtime results are unchanged.")
    return result


def _github_repository(
    repo_root: Path, environment: dict[str, str]
) -> tuple[str, str]:
    result = _run_git(
        repo_root,
        ["config", "--get", "remote.origin.url"],
        environment,
    )
    remote = result.stdout.strip()
    parsed = urlparse(remote)
    if parsed.scheme in {"https", "http"}:
        host = parsed.hostname
        path = parsed.path
        has_embedded_credentials = parsed.username is not None or parsed.password is not None
    elif remote.startswith("git@github.com:"):
        host = "github.com"
        path = remote.split(":", 1)[1]
        has_embedded_credentials = False
    else:
        host = None
        path = ""
        has_embedded_credentials = False

    if host != "github.com":
        raise PublishError(
            "GitHub publishing requires origin to point to github.com. "
            "Local runtime results are unchanged."
        )
    if has_embedded_credentials:
        raise PublishError(
            "Remove credentials from the origin URL; publishing credentials must "
            "come only from GITHUB_TOKEN_PUBLISH. Local runtime results are unchanged."
        )
    parts = path.strip("/").removesuffix(".git").split("/")
    if len(parts) != 2 or not all(parts):
        raise PublishError(
            "Could not identify the GitHub repository from origin. "
            "Local runtime results are unchanged."
        )
    return parts[0], parts[1]


def _publish_commit(
    repo_root: Path,
    runtime_data_dir: Path,
    environment: dict[str, str],
) -> None:
    worktree_parent = repo_root / "runtime" / "publish-worktrees"
    worktree_parent.mkdir(parents=True, exist_ok=True)
    worktree = worktree_parent / uuid.uuid4().hex
    branch = f"local-publish-{uuid.uuid4().hex}"
    worktree_added = False
    try:
        _run_git(repo_root, ["fetch", "origin", "main"], environment)
        _run_git(
            repo_root,
            ["worktree", "add", "-b", branch, str(worktree), "origin/main"],
            environment,
        )
        worktree_added = True

        for relative_path in PUBLISHABLE_FILES:
            destination = worktree / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runtime_data_dir / relative_path, destination)

        _run_git(
            worktree,
            ["add", "-f", "--", *(str(path) for path in PUBLISHABLE_FILES)],
            environment,
        )
        staged = _run_git(
            worktree,
            ["diff", "--cached", "--quiet"],
            environment,
            check=False,
        )
        if staged.returncode == 0:
            return

        _run_git(
            worktree,
            [
                "-c",
                "user.name=store-sales-pipeline",
                "-c",
                "user.email=store-sales-pipeline@users.noreply.github.com",
                "commit",
                "-m",
                "Publish local store sales forecast outputs",
                "-m",
                "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
            ],
            environment,
        )
        _run_git(worktree, ["pull", "--rebase", "origin", "main"], environment)

        for attempt in range(3):
            pushed = _run_git(
                worktree,
                ["push", "origin", "HEAD:refs/heads/main"],
                environment,
                check=False,
            )
            if pushed.returncode == 0:
                break
            if attempt == 2:
                raise PublishError(
                    "Could not push the derived outputs to public main after "
                    "retrying against the latest commit. Check token permissions "
                    "and branch protection. Local runtime results are unchanged."
                )
            _run_git(worktree, ["pull", "--rebase", "origin", "main"], environment)
    finally:
        if worktree_added:
            _run_git(
                repo_root,
                ["worktree", "remove", "--force", str(worktree)],
                environment,
                check=False,
            )
        _run_git(
            repo_root,
            ["branch", "-D", branch],
            environment,
            check=False,
        )


def _dispatch_workflow(owner: str, repository: str, token: str) -> None:
    endpoint = (
        f"{GITHUB_API}/repos/{owner}/{repository}/actions/workflows/"
        "data_pipeline.yml/dispatches"
    )
    request = Request(
        endpoint,
        data=json.dumps({"ref": "main", "inputs": {}}).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 204:
                raise PublishError(
                    "GitHub returned an unexpected response while starting the workflow."
                )
    except HTTPError as exc:
        if exc.code in {401, 403}:
            detail = (
                "GitHub rejected the token. It needs Contents: write and "
                "Actions: write access to this repository."
            )
        elif exc.code == 404:
            detail = (
                "GitHub could not find the workflow on main. Confirm "
                "data_pipeline.yml is enabled on the default branch."
            )
        else:
            detail = f"GitHub returned HTTP {exc.code}."
        raise PublishError(detail) from None
    except (URLError, TimeoutError, OSError):
        raise PublishError(
            "Could not reach the GitHub Actions API. Check the network and token."
        ) from None
