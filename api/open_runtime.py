"""Open the local runtime folder in the operating system's file browser."""

from __future__ import annotations

import os
import subprocess
import sys

from api import jobs


def main() -> None:
    jobs.ensure_runtime()
    runtime_path = str(jobs.RUNTIME_DIR)
    if sys.platform == "win32":
        os.startfile(runtime_path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", runtime_path])
    else:
        subprocess.Popen(["xdg-open", runtime_path])
    print(f"Opened runtime directory: {runtime_path}")


if __name__ == "__main__":
    main()
