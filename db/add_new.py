from __future__ import annotations

import argparse
from pathlib import Path

from db.ingest import add_batch, normalize_upload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and add a store-level upload to sales_new")
    parser.add_argument("path", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--source", default="upload")
    args = parser.parse_args()
    if args.path.suffix.lower() not in {".csv", ".xlsx"}:
        print("Upload must be a CSV or Excel (.xlsx) file.")
        return 1
    try:
        frame = normalize_upload(args.path)
        batch_id = add_batch(frame, args.source, args.path.name, replace=args.replace)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc))
        return 1
    print(f"Added batch {batch_id} with {len(frame)} pending rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
