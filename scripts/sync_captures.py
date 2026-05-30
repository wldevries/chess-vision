"""Sync the dataset between the local `data/` tree and MinIO.

Credentials/endpoint come from `.env` (see .env.example). Sync is size-based:
unchanged files are skipped, so re-running is cheap and resumable.

Usage:
    uv run python scripts/sync_captures.py up            # local -> bucket
    uv run python scripts/sync_captures.py down          # bucket -> local
    uv run python scripts/sync_captures.py up --dry-run  # show what would move
    uv run python scripts/sync_captures.py up --local data/store --prefix store

The flat layout keeps the store images under `data/store/`, the index at
`data/labels.jsonl`, originals under `data/source/`, and the reference JSON at the
`data/` root -- sync whichever subtree you want with `--local`/`--prefix`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessvision.data.storage import (
    StorageConfig,
    download_prefix,
    get_client,
    upload_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "direction",
        choices=["up", "down"],
        help="up: local->bucket, down: bucket->local",
    )
    p.add_argument("--local", type=Path, default=Path("data/store"), help="local dataset dir")
    p.add_argument("--prefix", default="store", help="key prefix within the bucket")
    p.add_argument("--dry-run", action="store_true", help="list transfers without moving bytes")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-upload even same-size objects (e.g. to repair Content-Type); up only",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = StorageConfig.from_env()
    client = get_client(config)

    where = f"{config.bucket}/{args.prefix.strip('/')}"
    if args.direction == "up":
        print(f"uploading {args.local} -> {config.endpoint_url}/{where}")
        result = upload_dir(
            args.local,
            args.prefix,
            config=config,
            client=client,
            dry_run=args.dry_run,
            force=args.force,
        )
    else:
        print(f"downloading {config.endpoint_url}/{where} -> {args.local}")
        result = download_prefix(
            args.prefix,
            args.local,
            config=config,
            client=client,
            dry_run=args.dry_run,
        )

    for key in result.transferred:
        print(f"  {'(dry-run) ' if args.dry_run else ''}{key}")
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
