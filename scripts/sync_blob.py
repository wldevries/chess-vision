"""Sync the captured dataset between the local `data/` tree and Azure Blob Storage.

The off-machine home for the dataset is the `chess` container (account from the
`STORAGE_CONNECTION_STRING` in `.env`, shared with hive-vision). Sync is size-based and
resumable: a file moves only when it is missing on the destination or its size differs.

By default this syncs the three dataset parts and *nothing else* — an allowlist, so the
external/derived trees (`othersets/`, `yolo_chessred/`, `yolo_pose/`, `data.zip`,
`_blur_*`) are never touched:

    source/<relpath>   raw originals (incl. source/inbox/<photo>)  <- data/source/
    store/<relpath>    EXIF-normalized JPEGs                        <- data/store/
    labels.jsonl       the label index                             <- data/labels.jsonl

Usage:
    uv run python scripts/sync_blob.py up               # local -> container (all parts)
    uv run python scripts/sync_blob.py down             # container -> local
    uv run python scripts/sync_blob.py up --dry-run     # show what would move
    uv run python scripts/sync_blob.py up --part store  # just one part
    uv run python scripts/sync_blob.py up --local models/best.pt --prefix models/best.pt

Models (later) live in the same container under `models/`; upload one with --local/--prefix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from chessvision.data.blob import (
    BlobConfig,
    SyncResult,
    download_prefix,
    get_container_client,
    upload_dir,
    upload_file,
)


@dataclass(frozen=True)
class Part:
    name: str
    local: Path
    prefix: str
    is_file: bool


# The allowlist of what belongs off-machine. Anything not here stays local.
PARTS: tuple[Part, ...] = (
    Part("source", Path("data/source"), "source", is_file=False),
    Part("store", Path("data/store"), "store", is_file=False),
    Part("labels", Path("data/labels.jsonl"), "labels.jsonl", is_file=True),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("direction", choices=["up", "down"], help="up: local->container, down: reverse")
    p.add_argument(
        "--part",
        choices=[part.name for part in PARTS],
        action="append",
        help="limit to one or more dataset parts (default: all). Repeatable.",
    )
    p.add_argument("--local", type=Path, help="override: local path (with --prefix)")
    p.add_argument("--prefix", help="override: blob key/prefix (with --local)")
    p.add_argument("--dry-run", action="store_true", help="list transfers without moving bytes")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-transfer even same-size objects (e.g. to repair Content-Type); up only",
    )
    return p.parse_args(argv)


def _sync_one(
    direction: str,
    local: Path,
    prefix: str,
    is_file: bool,
    *,
    config: BlobConfig,
    container,
    dry_run: bool,
    force: bool,
) -> SyncResult:
    if direction == "up":
        if is_file:
            return upload_file(
                local, prefix, config=config, container=container, dry_run=dry_run, force=force
            )
        return upload_dir(
            local, prefix, config=config, container=container, dry_run=dry_run, force=force
        )
    # down: a file-part downloads its single blob into the parent dir.
    dest = local.parent if is_file else local
    return download_prefix(prefix, dest, config=config, container=container, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = BlobConfig.from_env()
    container = get_container_client(config)

    if bool(args.local) ^ bool(args.prefix):
        raise SystemExit("--local and --prefix must be given together")

    if args.local:
        jobs = [Part("custom", args.local, args.prefix, is_file=args.local.is_file())]
    else:
        wanted = set(args.part) if args.part else {part.name for part in PARTS}
        jobs = [part for part in PARTS if part.name in wanted]

    arrow = "->" if args.direction == "up" else "<-"
    print(f"{args.direction}: data/ {arrow} {config.container}/  (STORAGE_CONNECTION_STRING)")

    total = SyncResult(transferred=[], skipped=0)
    for part in jobs:
        if args.direction == "up" and not part.is_file and not part.local.exists():
            print(f"  [skip] {part.name}: {part.local} not present locally")
            continue
        result = _sync_one(
            args.direction,
            part.local,
            part.prefix,
            part.is_file,
            config=config,
            container=container,
            dry_run=args.dry_run,
            force=args.force,
        )
        for key in result.transferred:
            print(f"  {'(dry-run) ' if args.dry_run else ''}{key}")
        print(f"  [{part.name}] {result.summary()}")
        total.transferred.extend(result.transferred)
        total.skipped += result.skipped

    print(total.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
