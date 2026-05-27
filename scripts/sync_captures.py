"""Sync the captured dataset between `data/captures/` and MinIO.

Credentials/endpoint come from `.env` (see .env.example). Sync is size-based:
unchanged files are skipped, so re-running is cheap and resumable.

Usage:
    uv run python scripts/sync_captures.py up            # local -> bucket
    uv run python scripts/sync_captures.py down          # bucket -> local
    uv run python scripts/sync_captures.py up --dry-run  # show what would move
    uv run python scripts/sync_captures.py up --prefix captures --local data/captures
    uv run python scripts/sync_captures.py tasks         # build LS pre-annotations -> bucket tasks/

The `tasks` command reads the local captures (the source of truth for marked
corners and FEN-projected piece estimates) and writes one Label Studio task-JSON
per frame straight to the bucket under `--tasks-prefix` (default `tasks`); nothing
is written locally. Point a Label Studio source storage at that prefix with
"Treat every bucket object as a source file" OFF. See chessvision/data/labelstudio.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chessvision.data import labelstudio as ls
from chessvision.data.publish import put_task
from chessvision.data.storage import (
    StorageConfig,
    download_prefix,
    get_bytes,
    get_client,
    upload_dir,
)

# Derived, regeneratable top-level files that must NOT travel to/from the bucket:
# label-studio.json is rebuilt by `annotations` from the export bucket, positions.json
# by save_capture_positions. Syncing them just clutters the (Label Studio source) bucket,
# and `down` would clobber the freshly regenerated local copies. The small reference
# sidecars (sets/boards/sessions/known_deviations.json) are NOT here — those do travel.
DERIVED_SIDECARS = frozenset({"label-studio.json", "positions.json"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "direction",
        choices=["up", "down", "tasks", "annotations"],
        help=(
            "up: local->bucket, down: bucket->local, tasks: build LS pre-annotations->bucket, "
            "annotations: pull LS annotation exports->merged label-studio.json"
        ),
    )
    p.add_argument("--local", type=Path, default=Path("data/captures"), help="local dataset dir")
    p.add_argument("--prefix", default="captures", help="key prefix within the bucket")
    p.add_argument("--dry-run", action="store_true", help="list transfers without moving bytes")
    p.add_argument(
        "--force",
        action="store_true",
        help="re-upload even same-size objects (e.g. to repair Content-Type); up only",
    )
    p.add_argument(
        "--tasks-prefix",
        default="tasks",
        help="bucket prefix for generated Label Studio tasks (tasks command)",
    )
    p.add_argument(
        "--model-version",
        default=ls.MODEL_VERSION,
        help="prediction model_version tag on generated tasks (tasks command)",
    )
    p.add_argument(
        "--with-boxes",
        action="store_true",
        help="also emit approximate piece bounding boxes (control 'boxes'); tasks command",
    )
    p.add_argument(
        "--annotations-bucket",
        default="chess-annotations",
        help="bucket Label Studio export storage writes to (annotations command)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/captures/label-studio.json"),
        help="merged export written by the annotations command",
    )
    return p.parse_args(argv)


def _image_size(args, config, client, session_id: str, filename: str, key: str):
    """Image dimensions for percentage conversion: local frame if present, else
    fetched from the bucket. Returns None if neither has it."""
    local = args.local / session_id / filename
    if local.exists():
        return ls.image_size_from_path(local)
    try:
        return ls.image_size_from_bytes(get_bytes(client, config.bucket, key))
    except Exception:
        return None


def run_tasks(args, config, client) -> int:
    where = f"{config.bucket}/{args.tasks_prefix.strip('/')}"
    print(f"building Label Studio tasks {args.local} -> {config.endpoint_url}/{where}")
    built = skipped = 0
    for session_id, record in ls.iter_records(args.local):
        filename = record["filename"]
        ikey = ls.image_key(args.prefix, session_id, filename)
        size = _image_size(args, config, client, session_id, filename, ikey)
        if size is None:
            print(f"  !! no image for {ikey}; skipping (run `down` first?)")
            skipped += 1
            continue
        tkey = put_task(
            client,
            config,
            session_id,
            record,
            size,
            captures_prefix=args.prefix,
            tasks_prefix=args.tasks_prefix,
            model_version=args.model_version,
            include_boxes=args.with_boxes,
            dry_run=args.dry_run,
        )
        print(f"  {'(dry-run) ' if args.dry_run else ''}{tkey}")
        built += 1
    print(f"{built} tasks built, {skipped} skipped")
    print("\nLabel Studio labelling config (paste into the project's Labeling Interface):\n")
    print(ls.LABELING_CONFIG)
    return 0


def run_annotations(args, config, client) -> int:
    """Pull every Label Studio annotation export from `--annotations-bucket` and
    fold them into one merged `label-studio.json` that the training stack reads.

    Export storage writes one object per annotation; we convert each to the
    merged-export task shape, dedup re-labels (latest wins), and overwrite `--out`.
    Images are not fetched — trainers load them from the local mirror or fall back
    to S3, so run `down` separately if you want them on disk.
    """
    from chessvision.data.captures import build_export_from_annotations

    bucket = args.annotations_bucket
    print(f"reading annotation exports from {config.endpoint_url}/{bucket}")
    paginator = client.get_paginator("list_objects_v2")
    keys = [
        obj["Key"] for page in paginator.paginate(Bucket=bucket) for obj in page.get("Contents", [])
    ]
    annotations = [json.loads(get_bytes(client, bucket, key)) for key in keys]
    tasks = build_export_from_annotations(annotations)
    print(f"{len(keys)} annotation objects -> {len(tasks)} deduped tasks")
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(tasks), encoding="utf-8")
    print(f"{'(dry-run) ' if args.dry_run else ''}wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = StorageConfig.from_env()
    client = get_client(config)

    if args.direction == "tasks":
        return run_tasks(args, config, client)
    if args.direction == "annotations":
        return run_annotations(args, config, client)

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
            ignore=DERIVED_SIDECARS,
        )
    else:
        print(f"downloading {config.endpoint_url}/{where} -> {args.local}")
        result = download_prefix(
            args.prefix,
            args.local,
            config=config,
            client=client,
            dry_run=args.dry_run,
            ignore=DERIVED_SIDECARS,
        )

    for key in result.transferred:
        print(f"  {'(dry-run) ' if args.dry_run else ''}{key}")
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
