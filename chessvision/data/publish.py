"""Publish a finished capture session: push its photos + metadata to the bucket
and generate the matching Label Studio point tasks.

This is the "done with this game/puzzle" step the capture app calls on finish —
it stitches together the generic S3 sync (`storage.upload_dir`) and the
points-only task builder (`labelstudio`). Kept separate from both so each stays
single-purpose and this orchestration is unit-testable with a fake client.
"""

from __future__ import annotations

import json
from pathlib import Path

from chessvision.data import labelstudio as ls
from chessvision.data.storage import StorageConfig, put_bytes, upload_dir


def put_task(
    client,
    config: StorageConfig,
    session_id: str,
    record: dict,
    image_size: tuple[int, int],
    *,
    captures_prefix: str = "captures",
    tasks_prefix: str = "tasks",
    model_version: str = ls.MODEL_VERSION,
    include_boxes: bool = False,
    dry_run: bool = False,
) -> str | None:
    """Build one Label Studio task for `record` and PUT it under `<tasks_prefix>/`.

    Returns the task key (None if the record has no filename). The single
    source of truth for the task shape + key, shared by `publish_session` and the
    `sync_captures.py tasks` command. Tasks overwrite by key, so re-running is safe.
    """
    filename = record.get("filename")
    if not filename:
        return None
    image_key = ls.image_key(captures_prefix, session_id, filename)
    task = ls.build_task(
        record,
        image_size,
        ls.image_uri(config.bucket, image_key),
        model_version=model_version,
        include_boxes=include_boxes,
    )
    task_key = ls.task_key(tasks_prefix, session_id, filename)
    if not dry_run:
        body = json.dumps(task).encode("utf-8")
        put_bytes(client, config.bucket, task_key, body, "application/json")
    return task_key


def publish_session(
    out_dir: str | Path,
    session_id: str,
    records: list[dict],
    *,
    config: StorageConfig,
    client,
    captures_prefix: str = "captures",
    tasks_prefix: str = "tasks",
    include_boxes: bool = False,
) -> dict:
    """Upload a session's folder to `<captures_prefix>/<session_id>/` and write a
    Label Studio task per record to `<tasks_prefix>/<session_id>/`.

    `records` are the session's capture rows (same shape as `captures.jsonl`).
    Photo upload is size-based (re-running skips unchanged files); tasks overwrite
    by key, so finishing twice is safe. Returns counts for a status message.
    """
    out_dir = Path(out_dir)
    sync = upload_dir(out_dir, f"{captures_prefix}/{session_id}", config=config, client=client)

    n_tasks = 0
    for record in records:
        filename = record.get("filename")
        if not filename:
            continue
        local_img = out_dir / filename
        if not local_img.exists():
            continue  # photo was deleted after capture; skip its task
        put_task(
            client,
            config,
            session_id,
            record,
            ls.image_size_from_path(local_img),
            captures_prefix=captures_prefix,
            tasks_prefix=tasks_prefix,
            include_boxes=include_boxes,
        )
        n_tasks += 1

    return {"uploaded": len(sync.transferred), "skipped": sync.skipped, "tasks": n_tasks}
