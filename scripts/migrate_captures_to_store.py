"""One-off: migrate the Label-Studio capture set into the unified corner store.

Post-Label-Studio, the capture set (`data/captures/label-studio.json` + images in MinIO/
per-session dirs) and the corner store (`data/corners/store`) do the same thing, so this
folds the capture frames into the corner store as `CornerLabel` rows + `store/images/<id>.jpg`.

**Frame safety:** capture labels live in the cv2-decoded raw-pixel frame (cv2 ignores EXIF
orientation). We re-encode via cv2 decode -> encode (strips all metadata incl. any GPS,
preserves the exact pixel frame) and **assert the decoded dims equal the label width/height**;
a mismatch (EXIF rotation) is skipped and reported rather than silently misaligning labels.

Each capture sample becomes one record: corners, pieces (verbose label + base point),
board/set/device/surface from session metadata, the real capture `session` (NOT a synthesized
one), and `captured_at` parsed from the filename timestamp when present. Idempotent (id derived
from the s3 uri) and atomic. The store images are written EXIF-free, so the merged store stays
publish-safe. Existing corner rows are preserved; only capture rows are added/overwritten.

    uv run python scripts/migrate_captures_to_store.py --dry-run
    uv run python scripts/migrate_captures_to_store.py            # apply
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from chessvision.data.captures import CaptureDataset, _split_s3_uri
from chessvision.data.corner_capture import (
    CornerLabel,
    CornerStore,
    _atomic_write_text,
    _stable_id,
    _task_id,
    encode_jpeg,
)
from chessvision.data.session_meta import SessionMetadata

_TS = re.compile(r"(\d{8})-(\d{6})")


def _captured_at_from_name(name: str) -> str:
    """Parse a `YYYYMMDD-HHMMSS` stamp out of a capture filename -> ISO seconds, or ""."""
    m = _TS.search(name)
    if not m:
        return ""
    try:
        return datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%Y%m%d-%H%M%S").isoformat(
            timespec="seconds"
        )
    except ValueError:
        return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    p.add_argument("--corners-root", default="data/corners")
    p.add_argument("--limit", type=int, default=None, help="cap samples (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = p.parse_args()

    ds = CaptureDataset.load(args.captures)
    meta = SessionMetadata.load(args.captures.parent)
    store = CornerStore(args.corners_root)
    store.images_dir.mkdir(parents=True, exist_ok=True)

    samples = ds.samples[: args.limit] if args.limit else ds.samples
    existing = store.load_labels()  # {id: row}; corner rows preserved
    n_existing = len(existing)

    added, skipped_nocorners, skipped_frame, skipped_decode = 0, 0, 0, 0
    boards: Counter = Counter()
    for s in samples:
        if not s.has_all_corners:
            skipped_nocorners += 1
            continue
        try:
            raw = s.read_bytes(ds.s3)
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:  # noqa: BLE001 - report and skip any unreadable image
            print(f"  decode fail {s.s3_uri}: {e}")
            skipped_decode += 1
            continue
        if bgr is None:
            skipped_decode += 1
            continue
        h, w = bgr.shape[:2]
        if (w, h) != (s.width, s.height):
            # label frame != decoded frame (EXIF rotation) -> would misalign; skip loudly.
            print(f"  FRAME MISMATCH {s.s3_uri}: decoded {w}x{h} != label {s.width}x{s.height}")
            skipped_frame += 1
            continue

        _, key = _split_s3_uri(s.s3_uri)
        label_id = _stable_id(s.s3_uri)
        info = meta.info(s.session) if meta else {} or {}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_rel = f"images/{label_id}.jpg"
        if not args.dry_run:
            (store.store / image_rel).write_bytes(encode_jpeg(rgb))
        rec = CornerLabel(
            id=label_id,
            task_id=_task_id(label_id),
            src=key,  # provenance: the original s3 key
            image=image_rel,
            width=s.width,
            height=s.height,
            corners={k: (float(x), float(y)) for k, (x, y) in s.corners.items()},
            board=(info or {}).get("board", "") or "",
            piece_set=(info or {}).get("set", "") or "",
            session=s.session,
            device=(info or {}).get("device", "") or "",
            surface=(info or {}).get("surface", "") or "",
            captured_at=_captured_at_from_name(Path(key).name),
            pieces=tuple((kp.label, float(kp.point[0]), float(kp.point[1])) for kp in s.pieces),
        )
        existing[label_id] = rec.to_row()
        added += 1
        boards[rec.board or "(untagged)"] += 1

    print(
        f"capture samples={len(samples)}  added={added}  "
        f"skipped: no-corners={skipped_nocorners} frame-mismatch={skipped_frame} "
        f"decode-fail={skipped_decode}"
    )
    print(f"  by board: {dict(boards)}")
    print(f"  store rows: {n_existing} existing -> {len(existing)} total")

    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    ordered = sorted(existing.values(), key=lambda r: r["id"])
    body = "\n".join(json.dumps(r) for r in ordered) + "\n"
    _atomic_write_text(store.labels_path, body)
    print(f"wrote {len(ordered)} rows -> {store.labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
