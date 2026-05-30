"""One-off: backfill EXIF metadata + synthesized sessions onto corner-store labels.

Rows labelled before the EXIF step have empty `captured_at`/`device`, and nothing has a
`session` yet. This re-reads each row's **inbox original** (still on disk), extracts the
publish-safe whitelist (`captured_at`, lens-aware `device`; GPS/serials never touched), then
assigns sessions by `(board, capture-time gap)` via `assign_sessions`. Rewrites labels.jsonl
through `CornerLabel.from_row -> to_row` so every row gets the canonical field order.

Idempotent (recomputes from EXIF + reassigns deterministically) and atomic (temp + replace).
Rows whose inbox original is missing keep whatever they had and are reported. Usage:

    uv run python scripts/backfill_corner_exif_sessions.py [--corners-root data/corners]
                                                           [--gap-minutes 20] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace

from chessvision.data.corner_capture import (
    CornerLabel,
    CornerStore,
    assign_sessions,
    extract_exif_meta,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corners-root", default="data")
    p.add_argument("--gap-minutes", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true", help="report only; do not write")
    args = p.parse_args()

    store = CornerStore(args.corners_root)
    rows = store.load_labels()  # {id: row}
    if not rows:
        print(f"no labels under {store.labels_path}")
        return 0

    no_exif, missing_inbox = 0, 0
    labels: list[CornerLabel] = []
    for row in rows.values():
        src = row.get("src", "")
        try:
            raw = store.inbox_path(src).read_bytes()
        except (FileNotFoundError, KeyError):
            missing_inbox += 1
            labels.append(CornerLabel.from_row(row))
            continue
        meta = extract_exif_meta(raw)
        if not meta.get("captured_at"):
            no_exif += 1
        if meta.get("captured_at"):
            row["captured_at"] = meta["captured_at"]
        if not row.get("device") and meta.get("device"):
            row["device"] = meta["device"]
        labels.append(CornerLabel.from_row(row))

    sessions = assign_sessions(labels, gap_minutes=args.gap_minutes)
    labels = [replace(lab, session=sessions.get(lab.id, lab.session)) for lab in labels]

    # Report
    by_board_sessions: dict[str, set[str]] = {}
    sess_sizes: Counter = Counter()
    for lab in labels:
        if lab.session:
            by_board_sessions.setdefault(lab.board or "(untagged)", set()).add(lab.session)
            sess_sizes[lab.session] += 1
    print(f"rows={len(labels)}  missing inbox original={missing_inbox}  no EXIF time={no_exif}")
    print(f"sessions={len(sess_sizes)} (gap={args.gap_minutes}min)")
    for board in sorted(by_board_sessions):
        sizes = sorted((sess_sizes[s] for s in by_board_sessions[board]), reverse=True)
        print(f"  {board:16s}: {len(by_board_sessions[board])} sessions, frames/session={sizes}")

    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    # Rewrite labels.jsonl, canonical field order, sorted by id (matches CornerStore._write).
    ordered = sorted((lab.to_row() for lab in labels), key=lambda r: r["id"])
    body = "\n".join(json.dumps(r) for r in ordered) + "\n"
    tmp = store.labels_path.with_name(store.labels_path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, store.labels_path)
    print(f"wrote {len(ordered)} rows -> {store.labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
