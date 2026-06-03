"""One-off: backfill the `set` (piece-set) tag on corner-store labels from the board tag.

Each corner/position label records its `board` but predates the `set` field. Every board
in this dataset was shot with exactly one piece set, so the set is a pure function of the
board. This injects `"set"` into each row of `labels.jsonl` accordingly.

Idempotent (keyed on board) and atomic (temp file + replace). Unknown boards abort rather
than mistag. Back up labels.jsonl first; usage:

    uv run python scripts/backfill_corner_sets.py [data/labels.jsonl]
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

# board (boards.json key) -> set (sets.json key). The staunton-wood set is used on both
# its own rimmed board and the snug rimless board; dennis-plastic is the new colleague set.
BOARD_TO_SET = {
    "dennis-bord": "dennis-plastic",
    "rimless-45mm": "staunton-wood",
    "staunton-56mm": "staunton-wood",
    "cheap-30mm": "cheap-wood",
}


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/labels.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    unknown = sorted({r.get("board", "") for r in rows} - BOARD_TO_SET.keys())
    if unknown:
        raise SystemExit(f"unmapped board(s) in {path}: {unknown} — add them to BOARD_TO_SET")

    before, after = Counter(), Counter()
    for r in rows:
        before[r.get("set", "")] += 1
        # Rebuild so `set` sits right after `board`, matching CornerLabel.to_row order.
        new = {}
        for k, v in r.items():
            if k == "set":
                continue
            new[k] = v
            if k == "board":
                new["set"] = BOARD_TO_SET[r["board"]]
        new.setdefault("set", BOARD_TO_SET[r["board"]])  # in case there was no board key position
        r.clear()
        r.update(new)
        after[r["set"]] += 1

    body = "\n".join(json.dumps(r) for r in rows) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)

    print(f"updated {len(rows)} rows in {path}")
    print("  before (set tags):", dict(before))
    print("  after  (set tags):", dict(after))


if __name__ == "__main__":
    main()
