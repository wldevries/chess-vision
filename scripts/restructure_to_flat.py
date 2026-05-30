"""One-off: restructure data/ to the flat layout, re-keying records by source-relative path.

From the OLD layout:
    data/corners/store/{images/<sha1>.jpg, labels.jsonl}
    data/corners/inbox/<inboxrel>
    data/captures/<session>/<file>.jpg   + sets.json/boards.json/sessions.json + label-studio.json

To the NEW flat layout (mirror-source ids: id == image == src == <relpath under source/>):
    data/store/<relpath>.jpg     # normalized image (was images/<sha1>.jpg)
    data/labels.jsonl            # id=image=src=relpath; task_id dropped (derived on read)
    data/source/<relpath>.jpg    # original (inbox/<inboxrel> or <session>/<file>)
    data/sets.json data/boards.json data/sessions.json   # moved from data/captures/

relpath: capture rows (src "captures/<session>/<file>") -> "<session>/<file>";
inbox rows (src bare "<inboxrel>") -> "inbox/<inboxrel>". store images are REQUIRED (the
labelled artifact); source originals are best-effort (copied when present, reported if not).

COPIES (does not delete) so the old dirs remain until validated. Re-runnable. After you've
confirmed training/app work on the new layout, delete data/corners and the migrated bits of
data/captures. Usage:

    uv run python scripts/restructure_to_flat.py --dry-run
    uv run python scripts/restructure_to_flat.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def relpath_for(src: str) -> str:
    """Old `src` -> new source-relative path."""
    if src.startswith("captures/"):
        return src[len("captures/") :]
    return f"inbox/{src}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data = args.data
    old_corners = data / "corners"
    old_store = old_corners / "store"
    old_inbox = old_corners / "inbox"
    old_captures = data / "captures"
    old_labels = old_store / "labels.jsonl"

    new_store = data / "store"
    new_source = data / "source"
    new_labels = data / "labels.jsonl"

    if not old_labels.exists():
        raise SystemExit(f"no old labels at {old_labels}")
    rows = [
        json.loads(line)
        for line in old_labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    n_img, n_src_ok, n_src_missing = 0, 0, 0
    out_rows = []
    for row in rows:
        relpath = relpath_for(row["src"])
        # store image (normalized) -- required
        old_img = old_store / row["image"]  # images/<sha1>.jpg
        new_img = new_store / relpath
        if not old_img.exists():
            raise SystemExit(f"missing store image {old_img} for {row['id']}")
        if not args.dry_run:
            new_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_img, new_img)
        n_img += 1
        # source original -- best effort
        old_orig = (old_captures / relpath) if not relpath.startswith("inbox/") else (
            old_inbox / relpath[len("inbox/") :]
        )
        if old_orig.exists():
            if not args.dry_run:
                dst = new_source / relpath
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_orig, dst)
            n_src_ok += 1
        else:
            n_src_missing += 1
        # rewrite the record: id == image == src == relpath; drop task_id (derived)
        new = dict(row)
        new["id"] = new["image"] = new["src"] = relpath
        new.pop("task_id", None)
        out_rows.append(new)

    out_rows.sort(key=lambda r: r["id"])
    body = "\n".join(json.dumps(r) for r in out_rows) + "\n"

    # metadata json moved to data/ root (still drives app dropdowns / box sizing)
    meta_moved = []
    for name in ("sets.json", "boards.json", "sessions.json"):
        srcp = old_captures / name
        if srcp.exists():
            meta_moved.append(name)
            if not args.dry_run:
                shutil.copy2(srcp, data / name)

    print(
        f"rows={len(rows)}  store images copied={n_img}  "
        f"source originals: ok={n_src_ok} missing={n_src_missing}"
    )
    print(f"metadata copied to data/: {meta_moved}")
    print(f"labels -> {new_labels} ({len(out_rows)} rows)")
    if args.dry_run:
        print("dry-run: nothing written")
        return 0
    new_labels.write_text(body, encoding="utf-8")
    print("done. OLD dirs intact (data/corners + data/captures) -- delete after validating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
