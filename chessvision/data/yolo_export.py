"""Export ChessReD chessred2k as a YOLO-format detection dataset (boxes only).

Mirrors the box pipeline in `chessvision/data/detection.py` -- same official
chessred2k split, same 12 piece classes -- but writes the on-disk layout
Ultralytics expects:

    <out_dir>/
      images/{train,val,test}/<image_id>.jpg   (hardlink to the ChessReD image; copy fallback)
      labels/{train,val,test}/<image_id>.txt    (one `cls cx cy w h` row per box, normalized)
      data.yaml

YOLO class id == ChessReD `category_id` (0..11) and has **no background class**
(unlike the Faster R-CNN head, which reserves id 0 -- see
`detection.LABEL_NAMES`). `PIECE_NAMES[i]` is the name for YOLO class `i`, derived
from `detection.LABEL_NAMES` so the two label spaces stay in lockstep.

Images are hardlinked (no copy, no extra disk) when the ChessReD tree and `out_dir`
share a volume; otherwise we fall back to copying. Files are named by `image_id` so
names are unique across ChessReD's per-game subfolders. Rebuilds are idempotent.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from chessvision.data.chessred import ChessReD
from chessvision.data.detection import LABEL_NAMES, NUM_PIECE_CLASSES

# YOLO class i == ChessReD category_id i (FRCNN label i+1). Keep the two in lockstep.
PIECE_NAMES = [LABEL_NAMES[i + 1] for i in range(NUM_PIECE_CLASSES)]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src->dst (cheap, same-volume); fall back to copy. Idempotent."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_split(chessred: ChessReD, split: str, out_dir: Path, limit: int | None = None) -> int:
    """Write images/<split> + labels/<split> for one chessred2k split. Returns #images."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for image_id in chessred.chessred2k_split(split):
        pieces = [p for p in chessred.pieces(image_id) if p.bbox is not None]
        if not pieces:
            continue  # defensive: chessred2k images all carry boxes
        meta = chessred.meta(image_id)
        width, height = float(meta.width), float(meta.height)
        suffix = Path(meta.file_name).suffix or ".jpg"
        _link_or_copy(chessred.resolve_image_path(meta), img_dir / f"{image_id}{suffix}")

        lines = []
        for p in pieces:
            x, y, w, h = p.bbox  # COCO xywh, original pixels
            cx, cy = (x + w / 2) / width, (y + h / 2) / height
            lines.append(f"{p.category_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
        (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        n += 1
        if limit and n >= limit:
            break
    return n


def build_yolo_dataset(
    data_root: str | Path,
    out_dir: str | Path,
    images_root: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "val"),
    limit: int | None = None,
) -> tuple[Path, dict[str, int]]:
    """Build the full YOLO dataset + data.yaml. Returns (yaml_path, {split: count})."""
    chessred = ChessReD.load(data_root, images_root)
    out_dir = Path(out_dir)
    counts = {s: export_split(chessred, s, out_dir, limit=limit) for s in splits}

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(PIECE_NAMES))
    # `val:` points at the validation split; test (if built) is selectable via split=test.
    yaml_path = out_dir / "data.yaml"
    yaml_lines = [f"path: {out_dir.resolve().as_posix()}"]
    for key, sub in (("train", "train"), ("val", "val"), ("test", "test")):
        if sub in counts:
            yaml_lines.append(f"{key}: images/{sub}")
    yaml_lines += [f"nc: {NUM_PIECE_CLASSES}", "names:", names_block, ""]
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    return yaml_path, counts


def parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)"
    )
    p.add_argument("--images-root", type=Path, default=None, help="image tree root override")
    p.add_argument("--out-dir", type=Path, default=Path("data/yolo_chessred"))
    p.add_argument(
        "--splits", nargs="+", default=["train", "val"], help="chessred2k splits to export"
    )
    p.add_argument("--limit", type=int, default=None, help="cap images per split (smoke tests)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    yaml_path, counts = build_yolo_dataset(
        args.data_root, args.out_dir, args.images_root, tuple(args.splits), args.limit
    )
    print(f"wrote {yaml_path}")
    for split, n in counts.items():
        print(f"  {split}: {n} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
