"""Export a YOLO-**pose** dataset (box + board-contact keypoint) for the piece detector.

The pose counterpart to `yolo_export.py`. Each label row is

    cls  cx cy w h  px py pv          (all box/point coords normalized to [0,1])

where the single keypoint `(px, py)` is the piece's **board-contact point** -- the
doctrine-pure base (square-center projected through the homography), exactly the target
the Keypoint R-CNN's contact head learns (see chessvision/data/contact.py and the
contact-point anti-pattern). `pv=2` = labelled/visible.

Built to match the champion joint trainer (`scripts/train_keypoint_joint.py`,
`runs/keypoint_joint_aug_crop`) so the comparison is like-for-like on the two things
that move the number:

  - **50/50 data mix** -- ChessReD outnumbers the store ~3:1, and the champion rebalances
    with a `WeightedRandomSampler` (`mix=0.5`). YOLO has no sampler weighting, so we
    **oversample the store** (`store_oversample`, default auto) by duplicating its frames
    (image hardlinked, label copied) until the two domains are ~balanced. With YOLO's
    per-sample augmentation each duplicate trains under different aug, approximating the
    sampler. This is the lever against drifting back to the ChessReD appearance domain.
  - **board crop** -- the champion trains AND infers on a rectangle around the board (GT
    corners here; the corner model's predicted board at deploy) for higher-resolution
    piece detection/classification. We slice both domains to `board_crop_bbox` (margins
    side/top/bottom) and translate boxes+contact points into the crop frame. The eval
    must use the SAME crop (see eval_yolo_pose_vs_keypoint.py).

  - train = ChessReD chessred2k **train** + capture-store **train** (oversampled)
  - val   = capture-store **val** (pose-held-out poses of trained boards)

The store **test** boards (e.g. dennis -- the unseen-board number) are NOT exported here;
the comparison eval scores them directly from `CaptureSample`s. Store splits come from
`split_store_for_keypoints` with the same params the joint trainer/eval use.

YOLO class id == ChessReD category_id (0..11); the keypoint detector / capture truth use
labels 1..12 (LABEL_NAMES), so the wrapper at inference shifts back by +1.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from chessvision.data.capture_detection import synthesize_piece_targets
from chessvision.data.captures import CaptureSample
from chessvision.data.chessred import AnnotatedImage, ChessReD
from chessvision.data.contact import contact_points
from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.detection import NUM_PIECE_CLASSES, apply_board_crop
from chessvision.data.positions import store_label_to_capture
from chessvision.data.yolo_export import PIECE_NAMES, _link_or_copy
from chessvision.geometry import board_crop_bbox

# Champion board-crop margins (fraction of the corner bbox), from keypoint_joint_aug_crop.
CROP = dict(side=0.12, top=0.30, bottom=0.08)


def _pose_lines(
    boxes: np.ndarray, labels, kpts: np.ndarray, width: float, height: float
) -> list[str]:
    """Normalized YOLO-pose rows from xyxy boxes + (N,1,3) keypoints. Drops degenerate boxes."""
    lines = []
    for (x1, y1, x2, y2), cls, kpt in zip(boxes, labels, kpts, strict=True):
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        px = min(max(float(kpt[0, 0]) / width, 0.0), 1.0)
        py = min(max(float(kpt[0, 1]) / height, 0.0), 1.0)
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {px:.6f} {py:.6f} 2")
    return lines


def _write_pose(
    img_bgr: np.ndarray, lines: list[str], stem: str, img_dir: Path, lbl_dir: Path
) -> None:
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), img_bgr)
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_chessred_pose(
    chessred: ChessReD, split: str, out_dir: Path, board_crop: bool, limit: int | None
) -> int:
    """ChessReD chessred2k split -> YOLO-pose. With board_crop the image is decoded and sliced to
    the board rectangle (boxes/contact points translated); else hardlinked full-frame. #images."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for image_id in chessred.chessred2k_split(split):
        meta = chessred.meta(image_id)
        annotated = AnnotatedImage(
            meta=meta, corners=chessred.corners(image_id), pieces=chessred.pieces(image_id)
        )
        cps = contact_points(annotated)
        boxes, labels, kpts = [], [], []
        for piece, cp in zip(annotated.pieces, cps, strict=True):
            if piece.bbox is None:
                continue
            x, y, w, h = piece.bbox  # COCO xywh
            boxes.append([x, y, x + w, y + h])
            labels.append(piece.category_id)
            kpts.append([[cp.xy[0], cp.xy[1], 2.0]])
        if not boxes:
            continue
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        kpts = np.asarray(kpts, np.float32).reshape(-1, 1, 3)
        src = chessred.resolve_image_path(meta)

        if board_crop:
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            bh0, bw0 = bgr.shape[:2]
            bbox = board_crop_bbox(annotated.corners, bw0, bh0, **CROP)
            bgr, boxes, kpts = apply_board_crop(bgr, boxes, kpts, bbox)
            height, width = float(bgr.shape[0]), float(bgr.shape[1])
            lines = _pose_lines(boxes, labels, kpts, width, height)
            if lines:
                _write_pose(bgr, lines, f"cr_{image_id}", img_dir, lbl_dir)
                n += 1
        else:
            width, height = float(meta.width), float(meta.height)
            lines = _pose_lines(boxes, labels, kpts, width, height)
            if lines:
                _link_or_copy(
                    src, img_dir / f"cr_{image_id}{Path(meta.file_name).suffix or '.jpg'}"
                )
                (lbl_dir / f"cr_{image_id}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                n += 1
        if limit and n >= limit:
            break
    return n


def export_store_pose(
    samples: list[CaptureSample], split: str, out_dir: Path, board_crop: bool, oversample: int = 1
) -> int:
    """Capture-store samples -> YOLO-pose. Images are DECODED (pixel frame provably matches labels
    regardless of EXIF) and, with board_crop, sliced to the board rectangle. Each frame is written
    `oversample` times (duplicate image hardlinked, label copied) to rebalance vs ChessReD. Returns
    the number of *distinct* frames written."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for sample in samples:
        if not sample.has_all_corners:
            continue
        boxes, labels, kpts = synthesize_piece_targets(sample)
        if not len(boxes):
            continue
        rgb = sample.load_image(None)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if board_crop:
            h0, w0 = bgr.shape[:2]
            bbox = board_crop_bbox(sample.corners, w0, h0, **CROP)
            bgr, boxes, kpts = apply_board_crop(bgr, boxes, kpts, bbox)
        height, width = float(bgr.shape[0]), float(bgr.shape[1])
        lines = _pose_lines(boxes, [int(v) - 1 for v in labels.tolist()], kpts, width, height)
        if not lines:
            continue
        base = f"st_{sample.session}_{sample.task_id}"
        _write_pose(bgr, lines, base, img_dir, lbl_dir)
        for k in range(1, max(1, oversample)):  # duplicate to oversample the store
            dup_img = img_dir / f"{base}_d{k}.jpg"
            if not dup_img.exists():
                try:
                    os.link(img_dir / f"{base}.jpg", dup_img)
                except OSError:
                    cv2.imwrite(str(dup_img), bgr)
            (lbl_dir / f"{base}_d{k}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    return n


def build_yolo_pose_dataset(
    data_root: str | Path,
    store_root: str | Path,
    out_dir: str | Path,
    images_root: str | Path | None = None,
    test_boards: tuple[str, ...] = ("dennis-bord",),
    val_pose_frac: float = 0.25,
    dedup_thr: float = 0.02,
    board_crop: bool = True,
    store_oversample: int | None = None,
    limit: int | None = None,
) -> tuple[Path, dict[str, int]]:
    """Build train (ChessReD-train + store-train, store oversampled ~50/50) and val (store-val).

    `store_oversample=None` auto-picks round(chessred_train / store_train) so the two domains are
    ~balanced (mirrors the champion's mix=0.5). Returns (yaml_path, counts)."""
    out_dir = Path(out_dir)
    chessred = ChessReD.load(data_root, images_root)
    store = CornerStore(store_root)
    tr, va, _te = split_store_for_keypoints(
        store, test_boards=list(test_boards), val_pose_frac=val_pose_frac, dedup_thr=dedup_thr
    )
    to_cap = lambda labels: [store_label_to_capture(lb, store) for lb in labels]  # noqa: E731

    n_cr = export_chessred_pose(chessred, "train", out_dir, board_crop, limit)
    store_train = to_cap(tr)
    if store_oversample is None:
        store_oversample = max(1, round(n_cr / max(len(store_train), 1)))
    counts = {
        "train_chessred": n_cr,
        "train_store_distinct": export_store_pose(
            store_train, "train", out_dir, board_crop, oversample=store_oversample
        ),
        "store_oversample": store_oversample,
        "val_store": export_store_pose(to_cap(va), "val", out_dir, board_crop, oversample=1),
    }

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(PIECE_NAMES))
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [1, 3]\n"  # one contact keypoint, (x, y, visibility)
        "flip_idx: [0]\n"  # single keypoint maps to itself under horizontal flip
        f"nc: {NUM_PIECE_CLASSES}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return yaml_path, counts


def parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", required=True, type=Path, help="ChessReD dir (annotations.json)")
    p.add_argument("--images-root", type=Path, default=None)
    p.add_argument("--store", type=Path, default=Path("data"), help="unified corner store root")
    p.add_argument("--out-dir", type=Path, default=Path("data/yolo_pose"))
    p.add_argument(
        "--test-boards", default="dennis-bord", help="boards held out as TEST (excluded)"
    )
    p.add_argument("--val-pose-frac", type=float, default=0.25)
    p.add_argument("--dedup-thr", type=float, default=0.02)
    p.add_argument(
        "--board-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="slice both domains to the board rectangle (matches the champion); default on",
    )
    p.add_argument(
        "--store-oversample",
        type=int,
        default=None,
        help="duplicate store frames N times (default auto ~50/50 vs ChessReD)",
    )
    p.add_argument("--limit", type=int, default=None, help="cap ChessReD train images (smoke)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    yaml_path, counts = build_yolo_pose_dataset(
        args.data_root,
        args.store,
        args.out_dir,
        args.images_root,
        tuple(b for b in args.test_boards.split(",") if b),
        args.val_pose_frac,
        args.dedup_thr,
        args.board_crop,
        args.store_oversample,
        args.limit,
    )
    print(f"wrote {yaml_path}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
