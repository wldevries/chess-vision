"""Head-to-head: YOLO-pose vs the Keypoint R-CNN, per board, same capture metrics.

The FEN-relevant comparison. Both models are scored on the user's boards with the SAME
metric path (`chessvision.capture_eval.evaluate_captures`): each detected **contact
keypoint** is mapped through the GT-corner homography to a square, then per-square
occupancy/class is compared to the labelled board. Reports the standard trio per board:
localization (occupancy recall), class_acc (the deployment metric / bottleneck), board_exact.

The YOLO-pose model is presented through `YoloKeypointDetector`, which exposes the exact
`model([image]) -> [{scores,labels,keypoints}]` interface the R-CNN does, so neither the
metric nor the square-assignment code is forked.

Split mirrors the joint trainer / `eval_per_board_captures.py`:
  - val  rows = pose-held-out poses of TRAINED boards  -> generalization to new positions
  - test rows = whole boards held out (`--test-boards`, e.g. dennis) -> unseen-board number

Each model is evaluated in its NATIVE framing: the Keypoint R-CNN with the board-crop config
stamped in its checkpoint; YOLO-pose full-frame (--yolo-max-size keeps the image near-original
so YOLO does its own single letterbox to --yolo-imgsz). Thresholds default per model and are
tunable (--kp-score-thresh / --yolo-score-thresh) since board_exact is false-positive sensitive.

    uv sync --group yolo
    uv run --group yolo python scripts/eval_yolo_pose_vs_keypoint.py \
        --data-root data/othersets/ChessReD --store data --test-boards dennis-bord \
        --yolo-pose-ckpt runs/yolo_pose/weights/best.pt \
        --keypoint-ckpt runs/keypoint_joint/best.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from chessvision.capture_eval import evaluate_captures
from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.positions import store_label_to_capture
from chessvision.keypoint_detector import load_keypoint_detector, read_keypoint_preprocess
from chessvision.yolo_keypoint import load_yolo_keypoint_detector

KEYS = ("frames", "gt_pieces", "localization", "class_acc", "board_exact", "board_exact_rate")


def _round(d: dict) -> dict:
    return {k: (round(d[k], 4) if isinstance(d[k], float) else d[k]) for k in KEYS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", type=Path, default=None, help="unused; kept for symmetry with trainer")
    add("--store", type=Path, default=Path("data"), help="unified corner store root")
    add("--yolo-pose-ckpt", type=Path, default=Path("runs/yolo_pose/weights/best.pt"))
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint_joint/best.pt"))
    add("--test-boards", default="dennis-bord", help="boards held out as TEST (unseen-board)")
    add("--val-pose-frac", type=float, default=0.25)
    add("--dedup-thr", type=float, default=0.02)
    add("--only-board", default=None, help="restrict to one board id")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Keypoint R-CNN framing (board-crop read from its checkpoint unless overridden).
    add("--max-size", type=int, default=1333, help="Keypoint R-CNN resize long-side")
    add("--kp-score-thresh", type=float, default=0.5)
    add(
        "--board-crop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="R-CNN crop; default reads the checkpoint's recorded crop config",
    )
    # YOLO-pose framing. Board-crop ON by default to match the trained dataset (the champion
    # crops too); --yolo-max-size high => the cropped board isn't pre-resized (YOLO letterboxes).
    add("--yolo-imgsz", type=int, default=1280)
    add(
        "--yolo-max-size",
        type=int,
        default=8192,
        help="high => no pre-resize; YOLO letterboxes the (cropped) board once",
    )
    add(
        "--yolo-board-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="crop to the board rectangle for YOLO (must match how it trained); default on",
    )
    add("--yolo-score-thresh", type=float, default=0.25)
    add("--skip-yolo", action="store_true")
    add("--skip-kp", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    store = CornerStore(args.store)
    test_boards = [b for b in args.test_boards.split(",") if b]

    tr, va, te = split_store_for_keypoints(
        store, test_boards=test_boards, val_pose_frac=args.val_pose_frac, dedup_thr=args.dedup_thr
    )
    # board -> (split_tag, [CaptureSample]); test boards are excluded from val by construction.
    by_board: dict[str, list] = defaultdict(list)
    split_of: dict[str, str] = {}
    for tag, labels in (("val", va), ("test", te)):
        for lb in labels:
            b = lb.board or "<untagged>"
            if args.only_board and b != args.only_board:
                continue
            by_board[b].append(store_label_to_capture(lb, store))
            split_of[b] = tag

    print(
        "boards: " + ", ".join(f"{b}[{split_of[b]}]({len(g)})" for b, g in sorted(by_board.items()))
    )

    # Keypoint R-CNN: match the framing it trained on (board-crop from the checkpoint).
    models = {}
    if not args.skip_kp:
        pre = read_keypoint_preprocess(args.keypoint_ckpt)
        use_crop = pre.get("board_crop", False) if args.board_crop is None else args.board_crop
        kp_crop = dict(
            board_crop=use_crop,
            crop_side=pre.get("crop_side", 0.12),
            crop_top=pre.get("crop_top", 0.30),
            crop_bottom=pre.get("crop_bottom", 0.08),
        )
        models["keypoint_rcnn"] = (
            load_keypoint_detector(args.keypoint_ckpt, device),
            dict(max_size=args.max_size, score_thresh=args.kp_score_thresh, **kp_crop),
        )
        print(f"keypoint_rcnn: board_crop={use_crop}, score_thresh={args.kp_score_thresh}")
    if not args.skip_yolo:
        from chessvision.data.yolo_pose_export import CROP

        yolo_crop = dict(
            board_crop=args.yolo_board_crop,
            crop_side=CROP["side"],
            crop_top=CROP["top"],
            crop_bottom=CROP["bottom"],
        )
        models["yolo_pose"] = (
            load_yolo_keypoint_detector(
                args.yolo_pose_ckpt, device=str(device), imgsz=args.yolo_imgsz
            ),
            dict(max_size=args.yolo_max_size, score_thresh=args.yolo_score_thresh, **yolo_crop),
        )
        print(
            f"yolo_pose: board_crop={args.yolo_board_crop}, imgsz={args.yolo_imgsz}, "
            f"score_thresh={args.yolo_score_thresh}"
        )

    print("\n" + "=" * 72)
    summary: dict[str, dict] = {}
    for b, grp in sorted(by_board.items()):
        print(f"\n### {b} [{split_of[b]}]  ({len(grp)} frames)")
        summary[b] = {"split": split_of[b], "frames": len(grp)}
        for name, (model, kw) in models.items():
            res = evaluate_captures(model, grp, None, device, **kw)
            print(f"  {name:14s}:", json.dumps(_round(res)))
            summary[b][name] = {
                k: res[k] for k in ("localization", "class_acc", "board_exact_rate")
            }

    # Compact class_acc table (the headline number).
    if len(models) == 2:
        print("\n" + "=" * 72)
        print(f"{'board':<18}{'split':<6}{'kp class_acc':>14}{'yolo class_acc':>16}")
        for b in sorted(summary):
            row = summary[b]
            ka = row.get("keypoint_rcnn", {}).get("class_acc", float("nan"))
            ya = row.get("yolo_pose", {}).get("class_acc", float("nan"))
            print(f"{b:<18}{row['split']:<6}{ka:>14.4f}{ya:>16.4f}")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
