"""End-to-end capture eval: corner MODEL -> homography -> keypoint head -> squares -> board.

`capture_eval.evaluate_captures` builds the square-assignment homography from the
hand-labelled GT corners -- a *perfect-corners ceiling* for the piece head. This script
chains the actual corner model in instead, so the metrics reflect the real image->FEN
pipeline. It prints BOTH side by side (same frames, same metrics): the GT-corner ceiling
and the end-to-end number. The gap between them = the cost of imperfect board localization.

    uv run python scripts/eval_end_to_end_captures.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from chessvision.capture_eval import _gt_board, evaluate_captures
from chessvision.corner_regressor import corners_from_lattice, load_corner_regressor
from chessvision.data.capture_detection import CaptureKeypointConfig, split_by_sessions
from chessvision.data.captures import CaptureDataset
from chessvision.data.detection import resize_targets
from chessvision.geometry import board_crop_bbox, compute_homography, square_for_point
from chessvision.keypoint_detector import load_keypoint_detector
from scripts.finetune_keypoint_captures import DEFAULT_VAL_SESSIONS


@torch.no_grad()
def evaluate_end_to_end(
    kp_model,
    corner_model,
    samples,
    s3,
    device,
    *,
    max_size: int = 1333,
    score_thresh: float = 0.5,
    use_conf: bool = True,
    board_crop: bool = False,
    crop_side: float = 0.12,
    crop_top: float = 0.30,
    crop_bottom: float = 0.08,
) -> dict[str, float | int]:
    """Same logic/metrics as `evaluate_captures`, but the detection-mapping homography
    comes from the corner MODEL's prediction (not the GT corners). Truth (`_gt_board`)
    still uses GT corners. With `board_crop`, the image is cropped to the predicted board
    bbox (+asymmetric headroom for tall back-leaning pieces) before the resize, so pieces
    get more pixels and off-board clutter is removed -- an Approach-A crop, not a warp."""
    kp_model.eval()
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        gt = _gt_board(sample)  # truth via GT corners
        counts["gt_pieces"] += len(gt)
        counts["frames"] += 1
        if not gt:
            continue

        rgb_full = sample.load_image(s3)
        h, w = rgb_full.shape[:2]

        # Corner model on the full-res image -> native-px corners.
        pred = corners_from_lattice(corner_model, rgb_full, device=device, use_conf=use_conf)
        pred_pts = [(float(x), float(y)) for x, y in pred.values()]

        # Optional board crop: bbox of predicted corners + margins. Extra top headroom
        # (min-y side) because pieces lean away from the camera into the back ranks; a
        # tight bbox clips back-rank king/queen tops.
        ox, oy = 0.0, 0.0
        rgb_src = rgb_full
        if board_crop:
            # Same geometry as training, but from the corner MODEL's points (deployment has no GT).
            x0, y0, x1, y1 = board_crop_bbox(
                dict(enumerate(pred_pts)), w, h, side=crop_side, top=crop_top, bottom=crop_bottom
            )
            rgb_src = rgb_full[y0:y1, x0:x1]
            ox, oy = float(x0), float(y0)

        hs, ws = rgb_src.shape[:2]
        scale = min(1.0, max_size / max(hs, ws))

        # Assign each predicted point to the nearest GT corner KEY so the predicted
        # homography lands in the SAME orientation frame as the GT board (orientation is a
        # manual toggle by design, not a corner-model task). Express in crop+resized coords.
        mapped: dict[str, tuple[float, float]] = {}
        used: set[int] = set()
        for key, (gx, gy) in sample.corners.items():
            j = min(
                (i for i in range(len(pred_pts)) if i not in used),
                key=lambda i: (pred_pts[i][0] - gx) ** 2 + (pred_pts[i][1] - gy) ** 2,
            )
            used.add(j)
            mapped[key] = ((pred_pts[j][0] - ox) * scale, (pred_pts[j][1] - oy) * scale)
        try:
            homography = compute_homography(mapped)
        except Exception:
            continue  # degenerate predicted quad -> all gt squares miss, frame not ok

        rgb, _, _ = resize_targets(rgb_src, np.zeros((0, 4), np.float32), None, max_size)
        image = (
            torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255
        )
        out = kp_model([image])[0]

        best: dict[str, tuple[float, int]] = {}
        for score, label, kps in zip(
            out["scores"].tolist(),
            out["labels"].tolist(),
            out["keypoints"].cpu().numpy(),
            strict=True,
        ):
            if score < score_thresh:
                continue
            sq = square_for_point(homography, (float(kps[0, 0]), float(kps[0, 1])))
            if sq is None:
                continue
            if sq not in best or score > best[sq][0]:
                best[sq] = (score, label)

        frame_ok = True
        for sq, gt_label in gt.items():
            if sq in best:
                counts["localized"] += 1
                if best[sq][1] == gt_label:
                    counts["class_correct"] += 1
                else:
                    frame_ok = False
            else:
                frame_ok = False
        if any(sq not in gt for sq in best):
            frame_ok = False
        counts["board_exact"] += int(frame_ok)

    gt_pieces = max(counts["gt_pieces"], 1)
    frames = max(counts["frames"], 1)
    return {
        **counts,
        "localization": counts["localized"] / gt_pieces,
        "class_acc": counts["class_correct"] / gt_pieces,
        "board_exact_rate": counts["board_exact"] / frames,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint_captures/best.pt"))
    add("--corner-ckpt", type=Path, default=Path("runs/corners/best.pt"))
    add("--val-sessions", default=DEFAULT_VAL_SESSIONS)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--max-size", type=int, default=1333)
    add("--score-thresh", type=float, default=0.5)
    add("--no-conf", action="store_true", help="lattice H-fit ignores per-point confidence")
    add("--board-crop", action="store_true", help="also eval with a board-bbox crop (+headroom)")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    dataset = CaptureDataset.load(args.captures)
    cfg = CaptureKeypointConfig(max_size=args.max_size)
    val_sessions = [s for s in args.val_sessions.split(",") if s]
    _, val_ds = split_by_sessions(dataset, val_sessions, cfg)
    print(f"val {len(val_ds)} frames (held-out: {val_sessions})")

    kp = load_keypoint_detector(args.keypoint_ckpt, device)
    corner = load_corner_regressor(args.corner_ckpt, device=device)

    keys = ("localization", "class_acc", "board_exact", "board_exact_rate")
    ceiling = evaluate_captures(
        kp,
        val_ds.samples,
        dataset.s3,
        device,
        max_size=args.max_size,
        score_thresh=args.score_thresh,
    )
    e2e = evaluate_end_to_end(
        kp,
        corner,
        val_ds.samples,
        dataset.s3,
        device,
        max_size=args.max_size,
        score_thresh=args.score_thresh,
        use_conf=not args.no_conf,
    )
    print("GT-corner ceiling  :", json.dumps({k: round(ceiling[k], 4) for k in keys}))
    print("end-to-end (model) :", json.dumps({k: round(e2e[k], 4) for k in keys}))
    if args.board_crop:
        cropped = evaluate_end_to_end(
            kp,
            corner,
            val_ds.samples,
            dataset.s3,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
            use_conf=not args.no_conf,
            board_crop=True,
        )
        print("end-to-end + crop  :", json.dumps({k: round(cropped[k], 4) for k in keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
