"""Evaluate a keypoint detector on capture photos: square + class accuracy.

The capture truth is per-piece (contact point + class), with squares derived through
the homography. For each held-out frame we run the model, map every detected
**contact keypoint** to its square (never a box bottom -- see the anti-pattern), and
compare the per-square occupancy and class against the labelled board.

Metrics (summed over frames, reported as rates):
    localization : GT squares that got >=1 detection           (recall of occupancy)
    class_acc    : GT squares whose top detection has the right class
    board_exact  : frames where every GT square is class-correct AND no false piece

Predicted keypoints come back in the model's (resized) input frame, so we build the
homography from corners scaled by the same factor and keep everything in that frame.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from chessvision.data.capture_detection import synthesize_piece_targets
from chessvision.data.captures import CaptureSample, S3Config
from chessvision.data.detection import resize_targets
from chessvision.geometry import Orientation, compute_homography, square_for_point


def _scaled_homography(sample: CaptureSample, scale: float) -> np.ndarray:
    corners = {k: (x * scale, y * scale) for k, (x, y) in sample.corners.items()}
    return compute_homography(corners, Orientation.R0)


def _gt_board(sample: CaptureSample) -> dict[str, int]:
    """square -> label for on-board pieces (drops off-board, same as training)."""
    homography = compute_homography(sample.corners, Orientation.R0)
    _, labels, kpts = synthesize_piece_targets(sample)
    board: dict[str, int] = {}
    for label, kp in zip(labels.tolist(), kpts, strict=True):
        sq = square_for_point(homography, (float(kp[0, 0]), float(kp[0, 1])))
        if sq is not None:
            board[sq] = label  # one piece per square
    return board


@torch.no_grad()
def evaluate_captures(
    model: torch.nn.Module,
    samples: Sequence[CaptureSample],
    s3: S3Config | None,
    device: torch.device,
    *,
    max_size: int = 1333,
    score_thresh: float = 0.5,
) -> dict[str, float | int]:
    """Run `model` over capture `samples`; return summed counts + derived rates."""
    model.eval()
    counts = defaultdict(int)
    for sample in samples:
        gt = _gt_board(sample)
        counts["gt_pieces"] += len(gt)
        counts["frames"] += 1
        if not gt:
            continue

        rgb = sample.load_image(s3)
        h, w = rgb.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        rgb, _, _ = resize_targets(rgb, np.zeros((0, 4), np.float32), None, max_size)
        homography = _scaled_homography(sample, scale)
        image = (
            torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255
        )
        out = model([image])[0]

        # Best detection per square (by score), mapped via its contact keypoint.
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
        if any(sq not in gt for sq in best):  # detected a piece on an empty square
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
