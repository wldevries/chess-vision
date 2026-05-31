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

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from chessvision.data.capture_detection import synthesize_piece_targets
from chessvision.data.captures import CaptureSample
from chessvision.data.detection import resize_targets
from chessvision.data.storage import StorageConfig
from chessvision.geometry import (
    Orientation,
    board_crop_bbox,
    compute_homography,
    square_for_point,
)


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
def _detect_squares(
    model: torch.nn.Module,
    sample: CaptureSample,
    s3: StorageConfig | None,
    device: torch.device,
    *,
    max_size: int,
    score_thresh: float,
    board_crop: bool = False,
    crop_side: float = 0.12,
    crop_top: float = 0.30,
    crop_bottom: float = 0.08,
) -> dict[str, tuple[float, int]]:
    """square -> (score, label) of the best detection on it, via the contact keypoint
    mapped through the GT-corner homography. Shared by `evaluate_captures` and
    `confusion_captures` so the two never drift apart.

    With `board_crop` the image is sliced to the GT board bbox first (same geometry as the
    board-crop training path) and the homography is built in the crop+resized frame, so a
    crop-trained model is evaluated on the matching framing (train == eval)."""
    rgb = sample.load_image(s3)
    h, w = rgb.shape[:2]
    ox, oy = 0.0, 0.0
    if board_crop:
        x0, y0, x1, y1 = board_crop_bbox(
            sample.corners, w, h, side=crop_side, top=crop_top, bottom=crop_bottom
        )
        rgb = rgb[y0:y1, x0:x1]
        ox, oy = float(x0), float(y0)
    hs, ws = rgb.shape[:2]
    scale = min(1.0, max_size / max(hs, ws))
    rgb, _, _ = resize_targets(rgb, np.zeros((0, 4), np.float32), None, max_size)
    corners = {k: ((x - ox) * scale, (y - oy) * scale) for k, (x, y) in sample.corners.items()}
    homography = compute_homography(corners, Orientation.R0)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255
    out = model([image])[0]
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
    return best


@torch.no_grad()
def evaluate_captures(
    model: torch.nn.Module,
    samples: Sequence[CaptureSample],
    s3: StorageConfig | None,
    device: torch.device,
    *,
    max_size: int = 1333,
    score_thresh: float = 0.5,
    board_crop: bool = False,
    crop_side: float = 0.12,
    crop_top: float = 0.30,
    crop_bottom: float = 0.08,
) -> dict[str, float | int]:
    """Run `model` over capture `samples`; return summed counts + derived rates. Pass
    `board_crop=True` (matching the training crop) to eval a board-crop-trained model."""
    model.eval()
    crop = dict(
        board_crop=board_crop, crop_side=crop_side, crop_top=crop_top, crop_bottom=crop_bottom
    )
    counts = defaultdict(int)
    for sample in samples:
        gt = _gt_board(sample)
        counts["gt_pieces"] += len(gt)
        counts["frames"] += 1
        if not gt:
            continue

        best = _detect_squares(
            model, sample, s3, device, max_size=max_size, score_thresh=score_thresh, **crop
        )

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


@torch.no_grad()
def confusion_captures(
    model: torch.nn.Module,
    samples: Sequence[CaptureSample],
    s3: StorageConfig | None,
    device: torch.device,
    *,
    max_size: int = 1333,
    score_thresh: float = 0.5,
    board_crop: bool = False,
    crop_side: float = 0.12,
    crop_top: float = 0.30,
    crop_bottom: float = 0.08,
) -> tuple[Counter, Counter]:
    """Square-level confusion over the held-out GT pieces, to show *where* class_acc
    leaks. Returns ``(confusion, false_pos)``:

    - ``confusion[(gt_label, pred_label)]`` counts each GT piece by what was predicted on
      its square; ``pred_label`` is ``None`` when the square got no detection (a miss).
      So the diagonal ``(c, c)`` is correct, ``(c, None)`` is missed, and ``(c, other)``
      is misclassified-as-other.
    - ``false_pos[pred_label]`` counts detections on squares that are empty in GT.

    Uses the GT-corner homography (the ceiling path). On captures end-to-end == ceiling
    (2026-05-29), so this faithfully attributes the class error without the corner model.
    """
    model.eval()
    crop = dict(
        board_crop=board_crop, crop_side=crop_side, crop_top=crop_top, crop_bottom=crop_bottom
    )
    confusion: Counter = Counter()
    false_pos: Counter = Counter()
    for sample in samples:
        gt = _gt_board(sample)
        if not gt:
            continue
        best = _detect_squares(
            model, sample, s3, device, max_size=max_size, score_thresh=score_thresh, **crop
        )
        for sq, gt_label in gt.items():
            pred = best[sq][1] if sq in best else None
            confusion[(gt_label, pred)] += 1
        for sq, (_, label) in best.items():
            if sq not in gt:
                false_pos[label] += 1
    return confusion, false_pos
