"""Square-accuracy eval for the contact-keypoint head vs the box-bottom baseline.

For each image we match predicted boxes to ground-truth pieces (same class, best
IoU), then for every matched piece compare two ways of getting its square:

  - **keypoint**: the predicted contact point -> `square_for_point`
  - **box-bottom** (baseline): `bbox_base_point` of the predicted box -> `square_for_point`

against the piece's true square. Reported overall and on the `occluded_pieces`
subset (where box-bottom is expected to fail). Restricting to *matched* pieces
isolates the contact-point question from detection recall: both methods see the
same detections, so the delta is purely "where on the board did it land".

Inference runs at full image resolution and the homography is built from the
full-res corners, so predictions and H share one coordinate frame.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from chessvision.data.chessred import AnnotatedImage, ChessReD
from chessvision.data.contact import occluded_pieces
from chessvision.geometry import (
    Orientation,
    bbox_base_point,
    compute_homography,
    square_for_point,
)

_KEYS = ("n_gt", "matched", "kp_correct", "bb_correct")


def _blank() -> dict[str, int]:
    return dict.fromkeys(_KEYS, 0)


@torch.no_grad()
def evaluate_squares(
    model: torch.nn.Module,
    chessred: ChessReD,
    image_ids,
    device,
    *,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
    tol: float = 0.06,
    offset: float = 0.05,
) -> dict[str, dict[str, int]]:
    """Accumulate keypoint vs box-bottom square correctness over `image_ids`."""
    model.eval()
    overall, occ = _blank(), _blank()
    for image_id in image_ids:
        corners = chessred.corners(image_id)
        if not corners:
            continue
        homography = compute_homography(corners, Orientation.R0)
        meta = chessred.meta(image_id)
        pieces = [p for p in chessred.pieces(image_id) if p.bbox is not None]
        if not pieces:
            continue
        occ_ids = {
            p.piece_id
            for p in occluded_pieces(AnnotatedImage(meta=meta, corners=corners, pieces=pieces))
        }

        bgr = cv2.imread(str(chessred.resolve_image_path(meta)), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).to(device)
        out = model([t])[0]
        keep = out["scores"] >= score_thresh
        p_boxes, p_labels, p_kps = out["boxes"][keep], out["labels"][keep], out["keypoints"][keep]

        gt_boxes = torch.tensor(
            [[x, y, x + w, y + h] for (x, y, w, h) in (p.bbox for p in pieces)],
            dtype=torch.float32,
            device=device,
        )
        ious = (
            box_iou(gt_boxes, p_boxes)
            if len(p_boxes)
            else torch.zeros((len(pieces), 0), device=device)
        )
        for gi, piece in enumerate(pieces):
            is_occ = piece.piece_id in occ_ids
            overall["n_gt"] += 1
            if is_occ:
                occ["n_gt"] += 1
            if not len(p_boxes):
                continue
            row = ious[gi].clone()
            row[p_labels != piece.category_id + 1] = 0.0  # only same-class predictions
            best = int(row.argmax())
            if float(row[best]) < iou_thresh:
                continue  # piece not detected -> not a contact-point question
            overall["matched"] += 1
            if is_occ:
                occ["matched"] += 1

            kp_xy = p_kps[best, 0, :2].cpu().numpy()
            x1, y1, x2, y2 = (float(v) for v in p_boxes[best])
            bb = bbox_base_point((x1, y1, x2 - x1, y2 - y1), vertical_offset=offset)
            kp_ok = square_for_point(homography, kp_xy, tol) == piece.square
            bb_ok = square_for_point(homography, bb, tol) == piece.square
            overall["kp_correct"] += kp_ok
            overall["bb_correct"] += bb_ok
            if is_occ:
                occ["kp_correct"] += kp_ok
                occ["bb_correct"] += bb_ok
    return {"overall": overall, "occluded": occ}


def rates(c: dict[str, int]) -> dict[str, float | int | None]:
    """Turn raw counts into match-rate + keypoint/box-bottom square accuracy."""
    m, n = c["matched"], c["n_gt"]
    return {
        "n_gt": n,
        "matched": m,
        "match_rate": round(m / n, 4) if n else None,
        "kp_square_acc": round(c["kp_correct"] / m, 4) if m else None,
        "box_bottom_square_acc": round(c["bb_correct"] / m, 4) if m else None,
    }
