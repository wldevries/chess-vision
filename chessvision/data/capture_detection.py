"""Capture-set keypoint-detection dataset (domain fine-tuning, Approach A).

Turns the hand-labelled capture photos (`CaptureDataset`: four corners + one
contact keypoint per piece, *no boxes*) into the same `(image, target)` shape the
keypoint detector trains on -- so `runs/keypoint/best.pt` can be fine-tuned on the
user's own boards/pieces (see the keypoint-head-purpose note).

Per piece we emit:
    boxes     : (N, 4) float32 xyxy -- **synthesized** with `geometry.project_piece_box`
                (a vertical-cylinder RoI around the contact point; the box is only a
                region hint, never a contact source -- see the box-bottom anti-pattern).
    labels    : (N,) int64 in 1..12, the SAME class ids as `detection.LABEL_NAMES`
                (so the pretrained box head transfers without remapping).
    keypoints : (N, 1, 3) float32 [x, y, visibility=2] -- the hand-tagged contact point,
                which is the exact supervision target.

Off-board pieces (a captured piece resting beside the board: `square_for_point`
returns None) are **dropped** -- they have no square and would poison the RoI.

.. warning::
    Split by **session**, never by random image (a session = one board/set/room/pose;
    random splits leak near-duplicate frames). Use `leave_one_session_out`.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from chessvision.data.captures import CaptureDataset, CaptureSample
from chessvision.data.detection import DetectionConfig, augment_targets, resize_targets
from chessvision.data.storage import StorageConfig
from chessvision.geometry import (
    PIECE_HEIGHT_SCALE,
    Orientation,
    compute_homography,
    project_piece_box,
    square_for_point,
)

# FEN letter -> detector class id (1..12), matching detection._PIECE_NAMES /
# ChessReD category_id + 1: pawn,rook,knight,bishop,queen,king for white then black.
_FEN_ORDER = "PRNBQKprnbqk"


def fen_to_label(fen: str) -> int:
    """FEN piece letter (e.g. 'R', 'q') -> detector label in 1..12."""
    return _FEN_ORDER.index(fen) + 1


def synthesize_piece_targets(
    sample: CaptureSample,
    *,
    radius_squares: float = 0.3,
    margin: float = 0.15,
    orientation: Orientation = Orientation.R0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (boxes xyxy, labels, keypoints) for one capture sample.

    Pure (no image decode) so it is unit-testable without S3/image bytes. Drops
    pieces whose contact point falls off the board. Boxes are the tight cylinder
    bound expanded by `margin` (fraction of box size, each side) so the keypoint
    head's RoI crop has context around the piece -- and the contact point sits
    comfortably inside rather than on the bottom edge. Boxes are clipped to the
    image; the contact keypoint always stays inside its (clipped) box.

    Per-piece height/radius come from the sample's resolved `box_sizes` (physical
    mm / board square, via session metadata) when present; otherwise they fall back
    to `PIECE_HEIGHT_SCALE` and `radius_squares`. This is what lets the same set on a
    smaller-grid board get correctly larger boxes without retuning constants.
    """
    homography = compute_homography(sample.corners, orientation)
    w, h = sample.width, sample.height
    sizes = sample.box_sizes or {}
    boxes, labels, kpts = [], [], []
    for kp in sample.pieces:
        if square_for_point(homography, kp.point) is None:
            continue  # off-board piece (resting beside the board) -> no square
        fen = kp.fen.lower()
        height_squares, piece_radius = sizes.get(
            fen, (PIECE_HEIGHT_SCALE.get(fen, 1.0), radius_squares)
        )
        x1, y1, x2, y2 = project_piece_box(
            homography,
            kp.point,
            (w, h),
            height_squares=height_squares,
            radius_squares=piece_radius,
        )
        # Margin so the RoI crop has context (esp. below the base) and the contact
        # point is not pinned to an edge.
        mx, my = margin * (x2 - x1), margin * (y2 - y1)
        x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
        # Clip to image; the contact point is in-image so it survives the clip.
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue  # degenerate after clipping
        boxes.append([x1, y1, x2, y2])
        labels.append(fen_to_label(kp.fen))
        kpts.append([[float(kp.point[0]), float(kp.point[1]), 2.0]])
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
        np.asarray(kpts, dtype=np.float32).reshape(-1, 1, 3),
    )


@dataclass
class CaptureKeypointConfig(DetectionConfig):
    radius_squares: float = 0.3  # synthesized box radius (squares); see project_piece_box
    margin: float = 0.15  # expand each synthesized box this fraction per side (RoI context)


class CaptureKeypointDetection(Dataset):
    """`(image, target)` items from capture photos with synthesized boxes + contact
    keypoints. Mirrors `ChessReDKeypointDetection`'s target format and reuses the
    shared `resize_targets`/`augment_targets` so scaling/flips track identically."""

    def __init__(
        self,
        samples: Sequence[CaptureSample],
        s3: StorageConfig | None = None,
        config: CaptureKeypointConfig | None = None,
        train: bool = False,
    ):
        self.config = config or CaptureKeypointConfig()
        self.s3 = s3
        self.train = train
        # Keep only fully-cornered samples that yield >=1 on-board piece.
        kept = []
        for s in samples:
            if not s.has_all_corners:
                continue
            boxes, _, _ = synthesize_piece_targets(
                s, radius_squares=self.config.radius_squares, margin=self.config.margin
            )
            if len(boxes):
                kept.append(s)
        self.samples = kept

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rgb = sample.load_image(self.s3)
        boxes, labels, keypoints = synthesize_piece_targets(
            sample, radius_squares=self.config.radius_squares, margin=self.config.margin
        )

        rgb, boxes, keypoints = resize_targets(rgb, boxes, keypoints, self.config.max_size)
        if self.train:
            rgb, boxes, keypoints = augment_targets(
                rgb,
                boxes,
                keypoints,
                hflip_prob=self.config.hflip_prob,
                jitter=self.config.jitter,
                color=self.config.color,
                blur=self.config.blur,
                noise=self.config.noise,
            )

        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        return image, {
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
            "labels": torch.from_numpy(labels),
            "keypoints": torch.from_numpy(np.ascontiguousarray(keypoints)),
            "image_id": torch.tensor([sample.task_id], dtype=torch.int64),
        }


def split_by_sessions(
    dataset: CaptureDataset,
    val_sessions: Collection[str],
    config: CaptureKeypointConfig | None = None,
) -> tuple[CaptureKeypointDetection, CaptureKeypointDetection]:
    """Split capture samples into (train, val) by session: sessions in `val_sessions`
    validate, the rest train. Splitting by **session** (not random frame) is mandatory
    -- frames within a session are near-duplicate, so a random split leaks them.

    With only two physical sets in the capture data (see the captures-two-boards note),
    hold out a few sessions from *each* set so both train and val cover both boards: the
    model must have seen a board to classify its pieces, and val then measures
    generalization to unseen sessions/positions of boards it knows. Train augments;
    val does not.
    """
    val_set = set(val_sessions)
    unknown = val_set - set(dataset.sessions)
    if unknown:
        raise ValueError(f"unknown sessions {sorted(unknown)}; have {dataset.sessions}")
    by_session = dataset.by_session()
    train_samples = [s for sess, group in by_session.items() if sess not in val_set for s in group]
    val_samples = [s for sess in val_set for s in by_session[sess]]
    train = CaptureKeypointDetection(train_samples, dataset.s3, config, train=True)
    val = CaptureKeypointDetection(val_samples, dataset.s3, config, train=False)
    return train, val
