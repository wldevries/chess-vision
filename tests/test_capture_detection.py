"""Capture keypoint-detection targets: synthesized boxes + contact keypoints.

Pure-geometry tests (no image decode / no S3) on a hand-built CaptureSample, so
they run offline. Locks in: off-board drop, label mapping to the 1..12 scheme,
and the contact point sitting strictly inside its (margined) box."""

from __future__ import annotations

import numpy as np

from chessvision.data.capture_detection import (
    fen_to_label,
    synthesize_piece_targets,
)
from chessvision.data.captures import CaptureSample, PieceKeypoint
from chessvision.data.detection import LABEL_NAMES
from chessvision.geometry import Orientation, canonical_to_image, compute_homography

# A perspective-distorted board filling a 1000x800 image.
CORNERS = {
    "top_left": (120.0, 140.0),
    "top_right": (880.0, 110.0),
    "bottom_left": (70.0, 690.0),
    "bottom_right": (940.0, 660.0),
}


def _img_pt(uv: tuple[float, float]) -> tuple[float, float]:
    h = compute_homography(CORNERS, Orientation.R0)
    x, y = canonical_to_image(h, np.array([uv], dtype=np.float32))[0]
    return float(x), float(y)


def _sample(pieces: list[PieceKeypoint]) -> CaptureSample:
    return CaptureSample(
        task_id=1,
        session="s",
        image_path=__import__("pathlib").Path("nope.jpg"),
        s3_uri="s3://b/captures/s/x.jpg",
        width=1000,
        height=800,
        corners=CORNERS,
        pieces=pieces,
    )


def test_fen_label_matches_detection_scheme():
    assert fen_to_label("P") == 1 and LABEL_NAMES[1] == "white-pawn"
    assert fen_to_label("k") == 12 and LABEL_NAMES[12] == "black-king"
    # every label lands in the 12-piece range
    for fen in "PRNBQKprnbqk":
        assert 1 <= fen_to_label(fen) <= 12


def test_on_board_piece_has_box_and_contained_keypoint():
    e4 = _img_pt((4.5 / 8.0, 4.5 / 8.0))  # center of a square, well on-board
    boxes, labels, kpts = synthesize_piece_targets(_sample([PieceKeypoint("WhiteQueen", e4)]))
    assert boxes.shape == (1, 4) and kpts.shape == (1, 1, 3)
    assert labels.tolist() == [fen_to_label("Q")]
    (x1, y1, x2, y2) = boxes[0]
    kx, ky = kpts[0, 0, 0], kpts[0, 0, 1]
    assert x1 < kx < x2 and y1 < ky < y2  # strictly inside (margin gives breathing room)
    assert kpts[0, 0, 2] == 2.0  # visibility = labelled


def test_off_board_piece_dropped():
    off = _img_pt((1.6, 1.6))  # projected well outside the board
    boxes, labels, kpts = synthesize_piece_targets(_sample([PieceKeypoint("BlackRook", off)]))
    assert len(boxes) == 0 and len(labels) == 0 and len(kpts) == 0


def test_margin_enlarges_box():
    e4 = _img_pt((4.5 / 8.0, 4.5 / 8.0))
    s = _sample([PieceKeypoint("WhiteKing", e4)])
    tight, _, _ = synthesize_piece_targets(s, margin=0.0)
    padded, _, _ = synthesize_piece_targets(s, margin=0.2)
    ta = (tight[0, 2] - tight[0, 0]) * (tight[0, 3] - tight[0, 1])
    pa = (padded[0, 2] - padded[0, 0]) * (padded[0, 3] - padded[0, 1])
    assert pa > ta
