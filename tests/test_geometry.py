"""Geometry tests: synthetic round-trips (no dataset) + one golden test pinned to
real ChessReD image-0 numbers (inline, so no 22MB file is needed)."""

from __future__ import annotations

import numpy as np
import pytest

from chessvision.geometry import (
    FILES,
    Orientation,
    bbox_base_point,
    canonical_to_image,
    compute_homography,
    quad_area,
    square_for_point,
    uv_to_square,
)

# An arbitrary convex, perspective-distorted quad (not a rectangle).
SYNTH_CORNERS = {
    "top_left": [100.0, 120.0],
    "top_right": [900.0, 80.0],
    "bottom_left": [60.0, 820.0],
    "bottom_right": [1000.0, 760.0],
}


def _all_squares():
    for f in range(8):
        for r_top in range(8):  # 0 == rank 8
            yield f, r_top, f"{FILES[f]}{8 - r_top}"


def test_square_center_round_trip():
    """Project every square's canonical center into the image, then ask which
    square the image point belongs to: must recover the original square."""
    h = compute_homography(SYNTH_CORNERS, Orientation.R0)
    for f, r_top, square in _all_squares():
        center = np.array([[(f + 0.5) / 8.0, (r_top + 0.5) / 8.0]], dtype=np.float32)
        img_pt = canonical_to_image(h, center)[0]
        assert square_for_point(h, img_pt) == square


def test_bbox_base_point():
    assert bbox_base_point((10.0, 20.0, 40.0, 60.0)) == (30.0, 80.0)
    assert bbox_base_point((10.0, 20.0, 40.0, 60.0), vertical_offset=0.5) == (30.0, 50.0)


@pytest.mark.parametrize(
    "orientation",
    [Orientation.R0, Orientation.R90, Orientation.R180, Orientation.R270],
)
def test_orientation_places_a8(orientation):
    """The a8 square center must land nearest the image corner that each
    orientation assigns to the a8 anchor (the first anchor)."""
    h = compute_homography(SYNTH_CORNERS, orientation)
    a8_center = np.array([[0.5 / 8.0, 0.5 / 8.0]], dtype=np.float32)
    img_pt = canonical_to_image(h, a8_center)[0]

    from chessvision.geometry import ORIENTATION_TO_KEYS

    expected_key = ORIENTATION_TO_KEYS[orientation][0]
    dists = {
        k: float(np.hypot(img_pt[0] - v[0], img_pt[1] - v[1])) for k, v in SYNTH_CORNERS.items()
    }
    assert min(dists, key=dists.get) == expected_key


def test_off_board_and_edge():
    h = compute_homography(SYNTH_CORNERS, Orientation.R0)
    # A canonical point well outside the board -> off board.
    far = canonical_to_image(h, np.array([[1.5, 1.5]], dtype=np.float32))[0]
    assert square_for_point(h, far) is None
    # Exactly on the far edges clips to the last square rather than raising.
    assert uv_to_square(1.0, 0.0) == "h8"
    assert uv_to_square(0.0, 1.0) == "a1"
    assert uv_to_square(1.0, 1.0) == "h1"


def test_quad_area_rejects_degenerate():
    good = quad_area(SYNTH_CORNERS)
    assert good > 1.0
    collinear = {
        "top_left": [0.0, 0.0],
        "top_right": [1.0, 1.0],
        "bottom_right": [2.0, 2.0],
        "bottom_left": [3.0, 3.0],
    }
    assert quad_area(collinear) == pytest.approx(0.0, abs=1e-9)


# --- Golden test: real ChessReD image-0 corners + corner-piece bboxes (3072x3072) ---
GOLDEN_CORNERS = {
    "top_left": [488.7, 1078.7],
    "top_right": [1772.23, 638.59],
    "bottom_left": [1063.3, 2304.1],
    "bottom_right": [2610.3, 1560.9],
}
# (chessboard_position, COCO bbox [x, y, w, h]) for the four corner rooks of image 0.
GOLDEN_PIECES = [
    ("a8", [510.34, 963.65, 155.75, 186.14]),
    ("h8", [1689.96, 562.0, 110.16, 176.01]),
    ("a1", [1027.34, 2014.62, 167.14, 221.59]),
    ("h1", [2413.86, 1387.84, 144.35, 200.06]),
]


def test_golden_chessred_image0_corner_pieces():
    """R0 homography from image-0 corners must map each corner rook's base point
    to its labelled square -- locks in the orientation convention against real data."""
    h = compute_homography(GOLDEN_CORNERS, Orientation.R0)
    for square, bbox in GOLDEN_PIECES:
        base = bbox_base_point(bbox)
        assert square_for_point(h, base) == square
