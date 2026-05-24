"""Contact-point generator checks. Skipped when ChessReD isn't present."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from chessvision.data.chessred import ChessReD
from chessvision.data.contact import contact_points, occluded_pieces, occlusion_score
from chessvision.geometry import compute_homography, square_for_point, square_polygons

DATA_ROOT = Path("data/Chess Recognition Dataset (ChessReD)_2_all")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "annotations.json").exists(),
    reason="ChessReD dataset not present",
)


@pytest.fixture(scope="module")
def chessred() -> ChessReD:
    return ChessReD.load(DATA_ROOT)


def test_contact_point_round_trips_to_its_square(chessred: ChessReD):
    """The doctrine check: a projected square-center maps back to that square."""
    img = next(chessred.images_with_corners())
    h = compute_homography(img.corners)
    cps = contact_points(img)
    assert cps
    for cp in cps:
        assert square_for_point(h, cp.xy) == cp.square


def test_contact_point_lies_in_its_square_polygon(chessred: ChessReD):
    img = next(chessred.images_with_corners())
    polys = square_polygons(compute_homography(img.corners))
    for cp in contact_points(img):
        poly = polys[cp.square].astype(np.float32)
        # +1.0 tolerance: point should be inside, not merely on the edge
        assert cv2.pointPolygonTest(poly, (float(cp.xy[0]), float(cp.xy[1])), True) > -1.0


def test_occlusion_score_nonnegative_and_some_images_occluded(chessred: ChessReD):
    scores = []
    for i, img in enumerate(chessred.images_with_corners()):
        scores.append(occlusion_score(img))
        if i >= 60:
            break
    assert all(s >= 0 for s in scores)
    assert max(scores) > 0  # full starting positions are densely packed → occlusion


def test_occluded_pieces_are_a_subset(chessred: ChessReD):
    img = next(chessred.images_with_corners())
    occ = occluded_pieces(img)
    assert all(p in img.pieces for p in occ)
    assert len(occ) == occlusion_score(img)
