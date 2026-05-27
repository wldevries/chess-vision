"""Lattice target generation + decode geometry (the 81-point board model)."""

from __future__ import annotations

import cv2
import numpy as np

from chessvision.corner_regressor import homography_from_lattice
from chessvision.data.corners import LATTICE_CANONICAL, NUM_LATTICE, corners_to_lattice
from chessvision.geometry import CANONICAL_ANCHORS

# A valid (non-degenerate) board quad in CORNER_ORDER = (TL, TR, BR, BL), pixel coords.
QUAD = np.array([[120, 140], [880, 180], [820, 700], [160, 660]], dtype=np.float32)


def test_lattice_canonical_layout():
    assert LATTICE_CANONICAL.shape == (NUM_LATTICE, 2) == (81, 2)
    # row-major (rank j, file i): corners sit at indices 0, 8, 72, 80.
    assert tuple(LATTICE_CANONICAL[0]) == (0.0, 0.0)
    assert tuple(LATTICE_CANONICAL[8]) == (1.0, 0.0)
    assert tuple(LATTICE_CANONICAL[72]) == (0.0, 1.0)
    assert tuple(LATTICE_CANONICAL[80]) == (1.0, 1.0)


def test_corners_to_lattice_places_the_four_corners():
    lat = corners_to_lattice(QUAD)
    assert lat.shape == (81, 2)
    # the 4 lattice-grid corners must coincide with the input quad (TL,TR,BR,BL).
    for lattice_idx, quad_idx in {0: 0, 8: 1, 80: 2, 72: 3}.items():
        assert np.allclose(lat[lattice_idx], QUAD[quad_idx], atol=1e-3)


def test_lattice_fit_roundtrip_recovers_corners():
    """corners -> 81 lattice points -> fit H -> project anchors back == original corners."""
    lat = corners_to_lattice(QUAD)
    h = homography_from_lattice(lat, conf=None)
    recovered = cv2.perspectiveTransform(
        CANONICAL_ANCHORS.reshape(-1, 1, 2).astype(np.float32), h
    ).reshape(-1, 2)
    # anchors (a8,h8,a1,h1) -> visual (TL,TR,BL,BR); QUAD is (TL,TR,BR,BL).
    expected = QUAD[[0, 1, 3, 2]]
    assert np.allclose(recovered, expected, atol=1e-2)


def test_confidence_weighting_is_robust_to_one_bad_point():
    """A single grossly-wrong point with low confidence should barely move the fit."""
    lat = corners_to_lattice(QUAD)
    conf = np.ones(81, dtype=np.float32)
    bad = lat.copy()
    bad[40] += np.array([300.0, -250.0], dtype=np.float32)  # wreck an interior point
    conf[40] = 1e-3  # ...but flag it as unconfident
    h_clean = homography_from_lattice(lat, conf=None)
    h_weighted = homography_from_lattice(bad, conf=conf)
    c_clean = cv2.perspectiveTransform(
        CANONICAL_ANCHORS.reshape(-1, 1, 2).astype(np.float32), h_clean
    ).reshape(-1, 2)
    c_weighted = cv2.perspectiveTransform(
        CANONICAL_ANCHORS.reshape(-1, 1, 2).astype(np.float32), h_weighted
    ).reshape(-1, 2)
    # down-weighting keeps the corners within a few px despite the gross outlier.
    assert np.max(np.linalg.norm(c_clean - c_weighted, axis=1)) < 5.0
