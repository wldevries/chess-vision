"""Board geometry: corners -> homography -> square assignment.

Pure geometry, no dataset/IO knowledge. This is the Phase-1 utility (plan.md
section 2): turn 4 board corners into a homography, project the 81-point lattice
and 64 square polygons, and map an image point (a piece's base point) to its
algebraic square.

Canonical board = the unit square [0, 1]^2 covering the 8x8 playing area:
    u = file axis (0 -> a-side, 1 -> h-side)
    v = rank axis (0 -> rank 8 / top, 1 -> rank 1 / bottom)
so the four canonical anchors are a8->(0,0), h8->(1,0), a1->(0,1), h1->(1,1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum

import cv2
import numpy as np

FILES = "abcdefgh"

# Canonical anchors, in the fixed order (a8, h8, a1, h1).
CANONICAL_ANCHORS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

# Image-corner keys as a closed ring (TL -> TR -> BR -> BL), used for area/shoelace.
IMAGE_CORNER_RING = ("top_left", "top_right", "bottom_right", "bottom_left")


class Orientation(IntEnum):
    """How the board (and its a8/h8/a1/h1 anchors) sits relative to the image.

    Each member maps the canonical anchors (a8, h8, a1, h1) to image-corner keys.
    R0 is the identity and matches ChessReD image 0 (top_left == a8 corner).
    """

    R0 = 0
    R90 = 90
    R180 = 180
    R270 = 270


# anchor order is always (a8, h8, a1, h1); values are the image-corner keys.
ORIENTATION_TO_KEYS: dict[Orientation, tuple[str, str, str, str]] = {
    Orientation.R0: ("top_left", "top_right", "bottom_left", "bottom_right"),
    Orientation.R90: ("top_right", "bottom_right", "top_left", "bottom_left"),
    Orientation.R180: ("bottom_right", "bottom_left", "top_right", "top_left"),
    Orientation.R270: ("bottom_left", "top_left", "bottom_right", "top_right"),
}

Corners = Mapping[str, Sequence[float]]
Point = Sequence[float]


def quad_area(corners: Corners) -> float:
    """Absolute area (px^2) of the corner quad, via the shoelace formula.

    Used to reject degenerate / collinear corner labels before building H.
    """
    ring = np.array([corners[k] for k in IMAGE_CORNER_RING], dtype=np.float64)
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_homography(corners: Corners, orientation: Orientation = Orientation.R0) -> np.ndarray:
    """Homography mapping canonical board coords (u, v) -> image pixels (x, y)."""
    keys = ORIENTATION_TO_KEYS[orientation]
    dst = np.array([corners[k] for k in keys], dtype=np.float32)
    return cv2.getPerspectiveTransform(CANONICAL_ANCHORS, dst)


def _transform(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 perspective transform to an (N, 2) array, returning (N, 2)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, matrix.astype(np.float32))
    return out.reshape(-1, 2)


def canonical_to_image(homography: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project canonical (u, v) points to image pixels."""
    return _transform(homography, pts)


def image_to_canonical(homography: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project image pixels back to canonical (u, v) via the inverse homography."""
    return _transform(np.linalg.inv(homography), pts)


def uv_to_square(u: float, v: float, tol: float = 0.0) -> str | None:
    """Map a canonical (u, v) point to an algebraic square, or None if off-board.

    `tol` (in canonical units, where one square is 1/8) widens the on-board test
    so a tall piece whose base lands slightly past the far edge still resolves.
    """
    if u < -tol or u > 1.0 + tol or v < -tol or v > 1.0 + tol:
        return None
    file_idx = int(np.clip(int(np.floor(u * 8)), 0, 7))
    rank = 8 - int(np.clip(int(np.floor(v * 8)), 0, 7))
    return f"{FILES[file_idx]}{rank}"


def square_for_point(homography: np.ndarray, pt: Point, tol: float = 0.0) -> str | None:
    """Map a single image point to its algebraic square (None if off-board)."""
    u, v = image_to_canonical(homography, np.asarray([pt], dtype=np.float32))[0]
    return uv_to_square(float(u), float(v), tol)


def squares_for_points(
    homography: np.ndarray, pts: np.ndarray, tol: float = 0.0
) -> list[str | None]:
    """Batch version of `square_for_point` (one inverse-transform for all points)."""
    uvs = image_to_canonical(homography, pts)
    return [uv_to_square(float(u), float(v), tol) for u, v in uvs]


def bbox_base_point(bbox: Sequence[float], vertical_offset: float = 0.0) -> tuple[float, float]:
    """Base point of a COCO bbox [x, y, w, h]: bottom-center, where the piece meets
    the board. `vertical_offset` (fraction of height) lifts the point off the bbox
    bottom to compensate for a piece leaning away from the camera."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h * (1.0 - vertical_offset))


def lattice_points(homography: np.ndarray) -> np.ndarray:
    """The 9x9 = 81 grid-corner points projected into the image, shape (81, 2),
    ordered row-major by (v, u) so it reshapes to (9, 9, 2) for drawing lines."""
    coords = np.array(
        [[i / 8.0, j / 8.0] for j in range(9) for i in range(9)], dtype=np.float32
    )
    return canonical_to_image(homography, coords)


def square_polygons(homography: np.ndarray) -> dict[str, np.ndarray]:
    """Image-space quad (TL, TR, BR, BL), shape (4, 2), for each of the 64 squares."""
    polys: dict[str, np.ndarray] = {}
    for r_top in range(8):  # 0 == rank 8 (top)
        for f in range(8):
            canon = np.array(
                [
                    [f / 8.0, r_top / 8.0],
                    [(f + 1) / 8.0, r_top / 8.0],
                    [(f + 1) / 8.0, (r_top + 1) / 8.0],
                    [f / 8.0, (r_top + 1) / 8.0],
                ],
                dtype=np.float32,
            )
            polys[f"{FILES[f]}{8 - r_top}"] = canonical_to_image(homography, canon)
    return polys
