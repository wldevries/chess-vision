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
CornerDict = dict[str, list[float]]
Point = Sequence[float]


def quad_area(corners: Corners) -> float:
    """Absolute area (px^2) of the corner quad, via the shoelace formula.

    Used to reject degenerate / collinear corner labels before building H.
    """
    ring = np.array([corners[k] for k in IMAGE_CORNER_RING], dtype=np.float64)
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def order_corners(points: Sequence[Point]) -> CornerDict:
    """Sort 4 arbitrary board-corner points into visual TL/TR/BR/BL slots.

    Lets the caller click/detect the corners in *any* order: we split the four by
    `y` into a top pair and a bottom pair, then split each pair by `x`. Robust for
    any realistic board pose (it only mislabels if the board is shot rotated ~45°,
    like a diamond -- not a real seated-player capture). This solves only the
    *visual slot* assignment; which physical corner is a8 (R0/R90/R180/R270) is a
    separate, non-geometric choice the caller still has to make.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) != 4:
        raise ValueError(f"need exactly 4 corner points, got {len(pts)}")
    top, bottom = sorted(pts, key=lambda p: p[1])[:2], sorted(pts, key=lambda p: p[1])[2:]
    tl, tr = sorted(top, key=lambda p: p[0])
    bl, br = sorted(bottom, key=lambda p: p[0])
    return {
        "top_left": list(tl),
        "top_right": list(tr),
        "bottom_right": list(br),
        "bottom_left": list(bl),
    }


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


def square_center_uv(square: str) -> tuple[float, float]:
    """Canonical (u, v) of a square's center, e.g. "e4" -> (0.5625, 0.5625).

    The inverse of `uv_to_square` at square granularity: where a piece standing
    on `square` is expected to meet the board. Project it through a homography to
    get the piece's approximate base point in the image.
    """
    file_idx = FILES.index(square[0])
    rank = int(square[1])
    return ((file_idx + 0.5) / 8.0, (8.5 - rank) / 8.0)


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


# Per-piece box heights in squares, for `project_piece_box`. Keyed by lowercase FEN
# letter, pawn shortest -> king tallest. The box is only an RoI crop, so a piece must
# never poke out the top (clipping starves the head) while over-covering is harmless.
# These are therefore biased to cover the *tallest* set we capture (a cheap wooden set
# whose pieces are tall relative to its squares), which over-covers shorter Staunton-
# style sets slightly -- exactly the safe direction. Tune together via a multiplier.
PIECE_HEIGHT_SCALE: dict[str, float] = {
    "p": 0.85,
    "r": 1.05,
    "n": 1.1,
    "b": 1.25,
    "q": 1.45,
    "k": 1.75,
}


def focal_from_homography(homography: np.ndarray, cx: float, cy: float) -> float | None:
    """Estimate focal length from a single board->image homography (Zhang).

    With principal point `(cx, cy)` and square, skewless pixels, the constraint that
    `r1, r2` are orthonormal (`h1' . h2' = 0`, `|h1'| = |h2'|`, with `h' = K^-1 h`)
    gives two equations linear in `w = 1/f^2`. Returns `f = 1/sqrt(w)` (averaging the
    two), or None if the view is too degenerate to give a positive `w`.
    """
    a1, a2, a3 = (float(x) for x in homography[:, 0])
    b1, b2, b3 = (float(x) for x in homography[:, 1])
    r2 = cx * cx + cy * cy
    # h1'.h2' = 0  ->  coef1 * w + a3*b3 = 0
    coef1 = a1 * b1 + a2 * b2 - cx * (a3 * b1 + a1 * b3) - cy * (a3 * b2 + a2 * b3) + r2 * a3 * b3
    # |h1'|^2 - |h2'|^2 = 0  ->  coef2 * w + (a3^2 - b3^2) = 0
    coef2 = (
        (a1 * a1 + a2 * a2)
        - (b1 * b1 + b2 * b2)
        - 2 * cx * (a1 * a3 - b1 * b3)
        - 2 * cy * (a2 * a3 - b2 * b3)
        + r2 * (a3 * a3 - b3 * b3)
    )
    ws = []
    if abs(coef1) > 1e-12:
        ws.append(-a3 * b3 / coef1)
    if abs(coef2) > 1e-12:
        ws.append(-(a3 * a3 - b3 * b3) / coef2)
    ws = [w for w in ws if w > 0]
    if not ws:
        return None
    return float((sum(ws) / len(ws)) ** -0.5)


def camera_from_homography(
    homography: np.ndarray, image_size: tuple[int, int], focal_scale: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover a pinhole camera (K, R, t) from a board->image homography.

    Principal point is the image center. Focal length is **estimated from the
    homography** (`focal_from_homography`); pass `focal_scale` to instead force
    `f = focal_scale * max(W, H)` (fallback when estimation fails). With
    `H = K [r1 r2 t]`, the rotation columns and translation follow up to scale;
    `r3 = r1 x r2` completes the (SVD-orthonormalized) rotation. Canonical board
    coords are the unit square [0,1]^2 (= 8 squares); Z is out of the board plane.
    """
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    f = (focal_scale * max(w, h)) if focal_scale else focal_from_homography(homography, cx, cy)
    if not f:
        f = max(w, h)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    M = np.linalg.inv(K) @ np.asarray(homography, dtype=np.float64)
    scale = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * scale, M[:, 1] * scale, M[:, 2] * scale
    if t[2] < 0:  # board must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    return K, u @ vt, t


def project_piece_box(
    homography: np.ndarray,
    base: Point,
    image_size: tuple[int, int],
    *,
    height_squares: float = 1.0,
    radius_squares: float = 0.3,
    focal_scale: float | None = None,
) -> tuple[float, float, float, float]:
    """Bounding box (xyxy) of a piece modelled as a vertical 3D cylinder on the board.

    Given only a board-contact point (e.g. a hand-tagged keypoint, no box), recover
    the camera (`camera_from_homography`) and frame a cylinder of radius
    `radius_squares` and height `height_squares` (in squares) standing at the base.

    Key subtlety: a piece is a 3D object standing *above* the board plane, so its
    silhouette half-width in the image is the **apparent radius** ``f * r / depth`` --
    NOT the radius of a board-plane circle (which foreshortens to near-zero for pieces
    seen at a steep angle and would leave the box stuck against the contact point on
    its near side). We therefore project the cylinder *axis* (base -> top) and inflate
    that segment by the apparent radius at each end. The contact point then sits a full
    radius in from the box on both sides; the lean of the axis adds the asymmetric
    extra width on the side the piece tilts toward.

    `height_squares` should be set per piece type (see `PIECE_HEIGHT_SCALE`). Still
    approximate (assumed focal length, straight cylinder); the contact point remains
    the exact keypoint target.
    """
    base = np.asarray(base, dtype=np.float64).reshape(2)
    K, rot, t = camera_from_homography(homography, image_size, focal_scale)
    f = float(K[0, 0])
    u, v = image_to_canonical(homography, base.reshape(1, 2).astype(np.float32))[0]
    z = height_squares / 8.0  # squares -> canonical units (board edge = 1 = 8 squares)
    r = radius_squares / 8.0

    def project(pts3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = pts3d @ rot.T + t  # camera-frame coords; cam[:, 2] is depth (>0 in front)
        img = cam @ K.T
        return img[:, :2] / img[:, 2:3], cam[:, 2]

    # +Z must point toward the camera (piece tops rise *up* in the image, smaller y).
    top_xy, _ = project(np.array([[u, v, z]]))
    sign = 1.0 if top_xy[0, 1] < base[1] else -1.0
    axis, depth = project(np.array([[u, v, 0.0], [u, v, z * sign]]))
    # Apparent silhouette radius (px) at each end: f * r / depth. A cylinder cross-
    # section projects to ~this circle regardless of board foreshortening, so it sits
    # symmetrically around the axis on both sides.
    rad = f * r / np.maximum(depth, 1e-6)
    x1 = float(min(axis[0, 0] - rad[0], axis[1, 0] - rad[1]))
    x2 = float(max(axis[0, 0] + rad[0], axis[1, 0] + rad[1]))
    y1 = float(min(axis[0, 1] - rad[0], axis[1, 1] - rad[1]))
    y2 = float(max(axis[0, 1] + rad[0], axis[1, 1] + rad[1]))
    # The contact point is the keypoint target -- guarantee it sits inside the box.
    bx, by = float(base[0]), float(base[1])
    return (min(x1, bx), min(y1, by), max(x2, bx), max(y2, by))


def lattice_points(homography: np.ndarray) -> np.ndarray:
    """The 9x9 = 81 grid-corner points projected into the image, shape (81, 2),
    ordered row-major by (v, u) so it reshapes to (9, 9, 2) for drawing lines."""
    coords = np.array([[i / 8.0, j / 8.0] for j in range(9) for i in range(9)], dtype=np.float32)
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
