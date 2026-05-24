"""Homography self-check: does projecting each piece's base point through the
board homography reproduce its labelled square?

Pure logic over already-loaded annotations; returns dataclasses and writes
nothing (the CLI in scripts/ owns all IO). Works on any source that yields
images exposing `.meta.image_id`, `.corners`, and `.pieces` (each piece a
`.piece_id`, `.category_id`, `.square`, `.bbox`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from chessvision.geometry import (
    Orientation,
    bbox_base_point,
    compute_homography,
    quad_area,
    squares_for_points,
)

DEFAULT_ORIENTATIONS: tuple[Orientation, ...] = tuple(Orientation)
DEFAULT_TOL = 0.06  # ~half a square; tolerates a base point just past the far edge
MIN_QUAD_AREA = 1.0  # px^2; below this the corners are degenerate/collinear


@dataclass(frozen=True)
class PieceResult:
    piece_id: int
    category_id: int
    label_square: str
    pred_square: str | None
    base_xy: tuple[float, float]
    matched: bool
    offboard: bool


@dataclass(frozen=True)
class ImageResult:
    image_id: int
    orientation: Orientation | None
    n_pieces: int
    n_matched: int
    n_offboard: int
    accuracy: float
    errored: bool
    results: list[PieceResult] = field(default_factory=list)

    @property
    def mismatches(self) -> list[PieceResult]:
        return [r for r in self.results if not r.matched]


@dataclass
class SelfCheckReport:
    per_image: list[ImageResult]
    n_images: int
    n_errored: int
    n_pieces: int
    n_matched: int
    n_offboard: int
    global_accuracy: float
    orientation_counts: dict[str, int]
    flagged: list[ImageResult]
    flag_threshold: float
    offset_table: dict[float, float] | None = None


def _base_points(pieces: Sequence, vertical_offset: float) -> np.ndarray:
    return np.array([bbox_base_point(p.bbox, vertical_offset) for p in pieces], dtype=np.float32)


def _best_orientation(
    corners: Mapping,
    base_pts: np.ndarray,
    labels: Sequence[str],
    orientations: Sequence[Orientation],
    tol: float,
) -> tuple[Orientation, list[str | None]]:
    """Pick the orientation whose homography reproduces the most labels.
    Ties resolve to the lowest Orientation (iteration order)."""
    best_orient = orientations[0]
    best_preds: list[str | None] = []
    best_score = -1
    for orient in orientations:
        homography = compute_homography(corners, orient)
        preds = squares_for_points(homography, base_pts, tol)
        score = sum(p == lab for p, lab in zip(preds, labels, strict=True))
        if score > best_score:
            best_score, best_orient, best_preds = score, orient, preds
    return best_orient, best_preds


def check_image(
    image_id: int,
    corners: Mapping,
    pieces: Sequence,
    orientations: Sequence[Orientation] = DEFAULT_ORIENTATIONS,
    tol: float = DEFAULT_TOL,
    vertical_offset: float = 0.0,
) -> ImageResult:
    usable = [p for p in pieces if p.bbox is not None]

    if quad_area(corners) < MIN_QUAD_AREA:
        return ImageResult(image_id, None, len(usable), 0, 0, 1.0, errored=True)

    if not usable:
        return ImageResult(image_id, Orientation.R0, 0, 0, 0, 1.0, errored=False)

    labels = [p.square for p in usable]
    base_pts = _base_points(usable, vertical_offset)
    orient, preds = _best_orientation(corners, base_pts, labels, orientations, tol)

    results: list[PieceResult] = []
    for piece, base, pred in zip(usable, base_pts, preds, strict=True):
        results.append(
            PieceResult(
                piece_id=piece.piece_id,
                category_id=piece.category_id,
                label_square=piece.square,
                pred_square=pred,
                base_xy=(float(base[0]), float(base[1])),
                matched=pred == piece.square,
                offboard=pred is None,
            )
        )
    n_matched = sum(r.matched for r in results)
    n_offboard = sum(r.offboard for r in results)
    return ImageResult(
        image_id=image_id,
        orientation=orient,
        n_pieces=len(usable),
        n_matched=n_matched,
        n_offboard=n_offboard,
        accuracy=n_matched / len(usable),
        errored=False,
        results=results,
    )


def run(
    images: Iterable,
    orientations: Sequence[Orientation] = DEFAULT_ORIENTATIONS,
    tol: float = DEFAULT_TOL,
    vertical_offset: float = 0.0,
    flag_threshold: float = 0.9,
) -> SelfCheckReport:
    per_image = [
        check_image(img.meta.image_id, img.corners, img.pieces, orientations, tol, vertical_offset)
        for img in images
    ]

    scored = [r for r in per_image if not r.errored and r.n_pieces > 0]
    n_pieces = sum(r.n_pieces for r in scored)
    n_matched = sum(r.n_matched for r in scored)
    n_offboard = sum(r.n_offboard for r in scored)
    orient_counts = Counter(r.orientation.name for r in scored)
    flagged = sorted((r for r in scored if r.accuracy < flag_threshold), key=lambda r: r.accuracy)
    return SelfCheckReport(
        per_image=per_image,
        n_images=len(per_image),
        n_errored=sum(r.errored for r in per_image),
        n_pieces=n_pieces,
        n_matched=n_matched,
        n_offboard=n_offboard,
        global_accuracy=(n_matched / n_pieces) if n_pieces else 0.0,
        orientation_counts=dict(orient_counts),
        flagged=flagged,
        flag_threshold=flag_threshold,
    )


def sweep_vertical_offset(
    images: Iterable,
    ks: Sequence[float],
    orientations: Sequence[Orientation] = DEFAULT_ORIENTATIONS,
    tol: float = DEFAULT_TOL,
) -> dict[float, float]:
    """Global accuracy as a function of base-point vertical offset k.

    Orientation is fixed once per image at k=0 so a large k can't mask a wrong
    orientation; only the base point moves up by k*height for each k tested.
    """
    # Freeze orientation + homography per image at offset 0.
    frozen = []  # (homography, base_pts_per_k_recomputable bboxes, labels)
    for img in images:
        usable = [p for p in img.pieces if p.bbox is not None]
        if not usable or quad_area(img.corners) < MIN_QUAD_AREA:
            continue
        labels = [p.square for p in usable]
        base0 = _base_points(usable, 0.0)
        orient, _ = _best_orientation(img.corners, base0, labels, orientations, tol)
        frozen.append((compute_homography(img.corners, orient), usable, labels))

    table: dict[float, float] = {}
    for k in ks:
        matched = total = 0
        for homography, usable, labels in frozen:
            base_pts = _base_points(usable, k)
            preds = squares_for_points(homography, base_pts, tol)
            matched += sum(p == lab for p, lab in zip(preds, labels, strict=True))
            total += len(labels)
        table[k] = (matched / total) if total else 0.0
    return table
