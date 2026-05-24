"""Board-contact points from ChessReD labels (the doctrine-pure base point).

A piece's **board-contact point** is where it meets the board. We compute it
*without any bounding box*: ChessReD gives each piece's true square plus the 4
board corners, so the contact point is simply the square's center projected
through the homography. This is the target a base-keypoint head should learn,
and it is exactly the thing a box's bottom-center fails to estimate (see
CLAUDE.md anti-patterns / plan.md section 4).

Also provides an **occlusion score** for selecting hard images: a piece counts
as base-occluded when its contact point falls inside a *nearer* piece's bounding
box (nearer = lower in the image). Those are precisely the cases where reading
the base off a box breaks and a learned contact point must extrapolate -- good
candidates to curate and hand-verify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chessvision.data.chessred import AnnotatedImage, Piece
from chessvision.geometry import (
    Orientation,
    canonical_to_image,
    compute_homography,
    square_center_uv,
)


@dataclass(frozen=True)
class ContactPoint:
    piece: Piece
    xy: tuple[float, float]  # image pixels; projected square center, box-independent

    @property
    def square(self) -> str:
        return self.piece.square

    @property
    def category_id(self) -> int:
        return self.piece.category_id


def contact_points(
    image: AnnotatedImage, orientation: Orientation = Orientation.R0
) -> list[ContactPoint]:
    """Each piece's board-contact point in image pixels (square center → H → image).

    ChessReD self-check found every corner-annotated image resolves to R0, so
    that is the default orientation.
    """
    if not image.pieces:
        return []
    homography = compute_homography(image.corners, orientation)
    uvs = np.array([square_center_uv(p.square) for p in image.pieces], dtype=np.float32)
    pts = canonical_to_image(homography, uvs)
    return [
        ContactPoint(p, (float(x), float(y))) for p, (x, y) in zip(image.pieces, pts, strict=True)
    ]


def occluded_pieces(
    image: AnnotatedImage, orientation: Orientation = Orientation.R0
) -> list[Piece]:
    """Pieces whose contact point lies inside a *nearer* piece's bbox.

    "Nearer" = larger contact-point y (lower in image ⇒ closer to the camera).
    This proxies "the base is hidden behind a piece in front" -- the failure case
    for box-bottom-center. (The bbox here is only used to test coverage; the
    contact point itself is box-independent, per the doctrine.)
    """
    cps = [cp for cp in contact_points(image, orientation) if cp.piece.bbox is not None]
    hidden: list[Piece] = []
    for b in cps:
        bx, by = b.xy
        for a in cps:
            if a is b:
                continue
            ax, ay, aw, ah = a.piece.bbox  # COCO xywh
            inside = ax <= bx <= ax + aw and ay <= by <= ay + ah
            nearer = a.xy[1] > by  # a sits lower in the image ⇒ in front of b
            if inside and nearer:
                hidden.append(b.piece)
                break
    return hidden


def occlusion_score(image: AnnotatedImage, orientation: Orientation = Orientation.R0) -> int:
    """Number of base-occluded pieces; higher = harder for box-based base points."""
    return len(occluded_pieces(image, orientation))
