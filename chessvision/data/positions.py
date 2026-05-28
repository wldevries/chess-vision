"""Project a known position (FEN) onto a labelled corner photo.

This is the geometry behind the in-app *position-labelling* tool. Many corner photos
share the same physical setup, so rather than hand-label every piece we:

  1. mark the board corners once (the corner-capture flow already does this),
  2. pick the orientation that matches the photo, and
  3. project each piece's **square center through the homography** to get a starting
     board-contact point, which the user then nudges onto the true base.

Projecting a square center through ``H`` is exactly the doctrine-pure contact point
(see ``chessvision/data/contact.py`` and the CLAUDE.md anti-patterns) -- box-independent,
geometric truth. The only manual step is the nudge, which corrects for the piece leaning
and any small corner error. The output (image + corners + per-piece contact keypoints) is
the same shape the keypoint head trains on, with **no Label Studio round-trip**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chessvision.data.captures import PIECE_FEN, CaptureSample, PieceKeypoint
from chessvision.data.corner_capture import CornerStore
from chessvision.geometry import (
    FILES,
    Corners,
    Orientation,
    canonical_to_image,
    compute_homography,
    square_center_uv,
)

# FEN letter -> verbose keypoint label (the inverse of captures.PIECE_FEN, e.g. "P" ->
# "WhitePawn"). These verbose labels are what the capture/keypoint training stack uses.
LABEL_BY_FEN: dict[str, str] = {fen: label for label, fen in PIECE_FEN.items()}


def parse_board_fen(board_fen: str) -> list[tuple[str, str]]:
    """A FEN placement field -> [(square, fen_letter), ...] (e.g. ("e1", "K")).

    Accepts either a bare placement field ("rnbq...") or a full FEN (only the first
    space-separated field is read). Ranks run 8->1, files a->h, digits are empty runs.
    Raises ValueError on a malformed field so a bad paste fails loudly in the UI.
    """
    placement = board_fen.strip().split(" ", 1)[0]
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN placement must have 8 ranks, got {len(ranks)}")
    out: list[tuple[str, str]] = []
    for r, row in enumerate(ranks):
        rank = 8 - r
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
                continue
            if ch not in PIECE_FEN.values():
                raise ValueError(f"unknown FEN piece char {ch!r}")
            if file_idx > 7:
                raise ValueError(f"rank {rank} overflows 8 files")
            out.append((f"{FILES[file_idx]}{rank}", ch))
            file_idx += 1
        if file_idx != 8:
            raise ValueError(f"rank {rank} does not fill 8 files (got {file_idx})")
    return out


def project_position(
    corners: Corners, board_fen: str, orientation: Orientation = Orientation.R0
) -> list[dict]:
    """Project every piece in `board_fen` to its image-pixel contact point.

    `corners` is a CornerDict (TL/TR/BR/BL); `orientation` selects which physical corner
    is a8 (the manual 4-way choice -- not geometry-recoverable). Returns one dict per
    piece: ``{label, fen, square, x, y}`` with (x, y) in the same pixel frame as the
    corners. The point is the square *center* through H -- the starting handle position
    the user then nudges onto the real base.
    """
    pieces = parse_board_fen(board_fen)
    if not pieces:
        return []
    homography = compute_homography(corners, orientation)
    uvs = np.array([square_center_uv(sq) for sq, _ in pieces], dtype=np.float32)
    pts = canonical_to_image(homography, uvs)
    return [
        {
            "label": LABEL_BY_FEN[letter],
            "fen": letter,
            "square": sq,
            "x": float(x),
            "y": float(y),
        }
        for (sq, letter), (x, y) in zip(pieces, pts, strict=True)
    ]


def position_samples_as_captures(store: CornerStore | str | Path) -> list[CaptureSample]:
    """Position-labelled corner photos -> `CaptureSample`s for the keypoint fine-tune.

    Each labelled photo becomes one capture sample: the stored (EXIF-normalized) image,
    its four corners, and one `PieceKeypoint` per nudged contact point. The **session is
    keyed on the board** (``pos-<board>``) so the existing session-grouped split can hold
    out a whole physical board as a generalization test -- exactly what the 2-board
    capture set lacks (see captures-two-boards). Box sizing falls back to the global
    `PIECE_HEIGHT_SCALE` (positions carry no piece-set tag), which only affects the
    synthesized RoI, never the contact-point supervision.
    """
    store = store if isinstance(store, CornerStore) else CornerStore(store)
    samples: list[CaptureSample] = []
    for label in store.position_samples():
        image_path = store.store / label.image
        pieces = [PieceKeypoint(label=lbl, point=(x, y)) for lbl, x, y in label.pieces]
        samples.append(
            CaptureSample(
                task_id=label.task_id,
                session=f"pos-{label.board or '(untagged)'}",
                image_path=image_path,
                s3_uri=f"s3://positions/{label.image}",  # local file exists; S3 never hit
                width=label.width,
                height=label.height,
                corners={k: (float(x), float(y)) for k, (x, y) in label.corners.items()},
                pieces=pieces,
            )
        )
    return samples
