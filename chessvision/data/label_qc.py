"""Capture-label quality checks, two strengths.

1. `count_problems` -- offline heuristic from labels + corners alone: duplicate squares,
   piece counts above the no-promotion maximum, wrong king count, >32 pieces. Catches
   gross errors but not a wrong-class piece that keeps counts legal.

2. `game_truth_problems` -- authoritative: compare the labelled board to the **true game
   position**. Capture filenames are ``<game_id>_ply<NNN>_<ts>.jpg``; for a PGN-sourced
   game, replaying to that ply gives the exact position, so every label is checked for
   wrong square / wrong class / missing / phantom. (Also validates the orientation
   convention -- a wrong orientation mismatches every frame.)

Pure (labels/corners/FEN only); the scripts own IO/CLI. python-chess parses FENs.
"""

from __future__ import annotations

from collections import Counter

import chess

from chessvision.geometry import Orientation, compute_homography, square_for_point

# Per-colour max without promotions; FEN letter -> count.
LEGAL_MAX = {"P": 8, "p": 8, "N": 2, "n": 2, "B": 2, "b": 2, "R": 2, "r": 2, "Q": 1, "q": 1}


def label_board(sample) -> dict[str, str]:
    """Hand-labelled {square: fen-letter} via each contact point's square (R0).
    A duplicate square (two pieces) collapses here -- `count_problems` flags it."""
    homography = compute_homography(sample.corners, Orientation.R0)
    out: dict[str, str] = {}
    for p in sample.pieces:
        sq = square_for_point(homography, p.point)
        if sq is not None:
            out[sq] = p.fen
    return out


def count_problems(sample) -> list[str]:
    """Heuristic problems from labels alone (empty == looks fine)."""
    homography = compute_homography(sample.corners, Orientation.R0)
    on_board = [
        (p.fen, sq)
        for p in sample.pieces
        if (sq := square_for_point(homography, p.point)) is not None
    ]
    problems: list[str] = []
    dup = [sq for sq, c in Counter(sq for _, sq in on_board).items() if c > 1]
    if dup:
        problems.append(f"two pieces on square(s) {dup}")
    counts = Counter(fen for fen, _ in on_board)
    over = [f"{k}={v}" for k, v in counts.items() if v > LEGAL_MAX.get(k, 99)]
    if over:
        problems.append(f"too many: {', '.join(over)}")
    for king, colour in (("K", "white"), ("k", "black")):
        if counts.get(king, 0) != 1:
            problems.append(f"{counts.get(king, 0)} {colour} king(s)")
    if len(on_board) > 32:
        problems.append(f"{len(on_board)} pieces (>32)")
    return problems


def parse_source(image_uri: str) -> tuple[str, int] | None:
    """`.../euwe-0000_ply042_20260524-...jpg` -> ("euwe-0000", 42), or None."""
    stem = image_uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2 or not parts[1].startswith("ply"):
        return None
    try:
        return parts[0], int(parts[1][3:])
    except ValueError:
        return None


def true_board(fen: str) -> dict[str, str]:
    """FEN -> {square_name: piece_symbol}, e.g. {'e1': 'K', 'c6': 'n'}."""
    board = chess.Board(fen)
    return {chess.square_name(sq): pc.symbol() for sq, pc in board.piece_map().items()}


def game_truth_problems(sample, truth: dict[str, str]) -> list[str]:
    """Compare labels to the true `square -> piece`: missing / extra / wrong-class."""
    labelled = label_board(sample)
    missing = sorted(set(truth) - set(labelled))
    extra = sorted(set(labelled) - set(truth))
    wrong = {
        sq: f"{truth[sq]}!={labelled[sq]}"
        for sq in set(truth) & set(labelled)
        if truth[sq] != labelled[sq]
    }
    problems: list[str] = []
    if missing:
        problems.append(f"missing {missing}")
    if extra:
        problems.append(f"extra {extra}")
    if wrong:
        problems.append(f"wrong {wrong}")
    return problems
