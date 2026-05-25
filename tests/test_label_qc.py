"""Label QC: count heuristic + game-truth comparison (offline, no images)."""

from __future__ import annotations

import numpy as np

from chessvision.data.captures import CaptureSample, PieceKeypoint
from chessvision.data.label_qc import (
    count_problems,
    game_truth_problems,
    label_board,
    parse_source,
    true_board,
)
from chessvision.geometry import Orientation, canonical_to_image, compute_homography

CORNERS = {
    "top_left": (120.0, 140.0),
    "top_right": (880.0, 110.0),
    "bottom_left": (70.0, 690.0),
    "bottom_right": (940.0, 660.0),
}


def _at(square: str) -> tuple[float, float]:
    from chessvision.geometry import square_center_uv

    h = compute_homography(CORNERS, Orientation.R0)
    uv = np.array([square_center_uv(square)], dtype=np.float32)
    x, y = canonical_to_image(h, uv)[0]
    return float(x), float(y)


def _sample(pieces):
    return CaptureSample(
        task_id=1,
        session="s",
        image_path=__import__("pathlib").Path("x.jpg"),
        s3_uri="s3://b/captures/s/euwe-0000_ply007_x.jpg",
        width=1000,
        height=800,
        corners=CORNERS,
        pieces=pieces,
    )


def test_parse_source():
    assert parse_source("s3://b/captures/s/euwe-0000_ply042_20260524.jpg") == ("euwe-0000", 42)
    assert parse_source("s3://b/captures/s/puzzle-1dMGZ_ply000_x.jpg") == ("puzzle-1dMGZ", 0)
    assert parse_source("s3://b/x/no_ply_here.jpg") is None


def test_true_board_startpos():
    tb = true_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert tb["e1"] == "K" and tb["e8"] == "k" and tb["a2"] == "P" and len(tb) == 32


def test_count_problems_flags_two_kings_and_clean():
    two_kings = _sample(
        [PieceKeypoint("WhiteKing", _at("e1")), PieceKeypoint("WhiteKing", _at("e2"))]
    )
    probs = count_problems(two_kings)
    assert any("black king" in p for p in probs)  # no black king
    clean = _sample([PieceKeypoint("WhiteKing", _at("e1")), PieceKeypoint("BlackKing", _at("e8"))])
    assert count_problems(clean) == []


def test_game_truth_problems_detects_wrong_missing_extra():
    # truth: white king e1, black knight c6. Label e1 king (ok), c6 BISHOP (wrong), extra a1 rook.
    truth = {"e1": "K", "c6": "n"}
    sample = _sample(
        [
            PieceKeypoint("WhiteKing", _at("e1")),
            PieceKeypoint("BlackBishop", _at("c6")),
            PieceKeypoint("WhiteRook", _at("a1")),
        ]
    )
    problems = game_truth_problems(sample, truth)
    joined = " ".join(problems)
    assert "wrong" in joined and "c6" in joined  # n != b
    assert "extra" in joined and "a1" in joined
    assert label_board(sample)["e1"] == "K"
