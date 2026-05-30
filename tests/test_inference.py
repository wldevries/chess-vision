"""Tests for the live image -> FEN glue (model-free path) and corner ordering."""

from __future__ import annotations

import numpy as np
import pytest

from chessvision.geometry import order_corners
from chessvision.inference import (
    board_fen_from_squares,
    build_prediction,
    full_fen,
    label_to_symbol,
)

# An axis-aligned 800x800 board in image pixels. Under R0 the canonical unit square
# maps a8->TL(0,0), h8->TR(800,0), a1->BL(0,800), h1->BR(800,800), so a square's
# center projects to (u*800, v*800).
SQUARE_CORNERS = [[0, 0], [800, 0], [800, 800], [0, 800]]


def _center(square: str) -> list[float]:
    from chessvision.geometry import square_center_uv

    u, v = square_center_uv(square)
    return [u * 800.0, v * 800.0]


@pytest.mark.parametrize(
    "shuffled",
    [
        [[0, 0], [800, 0], [800, 800], [0, 800]],
        [[800, 800], [0, 0], [0, 800], [800, 0]],  # reordered
        [[0, 800], [800, 0], [0, 0], [800, 800]],  # reordered again
    ],
)
def test_order_corners_is_click_order_invariant(shuffled):
    ordered = order_corners(shuffled)
    assert ordered["top_left"] == [0.0, 0.0]
    assert ordered["top_right"] == [800.0, 0.0]
    assert ordered["bottom_right"] == [800.0, 800.0]
    assert ordered["bottom_left"] == [0.0, 800.0]


def test_order_corners_tolerates_perspective_skew():
    # A trapezoid (far edge foreshortened) shot slightly rotated: top still above bottom.
    pts = [[120, 40], [700, 60], [780, 760], [30, 740]]
    ordered = order_corners(pts)
    assert ordered["top_left"] == [120.0, 40.0]
    assert ordered["top_right"] == [700.0, 60.0]
    assert ordered["bottom_right"] == [780.0, 760.0]
    assert ordered["bottom_left"] == [30.0, 740.0]


def test_order_corners_rejects_wrong_count():
    with pytest.raises(ValueError):
        order_corners([[0, 0], [1, 0], [1, 1]])


def test_label_to_symbol_round_trips_fen_order():
    assert label_to_symbol(1) == "P"
    assert label_to_symbol(6) == "K"
    assert label_to_symbol(7) == "p"
    assert label_to_symbol(12) == "k"


def test_board_fen_from_squares_collapses_gaps():
    assert board_fen_from_squares({}) == "8/8/8/8/8/8/8/8"
    assert board_fen_from_squares({"a8": "P"}) == "P7/8/8/8/8/8/8/8"
    assert board_fen_from_squares({"h1": "k"}) == "8/8/8/8/8/8/8/7k"
    # Two pieces on rank 8 with a gap between them.
    assert board_fen_from_squares({"a8": "r", "c8": "r"}) == "r1r5/8/8/8/8/8/8/8"


def test_full_fen_wraps_placement():
    assert full_fen("8/8/8/8/8/8/8/8") == "8/8/8/8/8/8/8/8 w - - 0 1"


def test_build_prediction_places_piece_on_expected_square():
    points = np.array([_center("e2"), _center("d7")], dtype=np.float32)
    result = build_prediction(SQUARE_CORNERS, points, labels=[1, 7], scores=[0.9, 0.8])
    assert result.n_detected == 2
    r0 = result.orientations["R0"]
    assert r0.n_placed == 2
    # 'P' (white pawn) on e2, 'p' (black pawn) on d7.
    assert board_fen_from_squares({"e2": "P", "d7": "p"}) == r0.board_fen
    # Each detection records its square under every orientation.
    assert result.pieces[0].squares["R0"] == "e2"
    assert all(name in result.pieces[0].squares for name in ("R0", "R90", "R180", "R270"))


def test_build_prediction_resolves_one_piece_per_square_by_score():
    # Two detections collapse to the same square; the higher score wins.
    pt = _center("e4")
    points = np.array([pt, pt], dtype=np.float32)
    result = build_prediction(SQUARE_CORNERS, points, labels=[5, 1], scores=[0.4, 0.95])
    r0 = result.orientations["R0"]
    assert r0.n_placed == 1
    assert r0.board_fen == board_fen_from_squares({"e4": "P"})  # label 1, score 0.95


def test_build_prediction_accepts_corner_dict():
    result = build_prediction(
        order_corners(SQUARE_CORNERS), np.array([_center("a1")]), labels=[6], scores=[0.99]
    )
    assert result.orientations["R0"].board_fen == board_fen_from_squares({"a1": "K"})


def test_build_prediction_handles_no_detections():
    result = build_prediction(SQUARE_CORNERS, np.empty((0, 2)), labels=[], scores=[])
    assert result.n_detected == 0
    assert result.orientations["R0"].board_fen == "8/8/8/8/8/8/8/8"


# --- /api/live endpoint wiring (no torch / model: a stub predictor is injected) ---


class _StubPredictor:
    """Stands in for LivePredictor: ignores the image, returns fixed detections."""

    def __init__(self, points, labels, scores):
        self.points, self.labels, self.scores = points, labels, scores

    def predict(self, rgb, corners_in):
        return build_prediction(corners_in, np.asarray(self.points), self.labels, self.scores)


class _StubCornerPredictor:
    """Stands in for CornerPredictor: ignores the image, returns fixed board corners."""

    def predict(self, rgb):
        return order_corners(SQUARE_CORNERS)


def _app(predictor=None, corner_predictor=None):
    from chessvision.capture.app import create_app

    return create_app(
        [], "out_unused", predictor=predictor, corner_predictor=corner_predictor
    )


def _jpeg(h=80, w=80):
    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_live_mode_off_without_both_checkpoints():
    from fastapi.testclient import TestClient

    # Needs *both* the piece and corner predictors; either one missing -> off.
    for predictor, corner in (
        (None, None),
        (_StubPredictor([], [], []), None),
        (None, _StubCornerPredictor()),
    ):
        client = TestClient(_app(predictor=predictor, corner_predictor=corner))
        assert client.get("/api/live/available").json() == {"available": False}
        resp = client.post(
            "/api/live/predict", files={"image": ("f.jpg", _jpeg(), "image/jpeg")}
        )
        assert resp.status_code == 503


def test_live_predict_auto_detects_corners_then_pieces():
    from fastapi.testclient import TestClient

    stub = _StubPredictor([_center("e2")], labels=[1], scores=[0.9])
    client = TestClient(_app(predictor=stub, corner_predictor=_StubCornerPredictor()))
    assert client.get("/api/live/available").json() == {"available": True}

    # No corners in the request: the server detects them on the uploaded frame.
    resp = client.post(
        "/api/live/predict", files={"image": ("f.jpg", _jpeg(), "image/jpeg")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_detected"] == 1
    assert set(body["orientations"]) == {"R0", "R90", "R180", "R270"}
    assert body["orientations"]["R0"]["board_fen"] == board_fen_from_squares({"e2": "P"})
    assert "<svg" in body["orientations"]["R0"]["board_svg"]
    assert len(body["lattice"]) == 81
    assert body["corners"]["top_left"] == [0.0, 0.0]
    assert body["pieces"][0]["symbol"] == "P" and body["pieces"][0]["color"] == "w"
