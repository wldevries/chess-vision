"""Tests for the capture app's surviving endpoints: Set/Board metadata + corner-assist.

Capture mode (play a game/puzzle and snap photos) and its Label Studio pipeline were
retired; labelling now happens over the unified store via the corner/position tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chessvision.capture.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


# --- Set/Board metadata (the dropdowns) ------------------------------------- #


def test_add_board_appends_to_boards_json(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/api/meta").json()["boards"] == []
    resp = client.post("/api/meta/board", json={"id": "tournament-50mm", "square_mm": 50})
    assert resp.status_code == 200
    assert "tournament-50mm" in resp.json()["boards"]
    # Persisted with the square_mm so box synthesis stays correct on the new board.
    saved = json.loads((tmp_path / "boards.json").read_text())
    assert saved["tournament-50mm"] == {"square_mm": 50}
    # Shows up on the next fetch too.
    assert "tournament-50mm" in client.get("/api/meta").json()["boards"]


def test_add_set_appends_and_rejects_duplicates(client: TestClient) -> None:
    assert client.post("/api/meta/set", json={"id": "tournament-plastic"}).status_code == 200
    assert "tournament-plastic" in client.get("/api/meta").json()["sets"]
    dup = client.post("/api/meta/set", json={"id": "tournament-plastic"})
    assert dup.status_code == 409


def test_add_meta_rejects_bad_ids(client: TestClient) -> None:
    assert client.post("/api/meta/board", json={"id": ""}).status_code == 400
    assert client.post("/api/meta/board", json={"id": "_comment"}).status_code == 400


# --- corner-assist endpoints ------------------------------------------------ #


class _FakeCornerPredictor:
    """Stand-in for CornerPredictor: returns a fixed corner dict, no model/torch."""

    CORNERS = {
        "top_left": [1.0, 2.0],
        "top_right": [3.0, 4.0],
        "bottom_right": [5.0, 6.0],
        "bottom_left": [7.0, 8.0],
    }

    def predict(self, rgb):
        return dict(self.CORNERS)


def _jpeg_bytes() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_corner_assist_off_by_default(client: TestClient) -> None:
    assert client.get("/api/corners/available").json() == {"available": False}
    resp = client.post(
        "/api/corners/predict", files={"image": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 503  # not wired without --corner-ckpt


def test_corner_assist_predicts_when_injected(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, corner_predictor=_FakeCornerPredictor()))

    assert client.get("/api/corners/available").json() == {"available": True}

    resp = client.post(
        "/api/corners/predict", files={"image": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 200
    assert resp.json()["corners"] == _FakeCornerPredictor.CORNERS

    # empty upload is a 400, not a 500
    empty = client.post("/api/corners/predict", files={"image": ("f.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
