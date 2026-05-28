"""Tests for the capture tool: PGN -> ply model, and the FastAPI endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from chessvision.capture.app import create_app
from chessvision.capture.games import (
    Game,
    Ply,
    game_from_lichess_puzzle,
    load_pgn_file,
    load_pgn_text,
)

# A synthetic Lichess puzzle payload (no network): a Ruy-Lopez-ish position
# reached after 4 game half-moves, with a 2-move solution.
PUZZLE_PAYLOAD = {
    "game": {"pgn": "e4 e5 Nf3 Nc6"},
    "puzzle": {
        "id": "abc12",
        "initialPly": 3,  # replay moves 0..3 inclusive -> position after 2...Nc6
        "solution": ["f1b5", "a7a6"],
        "themes": ["middlegame", "short"],
        "rating": 1500,
    },
}

SAMPLE_PGN = Path("chessvision/capture/samples/opera_game.pgn")

TWO_GAME_PGN = """\
[White "A"]
[Black "B"]

1. e4 e5 2. Nf3 1-0

[White "C"]
[Black "D"]

1. d4 d5 *
"""


def test_sample_loads_as_one_game_with_startpos_ply_zero() -> None:
    games = load_pgn_file(SAMPLE_PGN)
    assert len(games) == 1
    game = games[0]
    assert game.white == "Paul Morphy"
    # 17 white + 16 black half-moves, plus ply 0 (the starting position).
    assert game.n_plies == 34
    start = game.plies[0]
    assert start.is_start and start.san is None
    assert start.fen == chess.STARTING_FEN
    assert start.move_label == "start"


def test_ply_fen_and_move_labels_track_the_game() -> None:
    game = load_pgn_file(SAMPLE_PGN)[0]
    # 1. e4 -> black to move. (python-chess omits the ep square when no legal
    # en-passant capture exists, so the ep field is "-" here.)
    assert game.plies[1].san == "e4"
    assert game.plies[1].turn == "b"
    assert game.plies[1].move_label == "1. e4"
    assert game.plies[1].fen.split()[:3] == [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",
        "b",
        "KQkq",
    ]
    # 1... e5 is a black move -> "1... e5".
    assert game.plies[2].move_label == "1... e5"
    # The game ends in mate by white: 17. Rd8#, black to move in a checkmated pos.
    last = game.plies[-1]
    assert last.san == "Rd8#"
    assert last.move_label == "17. Rd8#"
    assert last.turn == "b"
    assert chess.Board(last.fen).is_checkmate()


def test_highlight_squares_are_named() -> None:
    ply = load_pgn_file(SAMPLE_PGN)[0].plies[1]  # 1. e4
    assert (ply.from_square, ply.to_square) == ("e2", "e4")


def test_load_text_skips_moveless_entries_and_assigns_unique_ids() -> None:
    games = load_pgn_text(TWO_GAME_PGN, source="t")
    assert len(games) == 2
    assert len({g.game_id for g in games}) == 2


def test_moveless_pgn_yields_no_games() -> None:
    assert load_pgn_text('[White "x"]\n[Black "y"]\n\n*\n') == []


def test_puzzle_payload_becomes_a_game_at_the_solver_position() -> None:
    game = game_from_lichess_puzzle(PUZZLE_PAYLOAD)
    assert game.game_id == "puzzle-abc12"
    assert game.white == "Lichess puzzle" and game.black == "#abc12"
    assert "rating 1500" in game.event and "middlegame" in game.event
    # Ply 0 is the puzzle position (after 2...Nc6): White to move, full board.
    start = game.plies[0]
    assert start.is_start and start.turn == "w"
    assert start.fen.startswith("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R")
    # The solution moves become the subsequent plies.
    assert game.n_plies == 3
    assert (game.plies[1].san, game.plies[1].uci) == ("Bb5", "f1b5")
    assert game.plies[2].san == "a6"


def test_puzzle_without_themes_still_builds() -> None:
    payload = {"game": {"pgn": "e4 e5"}, "puzzle": {"id": "z", "initialPly": 1, "solution": []}}
    game = game_from_lichess_puzzle(payload)
    assert game.n_plies == 1  # just the position, no solution moves
    assert game.event == "rating ?"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    games = load_pgn_file(SAMPLE_PGN)
    return TestClient(create_app(games, tmp_path))


def test_finish_requires_a_capture(client: TestClient) -> None:
    game_id = client.get("/api/games").json()[0]["game_id"]
    sid = client.post("/api/session", data={"game_id": game_id}).json()["session_id"]
    resp = client.post(f"/api/session/{sid}/finish")
    assert resp.status_code == 400


def test_finish_publishes_the_session(client: TestClient, monkeypatch) -> None:
    # Storage config present (no real network — publish_session is stubbed).
    monkeypatch.setenv("MINIO_ENDPOINT_URL", "http://fake:9000")
    for var in ("MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET"):
        monkeypatch.setenv(var, "x")
    calls: list = []
    monkeypatch.setattr(
        "chessvision.data.publish.publish_session",
        lambda *a, **k: calls.append((a, k)) or {"uploaded": 2, "skipped": 0, "tasks": 1},
    )

    game_id = client.get("/api/games").json()[0]["game_id"]
    sid = client.post("/api/session", data={"game_id": game_id}).json()["session_id"]
    client.post(
        f"/api/session/{sid}/snap", files={"image": ("f.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")}
    )

    resp = client.post(f"/api/session/{sid}/finish")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": sid, "uploaded": 2, "skipped": 0, "tasks": 1}
    assert len(calls) == 1  # publish_session was invoked once


def test_api_lists_games_and_runs_a_snap_cycle(client: TestClient, tmp_path: Path) -> None:
    listed = client.get("/api/games").json()
    assert len(listed) == 1
    game_id = listed[0]["game_id"]

    state = client.post("/api/session", data={"game_id": game_id}).json()
    sid = state["session_id"]
    assert state["ply_index"] == 0
    assert state["ply"]["is_start"]
    assert "<svg" in state["board_svg"]

    # Snap the starting position; the server advances to ply 1 and records a row.
    resp = client.post(
        f"/api/session/{sid}/snap", files={"image": ("f.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")}
    )
    state = resp.json()
    assert state["ply_index"] == 1
    assert len(state["captures"]) == 1
    cap = state["captures"][0]
    assert cap["ply_index"] == 0
    assert cap["fen"] == chess.STARTING_FEN

    # The photo and a JSONL row are on disk.
    session_dir = tmp_path / sid
    assert (session_dir / cap["filename"]).read_bytes() == b"\xff\xd8\xff\xd9"
    assert (session_dir / "captures.jsonl").read_text(encoding="utf-8").strip()


def _new_session(client: TestClient) -> str:
    game_id = client.get("/api/games").json()[0]["game_id"]
    return client.post("/api/session", data={"game_id": game_id}).json()["session_id"]


def test_api_goto_clamps_and_view_flips(client: TestClient) -> None:
    sid = _new_session(client)

    # goto past the end clamps to the last ply.
    state = client.post(f"/api/session/{sid}/goto", data={"ply_index": 9999}).json()
    assert state["ply_index"] == state["n_plies"] - 1

    state = client.post(f"/api/session/{sid}/view", data={"view": "black"}).json()
    assert state["view"] == "black"
    assert client.post(f"/api/session/{sid}/view", data={"view": "sideways"}).status_code == 400


def test_api_orientation_takes_rotation_values(client: TestClient) -> None:
    sid = _new_session(client)
    state = client.post(f"/api/session/{sid}/orientation", data={"orientation": "R90"}).json()
    assert state["orientation"] == "R90"
    assert (
        client.post(f"/api/session/{sid}/orientation", data={"orientation": "x"}).status_code == 400
    )


# A clean 800x800 board square: canonical (u, v) -> (100 + u*800, 100 + v*800) under R0.
SQUARE_CORNERS = {
    "top_left": [100, 100],
    "top_right": [900, 100],
    "bottom_right": [900, 900],
    "bottom_left": [100, 900],
}


def test_corners_produce_overlay_with_piece_base_points(client: TestClient) -> None:
    sid = _new_session(client)
    assert client.get(f"/api/session/{sid}").json()["overlay"] is None

    state = client.post(
        f"/api/session/{sid}/corners", json={**SQUARE_CORNERS, "orientation": "R0"}
    ).json()
    assert state["corners"]["top_left"] == [100, 100]
    overlay = state["overlay"]
    assert len(overlay["lattice"]) == 81
    assert len(overlay["pieces"]) == 32  # full starting position

    a8 = next(p for p in overlay["pieces"] if p["square"] == "a8")
    assert a8["piece"] == "r" and a8["color"] == "b"
    # a8 center -> (100 + 0.0625*800,) twice = (150, 150).
    assert a8["base"] == pytest.approx([150.0, 150.0], abs=0.5)

    cleared = client.request("DELETE", f"/api/session/{sid}/corners").json()
    assert cleared["corners"] is None and cleared["overlay"] is None


def test_snap_records_corners_and_guesstimated_pieces(client: TestClient, tmp_path: Path) -> None:
    sid = _new_session(client)
    client.post(f"/api/session/{sid}/corners", json={**SQUARE_CORNERS, "orientation": "R0"})

    state = client.post(
        f"/api/session/{sid}/snap", files={"image": ("f.jpg", b"x", "image/jpeg")}
    ).json()
    cap = state["captures"][0]
    assert cap["corners"]["top_left"] == [100, 100]
    assert cap["orientation"] == "R0"
    assert len(cap["pieces"]) == 32
    # The JSONL row carries the same weak labels for offline use.
    row = json.loads(
        (tmp_path / sid / "captures.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["fen"] == cap["fen"] and len(row["pieces"]) == 32


def test_api_delete_capture_removes_file_and_row(client: TestClient, tmp_path: Path) -> None:
    game_id = client.get("/api/games").json()[0]["game_id"]
    sid = client.post("/api/session", data={"game_id": game_id}).json()["session_id"]
    state = client.post(
        f"/api/session/{sid}/snap", files={"image": ("f.jpg", b"data", "image/jpeg")}
    ).json()
    fn = state["captures"][0]["filename"]
    assert (tmp_path / sid / fn).exists()

    state = client.request("DELETE", f"/api/session/{sid}/capture/{fn}").json()
    assert state["captures"] == []
    assert not (tmp_path / sid / fn).exists()


def test_unknown_session_is_404(client: TestClient) -> None:
    assert client.get("/api/session/nope").status_code == 404


def test_dataclasses_are_constructible() -> None:
    # Cheap guard that the public model stays importable/typed as expected.
    ply = Ply(0, chess.STARTING_FEN, 0, "w", None, None, None, None)
    game = Game("g", "W", "B", "E", "D", "*", chess.STARTING_FEN, [ply])
    assert game.label.startswith("W - B")
    assert game.n_plies == 1


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
    assert client.get("/api/corners/available").json() == {"available": False, "heatmap": False}
    resp = client.post(
        "/api/corners/predict", files={"image": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 503  # not wired without --corner-ckpt


def test_corner_assist_predicts_when_injected(tmp_path: Path) -> None:
    games = load_pgn_file(SAMPLE_PGN)
    client = TestClient(create_app(games, tmp_path, corner_predictor=_FakeCornerPredictor()))

    assert client.get("/api/corners/available").json() == {"available": True, "heatmap": False}

    resp = client.post(
        "/api/corners/predict", files={"image": ("f.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 200
    assert resp.json()["corners"] == _FakeCornerPredictor.CORNERS

    # empty upload is a 400, not a 500
    empty = client.post("/api/corners/predict", files={"image": ("f.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
