"""Tests for the capture tool: PGN -> ply model, and the FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest
from fastapi.testclient import TestClient

from chessvision.capture.app import create_app
from chessvision.capture.games import (
    Game,
    Ply,
    load_pgn_file,
    load_pgn_text,
)

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


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    games = load_pgn_file(SAMPLE_PGN)
    return TestClient(create_app(games, tmp_path))


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


def test_api_goto_clamps_and_flip_toggles_orientation(client: TestClient) -> None:
    game_id = client.get("/api/games").json()[0]["game_id"]
    sid = client.post("/api/session", data={"game_id": game_id}).json()["session_id"]

    # goto past the end clamps to the last ply.
    state = client.post(f"/api/session/{sid}/goto", data={"ply_index": 9999}).json()
    assert state["ply_index"] == state["n_plies"] - 1

    state = client.post(f"/api/session/{sid}/orientation", data={"orientation": "black"}).json()
    assert state["orientation"] == "black"
    bad = client.post(f"/api/session/{sid}/orientation", data={"orientation": "sideways"})
    assert bad.status_code == 400


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
