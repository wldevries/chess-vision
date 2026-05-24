"""FastAPI app for the position-capture workflow.

Holds the loaded games and in-memory capture sessions, renders each position as
an SVG board (via `chess.svg`), and on each "snap" writes the uploaded photo plus
a JSONL metadata row whose `fen` field is the ground-truth label. Single local
user, so session state lives in memory; photos and metadata are the durable
output, organised per session under `out_root/<session_id>/`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.svg
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chessvision.capture.games import Game, Ply, fetch_lichess_puzzle_next
from chessvision.geometry import (
    Orientation,
    canonical_to_image,
    compute_homography,
    lattice_points,
    square_center_uv,
    square_polygons,
)

STATIC_DIR = Path(__file__).parent / "static"

CornerDict = dict[str, list[float]]


class CornersIn(BaseModel):
    """The four board corners as image pixels (native camera resolution),
    in the visual order top-left, top-right, bottom-right, bottom-left."""

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]
    orientation: str | None = None  # R0/R90/R180/R270; defaults to the session's

    def as_dict(self) -> CornerDict:
        return {
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "bottom_left": list(self.bottom_left),
        }


def render_board_svg(fen: str, lastmove_uci: str | None, view: str, size: int = 480) -> str:
    board = chess.Board(fen)
    lastmove = chess.Move.from_uci(lastmove_uci) if lastmove_uci else None
    orient = chess.WHITE if view == "white" else chess.BLACK
    return chess.svg.board(
        board, lastmove=lastmove, orientation=orient, size=size, coordinates=True
    )


def _round_pts(pts: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def compute_overlay(
    corners: CornerDict,
    orientation: Orientation,
    fen: str,
    from_square: str | None = None,
    to_square: str | None = None,
) -> dict:
    """Project the board grid and the FEN's occupied squares into the image.

    With fixed corners, every square's image location is known, so the known FEN
    gives a guesstimated base point (square center) and footprint quad for each
    piece -- a weak label and a live alignment check, no detector needed. If a
    move is supplied, its from/to square quads come back too so the client can
    highlight them on the live feed (mirroring the SVG board's lastmove).
    """
    homography = compute_homography(corners, orientation)
    polys = square_polygons(homography)
    board = chess.Board(fen)

    pieces: list[dict] = []
    for square_index, piece in board.piece_map().items():
        name = chess.square_name(square_index)
        u, v = square_center_uv(name)
        base = canonical_to_image(homography, np.array([[u, v]], dtype=np.float32))[0]
        pieces.append(
            {
                "square": name,
                "piece": piece.symbol(),  # 'P'/'n'/... (case = colour)
                "color": "w" if piece.color == chess.WHITE else "b",
                "base": [round(float(base[0]), 1), round(float(base[1]), 1)],
                "quad": _round_pts(polys[name]),
            }
        )

    move: dict | None = None
    if from_square in polys and to_square in polys:
        move = {
            "from": {"square": from_square, "quad": _round_pts(polys[from_square])},
            "to": {"square": to_square, "quad": _round_pts(polys[to_square])},
        }
    return {
        "lattice": _round_pts(lattice_points(homography)),
        "pieces": pieces,
        "move": move,
    }


@dataclass
class Session:
    session_id: str
    game: Game
    out_dir: Path
    ply_index: int = 0
    view: str = "white"  # cosmetic: which side is at the bottom of the SVG board
    corners: CornerDict | None = None  # image-pixel board corners, fixed for the session
    orientation: Orientation = Orientation.R0  # which canonical anchor maps to which corner
    captures: list[dict] = field(default_factory=list)

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / "captures.jsonl"

    def clamp(self, index: int) -> int:
        return max(0, min(index, self.game.n_plies - 1))

    def overlay(self) -> dict | None:
        if self.corners is None:
            return None
        ply = self.game.plies[self.ply_index]
        return compute_overlay(
            self.corners, self.orientation, ply.fen, ply.from_square, ply.to_square
        )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def create_app(games: list[Game], out_root: Path, *, lichess_token: str | None = None) -> FastAPI:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    games_by_id = {g.game_id: g for g in games}
    sessions: dict[str, Session] = {}

    app = FastAPI(title="chessvision capture")
    app.mount("/captures", StaticFiles(directory=str(out_root)), name="captures")

    def get_session(session_id: str) -> Session:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, f"unknown session: {session_id}")
        return session

    def ply_payload(ply: Ply) -> dict:
        return {
            "index": ply.index,
            "fen": ply.fen,
            "move_number": ply.move_number,
            "turn": ply.turn,
            "san": ply.san,
            "uci": ply.uci,
            "from": ply.from_square,
            "to": ply.to_square,
            "is_start": ply.is_start,
            "move_label": ply.move_label,
        }

    def result_label(result: str) -> str | None:
        """Human-readable outcome for the PGN Result tag, or None if unfinished."""
        return {"1-0": "White wins", "0-1": "Black wins", "1/2-1/2": "Draw"}.get(result)

    def instruction(ply: Ply, is_final: bool) -> str:
        if ply.is_start:
            return "Set up the starting position, then Snap."
        mover = "White" if ply.mover_is_white else "Black"
        prefix = "Final move · " if is_final else ""
        return f"{prefix}{mover} plays {ply.move_label} — set the board to match, then Snap."

    def state_payload(session: Session) -> dict:
        ply = session.game.plies[session.ply_index]
        captured_plies = sorted({c["ply_index"] for c in session.captures})
        is_final_ply = session.ply_index == session.game.n_plies - 1
        game_complete = is_final_ply and session.ply_index in captured_plies
        return {
            "session_id": session.session_id,
            "game": {
                "game_id": session.game.game_id,
                "label": session.game.label,
                "white": session.game.white,
                "black": session.game.black,
                "result": session.game.result,
                "result_label": result_label(session.game.result),
            },
            "ply_index": session.ply_index,
            "n_plies": session.game.n_plies,
            "view": session.view,
            "corners": session.corners,
            "orientation": session.orientation.name,
            "overlay": session.overlay(),
            "ply": ply_payload(ply),
            "instruction": instruction(ply, is_final_ply),
            "is_final_ply": is_final_ply,
            "game_complete": game_complete,
            "board_svg": render_board_svg(ply.fen, ply.uci, session.view),
            "captures": session.captures,
            "captured_plies": captured_plies,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/games")
    def list_games() -> list[dict]:
        return [
            {
                "game_id": g.game_id,
                "label": g.label,
                "white": g.white,
                "black": g.black,
                "n_plies": g.n_plies,
            }
            for g in games
        ]

    @app.post("/api/puzzles/next")
    def next_puzzle(
        theme: str | None = Form(None),
        difficulty: str | None = Form(None),
        min_pieces: int | None = Form(None),
    ) -> dict:
        """Fetch a fresh Lichess puzzle on the fly, register it, and return its
        summary so the client can immediately start a session on it."""
        try:
            game = fetch_lichess_puzzle_next(
                theme=theme or None,
                difficulty=difficulty or None,
                min_pieces=min_pieces,
                token=lichess_token,
            )
        except Exception as exc:  # network / API error -> surface as 502
            raise HTTPException(502, f"could not fetch puzzle: {exc}") from exc
        # Ids are unique per puzzle; re-fetching the same one just refreshes it.
        games_by_id[game.game_id] = game
        if all(g.game_id != game.game_id for g in games):
            games.append(game)
        return {
            "game_id": game.game_id,
            "label": game.label,
            "white": game.white,
            "black": game.black,
            "n_plies": game.n_plies,
        }

    @app.post("/api/session")
    def start_session(game_id: str = Form(...)) -> dict:
        game = games_by_id.get(game_id)
        if game is None:
            raise HTTPException(404, f"unknown game: {game_id}")
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if session_id in sessions:  # sub-second collision
            session_id = f"{session_id}-{len(sessions)}"
        out_dir = out_root / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "session.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "started_at": _now_iso(),
                    "game_id": game.game_id,
                    "white": game.white,
                    "black": game.black,
                    "event": game.event,
                    "date": game.date,
                    "result": game.result,
                    "start_fen": game.start_fen,
                    "n_plies": game.n_plies,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        session = Session(session_id=session_id, game=game, out_dir=out_dir)
        sessions[session_id] = session
        return state_payload(session)

    @app.get("/api/session/{session_id}")
    def get_state(session_id: str) -> dict:
        return state_payload(get_session(session_id))

    @app.post("/api/session/{session_id}/goto")
    def goto(session_id: str, ply_index: int = Form(...)) -> dict:
        session = get_session(session_id)
        session.ply_index = session.clamp(ply_index)
        return state_payload(session)

    @app.post("/api/session/{session_id}/view")
    def set_view(session_id: str, view: str = Form(...)) -> dict:
        session = get_session(session_id)
        if view not in ("white", "black"):
            raise HTTPException(400, "view must be 'white' or 'black'")
        session.view = view
        return state_payload(session)

    @app.post("/api/session/{session_id}/orientation")
    def set_orientation(session_id: str, orientation: str = Form(...)) -> dict:
        session = get_session(session_id)
        try:
            session.orientation = Orientation[orientation]
        except KeyError as exc:
            raise HTTPException(400, "orientation must be one of R0/R90/R180/R270") from exc
        return state_payload(session)

    @app.post("/api/session/{session_id}/corners")
    def set_corners(session_id: str, body: CornersIn) -> dict:
        session = get_session(session_id)
        if body.orientation is not None:
            try:
                session.orientation = Orientation[body.orientation]
            except KeyError as exc:
                raise HTTPException(400, "orientation must be one of R0/R90/R180/R270") from exc
        session.corners = body.as_dict()
        return state_payload(session)

    @app.delete("/api/session/{session_id}/corners")
    def clear_corners(session_id: str) -> dict:
        session = get_session(session_id)
        session.corners = None
        return state_payload(session)

    @app.post("/api/session/{session_id}/snap")
    async def snap(session_id: str, image: UploadFile, advance: bool = Form(True)) -> dict:
        session = get_session(session_id)
        ply = session.game.plies[session.ply_index]
        data = await image.read()
        if not data:
            raise HTTPException(400, "empty image upload")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{session.game.game_id}_ply{ply.index:03d}_{stamp}.jpg"
        (session.out_dir / filename).write_bytes(data)

        overlay = session.overlay()
        record = {
            "filename": filename,
            "url": f"/captures/{session.session_id}/{filename}",
            "session_id": session.session_id,
            "captured_at": _now_iso(),
            "game_id": session.game.game_id,
            "white": session.game.white,
            "black": session.game.black,
            "event": session.game.event,
            "date": session.game.date,
            "ply_index": ply.index,
            "move_number": ply.move_number,
            "turn": ply.turn,
            "san": ply.san,
            "uci": ply.uci,
            "from": ply.from_square,
            "to": ply.to_square,
            "fen": ply.fen,
            "move_label": ply.move_label,
            "view": session.view,
            # Geometry weak-labels: fixed corners + the FEN's piece base points.
            "corners": session.corners,
            "orientation": session.orientation.name,
            "pieces": overlay["pieces"] if overlay else None,
            "bytes": len(data),
        }
        with session.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        session.captures.append(record)

        if advance:
            session.ply_index = session.clamp(session.ply_index + 1)
        return state_payload(session)

    @app.post("/api/session/{session_id}/finish")
    def finish(session_id: str) -> dict:
        """Push this session's photos + metadata to the bucket and generate its
        Label Studio point tasks. Called on game/puzzle completion (and on demand).
        Size-based upload + overwriting tasks make it safe to call more than once."""
        session = get_session(session_id)
        if not session.captures:
            raise HTTPException(400, "nothing captured in this session yet")
        try:
            from chessvision.data.publish import publish_session
            from chessvision.data.storage import StorageConfig, get_client
        except Exception as exc:  # pragma: no cover - import guard
            raise HTTPException(503, f"storage support unavailable: {exc}") from exc
        try:
            config = StorageConfig.from_env()
            client = get_client(config)
            result = publish_session(
                session.out_dir, session.session_id, session.captures, config=config, client=client
            )
        except RuntimeError as exc:  # missing/incomplete .env config
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:  # network / S3 failure
            raise HTTPException(502, f"publish failed: {exc}") from exc
        return {"session_id": session_id, **result}

    @app.delete("/api/session/{session_id}/capture/{filename}")
    def delete_capture(session_id: str, filename: str) -> dict:
        session = get_session(session_id)
        match = next((c for c in session.captures if c["filename"] == filename), None)
        if match is None:
            raise HTTPException(404, f"no such capture: {filename}")
        (session.out_dir / filename).unlink(missing_ok=True)
        session.captures = [c for c in session.captures if c["filename"] != filename]
        # Rewrite the JSONL so it stays consistent with the kept photos.
        with session.jsonl_path.open("w", encoding="utf-8") as fh:
            for c in session.captures:
                fh.write(json.dumps(c) + "\n")
        return state_payload(session)

    @app.exception_handler(HTTPException)
    async def _http_error(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    return app
