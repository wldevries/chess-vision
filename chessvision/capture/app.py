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
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chessvision.capture.games import Game, Ply

STATIC_DIR = Path(__file__).parent / "static"


def render_board_svg(fen: str, lastmove_uci: str | None, orientation: str, size: int = 480) -> str:
    board = chess.Board(fen)
    lastmove = chess.Move.from_uci(lastmove_uci) if lastmove_uci else None
    orient = chess.WHITE if orientation == "white" else chess.BLACK
    return chess.svg.board(
        board, lastmove=lastmove, orientation=orient, size=size, coordinates=True
    )


@dataclass
class Session:
    session_id: str
    game: Game
    out_dir: Path
    ply_index: int = 0
    orientation: str = "white"  # which side is at the bottom of the displayed board
    captures: list[dict] = field(default_factory=list)

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / "captures.jsonl"

    def clamp(self, index: int) -> int:
        return max(0, min(index, self.game.n_plies - 1))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def create_app(games: list[Game], out_root: Path) -> FastAPI:
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

    def instruction(ply: Ply) -> str:
        if ply.is_start:
            return "Set up the starting position, then Snap."
        mover = "White" if ply.mover_is_white else "Black"
        return f"{mover} plays {ply.move_label} — set the board to match, then Snap."

    def state_payload(session: Session) -> dict:
        ply = session.game.plies[session.ply_index]
        return {
            "session_id": session.session_id,
            "game": {
                "game_id": session.game.game_id,
                "label": session.game.label,
                "white": session.game.white,
                "black": session.game.black,
            },
            "ply_index": session.ply_index,
            "n_plies": session.game.n_plies,
            "orientation": session.orientation,
            "ply": ply_payload(ply),
            "instruction": instruction(ply),
            "board_svg": render_board_svg(ply.fen, ply.uci, session.orientation),
            "captures": session.captures,
            "captured_plies": sorted({c["ply_index"] for c in session.captures}),
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

    @app.post("/api/session/{session_id}/orientation")
    def set_orientation(session_id: str, orientation: str = Form(...)) -> dict:
        session = get_session(session_id)
        if orientation not in ("white", "black"):
            raise HTTPException(400, "orientation must be 'white' or 'black'")
        session.orientation = orientation
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
            "board_orientation": session.orientation,
            "bytes": len(data),
        }
        with session.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        session.captures.append(record)

        if advance:
            session.ply_index = session.clamp(session.ply_index + 1)
        return state_payload(session)

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
