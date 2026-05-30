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
import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chessvision.capture.games import Game, Ply, fetch_lichess_puzzle_next
from chessvision.geometry import (
    Orientation,
    canonical_to_image,
    compute_homography,
    lattice_points,
    order_corners,
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


class CornerLabelIn(BaseModel):
    """A saved corner label for an inbox photo (corner-label mode). `corners` are four
    [x, y] points in the *normalized* (EXIF-applied) image frame, any order — sorted
    server-side. `src` is the inbox-relative path of the photo being labelled."""

    src: str
    corners: list[tuple[float, float]]
    board: str | None = None
    device: str | None = None
    surface: str | None = None


class PiecePointIn(BaseModel):
    """One placed piece: a verbose keypoint label (e.g. "WhiteRook") and its contact
    point [x, y] in the normalized image frame."""

    label: str
    x: float
    y: float


class PositionProjectIn(BaseModel):
    """Project a known FEN onto a photo's corners (no save). `corners` are four [x, y]
    points in the normalized frame, any order; `orientation` is R0..R270 (which physical
    corner is a8 — the manual choice); `fen` is a placement field or full FEN."""

    corners: list[tuple[float, float]]
    fen: str
    orientation: str = "R0"


class PositionSaveIn(BaseModel):
    """Save a position label: the photo `src`, its corners, the known FEN + orientation,
    and the nudged piece contact keypoints. Written into the corner store as a piece-
    keypoint training sample (no Label Studio)."""

    src: str
    corners: list[tuple[float, float]]
    fen: str
    orientation: str = "R0"
    pieces: list[PiecePointIn]
    board: str | None = None
    piece_set: str | None = Field(None, alias="set")  # sets.json key for these pieces

    model_config = {"populate_by_name": True}


class PositionLibraryIn(BaseModel):
    """Add/replace a named position in the reusable FEN library."""

    name: str
    fen: str


class SessionMetaIn(BaseModel):
    """Editable per-session domain tags (metadata editor). `set` is a Python builtin,
    so it rides in under the `piece_set` field with a JSON alias. Omitted fields are
    left unchanged; an empty string clears a tag."""

    piece_set: str | None = Field(None, alias="set")
    board: str | None = None
    device: str | None = None
    surface: str | None = None

    model_config = {"populate_by_name": True}


class NewBoardIn(BaseModel):
    """Register a new board in boards.json (corner/capture Board dropdowns). `id` is
    the board key, named by square size by convention (e.g. "tournament-50mm").
    `square_mm` is the edge length of one playing square — the one number that keeps
    box synthesis correct across boards; null/omitted means unmeasured (falls back)."""

    id: str
    square_mm: float | None = None
    note: str | None = None


class NewSetIn(BaseModel):
    """Register a new piece set in sets.json (capture Set dropdown). `id` is the set
    key (e.g. "tournament-plastic"); per-piece measurements are left unset (box
    synthesis falls back to PIECE_HEIGHT_SCALE) — fill them into sets.json later."""

    id: str
    note: str | None = None


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
    started_at: str = ""  # set at start; written into session.json on the first snap
    ply_index: int = 0
    view: str = "white"  # cosmetic: which side is at the bottom of the SVG board
    corners: CornerDict | None = None  # image-pixel board corners, fixed for the session
    orientation: Orientation = Orientation.R0  # which canonical anchor maps to which corner
    # Domain axes, chosen in the UI at session start and written into session.json so
    # SessionMetadata reads them from the session itself (no hand-maintained central file).
    piece_set: str | None = None  # sets.json key (the physical piece set)
    board: str | None = None  # boards.json key (board geometry)
    device: str | None = None  # camera label, e.g. "HD Pro Webcam C920"
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


def _meta_options(out_root: Path) -> dict[str, list[str]]:
    """Available piece-set and board ids for the capture UI's Set/Board dropdowns,
    read from the central `sets.json`/`boards.json` reference files (shared physical
    measurements). Comment keys (leading `_`) are skipped; a missing file -> empty list."""

    def keys(name: str) -> list[str]:
        path = out_root / name
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, AttributeError):
            return []
        return sorted(k for k in data if not k.startswith("_"))

    return {"sets": keys("sets.json"), "boards": keys("boards.json")}


def _write_session_meta(session: Session) -> None:
    """Create the session's output dir and write session.json (idempotent).

    Deferred to the first snap, not done at session start, so a session that is opened
    but never photographed leaves nothing on disk (and so nothing to sync to the bucket).
    `set`/`board`/`device` mirror the sessions.json schema so SessionMetadata reads the
    domain axes straight off this per-session file (the central file is the fallback)."""
    session.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = session.out_dir / "session.json"
    if meta_path.exists():
        return
    g = session.game
    meta_path.write_text(
        json.dumps(
            {
                "session_id": session.session_id,
                "started_at": session.started_at,
                "game_id": g.game_id,
                "white": g.white,
                "black": g.black,
                "event": g.event,
                "date": g.date,
                "result": g.result,
                "start_fen": g.start_fen,
                "n_plies": g.n_plies,
                "set": session.piece_set or "",
                "board": session.board or "",
                "device": session.device or "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def create_app(
    games: list[Game],
    out_root: Path,
    *,
    lichess_token: str | None = None,
    keypoint_ckpt: str | Path | None = None,
    corner_ckpt: str | Path | None = None,
    device: str | None = None,
    predictor=None,
    corner_predictor=None,
    corners_root: str | Path | None = None,
    corner_store=None,
) -> FastAPI:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    games_by_id = {g.game_id: g for g in games}
    sessions: dict[str, Session] = {}

    # sets.json/boards.json (the Set/Board dropdowns + mm reference) live at the unified
    # store root in the flat layout; fall back to out_root only when no store is wired.
    meta_root = Path(corners_root) if corners_root is not None else out_root

    # Corner-label mode: an import-and-label flow over phone photos staged in a
    # `data/corners/inbox/` tree, writing a standalone corner dataset (no FEN / Label
    # Studio). Off unless a corners root is provided; a pre-built store can be injected
    # for tests. See chessvision/data/corner_capture.py and corner-capture-mode.md.
    if corner_store is None and corners_root is not None:
        from chessvision.data.corner_capture import CornerStore

        corner_store = CornerStore(corners_root)

    # Read-position (live FEN) mode. The predictor is constructed cheaply -- the
    # model + torch only load on the first /api/live/predict call -- so the app
    # starts instantly even with no checkpoint requested. A pre-built `predictor`
    # can be injected (tests); otherwise it's built from `keypoint_ckpt`.
    if predictor is None and keypoint_ckpt is not None:
        from chessvision.inference import LivePredictor

        predictor = LivePredictor(keypoint_ckpt, device=device)

    # Corner-assist: pre-fill the corner-marking UI from the trained corner regressor.
    # Same lazy contract -- the model loads on the first /api/corners/predict call.
    if corner_predictor is None and corner_ckpt is not None:
        from chessvision.inference import CornerPredictor

        corner_predictor = CornerPredictor(corner_ckpt, device=device)

    app = FastAPI(title="chessvision capture")
    app.mount("/captures", StaticFiles(directory=str(out_root)), name="captures")
    # Shared static assets (cv.js etc.) for the page templates served below. The HTML
    # pages themselves are returned by explicit routes (/, /capture, /live, ...) so the
    # mount is for sidecar assets only, not the entry pages.
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
        """Launcher hub: cards linking to each capture/labelling tool."""
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/capture", response_class=HTMLResponse)
    def capture_page() -> str:
        """Capture mode: label a known game/puzzle — mark corners, snap each ply."""
        return (STATIC_DIR / "capture.html").read_text(encoding="utf-8")

    @app.get("/corners", response_class=HTMLResponse)
    def corners_page() -> str:
        """Corner-label mode: mark board corners on staged phone photos."""
        return (STATIC_DIR / "corners.html").read_text(encoding="utf-8")

    @app.get("/positions", response_class=HTMLResponse)
    def positions_page() -> str:
        """Position-label mode: project a known FEN onto a corner photo and nudge bases."""
        return (STATIC_DIR / "positions.html").read_text(encoding="utf-8")

    @app.get("/sessions", response_class=HTMLResponse)
    def sessions_page() -> str:
        """Session-metadata editor: tag past sessions with set/board/camera/surface."""
        return (STATIC_DIR / "sessions.html").read_text(encoding="utf-8")

    @app.get("/live", response_class=HTMLResponse)
    def live_page() -> str:
        """Standalone Read-position view: camera -> Read (auto-detects corners then
        pieces on the same frame) -> predicted FEN."""
        return (STATIC_DIR / "live.html").read_text(encoding="utf-8")

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
    def start_session(
        game_id: str = Form(...),
        piece_set: str | None = Form(None),
        board: str | None = Form(None),
        device: str | None = Form(None),
    ) -> dict:
        game = games_by_id.get(game_id)
        if game is None:
            raise HTTPException(404, f"unknown game: {game_id}")
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        if session_id in sessions:  # sub-second collision
            session_id = f"{session_id}-{len(sessions)}"
        # The output dir and session.json are written lazily on the first snap (see
        # _write_session_meta), so opening a game without photographing it leaves no trace.
        session = Session(
            session_id=session_id,
            game=game,
            out_dir=out_root / session_id,
            started_at=_now_iso(),
            piece_set=piece_set or None,
            board=board or None,
            device=device or None,
        )
        sessions[session_id] = session
        return state_payload(session)

    @app.get("/api/meta")
    def meta() -> dict:
        """Piece-set and board ids for the capture UI's Set/Board dropdowns."""
        return _meta_options(meta_root)

    def _add_meta_entry(name: str, kind: str, item_id: str, entry: dict) -> dict:
        """Append a new entry to sets.json/boards.json, preserving existing content
        (including comments). Returns the refreshed `_meta_options` so the caller's
        dropdowns update in one round-trip."""
        item_id = (item_id or "").strip()
        if not item_id or item_id.startswith("_"):
            raise HTTPException(400, "id must be non-empty and must not start with '_'")
        path = meta_root / name
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(500, f"{name} is unreadable: {exc}") from exc
        if item_id in data:
            raise HTTPException(409, f"{kind} '{item_id}' already exists")
        data[item_id] = entry
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return _meta_options(meta_root)

    @app.post("/api/meta/board")
    def add_board(body: NewBoardIn) -> dict:
        """Add a new board to boards.json (so it shows in the Board dropdowns)."""
        entry: dict = {}
        if body.note:
            entry["_note"] = body.note
        entry["square_mm"] = body.square_mm  # may be null -> unmeasured (falls back)
        return _add_meta_entry("boards.json", "board", body.id, entry)

    @app.post("/api/meta/set")
    def add_set(body: NewSetIn) -> dict:
        """Add a new piece set to sets.json (so it shows in the Set dropdown)."""
        entry: dict = {}
        if body.note:
            entry["_note"] = body.note
        return _add_meta_entry("sets.json", "set", body.id, entry)

    @app.get("/api/sessions")
    def list_sessions() -> list[dict]:
        """Every on-disk capture session with its current domain tags and the
        filenames of its photos — drives the metadata editor's master-detail view.
        Tags come from SessionMetadata (per-session session.json overlaying the
        central sessions.json), so already-tagged legacy sessions show correctly."""
        from chessvision.data.session_meta import SessionMetadata

        meta = SessionMetadata.load(out_root)
        rows: list[dict] = []
        for d in sorted((p for p in out_root.iterdir() if p.is_dir()), reverse=True):
            images = sorted(f.name for f in d.glob("*.jpg"))
            if not images and not (d / "session.json").exists():
                continue  # not a capture session (stray dir)
            info = (meta.info(d.name) if meta else None) or {}
            rows.append(
                {
                    "session_id": d.name,
                    "set": info.get("set", ""),
                    "board": info.get("board", ""),
                    "device": info.get("device", ""),
                    "surface": info.get("surface", ""),
                    "n_captures": len(images),
                    "images": images,
                }
            )
        return rows

    @app.post("/api/sessions/{session_id}/meta")
    def update_session_meta(session_id: str, body: SessionMetaIn) -> dict:
        """Write the chosen tags into the session's session.json, preserving every
        other key (game info, notes, lighting, ...). Only fields present in the body
        are touched; an empty string clears that tag."""
        sdir = out_root / session_id
        if not sdir.is_dir():
            raise HTTPException(404, f"unknown session: {session_id}")
        path = sdir / "session.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.setdefault("session_id", session_id)
        updates = {
            "set": body.piece_set,
            "board": body.board,
            "device": body.device,
            "surface": body.surface,
        }
        for key, val in updates.items():
            if val is not None:
                data[key] = val
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Keep an active in-memory session (if this one is live) consistent.
        live = sessions.get(session_id)
        if live is not None:
            if body.piece_set is not None:
                live.piece_set = body.piece_set or None
            if body.board is not None:
                live.board = body.board or None
            if body.device is not None:
                live.device = body.device or None
        return {k: data.get(k, "") for k in ("session_id", "set", "board", "device", "surface")}

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

        _write_session_meta(session)  # creates the dir + session.json on first snap
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
        """Finalize a capture session locally (photos + session.json are already on disk).

        The Label Studio publish step is retired; capture mode itself is being phased out
        in favour of the in-app corner/position labelling over the unified store."""
        session = get_session(session_id)
        if not session.captures:
            raise HTTPException(400, "nothing captured in this session yet")
        return {"session_id": session_id, "captures": len(session.captures)}

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

    @app.get("/api/live/available")
    def live_available() -> dict:
        """Whether Read-position mode is wired. It needs *both* checkpoints: the corner
        regressor (auto-detects the board) and the keypoint detector (the pieces)."""
        return {"available": predictor is not None and corner_predictor is not None}

    @app.post("/api/live/predict")
    async def live_predict(image: UploadFile) -> dict:
        """Read an unknown position end-to-end from a single frame.

        Auto-detects the 4 board corners on the uploaded frame, then detects pieces on
        the *same* frame and maps each contact point -> square -> FEN. No corners come
        from the client: both the board and the pieces are read off this one photo.
        Returns one board SVG + FEN per orientation (R0..R270) so the client can let the
        user rotate to the reading that matches reality, plus the detected corners, grid
        lattice, and per-piece contact points for the live overlay.
        """
        if predictor is None or corner_predictor is None:
            raise HTTPException(
                503,
                "Read-position mode is off (launch with --keypoint-ckpt and --corner-ckpt)",
            )
        data = await image.read()
        if not data:
            raise HTTPException(400, "empty image upload")

        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "could not decode image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            corners = corner_predictor.predict(rgb)
        except Exception as exc:  # corner model failure -> 500 with the reason
            raise HTTPException(500, f"corner prediction failed: {exc}") from exc
        try:
            result = predictor.predict(rgb, corners)
        except Exception as exc:  # model / geometry failure -> 500 with the reason
            raise HTTPException(500, f"prediction failed: {exc}") from exc

        # Grid lines are orientation-independent (orientation only relabels squares),
        # so one lattice suffices for the overlay.
        homography = compute_homography(result.corners, Orientation.R0)
        return {
            "corners": result.corners,
            "n_detected": result.n_detected,
            "lattice": _round_pts(lattice_points(homography)),
            "orientations": {
                name: {
                    "fen": o.fen,
                    "board_fen": o.board_fen,
                    "n_placed": o.n_placed,
                    "board_svg": render_board_svg(o.fen, None, "white"),
                }
                for name, o in result.orientations.items()
            },
            "pieces": [
                {
                    "point": [round(p.point[0], 1), round(p.point[1], 1)],
                    "symbol": p.symbol,
                    "color": "w" if p.symbol.isupper() else "b",
                    "score": round(p.score, 3),
                    "squares": p.squares,
                }
                for p in result.pieces
            ],
        }

    @app.get("/api/corners/available")
    def corners_available() -> dict:
        """Whether corner-assist is wired (a corner checkpoint was provided at launch)."""
        return {"available": corner_predictor is not None}

    @app.post("/api/corners/predict")
    async def corners_predict(image: UploadFile) -> dict:
        """Predict the 4 board corners from a frame, to pre-fill the marking UI.

        Returns `{"corners": {top_left, top_right, bottom_right, bottom_left}}` in native
        image pixels; the client seeds the draggable handles with these and the user nudges.
        """
        if corner_predictor is None:
            raise HTTPException(503, "Corner-assist is off (launch with --corner-ckpt)")
        data = await image.read()
        if not data:
            raise HTTPException(400, "empty image upload")
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "could not decode image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            corners = corner_predictor.predict(rgb)
        except Exception as exc:  # model failure -> 500 with the reason
            raise HTTPException(500, f"corner prediction failed: {exc}") from exc
        return {"corners": corners}

    @app.get("/api/corners-label/available")
    def corner_label_available() -> dict:
        """Whether corner-label mode is wired (launched with --corners-root) and whether
        corner-assist prediction is also available (--corner-ckpt)."""
        return {"available": corner_store is not None, "predict": corner_predictor is not None}

    def _require_corner_store():
        if corner_store is None:
            raise HTTPException(503, "Corner-label mode is off (launch with --corners-root)")
        return corner_store

    @app.get("/api/corners-label/inbox")
    def corner_label_inbox() -> list[dict]:
        """Every decodable inbox photo, date-ordered, with its labelled state — drives
        the photo browser. The `group` is the immediate parent folder for a hierarchical,
        date-named listing."""
        store = _require_corner_store()
        return [
            {
                "id": p.id,
                "src": p.src,
                "group": p.group,
                "date": p.date,
                "labeled": p.labeled,
                "board": p.board,
                "corners": p.corners,
            }
            for p in store.list_inbox()
        ]

    @app.get("/api/corners-label/image")
    def corner_label_image(src: str, w: int | None = None) -> Response:
        """The EXIF-normalized JPEG for an inbox photo — the exact pixels the marking
        grid is drawn on and the corners are stored against. `w` downscales for thumbnails."""
        store = _require_corner_store()
        try:
            data = store.normalized_bytes(src, max_width=w)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # decode failure -> surface the reason
            raise HTTPException(400, f"could not read image: {exc}") from exc
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/corners-label/save")
    def corner_label_save(body: CornerLabelIn) -> dict:
        """Normalize the inbox photo, store its JPEG, and upsert its corner label."""
        store = _require_corner_store()
        if len(body.corners) != 4:
            raise HTTPException(400, "corners must be exactly four [x, y] points")
        try:
            label = store.save_label(
                body.src,
                body.corners,
                board=body.board or "",
                device=body.device or "",
                surface=body.surface or "",
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # save / encode failure
            raise HTTPException(500, f"save failed: {exc}") from exc
        return {"id": label.id, "src": label.src, "board": label.board, "labeled": True}

    # ---- position-label mode (project a known FEN, nudge piece bases) ------ #
    # Builds on the corner store: a photo's corners + a known FEN + the chosen
    # orientation project to per-piece contact points (geometry, no manual class
    # labelling), which the user nudges and saves as keypoint training samples.

    @app.get("/api/positions/available")
    def positions_available() -> dict:
        """Whether position-label mode is wired (--corners-root) and whether corner-assist
        prediction is also available (--corner-ckpt)."""
        return {"available": corner_store is not None, "predict": corner_predictor is not None}

    @app.get("/api/positions/inbox")
    def positions_inbox() -> list[dict]:
        """Every inbox photo with its corner + position state — drives the position
        browser. `positioned` flags photos that already have pieces placed."""
        store = _require_corner_store()
        return [
            {
                "id": p.id,
                "src": p.src,
                "group": p.group,
                "date": p.date,
                "labeled": p.labeled,  # has corners
                "board": p.board,
                "set": p.piece_set,
                "corners": p.corners,
                "positioned": p.positioned,
                "fen": p.fen,
                "orientation": p.orientation,
                "pieces": p.pieces,
            }
            for p in store.list_inbox()
        ]

    @app.get("/api/positions/image")
    def positions_image(src: str, w: int | None = None) -> Response:
        """The EXIF-normalized JPEG for an inbox photo (same pixels corners/pieces are
        stored against). `w` downscales for thumbnails."""
        store = _require_corner_store()
        try:
            data = store.normalized_bytes(src, max_width=w)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # decode failure -> surface the reason
            raise HTTPException(400, f"could not read image: {exc}") from exc
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/positions/library")
    def positions_library() -> dict[str, str]:
        """Saved {name: FEN} entries reused across same-setup photos."""
        return _require_corner_store().load_positions_library()

    @app.post("/api/positions/library")
    def positions_library_add(body: PositionLibraryIn) -> dict[str, str]:
        store = _require_corner_store()
        try:
            from chessvision.data.positions import parse_board_fen

            parse_board_fen(body.fen)  # validate before storing
        except ValueError as exc:
            raise HTTPException(400, f"invalid FEN: {exc}") from exc
        return store.save_position_entry(body.name.strip(), body.fen.strip())

    def _orientation(name: str) -> Orientation:
        try:
            return Orientation[name]
        except KeyError as exc:
            raise HTTPException(400, "orientation must be one of R0/R90/R180/R270") from exc

    @app.post("/api/positions/project")
    def positions_project(body: PositionProjectIn) -> dict:
        """Project a known FEN through the photo's corners -> starting contact points.

        Returns one point per piece (`{label, fen, square, x, y}` in the normalized
        frame) plus a reference board SVG so the user can confirm the orientation."""
        _require_corner_store()
        if len(body.corners) != 4:
            raise HTTPException(400, "corners must be exactly four [x, y] points")
        from chessvision.data.positions import project_position
        from chessvision.inference import full_fen

        corners = order_corners(body.corners)
        try:
            pieces = project_position(corners, body.fen, _orientation(body.orientation))
        except ValueError as exc:
            raise HTTPException(400, f"invalid FEN: {exc}") from exc
        return {
            "pieces": pieces,
            "board_svg": render_board_svg(full_fen(body.fen.split(" ", 1)[0]), None, "white"),
        }

    @app.post("/api/positions/save")
    def positions_save(body: PositionSaveIn) -> dict:
        """Store corners + the known FEN/orientation + nudged contact keypoints as a
        piece-keypoint training sample."""
        store = _require_corner_store()
        if len(body.corners) != 4:
            raise HTTPException(400, "corners must be exactly four [x, y] points")
        if not body.pieces:
            raise HTTPException(400, "no pieces placed")
        _orientation(body.orientation)  # validate
        try:
            label = store.save_label(
                body.src,
                body.corners,
                board=body.board or "",
                piece_set=body.piece_set or "",
                fen=body.fen.split(" ", 1)[0],
                orientation=body.orientation,
                pieces=[p.model_dump() for p in body.pieces],
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:  # save / encode failure
            raise HTTPException(500, f"save failed: {exc}") from exc
        return {
            "id": label.id,
            "src": label.src,
            "board": label.board,
            "set": label.piece_set,
            "positioned": True,
            "n_pieces": len(label.pieces),
        }

    @app.exception_handler(HTTPException)
    async def _http_error(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    return app
