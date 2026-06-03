"""FastAPI app for the position-capture workflow.

Holds the loaded games and in-memory capture sessions, renders each position as
an SVG board (via `chess.svg`), and on each "snap" writes the uploaded photo plus
a JSONL metadata row whose `fen` field is the ground-truth label. Single local
user, so session state lives in memory; photos and metadata are the durable
output, organised per session under `out_root/<session_id>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.svg
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chessvision.geometry import (
    Orientation,
    compute_homography,
    lattice_points,
    order_corners,
)

STATIC_DIR = Path(__file__).parent / "static"

CornerDict = dict[str, list[float]]


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


def create_app(
    out_root: Path,
    *,
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

    # sets.json/boards.json (the Set/Board dropdowns + mm reference) live at the unified
    # store root in the flat layout; fall back to out_root only when no store is wired.
    meta_root = Path(corners_root) if corners_root is not None else out_root

    # Corner-label mode: an import-and-label flow over phone photos staged in the
    # `data/source/inbox/` tree, writing corner-only labels (no FEN / Label
    # Studio). Off unless a corners root is provided; a pre-built store can be injected
    # for tests. See chessvision/data/corner_capture.py and docs/corner-capture-mode.md.
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
    # Shared static assets (cv.js etc.) for the page templates served below. The HTML
    # pages themselves are returned by explicit routes (/, /corners, /live, ...) so the
    # mount is for sidecar assets only, not the entry pages.
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Launcher hub: cards linking to each capture/labelling tool."""
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

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
        """Every session in the unified store with its domain tags + photo count, newest
        first -- drives the session editor. Sessions are the real capture sessions baked
        into each record (labels.jsonl), not a directory scan."""
        return _require_corner_store().sessions_summary()

    @app.post("/api/sessions/{session_id}/meta")
    def update_session_meta(session_id: str, body: SessionMetaIn) -> dict:
        """Bulk-set board/set/device/surface on every record of a session (labels.jsonl).
        Only fields present in the body are touched; an empty string clears that tag."""
        updated = _require_corner_store().retag_session(
            session_id,
            board=body.board,
            piece_set=body.piece_set,
            device=body.device,
            surface=body.surface,
        )
        if updated is None:
            raise HTTPException(404, f"unknown session: {session_id}")
        return updated

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

    @app.get("/api/store/image")
    def store_image(path: str, w: int | None = None) -> Response:
        """Serve any normalized store image by its relpath (== record id) -- e.g. the
        session editor's photo previews. `w` downscales for thumbnails."""
        store = _require_corner_store()
        try:
            data = store.store_image_bytes(path, max_width=w)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
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
