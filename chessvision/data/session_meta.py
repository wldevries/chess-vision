"""Per-session capture metadata: which board/piece-set each session used, plus the
physical measurements that let box synthesis size pieces correctly across boards.

Three small JSON sidecars live next to the Label Studio export (`data/captures/`,
synced with the rest of the capture data, not committed):

    sets.json     set_id   -> {fen_letter: {"height_mm", "base_mm"}}   (measure once/set)
    boards.json   board_id -> {"square_mm"}                            (one number/board)
    sessions.json session  -> {"set", "board", "lighting", "device", ...}

The payoff: a piece's box height in *squares* is ``height_mm / square_mm`` and its
radius is ``(base_mm / 2) / square_mm``. So the same set on a board with smaller cells
spans more squares and its boxes grow automatically -- no per-board hand-tuning of
``geometry.PIECE_HEIGHT_SCALE``. Any missing measurement (null) falls back to that
constant for the affected piece, so a half-filled file still works.

Keys beginning with ``_`` (e.g. ``_comment``/``_note``) are documentation and ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_FEN_LETTERS = "prnbqk"
_DEFAULT_RADIUS_SQUARES = 0.3

# Domain-axis keys read off a per-session session.json (mirrors the sessions.json schema).
_DOMAIN_KEYS = ("set", "board", "device", "lighting", "surface", "notes")

# fen_letter -> (height_squares, radius_squares)
BoxSizes = dict[str, tuple[float, float]]


def resolve_box_sizes(
    set_def: dict, square_mm: float, default_radius_squares: float = _DEFAULT_RADIUS_SQUARES
) -> BoxSizes:
    """Per-piece (height_squares, radius_squares) from a set's mm measurements and a
    board's square size. Pieces with a missing/non-positive ``height_mm`` are omitted
    (caller falls back to ``PIECE_HEIGHT_SCALE``); a missing ``base_mm`` uses the
    default radius."""
    out: BoxSizes = {}
    if not square_mm or square_mm <= 0:
        return out
    for fen in _FEN_LETTERS:
        piece = set_def.get(fen)
        if not isinstance(piece, dict):
            continue
        height = piece.get("height_mm")
        if not height or height <= 0:
            continue
        base = piece.get("base_mm")
        radius_sq = (
            (float(base) / 2.0) / square_mm if (base and base > 0) else default_radius_squares
        )
        out[fen] = (float(height) / square_mm, radius_sq)
    return out


def _strip_comments(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


@dataclass(frozen=True)
class SessionMetadata:
    sets: dict
    boards: dict
    sessions: dict

    @classmethod
    def load(cls, root: str | Path) -> SessionMetadata | None:
        """Read the three sidecars from `root`. Returns None if none exist, so callers
        stay fully functional without any metadata."""
        root = Path(root)

        def read(name: str) -> dict:
            path = root / name
            if not path.exists():
                return {}
            return _strip_comments(json.loads(path.read_text(encoding="utf-8")))

        sets, boards, sessions = read("sets.json"), read("boards.json"), read("sessions.json")

        # Overlay per-session session.json (written by the capture app at session start):
        # the set/board/device live with the session itself, so it is the source of truth.
        # Only non-empty domain keys override, so a session.json carrying just game info
        # never blanks a central entry (e.g. a held-out `notes` tag set centrally).
        for sj in sorted(root.glob("*/session.json")):
            try:
                data = json.loads(sj.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            domain = {k: data[k] for k in _DOMAIN_KEYS if data.get(k)}
            if domain:
                sessions[sj.parent.name] = {**sessions.get(sj.parent.name, {}), **domain}

        if not (sets or boards or sessions):
            return None
        return cls(sets=sets, boards=boards, sessions=sessions)

    def info(self, session: str) -> dict | None:
        """Raw metadata row for a session (set/board/lighting/...), or None."""
        return self.sessions.get(session)

    def piece_box_sizes(
        self, session: str, *, default_radius_squares: float = _DEFAULT_RADIUS_SQUARES
    ) -> BoxSizes:
        """Resolve a session -> per-piece (height_squares, radius_squares), or {} if its
        set/board/square_mm aren't (yet) measured."""
        info = self.sessions.get(session)
        if not info:
            return {}
        set_def = self.sets.get(info.get("set"))
        board = self.boards.get(info.get("board"))
        if not set_def or not board:
            return {}
        return resolve_box_sizes(set_def, board.get("square_mm"), default_radius_squares)
