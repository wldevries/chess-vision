"""Load chess games (PGN files or the Lichess API) into a ply-by-ply sequence
the capture app can step through.

Each game becomes a list of `Ply` records: ply 0 is the starting position; ply i
is the position *after* the i-th half-move, tagged with the move that produced
it. The capture app shows the board at a ply, you set the pieces to match and
snap a photo, and that ply's FEN is the ground-truth label for the photo.

Pure parsing/model code -- no web or filesystem-output concerns (the app owns
those). `chess` (python-chess) handles PGN parsing, legality, SAN, and FEN.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "game"


@dataclass(frozen=True)
class Ply:
    """One position in a game's mainline.

    `fen` is the position *after* `san` has been played -- it is the ground-truth
    label for a photo taken at this ply. `turn` is the side to move *in* this
    position, so the move that produced it was played by the other side.
    """

    index: int  # 0 == starting position
    fen: str
    move_number: int  # full-move number of the move that produced this position
    turn: str  # side to move in this position: "w" or "b"
    san: str | None  # move that produced this position (None at the start)
    uci: str | None
    from_square: str | None  # e.g. "g1", for board highlighting
    to_square: str | None

    @property
    def is_start(self) -> bool:
        return self.index == 0

    @property
    def mover_is_white(self) -> bool:
        """Side that played `san` -- the opposite of the side to move now."""
        return self.turn == "b"

    @property
    def move_label(self) -> str:
        """Human move label, e.g. "12. Nf3" (white) or "12... Nf6" (black)."""
        if self.san is None:
            return "start"
        sep = "." if self.mover_is_white else "..."
        return f"{self.move_number}{sep} {self.san}"


@dataclass(frozen=True)
class Game:
    game_id: str
    white: str
    black: str
    event: str
    date: str
    result: str
    start_fen: str
    plies: list[Ply]

    @property
    def label(self) -> str:
        who = f"{self.white} - {self.black}"
        meta = ", ".join(p for p in (self.event, self.date) if p and p != "?")
        return f"{who} ({meta})" if meta else who

    @property
    def n_plies(self) -> int:
        return len(self.plies)


def game_from_node(node: chess.pgn.Game, game_id: str) -> Game:
    """Convert a parsed PGN game into a `Game` with one `Ply` per half-move."""
    headers = node.headers
    board = node.board()  # honours a FEN/SetUp header if present
    start_fen = board.fen()

    plies = [
        Ply(
            index=0,
            fen=start_fen,
            move_number=0,
            turn="w" if board.turn == chess.WHITE else "b",
            san=None,
            uci=None,
            from_square=None,
            to_square=None,
        )
    ]
    for move in node.mainline_moves():
        san = board.san(move)
        move_number = board.fullmove_number
        board.push(move)
        plies.append(
            Ply(
                index=len(plies),
                fen=board.fen(),
                move_number=move_number,
                turn="w" if board.turn == chess.WHITE else "b",
                san=san,
                uci=move.uci(),
                from_square=chess.square_name(move.from_square),
                to_square=chess.square_name(move.to_square),
            )
        )

    return Game(
        game_id=game_id,
        white=headers.get("White", "?"),
        black=headers.get("Black", "?"),
        event=headers.get("Event", "?"),
        date=headers.get("Date", "?"),
        result=headers.get("Result", "*"),
        start_fen=start_fen,
        plies=plies,
    )


def load_pgn_text(text: str, source: str = "pgn") -> list[Game]:
    """Parse every game in a PGN string (skipping any with no moves)."""
    games: list[Game] = []
    stream = io.StringIO(text)
    while (node := chess.pgn.read_game(stream)) is not None:
        if node.next() is None:
            continue  # header-only entry, nothing to step through
        games.append(game_from_node(node, f"{source}-{len(games):04d}"))
    return games


def load_pgn_file(path: str | Path) -> list[Game]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return load_pgn_text(text, source=_slug(path.stem))


def load_pgn_paths(paths: Iterable[str | Path]) -> list[Game]:
    """Load games from any mix of `.pgn` files and directories (globbed `*.pgn`).

    Game ids are made unique across all sources by appending a running counter.
    """
    games: list[Game] = []
    for raw in paths:
        path = Path(raw)
        files = sorted(path.glob("*.pgn")) if path.is_dir() else [path]
        for f in files:
            games.extend(load_pgn_file(f))
    return _dedupe_ids(games)


def fetch_lichess_user(username: str, max_games: int = 10, token: str | None = None) -> list[Game]:
    """Fetch a Lichess user's most recent games as PGN via the public API.

    Note: this only sees games played on lichess.org -- pros' over-the-board
    classics are not here. For historical master games, download a PGN collection
    (see the module/CLI docs) and pass it with --pgn instead.
    """
    import httpx  # lazy: only needed when fetching online

    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.get(
        f"https://lichess.org/api/games/user/{username}",
        params={"max": max_games, "moves": "true", "tags": "true"},
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    return load_pgn_text(resp.text, source=_slug(f"lichess-{username}"))


def _dedupe_ids(games: list[Game]) -> list[Game]:
    seen: dict[str, int] = {}
    out: list[Game] = []
    for g in games:
        if g.game_id in seen:
            seen[g.game_id] += 1
            g = Game(**{**g.__dict__, "game_id": f"{g.game_id}-{seen[g.game_id]}"})
        else:
            seen[g.game_id] = 0
        out.append(g)
    return out
