"""Launch the position-capture web app.

    # bundled sample game, opens on http://127.0.0.1:8000
    uv run python -m chessvision.capture

    # your own PGN collection (a file or a folder of .pgn files)
    uv run python -m chessvision.capture --pgn games/carlsen.pgn

    # pull a lichess user's recent online games
    uv run python -m chessvision.capture --lichess-user DrNykterstein --max-games 20

Puzzles can also be pulled live from the UI ("Next puzzle"), one fresh puzzle per
click via the Lichess API — good for diverse, occlusion-heavy positions. Set
LICHESS_TOKEN (a personal access token from lichess.org/account/oauth/token; the
API has no user/password auth) in the environment or .env to de-duplicate against
your solved puzzles and honour difficulty.

Where to get historical pro games (download a .pgn and pass it with --pgn):
  - pgnmentor.com/files.html       per-player collections (Carlsen, Fischer, ...)
  - The Week in Chess (theweekinchess.com)   weekly tournament PGNs
  - Lichess Elite Database          large filtered online-game dump

Photos and a captures.jsonl (one row per photo, with the ground-truth FEN) are
written under --out, one folder per session.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from chessvision.capture.app import create_app
from chessvision.capture.games import Game, fetch_lichess_user, load_pgn_paths

SAMPLE_PGN = Path(__file__).parent / "samples" / "opera_game.pgn"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m chessvision.capture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pgn",
        type=Path,
        action="append",
        default=[],
        help="PGN file or directory (repeatable); omit to use the bundled sample",
    )
    p.add_argument("--lichess-user", default=None, help="also fetch this lichess user's games")
    p.add_argument("--max-games", type=int, default=20, help="max games to fetch from lichess")
    p.add_argument("--out", type=Path, default=Path("data/captures"), help="output directory")
    p.add_argument("--host", default="127.0.0.1", help="bind host (use 0.0.0.0 for LAN access)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--keypoint-ckpt",
        type=Path,
        default=None,
        help="piece keypoint checkpoint for Read-position (live FEN) mode "
        "(e.g. runs/keypoint_captures/best.pt); needs --corner-ckpt too, since a Read "
        "auto-detects the board corners before the pieces",
    )
    p.add_argument(
        "--corner-ckpt",
        type=Path,
        default=None,
        help="corner-regressor checkpoint (e.g. runs/corners/best.pt): drives corner-assist "
        "(the 'Predict' button pre-fills the corner-marking handles) and the automatic "
        "corner detection in Read-position mode",
    )
    p.add_argument(
        "--corners-root",
        type=Path,
        default=Path("data"),
        help="unified store root (flat layout): phone photos staged in <root>/source/inbox/ are "
        "labelled in the app and written as normalized images to <root>/store/ with the index at "
        "<root>/labels.jsonl. On by default; pass --no-corners-root to disable.",
    )
    p.add_argument(
        "--no-corners-root",
        dest="corners_root",
        action="store_const",
        const=None,
        help="disable corner-label mode (hide the Corners tab)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device for live inference (default: cuda if available, else cpu)",
    )
    return p.parse_args(argv)


def load_games(args: argparse.Namespace) -> list[Game]:
    games: list[Game] = []
    if args.pgn:
        games.extend(load_pgn_paths(args.pgn))
    if args.lichess_user:
        games.extend(fetch_lichess_user(args.lichess_user, args.max_games))
    if not games:
        games.extend(load_pgn_paths([SAMPLE_PGN]))
    return games


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    games = load_games(args)
    if not games:
        print("No games loaded. Pass --pgn FILE or --lichess-user NAME.")
        return 1

    token = os.environ.get("LICHESS_TOKEN")
    print(f"Loaded {len(games)} game(s); writing captures under {args.out.resolve()}")
    print("Live puzzles: " + ("token set" if token else "anonymous (set LICHESS_TOKEN to dedupe)"))
    print(
        f"Open http://{args.host}:{args.port} on the tablet, then pick a game and start snapping."
    )
    if args.keypoint_ckpt and args.corner_ckpt:
        print(
            f"Read-position mode ON · pieces {args.keypoint_ckpt} · corners {args.corner_ckpt}"
        )
    elif args.keypoint_ckpt:
        print(
            "Read-position mode OFF: --keypoint-ckpt also needs --corner-ckpt "
            "(corners are auto-detected per read)"
        )
    if args.corner_ckpt:
        print(f"Corner-assist ON · checkpoint {args.corner_ckpt}")
    if args.corners_root:
        print(
            f"Corner-label mode ON · staging {args.corners_root}/inbox -> {args.corners_root}/store"
        )
    app = create_app(
        games,
        args.out,
        lichess_token=token,
        keypoint_ckpt=args.keypoint_ckpt,
        corner_ckpt=args.corner_ckpt,
        corners_root=args.corners_root,
        device=args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
