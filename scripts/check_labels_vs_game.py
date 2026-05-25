"""Validate capture labels against the **ground-truth game position**.

Capture filenames encode their source: ``<game_id>_ply<NNN>_<timestamp>.jpg``. For
PGN-sourced games (e.g. ``euwe-0000`` from ``games/Euwe.pgn``) we replay the game to
that ply and check every hand-labelled piece against the true position -- catching
wrong-square and wrong-class errors the count heuristic cannot, and confirming the
board-orientation convention (a wrong orientation mismatches every frame).

Frames whose ``game_id`` isn't in the loaded PGNs (e.g. Lichess ``puzzle-*``, whose
truth lived only in the API at capture time) fall back to the count heuristic.

Usage:
    uv run python scripts/check_labels_vs_game.py
    uv run python scripts/check_labels_vs_game.py --pgn games --pgn chessvision/capture/samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessvision.capture.games import load_pgn_paths
from chessvision.data.captures import CaptureDataset
from chessvision.data.label_qc import (
    count_problems,
    game_truth_problems,
    parse_source,
    true_board,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    p.add_argument(
        "--pgn",
        action="append",
        type=Path,
        help="PGN file or dir (repeatable). Default: games/ + chessvision/capture/samples",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pgn_paths = args.pgn or [Path("games"), Path("chessvision/capture/samples")]
    games = {g.game_id: g for g in load_pgn_paths([p for p in pgn_paths if p.exists()])}
    print(f"loaded {len(games)} games from {[str(p) for p in pgn_paths]}")

    dataset = CaptureDataset.load(args.captures)
    checked = flagged = heuristic = 0
    for sample in dataset.with_all_corners():
        src = parse_source(sample.s3_uri)
        game = games.get(src[0]) if src else None
        if game is not None and src[1] < game.n_plies:
            checked += 1
            problems = game_truth_problems(sample, true_board(game.plies[src[1]].fen))
            tag = "vs-game"
        else:
            heuristic += 1
            problems = count_problems(sample)  # puzzles / unknown source
            tag = "heuristic"
        if problems:
            flagged += 1
            print(f"task {sample.task_id} ({tag}): {'; '.join(problems)}")

    print(f"\n{flagged} flagged | {checked} checked vs game truth, {heuristic} by count heuristic.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
