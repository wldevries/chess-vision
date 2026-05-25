"""Harvest ground-truth positions from each session's captures.jsonl into one file.

The capture app writes a per-session `captures.jsonl` (fen, san/uci, from/to, game_id)
and `session.json` to MinIO at snap time. The Label Studio export drops those, so this
consolidates them into `data/captures/positions.json`, keyed by image filename -- an
offline ground-truth store for every frame, **puzzles included** (the Lichess puzzle id,
parsed from `game_id`, is recorded too). QC (`check_labels_vs_game.py`) and any
FEN-aware eval can then read truth directly, without re-deriving from PGN or hitting
the Lichess API.

Reads each session's `captures.jsonl` locally if present, else from S3/MinIO (creds in
.env, same as the dataset loader). Writes positions.json next to the labels (rides the
bucket with the rest of the capture data; not committed to git).

Usage:
    uv run python scripts/save_capture_positions.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chessvision.data.captures import CaptureDataset, _read_bytes

_FIELDS = ("ply_index", "fen", "san", "uci", "from", "to", "move_label")


def session_jsonl_uri(sample_s3_uri: str) -> str:
    """`s3://.../<session>/<file>.jpg` -> `s3://.../<session>/captures.jsonl`."""
    return sample_s3_uri.rsplit("/", 1)[0] + "/captures.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    p.add_argument("--out", type=Path, default=Path("data/captures/positions.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = CaptureDataset.load(args.captures)
    positions: dict[str, dict] = {}
    sessions_ok = 0
    for session, samples in dataset.by_session().items():
        uri = session_jsonl_uri(samples[0].s3_uri)
        local = str(Path(dataset.captures_root) / session / "captures.jsonl")
        try:
            raw = _read_bytes(local, uri, dataset.s3)
        except Exception as exc:  # missing jsonl / no S3 -> skip, keep going
            print(f"skip {session}: {exc}")
            continue
        sessions_ok += 1
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            game_id = row.get("game_id", "")
            entry = {"game_id": game_id, **{k: row.get(k) for k in _FIELDS}}
            if game_id.startswith("puzzle-"):
                entry["lichess_puzzle_id"] = game_id[len("puzzle-") :]
            positions[row["filename"]] = entry

    args.out.write_text(json.dumps(positions, indent=1), encoding="utf-8")
    puzzles = sum(1 for v in positions.values() if v.get("lichess_puzzle_id"))
    print(
        f"{len(positions)} positions from {sessions_ok} sessions "
        f"({puzzles} puzzle frames) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
