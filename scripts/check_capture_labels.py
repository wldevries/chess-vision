"""Sanity-check capture labels with the offline count heuristic.

Flags frames whose piece keypoints are implausible from labels + corners alone:
duplicate squares, piece counts above the no-promotion maximum, wrong king count,
>32 pieces. Cannot catch a wrong-class piece that keeps counts legal -- for that, use
`check_labels_vs_game.py` (authoritative, where the source PGN is available).

Exits non-zero if any frame is flagged, so it can gate a label-cleaning pass.

Usage:
    uv run python scripts/check_capture_labels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessvision.data.captures import CaptureDataset
from chessvision.data.label_qc import count_problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = CaptureDataset.load(args.captures)
    flagged = total = 0
    for sample in dataset.with_all_corners():
        total += 1
        problems = count_problems(sample)
        if problems:
            flagged += 1
            print(f"task {sample.task_id} ({sample.session}): {'; '.join(problems)}")
    print(f"\n{flagged} / {total} full-corner frames flagged.")
    if flagged:
        print("Review in Label Studio (a count check misses wrong-class-but-right-count).")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
