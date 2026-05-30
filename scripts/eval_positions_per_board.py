"""Per-board piece eval on the in-app POSITION labels (data/corners), not captures.

Every corner photo now also carries piece keypoints (the /positions tool projects a
known FEN through the homography, user nudges each base). `position_samples_as_captures`
turns those into `CaptureSample`s keyed `pos-<board>`, so we can score the deployed
keypoint head on them with the exact same two metrics as `eval_per_board_captures.py`:

  - GT-corner ceiling  (`evaluate_captures`): pieces only, using the labelled corners ->
    isolates the keypoint head's per-piece localization + classification.
  - end-to-end (model) (`evaluate_end_to_end`): corner model -> homography -> keypoint head,
    the real image->FEN path.

A board the checkpoint never trained on (e.g. dennis-bord) is a true unseen-board piece
test -- the FEN bottleneck the corner set alone could not measure.

    uv run python scripts/eval_positions_per_board.py --device cuda
    uv run python scripts/eval_positions_per_board.py --only-board dennis-bord --confusion
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from chessvision.capture_eval import confusion_captures, evaluate_captures
from chessvision.corner_regressor import load_corner_regressor
from chessvision.data.positions import position_samples_as_captures
from chessvision.keypoint_detector import load_keypoint_detector
from scripts.eval_end_to_end_captures import evaluate_end_to_end
from scripts.eval_per_board_captures import _round, print_confusion


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--corners-root", type=Path, default=Path("data/corners"))
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint_captures/best.pt"))
    add("--corner-ckpt", type=Path, default=Path("runs/corners/best.pt"))
    add("--only-board", default=None, help="restrict to one board id (e.g. dennis-bord)")
    add("--confusion", action="store_true", help="also dump a per-GT-class confusion breakdown")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--max-size", type=int, default=1333)
    add("--score-thresh", type=float, default=0.5)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    samples = position_samples_as_captures(args.corners_root)
    if not samples:
        raise SystemExit(f"no position-labelled samples under {args.corners_root}")

    by_board: dict[str, list] = defaultdict(list)
    for s in samples:
        board = s.session.removeprefix("pos-")  # session is pos-<board>
        if args.only_board and board != args.only_board:
            continue
        by_board[board].append(s)

    print(f"source: {args.corners_root} positions  (all frames are out-of-sample unless the")
    print("        board was folded into keypoint training via --positions-root)")
    for b, grp in sorted(by_board.items()):
        print(f"  {b:16s}: {len(grp)} frames")

    kp = load_keypoint_detector(args.keypoint_ckpt, device)
    corner = load_corner_regressor(args.corner_ckpt, device=device)

    print("\n", "=" * 70, sep="")
    for b, grp in sorted(by_board.items()):
        ceiling = evaluate_captures(
            kp, grp, None, device, max_size=args.max_size, score_thresh=args.score_thresh
        )
        e2e = evaluate_end_to_end(
            kp, corner, grp, None, device, max_size=args.max_size, score_thresh=args.score_thresh
        )
        print(f"\n### {b}  ({len(grp)} frames)")
        print("  GT-corner ceiling  :", json.dumps(_round(ceiling)))
        print("  end-to-end (model) :", json.dumps(_round(e2e)))
        if args.confusion:
            confusion, false_pos = confusion_captures(
                kp, grp, None, device, max_size=args.max_size, score_thresh=args.score_thresh
            )
            print("  confusion (GT class -> prediction):")
            print_confusion(confusion, false_pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
