"""Evaluate the trained keypoint head: square-accuracy vs the box-bottom baseline.

For each requested chessred2k split, match predicted boxes to GT pieces and, for
the matched ones, compare the predicted **contact keypoint -> square** against the
**box-bottom -> square** baseline (and the piece's true square). Reports overall
and on the occluded subset. On ChessReD box-bottom is already ~100%, so the
interesting columns are (a) does the keypoint head match it, and (b) the occluded
subset — the head's value only shows on steeper real-world angles (see captures).

Usage:
    uv run python scripts/eval_keypoint_head.py \
        --data-root "data/othersets/ChessReD" --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chessvision.data.chessred import ChessReD
from chessvision.keypoint_detector import load_keypoint_detector
from chessvision.keypoint_eval import evaluate_squares, rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path)
    add("--images-root", type=Path, default=None)
    add("--ckpt", type=Path, default=Path("runs/keypoint/best.pt"))
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--splits", default="val,test", help="comma-separated chessred2k splits")
    add("--score-thresh", type=float, default=0.5)
    add("--iou-thresh", type=float, default=0.5)
    add("--tol", type=float, default=0.06)
    add("--offset", type=float, default=0.05, help="box-bottom baseline vertical offset")
    add("--limit", type=int, default=None, help="cap images per split")
    add("--out", type=Path, default=Path("runs/keypoint/eval.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    model = load_keypoint_detector(args.ckpt, device)
    chessred = ChessReD.load(args.data_root, args.images_root)

    report = {"ckpt": str(args.ckpt), "splits": {}}
    for split in args.splits.split(","):
        ids = chessred.chessred2k_split(split)
        if args.limit:
            ids = ids[: args.limit]
        counts = evaluate_squares(
            model,
            chessred,
            ids,
            device,
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
            tol=args.tol,
            offset=args.offset,
        )
        report["splits"][split] = {k: rates(v) for k, v in counts.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
