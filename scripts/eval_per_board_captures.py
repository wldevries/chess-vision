"""Per-BOARD capture eval, focused on the user's own boards (no ChessReD).

`eval_end_to_end_captures.py` reports one number over all held-out frames, mixing
boards. But the boards differ wildly in data volume and difficulty (Staunton 56mm vs
the cheap 30mm vs the rimless 45mm), so a single number hides where the pipeline
actually stands. This script runs the SAME two metrics -- the GT-corner ceiling
(`evaluate_captures`) and the real image->FEN end-to-end (`evaluate_end_to_end`, corner
model -> homography -> keypoint head) -- but groups the held-out val frames by their
`board` tag (from session metadata) and prints a row per board plus an overall row.

    uv run python scripts/eval_per_board_captures.py --device cuda

Use `--all-frames` to evaluate every frame of each board (not just held-out sessions):
a looser, higher number (the head has seen most of these), useful only as an upper
sanity bound -- the held-out rows are the honest generalization numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from chessvision.capture_eval import confusion_captures, evaluate_captures
from chessvision.corner_regressor import load_corner_regressor
from chessvision.data.capture_detection import CaptureKeypointConfig, split_by_sessions
from chessvision.data.captures import CaptureDataset
from chessvision.data.detection import LABEL_NAMES
from chessvision.data.session_meta import SessionMetadata
from chessvision.keypoint_detector import load_keypoint_detector
from scripts.eval_end_to_end_captures import evaluate_end_to_end
from scripts.finetune_keypoint_captures import DEFAULT_VAL_SESSIONS

KEYS = ("frames", "gt_pieces", "localization", "class_acc", "board_exact", "board_exact_rate")


def _round(d: dict) -> dict:
    return {k: (round(d[k], 4) if isinstance(d[k], float) else d[k]) for k in KEYS}


def _short(label: int | None) -> str:
    if label is None:
        return "MISSED"
    return LABEL_NAMES[label]


def print_confusion(confusion, false_pos) -> None:
    """Per-GT-class breakdown: how many were correct / missed / misclassified-as-what,
    then false positives on empty squares. Sparse (only nonzero entries) and sorted by
    error so the worst offenders surface first."""
    by_gt: dict[int, dict] = {}
    for (gt, pred), n in confusion.items():
        row = by_gt.setdefault(gt, {"total": 0, "correct": 0, "missed": 0, "wrong": Counter()})
        row["total"] += n
        if pred == gt:
            row["correct"] += n
        elif pred is None:
            row["missed"] += n
        else:
            row["wrong"][pred] += n

    # Worst classes first: lowest per-class accuracy.
    for gt in sorted(by_gt, key=lambda g: by_gt[g]["correct"] / max(by_gt[g]["total"], 1)):
        row = by_gt[gt]
        acc = row["correct"] / max(row["total"], 1)
        parts = [f"correct {row['correct']}"]
        if row["missed"]:
            parts.append(f"MISSED {row['missed']}")
        for pred, n in row["wrong"].most_common():
            parts.append(f"as-{_short(pred)} {n}")
        print(f"    {_short(gt):14s} n={row['total']:4d} acc={acc:.3f}  " + ", ".join(parts))

    if false_pos:
        fp = ", ".join(f"{_short(p)} {n}" for p, n in false_pos.most_common())
        print(f"    false positives on empty squares: {fp}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint_captures/best.pt"))
    add("--corner-ckpt", type=Path, default=Path("runs/corners/best.pt"))
    add("--val-sessions", default=DEFAULT_VAL_SESSIONS)
    add("--all-frames", action="store_true", help="eval every frame per board, not just held-out")
    add("--only-board", default=None, help="restrict to one board id (e.g. staunton-56mm)")
    add("--confusion", action="store_true", help="also dump a per-GT-class confusion breakdown")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--max-size", type=int, default=1333)
    add("--score-thresh", type=float, default=0.5)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    dataset = CaptureDataset.load(args.captures)
    meta = SessionMetadata.load(args.captures.parent)
    cfg = CaptureKeypointConfig(max_size=args.max_size)

    if args.all_frames:
        samples = list(dataset.samples)
        scope = "ALL frames (head has seen most -> upper bound, NOT honest)"
    else:
        val_sessions = [s for s in args.val_sessions.split(",") if s]
        _, val_ds = split_by_sessions(dataset, val_sessions, cfg)
        samples = val_ds.samples
        scope = f"held-out sessions {val_sessions}"

    def board_of(sess: str) -> str:
        return (meta.info(sess) or {}).get("board", "<untagged>") if meta else "<untagged>"

    by_board: dict[str, list] = defaultdict(list)
    for s in samples:
        b = board_of(s.session)
        if args.only_board and b != args.only_board:
            continue
        by_board[b].append(s)

    print(f"scope: {scope}")
    for b, grp in sorted(by_board.items()):
        sess = sorted({s.session for s in grp})
        print(f"\n=== {b}  ({len(grp)} frames, {len(sess)} sessions: {','.join(sess)}) ===")

    kp = load_keypoint_detector(args.keypoint_ckpt, device)
    corner = load_corner_regressor(args.corner_ckpt, device=device)

    print("\n", "=" * 70, sep="")
    for b, grp in sorted(by_board.items()):
        ceiling = evaluate_captures(
            kp, grp, dataset.s3, device, max_size=args.max_size, score_thresh=args.score_thresh
        )
        e2e = evaluate_end_to_end(
            kp,
            corner,
            grp,
            dataset.s3,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
        )
        print(f"\n### {b}  ({len(grp)} frames)")
        print("  GT-corner ceiling  :", json.dumps(_round(ceiling)))
        print("  end-to-end (model) :", json.dumps(_round(e2e)))
        if args.confusion:
            confusion, false_pos = confusion_captures(
                kp,
                grp,
                dataset.s3,
                device,
                max_size=args.max_size,
                score_thresh=args.score_thresh,
            )
            print("  confusion (GT class -> prediction):")
            print_confusion(confusion, false_pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
