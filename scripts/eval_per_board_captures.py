"""Per-BOARD capture eval, focused on the user's own boards (no ChessReD).

`eval_end_to_end_captures.py` reports one number over all held-out frames, mixing
boards. But the boards differ wildly in data volume and difficulty (Staunton 56mm vs
the cheap 30mm vs the rimless 45mm), so a single number hides where the pipeline
actually stands. This script runs the SAME two metrics -- the GT-corner ceiling
(`evaluate_captures`) and the real image->FEN end-to-end (`evaluate_end_to_end`, corner
model -> homography -> keypoint head) -- but groups the held-out frames by their
`board` tag and prints a row per board.

Source is the flat unified store (`CornerStore`, default `data/`); the split is the
SAME one the keypoint train/joint scripts use (`split_store_for_keypoints`):
  - **val** rows  = pose-held-out poses of boards the head trained on (honest
    generalization to unseen positions of a known board), tagged ``[val]``.
  - **test** rows = whole boards held out of train+val (`--test-boards`, e.g. dennis):
    the honest **unseen-board** number, tagged ``[test]``.

    uv run python scripts/eval_per_board_captures.py --device cuda \
        --keypoint-ckpt runs/keypoint_joint/best.pt

Use `--all-frames` to evaluate every frame of each board (ignores the split): a looser,
higher number (the head has seen most of these), useful only as an upper sanity bound --
the val/test rows are the honest generalization numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from chessvision.capture_eval import confusion_captures, evaluate_captures
from chessvision.corner_regressor import load_corner_regressor
from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.detection import LABEL_NAMES
from chessvision.data.positions import store_label_to_capture
from chessvision.keypoint_detector import load_keypoint_detector
from scripts.eval_end_to_end_captures import evaluate_end_to_end

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
    add("--store", type=Path, default=Path("data"), help="unified corner store root (flat layout)")
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint_captures/best.pt"))
    add("--corner-ckpt", type=Path, default=Path("runs/corners/best.pt"))
    add(
        "--test-boards",
        default="dennis-bord",
        help="comma-separated boards held out as TEST (whole board, the unseen-board number)",
    )
    add("--val-pose-frac", type=float, default=0.25, help="share of each board's poses -> val")
    add("--dedup-thr", type=float, default=0.02, help="pose-cluster dist (frac img)")
    add("--all-frames", action="store_true", help="eval every frame per board, ignoring the split")
    add("--only-board", default=None, help="restrict to one board id (e.g. staunton-56mm)")
    add("--confusion", action="store_true", help="also dump a per-GT-class confusion breakdown")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--max-size", type=int, default=1333)
    add("--score-thresh", type=float, default=0.5)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    store = CornerStore(args.store)
    test_boards = [b for b in args.test_boards.split(",") if b]

    # Pick which labels to eval and tag each by split. Store images are local (s3=None).
    if args.all_frames:
        labels_by_split = [("all", store.position_samples())]
        scope = "ALL frames (head has seen most -> upper bound, NOT honest)"
    else:
        tr, va, te = split_store_for_keypoints(
            store,
            test_boards=test_boards,
            val_pose_frac=args.val_pose_frac,
            dedup_thr=args.dedup_thr,
        )
        labels_by_split = [("val", va), ("test", te)]
        scope = f"held-out: val=pose-held-out (trained boards), test=whole boards {test_boards}"

    # Each board belongs to exactly one split (test boards are excluded from val/train),
    # so board -> (split_tag, [CaptureSample]).
    by_board: dict[str, list] = defaultdict(list)
    split_of: dict[str, str] = {}
    for tag, labels in labels_by_split:
        for lb in labels:
            b = lb.board or "<untagged>"
            if args.only_board and b != args.only_board:
                continue
            by_board[b].append(store_label_to_capture(lb, store))
            split_of[b] = tag

    print(f"scope: {scope}")
    for b, grp in sorted(by_board.items()):
        sess = sorted({s.session for s in grp})
        print(f"\n=== {b} [{split_of[b]}]  ({len(grp)} frames, {len(sess)} sessions) ===")

    kp = load_keypoint_detector(args.keypoint_ckpt, device)
    corner = load_corner_regressor(args.corner_ckpt, device=device)

    print("\n", "=" * 70, sep="")
    for b, grp in sorted(by_board.items()):
        ceiling = evaluate_captures(
            kp, grp, None, device, max_size=args.max_size, score_thresh=args.score_thresh
        )
        e2e = evaluate_end_to_end(
            kp,
            corner,
            grp,
            None,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
        )
        print(f"\n### {b} [{split_of[b]}]  ({len(grp)} frames)")
        print("  GT-corner ceiling  :", json.dumps(_round(ceiling)))
        print("  end-to-end (model) :", json.dumps(_round(e2e)))
        if args.confusion:
            confusion, false_pos = confusion_captures(
                kp,
                grp,
                None,
                device,
                max_size=args.max_size,
                score_thresh=args.score_thresh,
            )
            print("  confusion (GT class -> prediction):")
            print_confusion(confusion, false_pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
