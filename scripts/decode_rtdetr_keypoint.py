"""Compare board-decode strategies for the RT-DETR keypoint model (no retraining).

The keypoint head's per-piece class_acc is strong, but `board_exact` (whole-board FEN) is low
because RT-DETR emits 300 low-confidence queries that drop false pieces onto empty squares. The
fix is decode-side: how we turn (contact-point, class, score) per query into a 64-square board.

Strategies (all on the SAME cached model outputs, so only the decode differs):
  - fixed       : per-square top-1 query above a fixed score floor (the current eval's decode).
  - val-tuned   : same, but the floor is swept on store-VAL to maximize board_exact, then applied
                  FROZEN to dennis -- never tuned on test (that would leak).
  - gap         : parameter-free. Per square keep the best query, sort those, and cut at the
                  largest score gap within the plausible 2..32-piece range -- the board's own
                  score distribution sets the cutoff, so nothing is tuned on either split.

Reports localization / class_acc / board_exact on store-val and the held-out dennis test.

    uv run --group rtdetr python scripts/decode_rtdetr_keypoint.py \
        --ckpt runs/rtdetr_keypoint_aug/best --store data --test-boards dennis-bord --board-crop
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from chessvision.capture_eval import _gt_board
from chessvision.data.capture_detection import CaptureKeypointConfig, CaptureKeypointDetection
from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.detection import resize_targets
from chessvision.data.positions import store_label_to_capture
from chessvision.geometry import (
    Orientation,
    board_crop_bbox,
    compute_homography,
    square_for_point,
)
from chessvision.rtdetr_keypoint import TorchvisionDetAdapter, load_rtdetr_keypoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--ckpt", type=Path, default=Path("runs/rtdetr_keypoint_aug/best"))
    add("--store", type=Path, default=Path("data"))
    add("--test-boards", default="dennis-bord")
    add("--val-pose-frac", type=float, default=0.25)
    add("--dedup-thr", type=float, default=0.02)
    add("--max-size", type=int, default=1333)
    add("--board-crop", action="store_true")
    add("--crop-side", type=float, default=0.12)
    add("--crop-top", type=float, default=0.30)
    add("--crop-bottom", type=float, default=0.08)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


@torch.no_grad()
def square_candidates(adapter, sample, device, max_size, crop) -> list[tuple[str, float, int]]:
    """All (square, score, label) for queries whose contact point lands on a board square, in
    the same crop+resized frame the model trained on (mirrors capture_eval._detect_squares)."""
    rgb = sample.load_image(None)
    h, w = rgb.shape[:2]
    ox, oy = 0.0, 0.0
    if crop["board_crop"]:
        x0, y0, x1, y1 = board_crop_bbox(
            sample.corners,
            w,
            h,
            side=crop["crop_side"],
            top=crop["crop_top"],
            bottom=crop["crop_bottom"],
        )
        rgb = rgb[y0:y1, x0:x1]
        ox, oy = float(x0), float(y0)
    hs, ws = rgb.shape[:2]
    scale = min(1.0, max_size / max(hs, ws))
    rgb, _, _ = resize_targets(rgb, np.zeros((0, 4), np.float32), None, max_size)
    corners = {k: ((x - ox) * scale, (y - oy) * scale) for k, (x, y) in sample.corners.items()}
    homography = compute_homography(corners, Orientation.R0)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255
    out = adapter([image])[0]
    cands = []
    for score, label, kps in zip(
        out["scores"].tolist(), out["labels"].tolist(), out["keypoints"].numpy(), strict=True
    ):
        sq = square_for_point(homography, (float(kps[0, 0]), float(kps[0, 1])))
        if sq is not None:
            cands.append((sq, float(score), int(label)))
    return cands


def _per_square_best(cands: list[tuple[str, float, int]]) -> dict[str, tuple[float, int]]:
    best: dict[str, tuple[float, int]] = {}
    for sq, score, label in cands:
        if sq not in best or score > best[sq][0]:
            best[sq] = (score, label)
    return best


def decode_fixed(cands, tau: float) -> dict[str, int]:
    return {sq: lab for sq, (s, lab) in _per_square_best(cands).items() if s >= tau}


def decode_gap(cands, kmin: int = 2, kmax: int = 32, floor: float = 0.01) -> dict[str, int]:
    """Per-square best, sorted by score; cut at the largest score gap within [kmin, kmax] pieces.
    Parameter-free w.r.t. the dataset -- each board's own score distribution picks the cutoff."""
    items = sorted(_per_square_best(cands).items(), key=lambda kv: kv[1][0], reverse=True)
    items = [(sq, lab, s) for sq, (s, lab) in items if s >= floor]
    n = len(items)
    hi = min(kmax, n)
    if hi <= kmin:
        return {sq: lab for sq, lab, _ in items[:hi]}
    scores = [s for _, _, s in items]
    best_k, best_gap = hi, -1.0
    for k in range(kmin, hi):  # keep k pieces -> cut between scores[k-1] and scores[k]
        gap = scores[k - 1] - scores[k]
        if gap > best_gap:
            best_gap, best_k = gap, k
    return {sq: lab for sq, lab, _ in items[:best_k]}


def tally(board: dict[str, int], gt: dict[str, int], c: dict[str, int]) -> None:
    frame_ok = True
    for sq, lab in gt.items():
        if sq in board:
            c["localized"] += 1
            if board[sq] == lab:
                c["class_correct"] += 1
            else:
                frame_ok = False
        else:
            frame_ok = False
    if any(sq not in gt for sq in board):
        frame_ok = False
    c["board_exact"] += int(frame_ok)
    c["gt_pieces"] += len(gt)
    c["frames"] += 1


def rates(c: dict[str, int]) -> dict[str, float]:
    gp = max(c["gt_pieces"], 1)
    fr = max(c["frames"], 1)
    return {
        "localization": c["localized"] / gp,
        "class_acc": c["class_correct"] / gp,
        "board_exact": c["board_exact"] / fr,
    }


def evaluate(decode, cached) -> dict[str, float]:
    c: dict[str, int] = defaultdict(int)
    for cands, gt in cached:
        tally(decode(cands), gt, c)
    return rates(c)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    store = CornerStore(args.store)
    test_boards = [b for b in args.test_boards.split(",") if b]
    _, va, te = split_store_for_keypoints(
        store, test_boards=test_boards, val_pose_frac=args.val_pose_frac, dedup_thr=args.dedup_thr
    )
    cfg = CaptureKeypointConfig(max_size=args.max_size, board_crop=args.board_crop)
    mk = lambda L: CaptureKeypointDetection(  # noqa: E731
        [store_label_to_capture(lb, store) for lb in L], None, cfg, train=False
    )
    val_ds, test_ds = mk(va), mk(te)

    model, processor = load_rtdetr_keypoint(args.ckpt, device=device)
    adapter = TorchvisionDetAdapter(model, processor, device)
    crop = dict(
        board_crop=args.board_crop,
        crop_side=args.crop_side,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
    )

    # Run the model once per frame; cache (candidates, gt) so decode strategies are free to compare.
    def cache(ds):
        return [
            (square_candidates(adapter, s, device, args.max_size, crop), _gt_board(s))
            for s in ds.samples
        ]

    print("caching model outputs (val + test) ...", flush=True)
    val_cached, test_cached = cache(val_ds), cache(test_ds)

    # val-tuned threshold: pick the floor maximizing val board_exact, then FREEZE for test.
    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]
    tuned_tau = max(
        grid, key=lambda t: evaluate(lambda c: decode_fixed(c, t), val_cached)["board_exact"]
    )

    strategies = {
        "fixed tau=0.02": lambda c: decode_fixed(c, 0.02),
        f"val-tuned tau={tuned_tau}": lambda c: decode_fixed(c, tuned_tau),
        "gap (param-free)": decode_gap,
    }

    print(f"\n{'strategy':<22} {'val loc/cls/exact':<26} test loc/cls/exact")
    print("-" * 70)
    for name, dec in strategies.items():
        v, t = evaluate(dec, val_cached), evaluate(dec, test_cached)
        print(
            f"{name:<22} "
            f"{v['localization']:.3f}/{v['class_acc']:.3f}/{v['board_exact']:.3f}        "
            f"{t['localization']:.3f}/{t['class_acc']:.3f}/{t['board_exact']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
