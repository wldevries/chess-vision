"""Dense-position cell-assignment eval on ChessReD (the occlusion test).

ChessReD has OFFICIAL corners (trusted ground truth, unlike the capture set whose corner
labels may be off) and per-image piece counts. We eval on the held-out val+test splits
(the lattice trained on chessred2k train), filtered to boards with > MIN_PIECES pieces --
i.e. crowded boards where pieces occlude the rim/interior. Metric is the same as
eval_cell_assignment: project each square center through the GT homography, read it back
through the predicted homography, count cells that still land in the right square.

Reports board-ok / square-acc / cell-disp, bucketed by piece count, for the lattice model
with and without confidence-weighted H-fitting (the place confidence should finally matter).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from chessvision.corner_regressor import corners_from_lattice, load_corner_regressor
from chessvision.data.chessred import ChessReD
from chessvision.geometry import (
    FILES,
    canonical_to_image,
    compute_homography,
    image_to_canonical,
    order_corners,
    square_center_uv,
    uv_to_square,
)

SQUARES = [f"{f}{r}" for r in range(1, 9) for f in FILES]
CENTERS = np.array([square_center_uv(s) for s in SQUARES], dtype=np.float32)


def cell_scores(gt_corners, pred_corners):
    h_gt = compute_homography(gt_corners)
    h_pred = compute_homography(pred_corners)
    img_pts = canonical_to_image(h_gt, CENTERS)
    uv = image_to_canonical(h_pred, img_pts)
    correct = disp = 0.0
    for (u, v), (cu, cv), sq in zip(uv, CENTERS, SQUARES, strict=True):
        disp += float(np.hypot(u - cu, v - cv)) * 8.0
        correct += uv_to_square(float(u), float(v)) == sq
    return correct / 64.0, disp / 64.0, correct == 64


def bucket(n):
    return "17-24" if n <= 24 else "25-32" if n <= 32 else "33+"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/corners_lattice/l384/best.pt")
    p.add_argument("--data-root", default="data/othersets/ChessReD")
    p.add_argument("--min-pieces", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_corner_regressor(args.ckpt, device=device)
    chess = ChessReD.load(args.data_root)

    ids = []
    for sp in args.splits:
        ids += chess.chessred2k_split(sp)
    dense = [(i, len(chess.pieces(i))) for i in ids if chess.corners(i) is not None]
    dense = [(i, n) for i, n in dense if n > args.min_pieces]
    print(f"ckpt={args.ckpt}  splits={args.splits}  >{args.min_pieces} pieces: {len(dense)} images")

    results = {True: [], False: []}  # use_conf -> list of (acc, disp, ok, n_pieces)
    for img_id, npieces in dense:
        meta = chess.meta(img_id)
        rgb = cv2.cvtColor(cv2.imread(str(chess.resolve_image_path(meta))), cv2.COLOR_BGR2RGB)
        gt = order_corners(list(chess.corners(img_id).values()))  # semantic -> visual slots
        for use_conf in (True, False):
            pred = corners_from_lattice(model, rgb, device=device, use_conf=use_conf)
            acc, disp, ok = cell_scores(gt, pred)
            results[use_conf].append((acc, disp, ok, npieces))

    for use_conf in (True, False):
        rows = results[use_conf]
        tag = "conf-weighted" if use_conf else "UNweighted "
        accs = np.array([r[0] for r in rows])
        disps = np.array([r[1] for r in rows])
        oks = np.array([r[2] for r in rows])
        print(
            f"  [{tag}] ALL>{args.min_pieces}: n={len(rows)} "
            f"square-acc={100*accs.mean():.1f}% cell-disp={disps.mean():.3f} "
            f"board-ok={100*oks.mean():.0f}%"
        )
        for b in ("17-24", "25-32", "33+"):
            idx = [k for k, r in enumerate(rows) if bucket(r[3]) == b]
            if idx:
                print(
                    f"      {b:>6} pieces (n={len(idx):>3}): "
                    f"square-acc={100*accs[idx].mean():.1f}% "
                    f"cell-disp={disps[idx].mean():.3f} board-ok={100*oks[idx].mean():.0f}%"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
