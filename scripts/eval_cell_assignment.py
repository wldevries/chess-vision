"""Does the corner model locate the CELLS well enough? (project-goal metric)

For each held-out corner photo we have ground-truth corners. We build the true
homography H_gt and the predicted homography H_pred (from the model's corners),
then for all 64 squares:

    square center (u,v) --H_gt--> image pixel --H_pred(inverse)--> (u',v') -> square'

If square' != square, that cell would be mis-assigned *purely because of corner
error* (no piece detector involved). Reports, per board:
  - square-acc : fraction of the 64 centers that still map to the right square
  - cell-disp  : mean center displacement in SQUARE units (1.0 = one whole square)
  - board-ok   : fraction of images with all 64 squares correct
This is the number that matters for FEN, unlike raw corner-fraction error.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch

from chessvision.corner_regressor import (
    corners_from_lattice,
    load_corner_regressor,
    predict_corners,
)
from chessvision.data.corner_capture import CornerStore, select_corner_dataset_poses
from chessvision.geometry import (
    FILES,
    canonical_to_image,
    compute_homography,
    image_to_canonical,
    square_center_uv,
    uv_to_square,
)

SQUARES = [f"{f}{r}" for r in range(1, 9) for f in FILES]


def eval_image(model, rgb, gt_corners, device, predict):
    pred = predict(model, rgb, device)
    h_gt = compute_homography(gt_corners)
    h_pred = compute_homography(pred)
    centers = np.array([square_center_uv(s) for s in SQUARES], dtype=np.float32)
    img_pts = canonical_to_image(h_gt, centers)  # where each center sits in the image
    uv_pred = image_to_canonical(h_pred, img_pts)  # read back through predicted H
    correct = 0
    disp_sq = 0.0
    worst = 0.0
    for (u, v), (cu, cv), sq in zip(uv_pred, centers, SQUARES, strict=True):
        d = float(np.hypot(u - cu, v - cv)) * 8.0  # canonical -> square units
        disp_sq += d
        worst = max(worst, d)
        if uv_to_square(float(u), float(v)) == sq:
            correct += 1
    return correct / 64.0, disp_sq / 64.0, worst, correct == 64


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/corners/best.pt")
    p.add_argument("--corners-root", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dedup-thr", type=float, default=0.02)
    p.add_argument("--max-per-pose", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--lattice", action="store_true", help="checkpoint is an 81-point lattice model")
    p.add_argument("--no-conf", action="store_true", help="lattice: unweighted H-fit (ignore conf)")
    p.add_argument("--all-frames", action="store_true", help="score every labelled frame, not just "
                   "the held-out pose split (use for a board the ckpt never trained on)")
    p.add_argument("--board", default=None, help="restrict to one board tag")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_corner_regressor(args.ckpt, device=device)
    if args.lattice:
        use_conf = not args.no_conf
        def predict(m, rgb, dev):
            return corners_from_lattice(m, rgb, device=dev, use_conf=use_conf)
    else:
        def predict(m, rgb, dev):
            return predict_corners(m, rgb, device=dev)
    store = CornerStore(args.corners_root)
    if args.all_frames:
        heldout = store.samples()
    else:
        _, heldout = select_corner_dataset_poses(
            store, dedup_thr=args.dedup_thr, max_per_pose=args.max_per_pose, val_frac=args.val_frac
        )
    if args.board:
        heldout = [s for s in heldout if (s.board or "(untagged)") == args.board]

    rows = defaultdict(lambda: {"acc": [], "disp": [], "worst": [], "ok": []})
    for s in heldout:
        bgr = cv2.imread(str(store.store / s.image), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = {k: list(v) for k, v in s.corners.items()}
        acc, disp, worst, ok = eval_image(model, rgb, gt, device, predict)
        orient = "portrait" if s.width < s.height else "landscape"
        for key in (s.board or "(untagged)", f"  {s.board} [{orient}]"):
            r = rows[key]
            r["acc"].append(acc)
            r["disp"].append(disp)
            r["worst"].append(worst)
            r["ok"].append(ok)

    print(f"ckpt={args.ckpt}  held-out images={len(heldout)}")
    hdr = f"{'board':24s} {'n':>3} {'square-acc':>10} {'cell-disp':>10} {'worst-disp':>10}"
    print(f"{hdr} {'board-ok':>9}")
    # boards first (no leading spaces), then the orientation sub-rows
    def _order(k: str):
        return (k.strip().startswith(("staunton", "rimless", "cheap")) and k.startswith("  "), k)

    for key in sorted(rows, key=_order):
        r = rows[key]
        n = len(r["acc"])
        print(
            f"{key:24s} {n:>3} {100*np.mean(r['acc']):>9.1f}% {np.mean(r['disp']):>10.3f} "
            f"{np.mean(r['worst']):>10.3f} {100*np.mean(r['ok']):>8.0f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
