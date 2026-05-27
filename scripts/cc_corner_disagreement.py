"""How far off are the capture corners? Compare the (lattice) model's corners to the
labelled corners for every capture frame, positionally (both canonicalized to visual
slots, so the capture set's own corner-naming convention doesn't matter).

Large disagreement flags a frame worth reviewing -- either a mis-placed label or a model
miss on that (possibly domain-shifted) capture. Reports the distribution + worst frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chessvision.corner_regressor import load_corner_regressor, corners_from_lattice
from chessvision.data.captures import CaptureDataset
from chessvision.geometry import order_corners

SLOTS = ("top_left", "top_right", "bottom_right", "bottom_left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/corners/best.pt")
    args = ap.parse_args()
    model = load_corner_regressor(args.ckpt, device="cuda")
    ds = CaptureDataset.load("data/captures/label-studio.json", "data/captures")
    rows = []
    radial = []  # signed: label-radius minus model-radius from the quad center (rim-offset sign)
    for s in ds.samples:
        if not s.has_all_corners:
            continue
        rgb = s.load_image()
        pred = corners_from_lattice(model, rgb, device="cuda")  # visual-slot dict
        ov = order_corners(list(s.corners.values()))
        pv = order_corners(list(pred.values()))
        disp = max(np.hypot(ov[k][0] - pv[k][0], ov[k][1] - pv[k][1]) for k in SLOTS)
        # systematic rim bias: is the LABEL consistently farther from the board center
        # than the MODEL corner? center = mean of the 4 model corners.
        cx = np.mean([pv[k][0] for k in SLOTS]); cy = np.mean([pv[k][1] for k in SLOTS])
        for k in SLOTS:
            rl = np.hypot(ov[k][0] - cx, ov[k][1] - cy)
            rm = np.hypot(pv[k][0] - cx, pv[k][1] - cy)
            radial.append((rl - rm) / max(s.width, s.height))
        rows.append((disp / max(s.width, s.height), disp, s.session, Path(s.image_path).name))

    rows.sort(reverse=True)
    fr = np.array([r[0] for r in rows])
    print(f"{len(rows)} frames with all corners; model-vs-label max-corner displacement")
    print(f"  (fraction of long side)  median={np.median(fr):.4f}  mean={fr.mean():.4f}")
    for lo, hi in [(0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 9)]:
        n = int(((fr >= lo) & (fr < hi)).sum())
        print(f"    {lo:.2f}-{hi:.2f} long-side: {n:>3} frames ({100*n/len(fr):.0f}%)")
    rad = np.array(radial)
    print(
        f"\nsystematic rim bias (label radius - model radius, frac of long side): "
        f"mean={rad.mean():+.4f}  median={np.median(rad):+.4f}  "
        f"(>0 => labels sit OUTSIDE the model's corners = rim offset)"
    )
    print("\nworst 12 frames (review these first):")
    for frac, px, sess, name in rows[:12]:
        print(f"  {frac:.3f} ({px:.0f}px)  {sess}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
