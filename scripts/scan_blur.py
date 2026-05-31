"""Score every piece-labelled store image for blur, board-cropped, with a perceptual metric.

Variance-of-Laplacian (the obvious focus measure) tracks *total detail*, not sharpness, so a
sharp sparse position scores like a blurry busy one. This uses the **Crete re-blur metric**
(Crete et al. 2007): re-blur the image and measure how little the edge variation changes -- a
sharp frame changes a lot (low blur score), an already-soft frame barely changes (high blur
score). It's a *ratio*, so it's largely content-invariant. We crop to the board (from the corner
labels) first so background floor/table doesn't dominate.

Writes data/_blur_scores.json with both metrics per image (cb = Crete blur in [0,1], higher =
blurrier; fm = old Laplacian variance, kept for reference), sorted blurriest-first by cb.

    uv run python scripts/scan_blur.py
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from chessvision.data.corner_capture import CornerStore


def crete_blur(gray: np.ndarray) -> float:
    """Crete no-reference perceptual blur in [0,1] (higher = blurrier). Content-invariant
    because it's the fraction of edge variation lost to a re-blur, not absolute edge energy."""
    g = gray.astype(np.float32)
    bh = cv2.blur(g, (1, 9))  # horizontal low-pass
    bv = cv2.blur(g, (9, 1))  # vertical low-pass
    d_fh, d_fv = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    d_bh, d_bv = np.abs(np.diff(bh, axis=1)), np.abs(np.diff(bv, axis=0))
    vh = np.maximum(0.0, d_fh - d_bh)
    vv = np.maximum(0.0, d_fv - d_bv)
    s_fh, s_fv = d_fh.sum(), d_fv.sum()
    bm_h = (s_fh - vh.sum()) / s_fh if s_fh > 0 else 0.0
    bm_v = (s_fv - vv.sum()) / s_fv if s_fv > 0 else 0.0
    return float(max(bm_h, bm_v))


def board_crop(gray: np.ndarray, corners: dict) -> np.ndarray:
    """Crop to the board's corner bounding box (5% margin), so background isn't scored."""
    pts = np.array(list(corners.values()), dtype=np.float32)
    h, w = gray.shape
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    crop = gray[y1:y2, x1:x2]
    return crop if crop.size else gray


def main() -> int:
    store = CornerStore("data")
    samples = store.position_samples()
    rows = []
    for s in samples:
        bgr = cv2.imread(str(store.store / s.image), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if s.corners:
            gray = board_crop(gray, s.corners)
        h, w = gray.shape
        sc = 512 / max(h, w)
        if sc < 1:
            gray = cv2.resize(gray, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        fm = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        rows.append(
            {
                "image": s.image,
                "board": s.board or "?",
                "cb": round(crete_blur(gray), 4),
                "fm": round(fm, 1),
            }
        )
    rows.sort(key=lambda r: r["cb"], reverse=True)  # blurriest (highest cb) first
    json.dump(rows, open("data/_blur_scores.json", "w"))
    cbs = np.array([r["cb"] for r in rows])
    print(f"n={len(rows)} | cb percentiles (higher=blurrier):")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q}: {np.percentile(cbs, q):.3f}")
    print("--- 15 blurriest by Crete (cb) ---")
    for r in rows[:15]:
        print(f"  cb={r['cb']:.3f}  fm={r['fm']:7.1f}  {r['board']:14s} {r['image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
