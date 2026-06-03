"""Render the lattice model's 81 channel heatmaps in a 9x9 grid for one good image.

Picks a held-out corner-store image where the predicted homography gives all 64
cells correct (board-ok), then dumps the raw per-channel softmax maps as a 9x9
panel so we can see if each channel actually localizes its assigned intersection,
or if interior channels just put mass on the 4 corners and ride soft-argmax to
the right centroid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from chessvision.corner_regressor import (
    corners_from_lattice,
    load_corner_regressor,
)
from chessvision.data.corner_capture import CornerStore, select_corner_dataset_poses
from chessvision.data.corners import LATTICE_CANONICAL, corners_to_array
from chessvision.geometry import (
    FILES,
    canonical_to_image,
    compute_homography,
    image_to_canonical,
    square_center_uv,
    uv_to_square,
)

SQUARES = [f"{f}{r}" for r in range(1, 9) for f in FILES]


def is_board_ok(model, rgb, gt_corners, device) -> bool:
    pred = corners_from_lattice(model, rgb, device=device, use_conf=True)
    h_gt = compute_homography(gt_corners)
    h_pred = compute_homography(pred)
    centers = np.array([square_center_uv(s) for s in SQUARES], dtype=np.float32)
    img_pts = canonical_to_image(h_gt, centers)
    uv_pred = image_to_canonical(h_pred, img_pts)
    return all(
        uv_to_square(float(u), float(v)) == sq for (u, v), sq in zip(uv_pred, SQUARES, strict=True)
    )


def heatmaps_for(model, rgb, device):
    size = getattr(model, "image_size", 512)
    inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1).float().div_(255.0)
    with torch.no_grad():
        h = model.heatmaps(tensor.unsqueeze(0).to(device))  # (1, 81, H, W)
        _, k, hh, ww = h.shape
        prob = h.reshape(1, k, hh * ww).softmax(dim=-1).reshape(1, k, hh, ww)
    return inp, prob[0].cpu().numpy()  # (81, H, W)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/corners/best.pt")
    p.add_argument("--corners-root", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/corners/viz_lattice_channels.png")
    p.add_argument("--max-tries", type=int, default=20)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_corner_regressor(args.ckpt, device=device)

    store = CornerStore(args.corners_root)
    _, heldout = select_corner_dataset_poses(store)

    chosen = None
    for s in heldout[: args.max_tries]:
        bgr = cv2.imread(str(store.store / s.image), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = {k: list(v) for k, v in s.corners.items()}
        if is_board_ok(model, rgb, gt, device):
            chosen = (s, rgb, gt)
            print(f"chose {s.image} (board={s.board})")
            break
    if chosen is None:
        print("no board-ok image found in first", args.max_tries, "heldout samples")
        return 1

    s, rgb, gt = chosen
    inp, prob = heatmaps_for(model, rgb, device)  # inp 512x512x3, prob (81, ph, pw)
    h0, w0 = rgb.shape[:2]
    ph, pw = prob.shape[1:]

    # GT corners are stored in ORIGINAL-image pixels. Project the canonical lattice
    # into original-image pixels, then scale to heatmap-pixel coords.
    gt_corners4 = corners_to_array(gt)  # (4, 2) in original-image px
    h_gt = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
        gt_corners4[[0, 1, 3, 2]].astype(np.float32),  # TL,TR,BR,BL -> TL,TR,BL,BR
    )
    gt_lattice_rgb = cv2.perspectiveTransform(LATTICE_CANONICAL.reshape(-1, 1, 2), h_gt).reshape(
        -1, 2
    )
    gt_lattice_hm = gt_lattice_rgb * np.array([pw / w0, ph / h0], dtype=np.float32)

    # Per-channel soft-argmax in [0, 1] -> heatmap-pixel coords.
    xs = np.linspace(0.0, 1.0, pw)
    ys = np.linspace(0.0, 1.0, ph)
    pred_x = (prob.sum(axis=1) * xs).sum(axis=-1) * pw
    pred_y = (prob.sum(axis=2) * ys).sum(axis=-1) * ph
    pred_lattice_hm = np.stack([pred_x, pred_y], axis=-1)

    inp_small = cv2.resize(inp, (pw, ph), interpolation=cv2.INTER_AREA)
    fig, axes = plt.subplots(9, 9, figsize=(20, 20))
    fig.suptitle(
        f"81-channel softmax heatmaps — {s.image} (board-ok)\n"
        f"green = soft-argmax prediction · red = GT lattice point  (heatmap {pw}x{ph})",
        fontsize=12,
    )
    for j in range(9):
        for i in range(9):
            k = j * 9 + i
            ax = axes[j, i]
            ax.imshow(inp_small, alpha=0.4)
            ax.imshow(prob[k], cmap="hot", alpha=0.6)
            gx, gy = gt_lattice_hm[k]
            px, py = pred_lattice_hm[k]
            ax.scatter([gx], [gy], c="red", s=18, marker="x", linewidths=1.5)
            ax.scatter([px], [py], c="lime", s=12, marker="o", edgecolors="black", linewidths=0.4)
            ax.set_xlim(0, pw - 1)
            ax.set_ylim(ph - 1, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({i},{j})", fontsize=6, pad=1)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")

    # Context: source frame with predicted vs GT lattice.
    side = out.with_name(out.stem + "_context.png")
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    ax2.imshow(rgb)
    ax2.scatter(gt_lattice_rgb[:, 0], gt_lattice_rgb[:, 1], c="red", s=10, marker="x", label="GT")
    pred_rgb = pred_lattice_hm * np.array([w0 / pw, h0 / ph], dtype=np.float32)
    ax2.scatter(pred_rgb[:, 0], pred_rgb[:, 1], c="lime", s=6, label="pred (soft-argmax)")
    ax2.set_title(s.image)
    ax2.legend(loc="upper right")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig2.savefig(side, dpi=120, bbox_inches="tight")
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
