"""One-off: break the corner-dataset (cds) held-out eval down per board.

Mirrors train_corner_regressor's cds split (select_corner_dataset_poses with the
same defaults) and runs the same mean/worst corner-error metric, but reports it
per `board` tag so we can see staunton vs rimless separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from chessvision.corner_regressor import load_corner_regressor
from chessvision.data.corner_capture import (
    CornerCaptureDataset,
    CornerStore,
    select_corner_dataset_poses,
)
from chessvision.data.corners import CornerConfig


@torch.no_grad()
def eval_samples(model, samples, store, cfg, device):
    ds = CornerCaptureDataset(samples, store, cfg, train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    total_err = total_worst = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        corners = targets["corners"].to(device)
        pred = model(images)
        dist = torch.linalg.vector_norm(pred - corners, dim=-1)
        total_err += dist.mean(dim=1).sum().item()
        total_worst += dist.max(dim=1).values.sum().item()
        n += images.shape[0]
    return total_err / max(n, 1), total_worst / max(n, 1), n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/corners/best.pt")
    p.add_argument("--corners-root", default="data/corners")
    p.add_argument("--dedup-thr", type=float, default=0.02)
    p.add_argument("--max-per-pose", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_corner_regressor(args.ckpt, device=device)
    cfg = CornerConfig(image_size=getattr(model, "image_size", 384), cache=False)

    store = CornerStore(args.corners_root)
    _, cds_heldout = select_corner_dataset_poses(
        store,
        dedup_thr=args.dedup_thr,
        max_per_pose=args.max_per_pose,
        val_frac=args.val_frac,
    )

    by_board: dict[str, list] = defaultdict(list)
    by_board_orient: dict[str, list] = defaultdict(list)
    for s in cds_heldout:
        board = s.board or "(untagged)"
        orient = "portrait" if s.width < s.height else "landscape"
        by_board[board].append(s)
        by_board_orient[f"{board} [{orient}]"].append(s)

    print(f"ckpt={args.ckpt}  held-out frames={len(cds_heldout)}")
    mean_all, worst_all, n_all = eval_samples(model, cds_heldout, store, cfg, device)
    print(f"  ALL boards : mean={mean_all:.5f} worst={worst_all:.5f} (n={n_all})")
    for board in sorted(by_board):
        m, w, n = eval_samples(model, by_board[board], store, cfg, device)
        print(f"  {board:22s}: mean={m:.5f} worst={w:.5f} (n={n})")

    # only show the orientation split where a board actually has both
    boards_with_both = {
        k.rsplit(" [", 1)[0] for k in by_board_orient
    }
    mixed = {b for b in boards_with_both if sum(b == kk.rsplit(" [", 1)[0] for kk in by_board_orient) > 1}
    if mixed:
        print("  -- orientation split (boards with both) --")
        for key in sorted(by_board_orient):
            if key.rsplit(" [", 1)[0] in mixed:
                m, w, n = eval_samples(model, by_board_orient[key], store, cfg, device)
                print(f"  {key:22s}: mean={m:.5f} worst={w:.5f} (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
