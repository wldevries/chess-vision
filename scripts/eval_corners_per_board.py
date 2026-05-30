"""Eval the corner localizer on the standalone corner dataset, per board, on ALL frames.

Unlike `eval_cds_per_board.py` (which scores only the held-out pose split), this scores
*every* labelled frame of each board. That's the right lens for a board the deployed
checkpoint never trained on (e.g. a freshly-added board): all its frames are
out-of-sample, so the full set is the honest unseen-board number. For boards that WERE
in training, the all-frames number is optimistic (it includes train poses) -- use
`eval_cds_per_board.py` for those; here they're only a sanity reference.

Handles both the 81-point lattice checkpoint (default deployed) and the 4-corner model:
when the loaded model predicts 81 points, the labelled 4 corners are expanded to the
9x9 lattice (same as training) so pred/target shapes match.

    uv run python scripts/eval_corners_per_board.py --board dennis-bord
    uv run python scripts/eval_corners_per_board.py            # every board
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from chessvision.corner_regressor import load_corner_regressor
from chessvision.data.corner_capture import CornerCaptureDataset, CornerStore
from chessvision.data.corners import CornerConfig, LatticeTargets, collate_corners


@torch.no_grad()
def eval_samples(model, samples, store, cfg, device, lattice: bool):
    ds = CornerCaptureDataset(samples, store, cfg, train=False)
    if lattice:
        ds = LatticeTargets(ds)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_corners)
    total_err = total_worst = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device)
        corners = targets["corners"].to(device)
        pred = model(images)
        dist = torch.linalg.vector_norm(pred - corners, dim=-1)  # (B, N) per-point L2
        total_err += dist.mean(dim=1).sum().item()
        total_worst += dist.max(dim=1).values.sum().item()
        n += images.shape[0]
    return total_err / max(n, 1), total_worst / max(n, 1), n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/corners/best.pt")
    p.add_argument("--corners-root", default="data")
    p.add_argument("--board", default=None, help="restrict to one board tag (default: all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_corner_regressor(args.ckpt, device=device)
    model.eval()
    n_pts = int(getattr(model, "num_corners", 0)) or sum(
        1 for _ in model(torch.zeros(1, 3, model.image_size, model.image_size, device=device))[0]
    )
    lattice = n_pts == 81
    cfg = CornerConfig(image_size=getattr(model, "image_size", 512), cache=False)

    store = CornerStore(args.corners_root)
    samples = store.samples()
    by_board: dict[str, list] = defaultdict(list)
    for s in samples:
        board = s.board or "(untagged)"
        if args.board and board != args.board:
            continue
        by_board[board].append(s)

    print(f"ckpt={args.ckpt}  pred_points={n_pts} ({'lattice' if lattice else '4-corner'})")
    print("metric = mean per-point L2 error as fraction of image size (lower better)\n")
    for board in sorted(by_board):
        # representative long side for a rough px figure
        longs = [max(s.width, s.height) for s in by_board[board]]
        long_side = sorted(longs)[len(longs) // 2]
        m, w, n = eval_samples(model, by_board[board], store, cfg, device, lattice)
        print(
            f"  {board:16s}: mean={m:.5f} ({m * long_side:5.1f}px)  "
            f"worst={w:.5f} ({w * long_side:5.1f}px)  (n={n}, ~{long_side}px long side)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
