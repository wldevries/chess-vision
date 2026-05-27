"""Train the board-corner localizer on ChessReD chessred2k (Phase 3).

Trains a compact heatmap/soft-argmax corner model (`chessvision.corner_regressor`)
on the 1442/330/306 chessred2k corner split and reports mean per-corner error.
Corners are predicted in visual TL/TR/BR/BL slots (orientation stays a manual
toggle downstream -- see `chessvision.data.corners`).

Usage:
    uv run python scripts/train_corner_regressor.py \
        --data-root "data/Chess Recognition Dataset (ChessReD)_2_all" \
        --epochs 40 --batch-size 16 --device cuda --amp

The headline metric is **mean per-corner error** as a fraction of image size
(multiply by the native long side, ~3072 px for ChessReD, for pixels). A follow-up
eval ties this to Phase-1 square-assignment accuracy -- the project-goal number.

The entry point is guarded by `if __name__ == "__main__"` (required for DataLoader
workers / process pools on Windows spawn).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from chessvision.corner_regressor import build_corner_regressor, save_corner_checkpoint
from chessvision.data.chessred import ChessReD
from chessvision.data.corner_capture import (
    CornerCaptureDataset,
    CornerStore,
    select_corner_dataset_poses,
)
from chessvision.data.corners import (
    NUM_CORNERS,
    NUM_LATTICE,
    CaptureCorners,
    ChessReDCorners,
    CornerConfig,
    LatticeTargets,
    _cluster_by_corners,
    collate_corners,
    select_capture_corner_poses,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None, help="image tree root override")
    add("--backbone", default="mobilenet_v3_small", help="mobilenet_v3_small|_large, resnet18, ...")
    add("--image-size", type=int, default=384, help="square network input")
    add("--epochs", type=int, default=40)
    add("--batch-size", type=int, default=16)
    add("--lr", type=float, default=1e-3)
    add("--weight-decay", type=float, default=1e-4)
    add("--hflip", type=float, default=0.5, help="train horizontal-flip probability")
    add("--jitter", type=float, default=0.1, help="train brightness/contrast jitter magnitude")
    # Colour aug -- off by default: a controlled seed-0 sweep (runs/corners_sweep) did
    # NOT show it helping captures and the 16-pose held-out eval is too noisy to trust
    # single-run diffs <~0.005. Re-enable per-flag to experiment (needs multi-seed eval).
    add("--hue", type=float, default=0.0, help="HSV hue jitter, frac of full circle (0 off)")
    add("--saturation", type=float, default=0.0, help="HSV saturation jitter magnitude (0 off)")
    add("--grayscale-prob", type=float, default=0.0, help="prob. of dropping colour (0 off)")
    # Geometric aug -- off by default (same reason); auto-skips samples it would push off-frame.
    add("--rotate", type=float, default=0.0, help="max abs rotation in degrees (0 off)")
    add("--scale", type=float, default=0.0, help="scale jitter magnitude, e.g. 0.1 -> x[0.9,1.1]")
    add("--perspective", type=float, default=0.0, help="perspective jitter, frac of size (0 off)")
    add("--workers", type=int, default=4, help="DataLoader workers (ignored when caching)")
    add(
        "--no-cache",
        action="store_true",
        help="disable the in-RAM image cache (re-decode every epoch; decode-bound)",
    )
    add("--cache-workers", type=int, default=8, help="threads to pre-warm the image cache")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true", help="mixed precision (CUDA only)")
    add("--limit-train", type=int, default=None, help="cap train images (smoke tests)")
    add("--eval-every", type=int, default=1, help="run val every N epochs")
    add("--seed", type=int, default=0, help="RNG seed (model head init, aug draws, shuffle)")
    add(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply ImageNet mean/std normalization inside the model (--no-normalize to disable)",
    )
    add(
        "--lattice",
        action="store_true",
        help="train the 81-point grid lattice instead of 4 corners (predicts every 9x9 "
        "intersection; targets auto-derived from the labelled corners). Reuses the whole "
        "stack -- loss/eval are generic over point count.",
    )
    add("--out-dir", type=Path, default=Path("runs/corners"))
    # Capture set (the user's own boards): added to train for board-appearance variety,
    # deduped to distinct corner poses; held-out poses give an on-your-boards eval.
    add("--captures-export", type=Path, default=Path("data/captures/label-studio.json"))
    add("--no-captures", action="store_true", help="train on ChessReD only (ignore captures)")
    add("--dedup-thr", type=float, default=0.02, help="distinct-pose threshold (frac of img size)")
    add("--max-per-pose", type=int, default=2, help="frames kept per distinct corner pose")
    add("--val-frac", type=float, default=0.25, help="share of each board's poses held out")
    # Standalone corner dataset (phone photos labelled in the app, corner-only). Its
    # held-out poses become the `cds_*` eval and the checkpoint-selection metric when
    # present -- the larger, viewpoint-diverse "works on your boards" number. See
    # chessvision/data/corner_capture.py and corner-capture-mode.md.
    add("--corners-root", type=Path, default=Path("data/corners"), help="corner-dataset root")
    add("--no-corner-ds", action="store_true", help="ignore the standalone corner dataset")
    return p.parse_args(argv)


def build_loaders(args: argparse.Namespace, chessred: ChessReD):
    """Returns (train_loader, val_loader, capture_eval_loader, corner_ds_eval_loader). The
    capture / corner-ds loaders are None when disabled/absent. Train = ChessReD train
    (+ deduped capture poses + corner-dataset poses); val = ChessReD val (the stable,
    diverse metric); capture_eval / corner_ds_eval = held-out poses (the honest 'works on
    your boards' numbers)."""
    cache = not args.no_cache
    train_cfg = CornerConfig(
        image_size=args.image_size,
        hflip_prob=args.hflip,
        jitter=args.jitter,
        hue=args.hue,
        saturation=args.saturation,
        grayscale_prob=args.grayscale_prob,
        rotate=args.rotate,
        scale=args.scale,
        perspective=args.perspective,
        cache=cache,
    )
    eval_cfg = CornerConfig(image_size=args.image_size, cache=cache)

    chess_train = ChessReDCorners.from_split(chessred, "train", config=train_cfg)
    if args.limit_train:
        chess_train.image_ids = chess_train.image_ids[: args.limit_train]
    val_ds = ChessReDCorners.from_split(chessred, "val", config=eval_cfg)

    datasets = [chess_train, val_ds]
    train_parts: list = [chess_train]
    capture_eval_ds = None
    if not args.no_captures and args.captures_export.exists():
        cap_train, cap_heldout = select_capture_corner_poses(
            args.captures_export,
            dedup_thr=args.dedup_thr,
            max_per_pose=args.max_per_pose,
            val_frac=args.val_frac,
        )
        cap_train_ds = CaptureCorners(cap_train, train_cfg, train=True)
        capture_eval_ds = CaptureCorners(cap_heldout, eval_cfg, train=False)
        train_parts.append(cap_train_ds)
        datasets += [cap_train_ds, capture_eval_ds]
        # Report frames AND distinct corner poses: the eval's real sample size is poses
        # (near-duplicate frames of one camera setup aren't independent), so the pose
        # count is what matters -- not the frame count.
        n_train_poses = len(_cluster_by_corners(cap_train, args.dedup_thr))
        n_heldout_poses = len(_cluster_by_corners(cap_heldout, args.dedup_thr))
        print(
            f"captures: +{len(cap_train)} train frames ({n_train_poses} poses) | "
            f"{len(cap_heldout)} held-out frames ({n_heldout_poses} poses)"
        )

    # The standalone corner dataset (phone photos, corner-only). Same pose-deduped
    # train + held-out-by-board split, its own loader so the `cds_*` eval is separate.
    corner_ds_eval_ds = None
    if not args.no_corner_ds:
        store = CornerStore(args.corners_root)
        cds_samples = store.samples()
        if cds_samples:
            cds_train, cds_heldout = select_corner_dataset_poses(
                store,
                dedup_thr=args.dedup_thr,
                max_per_pose=args.max_per_pose,
                val_frac=args.val_frac,
            )
            cds_train_ds = CornerCaptureDataset(cds_train, store, train_cfg, train=True)
            corner_ds_eval_ds = CornerCaptureDataset(cds_heldout, store, eval_cfg, train=False)
            train_parts.append(cds_train_ds)
            datasets += [cds_train_ds, corner_ds_eval_ds]
            n_train_poses = len(_cluster_by_corners(cds_train, args.dedup_thr))
            n_heldout_poses = len(_cluster_by_corners(cds_heldout, args.dedup_thr))
            print(
                f"corner-ds: +{len(cds_train)} train frames ({n_train_poses} poses) | "
                f"{len(cds_heldout)} held-out frames ({n_heldout_poses} poses)"
            )

    train_ds: object = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)

    if cache:
        print(f"pre-warming image cache ({args.cache_workers} threads)...", flush=True)
        for ds in datasets:
            ds.prewarm(max_workers=args.cache_workers)

    # With the in-RAM cache the bottleneck is gone and the cache is per-process, so
    # worker processes would only duplicate memory -- run single-process.
    # Lattice mode: wrap the loader-facing datasets so targets become the 81-point grid.
    # Prewarm already ran on the base datasets above; the wrap just expands 4 -> 81 on read.
    if args.lattice:
        train_ds = LatticeTargets(train_ds)
        val_ds = LatticeTargets(val_ds)
        capture_eval_ds = LatticeTargets(capture_eval_ds) if capture_eval_ds is not None else None
        corner_ds_eval_ds = (
            LatticeTargets(corner_ds_eval_ds) if corner_ds_eval_ds is not None else None
        )

    workers = 0 if cache else args.workers
    common = dict(
        collate_fn=collate_corners,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    capture_eval_loader = (
        DataLoader(capture_eval_ds, batch_size=args.batch_size, shuffle=False, **common)
        if capture_eval_ds is not None
        else None
    )
    corner_ds_eval_loader = (
        DataLoader(corner_ds_eval_ds, batch_size=args.batch_size, shuffle=False, **common)
        if corner_ds_eval_ds is not None
        else None
    )
    return train_loader, val_loader, capture_eval_loader, corner_ds_eval_loader


def train_one_epoch(model, loader, optimizer, device, scaler, epoch) -> float:
    model.train()
    running = 0.0
    t0 = time.time()
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        corners = targets["corners"].to(device, non_blocking=True)  # (B, 4, 2) in [0, 1]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            pred = model(images)
            loss = F.smooth_l1_loss(pred, corners)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item()
        if step % 20 == 0:
            rate = (step + 1) * images.shape[0] / (time.time() - t0)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.5f} "
                f"({rate:.1f} img/s)",
                flush=True,
            )
    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Mean per-corner error in normalized units (fraction of image size).

    `mean_corner_err` is the headline number; `worst_corner_err` is the per-image
    max corner error averaged over the val set (a tail indicator -- one bad corner
    breaks the homography).
    """
    model.eval()
    total_err = 0.0
    total_worst = 0.0
    n_images = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        corners = targets["corners"].to(device, non_blocking=True)
        pred = model(images)
        dist = torch.linalg.vector_norm(pred - corners, dim=-1)  # (B, 4) per-corner L2
        total_err += dist.mean(dim=1).sum().item()
        total_worst += dist.max(dim=1).values.sum().item()
        n_images += images.shape[0]
    return {
        "mean_corner_err": total_err / max(n_images, 1),
        "worst_corner_err": total_worst / max(n_images, 1),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chessred = ChessReD.load(args.data_root, args.images_root)
    train_loader, val_loader, capture_loader, corner_ds_loader = build_loaders(args, chessred)
    print(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | device {device}")

    model = build_corner_regressor(
        backbone=args.backbone,
        pretrained=True,
        num_corners=NUM_LATTICE if args.lattice else NUM_CORNERS,
        normalize=args.normalize,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # Select best.pt on your real boards when present: chessred2k is a single foldable
    # board, so its val is a same-board *memorization* metric that can move opposite to
    # generalization. Prefer the standalone corner dataset (most viewpoint-diverse), then
    # captures, then fall back to ChessReD val only when neither is present.
    select_name = (
        "cds_mean"
        if corner_ds_loader is not None
        else "cap_mean"
        if capture_loader is not None
        else "val_mean"
    )
    history = []
    best_err = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, epoch)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "lr": scheduler.get_last_lr()[0]}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device)
            row.update({k: round(v, 5) for k, v in metrics.items()})
            select_err = metrics["mean_corner_err"]
            if capture_loader is not None:
                cap = evaluate(model, capture_loader, device)
                row.update({f"cap_{k}": round(v, 5) for k, v in cap.items()})
                select_err = cap["mean_corner_err"]
            if corner_ds_loader is not None:
                cds = evaluate(model, corner_ds_loader, device)
                row.update({f"cds_{k}": round(v, 5) for k, v in cds.items()})
                select_err = cds["mean_corner_err"]
            if select_err < best_err:
                best_err = select_err
                save_corner_checkpoint(
                    model,
                    args.out_dir / "best.pt",
                    image_size=args.image_size,
                    epoch=epoch,
                    metrics=row,
                )
        print(json.dumps(row), flush=True)
        history.append(row)
        save_corner_checkpoint(
            model, args.out_dir / "last.pt", image_size=args.image_size, epoch=epoch
        )
        # JSONL: one compact object per epoch (DuckDB: read_json_auto('history.jsonl')).
        jsonl = "\n".join(json.dumps(r) for r in history) + "\n"
        (args.out_dir / "history.jsonl").write_text(jsonl, encoding="utf-8")

    print(f"done. best {select_name} corner err: {best_err:.5f} (fraction of image size)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
