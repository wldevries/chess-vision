"""Train the Approach-A piece detector on ChessReD chessred2k (Phase 2).

Fine-tunes a COCO-pretrained Faster R-CNN (ResNet50-FPN) on the 1,442/330/306
chessred2k train/val/test split and reports COCO mAP on the val split.

Usage:
    uv run python scripts/train_detector.py \
        --data-root "data/Chess Recognition Dataset (ChessReD)_2_all" \
        --epochs 12 --batch-size 2 --device cuda --amp

mAP needs torchmetrics + pycocotools (`uv add torchmetrics pycocotools`); without
them training still runs and the val loss is reported instead.

This script's entry point is guarded by `if __name__ == "__main__"`, which is
required for DataLoader workers / any process pool on Windows (spawn).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessvision.data.chessred import ChessReD
from chessvision.data.detection import (
    ChessReDDetection,
    DetectionConfig,
    collate_detection,
)
from chessvision.detector import build_detector, save_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None, help="image tree root override")
    add("--epochs", type=int, default=12)
    add("--batch-size", type=int, default=2)
    add("--lr", type=float, default=0.005)
    add("--momentum", type=float, default=0.9)
    add("--weight-decay", type=float, default=5e-4)
    add("--max-size", type=int, default=1333, help="long-side cap; boxes scaled to match")
    add("--hflip", type=float, default=0.5, help="train horizontal-flip probability")
    add("--jitter", type=float, default=0.1, help="train brightness/contrast jitter magnitude")
    add("--workers", type=int, default=4, help="DataLoader workers (0 = main process)")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true", help="mixed precision (CUDA only)")
    add("--limit-train", type=int, default=None, help="cap train images (smoke tests)")
    add("--eval-every", type=int, default=1, help="run val mAP every N epochs")
    add("--out-dir", type=Path, default=Path("runs/detector"))
    return p.parse_args(argv)


def build_loaders(args: argparse.Namespace, chessred: ChessReD):
    train_cfg = DetectionConfig(max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter)
    eval_cfg = DetectionConfig(max_size=args.max_size)

    train_ds = ChessReDDetection.from_split(chessred, "train", config=train_cfg)
    if args.limit_train:
        train_ds.image_ids = train_ds.image_ids[: args.limit_train]
    val_ds = ChessReDDetection.from_split(chessred, "val", config=eval_cfg)

    common = dict(
        collate_fn=collate_detection,
        num_workers=args.workers,
        pin_memory=True,
        # keep workers alive across epochs so the large ChessReD object isn't
        # re-pickled to each worker every epoch (workers=0 can't be persistent)
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device, scaler, epoch) -> float:
    model.train()
    running = 0.0
    t0 = time.time()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item()
        if step % 20 == 0:
            rate = (step + 1) * len(images) / (time.time() - t0)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f} "
                f"({rate:.1f} img/s)",
                flush=True,
            )
    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict | None:
    """COCO mAP via torchmetrics; returns None (skips) if the dep is missing."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ModuleNotFoundError:
        print("  [skip mAP] torchmetrics not installed (uv add torchmetrics pycocotools)")
        return None

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)
    result = metric.compute()
    return {k: float(v) for k, v in result.items() if v.numel() == 1}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chessred = ChessReD.load(args.data_root, args.images_root)
    train_loader, val_loader = build_loaders(args, chessred)
    print(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | device {device}")

    model = build_detector(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)], gamma=0.1
    )
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    history = []
    best_map = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, epoch)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "lr": scheduler.get_last_lr()[0]}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device)
            if metrics:
                row.update({k: round(v, 4) for k, v in metrics.items()})
                if metrics.get("map", -1) > best_map:
                    best_map = metrics["map"]
                    save_checkpoint(model, args.out_dir / "best.pt", epoch=epoch, metrics=metrics)
        print(json.dumps(row), flush=True)
        history.append(row)
        save_checkpoint(model, args.out_dir / "last.pt", epoch=epoch)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"done. best val mAP: {best_map:.4f}" if best_map >= 0 else "done (mAP skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
