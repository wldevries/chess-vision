"""Train the board-contact keypoint head on the frozen detector trunk (Phase 2.5).

Grafts a 1-keypoint head onto `runs/detector/best.pt`, freezes the trunk, and
trains ONLY the keypoint branch on chessred2k contact-point targets. The detector
checkpoint is never modified; outputs go to `runs/keypoint/`. Best checkpoint is
chosen by val keypoint square-accuracy (predicted contact point -> square).

Usage:
    uv run python scripts/train_keypoint_head.py \
        --data-root "data/othersets/ChessReD" \
        --epochs 8 --batch-size 2 --device cuda --amp
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
    ChessReDKeypointDetection,
    DetectionConfig,
    collate_detection,
)
from chessvision.keypoint_detector import (
    freeze_trunk,
    graft_from_detector_checkpoint,
    keypoint_parameters,
    save_keypoint_checkpoint,
)
from chessvision.keypoint_eval import evaluate_squares, rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None)
    add("--detector-ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--epochs", type=int, default=8)
    add("--batch-size", type=int, default=2)
    add("--lr", type=float, default=0.01)
    add("--momentum", type=float, default=0.9)
    add("--weight-decay", type=float, default=1e-4)
    add("--max-size", type=int, default=1333)
    add("--hflip", type=float, default=0.5)
    add("--jitter", type=float, default=0.1)
    add("--workers", type=int, default=4)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true")
    add("--limit-train", type=int, default=None)
    add("--eval-every", type=int, default=1)
    add("--eval-limit", type=int, default=None, help="cap val images per eval (speed)")
    add("--out-dir", type=Path, default=Path("runs/keypoint"))
    return p.parse_args(argv)


def build_loader(args, chessred):
    cfg = DetectionConfig(max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter)
    train_ds = ChessReDKeypointDetection.from_split(chessred, "train", config=cfg)
    if args.limit_train:
        train_ds.image_ids = train_ds.image_ids[: args.limit_train]
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_detection,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )


def train_one_epoch(model, loader, optimizer, device, scaler, epoch) -> float:
    model.train()
    running, t0 = 0.0, time.time()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = model(images, targets)["loss_keypoint"]  # trunk is frozen
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item()
        if step % 50 == 0:
            rate = (step + 1) * len(images) / (time.time() - t0)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} kp_loss {loss.item():.3f} "
                f"({rate:.1f} img/s)",
                flush=True,
            )
    return running / max(len(loader), 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chessred = ChessReD.load(args.data_root, args.images_root)
    loader = build_loader(args, chessred)
    val_ids = chessred.chessred2k_split("val")
    if args.eval_limit:
        val_ids = val_ids[: args.eval_limit]

    model = graft_from_detector_checkpoint(args.detector_ckpt, device)
    freeze_trunk(model)
    optimizer = torch.optim.SGD(
        keypoint_parameters(model),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)], gamma=0.1
    )
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    print(
        f"train {len(loader.dataset)} | val {len(val_ids)} | device {device} | "
        f"trainable keypoint tensors {len(keypoint_parameters(model))}"
    )

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, epoch)
        scheduler.step()
        row = {"epoch": epoch, "kp_loss": round(loss, 4), "lr": scheduler.get_last_lr()[0]}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            counts = evaluate_squares(model, chessred, val_ids, device)
            row["val"] = {k: rates(v) for k, v in counts.items()}
            kp_acc = row["val"]["overall"]["kp_square_acc"] or -1.0
            if kp_acc > best:
                best = kp_acc
                save_keypoint_checkpoint(
                    model, args.out_dir / "best.pt", epoch=epoch, val=row["val"]
                )
        print(json.dumps(row), flush=True)
        history.append(row)
        save_keypoint_checkpoint(model, args.out_dir / "last.pt", epoch=epoch)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"done. best val kp square-acc: {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
