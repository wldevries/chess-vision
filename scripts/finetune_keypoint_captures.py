"""Fine-tune the keypoint detector on the capture photos (Phase 2.5, domain step).

Starts from the ChessReD-trained keypoint model (`runs/keypoint/best.pt`), unfreezes
the box **classifier** + keypoint heads (backbone/FPN/RPN stay frozen), and fine-tunes
on the user's own boards so it classifies their pieces. Split is by **session** across
both physical sets (see captures-two-boards): val holds out whole sessions of each
board to measure generalization to unseen positions without leaking near-duplicate
frames. Best checkpoint is chosen by held-out **class accuracy**. Outputs ->
`runs/keypoint_captures/`; `runs/keypoint/best.pt` and `runs/detector/best.pt` are
read-only.

Usage:
    uv run python scripts/finetune_keypoint_captures.py --device cuda --amp --epochs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessvision.capture_eval import evaluate_captures
from chessvision.data.capture_detection import CaptureKeypointConfig, split_by_sessions
from chessvision.data.captures import CaptureDataset
from chessvision.data.detection import collate_detection
from chessvision.keypoint_detector import (
    FINETUNE_SCOPES,
    graft_from_detector_checkpoint,
    load_keypoint_detector,
    save_keypoint_checkpoint,
    set_finetune_scope,
    trainable_parameters,
)

# Default held-out sessions: one Staunton session + two cheap-set sessions, so val
# covers BOTH boards (see captures-two-boards). Everything else trains.
DEFAULT_VAL_SESSIONS = "20260524-153712,20260524-230621,20260524-231529"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint/best.pt"))
    add("--detector-ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--val-sessions", default=DEFAULT_VAL_SESSIONS, help="comma-separated held-out sessions")
    add("--unfreeze", choices=FINETUNE_SCOPES, default="classifier", help="how much to adapt")
    add("--optimizer", choices=("adamw", "sgd"), default="adamw")
    add("--epochs", type=int, default=5)
    add("--batch-size", type=int, default=2)
    add("--lr", type=float, default=1e-4, help="AdamW: ~1e-4; SGD: raise to ~5e-3")
    add("--momentum", type=float, default=0.9, help="SGD only")
    add("--weight-decay", type=float, default=1e-2)
    add("--max-size", type=int, default=1333)
    add("--hflip", type=float, default=0.5)
    add("--jitter", type=float, default=0.1)
    add("--workers", type=int, default=4)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true")
    add("--score-thresh", type=float, default=0.5)
    add("--out-dir", type=Path, default=Path("runs/keypoint_captures"))
    return p.parse_args(argv)


def load_model(args, device):
    """Resume the ChessReD keypoint model if present, else graft a fresh head."""
    if args.keypoint_ckpt.exists():
        print(f"resuming keypoint model from {args.keypoint_ckpt}")
        return load_keypoint_detector(args.keypoint_ckpt, device)
    print(f"{args.keypoint_ckpt} absent; grafting head onto {args.detector_ckpt}")
    return graft_from_detector_checkpoint(args.detector_ckpt, device)


def train_one_epoch(model, loader, optimizer, device, scaler, epoch) -> float:
    model.train()
    running, t0 = 0.0, time.time()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())  # class + box + keypoint (+ frozen rpn, no grad)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.item())
        if step % 20 == 0:
            rate = (step + 1) * len(images) / (time.time() - t0)
            parts = {k: round(float(v.item()), 3) for k, v in loss_dict.items()}
            print(
                f"  epoch {epoch} step {step}/{len(loader)} {parts} ({rate:.1f} img/s)", flush=True
            )
    return running / max(len(loader), 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = CaptureDataset.load(args.captures)
    cfg = CaptureKeypointConfig(max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter)
    val_sessions = [s for s in args.val_sessions.split(",") if s]
    train_ds, val_ds = split_by_sessions(dataset, val_sessions, cfg)
    print(
        f"train {len(train_ds)} frames | val {len(val_ds)} frames "
        f"(held-out sessions: {val_sessions})"
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_detection,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = load_model(args, device)
    set_finetune_scope(model, args.unfreeze)
    n_train = sum(p.numel() for p in trainable_parameters(model))
    print(f"unfreeze={args.unfreeze} | optimizer={args.optimizer} | trainable params {n_train:,}")
    params = trainable_parameters(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)], gamma=0.1
    )
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # Baseline: held-out accuracy of the pretrained model before any fine-tuning.
    base = evaluate_captures(
        model,
        val_ds.samples,
        dataset.s3,
        device,
        max_size=args.max_size,
        score_thresh=args.score_thresh,
    )
    print("baseline (pretrained) val:", json.dumps(base), flush=True)

    history, best = [{"epoch": 0, "val": base}], base["class_acc"]
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, epoch)
        scheduler.step()
        val = evaluate_captures(
            model,
            val_ds.samples,
            dataset.s3,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
        )
        row = {"epoch": epoch, "loss": round(loss, 4), "lr": scheduler.get_last_lr()[0], "val": val}
        if val["class_acc"] > best:
            best = val["class_acc"]
            save_keypoint_checkpoint(model, args.out_dir / "best.pt", epoch=epoch, val=val)
        print(json.dumps(row), flush=True)
        history.append(row)
        save_keypoint_checkpoint(model, args.out_dir / "last.pt", epoch=epoch)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"done. baseline class_acc {base['class_acc']:.4f} -> best {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
