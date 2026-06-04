"""Train the RT-DETRv2 piece detector on ChessReD chessred2k (Phase 2, alternative).

Fine-tunes a COCO-pretrained RT-DETRv2 (ResNet-18 backbone by default) on the same official
chessred2k train/val split as the Faster R-CNN baseline, and reports COCO mAP on val with the
*same* torchmetrics metric -- so the number is directly comparable to the FRCNN (best val mAP
0.864) and the YOLO baseline. The Apache-2.0 license is the reason this exists: an end-to-end
transformer detector with no Ultralytics AGPL strings attached.

    uv sync --group rtdetr
    uv run --group rtdetr python scripts/train_rtdetr.py \
        --data-root "data/othersets/ChessReD" \
        --epochs 40 --batch-size 4 --device cuda --amp --board-crop

Pass `--store data` to **mix in the user's capture store** (their own boards: staunton, cheap,
rimless, ...) alongside ChessReD, the way `train_keypoint_joint.py` does for the keypoint head
-- this is the lever for generalization across boards (the project's whole point), not just
fitting ChessReD. A `WeightedRandomSampler` rebalances the ~3:1 ChessReD:store imbalance to
`--mix` (0.5 = balanced). The store's piece boxes are *synthesized* cylinder RoIs around each
hand-labelled contact point (see capture_detection); they teach piece appearance/location on
the user's boards, but the store mAP is vs those synthetic boxes, so read it as a detection
proxy, not a tight-localization score. `--test-boards` (e.g. dennis-bord) is held out as an
unseen-board TEST, reported but never used for checkpoint selection.

    uv run --group rtdetr python scripts/train_rtdetr.py \
        --data-root "data/othersets/ChessReD" --store data --test-boards dennis-bord \
        --epochs 40 --batch-size 4 --device cuda --amp --board-crop

LR schedule is constant by default (the step schedule overfit the first run); `--patience`
early-stops once the selection mAP plateaus, always keeping the best checkpoint. `--board-crop`
crops to the board before the 640 resize to recover the resolution lost vs the FRCNN@800 /
YOLO@1280 baselines. Selection metric is store val mAP when `--store` is set, else ChessReD val.

mAP needs torchmetrics + pycocotools (already core deps). The model + processor are saved in
HF-native layout under --out-dir/best (a directory, not a .pt); load with rtdetr.load_rtdetr.

This script's entry point is guarded by `if __name__ == "__main__"`, required for DataLoader
workers under Windows spawn.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from chessvision.data.capture_detection import CaptureKeypointConfig, CaptureKeypointDetection
from chessvision.data.chessred import ChessReD
from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.detection import ChessReDDetection, DetectionConfig
from chessvision.data.positions import store_label_to_capture
from chessvision.rtdetr import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE_SIZE,
    RTDetrCollate,
    build_processor,
    build_rtdetr,
    evaluate_map,
    save_rtdetr,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None, help="image tree root override")
    add("--checkpoint", default=DEFAULT_CHECKPOINT, help="pretrained RT-DETRv2 to fine-tune")
    add("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="square input size (640 best)")
    add("--epochs", type=int, default=50)
    add("--batch-size", type=int, default=4)
    add("--lr", type=float, default=1e-4, help="AdamW lr for the transformer + heads")
    add("--backbone-lr", type=float, default=1e-5, help="lower AdamW lr for the backbone")
    add("--weight-decay", type=float, default=1e-4)
    add(
        "--lr-schedule",
        choices=("constant", "cosine", "step"),
        default="constant",
        help="constant is the DETR-family default; the step schedule overfit on the first run",
    )
    add(
        "--patience",
        type=int,
        default=10,
        help="early-stop after N evals without a val-mAP gain (0 disables); keeps best regardless",
    )
    add(
        "--board-crop",
        action="store_true",
        help="crop to the board bbox before the 640 resize so pieces fill the frame",
    )
    # Capture-store mixing (generalization across the user's own boards). Off unless --store set.
    add("--store", type=Path, default=None, help="capture store root (e.g. data) to mix in")
    add("--test-boards", default="dennis-bord", help="comma-sep boards held out as store TEST")
    add("--val-pose-frac", type=float, default=0.25, help="store: share of each board's poses->val")
    add("--dedup-thr", type=float, default=0.02, help="store: pose-cluster dist (frac img)")
    add("--mix", type=float, default=0.5, help="target fraction of each epoch from the STORE")
    add(
        "--epoch-size",
        type=int,
        default=None,
        help="samples drawn per epoch (default: dataset sum)",
    )
    add("--max-size", type=int, default=1333, help="dataset long-side cap before the 640 resize")
    add("--hflip", type=float, default=0.5, help="train horizontal-flip probability")
    add("--jitter", type=float, default=0.1, help="train brightness/contrast jitter magnitude")
    add("--color", type=float, default=0.0, help="per-channel gain (white-balance drift)")
    add("--noise", type=float, default=0.0, help="additive Gaussian noise (fraction of 255)")
    add("--workers", type=int, default=4, help="DataLoader workers (0 = main process)")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true", help="mixed precision (CUDA only)")
    add("--grad-clip", type=float, default=0.1, help="max grad norm (DETR-family default 0.1)")
    add("--limit-train", type=int, default=None, help="cap train images (smoke tests)")
    add("--eval-every", type=int, default=1, help="run val mAP every N epochs")
    add("--out-dir", type=Path, default=Path("runs/rtdetr"))
    return p.parse_args(argv)


def make_sampler(n_chessred: int, n_store: int, mix: float, epoch_size: int | None):
    """WeightedRandomSampler over ConcatDataset([chessred, store]) so a fraction `mix` of each
    epoch's draws come from the store -- rebalances the ~3:1 ChessReD:store imbalance. Per-sample
    weight = group_target / group_size, so within a group every sample is equiprobable."""
    weights = [(1.0 - mix) / max(n_chessred, 1)] * n_chessred + [mix / max(n_store, 1)] * n_store
    return WeightedRandomSampler(
        weights, num_samples=epoch_size or (n_chessred + n_store), replacement=True
    )


def build_datasets(args: argparse.Namespace, chessred: ChessReD):
    """Returns (train_dataset, sampler_or_None, eval_sets, select_name). With --store, the train
    set is ConcatDataset([ChessReD, store]) drawn by a mix sampler and selection is on store val;
    otherwise it's ChessReD-only with shuffle and selection on ChessReD val. Eval cfgs must use
    the same board_crop as train, else the frames differ."""
    aug = dict(color=args.color, noise=args.noise, board_crop=args.board_crop)
    cr_train_cfg = DetectionConfig(
        max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter, **aug
    )
    cr_eval_cfg = DetectionConfig(max_size=args.max_size, board_crop=args.board_crop)
    cr_train = ChessReDDetection.from_split(chessred, "train", config=cr_train_cfg)
    if args.limit_train:
        cr_train.image_ids = cr_train.image_ids[: args.limit_train]
    cr_val = ChessReDDetection.from_split(chessred, "val", config=cr_eval_cfg)

    if not args.store:
        return cr_train, None, {"chessred_val": cr_val}, "chessred_val"

    store = CornerStore(args.store)
    test_boards = [b for b in args.test_boards.split(",") if b]
    tr, va, te = split_store_for_keypoints(
        store, test_boards=test_boards, val_pose_frac=args.val_pose_frac, dedup_thr=args.dedup_thr
    )
    to_cap = lambda labels: [store_label_to_capture(lb, store) for lb in labels]  # noqa: E731
    st_train_cfg = CaptureKeypointConfig(
        max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter, **aug
    )
    st_eval_cfg = CaptureKeypointConfig(max_size=args.max_size, board_crop=args.board_crop)
    store_train = CaptureKeypointDetection(to_cap(tr), None, st_train_cfg, train=True)
    store_val = CaptureKeypointDetection(to_cap(va), None, st_eval_cfg, train=False)
    store_test = (
        CaptureKeypointDetection(to_cap(te), None, st_eval_cfg, train=False) if te else None
    )

    sampler = make_sampler(len(cr_train), len(store_train), args.mix, args.epoch_size)
    evals = {"chessred_val": cr_val, "store_val": store_val}
    if store_test is not None:
        evals["store_test"] = store_test
    print(
        f"store: train {len(store_train)} | val {len(store_val)} | "
        f"test {len(store_test) if store_test else 0} (test boards: {test_boards})",
        flush=True,
    )
    return ConcatDataset([cr_train, store_train]), sampler, evals, "store_val"


def train_one_epoch(model, loader, optimizer, device, scaler, grad_clip, epoch) -> float:
    model.train()
    running = 0.0
    t0 = time.time()
    for step, enc in enumerate(loader):
        pixel_values = enc["pixel_values"].to(device, non_blocking=True)
        # RT-DETR's processor resizes every image to a fixed square, so it emits no pixel_mask
        # (nothing is padded); the model accepts pixel_mask=None.
        pixel_mask = (
            enc["pixel_mask"].to(device, non_blocking=True) if "pixel_mask" in enc else None
        )
        labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in enc["labels"]]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running += loss.item()
        if step % 20 == 0:
            rate = (step + 1) * pixel_values.size(0) / (time.time() - t0)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f} "
                f"({rate:.1f} img/s)",
                flush=True,
            )
    return running / max(len(loader), 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Quiet HF: drop the startup LOAD REPORT table and the per-save shard/loading progress bars,
    # so the log is just our per-epoch metric JSON. Imported here (not at module top) so --help
    # works without the optional `transformers` dep installed.
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    chessred = ChessReD.load(args.data_root, args.images_root)
    train_ds, sampler, evals, select_name = build_datasets(args, chessred)
    processor = build_processor(args.checkpoint, image_size=args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,  # sampler and shuffle are mutually exclusive
        collate_fn=RTDetrCollate(processor),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    print(
        f"train {len(train_ds)} | select on {select_name} | device {device}",
        flush=True,
    )

    model = build_rtdetr(args.checkpoint, pretrained=True).to(device)
    # Lower lr on the pretrained backbone, full lr on the encoder/decoder/heads (DETR recipe).
    backbone, other = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone if "backbone" in name else other).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": other, "lr": args.lr},
            {"params": backbone, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    # The first run showed the step schedule overfit (val mAP peaked then declined as the lr
    # dropped), so constant is the default; cosine/step stay available for ablations.
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)], gamma=0.1
        )
    else:  # constant
        scheduler = None
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    history = []
    best_map = -1.0
    since_improve = 0  # evals since the last val-mAP gain (drives early stopping)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args.grad_clip, epoch
        )
        if scheduler is not None:
            scheduler.step()
        cur_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "lr": cur_lr}

        stop = False
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Score every eval set (ChessReD val, and store val/test when mixing); select on one.
            metrics = {
                name: evaluate_map(model, processor, ds, device, args.batch_size, args.workers)
                for name, ds in evals.items()
            }
            row["eval"] = {n: {k: round(v, 4) for k, v in m.items()} for n, m in metrics.items()}
            sel = metrics.get(select_name, {}).get("map", -1.0)
            row["sel_map"] = round(sel, 4)
            if sel > best_map:
                best_map = sel
                since_improve = 0
                save_rtdetr(model, processor, args.out_dir / "best")
            else:
                since_improve += 1
                stop = bool(args.patience) and since_improve >= args.patience
        print(json.dumps(row), flush=True)
        history.append(row)
        save_rtdetr(model, processor, args.out_dir / "last")
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stop:
            print(
                f"early stop: no {select_name} mAP gain in {args.patience} evals "
                f"(best {best_map:.4f})",
                flush=True,
            )
            break

    print(
        f"done. best {select_name} mAP: {best_map:.4f}" if best_map >= 0 else "done (mAP skipped)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
