"""Joint ChessReD + capture-store keypoint training (experiment: mix, don't pretrain->finetune).

Instead of the sequential pipeline (box detector on ChessReD -> keypoint head on ChessReD ->
fine-tune on the store), this trains ONE model on a **combination** of both datasets in a
single stage. It starts from the torchvision **COCO source** Faster R-CNN v2 (not our own
`runs/detector/best.pt`): ChessReD is already in the training mix here, so initializing from a
ChessReD-baked checkpoint would double-count it and bias the init toward the ChessReD domain.
The 12-class box predictor + the 1-keypoint contact head are fresh; everything is trainable.

Both datasets already emit the identical `(image, target)` shape -- same labels 1..12, same
`keypoints (N,1,3)`, same `collate_detection` -- so they concatenate with no remapping. ChessReD
outnumbers the store ~3:1, so a `WeightedRandomSampler` rebalances each epoch to a target store
fraction (`--mix`, default 0.5 = balanced) -- the lever that keeps the model from drifting back
to the single ChessReD appearance domain (the project's explicit failure mode).

Eval each epoch reports BOTH worlds: store val **class_acc** (the deployment metric, and the
checkpoint-selection signal -- comparable to the current `runs/keypoint_captures` number),
ChessReD val keypoint square-acc (sanity that ChessReD didn't regress), and store **test**
(held-out whole boards, e.g. dennis -- an honest unseen-board number selection never peeks at).

Usage:
    uv run python scripts/train_keypoint_joint.py \
        --data-root "data/Chess Recognition Dataset (ChessReD)_2_all" \
        --store data --test-boards dennis-bord \
        --device cuda --amp --epochs 12 --mix 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from chessvision.capture_eval import evaluate_captures
from chessvision.data.capture_detection import CaptureKeypointConfig, CaptureKeypointDetection
from chessvision.data.chessred import ChessReD
from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.detection import (
    ChessReDKeypointDetection,
    DetectionConfig,
    collate_detection,
)
from chessvision.data.positions import store_label_to_capture
from chessvision.keypoint_detector import (
    build_keypoint_detector,
    save_keypoint_checkpoint,
    trainable_parameters,
)
from chessvision.keypoint_eval import evaluate_squares, rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None)
    add("--store", type=Path, default=Path("data"), help="unified corner store root (flat layout)")
    add(
        "--test-boards", default="dennis-bord", help="comma-separated boards held out as store TEST"
    )
    add(
        "--val-pose-frac",
        type=float,
        default=0.25,
        help="store: share of each board's poses -> val",
    )
    add("--dedup-thr", type=float, default=0.02, help="store: pose-cluster dist (frac img)")
    add(
        "--mix",
        type=float,
        default=0.5,
        help="target fraction of each epoch drawn from the STORE (0.5 = balanced 50/50; "
        "1.0 = store only; the rest is ChessReD). Rebalances the ~3:1 ChessReD:store imbalance.",
    )
    add(
        "--epoch-size",
        type=int,
        default=None,
        help="samples drawn per epoch (default: dataset sum)",
    )
    add("--epochs", type=int, default=12)
    add("--batch-size", type=int, default=2)
    add("--optimizer", choices=("sgd", "adamw"), default="sgd")
    add("--lr", type=float, default=0.005, help="SGD: ~0.005 (full train from COCO); AdamW: ~1e-4")
    add("--momentum", type=float, default=0.9, help="SGD only")
    add("--weight-decay", type=float, default=5e-4)
    add("--max-size", type=int, default=1333)
    add("--hflip", type=float, default=0.5)
    add("--jitter", type=float, default=0.1)
    # Appearance-only aug (image alone), applied to BOTH domains' train sets. Default off.
    add("--aug-color", type=float, default=0.0, help="per-channel gain / white-balance (e.g. 0.1)")
    add("--aug-blur", type=float, default=0.0, help="max Gaussian blur sigma px (e.g. 1.0)")
    add("--aug-motion-blur", type=float, default=0.0, help="max motion-blur length px (e.g. 5)")
    add("--aug-noise", type=float, default=0.0, help="max noise std /255, low-light (e.g. 0.03)")
    # Board crop: slice both domains to the board bbox (+margin); recorded in the .pt for eval
    add("--board-crop", action="store_true", help="train on board-sliced images (eval must match)")
    add("--crop-side", type=float, default=0.12, help="board-crop side margin (frac of bbox)")
    add("--crop-top", type=float, default=0.30, help="board-crop top margin (headroom)")
    add("--crop-bottom", type=float, default=0.08, help="board-crop bottom margin")
    add("--workers", type=int, default=4)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true")
    add("--score-thresh", type=float, default=0.5)
    add("--eval-chessred-limit", type=int, default=None, help="cap ChessReD val images per eval")
    add("--out-dir", type=Path, default=Path("runs/keypoint_joint"))
    return p.parse_args(argv)


def build_datasets(args):
    """Build the ChessReD-train and store-(train/val/test) keypoint datasets + the ChessReD
    val image ids. Train configs augment; eval configs do not."""
    aug = dict(
        color=args.aug_color,
        blur=args.aug_blur,
        motion_blur=args.aug_motion_blur,
        noise=args.aug_noise,
        board_crop=args.board_crop,  # slice both domains to the board; eval must match
        crop_side=args.crop_side,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
    )
    cfg_train = CaptureKeypointConfig(
        max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter, **aug
    )
    cr_train_cfg = DetectionConfig(
        max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter, **aug
    )

    chessred = ChessReD.load(args.data_root, args.images_root)
    cr_train = ChessReDKeypointDetection.from_split(chessred, "train", config=cr_train_cfg)
    cr_val_ids = chessred.chessred2k_split("val")
    if args.eval_chessred_limit:
        cr_val_ids = cr_val_ids[: args.eval_chessred_limit]

    store = CornerStore(args.store)
    test_boards = [b for b in args.test_boards.split(",") if b]
    tr, va, te = split_store_for_keypoints(
        store, test_boards=test_boards, val_pose_frac=args.val_pose_frac, dedup_thr=args.dedup_thr
    )
    to_cap = lambda labels: [store_label_to_capture(lb, store) for lb in labels]  # noqa: E731
    store_train = CaptureKeypointDetection(to_cap(tr), None, cfg_train, train=True)
    store_val = CaptureKeypointDetection(to_cap(va), None, cfg_train, train=False)
    store_test = CaptureKeypointDetection(to_cap(te), None, cfg_train, train=False) if te else None
    return chessred, cr_train, cr_val_ids, store_train, store_val, store_test, test_boards


def make_sampler(n_chessred: int, n_store: int, mix: float, epoch_size: int | None):
    """WeightedRandomSampler over ConcatDataset([chessred, store]) so a fraction `mix` of each
    epoch's draws come from the store. Per-sample weight = group_target / group_size, so within
    a group every sample is equiprobable and the two groups hit their target mass."""
    w_chessred = (1.0 - mix) / max(n_chessred, 1)
    w_store = mix / max(n_store, 1)
    weights = [w_chessred] * n_chessred + [w_store] * n_store
    num_samples = epoch_size or (n_chessred + n_store)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def train_one_epoch(model, loader, optimizer, device, scaler, epoch) -> float:
    model.train()
    running, t0 = 0.0, time.time()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())  # class + box + rpn + keypoint, all trainable
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

    chessred, cr_train, cr_val_ids, store_train, store_val, store_test, test_boards = (
        build_datasets(args)
    )
    print(
        f"chessred train {len(cr_train)} | store train {len(store_train)} | "
        f"store val {len(store_val)} | store test {len(store_test) if store_test else 0} "
        f"(test boards: {test_boards}) | chessred val {len(cr_val_ids)}",
        flush=True,
    )

    sampler = make_sampler(len(cr_train), len(store_train), args.mix, args.epoch_size)
    loader = DataLoader(
        ConcatDataset([cr_train, store_train]),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_detection,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # Split manifest: store frames per bucket (ChessReD ids come from the official split).
    def _store_bucket(ds):
        rows = sorted(
            (
                {"task_id": s.task_id, "session": s.session, "image": str(s.image_path)}
                for s in ds.samples
            ),
            key=lambda r: (r["session"], r["task_id"]),
        )
        return {
            "sessions": sorted({r["session"] for r in rows}),
            "n_frames": len(rows),
            "frames": rows,
        }

    manifest = {
        "args": {
            "store": str(args.store),
            "test_boards": test_boards,
            "mix": args.mix,
            "init": "coco-source",
            "optimizer": args.optimizer,
            "lr": args.lr,
            "epochs": args.epochs,
            "val_pose_frac": args.val_pose_frac,
            "dedup_thr": args.dedup_thr,
        },
        "chessred": {"train_n": len(cr_train), "val_ids": [int(i) for i in cr_val_ids]},
        "store_train": _store_bucket(store_train),
        "store_val": _store_bucket(store_val),
        "store_test": _store_bucket(store_test) if store_test is not None else {"n_frames": 0},
    }
    (args.out_dir / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote split manifest -> {args.out_dir / 'split.json'}")

    # Start from COCO source: pretrained trunk, fresh 12-class + keypoint heads, all trainable.
    model = build_keypoint_detector(pretrained=True).to(device)
    n_train = sum(p.numel() for p in trainable_parameters(model))
    print(
        f"init=coco-source | optimizer={args.optimizer} | trainable params {n_train:,}", flush=True
    )

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

    # Eval must use the same framing the model trained on, else scale-shift garbage.
    crop_eval = dict(
        board_crop=args.board_crop,
        crop_side=args.crop_side,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
    )
    # Stamped into the .pt so eval can auto-match the training framing (no silent mismatch).
    preprocess = {**crop_eval, "max_size": args.max_size}

    def eval_store(samples):
        return evaluate_captures(
            model,
            samples,
            None,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
            **crop_eval,
        )

    def eval_chessred():
        counts = evaluate_squares(model, chessred, cr_val_ids, device, **crop_eval)
        r = {k: rates(v) for k, v in counts.items()}
        return {"kp_square_acc": r["overall"]["kp_square_acc"], "overall": r["overall"]}

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, epoch)
        scheduler.step()
        val = eval_store(store_val.samples)
        cr_val = eval_chessred()
        row = {
            "epoch": epoch,
            "loss": round(loss, 4),
            "lr": scheduler.get_last_lr()[0],
            "store_val": val,
            "chessred_val": cr_val,
        }
        if store_test is not None:
            row["store_test"] = eval_store(store_test.samples)  # reported only; never selects
        # Select on store val class_acc -- the deployment metric, comparable to keypoint_captures.
        if val["class_acc"] > best:
            best = val["class_acc"]
            save_keypoint_checkpoint(
                model, args.out_dir / "best.pt", epoch=epoch, val=val, preprocess=preprocess
            )
        print(json.dumps(row), flush=True)
        history.append(row)
        save_keypoint_checkpoint(
            model, args.out_dir / "last.pt", epoch=epoch, preprocess=preprocess
        )
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"done. best store val class_acc {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
