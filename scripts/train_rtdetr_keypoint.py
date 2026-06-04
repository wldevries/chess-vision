"""Joint ChessReD + capture-store training of RT-DETRv2 **with the contact-keypoint head**.

This is the deployment path: the model emits, per piece, a class + a board-contact point, which
the homography turns into a square -> FEN. It trains on ChessReD (real contact points from the
square-center projection) mixed with the user's capture store (hand-labelled contact points) so
it generalizes across boards -- the same recipe as `train_keypoint_joint.py`, but on the
Apache-licensed RT-DETR instead of the 234MB Keypoint R-CNN.

Selection + the headline metric is **store-val class_acc** (per-piece contact-point class
accuracy), directly comparable to the Keypoint R-CNN's `keypoint_captures` numbers. The held-out
`--test-boards` (dennis) class_acc is reported as an honest unseen-board number; ChessReD val box
mAP is a sanity check that the source domain didn't regress.

    uv run --group rtdetr python scripts/train_rtdetr_keypoint.py \
        --data-root "data/othersets/ChessReD" --store data --test-boards dennis-bord \
        --epochs 40 --batch-size 4 --device cuda --amp --board-crop \
        --color 0.1 --noise 0.03

Appearance aug (`--color`/`--noise`, off by default to match the repo convention) is a **major**
lever here, not a nicety: it lifted store class_acc 0.673->0.870 and unseen-dennis 0.562->0.786
(+~0.2). Always include it for a real run. `--hflip 0.5` (default on) is safe -- the keypoint
datasets mirror contact points. `board_exact` (whole-board FEN) stays low (~0.05) from compounding
+ RT-DETR's 300-query false positives; that, not class_acc, is the remaining deployment gap.

Guarded by `if __name__ == "__main__"` for Windows-spawn DataLoader workers.
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
from chessvision.data.detection import ChessReDKeypointDetection, DetectionConfig
from chessvision.data.positions import store_label_to_capture
from chessvision.rtdetr import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE_SIZE,
    build_processor,
    evaluate_map,
    save_rtdetr,
)
from chessvision.rtdetr_keypoint import (
    RTDetrKeypointCollate,
    TorchvisionDetAdapter,
    build_rtdetr_keypoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None)
    add("--store", type=Path, default=Path("data"), help="capture store root to mix in")
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
    add("--checkpoint", default=DEFAULT_CHECKPOINT, help="pretrained RT-DETRv2 to fine-tune")
    add("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    add("--keypoint-coef", type=float, default=5.0, help="weight on the contact-point L1 loss")
    add("--epochs", type=int, default=40)
    add("--batch-size", type=int, default=4)
    add("--lr", type=float, default=1e-4, help="AdamW lr for the transformer + heads")
    add("--backbone-lr", type=float, default=1e-5, help="lower AdamW lr for the backbone")
    add("--weight-decay", type=float, default=1e-4)
    add(
        "--patience",
        type=int,
        default=10,
        help="early-stop after N evals w/o class_acc gain (0 off)",
    )
    add("--board-crop", action="store_true", help="crop to the board bbox before the 640 resize")
    add("--crop-side", type=float, default=0.12)
    add("--crop-top", type=float, default=0.30)
    add("--crop-bottom", type=float, default=0.08)
    add("--max-size", type=int, default=1333)
    add("--hflip", type=float, default=0.5, help="train hflip prob (kp datasets mirror points)")
    add("--jitter", type=float, default=0.1)
    add("--color", type=float, default=0.0)
    add("--noise", type=float, default=0.0)
    # RT-DETR's focal/sigmoid confidences are low-absolute (unlike Keypoint R-CNN's ~0.5): a
    # 0.3 floor filters every detection and the metric reads 0. ~0.02 is where store class_acc
    # plateaus; tune per checkpoint. (Selection uses class_acc on occupied squares, which a low
    # floor doesn't inflate -- only board_exact suffers from the extra low-conf false pieces.)
    add("--score-thresh", type=float, default=0.02, help="class_acc eval: per-query score floor")
    add("--workers", type=int, default=4)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true")
    add("--grad-clip", type=float, default=0.1)
    add("--limit-train", type=int, default=None, help="cap ChessReD train images (smoke tests)")
    add("--eval-every", type=int, default=1)
    add("--out-dir", type=Path, default=Path("runs/rtdetr_keypoint"))
    return p.parse_args(argv)


def make_sampler(n_chessred: int, n_store: int, mix: float, epoch_size: int | None):
    weights = [(1.0 - mix) / max(n_chessred, 1)] * n_chessred + [mix / max(n_store, 1)] * n_store
    return WeightedRandomSampler(
        weights, num_samples=epoch_size or (n_chessred + n_store), replacement=True
    )


def build_datasets(args: argparse.Namespace, chessred: ChessReD):
    """ConcatDataset([ChessReD-kp, store-kp]) + mix sampler for training; store val/test
    CaptureSamples for class_acc eval; ChessReD val kp dataset for box-mAP sanity."""
    aug = dict(
        color=args.color,
        noise=args.noise,
        board_crop=args.board_crop,
        crop_side=args.crop_side,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
    )
    cr_train_cfg = DetectionConfig(
        max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter, **aug
    )
    cr_eval_cfg = DetectionConfig(
        max_size=args.max_size,
        board_crop=args.board_crop,
        **{k: aug[k] for k in ("crop_side", "crop_top", "crop_bottom")},
    )
    cr_train = ChessReDKeypointDetection.from_split(chessred, "train", config=cr_train_cfg)
    if args.limit_train:
        cr_train.image_ids = cr_train.image_ids[: args.limit_train]
    cr_val = ChessReDKeypointDetection.from_split(chessred, "val", config=cr_eval_cfg)

    store = CornerStore(args.store)
    test_boards = [b for b in args.test_boards.split(",") if b]
    tr, va, te = split_store_for_keypoints(
        store, test_boards=test_boards, val_pose_frac=args.val_pose_frac, dedup_thr=args.dedup_thr
    )
    to_cap = lambda labels: [store_label_to_capture(lb, store) for lb in labels]  # noqa: E731
    st_train_cfg = CaptureKeypointConfig(
        max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter, **aug
    )
    st_eval_cfg = CaptureKeypointConfig(
        max_size=args.max_size,
        board_crop=args.board_crop,
        crop_side=args.crop_side,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
    )
    store_train = CaptureKeypointDetection(to_cap(tr), None, st_train_cfg, train=True)
    store_val = CaptureKeypointDetection(to_cap(va), None, st_eval_cfg, train=False)
    store_test = (
        CaptureKeypointDetection(to_cap(te), None, st_eval_cfg, train=False) if te else None
    )
    print(
        f"chessred train {len(cr_train)} | store train {len(store_train)} | "
        f"store val {len(store_val)} | store test {len(store_test) if store_test else 0} "
        f"(test boards: {test_boards})",
        flush=True,
    )
    sampler = make_sampler(len(cr_train), len(store_train), args.mix, args.epoch_size)
    return ConcatDataset([cr_train, store_train]), sampler, cr_val, store_val, store_test


def train_one_epoch(model, loader, optimizer, device, scaler, grad_clip, epoch) -> float:
    model.train()
    running, t0 = 0.0, time.time()
    for step, enc in enumerate(loader):
        pixel_values = enc["pixel_values"].to(device, non_blocking=True)
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
        running += float(loss.item())
        if step % 20 == 0:
            rate = (step + 1) * pixel_values.size(0) / (time.time() - t0)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} "
                f"loss {loss.item():.3f} ({rate:.1f} img/s)",
                flush=True,
            )
    return running / max(len(loader), 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    chessred = ChessReD.load(args.data_root, args.images_root)
    train_ds, sampler, cr_val, store_val, store_test = build_datasets(args, chessred)
    processor = build_processor(args.checkpoint, image_size=args.image_size)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=RTDetrKeypointCollate(processor),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = build_rtdetr_keypoint(
        args.checkpoint, pretrained=True, keypoint_loss_coef=args.keypoint_coef
    ).to(device)
    adapter = TorchvisionDetAdapter(model, processor, device)
    backbone, other = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (backbone if "backbone" in name else other).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": other, "lr": args.lr}, {"params": backbone, "lr": args.backbone_lr}],
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    crop = dict(
        board_crop=args.board_crop,
        crop_side=args.crop_side,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
    )

    def class_acc(samples):
        m = evaluate_captures(
            adapter,
            samples,
            None,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
            **crop,
        )
        return {k: round(m[k], 4) for k in ("localization", "class_acc", "board_exact_rate")}

    history, best = [], -1.0
    since_improve = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, args.grad_clip, epoch)
        row: dict = {"epoch": epoch, "train_loss": round(loss, 4)}
        stop = False
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            sv = class_acc(store_val.samples)
            cr_map = evaluate_map(model, processor, cr_val, device, args.batch_size, args.workers)
            row["store_val"] = sv
            row["chessred_val_map"] = round(cr_map.get("map", float("nan")), 4)
            if store_test is not None:
                row["store_test"] = class_acc(store_test.samples)
            if sv["class_acc"] > best:
                best = sv["class_acc"]
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
                f"early stop: no store class_acc gain in {args.patience} evals (best {best:.4f})",
                flush=True,
            )
            break

    print(f"done. best store val class_acc: {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
