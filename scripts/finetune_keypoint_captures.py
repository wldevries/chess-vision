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
from chessvision.data.capture_detection import (
    CaptureKeypointConfig,
    CaptureKeypointDetection,
    split_by_sessions,
)
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
    add(
        "--positions-root",
        type=Path,
        default=None,
        help="data/corners tree with in-app position labels; folded in as pos-<board> "
        "sessions so a whole board can be held out via --val-sessions",
    )
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint/best.pt"))
    add("--detector-ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--val-sessions", default=DEFAULT_VAL_SESSIONS, help="comma-separated held-out sessions")
    add(
        "--test-sessions",
        default="",
        help="comma-separated sessions held out of BOTH train and val (selection). Evaluated "
        "separately each epoch and reported as 'test' -- use for an honest unseen-board number "
        "(e.g. pos-dennis-bord) that checkpoint selection never peeks at.",
    )
    # Unified-store path (the merged corner+capture store; retires LS + the captures/positions
    # split). When --store is given it is the SINGLE source: every piece-labelled record, keyed
    # on its real session. Test = whole boards; val = pose-held-out fraction of the rest.
    add("--store", type=Path, default=None, help="unified corner store root (e.g. data/corners); "
        "single source when given -- ignores --captures/--positions-root/--val/test-sessions")
    add("--test-boards", default="", help="store path: comma-separated boards held out as test "
        "(whole board -> train+selection both exclude it), e.g. dennis-bord")
    add("--val-pose-frac", type=float, default=0.25, help="store path: share of each non-test "
        "board's pose clusters held out for val (selection)")
    add("--dedup-thr", type=float, default=0.02, help="store path: pose-cluster dist (frac img)")
    add("--unfreeze", choices=FINETUNE_SCOPES, default="classifier", help="how much to adapt")
    add("--optimizer", choices=("adamw", "sgd"), default="adamw")
    add("--epochs", type=int, default=5)
    add("--batch-size", type=int, default=2)
    add("--lr", type=float, default=1e-4, help="AdamW: ~1e-4; SGD: raise to ~5e-3")
    add("--momentum", type=float, default=0.9, help="SGD only")
    add("--weight-decay", type=float, default=1e-2)
    add("--max-size", type=int, default=1333)
    add("--hflip", type=float, default=0.5)
    add("--jitter", type=float, default=0.1, help="brightness/contrast jitter magnitude")
    # Appearance-only corruptions (image alone; targets untouched). Default off so behaviour
    # is unchanged unless requested. Simulate real capture variance without a render domain gap.
    add("--aug-color", type=float, default=0.0, help="per-channel gain / white-balance (e.g. 0.1)")
    add("--aug-blur", type=float, default=0.0, help="max Gaussian blur sigma px, DoF (e.g. 1.0)")
    add("--aug-motion-blur", type=float, default=0.0, help="max motion-blur length px (e.g. 5)")
    add("--aug-noise", type=float, default=0.0, help="max noise std /255, low-light (e.g. 0.03)")
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

    cfg = CaptureKeypointConfig(
        max_size=args.max_size,
        hflip_prob=args.hflip,
        jitter=args.jitter,
        color=args.aug_color,
        blur=args.aug_blur,
        motion_blur=args.aug_motion_blur,
        noise=args.aug_noise,
    )

    # Split manifest helper: exact per-bucket membership by id/session/path, so "which images
    # were in train/val/test" is answerable from disk, not re-derived from args + data state.
    def _bucket(samples):
        rows = sorted(
            (
                {
                    "task_id": s.task_id,
                    "session": s.session,
                    "image": str(s.image_path),
                    "s3_uri": s.s3_uri,
                }
                for s in samples
            ),
            key=lambda r: (r["session"], r["task_id"]),
        )
        sessions = sorted({r["session"] for r in rows})
        return {"sessions": sessions, "n_frames": len(rows), "frames": rows}

    if args.store:
        # Unified store: the single source. test = whole boards, val = pose-held-out fraction
        # of the rest, both at the CornerLabel level (which carries board), then -> CaptureSample.
        from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
        from chessvision.data.positions import store_label_to_capture

        store = CornerStore(args.store)
        test_boards = [b for b in args.test_boards.split(",") if b]
        tr, va, te = split_store_for_keypoints(
            store,
            test_boards=test_boards,
            val_pose_frac=args.val_pose_frac,
            dedup_thr=args.dedup_thr,
        )
        to_cap = lambda labels: [store_label_to_capture(lb, store) for lb in labels]  # noqa: E731
        train_ds = CaptureKeypointDetection(to_cap(tr), None, cfg, train=True)
        val_ds = CaptureKeypointDetection(to_cap(va), None, cfg, train=False)
        test_ds = (
            CaptureKeypointDetection(to_cap(te), None, cfg, train=False) if te else None
        )
        eval_s3 = None
        print(
            f"store={args.store} | train {len(train_ds)} | val {len(val_ds)} | "
            f"test {len(test_ds) if test_ds else 0} (test boards: {test_boards})"
        )
        manifest = {
            "args": {
                "store": str(args.store),
                "keypoint_ckpt": str(args.keypoint_ckpt),
                "unfreeze": args.unfreeze,
                "epochs": args.epochs,
                "test_boards": test_boards,
                "val_pose_frac": args.val_pose_frac,
                "dedup_thr": args.dedup_thr,
            },
            "train": _bucket(train_ds.samples),
            "val": _bucket(val_ds.samples),
            "test": _bucket(test_ds.samples if test_ds is not None else []),
        }
    else:
        # Legacy path: Label Studio capture export and/or in-app position labels (folded in as
        # pos-<board>). Kept for back-compat; the merged store (--store) supersedes it.
        if args.captures.exists():
            dataset = CaptureDataset.load(args.captures)
        elif args.positions_root:
            dataset = CaptureDataset(export_path=args.captures, captures_root=Path("."), samples=[])
        else:
            raise SystemExit(f"no captures at {args.captures} and no --positions-root given")
        if args.positions_root:
            from chessvision.data.positions import position_samples_as_captures

            pos = position_samples_as_captures(args.positions_root)
            print(f"positions: +{len(pos)} samples from {args.positions_root}")
            dataset.samples = dataset.samples + pos
        counts = {s: len(g) for s, g in sorted(dataset.by_session().items())}
        print(f"sessions ({len(counts)}): {json.dumps(counts)}")

        # Test sessions removed BEFORE the split -> excluded from train AND selection.
        test_sessions = [s for s in args.test_sessions.split(",") if s]
        test_ds = None
        if test_sessions:
            by_session = dataset.by_session()
            unknown = set(test_sessions) - set(dataset.sessions)
            if unknown:
                raise SystemExit(f"unknown test sessions {sorted(unknown)}; have {dataset.sessions}")  # noqa: E501
            test_samples = [s for sess in test_sessions for s in by_session[sess]]
            test_ds = CaptureKeypointDetection(test_samples, dataset.s3, cfg, train=False)
            dataset.samples = [s for s in dataset.samples if s.session not in set(test_sessions)]
            print(f"test (held out of train+selection): {len(test_samples)} frames {test_sessions}")

        val_sessions = [s for s in args.val_sessions.split(",") if s]
        train_ds, val_ds = split_by_sessions(dataset, val_sessions, cfg)
        eval_s3 = dataset.s3
        print(
            f"train {len(train_ds)} frames | val {len(val_ds)} frames "
            f"(held-out sessions: {val_sessions})"
        )
        manifest = {
            "args": {
                "captures": str(args.captures),
                "positions_root": str(args.positions_root) if args.positions_root else None,
                "keypoint_ckpt": str(args.keypoint_ckpt),
                "unfreeze": args.unfreeze,
                "epochs": args.epochs,
                "val_sessions": val_sessions,
                "test_sessions": test_sessions,
            },
            "train": _bucket(train_ds.samples),
            "val": _bucket(val_ds.samples),
            "test": _bucket(test_ds.samples if test_ds is not None else []),
        }

    (args.out_dir / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote split manifest -> {args.out_dir / 'split.json'}")

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

    def eval_test():
        if test_ds is None:
            return None
        return evaluate_captures(
            model,
            test_ds.samples,
            eval_s3,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
        )

    # Baseline: held-out accuracy of the pretrained model before any fine-tuning.
    base = evaluate_captures(
        model,
        val_ds.samples,
        eval_s3,
        device,
        max_size=args.max_size,
        score_thresh=args.score_thresh,
    )
    base_test = eval_test()
    print("baseline (pretrained) val:", json.dumps(base), flush=True)
    if base_test is not None:
        print("baseline (pretrained) test:", json.dumps(base_test), flush=True)

    history = [{"epoch": 0, "val": base, "test": base_test}]
    best = base["class_acc"]
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, epoch)
        scheduler.step()
        val = evaluate_captures(
            model,
            val_ds.samples,
            eval_s3,
            device,
            max_size=args.max_size,
            score_thresh=args.score_thresh,
        )
        test = eval_test()
        row = {"epoch": epoch, "loss": round(loss, 4), "lr": scheduler.get_last_lr()[0], "val": val}
        if test is not None:
            row["test"] = test  # reported only; selection uses val class_acc, never test
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
