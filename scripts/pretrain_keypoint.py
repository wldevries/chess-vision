"""Multi-domain keypoint-detector PRETRAIN (start from COCO, train the whole model).

This is the diverse pretraining stage: start from the torchvision **COCO source** Faster
R-CNN v2, graft the 1-keypoint contact head, **unlock everything**, and train the full
detector + keypoint on a mix of ChessReD + the external synthetic sets (Chesscog +
SyntheticBoards, `chessvision.data.synthetic_sets`). All three emit the identical
`(image, target)` shape, so they concatenate with no remapping. ChessReD is fully covered
every epoch; each other set contributes a `--domain-mult` fraction of ChessReD's size, drawn
without replacement and re-drawn each epoch (default 0.5 -> a 100/50/50 image mix), so the
larger Chesscog set (~4400) never swamps ChessReD (~1442) / SyntheticBoards (~1944).

The point is **generalization**: warming the trunk on varied boards/sets/viewpoints builds
transferable features (and a transferable contact-point *localizer* -- the lever for the
unseen-board localization gap, see new-colleague-board-incoming / joint-training-beats-
sequential). The output `runs/keypoint_pretrain/best.pt` is then a strong base to **finetune
on the store** (`finetune_keypoint_captures.py`) or to seed the joint train.

Contrast with the LEGACY `train_keypoint_head.py`, which trains ONLY the keypoint head on a
FROZEN ChessReD trunk (a narrow sub-step of the old sequential pipeline). This script is the
full-model pretrain; that one is head-only.

Selection = ChessReD chessred2k val keypoint square-accuracy (so runs stay comparable); the
real generalization test is downstream on the store + dennis.

Usage:
    uv run python scripts/pretrain_keypoint.py --device cuda --amp --epochs 12
    # ChessReD fully covered each epoch + half-that-many from each other set (100/50/50);
    # tune the non-anchor share or scope:
    uv run python scripts/pretrain_keypoint.py --domain-mult 0.5 --scope backbone ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from chessvision.data.chessred import ChessReD
from chessvision.data.detection import (
    ChessReDKeypointDetection,
    DetectionConfig,
    collate_detection,
)
from chessvision.data.synthetic_sets import (
    ChesscogKeypointDetection,
    SyntheticBoardsKeypointDetection,
)
from chessvision.keypoint_detector import (
    build_keypoint_detector,
    graft_from_detector_checkpoint,
    keypoint_parameters,
    save_keypoint_checkpoint,
    set_finetune_scope,
)
from chessvision.keypoint_eval import evaluate_squares, rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", type=Path, default=Path("data/othersets/ChessReD"), help="ChessReD dir")
    add("--images-root", type=Path, default=None)
    add(
        "--chesscog-root",
        type=Path,
        default=Path("data/othersets/Chesscog"),
        help="Chesscog (oblique, boxed); skipped if the dir is absent",
    )
    add(
        "--synthetic-root",
        type=Path,
        default=Path("data/othersets/synthetic-chess-board-images"),
        help="SyntheticBoards (top-down, varied materials); skipped if absent",
    )
    add(
        "--domain-mult",
        default="0.5",
        help="non-anchor domain size per epoch, as a multiple of the ANCHOR (first domain = "
        "ChessReD) full size. ChessReD is ALWAYS fully covered each epoch; each other domain "
        "contributes round(mult * n_chessred) samples drawn WITHOUT replacement (fresh each "
        "epoch). One value applies to all others (e.g. '0.5' -> 100/50/50); or a comma list "
        "per non-anchor domain in order (chesscog,synthetic), e.g. '0.5,0.3'.",
    )
    add(
        "--init",
        choices=("coco", "detector"),
        default="coco",
        help="coco: whole trunk from COCO source weights (the pretrain). detector: graft onto "
        "runs/detector/best.pt instead (a ChessReD-warmed trunk).",
    )
    add(
        "--scope",
        choices=("full", "backbone", "keypoint"),
        default="full",
        help="full: every parameter trainable (the pretrain). backbone: upper backbone/FPN/RPN/"
        "heads. keypoint: only the keypoint head (requires --init detector).",
    )
    add("--detector-ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--epochs", type=int, default=12)
    add("--batch-size", type=int, default=2)
    add("--lr", type=float, default=0.005, help="SGD; full COCO train ~0.005")
    add("--momentum", type=float, default=0.9)
    add("--weight-decay", type=float, default=5e-4)
    add("--max-size", type=int, default=1333)
    add("--hflip", type=float, default=0.5)
    add("--jitter", type=float, default=0.1)
    add("--workers", type=int, default=4)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--amp", action="store_true")
    add("--eval-every", type=int, default=1)
    add("--eval-limit", type=int, default=None, help="cap ChessReD val images per eval (speed)")
    add("--out-dir", type=Path, default=Path("runs/keypoint_pretrain"))
    return p.parse_args(argv)


def build_sources(args, chessred):
    """The enabled training domains as (name, dataset), all emitting the same target shape.
    A synthetic root that doesn't exist is skipped (with a note) rather than erroring."""
    cfg = DetectionConfig(max_size=args.max_size, hflip_prob=args.hflip, jitter=args.jitter)
    sources = [("chessred", ChessReDKeypointDetection.from_split(chessred, "train", config=cfg))]
    if args.chesscog_root and args.chesscog_root.exists():
        sources.append(
            ("chesscog", ChesscogKeypointDetection(args.chesscog_root, "train", cfg, train=True))
        )
    elif args.chesscog_root:
        print(f"note: --chesscog-root {args.chesscog_root} absent, skipping")
    if args.synthetic_root and args.synthetic_root.exists():
        sources.append(
            ("synthetic", SyntheticBoardsKeypointDetection(args.synthetic_root, cfg, train=True))
        )
    elif args.synthetic_root:
        print(f"note: --synthetic-root {args.synthetic_root} absent, skipping")
    return sources


class AnchoredCoverageSampler(Sampler):
    """Indices into a ConcatDataset that guarantee FULL coverage of the anchor domain (domain 0,
    ChessReD) each epoch, plus a `mult * n_anchor`-sized subsample of every other domain drawn
    WITHOUT replacement and re-drawn each epoch.

    Why not a WeightedRandomSampler: weighted sampling draws *with replacement*, so at 50% mass
    only ~63% of ChessReD's unique images appear per epoch. Here ChessReD is fully covered every
    epoch; the larger synthetic sets rotate a fresh subsample through their pools across epochs.
    A non-anchor domain whose requested count exceeds its pool tops up with replacement.
    """

    def __init__(self, sizes, mults):
        self.sizes = list(sizes)
        self.mults = list(mults)  # one per non-anchor domain (domains 1..k-1)
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.n_anchor = self.sizes[0]

    def counts(self):
        """Per-domain samples per epoch: anchor full, others round(mult * n_anchor)."""
        return [self.n_anchor] + [round(m * self.n_anchor) for m in self.mults]

    def __len__(self):
        return sum(self.counts())

    def __iter__(self):
        chunks = [self.offsets[0] + torch.randperm(self.sizes[0])]
        for i, count in enumerate(self.counts()[1:], start=1):
            n = self.sizes[i]
            perm = torch.randperm(n)
            if count <= n:
                pick = perm[:count]
            else:  # need more than the pool holds -> top up with replacement
                pick = torch.cat([perm, torch.randint(n, (count - n,))])
            chunks.append(self.offsets[i] + pick)
        order = torch.cat(chunks)
        return iter(order[torch.randperm(len(order))].tolist())  # interleave domains


def _resolve_mults(args, n_nonanchor, names):
    """Parse --domain-mult into one multiplier per non-anchor domain. A single value applies to
    all; a comma list must match the non-anchor count."""
    vals = [float(x) for x in str(args.domain_mult).split(",")]
    if len(vals) == 1:
        return vals * n_nonanchor
    if len(vals) != n_nonanchor:
        raise SystemExit(
            f"--domain-mult has {len(vals)} values but {n_nonanchor} non-anchor domains are "
            f"enabled ({', '.join(names[1:])}); give one value, or one per domain in that order"
        )
    return vals


def build_loader(args, sources):
    """DataLoader over the concatenated domains. One domain -> plain shuffle (full coverage). Many
    -> AnchoredCoverageSampler: all of ChessReD each epoch + a `--domain-mult` subsample of each
    other domain (no replacement, fresh per epoch)."""
    names = [n for n, _ in sources]
    datasets = [d for _, d in sources]
    concat = ConcatDataset(datasets)
    common = dict(
        batch_size=args.batch_size,
        collate_fn=collate_detection,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    if len(datasets) == 1:
        return DataLoader(concat, shuffle=True, **common)
    mults = _resolve_mults(args, len(datasets) - 1, names)
    sampler = AnchoredCoverageSampler([len(d) for d in datasets], mults)
    counts = sampler.counts()
    print(
        "sampler (anchored coverage): "
        + ", ".join(f"{nm}={c}" for nm, c in zip(names, counts, strict=True))
        + f" | epoch_size={sum(counts)}",
        flush=True,
    )
    return DataLoader(concat, sampler=sampler, **common)


def build_model(args, device):
    """Build + freeze per --init/--scope. Returns (model, full_loss): full_loss means the trunk
    trains, so we backprop the whole multi-task loss, not just the keypoint branch."""
    if args.init == "coco":
        if args.scope == "keypoint":
            raise SystemExit(
                "--init coco --scope keypoint = random frozen trunk; use full/backbone"
            )
        model = build_keypoint_detector(pretrained=True).to(device)
    else:
        model = graft_from_detector_checkpoint(args.detector_ckpt, device)

    if args.scope == "keypoint":
        from chessvision.keypoint_detector import freeze_trunk

        freeze_trunk(model)
        return model, False
    if args.scope == "full":
        for prm in model.parameters():
            prm.requires_grad_(True)
    else:  # backbone
        set_finetune_scope(model, "backbone")
    return model, True


def train_one_epoch(model, loader, optimizer, device, scaler, epoch, full_loss) -> float:
    model.train()
    running, t0 = 0.0, time.time()
    for step, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) if full_loss else loss_dict["loss_keypoint"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.item())
        if step % 50 == 0:
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

    chessred = ChessReD.load(args.data_root, args.images_root)
    sources = build_sources(args, chessred)
    loader = build_loader(args, sources)
    val_ids = chessred.chessred2k_split("val")
    if args.eval_limit:
        val_ids = val_ids[: args.eval_limit]

    model, full_loss = build_model(args, device)
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
    n_train = sum(p.numel() for p in keypoint_parameters(model))
    domains = ", ".join(f"{name}={len(d)}" for name, d in sources)
    print(
        f"init={args.init} scope={args.scope} (full_loss={full_loss}) | "
        f"domains: {domains} | val {len(val_ids)} | device {device} | trainable params {n_train:,}",
        flush=True,
    )

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, scaler, epoch, full_loss)
        scheduler.step()
        row = {"epoch": epoch, "loss": round(loss, 4), "lr": scheduler.get_last_lr()[0]}
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

    print(f"done. best ChessReD val kp square-acc: {best:.4f} -> {args.out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
