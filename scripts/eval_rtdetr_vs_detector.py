"""Head-to-head: RT-DETRv2 vs the Faster R-CNN baseline, same val set, same mAP metric.

Both detectors are scored on ChessReD chessred2k **val** with the *same* metric
(`torchmetrics.detection.MeanAveragePrecision`, COCO mAP) that produced the Phase-2 number
(best val mAP 0.864) -- so the comparison is apples-to-apples, mirroring
`scripts/eval_yolo_vs_detector.py`.

Both models are scored in the same dataset (max_size) frame against the same ground truth:
  - Faster R-CNN: the resized dataset frame, exactly as train_detector.evaluate.
  - RT-DETR: the processor resizes to 640 internally; `post_process_object_detection` maps the
    normalized boxes back to the dataset-frame size we pass as target_sizes, so predictions and
    GT share the frame. mAP is IoU-ratio based, so this is invariant to the intermediate resize.
Each model keeps its own class-id convention (FRCNN 1..12 with background=0; RT-DETR 0..11);
GT is shifted to match each, so per-class AP -- and thus the scalar `map` -- is comparable.

    uv sync --group rtdetr
    uv run --group rtdetr python scripts/eval_rtdetr_vs_detector.py \
        --data-root "data/othersets/ChessReD" \
        --rtdetr-ckpt runs/rtdetr/best \
        --frcnn-ckpt runs/detector/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessvision.data.chessred import ChessReD
from chessvision.data.detection import ChessReDDetection, DetectionConfig, collate_detection
from chessvision.detector import load_detector
from chessvision.rtdetr import _to_uint8_hwc, load_rtdetr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None)
    add("--split", default="val", help="chessred2k split to score (val|test)")
    add("--rtdetr-ckpt", type=Path, default=Path("runs/rtdetr/best"), help="save_rtdetr dir")
    add("--frcnn-ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--max-size", type=int, default=1333, help="dataset long-side (matches training)")
    add("--batch-size", type=int, default=4)
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--workers", type=int, default=4)
    add("--skip-rtdetr", action="store_true", help="only score the Faster R-CNN")
    add("--skip-frcnn", action="store_true", help="only score the RT-DETR model")
    return p.parse_args(argv)


def _metric():
    from torchmetrics.detection import MeanAveragePrecision

    return MeanAveragePrecision(box_format="xyxy")


def _scalars(result: dict) -> dict[str, float]:
    return {k: float(v) for k, v in result.items() if torch.as_tensor(v).numel() == 1}


def _val_dataset(args, chessred: ChessReD) -> ChessReDDetection:
    cfg = DetectionConfig(max_size=args.max_size)
    return ChessReDDetection.from_split(chessred, args.split, config=cfg, train=False)


@torch.no_grad()
def eval_frcnn(args, chessred: ChessReD, device) -> dict[str, float]:
    """Score the Faster R-CNN exactly as train_detector.evaluate (reproduces the baseline)."""
    ds = _val_dataset(args, chessred)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_detection,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = load_detector(args.frcnn_ckpt, device=device)
    metric = _metric()
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        preds = [{k: v.cpu() for k, v in p.items()} for p in model(images)]
        metric.update(preds, targets)
    return _scalars(metric.compute())


@torch.no_grad()
def eval_rtdetr(args, chessred: ChessReD, device) -> dict[str, float]:
    """Score RT-DETR in the dataset frame; GT shifted to RT-DETR's 0..11 ids."""
    ds = _val_dataset(args, chessred)
    model, processor = load_rtdetr(args.rtdetr_ckpt, device=device)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=list
    )
    metric = _metric()
    for batch in loader:
        images, sizes, gts = [], [], []
        for image, target in batch:
            images.append(_to_uint8_hwc(image))
            sizes.append((image.shape[1], image.shape[2]))  # (H, W)
            gts.append({"boxes": target["boxes"], "labels": target["labels"] - 1})
        enc = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**enc)
        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor(sizes), threshold=0.0
        )
        preds = [
            {"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(), "labels": r["labels"].cpu()}
            for r in results
        ]
        metric.update(preds, gts)
    return _scalars(metric.compute())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    chessred = ChessReD.load(args.data_root, args.images_root)

    results: dict[str, dict[str, float]] = {}
    if not args.skip_frcnn:
        print(f"scoring Faster R-CNN ({args.frcnn_ckpt}) on {args.split} ...", flush=True)
        results["faster_rcnn"] = eval_frcnn(args, chessred, device)
    if not args.skip_rtdetr:
        print(f"scoring RT-DETRv2 ({args.rtdetr_ckpt}) on {args.split} ...", flush=True)
        results["rtdetr_v2"] = eval_rtdetr(args, chessred, device)

    keys = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]
    width = max(len(k) for k in keys)
    header = f"{'metric':<{width}}  " + "  ".join(f"{m:>12}" for m in results)
    print("\n" + header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<{width}}  " + "  ".join(
            f"{results[m].get(k, float('nan')):>12.4f}" for m in results
        )
        print(row)
    print("\n" + json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
