"""Head-to-head: YOLO vs the Faster R-CNN baseline, same val set, same mAP metric.

Both detectors are scored on ChessReD chessred2k **val** with the *same* metric
(`torchmetrics.detection.MeanAveragePrecision`, COCO mAP) that produced the Phase-2
number (best val mAP 0.864) -- so the comparison is apples-to-apples and not an
artifact of two libraries' differing mAP implementations.

Coordinate frames differ by model but mAP is IoU-ratio based (invariant to a uniform
resize), so each model is fed the frame it expects and scored against ground truth in
that frame:
  - Faster R-CNN: the resized (max_size) dataset frame, exactly as train_detector.evaluate
    -> reproduces the published number.
  - YOLO: original-resolution images; Ultralytics resizes to imgsz internally and returns
    boxes in original pixels, scored against original-pixel ground truth.
Each model keeps its own class-id convention (FRCNN 1..12 with background=0; YOLO 0..11);
per-class AP is computed within each scheme, so the scalar `map` is comparable.

    uv sync --group yolo
    uv run --group yolo python scripts/eval_yolo_vs_detector.py \
        --data-root "data/othersets/ChessReD" \
        --yolo-ckpt runs/yolo_detector/weights/best.pt \
        --frcnn-ckpt runs/detector/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessvision.data.chessred import ChessReD
from chessvision.data.detection import (
    ChessReDDetection,
    DetectionConfig,
    collate_detection,
)
from chessvision.detector import load_detector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None)
    add("--split", default="val", help="chessred2k split to score (val|test)")
    add("--yolo-ckpt", type=Path, default=Path("runs/yolo_detector/weights/best.pt"))
    add("--frcnn-ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--max-size", type=int, default=1333, help="Faster R-CNN long-side (matches training)")
    add("--imgsz", type=int, default=1280, help="YOLO inference image size")
    add("--conf", type=float, default=0.001, help="YOLO confidence floor (low = full PR curve)")
    add("--iou", type=float, default=0.7, help="YOLO NMS IoU")
    add("--max-det", type=int, default=300, help="YOLO max detections per image")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--workers", type=int, default=4)
    add("--skip-yolo", action="store_true", help="only score the Faster R-CNN")
    add("--skip-frcnn", action="store_true", help="only score the YOLO model")
    return p.parse_args(argv)


def _metric():
    from torchmetrics.detection import MeanAveragePrecision

    return MeanAveragePrecision(box_format="xyxy")


def _scalars(result: dict) -> dict[str, float]:
    return {k: float(v) for k, v in result.items() if torch.as_tensor(v).numel() == 1}


@torch.no_grad()
def eval_frcnn(args, chessred: ChessReD, device) -> dict[str, float]:
    """Score the Faster R-CNN exactly as train_detector.evaluate (reproduces the baseline)."""
    cfg = DetectionConfig(max_size=args.max_size)
    ds = ChessReDDetection.from_split(chessred, args.split, config=cfg, train=False)
    loader = DataLoader(
        ds,
        batch_size=2,
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
def eval_yolo(args, chessred: ChessReD) -> dict[str, float]:
    """Score the YOLO model on original-resolution val images, same metric as FRCNN."""
    from ultralytics import YOLO

    model = YOLO(str(args.yolo_ckpt))
    metric = _metric()
    for image_id in chessred.chessred2k_split(args.split):
        gt_boxes, gt_labels = [], []
        for piece in chessred.pieces(image_id):
            if piece.bbox is None:
                continue
            x, y, w, h = piece.bbox
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(piece.category_id)  # YOLO scheme: 0..11, no background
        if not gt_boxes:
            continue

        path = chessred.resolve_image_path(chessred.meta(image_id))
        res = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0].boxes
        metric.update(
            [
                {
                    "boxes": res.xyxy.cpu(),
                    "scores": res.conf.cpu(),
                    "labels": res.cls.cpu().to(torch.int64),
                }
            ],
            [
                {
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.tensor(gt_labels, dtype=torch.int64),
                }
            ],
        )
    return _scalars(metric.compute())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    chessred = ChessReD.load(args.data_root, args.images_root)

    results: dict[str, dict[str, float]] = {}
    if not args.skip_frcnn:
        print(f"scoring Faster R-CNN ({args.frcnn_ckpt}) on {args.split} ...", flush=True)
        results["faster_rcnn"] = eval_frcnn(args, chessred, device)
    if not args.skip_yolo:
        print(f"scoring YOLO ({args.yolo_ckpt}) on {args.split} ...", flush=True)
        results["yolo"] = eval_yolo(args, chessred)

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
