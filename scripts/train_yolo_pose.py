"""Train a YOLO-pose piece detector (box + board-contact keypoint) -- the FEN-relevant model.

The browser-deployable counterpart to the Keypoint R-CNN (`scripts/train_keypoint_joint.py`):
single-stage, tiny, exports to ONNX/TF.js. Each piece gets a class + its **contact point**
(the doctrine-pure base keypoint), which downstream maps through the homography to a square ->
FEN. Trains on the SAME data mix as the champion joint trainer (ChessReD-train + capture-store),
so it can be compared head-to-head with `scripts/eval_yolo_pose_vs_keypoint.py`.

    uv sync --group yolo
    uv run --group yolo python scripts/train_yolo_pose.py \
        --data-root data/othersets/ChessReD --store data --test-boards dennis-bord \
        --model yolo11n-pose.pt --epochs 200 --imgsz 1280 --batch -1 --device 0

The dataset is auto-built into --pose-dir on first run (ChessReD images hardlinked; store images
decoded so the pixel frame matches labels); pass --rebuild to regenerate. The store TEST boards
(unseen-board number) are NOT in YOLO's train/val -- the comparison eval scores them separately.

Notes:
  - YOLO-pose trains box + keypoint jointly; the keypoint (OKS) loss weight is `pose=12.0` by
    default in Ultralytics. The contact point is the only thing square assignment needs.
  - --imgsz 1280: pieces are small in the 3072px boards. This is the main accuracy/speed lever.
  - Matches the champion on the two levers that move the number: board-crop (higher piece
    resolution) ON by default, and the store oversampled ~50/50 vs ChessReD (--store-oversample).
    The eval must use the SAME board-crop (eval_yolo_pose_vs_keypoint.py does, by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessvision.data.yolo_pose_export import build_yolo_pose_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None)
    add("--store", type=Path, default=Path("data"), help="unified corner store root (flat layout)")
    add("--pose-dir", type=Path, default=Path("data/yolo_pose"), help="built YOLO-pose dataset dir")
    add("--rebuild", action="store_true", help="regenerate the dataset even if it exists")
    add(
        "--test-boards",
        default="dennis-bord",
        help="boards held out as TEST (excluded from train/val)",
    )
    add("--val-pose-frac", type=float, default=0.25)
    add("--dedup-thr", type=float, default=0.02)
    add(
        "--board-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train on board-cropped images for higher piece res (matches champion); default on",
    )
    add(
        "--store-oversample",
        type=int,
        default=None,
        help="duplicate store frames N times to rebalance ~50/50 vs ChessReD (default auto)",
    )
    add("--model", default="yolo11n-pose.pt", help="Ultralytics pose model (yolo11n/s-pose .pt)")
    add("--epochs", type=int, default=200)
    add("--imgsz", type=int, default=1280, help="train/val image size (multiple of 32)")
    add("--batch", type=int, default=-1, help="batch size (-1 = Ultralytics auto)")
    add("--device", default="0")
    add("--workers", type=int, default=8)
    add("--seed", type=int, default=0)
    add("--patience", type=int, default=50, help="early-stop patience (epochs without val gain)")
    add("--project", type=Path, default=Path("runs"))
    add("--name", default="yolo_pose", help="run name under --project")
    add("--resume", action="store_true", help="resume an interrupted run from its last.pt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Resume an interrupted run from its last.pt (Ultralytics reloads the saved args/optimizer
    # and continues to the original epoch count); dataset is reused as-is.
    if args.resume:
        from ultralytics import YOLO

        last = args.project.resolve() / args.name / "weights" / "last.pt"
        print(f"resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        print(f"done. best weights: {Path(model.trainer.best)}")
        return 0

    yaml_path = args.pose_dir / "data.yaml"
    if args.rebuild or not yaml_path.exists():
        print(f"building YOLO-pose dataset -> {args.pose_dir}")
        yaml_path, counts = build_yolo_pose_dataset(
            args.data_root,
            args.store,
            args.pose_dir,
            args.images_root,
            tuple(b for b in args.test_boards.split(",") if b),
            args.val_pose_frac,
            args.dedup_thr,
            args.board_crop,
            args.store_oversample,
        )
        print(f"  {yaml_path}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    else:
        print(f"using existing dataset {yaml_path} (pass --rebuild to regenerate)")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=str(args.project.resolve()),  # absolute: avoid Ultralytics' runs/ nesting
        name=args.name,
        exist_ok=True,
    )
    best = Path(model.trainer.best)
    print(f"done. best weights: {best}")
    print(
        f"compare to Keypoint R-CNN:  uv run --group yolo python "
        f'scripts/eval_yolo_pose_vs_keypoint.py --data-root "{args.data_root}" '
        f'--yolo-pose-ckpt "{best}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
