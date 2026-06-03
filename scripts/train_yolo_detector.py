"""Train a modern YOLO (Ultralytics) piece detector on ChessReD chessred2k.

The browser-deployable counterpart to the Faster R-CNN baseline (scripts/train_detector.py):
single-stage, exports cleanly to ONNX/TF.js, and an order of magnitude smaller. This trains
*boxes only* (12 piece classes) on the same official chessred2k train/val split, so the result
is directly comparable to the Phase-2 detector -- score them head-to-head with
scripts/eval_yolo_vs_detector.py.

Ultralytics is AGPL-3.0; our project is GPL-3.0-or-later (compatible). It lives in the optional
`yolo` dependency group to keep the core install lean:

    uv sync --group yolo            # install ultralytics (+ its deps) once
    uv run --group yolo python scripts/train_yolo_detector.py \
        --data-root "data/Chess Recognition Dataset (ChessReD)_2_all" \
        --model yolo11s.pt --epochs 100 --imgsz 1280 --batch 8 --device 0

The dataset is auto-built (hardlinks, no copy) into --yolo-dir on first run via
chessvision.data.yolo_export; pass --rebuild to regenerate it.

Notes:
  - --model picks the size: yolo11n.pt (smallest, best for the web) .. yolo11x.pt (most accurate).
    yolo11s.pt is a sensible accuracy/size baseline to challenge the Faster R-CNN with; switch to
    yolo11n.pt once you know the accuracy budget you can spend for the static app.
  - --imgsz is the main accuracy/speed lever: pieces are small in the 3072px boards, so 1280 keeps
    far-rank pieces legible (640 likely under-detects them). Must be a multiple of 32.
  - Ultralytics' default augmentation (mosaic, hflip, HSV) is left on; the Faster R-CNN used
    hflip+jitter, so this is at least as much aug. Tune via --close-mosaic / Ultralytics args later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessvision.data.yolo_export import build_yolo_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None, help="image tree root override")
    add("--yolo-dir", type=Path, default=Path("data/yolo_chessred"), help="built YOLO dataset dir")
    add("--rebuild", action="store_true", help="regenerate the YOLO dataset even if it exists")
    add("--model", default="yolo11s.pt", help="Ultralytics model/size (yolo11n..x .pt)")
    add("--epochs", type=int, default=100)
    add("--imgsz", type=int, default=1280, help="train/val image size (multiple of 32)")
    add("--batch", type=int, default=8, help="batch size (-1 lets Ultralytics auto-pick)")
    add("--device", default="0", help="'0' for first GPU, 'cpu', or '0,1' for multi-GPU")
    add("--workers", type=int, default=8)
    add("--seed", type=int, default=0)
    add("--patience", type=int, default=10, help="early-stop patience (epochs without val gain)")
    add("--project", type=Path, default=Path("runs"), help="Ultralytics project dir")
    add("--name", default="yolo_detector", help="run name under --project")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    yaml_path = args.yolo_dir / "data.yaml"
    if args.rebuild or not yaml_path.exists():
        print(f"building YOLO dataset -> {args.yolo_dir}")
        yaml_path, counts = build_yolo_dataset(
            args.data_root, args.yolo_dir, args.images_root, splits=("train", "val")
        )
        print(f"  {yaml_path}: " + ", ".join(f"{s}={n}" for s, n in counts.items()))
    else:
        print(f"using existing dataset {yaml_path} (pass --rebuild to regenerate)")

    # Imported lazily so the module/CLI parse without the optional `yolo` group installed.
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
        # Absolute project dir: a relative one is resolved against Ultralytics' own
        # settings runs_dir and lands somewhere surprising (runs/detect/...).
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
    )
    # Report the path Ultralytics actually wrote, not a reconstruction of it.
    best = Path(model.trainer.best)
    print(f"done. best weights: {best}")
    print(
        f"compare to Faster R-CNN:  uv run --group yolo python scripts/eval_yolo_vs_detector.py "
        f'--data-root "{args.data_root}" --yolo-ckpt "{best}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
