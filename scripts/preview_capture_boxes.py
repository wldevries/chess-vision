"""Preview synthesized piece boxes on capture photos (eyeball before fine-tuning).

The capture set has hand-tagged contact points (base + class) but no boxes. To
fine-tune the region-based detector/keypoint head we synthesize a box per piece
with `geometry.project_piece_box`. This draws those boxes (+ the contact point) on a
few capture images so we can sanity-check the sizing before training on them.

Usage:
    uv run python scripts/preview_capture_boxes.py --limit 8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from chessvision.data.captures import CaptureDataset
from chessvision.geometry import (
    PIECE_HEIGHT_SCALE,
    Orientation,
    compute_homography,
    project_piece_box,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    add("--out-dir", type=Path, default=Path("runs/capture_boxes"))
    add("--limit", type=int, default=8)
    add("--seed", type=int, default=42, help="shuffle seed so samples span sessions")
    add("--max-size", type=int, default=1600, help="downscale longest side to this; 0 = full res")
    add("--radius-squares", type=float, default=0.3, help="piece base radius (squares)")
    add("--height-mult", type=float, default=1.0, help="global multiplier on piece heights")
    add("--focal-scale", type=float, default=1.0, help="assumed focal = scale * max(W,H)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = CaptureDataset.load(args.captures)
    samples = list(dataset.with_all_corners())
    random.Random(args.seed).shuffle(samples)  # spread across sessions, not one game
    samples = samples[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("sessions in sample:", sorted({s.session for s in samples}))

    for s in samples:
        homography = compute_homography(s.corners, Orientation.R0)
        img = cv2.cvtColor(s.load_image(dataset.s3), cv2.COLOR_RGB2BGR)
        for kp in s.pieces:
            x1, y1, x2, y2 = project_piece_box(
                homography,
                kp.point,
                (s.width, s.height),
                height_squares=PIECE_HEIGHT_SCALE.get(kp.fen.lower(), 1.0) * args.height_mult,
                radius_squares=args.radius_squares,
                focal_scale=args.focal_scale,
            )
            cv2.rectangle(img, (round(x1), round(y1)), (round(x2), round(y2)), (0, 200, 0), 2)
            cv2.circle(img, (round(kp.point[0]), round(kp.point[1])), 5, (0, 0, 255), -1)
        if args.max_size and max(img.shape[:2]) > args.max_size:
            sc = args.max_size / max(img.shape[:2])
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out_dir / f"{s.session}_{s.task_id}.jpg"), img)

    print(f"wrote {len(samples)} overlay(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
