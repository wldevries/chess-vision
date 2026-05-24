"""Run the homography self-check over ChessReD and write a report.

Builds a board homography from each image's labelled corners, projects every
piece's base point, and compares the predicted square to the labelled one. This
validates Approach A (plan.md section 4) on real data before any training.

Usage:
    uv run python scripts/check_homography.py \
        --data-root "data/Chess Recognition Dataset (ChessReD)_2_all" \
        --offset-sweep --worst-n 20

Outputs (under --out-dir): summary.json, per_image.csv, mismatches.csv,
flagged.json, and overlays/*.jpg for the worst-N images (needs --worst-n>0).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from chessvision.data.chessred import ChessReD
from chessvision.geometry import Orientation
from chessvision.selfcheck import ImageResult, SelfCheckReport, run, sweep_vertical_offset

OFFSET_KS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None,
        help="image tree root (default <data-root>/chessred/images)")
    add("--limit", type=int, default=None, help="process at most N corner-annotated images")
    add("--orientation", choices=["auto", "r0"], default="auto",
        help="try 4 rotations, or force R0")
    add("--tol", type=float, default=0.06, help="off-board tolerance in canonical units")
    add("--offset", type=float, default=0.0,
        help="base-point vertical offset (fraction of bbox height)")
    add("--offset-sweep", action="store_true", help="also report accuracy vs offset k")
    add("--flag-threshold", type=float, default=0.9,
        help="per-image accuracy below this is flagged")
    add("--worst-n", type=int, default=0,
        help="render overlays for the N lowest-accuracy images")
    add("--out-dir", type=Path, default=Path("runs/homography_check"), help="output dir")
    return p.parse_args(argv)


def write_per_image_csv(path: Path, per_image: list[ImageResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["image_id", "orientation", "n_pieces", "n_matched", "n_offboard", "accuracy", "err"]
        )
        for r in per_image:
            w.writerow([
                r.image_id,
                r.orientation.name if r.orientation is not None else "",
                r.n_pieces, r.n_matched, r.n_offboard, f"{r.accuracy:.6f}", int(r.errored),
            ])


def write_mismatches_csv(path: Path, per_image: list[ImageResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["image_id", "piece_id", "category", "label_square", "pred_square", "base_x", "base_y"]
        )
        for r in per_image:
            for m in r.mismatches:
                w.writerow([
                    r.image_id, m.piece_id, m.category_id, m.label_square,
                    m.pred_square or "OFF", f"{m.base_xy[0]:.2f}", f"{m.base_xy[1]:.2f}",
                ])


def write_flagged_json(path: Path, report: SelfCheckReport) -> None:
    payload = [
        {
            "image_id": r.image_id,
            "accuracy": round(r.accuracy, 6),
            "orientation": r.orientation.name if r.orientation is not None else None,
            "n_pieces": r.n_pieces,
            "mismatches": [
                {"piece_id": m.piece_id, "label": m.label_square, "pred": m.pred_square}
                for m in r.mismatches
            ],
        }
        for r in report.flagged
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(path: Path, report: SelfCheckReport, args: argparse.Namespace) -> dict:
    accs = [r.accuracy for r in report.per_image if not r.errored and r.n_pieces > 0]
    accs.sort()
    median = accs[len(accs) // 2] if accs else 0.0
    summary = {
        "params": {
            "data_root": str(args.data_root), "limit": args.limit, "orientation": args.orientation,
            "tol": args.tol, "offset": args.offset, "flag_threshold": args.flag_threshold,
        },
        "n_images": report.n_images,
        "n_errored": report.n_errored,
        "n_pieces": report.n_pieces,
        "n_matched": report.n_matched,
        "n_offboard": report.n_offboard,
        "global_accuracy": round(report.global_accuracy, 6),
        "mean_per_image_accuracy": round(sum(accs) / len(accs), 6) if accs else 0.0,
        "median_per_image_accuracy": round(median, 6),
        "orientation_counts": report.orientation_counts,
        "n_flagged": len(report.flagged),
        "offset_table": {
            str(k): round(v, 6) for k, v in (report.offset_table or {}).items()
        } or None,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_overlay(image_path: Path, corners: dict, result: ImageResult, out_path: Path) -> bool:
    import cv2  # lazy: only needed for overlays
    import numpy as np

    from chessvision.geometry import compute_homography, lattice_points

    img = cv2.imread(str(image_path))
    if img is None or result.orientation is None:
        return False

    h = compute_homography(corners, result.orientation)
    grid = lattice_points(h).reshape(9, 9, 2).round().astype(np.int32)
    for i in range(9):
        cv2.polylines(img, [np.ascontiguousarray(grid[i])], False, (0, 255, 255), 2)
        cv2.polylines(img, [np.ascontiguousarray(grid[:, i])], False, (0, 255, 255), 2)
    for m in result.results:
        x, y = int(m.base_xy[0]), int(m.base_xy[1])
        color = (0, 200, 0) if m.matched else (0, 0, 255)
        cv2.circle(img, (x, y), 10, color, -1)
        if not m.matched:
            cv2.putText(img, f"{m.label_square}/{m.pred_square or 'OFF'}", (x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    scale = 1024 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return bool(cv2.imwrite(str(out_path), img))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = ChessReD.load(args.data_root, args.images_root)
    orientations = (Orientation.R0,) if args.orientation == "r0" else tuple(Orientation)

    images = list(itertools.islice(dataset.images_with_corners(), args.limit) if args.limit
                  else dataset.images_with_corners())
    report = run(images, orientations=orientations, tol=args.tol,
                 vertical_offset=args.offset, flag_threshold=args.flag_threshold)
    if args.offset_sweep:
        report.offset_table = sweep_vertical_offset(images, OFFSET_KS, orientations, args.tol)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_per_image_csv(args.out_dir / "per_image.csv", report.per_image)
    write_mismatches_csv(args.out_dir / "mismatches.csv", report.per_image)
    write_flagged_json(args.out_dir / "flagged.json", report)
    summary = write_summary(args.out_dir / "summary.json", report, args)

    print(json.dumps(summary, indent=2))

    if args.worst_n > 0:
        by_id = {img.meta.image_id: img for img in images}
        scored = sorted((r for r in report.per_image if not r.errored and r.n_pieces > 0),
                        key=lambda r: r.accuracy)[: args.worst_n]
        overlay_dir = args.out_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        n = 0
        for r in scored:
            img = by_id[r.image_id]
            out = overlay_dir / f"{r.accuracy:.3f}_{r.image_id}.jpg"
            if render_overlay(dataset.resolve_image_path(img.meta), img.corners, r, out):
                n += 1
        print(f"wrote {n} overlay(s) to {overlay_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
