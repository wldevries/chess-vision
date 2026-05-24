"""Run the trained box detector on the capture photos and score box-bottom → square.

This is the real-world baseline the keypoint head must beat. For each capture image
(those with all 4 hand-marked corners), we run `runs/detector/best.pt`, derive each
box's contact point as the bottom-center (`bbox_base_point`, the doctrine's weak link),
map it through the corner homography to a grid cell, and compare against the
hand-corrected capture keypoints. Both predictions and labels go through the SAME
R0 homography, so cell identity is consistent without needing per-photo orientation.

Outputs (under --out-dir): summary.json (overall + per session), per_piece.csv, and
overlays/*.jpg (9x9 grid + base points; green = right cell & class, red/orange = wrong).

Usage:
    uv run python scripts/eval_detector_on_captures.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from chessvision.data.captures import PIECE_FEN, CaptureDataset
from chessvision.data.detection import LABEL_NAMES
from chessvision.detector import load_detector
from chessvision.geometry import (
    Orientation,
    bbox_base_point,
    compute_homography,
    lattice_points,
    square_for_point,
)


def _ls_label(name: str) -> str:
    return "".join(word.capitalize() for word in name.split("-"))


# detector label (1..12) -> FEN char, to compare with capture PieceKeypoint.fen
DETECTOR_LABEL_TO_FEN: dict[int, str] = {
    i: PIECE_FEN[_ls_label(LABEL_NAMES[i])] for i in range(1, len(LABEL_NAMES))
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--captures", type=Path, default=Path("data/captures/label-studio.json"))
    add("--ckpt", type=Path, default=Path("runs/detector/best.pt"))
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--score-thresh", type=float, default=0.5)
    add("--tol", type=float, default=0.06, help="off-board tolerance (canonical units)")
    add("--offset", type=float, default=0.05, help="base-point vertical offset (bbox-height frac)")
    add("--limit", type=int, default=None, help="process at most N images")
    add("--out-dir", type=Path, default=Path("runs/captures_eval"))
    add("--overlay-max", type=int, default=1600, help="overlay long-side px (0 disables overlays)")
    return p.parse_args(argv)


@torch.no_grad()
def detect(model, image_rgb: np.ndarray, device, score_thresh: float):
    t = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
    out = model([t.to(device)])[0]
    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep = scores >= score_thresh
    return boxes[keep], labels[keep], scores[keep]


def predicted_cells(
    boxes, labels, scores, homography, offset, tol
) -> dict[str, tuple[str, float, tuple]]:
    """cell -> (fen, score, base_xy); on collision keep the highest-scoring box."""
    pred: dict[str, tuple[str, float, tuple]] = {}
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores, strict=True):
        base = bbox_base_point((x1, y1, x2 - x1, y2 - y1), vertical_offset=offset)
        cell = square_for_point(homography, base, tol)
        if cell is None:
            continue
        if cell not in pred or sc > pred[cell][1]:
            pred[cell] = (DETECTOR_LABEL_TO_FEN[int(lab)], float(sc), base)
    return pred


def true_cells(sample, homography, tol) -> dict[str, tuple[str, tuple]]:
    """cell -> (fen, point) from the hand-corrected keypoints."""
    truth: dict[str, tuple[str, tuple]] = {}
    for kp in sample.pieces:
        cell = square_for_point(homography, kp.point, tol)
        if cell is not None:
            truth[cell] = (kp.fen, kp.point)
    return truth


def draw_overlay(image_rgb, homography, pred, truth, max_size: int) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    grid = lattice_points(homography).reshape(9, 9, 2).round().astype(np.int32)
    for i in range(9):
        cv2.polylines(img, [np.ascontiguousarray(grid[i])], False, (0, 255, 255), 1)
        cv2.polylines(img, [np.ascontiguousarray(grid[:, i])], False, (0, 255, 255), 1)
    # labels: hollow circle, green if correctly classified, red otherwise
    for cell, (fen, pt) in truth.items():
        ok = cell in pred and pred[cell][0] == fen
        cv2.circle(img, (round(pt[0]), round(pt[1])), 12, (0, 200, 0) if ok else (0, 0, 255), 2)
    # predictions: filled base point, green if matches a true cell+class, orange otherwise
    for cell, (fen, _sc, base) in pred.items():
        ok = cell in truth and truth[cell][0] == fen
        cv2.circle(
            img, (round(base[0]), round(base[1])), 6, (0, 200, 0) if ok else (0, 140, 255), -1
        )
    if max_size and max(img.shape[:2]) > max_size:
        s = max_size / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    model = load_detector(args.ckpt, device)
    dataset = CaptureDataset.load(args.captures)
    samples = [s for s in dataset.with_all_corners()]
    if args.limit:
        samples = samples[: args.limit]

    overlay_dir = args.out_dir / "overlays"
    if args.overlay_max:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # per-image stats
    agg = defaultdict(lambda: defaultdict(int))  # session -> counters
    for s in samples:
        homography = compute_homography(s.corners, Orientation.R0)
        image = s.load_image(dataset.s3)
        boxes, labels, scores = detect(model, image, device, args.score_thresh)
        pred = predicted_cells(boxes, labels, scores, homography, args.offset, args.tol)
        truth = true_cells(s, homography, args.tol)

        n_true, n_pred = len(truth), len(pred)
        occupied_found = sum(1 for c in truth if c in pred)
        class_correct = sum(1 for c in truth if c in pred and pred[c][0] == truth[c][0])
        extras = sum(1 for c in pred if c not in truth)
        board_correct = int(
            set(truth) == set(pred) and all(pred[c][0] == truth[c][0] for c in truth)
        )
        rows.append(
            {
                "task_id": s.task_id,
                "session": s.session,
                "n_true": n_true,
                "n_pred": n_pred,
                "occupied_found": occupied_found,
                "class_correct": class_correct,
                "extras": extras,
                "board_correct": board_correct,
            }
        )
        for key in (
            "n_true",
            "n_pred",
            "occupied_found",
            "class_correct",
            "extras",
            "board_correct",
        ):
            agg[s.session][key] += rows[-1][key]
            agg["__all__"][key] += rows[-1][key]
        agg[s.session]["n_images"] += 1
        agg["__all__"]["n_images"] += 1

        if args.overlay_max:
            ov = draw_overlay(image, homography, pred, truth, args.overlay_max)
            cv2.imwrite(str(overlay_dir / f"{s.session}_{s.task_id}.jpg"), ov)

    def rates(c: dict) -> dict:
        nt, npd = c["n_true"], c["n_pred"]
        return {
            "n_images": c["n_images"],
            "n_true": nt,
            "n_pred": npd,
            "recall_occupied": round(c["occupied_found"] / nt, 4) if nt else None,
            "recall_class": round(c["class_correct"] / nt, 4) if nt else None,
            "precision_class": round(c["class_correct"] / npd, 4) if npd else None,
            "extras": c["extras"],
            "board_correct": c["board_correct"],
            "board_acc": round(c["board_correct"] / c["n_images"], 4) if c["n_images"] else None,
        }

    summary = {
        "params": {
            "ckpt": str(args.ckpt),
            "score_thresh": args.score_thresh,
            "tol": args.tol,
            "offset": args.offset,
            "n_images": len(samples),
        },
        "overall": rates(agg["__all__"]),
        "per_session": {k: rates(v) for k, v in sorted(agg.items()) if k != "__all__"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "per_piece.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(summary, indent=2))
    if args.overlay_max:
        print(f"overlays -> {overlay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
