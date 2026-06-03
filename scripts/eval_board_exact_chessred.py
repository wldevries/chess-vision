"""Board-exact / per-square FEN eval on the ChessReD (chessred2k) test split.

Reports the *same* metrics as "End-to-End Chess Recognition" (Masouris & van Gemert,
VISAPP 2024) so our pipeline is directly comparable to their numbers on ChessReD:

    board_exact   : fraction of boards with all 64 squares correct        (paper: 15.26%)
    board_le1     : fraction of boards with <= 1 wrong square              (paper: 25.92%)
    per_sq_error  : wrong squares / (64 * boards)                          (paper:  5.31%)
    mean_wrong    : mean wrong squares per board                           (paper:  3.40)

A "square" is one of 64; its class is one of 13 (12 pieces + empty). GT comes straight
from ChessReD's per-piece `square` + `category_id` labels (no homography needed for truth);
the *prediction* runs our keypoint detector, maps each contact keypoint through the board
homography to a square, and takes the highest-scoring detection per square (empty if none).

Two square-assignment homographies, printed side by side (same frames, same metrics):
  - GT-corner ceiling : homography from the labelled corners -> isolates piece recognition.
  - end-to-end        : homography from the CORNER MODEL's corners -> the real image->FEN
                        number (the gap is the cost of board localization). Skipped with
                        --no-corner.

Orientation: ChessReD corners are labelled in the canonical (white-perspective) frame, so
R0 (top_left == a8) should reproduce the GT squares. We default to R0 and *also* report an
oracle that picks, per board, the rotation (R0/R90/R180/R270) maximising agreement with GT
-- if oracle == R0, orientation is a non-issue on ChessReD and the R0 number stands alone.

    uv run python scripts/eval_board_exact_chessred.py --device cuda          # chessred2k test
    uv run python scripts/eval_board_exact_chessred.py --full --device cuda   # full 2129-img test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from chessvision.corner_regressor import corners_from_lattice, load_corner_regressor
from chessvision.data.chessred import ChessReD
from chessvision.data.detection import resize_targets
from chessvision.geometry import (
    FILES,
    Orientation,
    board_crop_bbox,
    compute_homography,
    order_corners,
    square_for_point,
)
from chessvision.keypoint_detector import load_keypoint_detector, read_keypoint_preprocess

EMPTY = 12  # the 13th class; matches ChessReD's empty category id
ALL_SQUARES = [f"{f}{r}" for r in range(1, 9) for f in FILES]  # 64 algebraic squares
ORIENTS = tuple(Orientation)


def gt_board(ds: ChessReD, image_id: int) -> dict[str, int]:
    """square -> category_id (0..11) for the occupied squares; empties are absent."""
    return {p.square: p.category_id for p in ds.pieces(image_id) if p.square}


@torch.no_grad()
def detect(
    model: torch.nn.Module,
    rgb: np.ndarray,
    device: torch.device,
    *,
    max_size: int,
    score_thresh: float,
) -> list[tuple[float, float, float, int]]:
    """Run the keypoint detector on a (possibly cropped) RGB frame; return one
    (x, y, score, category_id) per detection above threshold, in that frame's pixels.
    `x, y` is the predicted board-contact keypoint (never a box bottom)."""
    rgb, _, _ = resize_targets(rgb, np.zeros((0, 4), np.float32), None, max_size)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255
    out = model([image])[0]
    dets = []
    for score, label, kps in zip(
        out["scores"].tolist(), out["labels"].tolist(), out["keypoints"].cpu().numpy(), strict=True
    ):
        if score >= score_thresh:
            dets.append((float(kps[0, 0]), float(kps[0, 1]), float(score), int(label) - 1))
    return dets


def assign_board(
    corners: dict, orient: Orientation, dets: list[tuple[float, float, float, int]]
) -> dict[str, int]:
    """square -> predicted category_id, keeping the highest-scoring detection per square."""
    homography = compute_homography(corners, orient)
    best: dict[str, tuple[float, int]] = {}
    for x, y, score, cat in dets:
        sq = square_for_point(homography, (x, y))
        if sq is not None and (sq not in best or score > best[sq][0]):
            best[sq] = (score, cat)
    return {sq: cat for sq, (_, cat) in best.items()}


def board_wrong(gt: dict[str, int], pred: dict[str, int]) -> int:
    """Wrong squares out of 64: gt/pred class differ (empty == EMPTY)."""
    return sum(gt.get(sq, EMPTY) != pred.get(sq, EMPTY) for sq in ALL_SQUARES)


def best_orient_wrong(gt: dict[str, int], by_orient: dict[Orientation, dict[str, int]]) -> int:
    return min(board_wrong(gt, pred) for pred in by_orient.values())


def summarize(wrongs: list[int]) -> dict[str, float]:
    n = max(len(wrongs), 1)
    total = sum(wrongs)
    return {
        "boards": len(wrongs),
        "board_exact": sum(w == 0 for w in wrongs) / n,
        "board_le1": sum(w <= 1 for w in wrongs) / n,
        "per_sq_error": total / (64 * n),
        "mean_wrong": total / n,
    }


def corners_native(ds: ChessReD, image_id: int) -> dict[str, tuple[float, float]]:
    raw = ds.corners(image_id)
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def scale_corners(corners: dict, ox: float, oy: float, scale: float) -> dict:
    return {k: ((x - ox) * scale, (y - oy) * scale) for k, (x, y) in corners.items()}


def map_pred_to_gt_keys(
    pred_pts: list[tuple[float, float]], gt_corners: dict
) -> dict[str, tuple[float, float]]:
    """Assign each predicted corner point to its nearest GT corner KEY, so the predicted
    homography lands in the same orientation frame as GT (orientation is not a corner task)."""
    mapped, used = {}, set()
    for key, (gx, gy) in gt_corners.items():
        j = min(
            (i for i in range(len(pred_pts)) if i not in used),
            key=lambda i: (pred_pts[i][0] - gx) ** 2 + (pred_pts[i][1] - gy) ** 2,
        )
        used.add(j)
        mapped[key] = pred_pts[j]
    return mapped


@torch.no_grad()
def run(
    kp_model,
    corner_model,
    ds: ChessReD,
    image_ids: list[int],
    device,
    *,
    max_size: int,
    score_thresh: float,
    board_crop: bool,
    crop: dict,
    full_mode: bool = False,
) -> dict:
    """Walk the split once.

    chessred2k mode (default): per-board wrong-square counts for the GT-corner ceiling
    (R0 + oracle orientation) and, if a corner model is given, the end-to-end pipeline
    (corners from the model, orientation frame anchored to the GT corner keys).

    full mode (`full_mode=True`, the 2,129-image full test split): NO GT corners exist, so
    there is no ceiling -- only the true image->FEN pipeline. Corners come from the model and
    are sorted into visual TL/TR/BR/BL slots by `order_corners`; since the physical-a8 corner
    is not recoverable from geometry, we score each board under R0 *and* under the best-of-4
    rotation (oracle) -- the oracle is the fair "orientation resolved" number (the paper's
    model gets orientation for free from its training labels; our app makes it a manual toggle).
    """
    res = {k: [] for k in ("ceiling_r0", "ceiling_oracle", "e2e_r0", "e2e_oracle")}
    occ_total = occ_localized = occ_class = 0  # occupied-square recall / class acc (best path)
    for image_id in image_ids:
        gt = gt_board(ds, image_id)
        if not gt:
            continue
        path = ds.resolve_image_path(ds.meta(image_id))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb_full.shape[:2]

        if not full_mode:
            # ---- GT-corner ceiling (chessred2k only) ----
            gt_corners = corners_native(ds, image_id)
            ox = oy = 0.0
            rgb_src = rgb_full
            if board_crop:
                x0, y0, x1, y1 = board_crop_bbox(gt_corners, w, h, **crop)
                rgb_src, ox, oy = rgb_full[y0:y1, x0:x1], float(x0), float(y0)
            scale = min(1.0, max_size / max(rgb_src.shape[:2]))
            corners_c = scale_corners(gt_corners, ox, oy, scale)
            dets = detect(kp_model, rgb_src, device, max_size=max_size, score_thresh=score_thresh)
            by_orient = {o: assign_board(corners_c, o, dets) for o in ORIENTS}
            pred_r0 = by_orient[Orientation.R0]
            res["ceiling_r0"].append(board_wrong(gt, pred_r0))
            res["ceiling_oracle"].append(best_orient_wrong(gt, by_orient))
            for sq, cat in gt.items():  # occupied-square diagnostics off the ceiling-R0 path
                occ_total += 1
                if sq in pred_r0:
                    occ_localized += 1
                    occ_class += int(pred_r0[sq] == cat)

        # ---- end-to-end (corner model) ----
        if corner_model is None:
            continue
        pred = corners_from_lattice(corner_model, rgb_full, device=device, use_conf=True)
        pred_pts = [(float(x), float(y)) for x, y in pred.values()]
        ox = oy = 0.0
        rgb_src = rgb_full
        if board_crop:
            x0, y0, x1, y1 = board_crop_bbox(dict(enumerate(pred_pts)), w, h, **crop)
            rgb_src, ox, oy = rgb_full[y0:y1, x0:x1], float(x0), float(y0)
        scale = min(1.0, max_size / max(rgb_src.shape[:2]))
        # Orientation frame: anchor to GT corner keys when we have them (chessred2k), else
        # fall back to pixel-sorted slots (full set) and let the oracle pick the rotation.
        if full_mode:
            corners_e = order_corners(pred_pts)
        else:
            corners_e = map_pred_to_gt_keys(pred_pts, corners_native(ds, image_id))
        corners_e = scale_corners(corners_e, ox, oy, scale)
        try:
            _ = compute_homography(corners_e, Orientation.R0)
        except Exception:
            res["e2e_r0"].append(64)  # degenerate quad -> whole board wrong
            res["e2e_oracle"].append(64)
            continue
        dets = detect(kp_model, rgb_src, device, max_size=max_size, score_thresh=score_thresh)
        by_orient = {o: assign_board(corners_e, o, dets) for o in ORIENTS}
        res["e2e_r0"].append(board_wrong(gt, by_orient[Orientation.R0]))
        res["e2e_oracle"].append(best_orient_wrong(gt, by_orient))
        if full_mode:  # diagnostics off the oracle-best e2e path (no ceiling in full mode)
            best_pred = min(by_orient.values(), key=lambda pr: board_wrong(gt, pr))
            for sq, cat in gt.items():
                occ_total += 1
                if sq in best_pred:
                    occ_localized += 1
                    occ_class += int(best_pred[sq] == cat)

    out = {
        "localization": occ_localized / max(occ_total, 1),
        "class_acc": occ_class / max(occ_total, 1),
    }
    if not full_mode:
        out["ceiling_r0"] = summarize(res["ceiling_r0"])
        out["ceiling_oracle"] = summarize(res["ceiling_oracle"])
        out["orient_disagreements"] = sum(
            a != b for a, b in zip(res["ceiling_r0"], res["ceiling_oracle"], strict=True)
        )
    if corner_model is not None:
        out["e2e_r0"] = summarize(res["e2e_r0"])
        out["e2e_oracle"] = summarize(res["e2e_oracle"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--chessred", type=Path, default=Path("data/othersets/ChessReD"))
    add("--split", default="test", choices=("train", "val", "test"))
    add("--keypoint-ckpt", type=Path, default=Path("runs/keypoint_joint_aug_crop/best.pt"))
    add("--corner-ckpt", type=Path, default=Path("runs/corners/best.pt"))
    add("--no-corner", action="store_true", help="skip the end-to-end (corner-model) pass")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--max-size", type=int, default=1333)
    add("--score-thresh", type=float, default=0.5)
    add(
        "--board-crop",
        choices=("auto", "on", "off"),
        default="auto",
        help="board-crop framing; 'auto' reads the flag stamped in the checkpoint",
    )
    add("--limit", type=int, default=0, help="cap #images (debug)")
    add(
        "--full",
        action="store_true",
        help="full ChessReD test split (2129 imgs, no GT corners) -- the true paper-comparable set",
    )
    args = p.parse_args(argv)

    device = torch.device(args.device)
    ds = ChessReD.load(args.chessred)
    if args.full:
        node = ds._splits[args.split]
        image_ids = list(node["image_ids"] if isinstance(node, dict) else node)
        if args.no_corner:
            p.error("--full needs the corner model (no GT corners exist); drop --no-corner")
    else:
        image_ids = ds.chessred2k_split(args.split)
    if args.limit:
        image_ids = image_ids[: args.limit]

    pre = read_keypoint_preprocess(args.keypoint_ckpt)
    if args.board_crop == "auto":
        board_crop = bool(pre.get("board_crop", False))
    else:
        board_crop = args.board_crop == "on"
    crop = dict(
        side=pre.get("crop_side", 0.12),
        top=pre.get("crop_top", 0.30),
        bottom=pre.get("crop_bottom", 0.08),
    )
    label = f"FULL {args.split} (no GT corners)" if args.full else f"chessred2k/{args.split}"
    print(
        f"{label}: {len(image_ids)} images | ckpt={args.keypoint_ckpt} | "
        f"board_crop={board_crop} | score_thresh={args.score_thresh}"
    )

    kp = load_keypoint_detector(args.keypoint_ckpt, device)
    corner = None if args.no_corner else load_corner_regressor(args.corner_ckpt, device=device)

    out = run(
        kp,
        corner,
        ds,
        image_ids,
        device,
        max_size=args.max_size,
        score_thresh=args.score_thresh,
        board_crop=board_crop,
        crop=crop,
        full_mode=args.full,
    )

    paper = {"board_exact": 0.1526, "board_le1": 0.2592, "per_sq_error": 0.0531, "mean_wrong": 3.40}
    keys = ("board_exact", "board_le1", "per_sq_error", "mean_wrong")
    # Columns present depend on mode: full = e2e R0/oracle only; chessred2k = + ceiling.
    cols = [("paper(E2E)", paper)]
    if not args.full:
        cols += [("ceiling", out["ceiling_r0"]), ("ceil-oracle", out["ceiling_oracle"])]
    if corner is not None:
        cols += [("e2e-R0", out["e2e_r0"]), ("e2e-oracle", out["e2e_oracle"])]
    print()
    print(f"{'metric':<14}" + "".join(f"{name:>13}" for name, _ in cols))
    for k in keys:
        print(f"{k:<14}" + "".join(f"{src[k]:>13.4f}" for _, src in cols))
    print()
    diag = f"occupied-square recall {out['localization']:.4f} | class_acc {out['class_acc']:.4f}"
    if not args.full:
        diag += (
            f" | orientation R0!=oracle on "
            f"{out['orient_disagreements']}/{out['ceiling_r0']['boards']} boards"
        )
    print(diag)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
