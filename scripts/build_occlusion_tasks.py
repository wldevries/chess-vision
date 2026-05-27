"""Mirror chessred2k to MinIO (downscaled) and build a Label Studio review set.

Two jobs:
  1. **Mirror** chessred2k images to MinIO, downscaled to `--max-size`, preserving
     ChessReD's layout: `s3://<bucket>/<prefix>/images/<game>/<file>.jpg`. This is
     the canonical S3 copy of the dataset.
  2. **Review tasks** for the most *base-occluded* images (a piece whose contact
     point sits inside a nearer piece's bbox -- where box-bottom fails, see
     chessvision.data.contact). Each task is a Label Studio import with predictions:
       - 4 board corners (from ChessReD), and
       - one keypoint per piece at its **contact point** (square center projected
         through the homography -- box-independent, the doctrine-pure base point),
     so a human only *corrects* the hard points. Tasks reference the mirrored URIs.

Contact points are stored as percentages, which are resolution-independent, so
downscaling for upload does not move any predicted point; we just report the
downscaled `original_width/height` (the dimensions Label Studio actually loads).

Usage:
    # 1) eyeball predictions locally first (no S3, no LS):
    uv run python scripts/build_occlusion_tasks.py --data-root <dir> --n 8 \
        --overlay-dir runs/occlusion_overlays --no-tasks
    # 2) mirror the whole subset + write review tasks (needs MinIO creds in .env):
    uv run python scripts/build_occlusion_tasks.py --data-root <dir> --mirror-all --upload --n 80
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from chessvision.data.chessred import AnnotatedImage, ChessReD
from chessvision.data.contact import contact_points, occluded_pieces
from chessvision.data.storage import StorageConfig, get_client

_CORNER_LS = {
    "top_left": "TopLeft",
    "top_right": "TopRight",
    "bottom_right": "BottomRight",
    "bottom_left": "BottomLeft",
}


def ls_piece_label(chessred_name: str) -> str:
    """'white-pawn' -> 'WhitePawn' (the capture LS piece-label spelling)."""
    return "".join(word.capitalize() for word in chessred_name.split("-"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = p.add_argument
    add("--data-root", required=True, type=Path, help="ChessReD dir (has annotations.json)")
    add("--images-root", type=Path, default=None, help="image tree root override")
    add("--n", type=int, default=80, help="number of most-occluded images to make review tasks for")
    add("--min-occlusion", type=int, default=1, help="require at least this many occluded pieces")
    add("--prefix", default="chessred2k", help="S3 key prefix; mirror lands at <prefix>/images/...")
    add("--max-size", type=int, default=1600, help="downscale long side before upload")
    add("--bucket", default=os.getenv("MINIO_BUCKET", "chess"))
    add("--model-version", default="contact-v1")
    add("--out", type=Path, default=Path("data/occlusion/label-studio-import.json"))
    add("--upload", action="store_true", help="actually upload to MinIO (default: dry run)")
    add("--mirror-all", action="store_true", help="upload ALL chessred2k, not just tasked images")
    add("--overlay-dir", type=Path, default=None, help="also write local prediction overlays here")
    add(
        "--no-tasks",
        action="store_true",
        help="skip writing the LS task JSON (overlay/mirror only)",
    )
    return p.parse_args(argv)


def scaled_dims(w: int, h: int, max_size: int) -> tuple[int, int]:
    long = max(w, h)
    if long <= max_size:
        return w, h
    s = max_size / long
    return round(w * s), round(h * s)


def downscale(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    s = max_size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def s3_key(prefix: str, meta) -> str:
    """Mirror key preserving ChessReD layout: <prefix>/images/<game>/<file>.jpg."""
    rel = meta.path.split("images/", 1)[-1]  # "<game>/<file>.jpg"
    return f"{prefix}/images/{rel}"


def keypoint_result(from_name: str, label: str, pct_x: float, pct_y: float, w: int, h: int) -> dict:
    return {
        "from_name": from_name,
        "to_name": "image",
        "type": "keypointlabels",
        "original_width": w,
        "original_height": h,
        "value": {"x": pct_x, "y": pct_y, "width": 0.5, "keypointlabels": [label]},
    }


def build_task(
    image: AnnotatedImage,
    categories: dict[int, str],
    s3_uri: str,
    model_version: str,
    max_size: int,
) -> dict:
    """LS import task; percentages from original coords, dims = downscaled (what LS loads)."""
    ow, oh = image.meta.width, image.meta.height
    sw, sh = scaled_dims(ow, oh, max_size)
    results = [
        keypoint_result("corners", _CORNER_LS[k], 100.0 * xy[0] / ow, 100.0 * xy[1] / oh, sw, sh)
        for k, xy in image.corners.items()
        if k in _CORNER_LS
    ]
    for cp in contact_points(image):
        label = ls_piece_label(categories[cp.category_id])
        results.append(
            keypoint_result("pieces", label, 100.0 * cp.xy[0] / ow, 100.0 * cp.xy[1] / oh, sw, sh)
        )
    return {
        "data": {"image": s3_uri},
        "predictions": [{"model_version": model_version, "result": results}],
    }


def write_overlay(image: AnnotatedImage, src_path: Path, max_size: int, out_path: Path) -> bool:
    """Draw contact points (green) + corners (red) on the downscaled image."""
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = downscale(img, max_size)
    scale = img.shape[1] / image.meta.width
    for k, xy in image.corners.items():
        if k in _CORNER_LS:
            cv2.circle(img, (round(xy[0] * scale), round(xy[1] * scale)), 9, (0, 0, 255), 2)
    for cp in contact_points(image):
        cv2.circle(img, (round(cp.xy[0] * scale), round(cp.xy[1] * scale)), 5, (0, 255, 0), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img))


def upload(client, bucket: str, key: str, src_path: Path, max_size: int) -> None:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src_path)
    ok, buf = cv2.imencode(".jpg", downscale(img, max_size), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"failed to encode {src_path}")
    client.put_object(Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/jpeg")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chessred = ChessReD.load(args.data_root, args.images_root)

    scored = [(len(occluded_pieces(img)), img) for img in chessred.images_with_corners()]
    scored = [t for t in scored if t[0] >= args.min_occlusion]
    scored.sort(key=lambda t: t[0], reverse=True)
    review = scored[: args.n]
    print(
        f"{len(scored)} images with >= {args.min_occlusion} occluded piece(s); "
        f"review batch = top {len(review)} (occlusion {review[-1][0]}..{review[0][0]})"
    )

    client = None
    if args.upload:
        cfg = StorageConfig.try_from_env()
        if cfg is None:
            raise SystemExit("no MinIO creds in .env; cannot --upload")
        client = get_client(cfg)

    # What to physically upload: all chessred2k, or just the review batch.
    to_upload = (
        list(chessred.images_with_corners()) if args.mirror_all else [img for _, img in review]
    )
    if args.upload:
        for i, img in enumerate(to_upload, 1):
            upload(
                client,
                args.bucket,
                s3_key(args.prefix, img.meta),
                chessred.resolve_image_path(img.meta),
                args.max_size,
            )
            if i % 100 == 0 or i == len(to_upload):
                print(f"  uploaded {i}/{len(to_upload)}", flush=True)

    if args.overlay_dir:
        n = 0
        for occ, img in review:
            out = args.overlay_dir / f"occ{occ:02d}_{img.meta.file_name}"
            n += write_overlay(img, chessred.resolve_image_path(img.meta), args.max_size, out)
        print(f"wrote {n} overlay(s) -> {args.overlay_dir}")

    if not args.no_tasks:
        tasks = [
            build_task(
                img,
                chessred.categories,
                f"s3://{args.bucket}/{s3_key(args.prefix, img.meta)}",
                args.model_version,
                args.max_size,
            )
            for _, img in review
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        print(
            f"wrote {len(tasks)} review tasks -> {args.out}"
            + ("" if args.upload else "  (dry run: images not uploaded yet)")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
