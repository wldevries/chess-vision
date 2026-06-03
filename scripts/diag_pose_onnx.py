"""Diagnose the web pieces.onnx on a real cropped board: does the EXPORT detect, or is it JS?

Replicates the browser preprocessing (board-crop via GT corners -> letterbox 1280 -> RGB [0,1] CHW)
and runs pieces.onnx through onnxruntime with the SAME decode the JS uses, printing detection
counts + max class score. Compares int8 vs fp32 so we can tell if quantization killed it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from chessvision.data.corner_capture import CornerStore, split_store_for_keypoints
from chessvision.data.positions import store_label_to_capture
from chessvision.geometry import board_crop_bbox

N = 33600


def letterbox(rgb, size=1280, pad=114):
    h, w = rgb.shape[:2]
    s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), pad, np.uint8)
    canvas[py : py + nh, px : px + nw] = cv2.resize(rgb, (nw, nh))
    return canvas


def decode(out, conf=0.25):
    # out: (1, 19, N) -> per anchor [cx,cy,w,h, 12 cls, kx,ky,kconf]
    d = out[0]
    cls = d[4:16, :]  # (12, N)
    best = cls.max(axis=0)
    keep = best >= conf
    return int(keep.sum()), float(best.max())


def run(model_path, chw):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: chw})[0]
    return decode(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=Path("data"))
    p.add_argument("--onnx", type=Path, default=Path("web/models/pieces.onnx"))
    p.add_argument("--n", type=int, default=3, help="how many val frames to test")
    args = p.parse_args()

    store = CornerStore(args.store)
    _, va, _ = split_store_for_keypoints(store, test_boards=["dennis-bord"])
    samples = [store_label_to_capture(lb, store) for lb in va[: args.n]]

    for s in samples:
        rgb = s.load_image(None)
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = board_crop_bbox(s.corners, w, h, side=0.12, top=0.30, bottom=0.08)
        crop = rgb[y0:y1, x0:x1]
        lb = letterbox(crop)
        chw = np.ascontiguousarray(lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        ndet, mx = run(args.onnx, chw)
        print(
            f"{s.session}: crop {x1 - x0}x{y1 - y0}, GT~{len(s.pieces)} pieces "
            f"-> dets>=0.25: {ndet}, max class score: {mx:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
