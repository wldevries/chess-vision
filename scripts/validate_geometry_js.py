"""Dump reference geometry (Python) so the JS port (web/geometry.js) can be checked against it.

Simulates the corner model: project the 81 canonical lattice points through a realistic board
homography to get 81 "predicted" image points, refit H with the SAME routine the JS ports
(corner_regressor.homography_from_lattice, unweighted), and record square_for_point for a set of
probe points. Writes web/_geom_ref.json; scripts/_validate_geometry.mjs reproduces it in JS.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chessvision.corner_regressor import homography_from_lattice
from chessvision.data.corners import LATTICE_CANONICAL
from chessvision.geometry import compute_homography, square_for_point


def main() -> int:
    # A realistic tilted-board quad in a 4000x3000 image (TL, TR, BR, BL).
    corners = {
        "top_left": [820.0, 760.0],
        "top_right": [3180.0, 690.0],
        "bottom_right": [3650.0, 2540.0],
        "bottom_left": [360.0, 2610.0],
    }
    h_corners = compute_homography(corners)  # canonical -> image (R0)
    pts81 = cv2.perspectiveTransform(
        LATTICE_CANONICAL.reshape(-1, 1, 2).astype(np.float32), h_corners
    ).reshape(-1, 2)

    h_lat = homography_from_lattice(pts81.astype(np.float32), None)  # the routine JS ports

    # Probe points across the board (centers of a few squares) + one off-board point.
    probes = []
    for sq in ["a8", "e4", "h1", "d5", "a1", "h8", "c6"]:
        from chessvision.geometry import canonical_to_image, square_center_uv

        uv = np.array([square_center_uv(sq)], dtype=np.float32)
        xy = canonical_to_image(h_lat, uv)[0]
        probes.append({"pt": [float(xy[0]), float(xy[1])], "expect": sq})
    probes.append(
        {"pt": [50.0, 50.0], "expect": square_for_point(h_lat, [50.0, 50.0])}
    )  # off-board -> None

    ref = {
        "points81": [[float(x), float(y)] for x, y in pts81],
        "H_py": [float(v) for v in h_lat.reshape(-1)],
        "probes": [{**p, "square_py": square_for_point(h_lat, p["pt"])} for p in probes],
    }
    out = Path("web/_geom_ref.json")
    out.write_text(json.dumps(ref), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
