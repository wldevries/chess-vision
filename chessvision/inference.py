"""Live image -> FEN inference (Phase 4 glue).

Runs the trained contact-keypoint detector on a natural photo, maps each piece's
predicted **board-contact point** through the board homography to a square, and
emits a board-only FEN. Square assignment is done for all four board orientations
(R0..R270) in one pass: the homography maps a point to *a* square regardless of
orientation, so the four FENs are the same detection relabelled four ways. The
caller (UI) picks the one that reads correctly -- which physical corner is a8 is
not recoverable from geometry, so it stays a human/semantic choice.

See the contact-point anti-pattern: the square comes from the predicted keypoint,
**never** from a box bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chessvision.geometry import (
    CornerDict,
    Orientation,
    board_crop_bbox,
    compute_homography,
    order_corners,
    quad_area,
    squares_for_points,
)

# Detector class id (1..12) -> FEN letter, the inverse of capture_detection.fen_to_label
# (ChessReD category_id + 1: pawn,rook,knight,bishop,queen,king, white then black).
_FEN_ORDER = "PRNBQKprnbqk"

DEFAULT_SCORE_THRESH = 0.5
DEFAULT_TOL = 0.06  # ~half a square; tolerates a base point just past the far edge


def label_to_symbol(label: int) -> str:
    """Detector label in 1..12 -> FEN piece letter (e.g. 1 -> 'P', 12 -> 'k')."""
    return _FEN_ORDER[label - 1]


def board_fen_from_squares(placement: dict[str, str]) -> str:
    """Build the FEN piece-placement field from a {square: symbol} map.

    `placement` maps algebraic squares ("e4") to FEN letters ("P"/"n"/...). Pure
    string assembly (no python-chess dependency) so it stays trivially testable.
    Empty squares collapse to run-length digits, ranks 8..1 joined by '/'.
    """
    files = "abcdefgh"
    rows = []
    for rank in range(8, 0, -1):
        row, gap = "", 0
        for f in files:
            sym = placement.get(f"{f}{rank}")
            if sym is None:
                gap += 1
                continue
            if gap:
                row += str(gap)
                gap = 0
            row += sym
        if gap:
            row += str(gap)
        rows.append(row)
    return "/".join(rows)


def full_fen(board_fen: str, turn: str = "w") -> str:
    """Wrap a placement field into a complete FEN (for python-chess / SVG rendering).

    Side-to-move/castling/clocks are unknowable from a still photo; we fill neutral
    defaults so the string parses. Only the placement field carries real signal.
    """
    return f"{board_fen} {turn} - - 0 1"


@dataclass(frozen=True)
class DetectedPiece:
    symbol: str  # FEN letter; case = colour
    score: float
    point: tuple[float, float]  # predicted board-contact point, image pixels
    squares: dict[str, str | None]  # orientation name -> square (or None if off-board)


@dataclass(frozen=True)
class OrientationResult:
    board_fen: str  # placement field only
    fen: str  # full FEN (neutral side-to-move/clocks)
    n_placed: int  # pieces that landed on a square (after collision resolution)


@dataclass(frozen=True)
class PredictionResult:
    corners: CornerDict  # the input corners, sorted into TL/TR/BR/BL
    n_detected: int  # detections kept after the score threshold
    pieces: list[DetectedPiece]
    orientations: dict[str, OrientationResult]  # keyed by Orientation name


def _resolve_placement(
    symbols: list[str], scores: list[float], squares: list[str | None]
) -> dict[str, str]:
    """One piece per square: on collision keep the highest-scoring detection."""
    best: dict[str, tuple[float, str]] = {}
    for sym, score, sq in zip(symbols, scores, squares, strict=True):
        if sq is None:
            continue
        if sq not in best or score > best[sq][0]:
            best[sq] = (score, sym)
    return {sq: sym for sq, (_, sym) in best.items()}


def build_prediction(
    corners_in,
    points: np.ndarray,
    labels: list[int],
    scores: list[float],
    *,
    tol: float = DEFAULT_TOL,
) -> PredictionResult:
    """Turn raw detections (contact points + class labels) into per-orientation FENs.

    Pure geometry + string assembly -- no torch -- so the model-free path is unit
    testable. `points` is (N, 2) image pixels, `labels` the 1..12 class ids, `scores`
    the detection confidences. `corners_in` is any 4 points (or a CornerDict); we
    sort them into TL/TR/BR/BL via `order_corners`.
    """
    raw = list(corners_in.values()) if isinstance(corners_in, dict) else corners_in
    corners = order_corners(raw)
    symbols = [label_to_symbol(label) for label in labels]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    per_orient_squares: dict[str, list[str | None]] = {}
    orientations: dict[str, OrientationResult] = {}
    degenerate = quad_area(corners) < 1.0
    for orient in Orientation:
        if degenerate or len(pts) == 0:
            squares: list[str | None] = [None] * len(pts)
        else:
            homography = compute_homography(corners, orient)
            squares = squares_for_points(homography, pts, tol)
        per_orient_squares[orient.name] = squares
        placement = _resolve_placement(symbols, scores, squares)
        board_fen = board_fen_from_squares(placement)
        orientations[orient.name] = OrientationResult(
            board_fen=board_fen, fen=full_fen(board_fen), n_placed=len(placement)
        )

    pieces = [
        DetectedPiece(
            symbol=symbols[i],
            score=float(scores[i]),
            point=(float(pts[i, 0]), float(pts[i, 1])),
            squares={name: sq[i] for name, sq in per_orient_squares.items()},
        )
        for i in range(len(pts))
    ]
    return PredictionResult(
        corners=corners, n_detected=len(pts), pieces=pieces, orientations=orientations
    )


class LivePredictor:
    """Lazy-loaded keypoint detector that turns an RGB frame + corners into FENs.

    The model (default: the captures-finetuned checkpoint) is loaded on first use so
    importing this module -- and starting the web app -- stays cheap and torch-free
    until a prediction is actually requested.
    """

    def __init__(
        self,
        ckpt: str | Path = Path("runs/keypoint_captures/best.pt"),
        device: str | None = None,
        score_thresh: float = DEFAULT_SCORE_THRESH,
        tol: float = DEFAULT_TOL,
    ):
        self.ckpt = Path(ckpt)
        self._device = device
        self.score_thresh = score_thresh
        self.tol = tol
        self._model = None
        self._crop: dict = {}  # board-crop config read from the checkpoint (empty = no crop)

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch

        from chessvision.keypoint_detector import load_keypoint_detector, read_keypoint_preprocess

        if not self.ckpt.exists():
            raise FileNotFoundError(f"keypoint checkpoint not found: {self.ckpt}")
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = load_keypoint_detector(self.ckpt, self._device)
        # Match the framing the model trained on: a crop-trained checkpoint records its crop
        # config, so live inference slices the same way (else scale-shift, as in the eval path).
        self._crop = read_keypoint_preprocess(self.ckpt)

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return str(self._device)

    def predict(self, rgb: np.ndarray, corners_in) -> PredictionResult:
        """Detect pieces in `rgb` (H, W, 3 uint8) and map contacts -> squares -> FEN.

        `corners_in` is 4 image points in any order (sorted internally). A crop-trained
        checkpoint (recorded in its metadata) slices the frame to the board bbox first and
        the predicted contact points are translated back to full-frame pixels, so the
        homography (built from the full-frame corners) and the points share one frame.
        """
        import torch

        self._ensure_loaded()
        arr = np.ascontiguousarray(rgb)
        ox, oy = 0.0, 0.0
        if self._crop.get("board_crop"):
            corners = corners_in if isinstance(corners_in, dict) else order_corners(corners_in)
            h, w = arr.shape[:2]
            x0, y0, x1, y1 = board_crop_bbox(
                corners,
                w,
                h,
                side=self._crop.get("crop_side", 0.12),
                top=self._crop.get("crop_top", 0.30),
                bottom=self._crop.get("crop_bottom", 0.08),
            )
            arr = np.ascontiguousarray(arr[y0:y1, x0:x1])
            ox, oy = float(x0), float(y0)
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255).to(self._device)
        with torch.no_grad():
            out = self._model([t])[0]
        keep = out["scores"] >= self.score_thresh
        points = out["keypoints"][keep][:, 0, :2].cpu().numpy()
        points[:, 0] += ox  # crop frame -> full-frame, to match the full-frame corners/homography
        points[:, 1] += oy
        labels = out["labels"][keep].cpu().tolist()
        scores = out["scores"][keep].cpu().tolist()
        return build_prediction(corners_in, points, labels, scores, tol=self.tol)


class CornerPredictor:
    """Lazy-loaded board-corner regressor: an RGB frame -> 4 board corners.

    Mirrors `LivePredictor`'s lazy-load contract (torch + weights load on first
    `predict`, so importing this and starting the web app stays cheap). Used to
    *pre-fill* the corner-marking UI so the user nudges instead of placing from
    scratch -- the predicted corners are a `CornerDict` in native image pixels,
    ready for `geometry.order_corners` / `compute_homography`.
    """

    def __init__(self, ckpt: str | Path = Path("runs/corners/best.pt"), device: str | None = None):
        self.ckpt = Path(ckpt)
        self._device = device
        self._model = None
        self._is_lattice = False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch

        from chessvision.corner_regressor import load_corner_regressor

        if not self.ckpt.exists():
            raise FileNotFoundError(f"corner checkpoint not found: {self.ckpt}")
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = load_corner_regressor(self.ckpt, self._device)
        # An 81-point lattice checkpoint is decoded to corners by the (parameter-free) lattice
        # head -- predict 81 grid points, fit H robustly over all of them, read off 4 corners.
        # Auto-detected from num_corners so the app/UI path is identical for either model.
        self._is_lattice = getattr(self._model, "num_corners", 4) != 4

    def predict(self, rgb: np.ndarray) -> CornerDict:
        """Predict the 4 board corners for `rgb` (H, W, 3 uint8), in native pixels.

        For a lattice checkpoint this runs the robust 81-point->H->corners decode; for a
        4-corner checkpoint it's the direct soft-argmax. Same `CornerDict` either way.
        """
        self._ensure_loaded()
        rgb = np.ascontiguousarray(rgb)
        if self._is_lattice:
            from chessvision.corner_regressor import corners_from_lattice

            return corners_from_lattice(self._model, rgb, device=self._device)

        from chessvision.corner_regressor import predict_corners

        return predict_corners(self._model, rgb, self._device)

    @property
    def is_lattice(self) -> bool:
        self._ensure_loaded()
        return self._is_lattice
