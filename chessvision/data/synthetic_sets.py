"""External synthetic chess sets as keypoint-detection PRETRAINING data.

Two Blender-rendered sets under ``data/othersets/`` are wrapped here so they emit
the *identical* ``(image, target)`` shape as
``chessvision.data.detection.ChessReDKeypointDetection`` -- same labels ``1..12``
(``_PIECE_NAMES`` order), same ``keypoints (N,1,3)`` with visibility 2, same
``collate_detection``. That means they concatenate into the existing training
stack with no remapping.

- **Chesscog** (``Chesscog/{train,val,test}/*.png`` + ``.json``): oblique
  viewpoint, single piece set, **has** per-piece boxes. The a8-corner orientation
  **varies per image**, so it is resolved per image by choosing the rotation whose
  projected square-centers best agree with the box centers.
- **SyntheticBoards / "thefamousrat"** (``synthetic-chess-board-images/data/*.jpg``
  + ``.json``): top-down, **varied boards and piece materials** (the overfit
  antidote), centered pieces, **no boxes**. Boxes are *fabricated* from the
  homography. That is acceptable **only** because these feed a pretraining stage
  whose job is to warm the trunk on varied appearance; the downstream finetune on
  real ChessReD/store boxes repairs box precision.

In both cases the contact-keypoint is the square center projected through the
homography (doctrine-pure), **never** read off a box (see CLAUDE.md anti-patterns).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from chessvision.data.detection import DetectionConfig, augment_targets, resize_targets
from chessvision.geometry import (
    Orientation,
    canonical_to_image,
    compute_homography,
    order_corners,
    square_center_uv,
)

# FEN letter -> detector label (1..12), matching detection._PIECE_NAMES order
# (white P,R,N,B,Q,K = 1..6; black p,r,n,b,q,k = 7..12). 0 stays background.
_FEN_TO_LABEL = {
    "P": 1, "R": 2, "N": 3, "B": 4, "Q": 5, "K": 6,
    "p": 7, "r": 8, "n": 9, "b": 10, "q": 11, "k": 12,
}  # fmt: skip

# "<type>_<color>" (thefamousrat `config` values) -> detector label.
_TYPE_IDX = {"pawn": 0, "rook": 1, "knight": 2, "bishop": 3, "queen": 4, "king": 5}


def _config_to_label(value: str) -> int:
    """Map a thefamousrat config value like 'knight_b' -> detector label 1..12."""
    type_name, color = value.rsplit("_", 1)
    return _TYPE_IDX[type_name] + 1 + (6 if color == "b" else 0)


def _to_tensor_target(rgb, boxes, labels, keypoints, image_id):
    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    target = {
        "boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
        "labels": torch.from_numpy(np.ascontiguousarray(labels)),
        "keypoints": torch.from_numpy(np.ascontiguousarray(keypoints)),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
    }
    return image, target


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resolve_orientation(corners: dict, squares: Sequence[str], box_centers: np.ndarray):
    """Pick the a8 orientation whose projected square centers best match box centers.

    Chesscog stores corners in an unnamed list and the physical a8 corner is not
    fixed across renders, so a single global orientation is wrong (~40% of boards).
    With boxes available we disambiguate per image: the correct rotation is the one
    whose square-center projections sit nearest the actual piece boxes.
    """
    uvs = np.array([square_center_uv(s) for s in squares], dtype=np.float32)
    best_o, best_err = Orientation.R0, float("inf")
    for o in Orientation:
        h = compute_homography(corners, o)
        proj = canonical_to_image(h, uvs)
        err = float(np.median(np.linalg.norm(proj - box_centers, axis=1)))
        if err < best_err:
            best_o, best_err = o, err
    return best_o


def _square_footprint_box(h: np.ndarray, square: str) -> list[float]:
    """Axis-aligned bbox of a square's projected polygon, nudged up for piece height.

    Used to FABRICATE a box for the box-less thefamousrat set. Each square spans
    1/8 in canonical coords; we project its four corners, take their bbox, then
    raise the top edge by ~0.6 square to crudely cover the standing piece. Good
    enough to define an RoI for pretraining (finetune fixes real extents).
    """
    u, v = square_center_uv(square)
    half = 1.0 / 16.0
    poly = np.array(
        [[u - half, v - half], [u + half, v - half], [u + half, v + half], [u - half, v + half]],
        dtype=np.float32,
    )
    pts = canonical_to_image(h, poly)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    height = y2 - y1
    return [float(x1), float(y1 - 0.6 * height), float(x2), float(y2)]


class _SyntheticKeypointBase(Dataset):
    """Shared resize/augment/tensorize for the two synthetic sets.

    Subclasses implement ``_load(idx) -> (rgb, boxes, labels, keypoints, image_id)``
    with boxes ``(N,4) xyxy``, labels ``(N,) int64``, keypoints ``(N,1,3)`` (vis 2).
    """

    def __init__(self, config: DetectionConfig | None = None, train: bool = False):
        self.config = config or DetectionConfig()
        self.train = train
        self.paths: list[Path] = []

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def __getitem__(self, idx: int):
        rgb, boxes, labels, keypoints, image_id = self._load(idx)
        rgb, boxes, keypoints = resize_targets(rgb, boxes, keypoints, self.config.max_size)
        if self.train:
            rgb, boxes, keypoints = augment_targets(
                rgb, boxes, keypoints,
                hflip_prob=self.config.hflip_prob, jitter=self.config.jitter,
                color=self.config.color, blur=self.config.blur, noise=self.config.noise,
            )  # fmt: skip
        return _to_tensor_target(rgb, boxes, labels, keypoints, image_id)


class ChesscogKeypointDetection(_SyntheticKeypointBase):
    """Chesscog split as box+contact-keypoint detection (orientation resolved per image)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        config: DetectionConfig | None = None,
        train: bool | None = None,
    ):
        super().__init__(config, train=(split == "train") if train is None else train)
        self.paths = sorted(Path(root, split).glob("*.json"))
        if not self.paths:
            raise FileNotFoundError(f"no chesscog labels under {Path(root, split)}")

    def _load(self, idx: int):
        label_path = self.paths[idx]
        d = json.loads(label_path.read_text(encoding="utf-8"))
        rgb = _read_rgb(label_path.with_suffix(".png"))

        pieces = d["pieces"]
        squares = [p["square"] for p in pieces]
        boxes_xywh = np.array([p["box"] for p in pieces], dtype=np.float32)  # x,y,w,h
        box_centers = boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2.0
        labels = np.array([_FEN_TO_LABEL[p["piece"]] for p in pieces], dtype=np.int64)

        corners = order_corners(d["corners"])
        orientation = _resolve_orientation(corners, squares, box_centers)
        h = compute_homography(corners, orientation)
        uvs = np.array([square_center_uv(s) for s in squares], dtype=np.float32)
        contacts = canonical_to_image(h, uvs)

        boxes = np.empty((len(pieces), 4), dtype=np.float32)
        boxes[:, :2] = boxes_xywh[:, :2]
        boxes[:, 2:] = boxes_xywh[:, :2] + boxes_xywh[:, 2:]  # xywh -> xyxy
        keypoints = np.concatenate(
            [contacts, np.full((len(pieces), 1), 2.0, dtype=np.float32)], axis=1
        ).reshape(-1, 1, 3)
        return rgb, boxes, labels, keypoints, idx


class SyntheticBoardsKeypointDetection(_SyntheticKeypointBase):
    """thefamousrat synthetic set: varied boards/materials, NO boxes (fabricated).

    Labels are ``config`` (square -> '<type>_<color>') plus normalized ``corners``.
    Pieces are centered, so the projected square-center contact label is exact;
    boxes are fabricated from the square footprint (pretraining only).
    """

    def __init__(
        self,
        root: str | Path,
        config: DetectionConfig | None = None,
        train: bool = True,
    ):
        super().__init__(config, train=train)
        # Skip the generator's stray `config.json` (cellsCoordinates/piecesTypes metadata,
        # not a per-image label) that also lives in data/.
        self.paths = sorted(
            p for p in Path(root, "data").glob("*.json") if p.name != "config.json"
        )
        if not self.paths:
            raise FileNotFoundError(f"no synthetic labels under {Path(root, 'data')}")

    def _load(self, idx: int):
        label_path = self.paths[idx]
        d = json.loads(label_path.read_text(encoding="utf-8"))
        cfg = d["config"]
        if isinstance(cfg, str):  # some records store the dict as a repr string
            cfg = json.loads(cfg.replace("'", '"'))
        rgb = _read_rgb(label_path.with_suffix(".jpg"))
        height, width = rgb.shape[:2]

        # Normalized corners -> pixels, in the file's native list order.
        raw = d["corners"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        pix = [[float(x) * width, float(y) * height] for x, y in raw]
        corners = order_corners(pix)

        # Square keys are uppercase in this set (e.g. 'A3'); geometry wants lowercase.
        squares = [s.lower() for s in cfg]
        labels = np.array([_config_to_label(v) for v in cfg.values()], dtype=np.int64)
        # No boxes to resolve orientation; centered pieces + this generator emit R0.
        h = compute_homography(corners, Orientation.R0)
        uvs = np.array([square_center_uv(s) for s in squares], dtype=np.float32)
        contacts = canonical_to_image(h, uvs)
        boxes = np.array([_square_footprint_box(h, s) for s in squares], dtype=np.float32)
        keypoints = np.concatenate(
            [contacts, np.full((len(squares), 1), 2.0, dtype=np.float32)], axis=1
        ).reshape(-1, 1, 3)
        return rgb, boxes, labels, keypoints, idx
