"""ChessReD piece-detection dataset (Approach A, Phase 2).

Wraps the **chessred2k** subset -- the only ChessReD images carrying piece
bounding boxes -- as a torchvision-detection `Dataset`. Each item is
`(image, target)` where:

    image  : float32 CxHxW tensor in [0, 1] (the FasterRCNN transform normalizes)
    target : {"boxes":  (N, 4) float32 xyxy,
              "labels": (N,) int64 in 1..12,
              "image_id": (1,) int64}

Class ids are the ChessReD `category_id` (0..11) shifted by **+1** so id 0 stays
free for torchvision's implicit background class. `LABEL_NAMES[label]` recovers
the human name.

Splits come from ChessReD's official `chessred2k` train/val/test (see
`ChessReD.chessred2k_split`), so reported mAP is comparable to the literature and
free of game-level leakage.

Images are 3072x3072; we downscale so the long side is <= `max_size` and scale
the boxes to match. This is the resize the detector's own transform would do
anyway -- doing it here just saves decode/copy cost. Set `train=True` to enable
light augmentation (horizontal flip + photometric jitter) for generalization
(plan.md section 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from chessvision.data.chessred import AnnotatedImage, ChessReD
from chessvision.data.contact import contact_points
from chessvision.geometry import board_crop_bbox

# ChessReD category_id 0..11 are the 12 pieces (12 == "empty", which has no box).
# Detector label = category_id + 1; label 0 is torchvision's background.
NUM_PIECE_CLASSES = 12
NUM_CLASSES = NUM_PIECE_CLASSES + 1  # + background, for the FastRCNNPredictor head
_PIECE_NAMES = (
    "white-pawn",
    "white-rook",
    "white-knight",
    "white-bishop",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-rook",
    "black-knight",
    "black-bishop",
    "black-queen",
    "black-king",
)
# Detector label (1..12) -> name; 0 -> background.
LABEL_NAMES: dict[int, str] = {
    0: "__background__",
    **{i + 1: n for i, n in enumerate(_PIECE_NAMES)},
}


@dataclass
class DetectionConfig:
    max_size: int = 1333  # long-side cap in pixels; boxes scaled to match
    hflip_prob: float = 0.0  # train-time horizontal flip probability
    jitter: float = 0.0  # train-time brightness/contrast jitter magnitude (0 disables)
    # Appearance-only corruptions (image untouched targets); each magnitude is a cap sampled
    # from zero, so a share of frames stay near-clean. See augment_targets.
    color: float = 0.0  # per-channel gain magnitude (white-balance / board-colour drift)
    blur: float = 0.0  # max Gaussian blur sigma in px (depth-of-field softening)
    motion_blur: float = 0.0  # max linear motion-blur kernel length px (directional camera shake)
    noise: float = 0.0  # max additive Gaussian noise std as a fraction of 255 (low-light grain)
    # Board crop: slice to the board bbox (+asymmetric margin) before resize, so pieces get more
    # pixels and clutter/padding shrink. MUST match the eval crop (geometry.board_crop_bbox).
    board_crop: bool = False
    crop_side: float = 0.12  # side margin as a fraction of the corner-bbox width
    crop_top: float = 0.30  # top margin (extra headroom for back-leaning pieces)
    crop_bottom: float = 0.08  # bottom margin


def collate_detection(batch):
    """Detection batches are ragged (variable #boxes), so keep them as tuples."""
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def resize_targets(
    rgb: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray | None, max_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Downscale so the long side is <= `max_size`, scaling boxes/keypoints to match.
    No-op (returns inputs) when the image already fits. Shared by the ChessReD and
    capture datasets so both scale targets identically."""
    h, w = rgb.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return rgb, boxes, keypoints
    rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if keypoints is not None and keypoints.size:
        keypoints = keypoints.copy()
        keypoints[:, :, :2] *= scale
    return rgb, boxes * scale, keypoints


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalized linear motion-blur kernel: a `length`-px streak rotated to `angle_deg`.
    Models directional camera shake (the real blur here is shake, not defocus)."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0  # horizontal streak
    center = ((length - 1) / 2.0, (length - 1) / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, m, (length, length))
    s = float(kernel.sum())
    return kernel / s if s > 0 else kernel


def apply_board_crop(
    rgb: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Crop `rgb` to `bbox` (x0,y0,x1,y1 from `geometry.board_crop_bbox`) and translate
    boxes/keypoints into the crop frame (boxes clipped to the crop; the contact keypoint is on
    the board so it always survives). Run BEFORE resize, on full-frame targets."""
    x0, y0, x1, y1 = bbox
    rgb = rgb[y0:y1, x0:x1]
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - x0, 0, x1 - x0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - y0, 0, y1 - y0)
    if keypoints is not None and keypoints.size:
        keypoints = keypoints.copy()
        keypoints[:, :, 0] -= x0
        keypoints[:, :, 1] -= y0
    return rgb, boxes, keypoints


def augment_targets(
    rgb: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray | None,
    *,
    hflip_prob: float,
    jitter: float,
    color: float = 0.0,
    blur: float = 0.0,
    motion_blur: float = 0.0,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Train-time augmentation. Shared by the ChessReD, capture, and synthetic datasets.

    Geometric (mirrors boxes/keypoints): horizontal flip. Appearance-only (image alone --
    the targets need no transform, so this is automatically "detection-safe"): brightness/
    contrast `jitter`, per-channel `color` gain (white-balance / board-colour drift), Gaussian
    `blur` (sigma ~U(0, blur) px -- motion / depth-of-field softening), additive Gaussian
    `noise` (std ~U(0, noise)*255 -- low-light sensor grain). Each appearance magnitude is
    sampled from zero up to its cap, so a share of frames stay near-clean; the corruptions are
    unconditional (not content-aware -- an already-soft frame may be blurred further, which is
    fine at these light caps). JPEG aug is deliberately omitted: a live camera feed is not
    re-encoded, so block artifacts never occur in deployment.
    """
    if hflip_prob and torch.rand(1).item() < hflip_prob:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        w = rgb.shape[1]
        if boxes.size:
            x1 = w - boxes[:, 2]
            x2 = w - boxes[:, 0]
            boxes = boxes.copy()
            boxes[:, 0], boxes[:, 2] = x1, x2
        if keypoints is not None and keypoints.size:
            keypoints = keypoints.copy()
            keypoints[:, :, 0] = w - keypoints[:, :, 0]  # mirror x; y, vis unchanged
    if jitter:
        # brightness + contrast jitter; cheap photometric variety
        alpha = 1.0 + (torch.rand(1).item() * 2 - 1) * jitter  # contrast
        beta = (torch.rand(1).item() * 2 - 1) * jitter * 255.0  # brightness
        rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if color:
        # independent per-channel gain: simulates white-balance / board-colour variation
        gains = 1.0 + (torch.rand(3).numpy() * 2 - 1) * color
        rgb = np.clip(rgb.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    if blur:
        sigma = torch.rand(1).item() * blur
        if sigma > 0.1:  # skip near-zero (cv2 no-op cost)
            rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma)
    if motion_blur:
        length = int(round(torch.rand(1).item() * motion_blur))
        if length >= 2:
            kernel = _motion_blur_kernel(length, torch.rand(1).item() * 360.0)
            rgb = cv2.filter2D(rgb, -1, kernel)
    if noise:
        std = torch.rand(1).item() * noise * 255.0
        if std > 0.5:
            field = torch.randn(*rgb.shape).numpy() * std
            rgb = np.clip(rgb.astype(np.float32) + field, 0, 255).astype(np.uint8)
    return rgb, boxes, keypoints


class ChessReDDetection(Dataset):
    def __init__(
        self,
        chessred: ChessReD,
        image_ids: Sequence[int],
        config: DetectionConfig | None = None,
        train: bool = False,
    ):
        self.ds = chessred
        self.config = config or DetectionConfig()
        self.train = train
        # Keep only images that actually carry >=1 box (defensive; chessred2k all do).
        self.image_ids = [i for i in image_ids if self._has_boxes(i)]

    @classmethod
    def from_split(
        cls,
        chessred: ChessReD,
        split: str,
        config: DetectionConfig | None = None,
        train: bool | None = None,
    ) -> ChessReDDetection:
        """Build from an official chessred2k split; `train` defaults to (split == 'train')."""
        return cls(
            chessred,
            chessred.chessred2k_split(split),
            config=config,
            train=(split == "train") if train is None else train,
        )

    def _has_boxes(self, image_id: int) -> bool:
        return any(p.bbox is not None for p in self.ds.pieces(image_id))

    def __len__(self) -> int:
        return len(self.image_ids)

    def _read_rgb(self, image_id: int) -> np.ndarray:
        path = self.ds.resolve_image_path(self.ds.meta(image_id))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _boxes_labels(self, image_id: int) -> tuple[np.ndarray, np.ndarray]:
        boxes, labels = [], []
        for p in self.ds.pieces(image_id):
            if p.bbox is None:
                continue
            x, y, w, h = p.bbox  # COCO xywh
            boxes.append([x, y, x + w, y + h])  # -> xyxy
            labels.append(p.category_id + 1)  # 0 reserved for background
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(labels, dtype=np.int64),
        )

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        rgb = self._read_rgb(image_id)
        boxes, labels = self._boxes_labels(image_id)

        if self.config.board_crop:
            rgb, boxes, labels = self._board_crop(image_id, rgb, boxes, labels)

        rgb, boxes, _ = self._resize(rgb, boxes)
        if self.train:
            rgb, boxes, _ = self._augment(rgb, boxes)

        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }
        return image, target

    def _board_crop(
        self, image_id: int, rgb: np.ndarray, boxes: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop to the board bbox (when corners exist) so pieces fill more of the frame, then
        drop any box the crop clipped to <=1px. Runs BEFORE resize, on full-frame targets.
        Shared by the box detectors; `ChessReDKeypointDetection` overrides `__getitem__` and
        does its own crop so it can carry the contact keypoint through too. MUST match the
        eval-time crop (same `geometry.board_crop_bbox` call)."""
        corners = self.ds.corners(image_id)
        if not corners:
            return rgb, boxes, labels
        h, w = rgb.shape[:2]
        c = self.config
        bbox = board_crop_bbox(
            corners, w, h, side=c.crop_side, top=c.crop_top, bottom=c.crop_bottom
        )
        rgb, boxes, _ = apply_board_crop(rgb, boxes, None, bbox)
        if boxes.size:
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes, labels = boxes[keep], labels[keep]
        return rgb, boxes, labels

    def _resize(
        self, rgb: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return resize_targets(rgb, boxes, keypoints, self.config.max_size)

    def _augment(
        self, rgb: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return augment_targets(
            rgb,
            boxes,
            keypoints,
            hflip_prob=self.config.hflip_prob,
            jitter=self.config.jitter,
            color=self.config.color,
            blur=self.config.blur,
            motion_blur=self.config.motion_blur,
            noise=self.config.noise,
        )


class ChessReDKeypointDetection(ChessReDDetection):
    """Detection dataset + per-piece board-contact keypoint target (Approach A).

    Adds `target["keypoints"] = (N, 1, 3)` `[x, y, visibility=2]`, one keypoint per
    box = that piece's contact point (square center projected through the homography,
    via `contact_points`). Keypoints are kept index-aligned with `boxes` (same piece
    order, same bbox-None skipping) and threaded through `_resize`/`_augment` so they
    track image scaling and horizontal flips. Everything else is the box pipeline.
    """

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        rgb = self._read_rgb(image_id)
        annotated = AnnotatedImage(
            meta=self.ds.meta(image_id),
            corners=self.ds.corners(image_id),
            pieces=self.ds.pieces(image_id),
        )
        cps = contact_points(annotated)  # aligned with annotated.pieces order

        boxes, labels, kpts = [], [], []
        for p, cp in zip(annotated.pieces, cps, strict=True):
            if p.bbox is None:
                continue
            x, y, w, h = p.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(p.category_id + 1)
            kpts.append([[cp.xy[0], cp.xy[1], 2.0]])  # COCO visibility 2 = labelled
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        keypoints = np.asarray(kpts, dtype=np.float32).reshape(-1, 1, 3)

        if self.config.board_crop and annotated.corners:
            h, w = rgb.shape[:2]
            c = self.config
            bbox = board_crop_bbox(
                annotated.corners, w, h, side=c.crop_side, top=c.crop_top, bottom=c.crop_bottom
            )
            rgb, boxes, keypoints = apply_board_crop(rgb, boxes, keypoints, bbox)

        rgb, boxes, keypoints = self._resize(rgb, boxes, keypoints)
        if self.train:
            rgb, boxes, keypoints = self._augment(rgb, boxes, keypoints)

        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
            "labels": torch.from_numpy(labels),
            "keypoints": torch.from_numpy(np.ascontiguousarray(keypoints)),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }
        return image, target
