"""ChessReD board-corner regression dataset (Phase 3, board localizer).

Wraps the **chessred2k** subset -- the only ChessReD images carrying a `corners`
annotation (exactly the 1442/330/306 detection split) -- as a dataset for direct
4-corner regression. Each item is `(image, target)` where:

    image  : float32 3xSxS tensor in [0, 1], the photo resized to a fixed square
    target : {"corners":   (4, 2) float32, the board corners normalized to [0, 1]
                           in (x/W, y/H), in visual TL/TR/BR/BL order,
              "image_id":  (1,) int64,
              "orig_size": (2,) int64 (W, H) -- native size, for px-error eval}

**Why visual slots, not the annotation's a8/h8/... keys.** ChessReD labels corners
by *board semantics* (`top_left` is the a8 corner, etc.), but which physical corner
is a8 is the deliberate manual orientation toggle and is **not** recoverable from
geometry (plan.md section 4 / the live-mode design). So we canonicalize every target
to *visual* slots with `geometry.order_corners` (sort by image position): the
regressor learns pure quad **localization**, and orientation stays a separate, manual
choice downstream. This also keeps the target well-posed -- a fixed mapping from image
appearance to output, with no hidden orientation variable to infer.

Coordinates are normalized by the **original** width/height (before the resize that
makes the network input), so targets are resolution-independent and the input resize
can be anisotropic without touching the labels.

Splits come from `ChessReD.chessred2k_split` (same as the detector), so corner numbers
are free of game-level leakage and comparable across phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from chessvision.data.chessred import ChessReD
from chessvision.geometry import order_corners

# Output corner order: visual slots as a closed-ish ring (matches geometry.IMAGE_CORNER_RING).
CORNER_ORDER: tuple[str, str, str, str] = ("top_left", "top_right", "bottom_right", "bottom_left")
NUM_CORNERS = 4


@dataclass
class CornerConfig:
    image_size: int = 384  # square network input (board ~fills the 3072^2 ChessReD frame)
    hflip_prob: float = 0.0  # train-time horizontal flip probability
    jitter: float = 0.0  # train-time brightness/contrast jitter magnitude (0 disables)


def corners_to_array(corners: dict[str, Sequence[float]]) -> np.ndarray:
    """Annotation corner dict -> (4, 2) float32 in visual TL/TR/BR/BL order.

    `order_corners` re-sorts the four points by image position, so the output is
    independent of the annotation's board-semantic keying (see module docstring).
    """
    ordered = order_corners(list(corners.values()))
    return np.array([ordered[k] for k in CORNER_ORDER], dtype=np.float32)


def collate_corners(batch):
    """Fixed-size targets -> a clean stacked batch (unlike ragged detection)."""
    images, targets = zip(*batch, strict=True)
    images = torch.stack(images, 0)
    out = {
        "corners": torch.stack([t["corners"] for t in targets], 0),
        "image_id": torch.cat([t["image_id"] for t in targets], 0),
        "orig_size": torch.stack([t["orig_size"] for t in targets], 0),
    }
    return images, out


def augment_corners(
    rgb: np.ndarray, pts: np.ndarray, config: CornerConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal flip (mirror x, then re-canonicalize visual slots) + photometric jitter.

    `pts` are normalized [0, 1] corners in visual-slot order. A flip swaps left/right,
    so the visual TL becomes TR and BL becomes BR; re-running `order_corners` in
    normalized space restores the slot ordering instead of hand-swapping pairs.
    """
    if config.hflip_prob and torch.rand(1).item() < config.hflip_prob:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        pts = pts.copy()
        pts[:, 0] = 1.0 - pts[:, 0]
        pts = corners_to_array({k: pts[i] for i, k in enumerate(CORNER_ORDER)})
    if config.jitter:
        alpha = 1.0 + (torch.rand(1).item() * 2 - 1) * config.jitter  # contrast
        beta = (torch.rand(1).item() * 2 - 1) * config.jitter * 255.0  # brightness
        rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return rgb, pts


def build_corner_item(
    rgb: np.ndarray, pts: np.ndarray, config: CornerConfig, train: bool, image_id: int
):
    """(rgb, normalized visual-slot corners) -> the `(image, target)` pair both the
    ChessReD and capture corner datasets emit. Augments (if `train`) then resizes to
    the square network input; the target keeps native `orig_size` for px-error eval."""
    h0, w0 = rgb.shape[:2]
    if train:
        rgb, pts = augment_corners(rgb, pts, config)
    inp = cv2.resize(rgb, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1).float() / 255.0
    target = {
        "corners": torch.from_numpy(np.ascontiguousarray(pts)),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "orig_size": torch.tensor([w0, h0], dtype=torch.int64),
    }
    return image, target


class ChessReDCorners(Dataset):
    def __init__(
        self,
        chessred: ChessReD,
        image_ids: Sequence[int],
        config: CornerConfig | None = None,
        train: bool = False,
    ):
        self.ds = chessred
        self.config = config or CornerConfig()
        self.train = train
        # Keep only images that actually carry corners (chessred2k all do; defensive).
        self.image_ids = [i for i in image_ids if self.ds.corners(i) is not None]

    @classmethod
    def from_split(
        cls,
        chessred: ChessReD,
        split: str,
        config: CornerConfig | None = None,
        train: bool | None = None,
    ) -> ChessReDCorners:
        """Build from an official chessred2k split; `train` defaults to (split == 'train')."""
        return cls(
            chessred,
            chessred.chessred2k_split(split),
            config=config,
            train=(split == "train") if train is None else train,
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _read_rgb(self, image_id: int) -> np.ndarray:
        path = self.ds.resolve_image_path(self.ds.meta(image_id))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        rgb = self._read_rgb(image_id)
        h0, w0 = rgb.shape[:2]
        pts = corners_to_array(self.ds.corners(image_id)) / np.array([w0, h0], dtype=np.float32)
        return build_corner_item(rgb, pts, self.config, self.train, image_id)


# --------------------------------------------------------------------------- #
# Capture corners (the user's own boards)
# --------------------------------------------------------------------------- #
#
# The capture set adds the user's two physical boards to corner training, but it is
# *low corner-geometry diversity*: within a session the camera/board is fixed, so its
# many frames share one corner pose (~10 distinct poses across all ~300 frames). Feeding
# every frame would over-represent those ~10 geometries and bias the localizer's pose
# prior, so we **dedup to distinct corner poses** and keep only a few frames per pose
# (enough lighting/piece variety to learn the board, not enough to distort the prior).
# See the warning in `chessvision.data.captures`: splits are grouped by *session*; here
# the held-out capture-eval poses come from the sessions tagged "held-out" in
# `sessions.json`, and an anti-leak check drops any train pose near a held-out one.


def _norm_corners(sample) -> np.ndarray:
    """A capture sample's corners as (4, 2) visual-slot coords normalized to [0, 1]."""
    return corners_to_array(sample.corners) / np.array([sample.width, sample.height], np.float32)


def _corner_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-corner Euclidean distance between two normalized corner quads."""
    return float(np.linalg.norm(a - b, axis=1).mean())


def _cluster_by_corners(
    samples: list, thr: float, exclude: list[np.ndarray] | None = None
) -> list[list]:
    """Greedily group samples whose corner quads are within `thr`, dropping any whose
    quad is within `thr` of an `exclude` quad (used to keep train poses clear of the
    held-out poses). Returns one list of samples per distinct pose."""
    exclude = exclude or []
    clusters: list[list] = []
    centroids: list[np.ndarray] = []
    for s in samples:
        c = _norm_corners(s)
        if any(_corner_dist(c, e) <= thr for e in exclude):
            continue
        hit = next((i for i, cc in enumerate(centroids) if _corner_dist(c, cc) <= thr), None)
        if hit is None:
            centroids.append(c)
            clusters.append([s])
        else:
            clusters[hit].append(s)
    return clusters


def _sample_evenly(items: list, k: int) -> list:
    """Up to `k` items spread evenly across `items` (for lighting/piece variety)."""
    if len(items) <= k:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in idx]


def _is_heldout(notes: str | None) -> bool:
    return bool(notes) and "held-out" in notes.lower()


def select_capture_corner_poses(
    export_path: str | Path,
    captures_root: str | Path | None = None,
    *,
    dedup_thr: float = 0.02,
    max_per_pose: int = 2,
) -> tuple[list, list]:
    """Pick deduped, distinct-corner-pose capture samples for corner training.

    Returns `(train_samples, heldout_samples)`. Held-out sessions are those tagged
    "held-out" in `sessions.json` (one+ pose per physical board); the rest are train.
    Each split is clustered to distinct poses (`dedup_thr`, fraction of image size) and
    thinned to `max_per_pose` evenly-spaced frames per pose. Train poses within
    `dedup_thr` of any held-out pose are dropped (anti-leak).
    """
    from chessvision.data.captures import CaptureDataset
    from chessvision.data.session_meta import SessionMetadata

    ds = CaptureDataset.load(export_path, captures_root)
    meta = SessionMetadata.load(ds.captures_root)

    def heldout_session(session: str) -> bool:
        info = meta.info(session) if meta else None
        return _is_heldout(info.get("notes") if info else None)

    with_corners = [s for s in ds.samples if s.has_all_corners]
    heldout_pool = [s for s in with_corners if heldout_session(s.session)]
    train_pool = [s for s in with_corners if not heldout_session(s.session)]

    heldout_clusters = _cluster_by_corners(heldout_pool, dedup_thr)
    heldout_centroids = [_norm_corners(c[0]) for c in heldout_clusters]
    train_clusters = _cluster_by_corners(train_pool, dedup_thr, exclude=heldout_centroids)

    train = [s for cluster in train_clusters for s in _sample_evenly(cluster, max_per_pose)]
    heldout = [s for cluster in heldout_clusters for s in _sample_evenly(cluster, max_per_pose)]
    return train, heldout


class CaptureCorners(Dataset):
    """Corner dataset over capture samples, emitting the same `(image, target)` as
    `ChessReDCorners`. Image bytes load locally with an S3/MinIO fallback (see
    `chessvision.data.captures`). Build the sample lists with
    `select_capture_corner_poses` so geometry is deduped and splits don't leak."""

    def __init__(self, samples: list, config: CornerConfig | None = None, train: bool = False):
        self.samples = list(samples)
        self.config = config or CornerConfig()
        self.train = train
        # S3 fallback config taken from any sample's parent dataset is not stored on the
        # sample, so resolve it from env once here (None -> local-only).
        from chessvision.data.captures import S3Config

        self.s3 = S3Config.from_env()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rgb = sample.load_image(self.s3)
        h0, w0 = rgb.shape[:2]
        pts = corners_to_array(sample.corners) / np.array([w0, h0], dtype=np.float32)
        return build_corner_item(rgb, pts, self.config, self.train, sample.task_id)
