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

from collections import Counter, defaultdict
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
    cache: bool = False  # keep the resized image_size^2 array in RAM (decode each image once)


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


class _CornerDataset(Dataset):
    """Shared corner-dataset machinery: optional in-RAM cache + the `(image, target)`
    finalize step. Subclasses implement `__len__`, `_image_id(idx)`, and
    `_load_raw(idx) -> (rgb_full, normalized_pts, (w0, h0))`.

    **Why the cache.** The images are huge (ChessReD is 3072^2 JPEGs) but the network
    input is tiny (`image_size`^2). Without caching, every epoch re-reads and re-decodes
    every image -- the run becomes decode-bound and the GPU idles (~340 s/epoch observed
    for a 1.7M-param model). With `config.cache`, each image is decoded + resized **once**
    and the small array is kept in RAM (~0.4 MB at 384^2), so later epochs are GPU-bound.
    Augmentation still runs per-epoch on the cached resized array (flip + photometric
    jitter are resolution-independent). Use `prewarm()` to fill the cache up front with a
    thread pool (cv2 decode releases the GIL, so threads parallelize it).

    The cache lives in the dataset object, so run with **DataLoader `num_workers=0`**: with
    worker processes each worker would build its own separate cache (no sharing, N x memory).
    """

    def __init__(self, config: CornerConfig | None = None, train: bool = False):
        self.config = config or CornerConfig()
        self.train = train
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}

    def __len__(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def _image_id(self, idx: int) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Decode the full image and return (rgb_full, normalized visual-slot corners,
        (w0, h0)). The expensive step that the cache exists to avoid repeating."""
        raise NotImplementedError  # pragma: no cover - overridden

    def _sized(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """Resized (image_size^2) rgb + normalized pts + native (w0, h0), from cache."""
        if idx in self._cache:
            return self._cache[idx]
        rgb, pts, owh = self._load_raw(idx)
        size = self.config.image_size
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        item = (np.ascontiguousarray(rgb), pts, owh)
        if self.config.cache:
            self._cache[idx] = item
        return item

    def prewarm(self, max_workers: int = 8) -> None:
        """Fill the cache with a thread pool so the first epoch isn't decode-bound."""
        if not self.config.cache:
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(self._sized, range(len(self))))

    def __getitem__(self, idx: int):
        rgb, pts, (w0, h0) = self._sized(idx)
        if self.train:
            rgb, pts = augment_corners(rgb, pts, self.config)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        target = {
            "corners": torch.from_numpy(np.ascontiguousarray(pts)),
            "image_id": torch.tensor([self._image_id(idx)], dtype=torch.int64),
            "orig_size": torch.tensor([w0, h0], dtype=torch.int64),
        }
        return image, target


class ChessReDCorners(_CornerDataset):
    def __init__(
        self,
        chessred: ChessReD,
        image_ids: Sequence[int],
        config: CornerConfig | None = None,
        train: bool = False,
    ):
        super().__init__(config, train)
        self.ds = chessred
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

    def _image_id(self, idx: int) -> int:
        return self.image_ids[idx]

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        image_id = self.image_ids[idx]
        path = self.ds.resolve_image_path(self.ds.meta(image_id))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        pts = corners_to_array(self.ds.corners(image_id)) / np.array([w0, h0], dtype=np.float32)
        return rgb, pts, (w0, h0)


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
# See the warning in `chessvision.data.captures`: splits must not leak near-duplicate
# frames. The held-out capture-eval set is derived automatically (no manual tag): every
# frame is clustered into a distinct corner *pose*, the pose-cluster is the atomic unit
# of the split, and a deterministic share of each *board's* poses is held out. An
# anti-leak pass still drops any train pose within `dedup_thr` of a held-out one.


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


def select_capture_corner_poses(
    export_path: str | Path,
    captures_root: str | Path | None = None,
    *,
    dedup_thr: float = 0.02,
    max_per_pose: int = 2,
    val_frac: float = 0.25,
) -> tuple[list, list]:
    """Pick deduped, distinct-corner-pose capture samples for corner training, with the
    train/held-out split derived automatically (no manual `sessions.json` tag).

    Returns `(train_samples, heldout_samples)`. Every frame with four corners is
    clustered into a distinct corner *pose* by geometry (`dedup_thr`, a fraction of
    image size); the **pose-cluster is the atomic unit of the split**, so an orientation
    that recurs across sessions lands wholly on one side — leakage is impossible by
    construction. Clusters are grouped by the session's **board** tag, and a
    deterministic share (`val_frac`, at least one pose where a board has >= 2 of them) of
    each board's poses is held out; the rest train. So every board with enough poses
    appears in both train and eval. Each kept pose is thinned to `max_per_pose`
    evenly-spaced frames (lighting/piece variety). A final anti-leak pass drops any train
    pose within `dedup_thr` of a held-out one (greedy clustering can leave cross-cluster
    near-neighbours). Sessions absent from the metadata group under "(untagged)".
    """
    from chessvision.data.captures import CaptureDataset
    from chessvision.data.session_meta import SessionMetadata

    ds = CaptureDataset.load(export_path, captures_root)
    meta = SessionMetadata.load(ds.captures_root)

    def board_of(session: str) -> str:
        info = meta.info(session) if meta else None
        return (info or {}).get("board") or "(untagged)"

    with_corners = [s for s in ds.samples if s.has_all_corners]
    clusters = _cluster_by_corners(with_corners, dedup_thr)  # one pose per cluster

    # Group poses by board (majority board of a cluster; clusters are board-pure in
    # practice), then hold out an evenly-spread share of each board's poses. Ordering is
    # by a stable key (a cluster's smallest task id) so the split is reproducible.
    by_board: dict[str, list[list]] = defaultdict(list)
    for cl in clusters:
        board = Counter(board_of(s.session) for s in cl).most_common(1)[0][0]
        by_board[board].append(cl)

    heldout_clusters: list[list] = []
    for board_clusters in by_board.values():
        pose_key = lambda i: min(s.task_id for s in board_clusters[i])  # noqa: E731,B023
        order = sorted(range(len(board_clusters)), key=pose_key)
        n = len(order)
        n_val = 0 if n < 2 else min(n - 1, max(1, round(val_frac * n)))
        val_idx = set(_sample_evenly(order, n_val)) if n_val else set()
        heldout_clusters.extend(board_clusters[i] for i in val_idx)

    heldout_ids = {id(cl) for cl in heldout_clusters}
    heldout_centroids = [_norm_corners(c[0]) for c in heldout_clusters]
    heldout = [s for cl in heldout_clusters for s in _sample_evenly(cl, max_per_pose)]

    # Train = every non-held-out frame, re-clustered with the held-out poses excluded so
    # no train pose sits within dedup_thr of a held-out one.
    train_pool = [s for cl in clusters if id(cl) not in heldout_ids for s in cl]
    train_clusters = _cluster_by_corners(train_pool, dedup_thr, exclude=heldout_centroids)
    train = [s for cluster in train_clusters for s in _sample_evenly(cluster, max_per_pose)]
    return train, heldout


class CaptureCorners(_CornerDataset):
    """Corner dataset over capture samples, emitting the same `(image, target)` as
    `ChessReDCorners`. Image bytes load locally with an S3/MinIO fallback (see
    `chessvision.data.captures`). Build the sample lists with
    `select_capture_corner_poses` so geometry is deduped and splits don't leak."""

    def __init__(self, samples: list, config: CornerConfig | None = None, train: bool = False):
        super().__init__(config, train)
        self.samples = list(samples)
        # S3 fallback config taken from any sample's parent dataset is not stored on the
        # sample, so resolve it from env once here (None -> local-only).
        from chessvision.data.captures import S3Config

        self.s3 = S3Config.from_env()

    def __len__(self) -> int:
        return len(self.samples)

    def _image_id(self, idx: int) -> int:
        return self.samples[idx].task_id

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        sample = self.samples[idx]
        rgb = sample.load_image(self.s3)
        h0, w0 = rgb.shape[:2]
        pts = corners_to_array(sample.corners) / np.array([w0, h0], dtype=np.float32)
        return rgb, pts, (w0, h0)
