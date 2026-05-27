"""A standalone board-corner dataset, fed by phone photos labelled in the app.

This is the data side of the "corner-capture mode" (see ``corner-capture-mode.md``).
It is **deliberately separate** from the capture set (`data/captures/`, Label Studio,
piece keypoints + FEN): the corner localizer needs *viewpoints*, not positions, so the
labels here are corner-only and there is no FEN/Label-Studio round-trip. Layout::

    data/corners/
      inbox/            raw phone dumps (subfolders OK, e.g. by date); local-only
      store/
        images/<id>.jpg EXIF-normalized JPEG, written when you label a photo
        labels.jsonl    one row per labelled photo (the trainable artifact)

**Why normalize on label.** Phone photos carry an EXIF orientation flag; browsers and
libraries disagree about when to apply it. So when a photo is labelled we bake the
rotation into the pixels once (`normalize_image`), store *that* JPEG, and record the
corners in its frame. The web app serves the same normalized pixels for marking, and
the trainer reads the stored JPEG directly (no EXIF) -- one pixel frame everywhere.

The corner dict uses the snake_case keys `geometry`/`corners.py` expect, so a label
flows straight into `order_corners` / `compute_homography` and the existing pose
clustering. `select_corner_dataset_poses` mirrors `select_capture_corner_poses` but
groups by the per-label ``board`` field (no SessionMetadata), so the held-out split
holds out whole *poses* per board -- viewpoint leakage is impossible by construction.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from chessvision.data.corners import (
    CORNER_ORDER,
    _cluster_by_corners,
    _CornerDataset,
    _sample_evenly,
    corners_to_array,
)
from chessvision.geometry import order_corners

CORNER_KEYS: frozenset[str] = frozenset(CORNER_ORDER)
# Decodable still-image suffixes. Phone HEIC/HEIF is intentionally excluded -- neither
# the browser <img> nor cv2 decode it without extra codecs; convert to JPEG first.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


# --------------------------------------------------------------------------- #
# Image normalization
# --------------------------------------------------------------------------- #


def normalize_image(data: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode JPEG/PNG bytes, bake in EXIF orientation, return (rgb, (w, h)).

    `rgb` is a contiguous (H, W, 3) uint8 array in the upright frame; the returned
    (w, h) are that frame's dimensions (post-rotation), which is what the stored
    corners are normalized against.
    """
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(im))
    h, w = rgb.shape[:2]
    return rgb, (w, h)


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array to JPEG bytes."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("failed to encode JPEG")
    return buf.tobytes()


def _stable_id(src: str) -> str:
    """A stable, filesystem-safe id for an inbox-relative source path."""
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def _task_id(label_id: str) -> int:
    """Deterministic int id for pose-clustering order (stable across re-labels)."""
    return int(label_id[:8], 16)


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CornerLabel:
    """One labelled corner photo. Shapes match `CaptureSample` closely enough to feed
    the shared pose-clustering helpers (`_cluster_by_corners` reads `corners`/`width`/
    `height`/`task_id`)."""

    id: str
    task_id: int
    src: str  # inbox-relative source path (forward slashes)
    image: str  # store-relative normalized JPEG, e.g. "images/<id>.jpg"
    width: int
    height: int
    corners: dict[str, tuple[float, float]]  # snake_case key -> (x, y) in the normalized frame
    board: str = ""  # boards.json key; "" -> untagged
    device: str = ""
    surface: str = ""
    labeled_at: str = ""

    @property
    def has_all_corners(self) -> bool:
        return set(self.corners) == CORNER_KEYS

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "src": self.src,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "corners": {k: [float(x), float(y)] for k, (x, y) in self.corners.items()},
            "board": self.board,
            "device": self.device,
            "surface": self.surface,
            "labeled_at": self.labeled_at,
        }

    @classmethod
    def from_row(cls, row: dict) -> CornerLabel:
        return cls(
            id=row["id"],
            task_id=_task_id(row["id"]),
            src=row["src"],
            image=row["image"],
            width=int(row["width"]),
            height=int(row["height"]),
            corners={k: (float(v[0]), float(v[1])) for k, v in row["corners"].items()},
            board=row.get("board", "") or "",
            device=row.get("device", "") or "",
            surface=row.get("surface", "") or "",
            labeled_at=row.get("labeled_at", "") or "",
        )


# --------------------------------------------------------------------------- #
# The on-disk store (inbox + labels)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InboxPhoto:
    id: str
    src: str  # inbox-relative path
    group: str  # immediate parent folder (e.g. "2026-05-27"), "" at the inbox root
    date: str  # EXIF capture time if present, else file mtime; ISO seconds
    labeled: bool
    board: str  # the label's board if labelled, else ""
    # Saved corners (TL/TR/BR/BL, [x, y] in the normalized frame) if labelled, else None —
    # lets the UI re-open a labelled photo with its handles already in place.
    corners: list[list[float]] | None = None


class CornerStore:
    """Read/write access to a `data/corners` tree: list the inbox, normalize-and-save a
    label, and enumerate labelled samples for training."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.store = self.root / "store"
        self.images_dir = self.store / "images"
        self.labels_path = self.store / "labels.jsonl"

    # ---- labels.jsonl -------------------------------------------------------

    def load_labels(self) -> dict[str, dict]:
        """All label rows keyed by id (last write wins on duplicate ids)."""
        rows: dict[str, dict] = {}
        if not self.labels_path.exists():
            return rows
        for line in self.labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row
        return rows

    def _write_labels(self, rows: dict[str, dict]) -> None:
        self.store.mkdir(parents=True, exist_ok=True)
        ordered = sorted(rows.values(), key=lambda r: r["id"])
        body = "\n".join(json.dumps(r) for r in ordered)
        self.labels_path.write_text(body + ("\n" if body else ""), encoding="utf-8")

    # ---- inbox listing ------------------------------------------------------

    def _iter_inbox_files(self):
        if not self.inbox.exists():
            return
        for path in sorted(self.inbox.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path

    @staticmethod
    def _photo_date(path: Path) -> str:
        """EXIF DateTimeOriginal (tag 36867) if present, else file mtime; ISO seconds."""
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                raw = exif.get(36867) or exif.get(306)  # DateTimeOriginal, then DateTime
                if raw:
                    dt = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
                    return dt.isoformat(timespec="seconds")
        except (OSError, ValueError):
            pass
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")

    def list_inbox(self) -> list[InboxPhoto]:
        """Every decodable inbox photo, date-ordered, with labelled state. The `group`
        is the immediate parent folder so the UI can show a hierarchical, date-named
        listing. Sorted by (date, src) so a date-named dump reads chronologically."""
        labels = self.load_labels()
        by_src = {r["src"]: r for r in labels.values()}
        photos: list[InboxPhoto] = []
        for path in self._iter_inbox_files():
            src = path.relative_to(self.inbox).as_posix()
            parent = path.parent.relative_to(self.inbox).as_posix()
            row = by_src.get(src)
            corners = None
            if row is not None:
                c = row["corners"]
                corners = [[float(c[k][0]), float(c[k][1])] for k in CORNER_ORDER]
            photos.append(
                InboxPhoto(
                    id=_stable_id(src),
                    src=src,
                    group="" if parent == "." else parent,
                    date=self._photo_date(path),
                    labeled=row is not None,
                    board=(row or {}).get("board", "") or "",
                    corners=corners,
                )
            )
        photos.sort(key=lambda p: (p.date, p.src))
        return photos

    def inbox_path(self, src: str) -> Path:
        """Resolve an inbox-relative `src` to an absolute path, rejecting traversal."""
        path = (self.inbox / src).resolve()
        if not path.is_file() or self.inbox.resolve() not in path.parents:
            raise FileNotFoundError(f"no such inbox photo: {src}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"not a supported image: {src}")
        return path

    def normalized_bytes(self, src: str, max_width: int | None = None) -> bytes:
        """The EXIF-normalized JPEG bytes for an inbox photo (for app display).

        `max_width` downscales for thumbnails (the browser loads many at once);
        omit it for the full-resolution image used to mark corners.
        """
        rgb, _ = normalize_image(self.inbox_path(src).read_bytes())
        if max_width and rgb.shape[1] > max_width:
            h, w = rgb.shape[:2]
            new_h = max(1, round(h * max_width / w))
            rgb = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
        return encode_jpeg(rgb)

    # ---- saving a label -----------------------------------------------------

    def save_label(
        self,
        src: str,
        corners: Sequence[Sequence[float]] | dict,
        *,
        board: str = "",
        device: str = "",
        surface: str = "",
    ) -> CornerLabel:
        """Normalize the inbox photo, write its store JPEG, and upsert its label row.

        `corners` may be four points in any order or a corner dict; they are sorted
        into visual TL/TR/BR/BL with `order_corners`. Re-saving the same `src`
        overwrites in place (the photo's id is derived from `src`).
        """
        path = self.inbox_path(src)
        rgb, (w, h) = normalize_image(path.read_bytes())
        label_id = _stable_id(src)
        image_rel = f"images/{label_id}.jpg"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        (self.store / image_rel).write_bytes(encode_jpeg(rgb))

        pts = list(corners.values()) if isinstance(corners, dict) else list(corners)
        ordered = order_corners(pts)  # {top_left, top_right, bottom_right, bottom_left}
        label = CornerLabel(
            id=label_id,
            task_id=_task_id(label_id),
            src=src,
            image=image_rel,
            width=w,
            height=h,
            corners={k: (float(ordered[k][0]), float(ordered[k][1])) for k in CORNER_ORDER},
            board=board or "",
            device=device or "",
            surface=surface or "",
            labeled_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        rows = self.load_labels()
        rows[label.id] = label.to_row()
        self._write_labels(rows)
        return label

    # ---- samples for training ----------------------------------------------

    def samples(self) -> list[CornerLabel]:
        """Labelled samples whose normalized image is present on disk and which carry
        all four corners (the corner-regression training set)."""
        out: list[CornerLabel] = []
        for row in self.load_labels().values():
            label = CornerLabel.from_row(row)
            if label.has_all_corners and (self.store / label.image).exists():
                out.append(label)
        return out


# --------------------------------------------------------------------------- #
# Train / held-out split (per board, by pose)
# --------------------------------------------------------------------------- #


def select_corner_dataset_poses(
    store: CornerStore | str | Path,
    *,
    dedup_thr: float = 0.02,
    max_per_pose: int = 2,
    val_frac: float = 0.25,
) -> tuple[list[CornerLabel], list[CornerLabel]]:
    """Split labelled corner photos into (train, held-out), holding out whole poses.

    Mirrors `corners.select_capture_corner_poses` but groups by each label's ``board``
    field directly (untagged -> "(untagged)"). Every labelled frame is clustered into
    a distinct corner *pose* by geometry (`dedup_thr`); the pose-cluster is the atomic
    unit of the split, so a viewpoint that recurs lands wholly on one side. A
    deterministic `val_frac` share of each board's poses is held out; the rest train
    (thinned to `max_per_pose` evenly-spaced frames). A final anti-leak pass drops any
    train pose within `dedup_thr` of a held-out one.
    """
    store = store if isinstance(store, CornerStore) else CornerStore(store)
    samples = store.samples()
    clusters = _cluster_by_corners(samples, dedup_thr)

    by_board: dict[str, list[list]] = defaultdict(list)
    for cl in clusters:
        # Clusters are board-pure in practice; pick the majority board as the key.
        boards = [s.board or "(untagged)" for s in cl]
        board = max(set(boards), key=boards.count)
        by_board[board].append(cl)

    heldout_clusters: list[list] = []
    for board_clusters in by_board.values():
        order = sorted(
            range(len(board_clusters)),
            key=lambda i: min(s.task_id for s in board_clusters[i]),  # noqa: B023
        )
        n = len(order)
        n_val = 0 if n < 2 else min(n - 1, max(1, round(val_frac * n)))
        val_idx = set(_sample_evenly(order, n_val)) if n_val else set()
        heldout_clusters.extend(board_clusters[i] for i in val_idx)

    heldout_ids = {id(cl) for cl in heldout_clusters}
    heldout_centroids = [
        corners_to_array(c[0].corners) / np.array([c[0].width, c[0].height], np.float32)
        for c in heldout_clusters
    ]
    heldout = [s for cl in heldout_clusters for s in _sample_evenly(cl, max_per_pose)]

    train_pool = [s for cl in clusters if id(cl) not in heldout_ids for s in cl]
    train_clusters = _cluster_by_corners(train_pool, dedup_thr, exclude=heldout_centroids)
    train = [s for cluster in train_clusters for s in _sample_evenly(cluster, max_per_pose)]
    return train, heldout


# --------------------------------------------------------------------------- #
# Training dataset
# --------------------------------------------------------------------------- #


class CornerCaptureDataset(_CornerDataset):
    """Corner dataset over labelled `data/corners` photos, emitting the same
    `(image, target)` as `ChessReDCorners`. Images load from the local store
    (already EXIF-normalized); corners are stored in that same frame."""

    def __init__(
        self,
        samples: Sequence[CornerLabel],
        store: CornerStore | str | Path,
        config=None,
        train: bool = False,
    ):
        super().__init__(config, train)
        self.samples = list(samples)
        self.store = store if isinstance(store, CornerStore) else CornerStore(store)

    def __len__(self) -> int:
        return len(self.samples)

    def _image_id(self, idx: int) -> int:
        return self.samples[idx].task_id

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        sample = self.samples[idx]
        path = self.store.store / sample.image
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read corner image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        pts = corners_to_array(sample.corners) / np.array([w0, h0], dtype=np.float32)
        return rgb, pts, (w0, h0)
