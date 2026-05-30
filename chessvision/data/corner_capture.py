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
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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


# --------------------------------------------------------------------------- #
# EXIF metadata extraction (publish-safe whitelist)
# --------------------------------------------------------------------------- #

# Per-device lens map: slugged (make, model) -> {(focal_mm, f_number): lens name}. Phones
# like the Fairphone 5 write no LensModel, so the focal length + aperture are the only way
# to tell which physical camera shot a photo. Unmapped combos fall back to a raw
# "<focal>mm-f<fnum>" tag (nothing lost); add rows here as new lenses/phones show up.
LENS_NAMES: dict[tuple[str, str], dict[tuple[float, float], str]] = {
    ("fairphone", "fp5"): {
        (5.56, 1.88): "main",  # FP5 rear wide; ultrawide has a shorter focal -> distinct tag
    },
}


def _slug(text: str) -> str:
    """Lowercase alphanumeric slug (non-alnum -> single hyphen, trimmed)."""
    out, prev_dash = [], False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def _device_id(make: str, model: str, focal_mm: float | None, f_number: float | None) -> str:
    """A stable, lens-aware device slug from EXIF camera fields (e.g. "fairphone-fp5-main"),
    or "" when there's no make/model. Falls back to a raw focal/aperture lens tag for an
    unmapped lens so the camera is always distinguishable."""
    mk, md = _slug(make), _slug(model)
    base = "-".join(p for p in (mk, md) if p)
    if not base:
        return ""
    lens = ""
    table = LENS_NAMES.get((mk, md))
    if table is not None and focal_mm is not None and f_number is not None:
        lens = table.get((focal_mm, f_number), "")
    if not lens and focal_mm is not None and f_number is not None:
        lens = f"{focal_mm:g}mm-f{f_number:g}"
    return f"{base}-{lens}" if lens else base


def extract_exif_meta(data: bytes) -> dict:
    """Whitelisted, **publish-safe** EXIF for a label row.

    Returns ONLY non-sensitive fields: ``captured_at`` (ISO seconds, the photo's capture
    time), ``device`` (lens-aware camera slug, e.g. ``"fairphone-fp5-main"``), and the
    provenance ``make``/``model``/``focal_mm``/``f_number`` used to derive it. **GPS
    (location) and serial/owner tags are never read**, so nothing location- or
    owner-identifying can leak into the dataset -- the stored JPEG is already EXIF-free, so
    the published artifact is those images plus this whitelist. Missing fields are omitted;
    returns ``{}`` when the image has no usable EXIF.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)  # Exif sub-IFD: focal length, aperture, capture time
    except (OSError, ValueError, SyntaxError):
        return {}

    out: dict = {}
    raw_dt = sub.get(36867) or exif.get(306)  # DateTimeOriginal, then DateTime
    if raw_dt:
        try:
            out["captured_at"] = datetime.strptime(
                str(raw_dt), "%Y:%m:%d %H:%M:%S"
            ).isoformat(timespec="seconds")
        except ValueError:
            pass

    make = str(exif.get(271, "") or "").strip()  # Make
    model = str(exif.get(272, "") or "").strip()  # Model
    focal = sub.get(37386)  # FocalLength (mm)
    f_number = sub.get(33437)  # FNumber
    focal_mm = round(float(focal), 2) if focal is not None else None
    f_num = round(float(f_number), 2) if f_number is not None else None
    if make:
        out["make"] = make
    if model:
        out["model"] = model
    if focal_mm is not None:
        out["focal_mm"] = focal_mm
    if f_num is not None:
        out["f_number"] = f_num
    device = _device_id(make, model, focal_mm, f_num)
    if device:
        out["device"] = device
    return out


def _stable_id(src: str) -> str:
    """A stable, filesystem-safe id for an inbox-relative source path."""
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: write a sibling temp file, then os.replace.

    The rename is atomic on the same filesystem, so a process killed mid-write leaves the
    old file intact instead of a truncated one -- the only real corruption mode for these
    whole-file JSON/JSONL stores (see the SQLite-vs-text discussion; text + atomic rename
    gives the durability without losing inspectability or the S3-object sync model)."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
    piece_set: str = ""  # sets.json key (the physical piece set); "" -> untagged. Stored as
    # "set" in the row to mirror sessions.json; matters for the *piece* (position) labels.
    session: str = ""  # synthesized capture session (YYYYMMDD-HHMMSS), grouping photos shot
    # close in time on one board -- the split unit; see assign_sessions / [[merge corner+capture]]
    device: str = ""  # camera slug; auto-derived from EXIF (lens-aware) when present
    surface: str = ""
    labeled_at: str = ""
    captured_at: str = ""  # EXIF DateTimeOriginal (photo capture time), ISO seconds -- NOT
    # labeled_at. Drives session synthesis; publish-safe (GPS/serials are never extracted).
    # Optional *position* labels (the in-app position tool, see chessvision/data/positions.py):
    # the known FEN placement field, the orientation chosen to match the photo, and the
    # nudged per-piece contact keypoints. Corner-only photos leave these empty.
    fen: str = ""
    orientation: str = ""  # "R0".."R270"
    # (verbose label, x, y) per placed piece, in the normalized frame.
    pieces: tuple[tuple[str, float, float], ...] = ()

    @property
    def has_all_corners(self) -> bool:
        return set(self.corners) == CORNER_KEYS

    @property
    def has_pieces(self) -> bool:
        return bool(self.pieces) and self.has_all_corners

    def to_row(self) -> dict:
        row = {
            "id": self.id,
            "src": self.src,
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "corners": {k: [float(x), float(y)] for k, (x, y) in self.corners.items()},
            "board": self.board,
            "set": self.piece_set,
            "session": self.session,
            "device": self.device,
            "surface": self.surface,
            "labeled_at": self.labeled_at,
            "captured_at": self.captured_at,
        }
        if self.pieces:
            row["fen"] = self.fen
            row["orientation"] = self.orientation
            row["pieces"] = [
                {"label": lbl, "x": float(x), "y": float(y)} for lbl, x, y in self.pieces
            ]
        return row

    @classmethod
    def from_row(cls, row: dict) -> CornerLabel:
        pieces = tuple(
            (p["label"], float(p["x"]), float(p["y"])) for p in row.get("pieces", []) or []
        )
        return cls(
            id=row["id"],
            task_id=_task_id(row["id"]),
            src=row["src"],
            image=row["image"],
            width=int(row["width"]),
            height=int(row["height"]),
            corners={k: (float(v[0]), float(v[1])) for k, v in row["corners"].items()},
            board=row.get("board", "") or "",
            piece_set=row.get("set", "") or "",
            session=row.get("session", "") or "",
            device=row.get("device", "") or "",
            surface=row.get("surface", "") or "",
            labeled_at=row.get("labeled_at", "") or "",
            captured_at=row.get("captured_at", "") or "",
            fen=row.get("fen", "") or "",
            orientation=row.get("orientation", "") or "",
            pieces=pieces,
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
    piece_set: str = ""  # the label's piece set (sets.json key) if tagged, else ""
    # Saved corners (TL/TR/BR/BL, [x, y] in the normalized frame) if labelled, else None —
    # lets the UI re-open a labelled photo with its handles already in place.
    corners: list[list[float]] | None = None
    # Position labels (the in-app position tool): whether pieces are placed, the FEN used,
    # the chosen orientation, and the nudged contact keypoints — so the positions browser
    # can flag done photos and re-open them with their handles in place.
    positioned: bool = False
    fen: str = ""
    orientation: str = ""
    pieces: list[dict] | None = None


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
        _atomic_write_text(self.labels_path, body + ("\n" if body else ""))

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
            pieces = (row or {}).get("pieces") or None
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
                    piece_set=(row or {}).get("set", "") or "",
                    corners=corners,
                    positioned=bool(pieces),
                    fen=(row or {}).get("fen", "") or "",
                    orientation=(row or {}).get("orientation", "") or "",
                    pieces=pieces,
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
        piece_set: str = "",
        device: str = "",
        surface: str = "",
        fen: str = "",
        orientation: str = "",
        pieces: Sequence[dict] | None = None,
    ) -> CornerLabel:
        """Normalize the inbox photo, write its store JPEG, and upsert its label row.

        `corners` may be four points in any order or a corner dict; they are sorted
        into visual TL/TR/BR/BL with `order_corners`. Re-saving the same `src`
        overwrites in place (the photo's id is derived from `src`).

        When `pieces` is given (the in-app position tool), the FEN, orientation, and
        nudged contact keypoints (`[{label, x, y}, ...]`, normalized-frame pixels) are
        stored too -- turning a corner photo into a piece-keypoint training sample. A
        corner-only re-save (``pieces=None``) drops any previously stored position, since
        re-marked corners invalidate the old projection.
        """
        path = self.inbox_path(src)
        raw = path.read_bytes()
        rgb, (w, h) = normalize_image(raw)
        exif_meta = extract_exif_meta(raw)  # publish-safe whitelist; GPS/serials never read
        label_id = _stable_id(src)
        image_rel = f"images/{label_id}.jpg"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        (self.store / image_rel).write_bytes(encode_jpeg(rgb))

        pts = list(corners.values()) if isinstance(corners, dict) else list(corners)
        ordered = order_corners(pts)  # {top_left, top_right, bottom_right, bottom_left}
        piece_tuples = tuple(
            (str(p["label"]), float(p["x"]), float(p["y"])) for p in (pieces or [])
        )
        label = CornerLabel(
            id=label_id,
            task_id=_task_id(label_id),
            src=src,
            image=image_rel,
            width=w,
            height=h,
            corners={k: (float(ordered[k][0]), float(ordered[k][1])) for k in CORNER_ORDER},
            board=board or "",
            piece_set=piece_set or "",
            device=device or exif_meta.get("device", ""),  # explicit arg wins; else EXIF
            surface=surface or "",
            labeled_at=datetime.now(UTC).isoformat(timespec="seconds"),
            captured_at=exif_meta.get("captured_at", ""),
            fen=fen or "",
            orientation=orientation or "",
            pieces=piece_tuples,
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

    def position_samples(self) -> list[CornerLabel]:
        """Labels that also carry placed pieces (the keypoint-head training set from the
        in-app position tool): all four corners, piece keypoints, and image on disk."""
        return [s for s in self.samples() if s.has_pieces]

    # ---- position library (named FENs reused across same-setup photos) ------

    @property
    def positions_library_path(self) -> Path:
        # Under store/ so it rides along with the synced labels (sync up --prefix corners).
        return self.store / "positions.json"

    def load_positions_library(self) -> dict[str, str]:
        """Saved {name: FEN placement field} entries; empty dict if none yet."""
        p = self.positions_library_path
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, AttributeError):
            return {}

    def save_position_entry(self, name: str, fen: str) -> dict[str, str]:
        """Upsert one named FEN into the library and return the full updated mapping."""
        lib = self.load_positions_library()
        lib[name] = fen
        self.store.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.positions_library_path, json.dumps(lib, indent=2, sort_keys=True))
        return lib


# --------------------------------------------------------------------------- #
# Session synthesis (group photos shot close in time on one board)
# --------------------------------------------------------------------------- #


def assign_sessions(
    samples: Sequence[CornerLabel], *, gap_minutes: float = 20.0
) -> dict[str, str]:
    """Map each label id -> a synthesized capture session id.

    Groups photos shot close in time on the same board into one session: walking the photos
    in capture-time order, a **new session starts when the board changes or the gap from the
    previous photo exceeds `gap_minutes`**. The session id is the burst's start time as
    ``YYYYMMDD-HHMMSS`` -- the same format the capture set uses, so the two are
    indistinguishable once merged. Samples without a ``captured_at`` are skipped (left
    unsessioned). Deterministic: ordered by (captured_at, id).
    """
    dated = sorted(
        (s for s in samples if s.captured_at), key=lambda s: (s.captured_at, s.id)
    )
    gap = timedelta(minutes=gap_minutes)
    out: dict[str, str] = {}
    cur: str | None = None
    prev_t: datetime | None = None
    prev_board: str | None = None
    for s in dated:
        try:
            t = datetime.fromisoformat(s.captured_at)
        except ValueError:
            continue
        board = s.board or "(untagged)"
        if cur is None or board != prev_board or (t - prev_t) > gap:
            cur = t.strftime("%Y%m%d-%H%M%S")
        out[s.id] = cur
        prev_t, prev_board = t, board
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
