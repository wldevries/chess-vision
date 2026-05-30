"""Capture-app dataset loader.

Parses the Label Studio JSON export (`data/captures/label-studio.json`) into
per-image samples: an on-disk image path, the four board corners, and the piece
keypoints. This is the labelled real-photo set produced by the capture web app
(`chessvision.capture`) and hand-corrected in Label Studio.

Corner keys are emitted in the snake_case form `geometry.compute_homography`
expects (`top_left`/`top_right`/`bottom_right`/`bottom_left`), so a sample's
`corners` can be fed straight into the Phase-1 homography utility.

Coordinates: Label Studio stores keypoints as **percentages** of the original
image size (0-100). We convert to absolute pixels on load.

Image bytes: the export stores `s3://<bucket>/captures/<session>/<file>.jpg`. The
loader prefers the **local** mirror under `<captures_root>/<session>/<file>.jpg`
and **falls back to S3/MinIO** when the local file is absent. S3 credentials come
from `.env` (see `StorageConfig.try_from_env`); the file is gitignored.

Parallel loading: `CaptureDataset.load_images` decodes many images at once.
Default is a thread pool (image load is I/O-bound: an S3 GET plus a JPEG decode).
A process pool is also supported and is **Windows-safe** -- the worker
(`_load_image_worker`) and everything sent to it (`_LoadSpec`, `StorageConfig`) are
module-level / frozen and therefore picklable under spawn, and the boto3 client
is built lazily *inside* each worker (clients are not picklable) rather than
shipped across the process boundary. When using a process pool on Windows, the
caller's entry point must be guarded by ``if __name__ == "__main__":``.

.. warning::
    **Splits MUST be grouped by session, never by random image.** Every image
    comes from one of a handful of capture *sessions* (the path segment after
    ``/captures/``), and a session is a single board / piece-set / room / camera
    pose. A random per-image split leaks near-duplicate frames of the same scene
    across train/val and the validation number becomes a lie -- exactly the
    overfit-one-domain failure plan.md section 5 is designed against. Use
    ``sample.session`` as the group key (see ``CaptureDataset.by_session``).

    The current export is also *low diversity* (a few sessions, likely one board
    and piece set). It is a bootstrap to prove the corner-regression pipeline and
    get a first number -- not a set that will generalize on its own.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from chessvision.data.storage import StorageConfig, get_client

# Label Studio corner label -> the snake_case key geometry.py uses.
_CORNER_KEY: dict[str, str] = {
    "TopLeft": "top_left",
    "TopRight": "top_right",
    "BottomRight": "bottom_right",
    "BottomLeft": "bottom_left",
}
CORNER_KEYS: frozenset[str] = frozenset(_CORNER_KEY.values())

# Label Studio piece label -> FEN character (uppercase = white, lowercase = black).
PIECE_FEN: dict[str, str] = {
    "WhitePawn": "P",
    "WhiteRook": "R",
    "WhiteKnight": "N",
    "WhiteBishop": "B",
    "WhiteQueen": "Q",
    "WhiteKing": "K",
    "BlackPawn": "p",
    "BlackRook": "r",
    "BlackKnight": "n",
    "BlackBishop": "b",
    "BlackQueen": "q",
    "BlackKing": "k",
}


# --------------------------------------------------------------------------- #
# S3 / MinIO access
# --------------------------------------------------------------------------- #


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/path/to/obj.jpg` -> ("bucket", "path/to/obj.jpg")."""
    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def _read_bytes(local_path: str, s3_uri: str, s3: StorageConfig | None) -> bytes:
    """Local file if it exists, else an S3 GET. Raises if neither is reachable."""
    p = Path(local_path)
    if p.exists():
        return p.read_bytes()
    if s3 is None:
        raise FileNotFoundError(
            f"{local_path} not found locally and no S3 config given for {s3_uri}"
        )
    bucket, key = _split_s3_uri(s3_uri)
    resp = get_client(s3).get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _decode_rgb(buf: bytes) -> np.ndarray:
    """JPEG bytes -> (H, W, 3) uint8 RGB array (ML-friendly channel order)."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class _LoadSpec:
    """Picklable unit of work for the parallel image loader (no live clients)."""

    task_id: int
    local_path: str
    s3_uri: str
    s3: StorageConfig | None


def _load_image_worker(spec: _LoadSpec) -> tuple[int, np.ndarray]:
    """Module-level so it is picklable for ProcessPoolExecutor under spawn."""
    return spec.task_id, _decode_rgb(_read_bytes(spec.local_path, spec.s3_uri, spec.s3))


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PieceKeypoint:
    label: str  # Label Studio class, e.g. "BlackRook"
    point: tuple[float, float]  # (x, y) in image pixels

    @property
    def fen(self) -> str:
        return PIECE_FEN[self.label]


@dataclass(frozen=True)
class CaptureSample:
    task_id: int
    session: str  # capture-session id; the group key for train/val/test splits
    image_path: Path  # local on-disk path (may not exist -> S3 fallback)
    s3_uri: str  # authoritative remote location, e.g. s3://chess/captures/.../x.jpg
    width: int
    height: int
    # snake_case corner key -> (x, y) in image pixels; ready for geometry.compute_homography.
    corners: dict[str, tuple[float, float]]
    pieces: list[PieceKeypoint]
    # Resolved per-piece box sizing from session metadata: fen-letter -> (height_squares,
    # radius_squares). None/absent -> box synthesis falls back to PIECE_HEIGHT_SCALE.
    box_sizes: dict[str, tuple[float, float]] | None = None

    @property
    def has_all_corners(self) -> bool:
        return set(self.corners) == CORNER_KEYS

    def read_bytes(self, s3: StorageConfig | None = None) -> bytes:
        """Raw JPEG bytes: local file if present, else S3 fallback."""
        return _read_bytes(str(self.image_path), self.s3_uri, s3)

    def load_image(self, s3: StorageConfig | None = None) -> np.ndarray:
        """Decoded (H, W, 3) uint8 RGB image: local if present, else S3 fallback."""
        return _decode_rgb(self.read_bytes(s3))


@dataclass
class CaptureDataset:
    export_path: Path
    captures_root: Path
    samples: list[CaptureSample] = field(repr=False)
    s3: StorageConfig | None = None  # used only when a local file is missing

    @classmethod
    def load(
        cls,
        export_path: str | Path,
        captures_root: str | Path | None = None,
        s3: StorageConfig | None = None,
    ) -> CaptureDataset:
        """Parse the Label Studio export.

        `captures_root` defaults to the export's own directory, which is correct
        for the committed layout (`data/captures/label-studio.json` alongside the
        `<session>/` image dirs). `s3` defaults to `StorageConfig.try_from_env()`,
        so the S3 fallback is wired up automatically when `.env` holds MinIO creds
        and is simply unavailable (local-only) otherwise.
        """
        export_path = Path(export_path)
        captures_root = Path(captures_root) if captures_root else export_path.parent
        if s3 is None:
            s3 = StorageConfig.try_from_env()
        with export_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        samples = [s for s in (cls._parse_task(t, captures_root) for t in raw) if s is not None]

        # Attach per-piece box sizing from session metadata (sets/boards/sessions JSON),
        # so box synthesis sizes pieces by physical mm/square instead of a global constant.
        from chessvision.data.session_meta import SessionMetadata

        meta = SessionMetadata.load(captures_root)
        if meta is not None:
            samples = [
                replace(s, box_sizes=meta.piece_box_sizes(s.session) or None) for s in samples
            ]
        return cls(export_path=export_path, captures_root=captures_root, samples=samples, s3=s3)

    @staticmethod
    def _parse_task(task: dict, captures_root: Path) -> CaptureSample | None:
        annotations = task.get("annotations") or []
        if not annotations:
            return None
        results = annotations[0].get("result") or []
        if not results:
            return None

        width = int(results[0]["original_width"])
        height = int(results[0]["original_height"])
        corners: dict[str, tuple[float, float]] = {}
        pieces: list[PieceKeypoint] = []
        for r in results:
            val = r["value"]
            label = val["keypointlabels"][0]
            # x/y are percentages of this result's own original_width/height.
            px = (
                val["x"] / 100.0 * r["original_width"],
                val["y"] / 100.0 * r["original_height"],
            )
            if r["from_name"] == "corners":
                key = _CORNER_KEY.get(label)
                if key is not None:
                    corners[key] = px  # last write wins on duplicate labels
            elif r["from_name"] == "pieces" and label in PIECE_FEN:
                pieces.append(PieceKeypoint(label=label, point=px))

        image_uri = task["data"]["image"]
        rel = image_uri.split("/captures/", 1)[-1]  # "<session>/<file>.jpg"
        session = rel.split("/", 1)[0]
        return CaptureSample(
            task_id=int(task["id"]),
            session=session,
            image_path=captures_root / Path(rel),
            s3_uri=image_uri,
            width=width,
            height=height,
            corners=corners,
            pieces=pieces,
        )

    def with_all_corners(self) -> Iterator[CaptureSample]:
        """Samples that carry all four corners -- the corner-regression training set."""
        for s in self.samples:
            if s.has_all_corners:
                yield s

    def by_session(self) -> dict[str, list[CaptureSample]]:
        """Group samples by capture session. Use this to build session-grouped
        splits (see the module warning)."""
        groups: dict[str, list[CaptureSample]] = defaultdict(list)
        for s in self.samples:
            groups[s.session].append(s)
        return dict(groups)

    @property
    def sessions(self) -> list[str]:
        return sorted({s.session for s in self.samples})

    def load_images(
        self,
        samples: Iterable[CaptureSample] | None = None,
        *,
        max_workers: int = 8,
        use_processes: bool = False,
    ) -> dict[int, np.ndarray]:
        """Decode images in parallel, keyed by `task_id`.

        Threads (default) are the right tool here -- each item is an S3 GET and/or
        a JPEG decode (both release the GIL), so threads scale without the pickle
        and spawn overhead of processes. Pass `use_processes=True` to fan decoding
        across cores; it is Windows-safe (see the module docstring) but the
        caller must guard their entry point with `if __name__ == "__main__":`.
        """
        chosen = list(samples if samples is not None else self.samples)
        specs = [
            _LoadSpec(task_id=s.task_id, local_path=str(s.image_path), s3_uri=s.s3_uri, s3=self.s3)
            for s in chosen
        ]
        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        with executor_cls(max_workers=max_workers) as pool:
            return {task_id: img for task_id, img in pool.map(_load_image_worker, specs)}

    def __len__(self) -> int:
        return len(self.samples)


