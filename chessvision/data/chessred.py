"""ChessReD loader.

Parses the single ChessReD `annotations.json` (COCO-style: flat arrays cross-
referenced by image_id) into image_id-indexed records, and joins corners with
pieces. Corners exist only for the ~2,078-image `chessred2k` subset, so the
self-check iterates `images_with_corners()`, not all 10,800 images.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

EMPTY_CATEGORY_ID = 12


@dataclass(frozen=True)
class Piece:
    piece_id: int
    category_id: int
    square: str  # chessboard_position, e.g. "e4"
    # COCO [x, y, w, h]. Present only for the chessred2k subset; the full set
    # carries position-only labels (category + square) with no bbox.
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ImageMeta:
    image_id: int
    file_name: str
    path: str
    width: int
    height: int
    camera: str
    game_id: int
    move_id: int


@dataclass(frozen=True)
class AnnotatedImage:
    meta: ImageMeta
    corners: dict[str, list[float]]
    pieces: list[Piece]


@dataclass
class ChessReD:
    data_root: Path
    images_root: Path
    categories: dict[int, str]
    _meta: dict[int, ImageMeta] = field(repr=False)
    _corners: dict[int, dict[str, list[float]]] = field(repr=False)
    _pieces: dict[int, list[Piece]] = field(repr=False)
    # raw["splits"]: official image-id splits. "train"/"val"/"test" cover the full
    # 10,800-image set; the bbox-carrying chessred2k subset has its own nested
    # train/val/test (the detection splits used in Phase 2).
    _splits: dict = field(repr=False, default_factory=dict)

    @classmethod
    def load(cls, data_root: str | Path, images_root: str | Path | None = None) -> ChessReD:
        data_root = Path(data_root)
        annotations_path = data_root / "annotations.json"
        with annotations_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)

        categories = {c["id"]: c["name"] for c in raw["categories"]}
        meta = {
            i["id"]: ImageMeta(
                image_id=i["id"],
                file_name=i["file_name"],
                path=i["path"],
                width=i["width"],
                height=i["height"],
                camera=i.get("camera", ""),
                game_id=i["game_id"],
                move_id=i["move_id"],
            )
            for i in raw["images"]
        }
        corners = {c["image_id"]: c["corners"] for c in raw["annotations"]["corners"]}

        pieces: dict[int, list[Piece]] = defaultdict(list)
        for p in raw["annotations"]["pieces"]:
            if p["category_id"] == EMPTY_CATEGORY_ID:
                continue  # defensive: "empty" should never carry a bbox
            raw_bbox = p.get("bbox")
            bbox = tuple(float(v) for v in raw_bbox) if raw_bbox is not None else None
            pieces[p["image_id"]].append(
                Piece(
                    piece_id=p["id"],
                    category_id=p["category_id"],
                    square=p["chessboard_position"],
                    bbox=bbox,
                )
            )

        return cls(
            data_root=data_root,
            images_root=Path(images_root) if images_root else data_root / "chessred" / "images",
            categories=categories,
            _meta=meta,
            _corners=corners,
            _pieces=dict(pieces),
            _splits=raw.get("splits", {}),
        )

    def images_with_corners(self) -> Iterator[AnnotatedImage]:
        """Iterate the images that have a corners annotation (the self-check set)."""
        for image_id, corners in self._corners.items():
            yield AnnotatedImage(
                meta=self._meta[image_id],
                corners=corners,
                pieces=self._pieces.get(image_id, []),
            )

    def chessred2k_split(self, split: str) -> list[int]:
        """Image ids for an official chessred2k detection split ('train'|'val'|'test').

        chessred2k is the only subset with piece bounding boxes, so it is the
        Phase-2 detector's train/val/test source. Using the published split keeps
        mAP comparable and avoids game-level leakage across splits.
        """
        node = self._splits["chessred2k"][split]
        return list(node["image_ids"] if isinstance(node, dict) else node)

    def meta(self, image_id: int) -> ImageMeta:
        return self._meta[image_id]

    def pieces(self, image_id: int) -> list[Piece]:
        return self._pieces.get(image_id, [])

    def corners(self, image_id: int) -> dict[str, list[float]] | None:
        return self._corners.get(image_id)

    def resolve_image_path(self, meta: ImageMeta) -> Path:
        """On-disk path. Annotation `path` is `images/<game>/<file>`; the real tree
        is `<images_root>/<game>/<file>`, so strip the leading `images/` segment."""
        rel = meta.path.split("images/", 1)[-1]
        return self.images_root / rel

    @property
    def n_corner_images(self) -> int:
        return len(self._corners)


CornersLike = Mapping[str, Sequence[float]]
