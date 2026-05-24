"""Build Label Studio pre-annotation tasks from capture records.

Each capture session's `captures.jsonl` carries, per frame, the manually-marked
board `corners` and — projected from the known FEN through the homography — a
`base` ground-contact point and square-footprint `quad` for every piece. This
module turns those into Label Studio *prediction* tasks so the labelling project
opens with the board outline, a labelled base keypoint per piece, and an
approximate (adjustable) bounding box per piece already drawn. The aim is to
nudge boxes into place rather than draw them from scratch.

Tasks are emitted as one JSON file per frame, meant to live under a separate
bucket prefix (e.g. `tasks/`, same bucket as the images) that a Label Studio
source storage reads with **"Treat every bucket object as a source file" OFF**,
so each JSON is parsed as a task. The image is referenced as `s3://<bucket>/<key>`
so Label Studio presigns it from the storage credentials for that bucket.

Configure the Label Studio project's labelling interface to match the control
names (`board`, `pieces`, `bases`) and label values below — see ``LABELING_CONFIG``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

# Piece char -> bounding-box height as a multiple of the local square size, in
# pixels. Deliberately rough: the box is meant to be dragged into place, not
# trusted. Pawns sit about one square tall; the king nearly two. Width is a flat
# fraction of the square — pieces are narrower than the square they stand on.
PIECE_HEIGHT = {"p": 1.0, "n": 1.25, "b": 1.45, "r": 1.2, "q": 1.6, "k": 1.85}
PIECE_WIDTH_FRAC = 0.6

MODEL_VERSION = "capture-estimate"

# Record corner key -> Label Studio label. The four corners are marked as
# separate labeled keypoints (not one polygon), matching the project schema.
_CORNER_LABELS = {
    "top_left": "TopLeft",
    "top_right": "TopRight",
    "bottom_right": "BottomRight",
    "bottom_left": "BottomLeft",
}

# Verbose, human-readable label values (e.g. "WhiteRook"). These strings must
# match the labels in your Label Studio project exactly — a mismatch silently
# creates a second, unrelated class. Capture records store the piece as a FEN
# char (uppercase = white, lowercase = black), which we expand here.
_PIECE_NAMES = {"p": "Pawn", "n": "Knight", "b": "Bishop", "r": "Rook", "q": "Queen", "k": "King"}

# The 12 piece labels, white first, in pawn..king order.
_PIECE_LABELS = [f"White{_PIECE_NAMES[c]}" for c in "pnbrqk"] + [
    f"Black{_PIECE_NAMES[c]}" for c in "pnbrqk"
]


def _piece_labels_xml(indent: str) -> str:
    return "\n".join(f'{indent}<Label value="{label}"/>' for label in _PIECE_LABELS)


# Label Studio labelling config. The control `name`s here MUST match the
# `from_name` values emitted in predictions (`board`, `pieces`, `bases`).
LABELING_CONFIG = f"""\
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <KeyPointLabels name="corners" toName="image" opacity="0.9" strokeWidth="3">
    <Label value="TopLeft" background="#1f77b4"/>
    <Label value="TopRight" background="#ff7f0e"/>
    <Label value="BottomRight" background="#2ca02c"/>
    <Label value="BottomLeft" background="#d62728"/>
  </KeyPointLabels>
  <KeyPointLabels name="pieces" toName="image" opacity="0.9">
{_piece_labels_xml("    ")}
  </KeyPointLabels>
</View>"""


def piece_label(piece: dict) -> str:
    """Verbose label for a piece record, e.g. "WhiteRook" / "BlackPawn"."""
    color = "White" if piece["color"] == "w" else "Black"
    return f"{color}{_PIECE_NAMES[piece['piece'].lower()]}"


def _square_size(quad: list[list[float]]) -> float:
    """Mean of the quad's four edge lengths — a local pixels-per-square estimate.

    More stable than the quad's axis-aligned bbox, which collapses for squares
    seen edge-on under perspective.
    """
    total = 0.0
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        total += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return total / 4


def _piece_rect(piece: dict) -> tuple[float, float, float, float]:
    """Approximate piece bounding box (left, top, right, bottom) in pixels.

    Centred on the base (the footprint centroid), as wide as a fraction of the
    square and as tall as the per-type factor. The bottom sits at the footprint's
    front edge (largest y); the body rises upward from there. Pieces lean away
    from the camera, so the real top shifts toward the board centre — the box
    can't capture that, hence "adjust the borders".
    """
    quad = piece["quad"]
    cx, _cy = piece["base"]
    size = _square_size(quad)
    half_w = PIECE_WIDTH_FRAC * size / 2
    height = PIECE_HEIGHT[piece["piece"].lower()] * size
    bottom = max(p[1] for p in quad)
    top = bottom - height
    return cx - half_w, top, cx + half_w, bottom


def _pct(value: float, size: int) -> float:
    """Pixel coordinate -> percentage of image dimension, clamped to [0, 100]."""
    return max(0.0, min(100.0, 100.0 * value / size))


def _corner_regions(corners: dict, w: int, h: int) -> list[dict]:
    regions = []
    for key, label in _CORNER_LABELS.items():
        x, y = corners[key]
        regions.append(
            {
                "type": "keypointlabels",
                "from_name": "corners",
                "to_name": "image",
                "original_width": w,
                "original_height": h,
                "value": {
                    "x": _pct(x, w),
                    "y": _pct(y, h),
                    "width": 0.5,
                    "keypointlabels": [label],
                },
            }
        )
    return regions


def _rect_region(piece: dict, w: int, h: int) -> dict:
    left, top, right, bottom = _piece_rect(piece)
    x, y = _pct(left, w), _pct(top, h)
    return {
        "type": "rectanglelabels",
        "from_name": "boxes",
        "to_name": "image",
        "original_width": w,
        "original_height": h,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": y,
            "width": max(0.0, _pct(right, w) - x),
            "height": max(0.0, _pct(bottom, h) - y),
            "rotation": 0,
            "rectanglelabels": [piece_label(piece)],
        },
    }


def _piece_point_region(piece: dict, w: int, h: int) -> dict:
    """A piece as a single labelled keypoint at its base — the points-only
    annotation that carries class + ground-contact location in one click."""
    x, y = piece["base"]
    return {
        "type": "keypointlabels",
        "from_name": "pieces",
        "to_name": "image",
        "original_width": w,
        "original_height": h,
        "value": {
            "x": _pct(x, w),
            "y": _pct(y, h),
            "width": 0.5,
            "keypointlabels": [piece_label(piece)],
        },
    }


def build_task(
    record: dict,
    image_size: tuple[int, int],
    image_uri: str,
    *,
    model_version: str = MODEL_VERSION,
    include_boxes: bool = False,
) -> dict:
    """One Label Studio task (data + a single prediction) for a capture record.

    Points-only by default: the four corner keypoints (control ``corners``) plus
    one labelled keypoint per piece at its base (control ``pieces``) — the simple,
    geometry-critical annotations. Pass ``include_boxes=True`` to additionally emit
    an approximate adjustable bounding box per piece (control ``boxes``), for the
    later box-detection pass; that control isn't in LABELING_CONFIG, so add a
    ``<RectangleLabels name="boxes">`` to the project before using it.
    """
    w, h = image_size
    result: list[dict] = []
    if record.get("corners"):
        result.extend(_corner_regions(record["corners"], w, h))
    for piece in record.get("pieces") or []:
        if include_boxes:
            result.append(_rect_region(piece, w, h))
        result.append(_piece_point_region(piece, w, h))
    return {
        "data": {"image": image_uri},
        "predictions": [{"model_version": model_version, "result": result}],
    }


def iter_records(captures_dir: str | Path) -> Iterator[tuple[str, dict]]:
    """Yield (session_id, record) for every line of every session's captures.jsonl."""
    for jsonl in sorted(Path(captures_dir).glob("*/captures.jsonl")):
        session_id = jsonl.parent.name
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield session_id, json.loads(line)


def image_key(captures_prefix: str, session_id: str, filename: str) -> str:
    return f"{captures_prefix.strip('/')}/{session_id}/{filename}"


def task_key(tasks_prefix: str, session_id: str, filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return f"{tasks_prefix.strip('/')}/{session_id}/{stem}.json"


def image_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def image_size_from_path(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def image_size_from_bytes(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size
