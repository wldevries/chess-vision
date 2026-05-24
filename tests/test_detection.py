"""ChessReD detection-dataset checks. Skipped when the dataset isn't present."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chessvision.data.chessred import ChessReD  # noqa: E402
from chessvision.data.detection import (  # noqa: E402
    LABEL_NAMES,
    NUM_CLASSES,
    ChessReDDetection,
    DetectionConfig,
    collate_detection,
)

DATA_ROOT = Path("data/Chess Recognition Dataset (ChessReD)_2_all")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "annotations.json").exists(),
    reason="ChessReD dataset not present",
)


@pytest.fixture(scope="module")
def chessred() -> ChessReD:
    return ChessReD.load(DATA_ROOT)


def test_split_sizes(chessred: ChessReD):
    assert len(chessred.chessred2k_split("train")) == 1442
    assert len(chessred.chessred2k_split("val")) == 330
    assert len(chessred.chessred2k_split("test")) == 306


def test_label_space():
    assert NUM_CLASSES == 13  # 12 pieces + background
    assert LABEL_NAMES[0] == "__background__"
    assert LABEL_NAMES[1] == "white-pawn"
    assert LABEL_NAMES[12] == "black-king"


def test_item_shapes(chessred: ChessReD):
    ds = ChessReDDetection.from_split(chessred, "val", config=DetectionConfig(max_size=640))
    img, target = ds[0]
    assert img.dtype == torch.float32 and img.ndim == 3 and img.shape[0] == 3
    assert max(img.shape[1:]) <= 640  # long side capped
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0

    boxes, labels = target["boxes"], target["labels"]
    assert boxes.ndim == 2 and boxes.shape[1] == 4 and len(boxes) == len(labels)
    assert boxes.numel() > 0
    # xyxy ordering and labels in the piece range
    assert torch.all(boxes[:, 2] >= boxes[:, 0]) and torch.all(boxes[:, 3] >= boxes[:, 1])
    assert int(labels.min()) >= 1 and int(labels.max()) <= 12


def test_collate_keeps_ragged(chessred: ChessReD):
    ds = ChessReDDetection.from_split(chessred, "val", config=DetectionConfig(max_size=640))
    images, targets = collate_detection([ds[0], ds[1]])
    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and "boxes" in targets[0]


def test_hflip_preserves_box_validity(chessred: ChessReD):
    cfg = DetectionConfig(max_size=640, hflip_prob=1.0)
    ds = ChessReDDetection.from_split(chessred, "val", config=cfg, train=True)
    _, target = ds[0]
    boxes = target["boxes"]
    assert torch.all(boxes[:, 2] > boxes[:, 0]) and torch.all(boxes[:, 3] > boxes[:, 1])
