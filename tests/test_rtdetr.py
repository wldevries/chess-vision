"""RT-DETRv2 adapter checks that don't need `transformers` installed.

The `rtdetr` group (transformers) is optional, so these cover only the pure-Python glue:
label maps, the uint8 conversion, and `RTDetrCollate`'s FRCNN->RT-DETR target rewrite (driven
with a stub processor that records what it was handed). Training/eval that actually loads the
model is exercised by the smoke run, not here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chessvision.data.detection import LABEL_NAMES, NUM_PIECE_CLASSES  # noqa: E402
from chessvision.rtdetr import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    RTDetrCollate,
    _to_uint8_hwc,
)


def test_label_maps_match_frcnn_names_zero_indexed():
    assert len(ID2LABEL) == NUM_PIECE_CLASSES == 12
    # RT-DETR id i corresponds to FRCNN label i+1 (no background slot).
    assert ID2LABEL[0] == LABEL_NAMES[1] == "white-pawn"
    assert ID2LABEL[11] == LABEL_NAMES[12] == "black-king"
    # round-trips
    assert all(LABEL2ID[name] == i for i, name in ID2LABEL.items())


def test_to_uint8_hwc_roundtrips_range_and_layout():
    img = torch.rand(3, 5, 7)  # CxHxW in [0, 1]
    arr = _to_uint8_hwc(img)
    assert arr.shape == (5, 7, 3) and arr.dtype.name == "uint8"
    assert arr.min() >= 0 and arr.max() <= 255


class _StubProcessor:
    """Records the images/annotations it receives instead of running a model processor."""

    def __init__(self):
        self.calls = []

    def __call__(self, images, annotations, return_tensors):
        self.calls.append((images, annotations))
        return {"pixel_values": None, "labels": annotations}


def test_collate_rewrites_targets_to_coco_zero_indexed():
    proc = _StubProcessor()
    collate = RTDetrCollate(proc)
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 60.0]]),  # xyxy
        "labels": torch.tensor([12]),  # FRCNN black-king
        "image_id": torch.tensor([7]),
    }
    collate([(torch.rand(3, 64, 64), target)])

    (_images, annotations) = proc.calls[0]
    ann = annotations[0]
    assert ann["image_id"] == 7
    box = ann["annotations"][0]
    assert box["category_id"] == 11  # 12 -> 0-indexed 11
    assert box["bbox"] == [10.0, 20.0, 20.0, 40.0]  # xyxy -> xywh
    assert box["area"] == 800.0 and box["iscrowd"] == 0
