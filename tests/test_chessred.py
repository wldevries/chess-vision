"""ChessReD loader sanity checks. Skipped when the dataset isn't present."""

from __future__ import annotations

from pathlib import Path

import pytest

from chessvision.data.chessred import ChessReD

DATA_ROOT = Path("data/othersets/ChessReD")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "annotations.json").exists(),
    reason="ChessReD dataset not present",
)


@pytest.fixture(scope="module")
def dataset() -> ChessReD:
    return ChessReD.load(DATA_ROOT)


def test_categories(dataset: ChessReD):
    assert len(dataset.categories) == 13
    assert dataset.categories[0] == "white-pawn"
    assert dataset.categories[12] == "empty"


def test_corner_image_count(dataset: ChessReD):
    assert dataset.n_corner_images == pytest.approx(2078, abs=50)


def test_annotated_image_shape(dataset: ChessReD):
    img = next(dataset.images_with_corners())
    assert set(img.corners) == {"top_left", "top_right", "bottom_left", "bottom_right"}
    assert all(p.category_id != 12 for p in img.pieces)


def test_corner_images_have_bboxes(dataset: ChessReD):
    """Corner-annotated images are the chessred2k subset, where pieces carry bboxes."""
    img = next(dataset.images_with_corners())
    assert img.pieces and all(p.bbox is not None and len(p.bbox) == 4 for p in img.pieces)
