"""Capture-app loader sanity checks. Skipped when the export isn't present."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chessvision.data.captures import (
    CORNER_KEYS,
    CaptureDataset,
    _split_s3_uri,
)
from chessvision.data.storage import StorageConfig
from chessvision.geometry import compute_homography, quad_area

EXPORT = Path("data/captures/label-studio.json")

pytestmark = pytest.mark.skipif(
    not EXPORT.exists(),
    reason="capture export not present",
)


@pytest.fixture(scope="module")
def dataset() -> CaptureDataset:
    return CaptureDataset.load(EXPORT)


def test_loads_all_tasks(dataset: CaptureDataset):
    # The export grows as more photos are labelled; assert we parse every
    # annotated task rather than hardcoding a count.
    raw = json.loads(EXPORT.read_text(encoding="utf-8"))
    annotated = sum(1 for t in raw if t.get("annotations"))
    assert len(dataset) == annotated
    assert len(dataset) >= 169  # never regress below the first labelled batch


def test_every_sample_has_four_corners(dataset: CaptureDataset):
    assert all(s.has_all_corners for s in dataset.samples)
    assert set(next(iter(dataset.samples)).corners) == set(CORNER_KEYS)


def test_local_image_paths_exist(dataset: CaptureDataset):
    missing = [s.image_path for s in dataset.samples if not s.image_path.exists()]
    assert not missing, f"{len(missing)} referenced images not found, e.g. {missing[:3]}"


def test_corners_are_pixels_within_bounds(dataset: CaptureDataset):
    s = next(iter(dataset.samples))
    for x, y in s.corners.values():
        assert 0 <= x <= s.width
        assert 0 <= y <= s.height


def test_corners_feed_geometry(dataset: CaptureDataset):
    """The emitted corner dict plugs straight into the Phase-1 homography."""
    s = next(iter(dataset.samples))
    assert quad_area(s.corners) > 0
    H = compute_homography(s.corners)
    assert H.shape == (3, 3)


def test_pieces_have_fen_chars(dataset: CaptureDataset):
    s = max(dataset.samples, key=lambda s: len(s.pieces))
    assert s.pieces
    assert all(p.fen in "PRNBQKprnbqk" for p in s.pieces)


def test_sessions_present(dataset: CaptureDataset):
    groups = dataset.by_session()
    assert len(groups) >= 4
    assert sum(len(v) for v in groups.values()) == len(dataset)


def test_split_s3_uri():
    bucket, key = _split_s3_uri("s3://chess/captures/sess/file.jpg")
    assert bucket == "chess"
    assert key == "captures/sess/file.jpg"


def test_sample_keeps_s3_uri(dataset: CaptureDataset):
    s = next(iter(dataset.samples))
    assert s.s3_uri.startswith("s3://")
    assert "/captures/" in s.s3_uri


def test_s3config_repr_hides_secret():
    cfg = StorageConfig(endpoint_url="http://h:9000", access_key="ak", secret_key="topsecret")
    assert "topsecret" not in repr(cfg)
    assert "***" in repr(cfg)


def test_load_images_threads(dataset: CaptureDataset):
    """Parallel decode of a few local images (I/O-bound thread-pool default)."""
    sample = dataset.samples[:4]
    images = dataset.load_images(sample, max_workers=4)
    assert set(images) == {s.task_id for s in sample}
    for s in sample:
        img = images[s.task_id]
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.shape[:2] == (s.height, s.width)


def test_load_images_processes(dataset: CaptureDataset):
    """Process-pool path must work under Windows spawn (proves picklability)."""
    sample = dataset.samples[:2]
    images = dataset.load_images(sample, max_workers=2, use_processes=True)
    assert set(images) == {s.task_id for s in sample}
    assert all(img.ndim == 3 for img in images.values())
