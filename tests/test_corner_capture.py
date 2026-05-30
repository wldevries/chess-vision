"""Tests for the standalone corner dataset + corner-label endpoints."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from chessvision.capture.app import create_app
from chessvision.data.corner_capture import (
    CornerStore,
    normalize_image,
    select_corner_dataset_poses,
)


def _write_inbox_jpeg(
    path: Path, size=(120, 80), color=(180, 60, 40), orientation: int | None = None
):
    """Write a JPEG into the inbox, optionally with an EXIF orientation flag."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    if orientation is not None:
        exif = im.getexif()
        exif[274] = orientation  # 274 = Orientation tag
        im.save(path, "JPEG", exif=exif)
    else:
        im.save(path, "JPEG")


# Four corners (any order) for a 120x80 image, well inside the frame.
SQUARE = [[20.0, 15.0], [100.0, 15.0], [100.0, 65.0], [20.0, 65.0]]


# --- normalization ---------------------------------------------------------- #


def test_normalize_bakes_in_exif_orientation() -> None:
    buf = io.BytesIO()
    im = Image.new("RGB", (120, 80), (10, 20, 30))
    exif = im.getexif()
    exif[274] = 6  # rotate 90° CW on display -> dimensions swap to 80x120
    im.save(buf, "JPEG", exif=exif)

    rgb, (w, h) = normalize_image(buf.getvalue())
    assert (w, h) == (80, 120)
    assert rgb.shape == (120, 80, 3)


# --- store: save / list / samples ------------------------------------------- #


def test_save_label_writes_store_image_and_row(tmp_path: Path) -> None:
    store = CornerStore(tmp_path / "corners")
    _write_inbox_jpeg(store.inbox / "2026-05-27" / "IMG_1.jpg")

    label = store.save_label("2026-05-27/IMG_1.jpg", SQUARE, board="staunton-56mm", device="Pixel")

    # The normalized store JPEG exists and the label row is in labels.jsonl.
    assert (store.store / label.image).exists()
    assert set(label.corners) == {"top_left", "top_right", "bottom_right", "bottom_left"}
    # order_corners sorted the points into visual slots regardless of input order.
    assert label.corners["top_left"] == (20.0, 15.0)
    assert label.corners["bottom_right"] == (100.0, 65.0)
    assert label.board == "staunton-56mm"
    rows = store.load_labels()
    # id == src == image == the source-relative path ("inbox/" + the inbox-relative path).
    assert len(rows) == 1 and next(iter(rows.values()))["src"] == "inbox/2026-05-27/IMG_1.jpg"
    assert label.id == label.image == label.src == "inbox/2026-05-27/IMG_1.jpg"


def test_resaving_same_src_overwrites_in_place(tmp_path: Path) -> None:
    store = CornerStore(tmp_path / "corners")
    _write_inbox_jpeg(store.inbox / "a.jpg")
    store.save_label("a.jpg", SQUARE, board="cheap-30mm")
    store.save_label("a.jpg", SQUARE, board="rimless-45mm")  # re-label
    rows = store.load_labels()
    assert len(rows) == 1  # not duplicated
    assert next(iter(rows.values()))["board"] == "rimless-45mm"


def test_list_inbox_reports_labeled_state_and_groups(tmp_path: Path) -> None:
    store = CornerStore(tmp_path / "corners")
    _write_inbox_jpeg(store.inbox / "day1" / "a.jpg")
    _write_inbox_jpeg(store.inbox / "b.jpg")  # at the inbox root
    store.save_label("day1/a.jpg", SQUARE, board="staunton-56mm")

    listing = {p.src: p for p in store.list_inbox()}
    assert listing["day1/a.jpg"].labeled and listing["day1/a.jpg"].group == "day1"
    assert listing["day1/a.jpg"].board == "staunton-56mm"
    assert not listing["b.jpg"].labeled and listing["b.jpg"].group == ""


def test_samples_require_corners_and_image(tmp_path: Path) -> None:
    store = CornerStore(tmp_path / "corners")
    _write_inbox_jpeg(store.inbox / "a.jpg")
    store.save_label("a.jpg", SQUARE)
    samples = store.samples()
    assert len(samples) == 1 and samples[0].has_all_corners


def test_inbox_path_rejects_traversal(tmp_path: Path) -> None:
    store = CornerStore(tmp_path / "corners")
    store.inbox.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        store.inbox_path("../secret.jpg")


# --- split: no board / pose leakage ----------------------------------------- #


def test_split_holds_out_whole_poses_per_board(tmp_path: Path) -> None:
    store = CornerStore(tmp_path / "corners")
    # Two boards, several distinct poses each (corner quads >> dedup_thr apart).
    for board in ("staunton-56mm", "cheap-30mm"):
        for k in range(4):
            shift = 4.0 + k * 8.0  # 120px image, dedup_thr=0.02 -> ~2.4px; 8px steps are distinct
            pts = [
                [shift, shift],
                [110 - shift, shift],
                [110 - shift, 70 - shift],
                [shift, 70 - shift],
            ]
            name = f"{board}_{k}.jpg"
            _write_inbox_jpeg(store.inbox / name)
            store.save_label(name, pts, board=board)

    train, heldout = select_corner_dataset_poses(store, val_frac=0.25)
    assert train and heldout
    # No held-out frame's pose appears in train (anti-leak): all train/heldout disjoint by src.
    train_src = {s.src for s in train}
    heldout_src = {s.src for s in heldout}
    assert train_src.isdisjoint(heldout_src)
    # Both boards contribute held-out poses.
    assert {s.board for s in heldout} == {"staunton-56mm", "cheap-30mm"}


# --- endpoints -------------------------------------------------------------- #


@pytest.fixture
def corner_client(tmp_path: Path) -> TestClient:
    store = CornerStore(tmp_path / "corners")
    _write_inbox_jpeg(store.inbox / "2026-05-27" / "IMG_1.jpg")
    return TestClient(create_app(tmp_path / "captures", corner_store=store))


def test_corner_label_off_by_default(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/corners-label/available").json()["available"] is False
    assert client.get("/api/corners-label/inbox").status_code == 503


def test_corner_label_inbox_image_and_save(corner_client: TestClient, tmp_path: Path) -> None:
    assert corner_client.get("/api/corners-label/available").json()["available"] is True

    inbox = corner_client.get("/api/corners-label/inbox").json()
    assert len(inbox) == 1 and inbox[0]["src"] == "2026-05-27/IMG_1.jpg"
    assert inbox[0]["labeled"] is False

    img = corner_client.get("/api/corners-label/image", params={"src": "2026-05-27/IMG_1.jpg"})
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(img.content)).size == (120, 80)

    resp = corner_client.post(
        "/api/corners-label/save",
        json={"src": "2026-05-27/IMG_1.jpg", "corners": SQUARE, "board": "staunton-56mm"},
    )
    assert resp.status_code == 200 and resp.json()["labeled"] is True
    # Now flagged labelled in the listing.
    assert corner_client.get("/api/corners-label/inbox").json()[0]["labeled"] is True
    # And persisted to disk under the corners root (labels.jsonl now at the root, flat layout).
    labels = (tmp_path / "corners" / "labels.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(labels)["board"] == "staunton-56mm"


def test_corner_label_save_rejects_wrong_point_count(corner_client: TestClient) -> None:
    resp = corner_client.post(
        "/api/corners-label/save",
        json={"src": "2026-05-27/IMG_1.jpg", "corners": [[1, 2], [3, 4], [5, 6]]},
    )
    assert resp.status_code == 400


def test_corner_label_image_404_for_unknown(corner_client: TestClient) -> None:
    assert (
        corner_client.get("/api/corners-label/image", params={"src": "nope.jpg"}).status_code == 404
    )


# --- training dataset ------------------------------------------------------- #


def test_corner_capture_dataset_yields_image_and_normalized_corners(tmp_path: Path) -> None:
    from chessvision.data.corner_capture import CornerCaptureDataset
    from chessvision.data.corners import CornerConfig

    store = CornerStore(tmp_path / "corners")
    _write_inbox_jpeg(store.inbox / "a.jpg")
    label = store.save_label("a.jpg", SQUARE)

    ds = CornerCaptureDataset([label], store, CornerConfig(image_size=64), train=False)
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    corners = target["corners"]
    assert corners.shape == (4, 2)
    # Stored in a 120x80 frame -> top_left (20,15) normalizes to (20/120, 15/80).
    assert float(corners[0, 0]) == pytest.approx(20 / 120, abs=1e-4)
    assert float(corners[0, 1]) == pytest.approx(15 / 80, abs=1e-4)
