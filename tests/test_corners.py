"""ChessReD corner-dataset + corner-model checks. Skipped when data/torch absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessvision.data.chessred import ChessReD  # noqa: E402
from chessvision.data.corners import (  # noqa: E402
    CORNER_ORDER,
    ChessReDCorners,
    CornerConfig,
    collate_corners,
    corners_to_array,
)

DATA_ROOT = Path("data/othersets/ChessReD")

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "annotations.json").exists(),
    reason="ChessReD dataset not present",
)


@pytest.fixture(scope="module")
def chessred() -> ChessReD:
    return ChessReD.load(DATA_ROOT)


def test_corners_to_array_is_visual_slots():
    # A square rotated so the annotation keys do NOT match visual position:
    # the min-y point must end up as TL regardless of its semantic key.
    ann = {
        "top_left": [10.0, 50.0],
        "top_right": [80.0, 10.0],  # actually the visually-topmost point
        "bottom_right": [90.0, 60.0],
        "bottom_left": [20.0, 95.0],
    }
    arr = corners_to_array(ann)
    assert arr.shape == (4, 2)
    tl, tr, br, bl = arr
    # visual slots: top pair above bottom pair; left.x < right.x within each pair
    assert tl[1] < bl[1] and tr[1] < br[1]
    assert tl[0] < tr[0] and bl[0] < br[0]


def test_split_sizes(chessred: ChessReD):
    # corners exist exactly for the chessred2k subset
    assert len(ChessReDCorners.from_split(chessred, "train")) == 1442
    assert len(ChessReDCorners.from_split(chessred, "val")) == 330
    assert len(ChessReDCorners.from_split(chessred, "test")) == 306


def test_item_shapes_and_normalization(chessred: ChessReD):
    ds = ChessReDCorners.from_split(chessred, "val", config=CornerConfig(image_size=256))
    img, target = ds[0]
    assert img.dtype == torch.float32 and img.shape == (3, 256, 256)
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0

    corners = target["corners"]
    assert corners.shape == (4, 2)
    # normalized to [0, 1]
    assert float(corners.min()) >= 0.0 and float(corners.max()) <= 1.0
    # visual-slot ordering holds on real data too
    tl, tr, br, bl = corners
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]


def test_collate_stacks(chessred: ChessReD):
    ds = ChessReDCorners.from_split(chessred, "val", config=CornerConfig(image_size=128))
    images, targets = collate_corners([ds[0], ds[1]])
    assert images.shape == (2, 3, 128, 128)
    assert targets["corners"].shape == (2, 4, 2)
    assert targets["orig_size"].shape == (2, 2)


def test_hflip_keeps_valid_visual_quad(chessred: ChessReD):
    cfg = CornerConfig(image_size=256, hflip_prob=1.0)
    ds = ChessReDCorners.from_split(chessred, "val", config=cfg, train=True)
    _, target = ds[0]
    tl, tr, br, bl = target["corners"]
    # after flip + re-canonicalization the slot ordering must still hold
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]
    assert float(target["corners"].min()) >= 0.0 and float(target["corners"].max()) <= 1.0


def test_augment_keeps_valid_in_frame_quad(chessred: ChessReD):
    """Colour + geometric augment, forced on, must keep corners in [0, 1] and in
    visual-slot order (geometric self-skips any sample it would push off-frame)."""
    cfg = CornerConfig(
        image_size=256,
        hflip_prob=0.5,
        jitter=0.2,
        hue=0.1,
        saturation=0.4,
        grayscale_prob=0.5,
        rotate=5.0,
        scale=0.1,
        perspective=0.04,
    )
    ds = ChessReDCorners.from_split(chessred, "val", config=cfg, train=True)
    for _ in range(20):  # augment is stochastic -- sample many draws of the same item
        img, target = ds[0]
        assert img.shape == (3, 256, 256)
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
        c = target["corners"]
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
        tl, tr, br, bl = c
        assert tl[0] < tr[0] and bl[0] < br[0]
        assert tl[1] < bl[1] and tr[1] < br[1]


def test_geometric_can_change_the_image(chessred: ChessReD):
    """Sanity: with geometric augment on, at least one draw differs from the unaugmented
    image (i.e. the warp is actually applied, not always skipped)."""
    from chessvision.data.corners import _CornerDataset  # noqa: F401

    base = ChessReDCorners.from_split(
        chessred, "val", config=CornerConfig(image_size=256), train=False
    )
    plain, _ = base[0]
    cfg = CornerConfig(image_size=256, rotate=4.0, scale=0.08, perspective=0.03)
    aug = ChessReDCorners.from_split(chessred, "val", config=cfg, train=True)
    assert any(not torch.equal(aug[0][0], plain) for _ in range(20))


def test_normalize_is_applied_and_roundtrips(tmp_path):
    from torch import nn

    from chessvision.corner_regressor import (
        build_corner_regressor,
        load_corner_regressor,
        save_corner_checkpoint,
    )

    # Stub backbone+head with identity so heatmaps() exposes exactly the input transform
    # (a random untrained net would squash the difference below tolerance).
    x = torch.rand(1, 3, 8, 8)
    norm = build_corner_regressor(pretrained=False, normalize=True)
    norm.features, norm.head = nn.Identity(), nn.Identity()
    assert torch.allclose(norm.heatmaps(x), (x - norm.norm_mean) / norm.norm_std)

    plain = build_corner_regressor(pretrained=False, normalize=False)
    plain.features, plain.head = nn.Identity(), nn.Identity()
    assert torch.allclose(plain.heatmaps(x), x)  # disabled -> passthrough

    # mean/std are constants, not weights: persistent=False keeps them out of state_dict
    assert "norm_mean" not in plain.state_dict() and "norm_std" not in plain.state_dict()

    # the normalize flag round-trips through the checkpoint
    ckpt = tmp_path / "m.pt"
    save_corner_checkpoint(build_corner_regressor(pretrained=False, normalize=True), ckpt)
    assert load_corner_regressor(ckpt).normalize is True
    save_corner_checkpoint(build_corner_regressor(pretrained=False, normalize=False), ckpt)
    assert load_corner_regressor(ckpt).normalize is False


def test_corner_order_constant():
    assert CORNER_ORDER == ("top_left", "top_right", "bottom_right", "bottom_left")


CAPTURES_EXPORT = Path("data/captures/label-studio.json")


@pytest.mark.skipif(not CAPTURES_EXPORT.exists(), reason="capture export not present")
def test_capture_pose_selection_dedups_and_holds_out():
    import numpy as np

    from chessvision.data.corners import (
        _corner_dist,
        _norm_corners,
        select_capture_corner_poses,
    )

    train, heldout = select_capture_corner_poses(CAPTURES_EXPORT, dedup_thr=0.02, max_per_pose=2)
    assert train and heldout

    # The split is by corner *pose*, not by session: the unit is the orientation
    # cluster, so a multi-pose session may appear on both sides (that's fine — the
    # guarantee below is geometric, not per-session).
    # Anti-leak: no train pose sits within the dedup threshold of any held-out pose.
    ho_corners = [_norm_corners(s) for s in heldout]
    for s in train:
        c = _norm_corners(s)
        assert all(_corner_dist(c, h) > 0.02 for h in ho_corners)

    # Per-board coverage: every board that contributes a held-out pose also has train
    # poses (no board is eval-only), so each board is both learned and measured.
    from chessvision.data.session_meta import SessionMetadata

    meta = SessionMetadata.load(CAPTURES_EXPORT.parent)
    board_of = lambda s: (meta.info(s.session) or {}).get("board") if meta else None  # noqa: E731
    assert {board_of(s) for s in heldout} <= {board_of(s) for s in train}

    # max_per_pose is respected: clustering train corners at the threshold yields no
    # cluster larger than the cap (frames are deduped to distinct poses).
    cs = [_norm_corners(s) for s in train]
    counts = np.zeros(len(cs), dtype=int)
    for i, ci in enumerate(cs):
        counts[i] = sum(_corner_dist(ci, cj) <= 0.02 for cj in cs)
    assert int(counts.max()) <= 2


def test_model_forward_and_predict_shapes():
    from chessvision.corner_regressor import (
        build_corner_regressor,
        predict_corners,
        soft_argmax2d,
    )

    # soft-argmax of a single hot cell recovers its normalized location
    heat = torch.full((1, 1, 9, 9), -1e4)
    heat[0, 0, 0, 8] = 1e4  # row 0 (y=0), col 8 (x=1)
    xy = soft_argmax2d(heat)[0, 0]
    assert pytest.approx(float(xy[0]), abs=1e-3) == 1.0  # x
    assert pytest.approx(float(xy[1]), abs=1e-3) == 0.0  # y

    model = build_corner_regressor(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128))
    assert out.shape == (2, 4, 2)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0  # bounded by soft-argmax grid

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = predict_corners(model, rgb, image_size=128)
    assert set(corners) == set(CORNER_ORDER)
    # predictions scaled back to native pixels
    assert all(0.0 <= x <= 640 and 0.0 <= y <= 480 for x, y in corners.values())
