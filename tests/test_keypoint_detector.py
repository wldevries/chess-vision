"""Keypoint-head transplant checks (CPU, no checkpoint/GPU needed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chessvision.detector import build_detector  # noqa: E402
from chessvision.keypoint_detector import (  # noqa: E402
    _is_keypoint_param,
    build_keypoint_detector,
    freeze_trunk,
    keypoint_parameters,
)


def test_graft_adds_keypoint_branch():
    m = build_keypoint_detector()
    assert m.roi_heads.keypoint_roi_pool is not None
    assert m.roi_heads.keypoint_head is not None
    assert m.roi_heads.keypoint_predictor is not None
    assert m.roi_heads.has_keypoint()


def test_box_checkpoint_grafts_with_strict_false():
    """A box-only state_dict loads into the grafted model; only keypoint keys missing."""
    box = build_detector(pretrained=False)
    kp = build_keypoint_detector()
    missing, unexpected = kp.load_state_dict(box.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all(_is_keypoint_param(k) for k in missing)


def test_freeze_trunk_trains_only_keypoint_branch():
    m = build_keypoint_detector()
    freeze_trunk(m)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable
    assert all(_is_keypoint_param(n) for n in trainable)
    assert len(keypoint_parameters(m)) == len(trainable)


def test_train_forward_has_keypoint_loss():
    m = build_keypoint_detector()
    m.train()
    images = [torch.rand(3, 256, 320)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 60.0, 90.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor([[[35.0, 80.0, 2.0]]]),  # (N,1,3) base inside box
        }
    ]
    losses = m(images, targets)
    assert "loss_keypoint" in losses


def test_eval_forward_returns_keypoints():
    m = build_keypoint_detector()
    m.eval()
    with torch.no_grad():
        out = m([torch.rand(3, 256, 256)])[0]
    assert "keypoints" in out
    kp = out["keypoints"]
    assert kp.ndim == 3 and kp.shape[1] == 1 and kp.shape[2] == 3
