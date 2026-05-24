"""Transplant a 1-keypoint "board-contact" head onto the trained box detector.

We keep the trained Faster R-CNN trunk (backbone + FPN + RPN + box head) from
`runs/detector/best.pt` and *add* a keypoint branch to `roi_heads` that predicts a
single keypoint per piece: its **board-contact point** (the base), which is what
square assignment needs (plan.md section 4). Only the new head is trained; the
trunk is frozen, so we never retrain or modify the detector checkpoint.

The keypoint head is the standard torchvision `KeypointRCNNHeads` /
`KeypointRCNNPredictor` with `num_keypoints=1`. It depends only on the FPN's
`out_channels=256` and featmap names `['0','1','2','3']`, both present on the v2
trunk, so it grafts cleanly onto `fasterrcnn_resnet50_fpn_v2`.

Note: torchvision's keypoint loss only supervises keypoints that fall *inside*
the proposal box, so a base-occluded piece whose true base is below its visible
box can't be learned/predicted here (the known v1 limitation — measured, not yet
mitigated).
"""

from __future__ import annotations

from pathlib import Path

import torch

from chessvision.data.detection import NUM_CLASSES
from chessvision.detector import build_detector

NUM_KEYPOINTS = 1  # the single board-contact point per piece


def build_keypoint_detector(
    num_classes: int = NUM_CLASSES, num_keypoints: int = NUM_KEYPOINTS
) -> torch.nn.Module:
    """A Faster R-CNN (v2) trunk with a keypoint branch grafted onto `roi_heads`.

    `pretrained=False` because weights come from the trained detector checkpoint
    (see `graft_from_detector_checkpoint`); only the keypoint branch is new.
    """
    from torchvision.models.detection.keypoint_rcnn import (
        KeypointRCNNHeads,
        KeypointRCNNPredictor,
    )
    from torchvision.ops import MultiScaleRoIAlign

    model = build_detector(num_classes=num_classes, pretrained=False)
    out_channels = model.backbone.out_channels  # 256 for ResNet50-FPN
    model.roi_heads.keypoint_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
    )
    model.roi_heads.keypoint_head = KeypointRCNNHeads(out_channels, tuple(512 for _ in range(8)))
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints)
    return model


def _is_keypoint_param(name: str) -> bool:
    return name.startswith("roi_heads.keypoint_")


def graft_from_detector_checkpoint(
    detector_ckpt: str | Path,
    device: str | torch.device = "cpu",
    num_keypoints: int = NUM_KEYPOINTS,
) -> torch.nn.Module:
    """Build the keypoint detector and load the trained trunk from a box-detector
    checkpoint. The checkpoint has no keypoint params, so `strict=False`; we assert
    the only missing keys are the new keypoint branch and nothing is unexpected.

    The detector checkpoint is opened **read-only** — it is never modified.
    """
    ckpt = torch.load(detector_ckpt, map_location=device, weights_only=True)
    model = build_keypoint_detector(ckpt.get("num_classes", NUM_CLASSES), num_keypoints)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not unexpected, f"unexpected keys loading trunk: {unexpected[:5]}"
    bad = [k for k in missing if not _is_keypoint_param(k)]
    assert not bad, f"missing non-keypoint trunk weights: {bad[:5]}"
    return model.to(device)


def freeze_trunk(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze everything except the keypoint head + predictor, so training only
    updates the new branch (the trunk's detection ability is preserved exactly)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.roi_heads.keypoint_head, model.roi_heads.keypoint_predictor):
        for p in module.parameters():
            p.requires_grad_(True)
    return model


def freeze_for_finetune(model: torch.nn.Module) -> torch.nn.Module:
    """Domain fine-tune freezing: train the box **classifier/regressor head** AND the
    keypoint branch, keep the backbone + FPN + RPN frozen. The capture set's real gap
    is piece *classification* on the user's own pieces (glass/wood), not localization
    (the trunk already localizes well and the contact geometry is exact), so we give
    the class head the capacity to adapt while the tiny low-diversity set can't corrupt
    the shared trunk. Mirrors `freeze_trunk` but also unfreezes `roi_heads.box_predictor`.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (
        model.roi_heads.box_predictor,
        model.roi_heads.keypoint_head,
        model.roi_heads.keypoint_predictor,
    ):
        for p in module.parameters():
            p.requires_grad_(True)
    return model


def keypoint_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """The trainable parameters (whatever the active freeze scheme leaves unfrozen)."""
    return [p for p in model.parameters() if p.requires_grad]


# Alias: reads better at fine-tune call sites where >1 head is trainable.
trainable_parameters = keypoint_parameters


def save_keypoint_checkpoint(model: torch.nn.Module, path: str | Path, **extra) -> None:
    from chessvision.data.detection import LABEL_NAMES

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "num_keypoints": NUM_KEYPOINTS,
            "label_names": LABEL_NAMES,
            **extra,
        },
        path,
    )


def load_keypoint_detector(path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Rebuild + load a trained keypoint detector (weights_only, like the detector)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_keypoint_detector(
        ckpt.get("num_classes", NUM_CLASSES), ckpt.get("num_keypoints", NUM_KEYPOINTS)
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
