"""Piece detector: a Faster R-CNN (ResNet50-FPN) for Approach A (plan.md section 4).

The detector finds pieces in the natural, un-warped photo. Downstream (Phase 4)
each predicted box's bottom-center is mapped through the board homography to a
square, then to FEN -- so the detector never sees a warped image.

Backbone is `fasterrcnn_resnet50_fpn_v2`, COCO-pretrained; only the box predictor
head is replaced to emit our 12 piece classes + background.
"""

from __future__ import annotations

from pathlib import Path

import torch

from chessvision.data.detection import LABEL_NAMES, NUM_CLASSES


def build_detector(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """Faster R-CNN with the head resized to `num_classes` (includes background).

    `pretrained=True` loads COCO weights for the backbone+FPN+RPN (a big head
    start); the final classification/regression layer is reinitialized for our
    classes.
    """
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_checkpoint(model: torch.nn.Module, path: str | Path, **extra) -> None:
    """Save a state_dict checkpoint plus metadata (label names, class count)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "label_names": LABEL_NAMES,
            **extra,
        },
        path,
    )


def load_detector(path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Rebuild the detector and load weights.

    `weights_only=True` is set explicitly: this checkpoint is a plain state_dict
    plus JSON-able metadata, so the safe loader is correct here, and the default
    flipped across torch versions (plan.md section 8) -- we never rely on it.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_detector(num_classes=ckpt.get("num_classes", NUM_CLASSES), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
