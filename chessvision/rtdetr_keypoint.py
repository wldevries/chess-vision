"""RT-DETRv2 with a contact-keypoint head (Approach A, the deployment path).

Box detectors find pieces; for FEN we need each piece's **board-contact point** (the base),
mapped through the homography to a square -- never a box bottom (see the contact-point
anti-pattern). This module grafts a per-query point head onto `RTDetrV2ForObjectDetection`:
each object query already owns one piece, so alongside its class+box it regresses one normalized
(x, y) contact point. The keypoint loss reuses the model's **own** `RTDetrHungarianMatcher`, so
the point is supervised on exactly the query the box loss matched to that piece -- no second,
divergent assignment.

Requires the optional `rtdetr` dependency group (transformers); this whole module imports it.

Training labels come straight from the keypoint datasets (`ChessReDKeypointDetection`,
`CaptureKeypointDetection`) which already emit boxes (1..12) + a contact keypoint per piece.
`RTDetrKeypointCollate` builds the HF label dicts (normalized cxcywh boxes + normalized points)
directly, so there's no processor-annotation alignment to babysit.

Eval reuses the existing `capture_eval`/`keypoint_eval` machinery via `TorchvisionDetAdapter`,
which makes the HF model quack like a torchvision detector (`model([img])[0] -> dict`) -- so the
contact-point class_acc is directly comparable to the Keypoint R-CNN numbers.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
from transformers.loss.loss_rt_detr import RTDetrForObjectDetectionLoss, RTDetrHungarianMatcher

from chessvision.rtdetr import DEFAULT_CHECKPOINT, ID2LABEL, LABEL2ID, _to_uint8_hwc

DEFAULT_KEYPOINT_LOSS_COEF = 5.0  # weight on the point L1 (≈ the box L1 weight, normalized coords)


class RTDetrV2Keypoint(RTDetrV2ForObjectDetection):
    """RT-DETRv2 + a contact-point head. `forward` returns the standard detection output with an
    extra `pred_points` (B, num_queries, 2) in normalized [0, 1] coords, and -- when `labels`
    carry a `points` key -- adds an L1 keypoint loss (on the matcher's assignment) to `loss`."""

    def __init__(self, config: RTDetrV2Config):
        super().__init__(config)
        d = config.d_model
        # Small MLP per query -> (x, y); sigmoid in forward keeps points in the image box.
        self.keypoint_embed = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2)
        )
        self.keypoint_matcher = RTDetrHungarianMatcher(config)
        self.keypoint_loss_coef = DEFAULT_KEYPOINT_LOSS_COEF
        # transformers infers the loss from the class-name suffix; our rename breaks that and it
        # falls back to the causal-LM loss, so pin the RT-DETR detection loss explicitly.
        self.loss_function = RTDetrForObjectDetectionLoss

    def _keypoint_loss(self, logits, pred_boxes, pred_points, labels) -> torch.Tensor:
        """L1 between predicted and GT contact points on matched (query, gt) pairs, normalized by
        the number of matched points. Uses the model's own matcher on (logits, pred_boxes) so the
        supervised query is the one the box loss also matched."""
        indices = self.keypoint_matcher({"logits": logits, "pred_boxes": pred_boxes}, labels)
        total = pred_points.new_zeros(())
        num = 0
        for b, (query_idx, gt_idx) in enumerate(indices):
            if query_idx.numel() == 0:
                continue
            gt_points = labels[b]["points"].to(pred_points.device)
            total = total + F.l1_loss(pred_points[b][query_idx], gt_points[gt_idx], reduction="sum")
            num += int(query_idx.numel())
        return total / max(num, 1)

    def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, **kwargs
        )
        pred_points = self.keypoint_embed(outputs.last_hidden_state).sigmoid()
        if labels is not None and all("points" in t for t in labels):
            kp_loss = self._keypoint_loss(outputs.logits, outputs.pred_boxes, pred_points, labels)
            outputs.loss = outputs.loss + self.keypoint_loss_coef * kp_loss
            if getattr(outputs, "loss_dict", None) is not None:
                outputs.loss_dict["loss_keypoint"] = kp_loss
        outputs["pred_points"] = pred_points
        return outputs


def build_rtdetr_keypoint(
    checkpoint: str = DEFAULT_CHECKPOINT,
    pretrained: bool = True,
    keypoint_loss_coef: float = DEFAULT_KEYPOINT_LOSS_COEF,
) -> RTDetrV2Keypoint:
    """RT-DETRv2+keypoint with class heads resized to 12 pieces. `pretrained` loads the COCO
    trunk; the class heads and the fresh keypoint head are (re)initialized."""
    if pretrained:
        model = RTDetrV2Keypoint.from_pretrained(
            checkpoint, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
        )
    else:
        config = RTDetrV2Config.from_pretrained(checkpoint, id2label=ID2LABEL, label2id=LABEL2ID)
        model = RTDetrV2Keypoint(config)
    model.keypoint_loss_coef = keypoint_loss_coef
    return model


def load_rtdetr_keypoint(path: str | Path, device: str | torch.device = "cpu"):
    """Rebuild the keypoint model + processor from a `save_rtdetr` directory."""
    from transformers import AutoImageProcessor

    model = RTDetrV2Keypoint.from_pretrained(path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(path)
    return model, processor


class RTDetrKeypointCollate:
    """Collate keypoint-dataset items into RT-DETR inputs + labels. Builds the HF label dicts
    ourselves (normalized cxcywh boxes + normalized contact points) instead of routing boxes
    through the processor's COCO path -- that keeps boxes and points perfectly aligned and lets
    the point label travel with its box. A class (not a closure) so it pickles to workers."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, labels = [], []
        for image, target in batch:
            _, h, w = image.shape
            images.append(_to_uint8_hwc(image))
            boxes = target["boxes"].float()  # xyxy abs
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
            bw = (boxes[:, 2] - boxes[:, 0]) / w
            bh = (boxes[:, 3] - boxes[:, 1]) / h
            pts = target["keypoints"][:, 0, :2].float()  # (N, 2) abs contact points
            labels.append(
                {
                    "class_labels": (target["labels"] - 1).long(),  # 1..12 -> 0..11
                    "boxes": torch.stack([cx, cy, bw, bh], dim=1),
                    "points": torch.stack([pts[:, 0] / w, pts[:, 1] / h], dim=1),
                }
            )
        enc = self.processor(images=images, return_tensors="pt")
        enc["labels"] = labels
        return enc


class TorchvisionDetAdapter:
    """Make an `RTDetrV2Keypoint` quack like a torchvision detector so `capture_eval` /
    `keypoint_eval` work unchanged: `adapter([image_chw_float]) -> [{scores, labels, boxes,
    keypoints}]` in the input image's pixel frame, labels shifted 0..11 -> 1..12 to match the
    GT scheme, keypoints shaped (Q, 1, 2). Returns *all* queries (no threshold) -- the eval
    applies its own `score_thresh`."""

    def __init__(self, model: RTDetrV2Keypoint, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def __call__(self, images):
        results = []
        for image in images:
            _, h, w = image.shape
            enc = self.processor(images=[_to_uint8_hwc(image)], return_tensors="pt").to(self.device)
            out = self.model(**enc)
            scores, labels = out.logits[0].sigmoid().max(-1)  # focal: per-query max class prob
            cx, cy, bw, bh = out.pred_boxes[0].unbind(-1)
            boxes = torch.stack(
                [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dim=1
            )
            scale = torch.tensor([w, h], device=self.device)
            points = (out["pred_points"][0] * scale).unsqueeze(1)  # (Q, 1, 2)
            results.append(
                {
                    "scores": scores.cpu(),
                    "labels": (labels + 1).cpu(),  # 0..11 -> 1..12
                    "boxes": boxes.cpu(),
                    "keypoints": points.cpu(),
                }
            )
        return results
