"""Adapt an Ultralytics YOLO-pose model to the torchvision keypoint-detector interface.

The capture eval (`chessvision.capture_eval`) and `eval_end_to_end_captures` drive a
detector purely through `model([image_tensor]) -> [{"scores","labels","keypoints"}]`,
mapping `keypoints[:, 0]` (the contact point) through the homography to a square. This
wrapper presents a YOLO-pose model behind that exact interface, so a YOLO-pose model is
scored by the SAME metric path as the Keypoint R-CNN -- no metric-code fork, no drift.

Two conventions are reconciled here:
  - class ids: YOLO emits 0..11; the eval/truth use 1..12 (LABEL_NAMES), so we shift +1.
  - channels: Ultralytics' numpy `predict` source is BGR; the eval hands us an RGB [0,1]
    tensor, so we convert RGB->uint8->BGR. Keypoints come back in the input frame's pixels
    (the frame the eval built its homography in), so coordinates line up with no rescaling.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from chessvision.data.detection import NUM_PIECE_CLASSES


class YoloKeypointDetector:
    """Callable that mimics a torchvision keypoint detector around a YOLO-pose model."""

    def __init__(
        self,
        yolo,
        *,
        imgsz: int = 1280,
        conf: float = 0.05,
        iou: float = 0.7,
        max_det: int = 300,
        device: str = "cpu",
    ):
        # Low `conf` floor so the eval's own `score_thresh` does the real filtering
        # (keeps the threshold knob in one place, comparable to the R-CNN path).
        self.yolo = yolo
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device

    def eval(self):  # torchvision models expose .eval(); evaluate_captures calls it
        return self

    def to(self, _device):  # no-op: device is handled by YOLO.predict
        return self

    @torch.no_grad()
    def __call__(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        outputs = []
        for img in images:
            arr_rgb = (
                (img.detach().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            )
            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)  # Ultralytics np source is BGR
            res = self.yolo.predict(
                source=arr_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
            )[0]

            n = len(res.boxes)
            scores = res.boxes.conf.detach().cpu()
            labels = res.boxes.cls.detach().cpu().to(torch.int64) + 1  # 0..11 -> 1..12
            if res.keypoints is not None and n:
                xy = res.keypoints.xy.detach().cpu().float()  # (N, 1, 2), input-frame px
                kconf = res.keypoints.conf
                kconf = (
                    kconf.detach().cpu().float().unsqueeze(-1)
                    if kconf is not None
                    else torch.ones(n, xy.shape[1], 1)
                )
                keypoints = torch.cat([xy, kconf], dim=-1)  # (N, 1, 3)
            else:
                keypoints = torch.zeros((0, 1, 3))
            outputs.append({"scores": scores, "labels": labels, "keypoints": keypoints})
        return outputs


def load_yolo_keypoint_detector(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    imgsz: int = 1280,
    conf: float = 0.05,
    iou: float = 0.7,
    max_det: int = 300,
) -> YoloKeypointDetector:
    """Load a trained YOLO-pose checkpoint behind the keypoint-detector interface."""
    from ultralytics import YOLO

    model = YOLO(str(path))
    # Sanity: a pose model trained on our 12 classes, single contact keypoint.
    names = getattr(model, "names", {}) or {}
    if names and len(names) != NUM_PIECE_CLASSES:
        raise ValueError(f"expected {NUM_PIECE_CLASSES} classes, checkpoint has {len(names)}")
    return YoloKeypointDetector(
        model, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=str(device)
    )
