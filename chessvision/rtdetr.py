"""RT-DETRv2 piece detector (HuggingFace `transformers`) -- an Apache-2.0-licensed,
ONNX-friendly alternative to the Faster R-CNN (`chessvision.detector`) and Ultralytics
YOLO (`scripts/train_yolo_detector.py`) box baselines.

RT-DETRv2 is a real-time DEtection TRansformer: an end-to-end detector with no anchors and
no NMS, so a single forward pass yields the boxes. It does the *same* job as the other two
detectors -- 12 piece-class boxes on the natural, un-warped photo (Approach A) -- and is
trained on the *same* official chessred2k split, so its COCO mAP is directly comparable
(`scripts/eval_rtdetr_vs_detector.py`). A contact-keypoint head grafted onto the object
queries is the natural Phase-3 follow-up (mirroring the FRCNN -> Keypoint R-CNN graft); this
module is box-only for now.

Everything `transformers` is imported lazily and lives behind the optional `rtdetr`
dependency group, so the core package imports and builds without it:

    uv sync --group rtdetr

Class-id convention: RT-DETR (like YOLO, unlike FRCNN) uses **0-indexed contiguous** class
ids 0..11 with no background slot -- DETR carries its own internal no-object class. The
`+1`/`-1` shift against the FRCNN scheme (1..12, background=0) lives in `RTDetrCollate` and
the eval scripts so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessvision.data.detection import LABEL_NAMES, NUM_PIECE_CLASSES

# Smallest pretrained RT-DETRv2 (ResNet-18 backbone): the best accuracy/size point for a
# browser-deployable detector. Swap to rtdetr_v2_r34vd / r50vd / r101vd for more capacity.
DEFAULT_CHECKPOINT = "PekingU/rtdetr_v2_r18vd"
# RT-DETRv2 is trained for 640x640; the docs warn other sizes degrade accuracy (the anchors
# and position embeddings are tuned for it), so this is the default and the main caveat vs
# YOLO@1280 -- far-rank pieces get fewer pixels here.
DEFAULT_IMAGE_SIZE = 640

# RT-DETR label maps (0..11 -> piece name). Built from the shared FRCNN names by dropping the
# background slot, so the two detectors stay name-consistent and only differ by the id offset.
ID2LABEL: dict[int, str] = {i: LABEL_NAMES[i + 1] for i in range(NUM_PIECE_CLASSES)}
LABEL2ID: dict[str, int] = {name: i for i, name in ID2LABEL.items()}


def build_processor(checkpoint: str = DEFAULT_CHECKPOINT, image_size: int = DEFAULT_IMAGE_SIZE):
    """Load the RT-DETR image processor (resize to `image_size`, rescale, pad, and
    COCO-annotation -> normalized-cxcywh label conversion). Going through the processor is how
    we inherit RT-DETR's exact preprocessing (notably: rescale to [0,1] with *no* ImageNet
    mean/std normalization) -- don't hand-roll it."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    processor.size = {"height": image_size, "width": image_size}
    return processor


def build_rtdetr(checkpoint: str = DEFAULT_CHECKPOINT, pretrained: bool = True):
    """RT-DETRv2 with the classification heads resized to our 12 piece classes.

    `pretrained=True` loads the COCO-pretrained backbone+encoder+decoder and reinitializes
    only the class heads for 12 labels (`ignore_mismatched_sizes` -- the box/keypoint heads
    are class-agnostic and transfer as-is). `pretrained=False` builds random weights from the
    checkpoint's config (for tests / from-scratch ablations)."""
    from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection

    # Pass id2label/label2id only (not num_labels): transformers derives num_labels from the
    # map, and passing both warns about a mismatch against the checkpoint's 80-class COCO map.
    if pretrained:
        return RTDetrV2ForObjectDetection.from_pretrained(
            checkpoint,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    config = RTDetrV2Config.from_pretrained(checkpoint, id2label=ID2LABEL, label2id=LABEL2ID)
    return RTDetrV2ForObjectDetection(config)


def save_rtdetr(model, processor, path: str | Path) -> None:
    """Save model + processor in HF-native layout (a directory: config.json + weights +
    preprocessor_config.json). RT-DETR doesn't fit the repo's single-`.pt` state_dict
    convention -- it needs its config -- so we use `save_pretrained`; `load_rtdetr` mirrors it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)


def load_rtdetr(path: str | Path, device: str | torch.device = "cpu"):
    """Rebuild model + processor from a `save_rtdetr` directory; returns `(model, processor)`
    with the model on `device` in eval mode."""
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    model = RTDetrV2ForObjectDetection.from_pretrained(path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(path)
    return model, processor


def _to_uint8_hwc(image: torch.Tensor):
    """A shared-dataset image tensor (CxHxW float in [0,1]) -> HxWxC uint8 numpy, the form the
    HF processor expects (it does its own rescale, so we must hand it 0..255, not 0..1)."""
    arr = image.mul(255).round().clamp(0, 255).to(torch.uint8)
    return arr.permute(1, 2, 0).cpu().numpy()


class RTDetrCollate:
    """Collate `ChessReDDetection` items into RT-DETR model inputs.

    Reuses the shared detection dataset unchanged -- each item is `(image[0,1] CxHxW,
    target{boxes xyxy-abs, labels 1..12})` -- and adapts it: image back to uint8, boxes to
    COCO `xywh`, label `-1` to 0-index, then the HF processor produces `pixel_values`,
    `pixel_mask`, and normalized-cxcywh `labels`. Implemented as a class (not a closure) so it
    pickles to DataLoader workers under Windows `spawn`.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, annotations = [], []
        for image, target in batch:
            images.append(_to_uint8_hwc(image))
            image_id = int(target["image_id"][0])
            anns = []
            for (x0, y0, x1, y1), label in zip(
                target["boxes"].tolist(), target["labels"].tolist(), strict=True
            ):
                w, h = x1 - x0, y1 - y0
                anns.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label) - 1,  # FRCNN 1..12 -> RT-DETR 0..11
                        "bbox": [x0, y0, w, h],  # COCO xywh
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
            annotations.append({"image_id": image_id, "annotations": anns})
        return self.processor(images=images, annotations=annotations, return_tensors="pt")


@torch.no_grad()
def evaluate_map(
    model, processor, dataset, device, batch_size: int = 4, workers: int = 0
) -> dict[str, float]:
    """COCO mAP (torchmetrics) over any `(image, target)` detection dataset -- ChessReD *or*
    the capture store, since both emit the same target shape. Scored in the dataset frame: the
    processor resizes to 640, `post_process_object_detection` maps boxes back to each image's
    (H, W). GT labels (1..12) are shifted to RT-DETR's 0..11 to share the scheme. `threshold=0.0`
    keeps all queries for a full PR curve. Returns `{}` if torchmetrics is missing or empty."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ModuleNotFoundError:
        return {}
    if len(dataset) == 0:
        return {}

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    # collate_fn=list keeps raw (image, target) pairs so we can track each image's size.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=list
    )
    for batch in loader:
        images, sizes, gts = [], [], []
        for image, target in batch:
            images.append(_to_uint8_hwc(image))
            sizes.append((image.shape[1], image.shape[2]))  # (H, W)
            gts.append({"boxes": target["boxes"], "labels": target["labels"] - 1})
        enc = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**enc)
        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor(sizes), threshold=0.0
        )
        preds = [
            {"boxes": r["boxes"].cpu(), "scores": r["scores"].cpu(), "labels": r["labels"].cpu()}
            for r in results
        ]
        metric.update(preds, gts)
    return {k: float(v) for k, v in metric.compute().items() if v.numel() == 1}
