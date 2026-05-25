"""Board-corner localizer: a compact CNN with a sub-pixel heatmap head (Phase 3).

The Phase-3 board localizer (plan.md): a small network predicts the 4 board
corners, replacing the manually-clicked corners that feed the Phase-1 homography.
This is a **separate, compact model** (not grafted on the detector trunk) so it
exports cleanly to mobile and adds little weight -- finding 4 corners on a board
that fills the frame is a low-frequency, global task, so a MobileNetV3 / ResNet18
backbone is ample (the heavy Faster R-CNN is overkill here). The default is
MobileNetV3-small (~2.5M params -> ~10 MB fp32, ~2.5 MB int8); the big 173 MB
detector is a separate model and a separate (Phase-6) size-optimization story.

**Head = soft-argmax (integral regression).** The head emits one low-res heatmap
per corner; a spatial softmax turns each into a probability map and we take its
**expected (x, y)** as the prediction. Properties:
  - sub-pixel: the expectation is continuous, not snapped to a heatmap cell;
  - supervises coordinates directly (Smooth-L1 on the (4, 2) target), so the
    dataset stays plain normalized corners -- no Gaussian-target rendering;
  - outputs are bounded to the grid range (~[0, 1]), matching in-frame corners.

Corners are predicted in **visual TL/TR/BR/BL slots**, normalized to [0, 1] (see
`chessvision.data.corners`): pure quad localization, with board orientation left
to the deliberate manual toggle downstream.

Checkpoints follow `detector.py`: plain `state_dict` + metadata, loaded with
`weights_only=True` (the default flipped across torch versions -- plan.md section 8).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from chessvision.data.corners import CORNER_ORDER, NUM_CORNERS

DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_IMAGE_SIZE = 384


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """A truncated torchvision classifier as a conv feature extractor.

    Returns `(features, out_channels)`; out_channels is probed with a dummy
    forward so we don't hardcode per-variant channel counts.
    """
    import torchvision.models as tvm

    weights = "DEFAULT" if pretrained else None
    if name.startswith("mobilenet"):
        features = getattr(tvm, name)(weights=weights).features
    elif name.startswith("resnet"):
        net = getattr(tvm, name)(weights=weights)
        features = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + fc
    else:
        raise ValueError(f"unsupported backbone {name!r} (use mobilenet_v3_* or resnet*)")
    with torch.no_grad():
        out_channels = features(torch.zeros(1, 3, 64, 64)).shape[1]
    return features, out_channels


def soft_argmax2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Expected (x, y) of each heatmap after a spatial softmax.

    `heatmaps`: (B, K, H, W). Returns (B, K, 2) in [0, 1] (normalized grid coords).
    The softmax is over the full HxW plane, then x = E[col/(W-1)], y = E[row/(H-1)].
    """
    b, k, h, w = heatmaps.shape
    prob = heatmaps.reshape(b, k, h * w).softmax(dim=-1).reshape(b, k, h, w)
    device, dtype = heatmaps.device, heatmaps.dtype
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    x = (prob.sum(dim=2) * xs).sum(dim=-1)  # marginal over rows -> weight columns
    y = (prob.sum(dim=3) * ys).sum(dim=-1)  # marginal over cols -> weight rows
    return torch.stack([x, y], dim=-1)


class CornerHeatmapNet(nn.Module):
    """Compact backbone -> upsampled per-corner heatmaps -> soft-argmax coordinates."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        num_corners: int = NUM_CORNERS,
        normalize: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_corners = num_corners
        self.normalize = normalize
        self.features, c = _build_backbone(backbone, pretrained)
        # ImageNet normalization baked in as the first op (when enabled) so it travels
        # with the exported graph -- this model exists to export to mobile, so keeping
        # the mean/std in Python preprocessing would force every caller (train, predict,
        # live, the native runtime) to re-apply it identically. Buffers are constants, so
        # persistent=False keeps them out of the state_dict (no key churn; old checkpoints
        # still load). Input contract: float CHW in [0, 1]. The torchvision backbone was
        # pretrained with exactly these statistics.
        self.register_buffer(
            "norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        # Two bilinear-upsample stages lift the stride-32 feature map (12x12 at 384)
        # to ~48x48 heatmaps -- comfortable resolution for a sub-pixel expectation.
        self.head = nn.Sequential(
            nn.Conv2d(c, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, num_corners, 1),
        )

    def heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = (x - self.norm_mean) / self.norm_std
        return self.head(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) -> (B, num_corners, 2) normalized corner coords."""
        return soft_argmax2d(self.heatmaps(x))


def build_corner_regressor(
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
    num_corners: int = NUM_CORNERS,
    normalize: bool = True,
) -> CornerHeatmapNet:
    return CornerHeatmapNet(
        backbone=backbone, pretrained=pretrained, num_corners=num_corners, normalize=normalize
    )


def save_corner_checkpoint(
    model: CornerHeatmapNet,
    path: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    **extra,
) -> None:
    """Save a state_dict checkpoint plus the metadata `load_corner_regressor` needs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": model.backbone_name,
            "image_size": image_size,
            "num_corners": model.num_corners,
            "normalize": model.normalize,
            "corner_order": list(CORNER_ORDER),
            **extra,
        },
        path,
    )


def load_corner_regressor(path: str | Path, device: str | torch.device = "cpu") -> CornerHeatmapNet:
    """Rebuild + load a trained corner localizer (`weights_only=True`, like the detector).

    The training `image_size` is stashed on the module as `.image_size` so
    `predict_corners` resizes inputs to match without a separate arg.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Default normalize=False for checkpoints predating this flag: those weights were
    # trained on un-normalized [0, 1] input, so the model must reproduce that at inference.
    model = build_corner_regressor(
        backbone=ckpt.get("backbone", DEFAULT_BACKBONE),
        pretrained=False,
        num_corners=ckpt.get("num_corners", NUM_CORNERS),
        normalize=ckpt.get("normalize", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.image_size = int(ckpt.get("image_size", DEFAULT_IMAGE_SIZE))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_corners(
    model: CornerHeatmapNet,
    rgb: np.ndarray,
    device: str | torch.device = "cpu",
    image_size: int | None = None,
) -> dict[str, list[float]]:
    """Predict board corners for one RGB image, in native pixel coordinates.

    Returns a snake_case dict (`top_left`/`top_right`/`bottom_right`/`bottom_left`)
    ready for `geometry.compute_homography` / `order_corners`. The network sees a
    fixed square; predictions are normalized [0, 1] and scaled by the image's own
    width/height, so this is correct for non-square images too.
    """
    size = image_size or getattr(model, "image_size", DEFAULT_IMAGE_SIZE)
    h0, w0 = rgb.shape[:2]
    inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1).float().div_(255.0)
    pred = model(tensor.unsqueeze(0).to(device))[0].cpu().numpy()  # (num_corners, 2) in [0, 1]
    pred = pred * np.array([w0, h0], dtype=np.float32)
    return {k: [float(x), float(y)] for k, (x, y) in zip(CORNER_ORDER, pred, strict=True)}
