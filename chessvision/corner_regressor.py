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

from chessvision.data.corners import CORNER_ORDER, LATTICE_CANONICAL, NUM_CORNERS
from chessvision.geometry import CANONICAL_ANCHORS

DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_IMAGE_SIZE = 512  # 384->512 tightened cell-disp on the weak boards (rimless worst-disp 0.909->0.685)


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


# --------------------------------------------------------------------------- #
# Lattice (81-point) inference: predict the 9x9 grid, fit H robustly over all
# points (confidence-weighted), derive 4 corners for drop-in compatibility.
# --------------------------------------------------------------------------- #


def _softargmax_with_conf(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, K, H, W) -> (coords (B, K, 2) in [0, 1], conf (B, K)).

    conf = 1 / spatial variance of the softmax map: a sharp, peaked heatmap is a
    confident localization; a diffuse one (an occluded interior point, the warped
    board middle) gets low confidence. No extra supervision -- it falls out of the
    same softmax the soft-argmax already computes.
    """
    b, k, h, w = heatmaps.shape
    prob = heatmaps.reshape(b, k, h * w).softmax(dim=-1).reshape(b, k, h, w)
    device, dtype = heatmaps.device, heatmaps.dtype
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    px, py = prob.sum(dim=2), prob.sum(dim=3)  # marginals over rows / cols
    ex = (px * xs).sum(dim=-1)
    ey = (py * ys).sum(dim=-1)
    var = (px * xs * xs).sum(dim=-1) - ex**2 + (py * ys * ys).sum(dim=-1) - ey**2
    return torch.stack([ex, ey], dim=-1), 1.0 / (var + 1e-6)


def predict_lattice(
    model: CornerHeatmapNet,
    rgb: np.ndarray,
    device: str | torch.device = "cpu",
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the 81 lattice points (native px, (81, 2)) and a per-point confidence (81,)."""
    size = image_size or getattr(model, "image_size", DEFAULT_IMAGE_SIZE)
    h0, w0 = rgb.shape[:2]
    inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1).float().div_(255.0)
    with torch.no_grad():
        coords, conf = _softargmax_with_conf(model.heatmaps(tensor.unsqueeze(0).to(device)))
    pts = coords[0].cpu().numpy() * np.array([w0, h0], dtype=np.float32)
    return pts.astype(np.float32), conf[0].cpu().numpy().astype(np.float32)


def _normalize_pts(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization (centroid to origin, mean distance sqrt(2)) for a stable DLT."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / (d + 1e-12)
    t = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return (pts - c) * s, t


def homography_from_lattice(points_px: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """Fit canonical->image H from the 81 lattice correspondences via a confidence-weighted
    normalized DLT. Spreading the fit over 81 overdetermined points averages out per-point
    noise (vs. a 4-corner fit with zero slack), and the confidence weights down-weight
    uncertain points so one bad (occluded) intersection can't skew the whole grid.
    Pass conf=None for an unweighted fit."""
    src = LATTICE_CANONICAL.astype(np.float64)
    dst = np.asarray(points_px, dtype=np.float64)
    w = np.ones(len(src)) if conf is None else np.sqrt(np.clip(conf, 1e-6, None))
    srcn, t_src = _normalize_pts(src)
    dstn, t_dst = _normalize_pts(dst)
    rows = []
    for (x, y), (u, v), wi in zip(srcn, dstn, w, strict=True):
        rows.append(wi * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u]))
        rows.append(wi * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    return (h / h[2, 2]).astype(np.float32)


def summed_heatmap(
    model: CornerHeatmapNet,
    rgb: np.ndarray,
    device: str | torch.device = "cpu",
    image_size: int | None = None,
) -> np.ndarray:
    """Additive heatmap for the whole lattice, with per-channel peak normalization.

    Each of the model's K keypoint channels is a per-point spatial softmax. The
    well-trained corner channels are *much* sharper than the often-occluded interior
    intersections, so a plain channel-sum is dominated by the 4 corners and the
    other 77 points vanish into a wash. We rescale each channel by its own peak
    first so every point contributes a unit-height blob, then sum and normalize.
    Returned as a (H, W) float32 in [0, 1].
    """
    size = image_size or getattr(model, "image_size", DEFAULT_IMAGE_SIZE)
    inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1).float().div_(255.0)
    with torch.no_grad():
        h = model.heatmaps(tensor.unsqueeze(0).to(device))  # (1, K, H, W)
        _, k, hh, ww = h.shape
        prob = h.reshape(1, k, hh * ww).softmax(dim=-1).reshape(1, k, hh, ww)
        peaks = prob.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        summed = (prob / peaks).sum(dim=1)[0]  # (H, W), each channel contributes max=1
    arr = summed.cpu().numpy().astype(np.float32)
    m = float(arr.max())
    if m > 0:
        arr = arr / m
    return arr


def corners_from_lattice(
    model: CornerHeatmapNet,
    rgb: np.ndarray,
    device: str | torch.device = "cpu",
    use_conf: bool = True,
) -> dict[str, list[float]]:
    """Drop-in replacement for `predict_corners` using the lattice model: predict 81 points,
    fit a robust H, and read off the 4 board corners (project the canonical anchors a8/h8/a1/h1,
    which land in visual TL/TR/BL/BR slots)."""
    pts, conf = predict_lattice(model, rgb, device=device)
    h = homography_from_lattice(pts, conf if use_conf else None)
    anchors = cv2.perspectiveTransform(
        CANONICAL_ANCHORS.reshape(-1, 1, 2).astype(np.float32), h
    ).reshape(-1, 2)
    keys = ("top_left", "top_right", "bottom_left", "bottom_right")
    return {k: [float(x), float(y)] for k, (x, y) in zip(keys, anchors, strict=True)}
