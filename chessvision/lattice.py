"""Lattice board model: the 81-point heatmap net + a parameter-free decode *head*.

The decode head turns the 81 predicted grid intersections into the 4 board corners +
homography + per-point confidence, entirely with arithmetic (soft-argmax, a
confidence-weighted normalized DLT, and an anchor projection) -- no learned weights.

Why a head and not caller-side code: the geometry then travels *inside the model*, so
every consumer (the web app's "Predict", read-position inference, a future mobile export)
calls one module and gets corners out, instead of re-implementing the DLT. It's a single
torch graph, so it also exports cleanly later (Phase 6).

`LatticeCornerNet(heatmap_net)` wraps a trained `CornerHeatmapNet(num_corners=81)`:
    corners, conf, H = net(image)             # corners (B,4,2) in [0,1], H canonical->image
The decode mirrors `corner_regressor.homography_from_lattice` (verified numerically).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from chessvision.corner_regressor import CornerHeatmapNet, _softargmax_with_conf
from chessvision.data.corners import LATTICE_CANONICAL
from chessvision.geometry import CANONICAL_ANCHORS

# visual-slot keys for the 4 corners derived from the canonical anchors (a8,h8,a1,h1).
CORNER_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _normalize_batch(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hartley normalization per sample. pts: (B, N, 2) -> (pts_n (B,N,2), T (B,3,3))."""
    c = pts.mean(dim=1, keepdim=True)  # (B,1,2)
    d = (pts - c).norm(dim=-1).mean(dim=1)  # (B,)
    s = np.sqrt(2.0) / (d + 1e-12)  # (B,)
    pts_n = (pts - c) * s[:, None, None]
    b = pts.shape[0]
    t = torch.zeros(b, 3, 3, dtype=pts.dtype, device=pts.device)
    t[:, 0, 0] = s
    t[:, 1, 1] = s
    t[:, 0, 2] = -s * c[:, 0, 0]
    t[:, 1, 2] = -s * c[:, 0, 1]
    t[:, 2, 2] = 1.0
    return pts_n, t


def _project(h: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply (B,3,3) homographies to (N,2) or (B,N,2) points -> (B,N,2)."""
    if pts.dim() == 2:
        pts = pts.unsqueeze(0).expand(h.shape[0], -1, -1)
    ones = torch.ones(*pts.shape[:-1], 1, dtype=pts.dtype, device=pts.device)
    hom = torch.cat([pts, ones], dim=-1)  # (B,N,3)
    out = hom @ h.transpose(1, 2)  # (B,N,3)
    return out[..., :2] / out[..., 2:3].clamp_min(1e-12)


class LatticeDecode(nn.Module):
    """Parameter-free: (B,81,H,W) heatmaps -> (corners (B,4,2), conf (B,81), H (B,3,3))."""

    def __init__(self, use_conf: bool = True):
        super().__init__()
        self.use_conf = use_conf
        self.register_buffer("canon", torch.from_numpy(LATTICE_CANONICAL.copy()), persistent=False)
        self.register_buffer(
            "anchors", torch.from_numpy(CANONICAL_ANCHORS.copy()), persistent=False
        )

    def fit_homography(self, dst: torch.Tensor, conf: torch.Tensor | None) -> torch.Tensor:
        """dst: (B,81,2) image coords in [0,1]; returns (B,3,3) canonical->image."""
        b, n, _ = dst.shape
        src = self.canon.to(dst.dtype).unsqueeze(0).expand(b, -1, -1)  # (B,81,2)
        src_n, t_src = _normalize_batch(src)
        dst_n, t_dst = _normalize_batch(dst)
        x, y = src_n[..., 0], src_n[..., 1]  # (B,N)
        u, v = dst_n[..., 0], dst_n[..., 1]
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        row1 = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u], dim=-1)
        row2 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v], dim=-1)
        a = torch.cat([row1, row2], dim=1)  # (B, 2N, 9)
        if conf is not None:
            w = conf.clamp_min(1e-6).sqrt()  # (B,N)
            a = a * torch.cat([w, w], dim=1)[..., None]
        _, _, vh = torch.linalg.svd(a)
        h_n = vh[:, -1, :].reshape(b, 3, 3)  # canonical_norm -> dst_norm
        h = torch.linalg.inv(t_dst) @ h_n @ t_src
        return h / h[:, 2:3, 2:3].clamp_min(1e-12)

    def forward(self, heatmaps: torch.Tensor):
        coords, conf = _softargmax_with_conf(heatmaps)  # (B,81,2),(B,81)
        h = self.fit_homography(coords, conf if self.use_conf else None)
        corners = _project(h, self.anchors.to(coords.dtype))  # (B,4,2)
        return corners, conf, h


class LatticeCornerNet(nn.Module):
    """A trained 81-point heatmap net + the decode head. `forward(image) -> corners`,
    `forward_full(image) -> (corners, conf, H)`. Image is float CHW in [0,1] at the net's
    `image_size` (use `predict` for an RGB ndarray at native resolution)."""

    def __init__(self, heatmap_net: CornerHeatmapNet, use_conf: bool = True):
        super().__init__()
        self.net = heatmap_net
        self.decode = LatticeDecode(use_conf=use_conf)
        self.image_size = getattr(heatmap_net, "image_size", 384)

    def forward_full(self, image: torch.Tensor):
        return self.decode(self.net.heatmaps(image))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_full(image)[0]

    @torch.no_grad()
    def predict(self, rgb: np.ndarray, device: str | torch.device = "cpu") -> dict:
        """RGB ndarray (native size) -> {corners: {key:[x,y] px}, homography, point_confidence,
        board_confidence}. Corners/H are in native pixels. board_confidence is the mean
        reprojection residual of the 81 points (lower = better)."""
        import cv2

        h0, w0 = rgb.shape[:2]
        inp = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(inp)).permute(2, 0, 1).float().div_(255.0)
        corners, conf, h_norm = self.forward_full(t.unsqueeze(0).to(device))
        scale = torch.tensor([w0, h0], dtype=corners.dtype, device=corners.device)
        corners_px = (corners[0] * scale).cpu().numpy()
        # rescale H from [0,1]-image coords to native px: diag(w,h) @ H
        s = np.diag([w0, h0, 1.0]).astype(np.float32)
        h_px = s @ h_norm[0].cpu().numpy()
        return {
            "corners": {k: [float(x), float(y)] for k, (x, y) in zip(CORNER_KEYS, corners_px)},
            "homography": h_px,
            "point_confidence": conf[0].cpu().numpy(),
            "board_confidence": float(conf[0].mean().cpu()),
        }


def load_lattice_corner_net(
    path, device: str | torch.device = "cpu", use_conf: bool = True
) -> LatticeCornerNet:
    """Load a lattice checkpoint (CornerHeatmapNet with num_corners=81) wrapped in the head."""
    from chessvision.corner_regressor import load_corner_regressor

    net = load_corner_regressor(path, device=device)
    model = LatticeCornerNet(net, use_conf=use_conf).to(device).eval()
    return model
