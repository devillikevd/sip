"""Composite Loss Functions for Phase 6 Hybrid Low-Light Enhancement Training.

Includes:
  - L1 reconstruction loss
  - Differentiable Gaussian-window SSIM loss
  - Color consistency loss (Gray-World + angular chromaticity alignment)
  - Structure-aware illumination & curve smoothness loss
  - Optional VGG16 perceptual loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.illumination_curve import (
    color_consistency_loss,
    illumination_smoothness_loss,
)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """Differentiable 1 - SSIM loss using a Gaussian spatial window."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        win = min(self.window_size, pred.size(2), pred.size(3))
        if win % 2 == 0:
            win = max(3, win - 1)
        window = _gaussian_window(win, self.sigma, channels, pred.device, pred.dtype)
        pad = win // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
        mu2 = F.conv2d(target, window, padding=pad, groups=channels)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(
            F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq, min=0.0
        )
        sigma2_sq = torch.clamp(
            F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq, min=0.0
        )
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        return 1.0 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Optional VGG16 feature perceptual loss (activated only when weight > 0)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        backbone = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.layer_indices = {3, 8, 15}

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        cur = (torch.clamp(x, 0.0, 1.0) - self.mean) / self.std
        feats = []
        for idx, layer in enumerate(self.backbone):
            cur = layer(cur)
            if idx in self.layer_indices:
                feats.append(cur)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_feats = self._extract(pred)
        with torch.no_grad():
            t_feats = self._extract(target)
        return sum(F.l1_loss(pf, tf) for pf, tf in zip(p_feats, t_feats)) / len(p_feats)


class HybridEnhancementLoss(nn.Module):
    """Weighted multi-term objective for Hybrid MIRNet + Illumination Curve Fusion:

        L_total = w_l1 * L1
                + w_ssim * (1 - SSIM)
                + w_color * L_color
                + w_smooth * L_smooth
                + w_perc * L_perceptual
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.25,
        color_weight: float = 0.10,
        smooth_weight: float = 0.05,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.smooth_weight = smooth_weight
        self.perceptual_weight = perceptual_weight

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perceptual = None
        if perceptual_weight > 0.0:
            try:
                self.perceptual = VGGPerceptualLoss()
            except Exception as exc:
                print(
                    f"[Warning] Could not initialize VGGPerceptualLoss ({exc}); "
                    "continuing with perceptual_weight=0.0."
                )
                self.perceptual_weight = 0.0

    def forward(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        target: torch.Tensor,
        input_image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(outputs, dict):
            pred = outputs["enhanced"]
            illum_map = outputs.get("illumination_map")
            curve_maps = outputs.get("curve_maps")
        else:
            pred = outputs
            illum_map = None
            curve_maps = None

        l_l1 = self.l1(pred, target)
        l_ssim = self.ssim(pred, target)
        l_color = color_consistency_loss(pred, reference=target)

        if illum_map is not None and self.smooth_weight > 0.0:
            l_smooth = illumination_smoothness_loss(
                illumination_map=illum_map,
                curve_maps=curve_maps,
                reference_image=input_image,
            )
        else:
            l_smooth = pred.new_zeros(())

        if self.perceptual is not None and self.perceptual_weight > 0.0:
            l_perc = self.perceptual(pred, target)
        else:
            l_perc = pred.new_zeros(())

        total = (
            self.l1_weight * l_l1
            + self.ssim_weight * l_ssim
            + self.color_weight * l_color
            + self.smooth_weight * l_smooth
            + self.perceptual_weight * l_perc
        )

        breakdown = {
            "total": float(total.item()),
            "l1": float(l_l1.item()),
            "ssim_loss": float(l_ssim.item()),
            "color": float(l_color.item()),
            "smooth": float(l_smooth.item()),
            "perceptual": float(l_perc.item()),
        }
        return total, breakdown

