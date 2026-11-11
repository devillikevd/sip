"""Adaptive Feature Fusion Module (Phase 4).

Combines three complementary feature representations:
  1. Original input features (extracted from the raw low-light image I).
  2. MIRNet restoration features (multi-scale denoised/restored representation).
  3. Illumination-enhanced features (global/local exposure & curve cues).

Uses both spatially varying pixel-wise branch weights W_s(h, w) in the 3-simplex
and channel-wise selective branch attention W_c(c) so the network dynamically
balances texture preservation, denoising, and illumination adjustment at every pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalFeatureEncoder(nn.Module):
    """Lightweight residual feature extractor for the raw low-light input image."""

    def __init__(self, in_channels: int = 3, feature_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels),
        )
        self.res = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.stem(x)
        return base + self.res(base)


class AdaptiveFeatureFusion(nn.Module):
    """Spatially Varying + Channel-Selective Three-Branch Feature Fusion.

    Inputs:
        input_or_orig_feat: Either raw input image ``(B, 3, H, W)`` (projected via
            ``OriginalFeatureEncoder``) or pre-extracted original features ``(B, C, H, W)``.
        mirnet_features: MIRNet restoration feature map ``(B, C, H, W)``.
        illum_features: Illumination-enhanced feature map ``(B, C, H, W)``.

    Outputs:
        fused_features: ``(B, C, H, W)``
        (optional) attention_info dict with ``spatial_weights`` ``(B, 3, H, W)``
        and ``channel_weights`` ``(B, 3, C, 1, 1)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 64,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        hidden = max(feature_channels // reduction, 8)

        self.orig_encoder = OriginalFeatureEncoder(in_channels, feature_channels)

        # Spatially varying 3-branch gating network: (B, 3C, H, W) -> W_s (B, 3, H, W)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1, bias=True),
            nn.PReLU(feature_channels),
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels // 2),
            nn.Conv2d(feature_channels // 2, 3, kernel_size=3, padding=1, bias=True),
        )

        # Channel-wise selective attention across the 3 branches: W_c (B, 3, C, 1, 1)
        self.channel_squeeze = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, kernel_size=1, bias=True),
            nn.PReLU(hidden),
        )
        self.channel_heads = nn.ModuleList(
            [nn.Conv2d(hidden, feature_channels, kernel_size=1, bias=True) for _ in range(3)]
        )

        # Post-fusion refinement residual block
        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
        )

    def _prepare_original_features(self, orig: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if orig.size(1) == self.in_channels:
            f_orig = self.orig_encoder(orig)
        elif orig.size(1) == self.feature_channels:
            f_orig = orig
        else:
            raise ValueError(
                f"Expected input_or_orig_feat channels to be {self.in_channels} or "
                f"{self.feature_channels}, got {orig.size(1)}"
            )
        if f_orig.shape[-2:] != (target_h, target_w):
            f_orig = F.interpolate(
                f_orig, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return f_orig

    def forward(
        self,
        input_or_orig_feat: torch.Tensor,
        mirnet_features: torch.Tensor,
        illum_features: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mirnet_features.shape != illum_features.shape:
            raise ValueError(
                f"Shape mismatch: mirnet_features {tuple(mirnet_features.shape)} "
                f"vs illum_features {tuple(illum_features.shape)}"
            )
        b, c, h, w = mirnet_features.shape
        if c != self.feature_channels:
            raise ValueError(
                f"Expected feature_channels={self.feature_channels}, got {c}"
            )

        f_orig = self._prepare_original_features(input_or_orig_feat, h, w)

        # 1. Spatially varying simplex weights W_s ∈ [0, 1]^(B×3×H×W), sum_b W_s = 1
        cat_feats = torch.cat([f_orig, mirnet_features, illum_features], dim=1)
        spatial_logits = self.spatial_gate(cat_feats)
        spatial_weights = torch.softmax(spatial_logits, dim=1)  # (B, 3, H, W)

        # Initial spatially fused summary U for global channel weighting
        stacked = torch.stack([f_orig, mirnet_features, illum_features], dim=1)  # (B, 3, C, H, W)
        u = (stacked * spatial_weights.unsqueeze(2)).sum(dim=1)  # (B, C, H, W)

        # 2. Channel-wise branch attention W_c ∈ [0, 1]^(B×3×C×1×1), sum_b W_c = 1
        z = self.channel_squeeze(F.adaptive_avg_pool2d(u, 1))
        ch_logits = torch.stack([head(z) for head in self.channel_heads], dim=1)  # (B, 3, C, 1, 1)
        channel_weights = torch.softmax(ch_logits, dim=1)

        # 3. Joint spatio-channel adaptive weighting (3.0 preserves scale at uniform init)
        reweighted = stacked * (3.0 * channel_weights)
        fused_base = (reweighted * spatial_weights.unsqueeze(2)).sum(dim=1)

        fused = fused_base + self.refine(fused_base)

        if not return_weights:
            return fused
        return fused, {
            "spatial_weights": spatial_weights,
            "channel_weights": channel_weights,
        }

