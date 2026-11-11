"""Deep Illumination Curve Estimation Module (Phase 3).

Author: Project Extension for
    "Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion"

This module implements a learnable, fully differentiable deep illumination and
pixel-wise iterative enhancement curve estimator. It combines:
  1. Multi-scale depthwise-separable feature extraction with Squeeze-and-Excitation.
  2. Learnable Retinex-inspired illumination map estimation L(x) in (0, 1].
  3. Pixel-wise, channel-wise iterative quadratic enhancement curves (K iterations):
         X_k = X_{k-1} + A_k * X_{k-1} * (1 - X_{k-1}),  A_k in [-1, 1]
  4. Illumination-guided brightness correction that lifts severely underexposed regions
     uniformly across RGB channels to avoid hue shift.
  5. Chromaticity-preserving color alignment and smoothness regularization losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelSE(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.PReLU(hidden)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(self.act(self.fc1(w))))
        return x * w


class _DepthwiseSeparableResBlock(nn.Module):
    """Multi-scale dilated depthwise-separable residual block preserving (H, W)."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.PReLU(channels)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.se = _ChannelSE(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.pw1(self.dw(x)))
        out = self.se(self.pw2(out))
        return x + out


def illumination_smoothness_loss(
    illumination_map: torch.Tensor,
    curve_maps: torch.Tensor | None = None,
    reference_image: torch.Tensor | None = None,
    edge_weight: float = 10.0,
) -> torch.Tensor:
    """Structure-aware Total Variation smoothness loss for illumination & curves.

    Penalizes rapid spatial variations in the estimated illumination map ``L`` (and
    optionally the pixel-wise curve maps ``A_k``). When ``reference_image`` is
    provided, gradients of ``L`` are weighted by ``exp(-edge_weight * |grad(I)|)``
    so illumination stays piecewise smooth while preserving object boundaries.

    Args:
        illumination_map: Tensor of shape ``(B, 1, H, W)`` in ``(0, 1]``.
        curve_maps: Optional tensor of shape ``(B, K, 3, H, W)`` or ``(B, 3*K, H, W)``.
        reference_image: Optional low-light image ``(B, 3, H, W)`` for edge-aware weighting.
        edge_weight: Edge attenuation coefficient for structure-aware smoothness.

    Returns:
        Scalar smoothness regularization loss tensor.
    """
    dh_l = torch.abs(illumination_map[:, :, 1:, :] - illumination_map[:, :, :-1, :])
    dw_l = torch.abs(illumination_map[:, :, :, 1:] - illumination_map[:, :, :, :-1])

    if reference_image is not None:
        gray = (
            0.299 * reference_image[:, 0:1]
            + 0.587 * reference_image[:, 1:2]
            + 0.114 * reference_image[:, 2:3]
        )
        dh_i = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        dw_i = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        weight_h = torch.exp(-edge_weight * dh_i.detach())
        weight_w = torch.exp(-edge_weight * dw_i.detach())
        tv_illum = (dh_l * weight_h).mean() + (dw_l * weight_w).mean()
    else:
        tv_illum = dh_l.mean() + dw_l.mean()

    if curve_maps is None:
        return tv_illum

    if curve_maps.dim() == 5:
        b, k, c, h, w = curve_maps.shape
        curves = curve_maps.reshape(b, k * c, h, w)
    else:
        curves = curve_maps

    dh_c = torch.pow(curves[:, :, 1:, :] - curves[:, :, :-1, :], 2).mean()
    dw_c = torch.pow(curves[:, :, :, 1:] - curves[:, :, :, :-1], 2).mean()
    return tv_illum + 0.5 * (dh_c + dw_c)


def color_consistency_loss(
    enhanced: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Color preservation loss combining Gray-World balance and chromaticity alignment.

    1. **Gray-World / Spatial Channel Balance**: Penalizes global channel-mean
       imbalances ``(mu_R - mu_G)^2 + (mu_R - mu_B)^2 + (mu_G - mu_B)^2``.
    2. **Chromaticity Direction Alignment** (when ``reference`` is provided):
       Penalizes cosine angular deviation ``1 - cos_sim(enhanced, reference)``
       per pixel so RGB ratios remain faithful without restricting brightness lift.
    """
    mean_rgb = enhanced.mean(dim=(2, 3))  # (B, 3)
    d_rg = (mean_rgb[:, 0] - mean_rgb[:, 1]) ** 2
    d_rb = (mean_rgb[:, 0] - mean_rgb[:, 2]) ** 2
    d_gb = (mean_rgb[:, 1] - mean_rgb[:, 2]) ** 2
    gray_world = torch.sqrt(d_rg + d_rb + d_gb + 1e-8).mean()

    if reference is None:
        return gray_world

    eps = 1e-4
    cos_sim = F.cosine_similarity(enhanced + eps, reference + eps, dim=1)
    angular_loss = (1.0 - cos_sim).mean()
    return gray_world + angular_loss


class IlluminationCurveEstimator(nn.Module):
    """Learnable Deep Illumination & Pixel-Wise Enhancement Curve Estimator.

    Architecture:
      Input I (B, 3, H, W)
        ├──► Stem Conv3x3(3 -> C) + Multi-Scale Dilated ResBlocks (d=1, 2, 4, 1)
        │      └──► Feature Aggregation Conv1x1(4C -> C)  [shared features F_shared]
        ├──► Illumination Head: [F_shared] -> L in (eps, 1]  (B, 1, H, W)
        ├──► Curve Head: [F_shared, L, I] -> A in [-1, 1]    (B, K, 3, H, W)
        ├──► Brightness & Color Gate Head: [F_shared, L] -> (beta, delta_lift, w_cp)
        └──► Feature Projection Head (for Hybrid Fusion): [F_shared, enhanced, L] -> (B, feature_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        feature_channels: int = 64,
        num_iters: int = 8,
    ):
        super().__init__()
        if num_iters < 1:
            raise ValueError(f"num_iters must be >= 1, got {num_iters}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.feature_channels = feature_channels
        self.num_iters = num_iters

        # Stem + multi-scale residual backbone (preserves arbitrary HxW)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(base_channels),
        )
        self.block1 = _DepthwiseSeparableResBlock(base_channels, dilation=1)
        self.block2 = _DepthwiseSeparableResBlock(base_channels, dilation=2)
        self.block3 = _DepthwiseSeparableResBlock(base_channels, dilation=4)
        self.block4 = _DepthwiseSeparableResBlock(base_channels, dilation=1)

        self.agg = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=1, bias=True),
            nn.PReLU(base_channels),
        )

        # 1. Learnable illumination map estimator L(x) in (1e-4, 1]
        self.illum_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.PReLU(base_channels // 2),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1, bias=True),
        )

        # 2. Pixel-wise curve parameter estimator A_k(x) in [-1, 1]
        curve_in = base_channels + in_channels + 1
        self.curve_head = nn.Sequential(
            nn.Conv2d(curve_in, base_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(base_channels),
            nn.Conv2d(base_channels, in_channels * num_iters, kernel_size=3, padding=1, bias=True),
        )

        # 3. Illumination-guided brightness correction & color preservation gate
        # Outputs 3 spatial maps: [beta_gain, delta_lift, w_color_preserve]
        self.bc_head = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels // 2, kernel_size=3, padding=1, bias=True),
            nn.PReLU(base_channels // 2),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1, bias=True),
        )
        self.gain_scale = nn.Parameter(torch.tensor(0.25))

        # 4. Feature projection for Phase 4 Adaptive Feature Fusion
        self.feature_proj = nn.Sequential(
            nn.Conv2d(base_channels + in_channels + 1, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels),
        )

    @staticmethod
    def _luminance(img: torch.Tensor) -> torch.Tensor:
        return (
            0.299 * img[:, 0:1]
            + 0.587 * img[:, 1:2]
            + 0.114 * img[:, 2:3]
        )

    def apply_iterative_curves(
        self, x: torch.Tensor, curve_maps: torch.Tensor
    ) -> torch.Tensor:
        """Apply K pixel-wise quadratic enhancement curves iteratively.

        Args:
            x: Clamped input image ``(B, 3, H, W)`` in ``[0, 1]``.
            curve_maps: Curve coefficients ``(B, K, 3, H, W)`` in ``[-1, 1]``.
        """
        cur = x
        for k in range(self.num_iters):
            a_k = curve_maps[:, k]
            cur = cur + a_k * cur * (1.0 - cur)
        return cur

    def compute_regularization_loss(
        self,
        outputs: dict[str, torch.Tensor],
        input_image: torch.Tensor,
        smooth_weight: float = 1.0,
        color_weight: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute illumination smoothness and color preservation regularization."""
        l_smooth = illumination_smoothness_loss(
            illumination_map=outputs["illumination_map"],
            curve_maps=outputs["curve_maps"],
            reference_image=input_image,
        )
        l_color = color_consistency_loss(
            enhanced=outputs["enhanced"],
            reference=input_image,
        )
        total = smooth_weight * l_smooth + color_weight * l_color
        return total, {
            "illumination_smoothness": l_smooth,
            "color_consistency": l_color,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x_clamped = torch.clamp(x, 0.0, 1.0)

        f0 = self.stem(x_clamped)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f_shared = self.agg(torch.cat([f1, f2, f3, f4], dim=1))

        # 1. Estimated illumination map L(x) in (1e-4, 1]
        illumination_map = torch.clamp(torch.sigmoid(self.illum_head(f_shared)), min=1e-4, max=1.0)

        # 2. Pixel-wise enhancement curve parameters A_k(x) in [-1, 1]
        curve_raw = self.curve_head(torch.cat([f_shared, x_clamped, illumination_map], dim=1))
        curve_maps = torch.tanh(curve_raw).view(
            x.size(0), self.num_iters, self.in_channels, x.size(2), x.size(3)
        )

        # 3. Iterative quadratic curve application
        curve_enhanced = self.apply_iterative_curves(x_clamped, curve_maps)

        # 4. Channel-shared brightness correction (lifts underexposed pixels neutrally)
        bc_params = self.bc_head(torch.cat([f_shared, illumination_map], dim=1))
        beta = torch.tanh(bc_params[:, 0:1])            # [-1, 1]
        delta_lift = 0.10 * torch.sigmoid(bc_params[:, 1:2]) * (1.0 - illumination_map)
        w_cp = 0.5 * torch.sigmoid(bc_params[:, 2:3])   # [0, 0.5] chromaticity blend weight

        brightness_gain = torch.clamp(
            1.0 + self.gain_scale * (1.0 - illumination_map) * beta, min=0.5, max=2.0
        )
        bright_corrected = torch.clamp(curve_enhanced * brightness_gain + delta_lift, 0.0, 1.0)

        # 5. Color preservation via luminance-matched chromaticity projection
        eps = 1e-4
        lum_in = self._luminance(x_clamped)
        lum_corr = self._luminance(bright_corrected)
        chroma_preserved = torch.clamp(
            x_clamped * ((lum_corr + eps) / (lum_in + eps)),
            0.0,
            1.0,
        )
        enhanced = torch.clamp(
            (1.0 - w_cp) * bright_corrected + w_cp * chroma_preserved,
            0.0,
            1.0,
        )

        if not return_dict:
            return enhanced

        features = self.feature_proj(torch.cat([f_shared, enhanced, illumination_map], dim=1))
        return {
            "enhanced": enhanced,
            "curve_enhanced": curve_enhanced,
            "illumination_map": illumination_map,
            "curve_maps": curve_maps,
            "brightness_gain": brightness_gain,
            "delta_lift": delta_lift,
            "color_blend_weight": w_cp,
            "features": features,
        }

