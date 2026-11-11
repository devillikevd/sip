"""Complete Hybrid MIRNet + Deep Illumination Curve Fusion Model (Phase 5)
and Ablation Model Factory (Phase 7).

Preserves the original `MIRNet` implementation from root `model.py` without
modifying its architecture, while integrating:
  Input Image
    -> Illumination Estimation (IlluminationCurveEstimator)
    -> MIRNet Restoration (original MIRNet)
    -> Adaptive Feature Fusion (AdaptiveFeatureFusion)
    -> Final Refinement
    -> Enhanced Image
"""

import contextlib
import io
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MIRNet
from models.adaptive_fusion import AdaptiveFeatureFusion
from models.illumination_curve import IlluminationCurveEstimator


def _pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad (B, C, H, W) so H and W are divisible by `multiple`."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    # Reflect padding requires pad < dim size; fall back to replicate for tiny tensors
    mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (pad_h, pad_w)


def _unpad(x: torch.Tensor, pads: tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pads
    h = x.size(2) - pad_h
    w = x.size(3) - pad_w
    return x[:, :, :h, :w]


def _run_mirnet_features(
    mirnet: MIRNet, x: torch.Tensor, silence_prints: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run unmodified MIRNet, returning both final image and pre-final-conv features."""
    x_pad, pads = _pad_to_multiple(x, multiple=4)
    ctx = contextlib.redirect_stdout(io.StringIO()) if silence_prints else contextlib.nullcontext()
    with ctx:
        feat = mirnet.first_conv(x_pad)
        feat = mirnet.mir_blocks(feat)
        out_pad = mirnet.final_conv(feat) + x_pad
    feat_unpad = _unpad(feat, pads)
    out_unpad = _unpad(out_pad, pads)
    return out_unpad, feat_unpad


class FinalRefinementBlock(nn.Module):
    """Lightweight residual refinement head mapping fused features to RGB residual."""

    def __init__(self, feature_channels: int = 64, out_channels: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(feature_channels),
        )
        self.to_rgb = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        res = fused_features + self.body(fused_features)
        return self.to_rgb(res)


class IlluminationOnlyNet(nn.Module):
    """Ablation Config A: Illumination Curve Estimation branch only."""

    def __init__(self, base_channels: int = 32, feature_channels: int = 64, num_iters: int = 8):
        super().__init__()
        self.illumination = IlluminationCurveEstimator(
            in_channels=3,
            base_channels=base_channels,
            feature_channels=feature_channels,
            num_iters=num_iters,
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        return self.illumination(x, return_dict=return_dict)


class BaselineMIRNetNet(nn.Module):
    """Ablation Config B: Original MIRNet architecture wrapped safely for any resolution."""

    def __init__(self, silence_prints: bool = True, **mirnet_kwargs):
        super().__init__()
        self.silence_prints = silence_prints
        self.mirnet = MIRNet(**mirnet_kwargs)

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        mir_out, mir_feat = _run_mirnet_features(
            self.mirnet, x_clamped, silence_prints=self.silence_prints
        )
        enhanced = torch.clamp(mir_out, 0.0, 1.0)
        if not return_dict:
            return enhanced
        return {"enhanced": enhanced, "mirnet_features": mir_feat}


class SequentialMIRNetIllumNet(nn.Module):
    """Ablation Config C: Sequential Illumination Estimation -> MIRNet (no Adaptive Fusion)."""

    def __init__(
        self,
        base_channels: int = 32,
        feature_channels: int = 64,
        num_iters: int = 8,
        silence_prints: bool = True,
        **mirnet_kwargs,
    ):
        super().__init__()
        self.silence_prints = silence_prints
        self.illumination = IlluminationCurveEstimator(
            in_channels=3,
            base_channels=base_channels,
            feature_channels=feature_channels,
            num_iters=num_iters,
        )
        self.mirnet = MIRNet(**mirnet_kwargs)

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        illum_out = self.illumination(x_clamped, return_dict=True)
        mir_out, mir_feat = _run_mirnet_features(
            self.mirnet, illum_out["enhanced"], silence_prints=self.silence_prints
        )
        enhanced = torch.clamp(mir_out, 0.0, 1.0)
        if not return_dict:
            return enhanced
        return {
            "enhanced": enhanced,
            "illum_enhanced": illum_out["enhanced"],
            "illumination_map": illum_out["illumination_map"],
            "curve_maps": illum_out["curve_maps"],
            "mirnet_features": mir_feat,
        }


class HybridMIRNet(nn.Module):
    """Ablation Config D (Proposed Full Model):
    Input -> Illumination Curve Estimator -> MIRNet -> Adaptive Feature Fusion -> Final Refinement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        illum_base_channels: int = 32,
        num_iters: int = 8,
        num_MSRB: int = 2,
        num_RRG: int = 3,
        height: int = 3,
        width: int = 2,
        bias: bool = False,
        silence_prints: bool = True,
    ):
        super().__init__()
        self.silence_prints = silence_prints

        # Stage 1: Deep Illumination Curve Estimation
        self.illumination = IlluminationCurveEstimator(
            in_channels=in_channels,
            base_channels=illum_base_channels,
            feature_channels=num_features,
            num_iters=num_iters,
        )

        # Stage 2: Original MIRNet Restoration (unmodified class from model.py)
        self.mirnet = MIRNet(
            in_c=in_channels,
            out_c=out_channels,
            num_features=num_features,
            kernel_size=3,
            stride=2,
            num_MSRB=num_MSRB,
            num_RRG=num_RRG,
            height=height,
            width=width,
            bias=bias,
        )

        # Stage 3: Adaptive Feature Fusion (original input + MIRNet + illumination features)
        self.fusion = AdaptiveFeatureFusion(
            in_channels=in_channels,
            feature_channels=num_features,
            reduction=4,
        )

        # Stage 4: Final RGB Residual Refinement
        self.refinement = FinalRefinementBlock(
            feature_channels=num_features,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        x_clamped = torch.clamp(x, 0.0, 1.0)

        # 1. Illumination curve estimation
        illum_out = self.illumination(x_clamped, return_dict=True)
        illum_enhanced = illum_out["enhanced"]
        illum_features = illum_out["features"]

        # 2. MIRNet restoration on illumination-enhanced image
        mir_out, mir_features = _run_mirnet_features(
            self.mirnet, illum_enhanced, silence_prints=self.silence_prints
        )

        # 3. Adaptive fusion of original input, MIRNet features, and illumination features
        fused_features, fusion_weights = self.fusion(
            x_clamped, mir_features, illum_features, return_weights=True
        )

        # 4. Final refinement over restored output
        refined_residual = self.refinement(fused_features)
        enhanced = torch.clamp(mir_out + refined_residual, 0.0, 1.0)

        if not return_dict:
            return enhanced

        return {
            "enhanced": enhanced,
            "illum_enhanced": illum_enhanced,
            "mirnet_output": torch.clamp(mir_out, 0.0, 1.0),
            "illumination_map": illum_out["illumination_map"],
            "curve_maps": illum_out["curve_maps"],
            "fusion_weights": fusion_weights,
            "fused_features": fused_features,
        }


CONFIG_DESCRIPTIONS = {
    "A": "Illumination branch only (IlluminationCurveEstimator)",
    "B": "Original MIRNet only (Baseline)",
    "C": "MIRNet + Illumination estimation (Sequential, no Adaptive Fusion)",
    "D": "MIRNet + Illumination estimation + Adaptive Feature Fusion (Proposed Hybrid)",
}


def build_ablation_model(config: str = "D", **kwargs) -> nn.Module:
    """Instantiate one of the four ablation configurations A, B, C, or D."""
    cfg = config.strip().upper()
    if cfg == "A":
        return IlluminationOnlyNet(**kwargs)
    if cfg == "B":
        return BaselineMIRNetNet(**kwargs)
    if cfg == "C":
        return SequentialMIRNetIllumNet(**kwargs)
    if cfg == "D":
        return HybridMIRNet(**kwargs)
    raise ValueError(f"Unknown ablation config '{config}'. Expected one of: A, B, C, D.")

