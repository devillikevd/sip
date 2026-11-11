"""Newly developed modules for Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion.

Note: The baseline MIRNet implementation is preserved
in root `model.py` and `antialias_downsample.py`. Modules in this package implement
our proposed illumination curve estimator, adaptive fusion, and hybrid MIRNet model.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from models.adaptive_fusion import AdaptiveFeatureFusion, OriginalFeatureEncoder
from models.hybrid_mirnet import (
    CONFIG_DESCRIPTIONS,
    BaselineMIRNetNet,
    FinalRefinementBlock,
    HybridMIRNet,
    IlluminationOnlyNet,
    SequentialMIRNetIllumNet,
    build_ablation_model,
)
from models.illumination_curve import (
    IlluminationCurveEstimator,
    color_consistency_loss,
    illumination_smoothness_loss,
)

__all__ = [
    "IlluminationCurveEstimator",
    "illumination_smoothness_loss",
    "color_consistency_loss",
    "OriginalFeatureEncoder",
    "AdaptiveFeatureFusion",
    "FinalRefinementBlock",
    "IlluminationOnlyNet",
    "BaselineMIRNetNet",
    "SequentialMIRNetIllumNet",
    "HybridMIRNet",
    "CONFIG_DESCRIPTIONS",
    "build_ablation_model",
]
