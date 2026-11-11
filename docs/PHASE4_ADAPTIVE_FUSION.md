# Phase 4: Adaptive Feature Fusion Module

**Module:** [`models/adaptive_fusion.py`](../models/adaptive_fusion.py)

---

## 1. Purpose

`AdaptiveFeatureFusion` integrates three feature streams at common spatial resolution `(B, C, H, W)` (`C = 64`):

1. **Original Input Features ($F_{\text{orig}}$):** Extracted directly from the raw low-light input $I \in [0, 1]^{B \times 3 \times H \times W}$ via [`OriginalFeatureEncoder`](../models/adaptive_fusion.py#L18-L35) (`Conv3×3 + PReLU` + residual `Conv3×3-PReLU-Conv3×3`), preserving high-frequency structural edges and raw color cues.
2. **MIRNet Restoration Features ($F_{\text{mir}}$):** Multi-scale denoised and detail-restored features from MIRNet (`mir_blocks` output before `final_conv`).
3. **Illumination-Enhanced Features ($F_{\text{illum}}$):** Exposure-corrected and illumination-conditioned features from [`IlluminationCurveEstimator`](../models/illumination_curve.py#L131-L263).

---

## 2. Tensor Shapes & Mathematical Operations

Let the stacked feature tensor be:
$$\mathbf{F} = \operatorname{stack}(F_{\text{orig}}, F_{\text{mir}}, F_{\text{illum}}) \in \mathbb{R}^{B \times 3 \times C \times H \times W}$$

1. **Spatially Varying Branch Attention ($W_s$):**
   $$W_s = \operatorname{Softmax}_{\text{branch}}\!\big(\operatorname{SpatialGate}([F_{\text{orig}}, F_{\text{mir}}, F_{\text{illum}}])\big) \in [0, 1]^{B \times 3 \times H \times W}$$
   such that $\sum_{b=0}^{2} W_{s, b}(h, w) = 1$ at every pixel $(h, w)$.

2. **Spatial Summary & Channel-Selective Attention ($W_c$):**
   $$U = \sum_{b=0}^{2} W_{s,b} \odot F_b \in \mathbb{R}^{B \times C \times H \times W}$$
   $$z = \operatorname{PReLU}\!\big(\operatorname{Conv}_{1\times 1}(\operatorname{GAP}(U))\big) \in \mathbb{R}^{B \times (C/4) \times 1 \times 1}$$
   $$W_c = \operatorname{Softmax}_{\text{branch}}\!\big(\operatorname{stack}(h_0(z), h_1(z), h_2(z))\big) \in [0, 1]^{B \times 3 \times C \times 1 \times 1}$$

3. **Joint Spatio-Channel Fusion & Residual Refinement:**
   $$F_{\text{base}} = \sum_{b=0}^{2} W_{s,b} \odot \big(3\,W_{c,b} \odot F_b\big) \in \mathbb{R}^{B \times C \times H \times W}$$
   $$F_{\text{fused}} = F_{\text{base}} + \operatorname{Refine}(F_{\text{base}}) \in \mathbb{R}^{B \times C \times H \times W}$$
   The factor $3$ ensures expectation preservation at initialization when $W_{c,b} \approx \frac{1}{3}$.

