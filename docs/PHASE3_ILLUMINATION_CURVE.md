# Phase 3: Deep Illumination Curve Estimation Module

**Module:** [`models/illumination_curve.py`](../models/illumination_curve.py)  
**Target Role:** Learnable illumination estimation, iterative pixel-wise tone-curve enhancement, brightness correction, color preservation, and feature extraction for hybrid fusion.

---

## 1. Architecture Overview

`IlluminationCurveEstimator` operates at full input resolution `(H, W)` using a lightweight multi-scale depthwise-separable CNN backbone (~72K parameters, `<0.25%` of MIRNet's 32.21M parameters), ensuring compatibility with arbitrary image dimensions without spatial divisibility constraints.

```text
Input I ∈ [0,1]^(B×3×H×W)
  │
  ▼
Stem Conv3×3 (3 → C) + PReLU
  ├──► ResBlock_1 (DWConv3×3, d=1 + PWConv + SE) ──┐
  ├──► ResBlock_2 (DWConv3×3, d=2 + PWConv + SE) ──┼─► Concat(4C) ─► Conv1×1 ─► F_shared ∈ ℝ^(B×C×H×W)
  ├──► ResBlock_3 (DWConv3×3, d=4 + PWConv + SE) ──┤
  └──► ResBlock_4 (DWConv3×3, d=1 + PWConv + SE) ──┘
         │
         ├──► Illumination Head ──────────────────► L ∈ (10⁻⁴, 1]^(B×1×H×W)
         ├──► Curve Head ([F_shared, I, L]) ──────► A ∈ [-1, 1]^(B×K×3×H×W)
         │       └─► K=8 Iterative Quadratic Curves: X_0=I → ... → X_K
         ├──► Brightness & Color Gate Head ───────► (β, δ_lift, w_cp)
         │       └─► Neutral Brightness Correction + Chromaticity Projection → I_enh
         └──► Feature Projection ([F_shared, I_enh, L]) ─► F_illum ∈ ℝ^(B×64×H×W)
```

---

## 2. Mathematical Formulation

### 2.1 Multi-Scale Feature Extraction & Illumination Estimation
Given clamped input $I \in [0, 1]^{B \times 3 \times H \times W}$, four dilated depthwise-separable residual blocks ($d \in \{1, 2, 4, 1\}$) with Squeeze-and-Excitation (`_ChannelSE`) capture multi-scale context into $F_{\text{shared}} \in \mathbb{R}^{B \times C \times H \times W}$ ($C = 32$).

A learnable Retinex-inspired illumination map $L \in [\epsilon, 1]^{B \times 1 \times H \times W}$ is predicted as:
$$L = \operatorname{clamp}\!\big(\sigma(\operatorname{Conv}_{\text{illum}}(F_{\text{shared}})),\; 10^{-4},\; 1\big)$$

### 2.2 Pixel-Wise Iterative Enhancement Curves
Conditioned on $[F_{\text{shared}}, I, L]$, the curve head predicts $K = 8$ pixel-wise, channel-wise curve parameter maps:
$$\mathcal{A} = \{A_1, \dots, A_K\}, \quad A_k = \tanh(\operatorname{Conv}_{\text{curve}}([F_{\text{shared}}, I, L])_k) \in [-1, 1]^{B \times 3 \times H \times W}$$

Starting from $X_0 = I$, the quadratic light-enhancement curve (following the Zero-DCE curve family, Guo et al., CVPR 2020) is applied iteratively:
$$X_k = X_{k-1} + A_k \odot X_{k-1} \odot (1 - X_{k-1}), \quad k = 1, \dots, K$$

**Boundedness & Monotonicity:** Since $A_k \in [-1, 1]$ and $X_{k-1} \in [0, 1]$, the derivative $\frac{\partial X_k}{\partial X_{k-1}} = 1 + A_k (1 - 2 X_{k-1}) \ge 1 - |1 - 2 X_{k-1}| \ge 0$ guarantees strict monotonicity and automatic preservation of the $[0, 1]$ dynamic range without intermediate clipping.

### 2.3 Illumination-Guided Brightness Correction
In deeply underexposed regions ($I \approx 0$), the quadratic factor $X(1 - X)$ vanishes. We apply an illumination-aware, **channel-shared** multiplicative gain $B_{\text{corr}}$ and dark-floor lift $\delta_{\text{lift}}$ conditioned on $(1 - L)$:
$$\beta = \tanh(\text{bc}_0) \in [-1, 1], \qquad \delta_{\text{lift}} = 0.10 \cdot \sigma(\text{bc}_1) \cdot (1 - L) \in [0, 0.10]$$
$$B_{\text{corr}} = \operatorname{clamp}\!\big(1 + \gamma_{\text{scale}} (1 - L) \odot \beta,\; 0.5,\; 2.0\big)$$
$$X_{\text{bright}} = \operatorname{clamp}\!\big(X_K \odot B_{\text{corr}} + \delta_{\text{lift}},\; 0,\; 1\big)$$
Because $B_{\text{corr}}$ and $\delta_{\text{lift}}$ are scalar maps shared across R, G, and B channels, brightness is adjusted without distorting chromaticity ratios.

### 2.4 Chromaticity-Preserving Color Alignment
Let luminance be $Y(X) = 0.299 X_R + 0.587 X_G + 0.114 X_B$. We compute a luminance-matched chromaticity reference and blend it smoothly via a learned spatial gate $w_{\text{cp}} = 0.5 \cdot \sigma(\text{bc}_2) \in [0, 0.5]$:
$$I_{\text{cp}} = \operatorname{clamp}\!\left(I \odot \frac{Y(X_{\text{bright}}) + \epsilon}{Y(I) + \epsilon},\; 0,\; 1\right), \quad \epsilon = 10^{-4}$$
$$I_{\text{enh}} = \operatorname{clamp}\!\big((1 - w_{\text{cp}}) \odot X_{\text{bright}} + w_{\text{cp}} \odot I_{\text{cp}},\; 0,\; 1\big)$$

---

## 3. Regularization Losses

### 3.1 Structure-Aware Illumination & Curve Smoothness (`illumination_smoothness_loss`)
$$\mathcal{L}_{\text{smooth}} = \mathbb{E}\!\left[|\nabla_x L| e^{-\lambda |\nabla_x Y(I)|} + |\nabla_y L| e^{-\lambda |\nabla_y Y(I)|}\right] + \frac{1}{2K} \sum_{k=1}^{K} \mathbb{E}\!\left[\|\nabla_x A_k\|_2^2 + \|\nabla_y A_k\|_2^2\right]$$
where $\lambda = 10.0$ suppresses smoothing across strong object edges in $I$ while encouraging piecewise-smooth illumination $L$ and spatially smooth curve parameters $A_k$.

### 3.2 Color Consistency Loss (`color_consistency_loss`)
$$\mathcal{L}_{\text{color}} = \mathbb{E}\!\left[\sqrt{(\mu_R - \mu_G)^2 + (\mu_R - \mu_B)^2 + (\mu_G - \mu_B)^2 + 10^{-8}}\right] + \mathbb{E}\!\left[1 - \cos(I_{\text{enh}} + \epsilon,\; I + \epsilon)\right]$$
combines global RGB channel balance (Gray-World prior) with pixel-wise RGB vector direction alignment against the input image.

