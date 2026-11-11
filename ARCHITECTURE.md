# System Architecture & Mathematical Formulation

**Project:** Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion  
**Team:** Vijendra, Vyom, Gautam  
**Base Architecture:** MIRNet (Zamir et al., ECCV 2020) ([`model.py`](model.py), [`antialias_downsample.py`](antialias_downsample.py)).

---

## 1. Existing MIRNet Architecture (Preserved)

[`model.py`](model.py) implements MIRNet (`32,212,944` parameters) using:
- Initial `3×3` convolution (`3 → 64` channels).
- `3` Residual Recurrent Groups (`RRG`), each containing `2` Multi-Scale Residual Blocks (`MSRB`) and a `3×3` residual convolution.
- Inside each `MSRB`: `3` parallel spatial streams (`H×W×64`, `H/2×W/2×128`, `H/4×W/4×256`), Dual Attention Units (`DAU` combining `SpatialAttention` and `ChannelAttention`), Selective Kernel Feature Fusion (`SKFF`), bilinear `UpSample`, and anti-aliased binomial `DownSample`.
- Final `3×3` convolution (`64 → 3`) plus global input residual skip.

Detailed file-by-file findings from the original codebase are documented in [`docs/PHASE1_REPOSITORY_ANALYSIS.md`](docs/PHASE1_REPOSITORY_ANALYSIS.md).

---

## 2. Proposed Hybrid Architecture (`HybridMIRNet`)

Our extension wraps the unmodified `MIRNet` inside a four-stage architecture in [`models/hybrid_mirnet.py`](models/hybrid_mirnet.py):

```text
                     ┌──────────────────────────────────────────────────────────┐
                     │           OriginalFeatureEncoder (3 → 64)                │
                     │                      F_orig                              │
                     └──────────────────────────┬───────────────────────────────┘
                                                │
Input I (B×3×H×W) ──┬───────────────────────────┼────────────────────────────────┐
                    ▼                           │                                │
     [Stage 1] IlluminationCurveEstimator       │                                │
       • Multi-scale dilated DW-ResBlocks       │                                │
       • Illumination map L(x) ∈ (0, 1]         │                                │
       • 8 iterative curves A_k(x) ∈ [-1, 1]    │                                │
       • Brightness & color alignment           │                                │
       ├──► F_illum (B×64×H×W) ─────────────────┼─────────┐                      │
       └──► I_illum (B×3×H×W)                   │         │                      │
               │                                │         │                      │
               ▼                                │         │                      │
     [Stage 2] Original MIRNet (model.py)       │         │                      │
       (auto reflect-padded to multiple of 4)   │         │                      │
       ├──► F_mir (B×64×H×W) ───────────────────┼───┐     │                      │
       └──► I_mir (B×3×H×W) ────────────────────┼───┼─────┼────────┐             │
                                                ▼   ▼     ▼        │             │
                                    [Stage 3] AdaptiveFeatureFusion│             │
                                      • Spatial simplex gate W_s   │             │
                                      • Channel branch gate W_c    │             │
                                      • Residual refinement        │             │
                                                │                  │             │
                                                ▼ F_fused          │             │
                                    [Stage 4] FinalRefinementBlock │             │
                                                │ ΔI_refine        │             │
                                                └────────► (+) ◄───┘             │
                                                            │                    │
                                                            ▼ clamp [0, 1]       │
                                                  Enhanced Output Î (B×3×H×W) ◄──┘
```

### Parameter Breakdown (Verified)
| Module | File | Parameters | Share |
| :--- | :--- | :---: | :---: |
| `IlluminationCurveEstimator` | [`models/illumination_curve.py`](models/illumination_curve.py) | `74,317` | `0.23%` |
| `MIRNet` (original) | [`model.py`](model.py) | `32,212,944` | `98.97%` |
| `AdaptiveFeatureFusion` | [`models/adaptive_fusion.py`](models/adaptive_fusion.py) | `185,795` | `0.57%` |
| `FinalRefinementBlock` | [`models/hybrid_mirnet.py`](models/hybrid_mirnet.py) | `75,715` | `0.23%` |
| **Total `HybridMIRNet` (Config D)** | [`models/hybrid_mirnet.py`](models/hybrid_mirnet.py) | **`32,548,771`** | **`100%`** |

---

## 3. Mathematical Formulation

### 3.1 Illumination & Iterative Curve Estimation
Given $I \in [0, 1]^{B \times 3 \times H \times W}$, shared multi-scale features $F_{\text{shared}} \in \mathbb{R}^{B \times 32 \times H \times W}$ predict:
1. **Illumination Map:**
   $$L = \operatorname{clamp}\!\big(\sigma(\operatorname{Conv}_{\text{illum}}(F_{\text{shared}})),\; 10^{-4},\; 1\big) \in (10^{-4}, 1]^{B \times 1 \times H \times W}$$
2. **Iterative Enhancement Curves ($K = 8$):**
   $$A_k = \tanh(\operatorname{Conv}_{\text{curve}}([F_{\text{shared}}, I, L])_k) \in [-1, 1]^{B \times 3 \times H \times W}$$
   $$X_k = X_{k-1} + A_k \odot X_{k-1} \odot (1 - X_{k-1}), \quad X_0 = I, \quad k = 1, \dots, 8$$
3. **Neutral Brightness Correction & Chromaticity Projection:**
   $$B_{\text{corr}} = \operatorname{clamp}\!\big(1 + \gamma (1 - L) \tanh(\text{bc}_0),\; 0.5,\; 2.0\big), \quad \delta_{\text{lift}} = 0.10 \cdot \sigma(\text{bc}_1)(1 - L)$$
   $$X_{\text{bright}} = \operatorname{clamp}(X_8 \odot B_{\text{corr}} + \delta_{\text{lift}}, 0, 1)$$
   $$I_{\text{cp}} = \operatorname{clamp}\!\left(I \odot \frac{Y(X_{\text{bright}}) + \epsilon}{Y(I) + \epsilon},\; 0,\; 1\right), \quad I_{\text{illum}} = \operatorname{clamp}\!\big((1 - w_{\text{cp}}) X_{\text{bright}} + w_{\text{cp}} I_{\text{cp}},\; 0,\; 1\big)$$
   where $w_{\text{cp}} = 0.5 \cdot \sigma(\text{bc}_2) \in [0, 0.5]$ and $Y(\cdot)$ is BT.601 luminance.

### 3.2 Adaptive Spatio-Channel Feature Fusion
Given $F_{\text{orig}}, F_{\text{mir}}, F_{\text{illum}} \in \mathbb{R}^{B \times 64 \times H \times W}$:
$$W_s = \operatorname{Softmax}_{\text{branch}}\!\big(\operatorname{SpatialGate}([F_{\text{orig}}, F_{\text{mir}}, F_{\text{illum}}])\big) \in [0, 1]^{B \times 3 \times H \times W}$$
$$U = \sum_{b=0}^{2} W_{s,b} \odot F_b, \quad W_c = \operatorname{Softmax}_{\text{branch}}\!\big(\operatorname{MLP}_{\text{ch}}(\operatorname{GAP}(U))\big) \in [0, 1]^{B \times 3 \times 64 \times 1 \times 1}$$
$$F_{\text{fused}} = F_{\text{base}} + \operatorname{Refine}(F_{\text{base}}), \quad F_{\text{base}} = \sum_{b=0}^{2} W_{s,b} \odot (3 W_{c,b} \odot F_b)$$

### 3.3 Composite Training Objective ([`losses.py`](losses.py))
$$\mathcal{L}_{\text{total}} = w_{\text{L1}}\mathcal{L}_{\text{L1}}(\hat{I}, Y) + w_{\text{SSIM}}(1 - \operatorname{SSIM}(\hat{I}, Y)) + w_{\text{color}}\mathcal{L}_{\text{color}}(\hat{I}, Y) + w_{\text{smooth}}\mathcal{L}_{\text{smooth}}(L, \mathcal{A}, I) + w_{\text{perc}}\mathcal{L}_{\text{VGG}}(\hat{I}, Y)$$
Default weights: $w_{\text{L1}}=1.0,\; w_{\text{SSIM}}=0.25,\; w_{\text{color}}=0.10,\; w_{\text{smooth}}=0.05,\; w_{\text{perc}}=0.0$ (optional VGG16 perceptual term).

