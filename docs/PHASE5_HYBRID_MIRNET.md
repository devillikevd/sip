# Phase 5: Complete Hybrid Model (`HybridMIRNet`)

**Module:** [`models/hybrid_mirnet.py`](../models/hybrid_mirnet.py)

---

## 1. End-to-End Integration Architecture

`HybridMIRNet` preserves the original `MIRNet` class from [`model.py`](../model.py) unmodified and wraps it into a four-stage pipeline:

```text
Input Image I ∈ [0,1]^(B×3×H×W)
  ├──► [Stage 1] IlluminationCurveEstimator ──► I_illum (B×3×H×W), L (B×1×H×W), F_illum (B×64×H×W)
  │
  ├──► [Stage 2] Original MIRNet (on I_illum, reflect-padded to multiple of 4)
  │       └─► F_mir = mir_blocks(first_conv(I_illum)) ∈ ℝ^(B×64×H×W)
  │       └─► I_mir = final_conv(F_mir) + I_illum      ∈ ℝ^(B×3×H×W)
  │
  ├──► [Stage 3] AdaptiveFeatureFusion(I, F_mir, F_illum)
  │       └─► F_fused ∈ ℝ^(B×64×H×W)  (+ spatial W_s and channel W_c weights)
  │
  └──► [Stage 4] FinalRefinementBlock(F_fused) ──► ΔI_refine
          └─► I_enhanced = clamp(I_mir + ΔI_refine, 0, 1)
```

---

## 2. Ablation Factory (`build_ablation_model`)

All four experimental configurations share an identical `forward(x, return_dict=False)` interface:

| Config | Class | Components Active |
| :---: | :--- | :--- |
| **A** | `IlluminationOnlyNet` | `IlluminationCurveEstimator` only |
| **B** | `BaselineMIRNetNet` | Original `MIRNet` only (resolution-safe wrapper) |
| **C** | `SequentialMIRNetIllumNet` | `IlluminationCurveEstimator` → `MIRNet` (no Adaptive Fusion) |
| **D** | `HybridMIRNet` | Full pipeline (`IlluminationCurveEstimator` → `MIRNet` → `AdaptiveFeatureFusion` → `FinalRefinementBlock`) |

