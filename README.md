# Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion

**Team:** Vijendra, Vyom, Gautam  
**Submission Date:** 17 November 2026  

---

## 1. Project Overview

Real-world low-light photographs suffer from severe underexposure, poor local visibility, amplified sensor noise, and color distortion. This research project proposes and implements a complete hybrid restoration framework that combines:

1. **Multi-Scale MIRNet Restoration (`model.py`, `antialias_downsample.py`)** — Multi-scale residual blocks, dual attention, and selective kernel feature fusion for detail restoration (MIT-licensed backbone, preserved unmodified).
2. **Deep Illumination Curve Estimation (`models/illumination_curve.py`)** — Learnable illumination map prediction $L(x) \in (0, 1]$, pixel-wise iterative quadratic curve enhancement (`74,317` parameters), chromaticity-preserving color consistency, and structure-aware smoothness regularization.
3. **Adaptive Feature Fusion (`models/adaptive_fusion.py`)** — Differentiable spatio-channel attention fusion (`185,795` parameters) blending raw input cues, MIRNet restoration features, and illumination features.
4. **Complete Hybrid Model & Ablation Suite (`models/hybrid_mirnet.py`)** — Integrated end-to-end architecture (`32,548,771` total parameters) with modular ablation configurations **A**, **B**, **C**, and **D**.
5. **Prism-Glass 4K AI Studio Interface (`app.py`)** — Interactive full-page Streamlit application featuring single & batch enhancement, Auto-Smart LIME/Retinex + exposure-safe fill-light pipeline, progressive **4K UHD (`3840px`)** super-resolution, 60fps draggable before/after wipe, 3.0× live magnifier loupe, luminance histogram telemetry, illumination heatmaps, and multi-format 4K PNG/JPG/ZIP exports.

---

## 2. Model Configurations & Modules

| Config | Architecture | Parameters |
|---|---|---|
| **A** | Illumination Curve Estimation branch only | `74,317` |
| **B** | Original MIRNet baseline | `32,212,944` |
| **C** | Sequential Illumination Curve → MIRNet (no Adaptive Fusion) | `32,287,261` |
| **D** | **Proposed Hybrid MIRNet** (Illumination + MIRNet + Adaptive Fusion + Refinement) | `32,548,771` |

---

## 3. Repository Structure

```text
├── model.py                         # MIRNet backbone architecture (MIT base, unmodified)
├── antialias_downsample.py          # Anti-aliased binomial downsampling layer
├── dataset.py                       # LoLDataset & PairedRandomPatchDataset loaders
├── utils.py                         # MixUp augmentation, PSNR, SSIM & image I/O helpers
├── losses.py                        # Composite L1 + SSIM + Color + Smoothness + VGG loss
├── train_enhancement.py             # Baseline MIRNet training script
├── train_hybrid.py                  # Complete Colab-ready training pipeline (Configs A–D)
├── app.py                           # Prism-Glass 4K AI Studio Streamlit application
├── conftest.py                      # Pytest environment & deterministic seed configuration
├── models/
│   ├── __init__.py
│   ├── illumination_curve.py        # Learnable IlluminationCurveEstimator & regularizers
│   ├── adaptive_fusion.py           # AdaptiveFeatureFusion & OriginalFeatureEncoder
│   └── hybrid_mirnet.py             # HybridMIRNet & build_ablation_model factory
├── scripts/
│   ├── baseline_verify.py           # Baseline verification & sample extraction
│   ├── run_ablation.py              # Config A–D ablation study & metric plots
│   ├── research_comparison.py       # Published literature vs local evaluation charts
│   ├── generate_visualizations.py   # Multi-panel qualitative comparison figures
│   └── build_notebook.py            # Reproducible Google Colab notebook builder
├── notebooks/
│   └── Complete_Project_Implementation.ipynb
├── tests/                           # Automated verification suite (21 tests across phases)
├── docs/                            # Detailed per-phase technical notes
├── baseline_results/                # Baseline diagnostics, samples & smoke-test logs
├── experiments/                     # Ablation CSVs, comparison charts & tables
├── Results/
│   ├── result1.png..result7.png     # Reference enhancement triptychs
│   └── comparisons/                 # Generated multi-method & internal feature plots
├── ARCHITECTURE.md                  # System design, tensor shapes & math formulation
├── EXPERIMENTS.md                   # Experimental setup, metrics & ablation analysis
├── LICENSE                          # MIT License
└── requirements.txt                 # Python dependencies
```

---

## 4. Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Automated Verification
```bash
python -m pytest tests/ -v
```

### 3. Launch the 4K AI Studio Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501`, upload any dark/night photo, adjust **Light Boost / Fill-Light / HD Clarity / Warmth** presets, inspect with **60fps Drag Wipe** or **3.0× 4K Live Zoom**, and export in **4K UHD (`3840px`)** PNG/JPG.

### 4. Train on LOL Dataset (Local GPU or Google Colab)
Place the LOL dataset (`our485/` and `eval15/`) under `data/LOL/`:
```bash
python train_hybrid.py --config D --train_low data/LOL/our485/low --train_high data/LOL/our485/high --val_low data/LOL/eval15/low --val_high data/LOL/eval15/high --epochs 50 --batch_size 4 --patch_size 128 --amp
```

### 5. Run Ablation, Literature Comparison & Visual Figures
```bash
python scripts/run_ablation.py
python scripts/research_comparison.py
python scripts/generate_visualizations.py
```

### 6. Run End-to-End in Google Colab
Upload or open [`notebooks/Complete_Project_Implementation.ipynb`](notebooks/Complete_Project_Implementation.ipynb) in Google Colab (T4 GPU recommended) and run all cells sequentially.

---

## 5. Documentation & Verification Notes

- Architectural diagrams and equations: see [`ARCHITECTURE.md`](ARCHITECTURE.md).
- Quantitative experiments, ablation CSVs, and published benchmarks: see [`EXPERIMENTS.md`](EXPERIMENTS.md).
- Local verification (`torch 2.12.1+cpu`, bundled LOL sample pairs) confirms model parameter counts, end-to-end forward/backward passes, checkpoint resume training, notebook execution, and real-time 4K inference UI; full 485-pair LOL GPU training is configured in `train_hybrid.py` and the Colab notebook.

---

## 6. References & License

- **MIRNet:** S. W. Zamir et al., *"Learning Enriched Features for Real Image Restoration and Enhancement"*, ECCV 2020.
- **LIME:** X. Guo et al., *"LIME: Low-Light Image Enhancement via Illumination Map Estimation"*, IEEE TIP 2017.
- **Zero-DCE:** C. Guo et al., *"Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"*, CVPR 2020.
- **License:** Distributed under the [MIT License](LICENSE) (base MIRNet PyTorch implementation preserved under MIT).
