# Low-Light Image Enhancement via Hybrid Illumination Curves and Multi-Scale Feature Restoration

**Project Submission:** Minor Project / SIP  
**Team Members:** Vijendra, Vyom, Gautam  
**Academic Year:** 2026  

---

## 1. Problem & Project Objective

Photographs taken in poor lighting conditions typically exhibit three major defects: severe underexposure, high sensor noise (especially in chrominance channels), and unnatural color shifts. Standard adjustment techniques like histogram equalization or global gamma correction brighten the image indiscriminately, blowing out background highlights and amplifying dark noise.

In this project, we built a hybrid deep learning model that divides the restoration task into two dedicated stages:
1. **Illumination Adjustment:** A lightweight curve estimation branch (`models/illumination_curve.py`, ~74K parameters) predicts pixel-wise quadratic tone curves to adjust exposure smoothly without clipping highlights.
2. **Detail Restoration:** The unmodified MIRNet backbone (`model.py`, ~32.2M parameters) extracts multi-scale contextual features across three resolutions to remove residual sensor noise and recover lost edges.
3. **Adaptive Fusion:** A custom gating module (`models/adaptive_fusion.py`, ~185K parameters) computes spatial and channel-wise weights to dynamically combine raw input details, restored MIRNet features, and illumination maps.

The combined architecture (**32.55M parameters**) achieves balanced exposure, higher structural similarity (SSIM > 0.85), and lower color distortion on the benchmark LOL dataset.

---

## 2. Model Configurations (Ablation Suite)

We implemented four modular configurations in `models/hybrid_mirnet.py` to evaluate each sub-system independently:

| Config | Architecture Description | Parameters | Purpose |
|:---:|---|:---:|---|
| **A** | Illumination Curve Branch Only | `74,317` | Evaluates pure curve-based brightening without heavy backbones. |
| **B** | Baseline MIRNet Architecture | `32,212,944` | Evaluates standard restoration without illumination conditioning. |
| **C** | Sequential Pipeline (Curve → MIRNet) | `32,287,261` | Direct feedforward pass without dynamic attention fusion. |
| **D** | **Proposed Hybrid Model (Full)** | `32,548,771` | Complete system with Adaptive Feature Fusion and residual refinement. |

---

## 3. Repository Structure

```text
├── model.py                         # MIRNet backbone (Zamir et al., ECCV 2020)
├── antialias_downsample.py          # Anti-aliased binomial downsampling layer
├── dataset.py                       # LOL Dataset loader with synchronized random patch cropping
├── utils.py                         # PSNR, SSIM, and image processing helpers
├── losses.py                        # Composite loss (L1 + SSIM + Color + Smoothness)
├── train_hybrid.py                  # Training pipeline with AdamW, CosineAnnealing, and AMP
├── app.py                           # Streamlit web interface for evaluation and before/after slider
├── conftest.py                      # Deterministic seed configuration for automated pytest
├── models/
│   ├── illumination_curve.py        # 8-step iterative quadratic curve estimator
│   ├── adaptive_fusion.py           # Spatio-channel gating and feature fusion
│   └── hybrid_mirnet.py             # HybridMIRNet wrapper and ablation model factory
├── scripts/
│   ├── baseline_verify.py           # Verification script for local sample pairs
│   ├── run_ablation.py              # Automated script running Configs A through D
│   ├── research_comparison.py       # Benchmark chart generation against published methods
│   └── build_notebook.py            # Colab notebook generation script
├── notebooks/
│   └── Complete_Project_Implementation.ipynb
├── tests/                           # Pytest test suite (21 unit tests)
├── experiments/                     # CSV logs, ablation tables, and comparative charts
├── Results/                         # Reference sample images and comparison grids
├── ARCHITECTURE.md                  # Math formulation and layer tensor shapes
├── EXPERIMENTS.md                   # Metric results and quantitative benchmarks
├── LICENSE                          # MIT License
└── requirements.txt                 # Project dependencies
```

---

## 4. How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
To verify module shapes, tensor passes, and loss functions across the pipeline:
```bash
python -m pytest tests/ -v
```

### Step 3: Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
This opens `http://localhost:8501`. You can upload any dark image, switch between Configs A, B, C, and D, inspect the result using the interactive split slider, and inspect the predicted illumination heatmap.

### Step 4: Training on LOL Dataset
Place the LOL dataset folders under `data/LOL/`:
```bash
python train_hybrid.py --config D --train_low data/LOL/our485/low --train_high data/LOL/our485/high --val_low data/LOL/eval15/low --val_high data/LOL/eval15/high --epochs 50 --batch_size 4 --patch_size 128 --amp
```

---

## 5. Quantitative Results (LOL Dataset Benchmark)

| Model | Publication / Type | Parameters | PSNR (dB) ↑ | SSIM ↑ |
|---|---|:---:|:---:|:---:|
| Raw Low-Light Image | Input Camera Capture | — | `6.80` | `0.1507` |
| Zero-DCE | CVPR 2020 (Curve only) | 0.079 M | `14.86` | `0.559` |
| RetinexNet | BMVC 2018 (Decomposition) | 0.84 M | `16.77` | `0.560` |
| EnlightenGAN | IEEE TIP 2021 (Unpaired) | 8.64 M | `17.48` | `0.651` |
| MIRNet (Paper) | ECCV 2020 (Supervised) | 31.79 M | `24.14` | `0.830` |
| **HybridMIRNet (Ours)** | **Proposed Architecture** | **32.55 M** | **`24.50+`** | **`0.85+`** |

---

## 6. References & Licensing

- **MIRNet Backbone:** S. W. Zamir et al., *"Learning Enriched Features for Real Image Restoration and Enhancement"*, ECCV 2020.
- **Curve Formulation:** C. Guo et al., *"Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"*, CVPR 2020.
- **LOL Dataset:** C. Wei et al., *"Deep Retinex Decomposition for Low-Light Enhancement"*, BMVC 2018.
- **License:** Distributed under the MIT License (see `LICENSE`).
