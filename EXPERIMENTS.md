# Experiments, Ablation Study & Research Comparison

---

## 1. Dataset Details

- **Target Benchmark Dataset:** **LOL (LOw-Light)** dataset (Wei et al., BMVC 2018), consisting of **485** paired low-light / normal-light training images (`our485`) and **15** evaluation pairs (`eval15`) at `600×400` resolution.
- **Local Verification Split:** Because the full LOL dataset and pretrained `.pth` weights are not distributed in this repository, local verification uses the **7** paired `184×276` samples extracted cleanly from [`Results/result1.png`](Results/result1.png)–[`Results/result7.png`](Results/result7.png) (stored in [`baseline_results/samples/`](baseline_results/samples)).

---

## 2. Training Configuration ([`train_hybrid.py`](train_hybrid.py))

| Hyperparameter | Full LOL (Colab GPU) | Local Smoke Test |
| :--- | :--- | :--- |
| Optimizer | `AdamW` (`lr=2e-4`, `wd=1e-4`) | `AdamW` (`lr=2e-4`) |
| Scheduler | `CosineAnnealingLR` (`eta_min=1e-6`) | `CosineAnnealingLR` |
| Patch Size & Augmentation | `128×128` random crop + H/V flips | `64×64` random crop + flips |
| Loss Function | `1.0·L1 + 0.25·SSIM + 0.10·Color + 0.05·Smooth` | Same |
| Precision & Clipping | CUDA AMP (`--amp`), grad norm `1.0` | FP32 CPU |
| Seed | `42` | `42` |

---

## 3. Baseline Verification Results (Phase 2)

Environment verified: Python `3.13.9`, `torch 2.12.1+cpu` (`cuda_available=False`; RTX 3050 6GB GPU present on host without CUDA PyTorch wheel).

| Source | Condition | Mean PSNR (dB) | Mean SSIM | Notes |
| :--- | :--- | :---: | :---: | :--- |
| Raw Low-Light Input | 7 extracted `Results/` pairs | `6.80` | `0.1507` | Baseline input quality |
| Untrained `MIRNet` (`model.py`) | Random init (`seed=42`, no repo `.pth`) | `6.69` | `0.1378` | `32,212,944` params; `4149.59 ms` CPU (`184×276`) |
| Upstream Author Panels | Cropped middle panels of `Results/result1..7.png` | `23.32` | `0.8877` | Upstream README claims `22.97 dB` on full `eval15` |

---

## 4. Ablation Study (Phase 7 — Configs A, B, C, D)

Generated automatically via [`scripts/run_ablation.py`](scripts/run_ablation.py) on the 7-sample verification split (`184×276`, CPU, `seed=42`):

| Config | Description | Status | Params | PSNR (dB) | SSIM | LPIPS | NIQE | Infer Time (ms) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Illumination branch only | `untrained_verification` | `74,317` | `7.29` | `0.2269` | N/A | N/A | `62.06` |
| **B** | Original MIRNet only | `untrained_verification` | `32,212,944` | `6.69` | `0.1378` | N/A | N/A | `4041.09` |
| **C** | MIRNet + Illumination (Sequential) | `untrained_verification` | `32,287,261` | `8.12` | `0.3358` | N/A | N/A | `4007.76` |
| **D** | MIRNet + Illumination + Adaptive Fusion | `untrained_verification` | `32,548,771` | `7.82` | `0.3031` | N/A | N/A | `4310.55` |

> **Honesty & Reproducibility Note:** Locally, `lpips` and `pyiqa` (NIQE) are not installed and no full-LOL checkpoints exist, so `status` is explicitly marked `untrained_verification` and `LPIPS`/`NIQE` are recorded as `N/A`. Running [`notebooks/Complete_Project_Implementation.ipynb`](notebooks/Complete_Project_Implementation.ipynb) on Google Colab GPU trains all four configurations on `our485`, installs `lpips` + `pyiqa`, and populates full trained `eval15` metrics directly into CSVs and plots.

Charts & CSV artifacts:
- [`experiments/ablation_results.csv`](experiments/ablation_results.csv)
- [`experiments/ablation_per_image.csv`](experiments/ablation_per_image.csv)
- [`experiments/ablation_metrics_comparison.png`](experiments/ablation_metrics_comparison.png)
- [`experiments/ablation_efficiency_tradeoff.png`](experiments/ablation_efficiency_tradeoff.png)

---

## 5. Published Research Comparison (Phase 8)

Published results on **LOL (`eval15`)** from peer-reviewed literature are kept strictly separated from local verification measurements ([`experiments/published_research_comparison.csv`](experiments/published_research_comparison.csv)):

| Method | Reference | Training Paradigm | Params (M) | Published PSNR (dB) | Published SSIM |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **RetinexNet** | Wei et al., BMVC 2018 | Paired Retinex Decomposition | `0.84` | `16.77` | `0.560` |
| **EnlightenGAN** | Jiang et al., IEEE TIP 2021 | Unpaired GAN | `8.64` | `17.48` | `0.651` |
| **Zero-DCE** | Guo et al., CVPR 2020 | Zero-Reference Curve | `0.079` | `14.86` | `0.559` |
| **Zero-DCE++** | Li et al., IEEE TPAMI 2021 | Zero-Reference Lightweight Curve | `0.011` | `14.86` | `0.550` |
| **MIRNet (Paper)** | Zamir et al., ECCV 2020 | Paired Supervised Multi-Scale | `31.79` | `24.14` | `0.830` |
| **MIRNet (Baseline Reference)** | Reference PyTorch Implementation (2021) | Paired Supervised (64×64 setup) | `32.21` | `22.97` | — |

---

## 6. Visualizations (Phase 9)

Generated in [`Results/comparisons/`](Results/comparisons):
- `comparison_sample_1.png`–`comparison_sample_7.png`: 5-panel comparisons (`Low-Light Input | MIRNet (B) | Illumination Only (A) | Hybrid MIRNet (D) | Ground Truth`).
- [`Results/comparisons/difficult_cases_grid.png`](Results/comparisons/difficult_cases_grid.png): Focuses on the darkest input scenes (`sample_7.png`, `sample_3.png`, `sample_5.png`).
- [`Results/comparisons/hybrid_internals_sample_1.png`](Results/comparisons/hybrid_internals_sample_1.png): Displays estimated illumination map $L(x)$ and spatially varying fusion weights.

---

## 7. Limitations

1. **Compute & Dataset Availability Locally:** Local PyTorch (`2.12.1+cpu`) lacks CUDA acceleration and full LOL (`our485`/`eval15`) training images are not stored locally; full multi-epoch convergence must be executed on Google Colab T4 GPU (or after installing CUDA PyTorch locally).
2. **MIRNet Footprint:** Preserving the original `MIRNet` architecture (`32.21M` parameters, plus `[DAU(...)] * width` weight-sharing quirk in [`model.py`](model.py#L324)) keeps parameter count at `32.55M`, yielding `~4.3s` CPU inference latency on `184×276` images (though GPU latency is interactive).
3. **Optional Perceptual/No-Reference Metrics:** `LPIPS` and `NIQE` require `lpips` / `pyiqa` downloads, configured automatically in the Colab notebook.

---

## 8. References

1. Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H., & Shao, L. (2020). *Learning Enriched Features for Real Image Restoration and Enhancement (MIRNet)*. ECCV 2020. arXiv:2003.06792.
2. Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., & Cong, R. (2020). *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (Zero-DCE)*. CVPR 2020.
3. Li, C., Guo, C., & Loy, C. C. (2021). *Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation (Zero-DCE++)*. IEEE TPAMI.
4. Wei, C., Wang, W., Yang, W., & Liu, J. (2018). *Deep Retinex Decomposition for Low-Light Enhancement (RetinexNet)*. BMVC 2018.
5. Jiang, Y., Gong, X., Liu, D., Cheng, Y., Fang, C., Shen, X., Yang, J., Zhou, P., & Wang, Z. (2021). *EnlightenGAN: Deep Light Enhancement Without Paired Supervision*. IEEE TIP.

