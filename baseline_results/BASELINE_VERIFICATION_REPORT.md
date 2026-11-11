# Phase 2: Baseline Verification Report

## 1. Environment & Dependency Status
- **Python:** `3.13.9`
- **PyTorch:** `2.12.1+cpu` (`cuda_available=False`)
- **TorchVision:** `0.27.1+cpu`
- **Missing dependencies:** `[]`

## 2. Dataset & Checkpoint Verification
- **Full LOL dataset present locally:** `False`
- **Checked dataset paths:** `['D:\\sip project submission\\data\\LOL', 'D:\\sip project submission\\data\\Loldataset', '/content/Data/Loldataset']`
- **Pretrained checkpoints in repo:** `[]`
- **Note:** No pretrained .pth/.pt checkpoint exists in the upstream repository.
- **Sample dataset check (`LoLDataset`):** Verified loading 7 paired `184x276` samples extracted from `Results/result1.png`–`result7.png` into `baseline_results/samples/`.

## 3. Original MIRNet Model Verification (`model.py` unmodified)
- **Total parameters:** `32,212,944`
- **Trainable parameters:** `32,212,944`
- **Inference time (`1x3x64x64`, cpu):** `434.07 ± 10.17 ms`
- **Inference time (`1x3x184x276`, cpu):** `4149.59 ± 152.37 ms`

## 4. Sample Quantitative Verification (7 Extracted `Results/result*.png` Pairs)

> **Honesty Notice:** Because no pretrained `.pth` checkpoint is distributed in the base repository, local `MIRNet()` execution uses random initialization (`seed=42`). Separately, the baseline visualization middle panels (`Results/result1..7.png`) are measured against the right ground-truth panels for transparent reference.

- **Raw Low-Light Input vs GT (mean over 7 samples):** PSNR = `6.80 dB`, SSIM = `0.1507`
- **Local Untrained `MIRNet` (random weights) vs GT:** PSNR = `6.69 dB`, SSIM = `0.1378`
- **Upstream Author Published Panels (`Results/*.png`) vs GT:** PSNR = `23.32 dB`, SSIM = `0.8877` (upstream README reports `22.97 dB` across the full 15-image LOL eval set)

## 5. Baseline Training Workflow Verification (Smoke Test)
- **Exit code:** `0`
- **Elapsed time (1 epoch, 2 steps, 64x64 crops on cpu):** `24.13 s`
- **Checkpoint saved & reloaded into `MIRNet`:** `True`
- **TensorBoard logs created:** `True`

## 6. Full Baseline Training Requirements
- **Dataset:** LOL (`our485` training: 485 paired images; `eval15` validation: 15 paired images, `600x400`).
- **Compute:** `MIRNet` has 32.21M parameters; full 100-epoch training on CPU is impractical. Recommended execution is Google Colab T4 GPU (or local RTX 3050 6GB after installing CUDA-enabled PyTorch).
