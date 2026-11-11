# Phase 1: Repository Analysis

**Project Team:** Vijendra, Vyom, Gautam  
**Submission Repository:** [devillikevd/sip](https://github.com/devillikevd/sip) (MIT License, [`LICENSE`](../LICENSE))  
**Target Project Title:** *Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion*  
**Analysis Date:** 2026-11-10

---

## 1. Repository Inventory

No files were modified or deleted prior to completing this analysis. All files in the cloned repository (`main` branch, commit `3c13a10a16eab8183b8fbd0c063d9815b662259a`) are listed below:

| File / Directory | Size | Lines | Role in Original Repository |
| :--- | :---: | :---: | :--- |
| `model.py` | 14,904 B | 457 | PyTorch implementation of MIRNet and all building blocks |
| `antialias_downsample.py` | 2,360 B | 64 | Anti-aliased downsampling layer (`DownSample`) used inside `model.py` |
| `dataset.py` | 1,116 B | 32 | PyTorch `Dataset` loader (`LoLDataset`) for paired low/normal images |
| `train_enhancement.py` | 5,733 B | 154 | Training & validation loop, Charbonnier loss (`Loss_fn`), TensorBoard logging |
| `utils.py` | 1,263 B | 56 | `MixUp` augmentation, PSNR helpers (`calc_PSNR`, `batch_psnr`), OpenCV I/O helpers |
| `README.md` | 1,155 B | 24 | Project summary, LOL dataset link, and reported `22.97` PSNR result |
| `LICENSE` | 1,097 B | 21 | MIT License |
| `Results/result1.png`–`result7.png` | 222–292 KB each | — | Seven `851×201` RGBA side-by-side triptych visualizations (`Input \| Restored \| Ground Truth`) |

### Missing Artifacts in the Original Repository
- **Testing / Inference Script:** None exists (`model.py` only has an `if __name__ == '__main__':` parameter count snippet).
- **Configuration Files:** None exist (all paths and hyperparameters are hardcoded in `train_enhancement.py:main()`).
- **Pretrained Checkpoints:** No `.pth` / `.pt` weight files are present in the repository or tracked releases.
- **Dependency Specification:** No `requirements.txt`, `pyproject.toml`, or `environment.yml` exists.

---

## 2. MIRNet Architecture Analysis (`model.py` & `antialias_downsample.py`)

The existing network implements **MIRNet** (*Learning Enriched Features for Real Image Restoration and Enhancement*, Zamir et al., ECCV 2020) in PyTorch.

### 2.1 High-Level Dataflow
```text
Input (B, 3, H, W)
  └─► first_conv: Conv2d(3 -> 64, 3x3, pad=1, bias=False)
  └─► mir_blocks: Sequential(3 × RRG)
        └─ Each RRG: 2 × MSRB -> Conv2d(64 -> 64, 3x3, pad=1) + Residual Skip
  └─► final_conv: Conv2d(64 -> 3, 3x3, pad=1, bias=False)
  └─► Global Residual Addition: out + x  ──► Output (B, 3, H, W)
```

### 2.2 Building Blocks
1. **`MIRNet`** (`model.py:432-449`):
   - Default configuration: `in_c=3`, `out_c=3`, `num_features=64`, `kernel_size=3`, `stride=2`, `num_MSRB=2`, `num_RRG=3`, `height=3`, `width=2`, `bias=False`.
   - Measured parameter count (verified directly via PyTorch): **32,212,944** trainable parameters.
2. **`RRG` (Residual Recurrent Group)** (`model.py:416-429`):
   - Stacks `num_MSRB=2` Multi-Scale Residual Blocks (`MSRB`), followed by a `3×3` convolution and residual skip connection.
3. **`MSRB` (Multi-Scale Residual Block)** (`model.py:315-414`):
   - Operates across `height=3` parallel spatial scales:
     - Scale `0`: `H × W`, `64` channels
     - Scale `1`: `H/2 × W/2`, `128` channels
     - Scale `2`: `H/4 × W/4`, `256` channels
   - Contains `width=2` stages of Dual Attention Units (`DAU`) per scale, cross-scale feature exchange via `DownSamp`/`UpSample`, multi-scale fusion via `SKFF`, a `3×3` `out_conv`, and residual addition `output + x`.
4. **`DAU` (Dual Attention Unit)** (`model.py:108-152`):
   - `Conv3x3 -> PReLU -> Conv3x3`, followed by parallel:
     - **`SpatialAttention`** (`model.py:20-58`): `CompressChannels` (channel-wise max + mean pooling into 2 channels) → `Conv5x5` → `Sigmoid` gating.
     - **`ChannelAttention`** (`model.py:61-105`): Squeeze-and-Excitation with `AdaptiveAvgPool2d(1)` → `Conv1x1(C -> C//8)` → `PReLU` → `Conv1x1(C//8 -> C)` → `Sigmoid`.
   - Concatenates spatial and channel attention outputs (`2C`), projects back to `C` with `Conv1x1`, and adds the input residual.
5. **`SKFF` (Selective Kernel Feature Fusion)** (`model.py:276-312`):
   - Fuses `3` scale-aligned feature maps by summing them, applying global average pooling + `Conv1x1(C -> max(C//3, 4)) + PReLU`, generating `3` softmax-normalized channel-wise attention weights (`attention_fcs`), and taking a weighted sum.
6. **Multi-Scale Resampling** (`model.py:155-273`, `antialias_downsample.py:6-64`):
   - **`DownSampleBlock` / `DownSamp`**: Residual two-branch downsampling using [`DownSample`](../antialias_downsample.py#L6-L52) (anti-aliased 2D binomial filter `[1, 2, 1]` outer product with reflection padding and `stride=2` depthwise convolution) + `1×1` channel doubling (`C -> 2C`).
   - **`UpsampleBlock` / `UpSample`**: Residual two-branch upsampling using bilinear interpolation (`scale_factor=2`, `align_corners=False`) + `1×1` channel halving (`C -> C//2`).

### 2.3 Observed Implementation Quirks in `model.py`
1. **Unintended weight sharing across MSRB width** (`model.py:324`):
   `[DAU(int(num_features*stride**i))]*width` duplicates the same Python `DAU` object reference `width=2` times instead of instantiating distinct `DAU` modules per stage.
2. **Hardcoded debug print** (`model.py:298`):
   `SKFF.forward()` executes `print(in_features.shape)` on every call (24 stdout lines per forward pass).
3. **Spatial divisibility constraint**:
   Because `height=3` downsamples by `4×`, input height and width must be multiples of `4`.
4. **Hardcoded 3-stream `SKFF`**:
   `SKFF.__init__` builds `3` attention heads (`for i in range(3)`), tying `SKFF` to `height=3`.

---

## 3. Data, Training, Metrics, and Dependencies

### 3.1 Dataset Loader (`dataset.py`)
- **`LoLDataset`** (`dataset.py:8-31`) reads all files in `low_light_root` and `target_root`.
- Applies `T.CenterCrop((img_size, img_size))` (default `img_size=64`), `T.ToTensor()`, and `T.Normalize([0.0,0.0,0.0], [1.0,1.0,1.0])` (identity scaling in `[0, 1]`).
- Note: `README.md` states *"All the images were resized to 64x64 during training"*, whereas the code actually performs a `64×64` center crop rather than resizing.
- Sets `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"` at line 6 to avoid OpenMP duplicate DLL crashes on Windows when multiple runtimes load.

### 3.2 Loss Function & Optimization (`train_enhancement.py`)
- **Loss (`Loss_fn`, lines 14–25)**: Charbonnier loss:
  $$\mathcal{L}_{\text{char}}(\hat{y}, y) = \frac{1}{N} \sum \sqrt{(\hat{y} - y)^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$
- **Optimizer**: `Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-8, eps=1e-8)`.
- **Scheduler**: `CosineAnnealingLR(opt, T_max=100, eta_min=5e-5, verbose=True)`.
- **DataLoaders**: `batch_size=4, shuffle=True, num_workers=2` (train); `batch_size=3, shuffle=False` (validation); `100` epochs.

### 3.3 Evaluation Metrics (`utils.py`)
- **`calc_PSNR(target, pred)`** (`utils.py:25-30`): Clamps tensors to `[0, 1]`, computes RMSE and `20 * log10(1 / rmse)`.
- **`batch_psnr(img1, img2)`** (`utils.py:33-41`): Intended to average per-sample PSNR across a batch.
- **Missing metrics**: SSIM, LPIPS, and NIQE are not implemented.

### 3.4 Existing Dependencies
Observed environment requirements from imports across all `.py` files:
- `torch`, `torchvision`
- `numpy`, `opencv-python` (`cv2`), `Pillow` (`PIL`)
- `tqdm`, `tensorboard`

---

## 4. Original Execution Workflow & Blocking Bugs

### 4.1 Intended Execution Flow (`train_enhancement.py:main()`)
1. Point hardcoded Google Colab dataset paths (`/content/Data/Loldataset/our485/...` and `eval15/...`) and Google Drive save paths (`/content/drive/MyDrive/MIRNet/Enhance`) to the LOL dataset.
2. Instantiate `MIRNet().to('cuda')`, `Loss_fn()`, `Adam`, `CosineAnnealingLR`, and `SummaryWriter`.
3. Create `LoLDataset` instances (`64×64` center crops) and `DataLoader` objects.
4. For 100 epochs: save a sample preview image, run `train()`, step scheduler, save `Mirnet_enhance.pth`, and run `validate()`.

### 4.2 Verified Defects Preventing Out-of-the-Box Execution
1. **Missing `tensorboard` dependency** in the local environment (`ModuleNotFoundError: No module named 'tensorboard'`).
2. **Hardcoded Google Colab / Drive paths** (`/content/Data/...`, `/content/drive/MyDrive/...`) and missing directory separator in `SAVE_DIR + 'Mirnet_enhance.pth'` and `SAVE_DIR + f'img{epoch}.png'`.
3. **Hardcoded CUDA device** (`torch.device('cuda')` in `train_enhancement.py` and `.cuda()` in `utils.py:MixUp.augment`), failing on CPU-only PyTorch environments.
4. **`CosineAnnealingLR(..., verbose=True)`** raises `TypeError` on PyTorch `2.12+` (deprecated in 2.2 and removed in newer releases).
5. **`NameError` in `train()` and `validate()`**: `train_enhancement.py` imports `from utils import *` (`line 11`) but invokes `utils.batch_psnr(...)` (`lines 48, 85`).
6. **`AttributeError` with `MixUp`**: `train()` declares `mixup = True` (`bool`) and calls `mixup.augment(...)` when `epoch > 5`.
7. **`model.eval()` inside training loop** (`train_enhancement.py:54`): Switches the model out of training mode after the first batch of every epoch.
8. **Bug in `utils.batch_psnr()`** (`utils.py:36-38`): Loops `for i, j in zip(img1, img2):` but calls `calc_PSNR(img1, img2)` instead of `calc_PSNR(i, j)`.
9. **Unsorted directory listing in `LoLDataset`** (`dataset.py:13-14`): `os.listdir(...)` without sorting or image-extension filtering can misalign low-light and target pairs.
10. **Mismatched validation preview target & unclosed file handles** (`train_enhancement.py:108-109, 139, 147`): Compares `eval15/low/111.png` against `our485/high/100.png` and uses unclosed `open(..., 'wb')` handles.
11. **OpenMP DLL conflict on Windows**: Running `import torch` without setting `KMP_DUPLICATE_LIB_OK=TRUE` beforehand fails with `OMP: Error #15` in the local Conda environment.

