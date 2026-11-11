"""Generator for Phase 11 Google Colab Notebook:
notebooks/Complete_Project_Implementation.ipynb
"""

import ast
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT_DIR, "notebooks", "Complete_Project_Implementation.ipynb")


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.rstrip("\n").split("\n")],
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            """# Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion

**Team:** Vijendra, Vyom, Gautam  
**Target Deadline:** 17 November 2026  
**Base Architecture:** MIRNet (Zamir et al., ECCV 2020) ([`model.py`](model.py), [`antialias_downsample.py`](antialias_downsample.py), MIT License)

This notebook executes the complete end-to-end pipeline sequentially:
1. Environment setup
2. Dependency installation
3. Dataset preparation (LOL dataset)
4. Model architecture verification (Configs A, B, C, D)
5. Training
6. Validation
7. Checkpoint loading
8. Image enhancement
9. Metric calculation (PSNR, SSIM, LPIPS, NIQE, latency)
10. Ablation study
11. Research comparison
12. Graph generation
13. Result visualization"""
        ),
        md_cell("## 1. Environment Setup"),
        code_cell(
            """import os
import sys
import subprocess

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

IN_COLAB = "google.colab" in sys.modules
print("Running in Google Colab:", IN_COLAB)

# Locate project root automatically when launched from notebooks/ or repo root
cwd = os.path.abspath(".")
if os.path.isfile(os.path.join(cwd, "models", "hybrid_mirnet.py")):
    PROJECT_ROOT = cwd
elif os.path.isfile(os.path.join(os.path.dirname(cwd), "models", "hybrid_mirnet.py")):
    PROJECT_ROOT = os.path.dirname(cwd)
    os.chdir(PROJECT_ROOT)
else:
    # In Google Colab, mount Drive and point to your uploaded project directory if needed:
    # from google.colab import drive
    # drive.mount("/content/drive")
    # PROJECT_ROOT = "/content/drive/MyDrive/sip_project_submission"
    # os.chdir(PROJECT_ROOT)
    PROJECT_ROOT = cwd

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Project root:", PROJECT_ROOT)
subprocess.run(["nvidia-smi"], check=False)"""
        ),
        md_cell("## 2. Dependency Installation"),
        code_cell(
            """import importlib

REQUIRED = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("tensorboard", "tensorboard"),
    ("skimage", "scikit-image"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("lpips", "lpips"),
    ("pyiqa", "pyiqa"),
    ("gdown", "gdown"),
]

missing_pkgs = []
for mod_name, pip_name in REQUIRED:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        missing_pkgs.append(pip_name)

if missing_pkgs and IN_COLAB:
    print("Installing missing packages in Colab:", missing_pkgs)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_pkgs])
elif missing_pkgs:
    print("Optional packages not installed locally (skipped outside Colab):", missing_pkgs)
else:
    print("All dependencies are installed.")

import torch
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))"""
        ),
        md_cell("## 3. Dataset Preparation (LOL Dataset)"),
        code_cell(
            """import zipfile
import glob

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "LOL")

def _find_lol_splits(base_dir):
    our485_hits = glob.glob(os.path.join(base_dir, "**", "our485"), recursive=True)
    eval15_hits = glob.glob(os.path.join(base_dir, "**", "eval15"), recursive=True)
    if our485_hits and eval15_hits:
        return (
            os.path.join(our485_hits[0], "low"),
            os.path.join(our485_hits[0], "high"),
            os.path.join(eval15_hits[0], "low"),
            os.path.join(eval15_hits[0], "high"),
        )
    return None

splits = _find_lol_splits(DATA_DIR)
LOL_GDRIVE_ID = "157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB"

if splits is None and IN_COLAB:
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "LOLdataset.zip")
    import gdown
    gdown.download(id=LOL_GDRIVE_ID, output=zip_path, quiet=False)
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print("Extracted LOL dataset into:", DATA_DIR)
    splits = _find_lol_splits(DATA_DIR)

if splits is not None:
    TRAIN_LOW, TRAIN_HIGH, VAL_LOW, VAL_HIGH = splits
    USING_FULL_LOL = True
else:
    print("[Notice] Full LOL dataset not found; using bundled 7 sample pairs for verification.")
    TRAIN_LOW = os.path.join(PROJECT_ROOT, "baseline_results", "samples", "low")
    TRAIN_HIGH = os.path.join(PROJECT_ROOT, "baseline_results", "samples", "high")
    VAL_LOW = TRAIN_LOW
    VAL_HIGH = TRAIN_HIGH
    USING_FULL_LOL = False

print("Train pairs:", len(glob.glob(os.path.join(TRAIN_LOW, "*"))))
print("Valid pairs:", len(glob.glob(os.path.join(VAL_LOW, "*"))))"""
        ),
        md_cell("## 4. Model Architecture Verification (Configs A, B, C, D)"),
        code_cell(
            """from models.hybrid_mirnet import CONFIG_DESCRIPTIONS, build_ablation_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy = torch.rand(1, 3, 64, 64, device=DEVICE)

for cfg in ("A", "B", "C", "D"):
    m = build_ablation_model(cfg).to(DEVICE).eval()
    params = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        out = m(dummy, return_dict=True)
    print(f"Config {cfg} ({CONFIG_DESCRIPTIONS[cfg]}): {params:,} params | output={tuple(out['enhanced'].shape)}")"""
        ),
        md_cell("## 5. Training (Hybrid MIRNet & Ablation Configurations)"),
        code_cell(
            """from torch.utils.data import DataLoader
from dataset import LoLDataset, PairedRandomPatchDataset
from losses import HybridEnhancementLoss
from train_hybrid import set_seed, train_one_epoch, validate_epoch, save_training_curves
import pandas as pd

SEED = 42
set_seed(SEED)

# Full Colab GPU run uses 50 epochs on LOL; local/CPU verification runs a 1-epoch 2-step pass
EPOCHS = 50 if (USING_FULL_LOL and DEVICE.type == "cuda") else 1
MAX_STEPS = None if (USING_FULL_LOL and DEVICE.type == "cuda") else 2
BATCH_SIZE = 4 if DEVICE.type == "cuda" else 2
PATCH_SIZE = 128 if DEVICE.type == "cuda" else 64
CONFIGS_TO_TRAIN = ["A", "B", "C", "D"] if (USING_FULL_LOL and DEVICE.type == "cuda") else ["A", "D"]

train_ds = PairedRandomPatchDataset(TRAIN_LOW, TRAIN_HIGH, patch_size=PATCH_SIZE, augment=True)
val_ds = LoLDataset(
    VAL_LOW,
    VAL_HIGH,
    img_size=None if USING_FULL_LOL else 64,
    mode="full" if USING_FULL_LOL else "resize",
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2 if DEVICE.type == "cuda" else 0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments_notebook")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

for cfg in CONFIGS_TO_TRAIN:
    print(f"\\n=== Training Config {cfg}: {CONFIG_DESCRIPTIONS[cfg]} ===")
    cfg_dir = os.path.join(EXPERIMENTS_DIR, f"config_{cfg}")
    ckpt_dir = os.path.join(cfg_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = build_ablation_model(cfg).to(DEVICE)
    criterion = HybridEnhancementLoss(
        l1_weight=1.0, ssim_weight=0.25, color_weight=0.10, smooth_weight=0.05, perceptual_weight=0.0
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    history = []
    best_psnr = -1.0
    for ep in range(EPOCHS):
        tr = train_one_epoch(train_loader, model, criterion, optimizer, DEVICE, scaler=scaler, max_steps=MAX_STEPS)
        va = validate_epoch(val_loader, model, criterion, DEVICE, max_steps=MAX_STEPS)
        scheduler.step()
        history.append({"epoch": ep + 1, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        print(f"Epoch {ep+1}/{EPOCHS} | Train Loss={tr['total']:.4f}, PSNR={tr['psnr']:.2f} | Val PSNR={va['psnr']:.2f}, SSIM={va['ssim']:.4f}")

        state = {
            "config": cfg,
            "epoch": ep,
            "best_psnr": max(best_psnr, va["psnr"]),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        }
        torch.save(state, os.path.join(ckpt_dir, "latest_checkpoint.pth"))
        if va["psnr"] > best_psnr:
            best_psnr = va["psnr"]
            torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(cfg_dir, "training_history.csv"), index=False)
    save_training_curves(hist_df, os.path.join(cfg_dir, "loss_psnr_curves.png"))"""
        ),
        md_cell("## 6. Validation"),
        code_cell(
            """best_d_ckpt = os.path.join(EXPERIMENTS_DIR, "config_D", "checkpoints", "best_model.pth")
val_model = build_ablation_model("D").to(DEVICE)
ckpt_val = torch.load(best_d_ckpt, map_location=DEVICE)
val_model.load_state_dict(ckpt_val["model_state_dict"])
val_criterion = HybridEnhancementLoss().to(DEVICE)
val_metrics = validate_epoch(val_loader, val_model, val_criterion, DEVICE, max_steps=MAX_STEPS)
print("Config D validation metrics:", val_metrics)"""
        ),
        md_cell("## 7. Checkpoint Loading"),
        code_cell(
            """loaded_model = build_ablation_model("D").to(DEVICE).eval()
ckpt = torch.load(best_d_ckpt, map_location=DEVICE)
loaded_model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded Config D checkpoint from epoch {ckpt['epoch'] + 1} (best_psnr={ckpt['best_psnr']:.4f} dB)")"""
        ),
        md_cell("## 8. Image Enhancement"),
        code_cell(
            """import numpy as np
from PIL import Image
from dataset import _list_image_files
from utils import save_img

ENHANCED_DIR = os.path.join(EXPERIMENTS_DIR, "enhanced_samples")
os.makedirs(ENHANCED_DIR, exist_ok=True)

val_low_files = _list_image_files(VAL_LOW)
val_high_files = _list_image_files(VAL_HIGH)

enhanced_records = []
for lp, hp in zip(val_low_files, val_high_files):
    arr = np.array(Image.open(lp).convert("RGB"), dtype=np.float32) / 255.0
    inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_dict = loaded_model(inp, return_dict=True)
    enh = torch.clamp(out_dict["enhanced"].squeeze(0).cpu(), 0.0, 1.0)
    save_path = os.path.join(ENHANCED_DIR, os.path.basename(lp))
    save_img(enh, save_path)
    enhanced_records.append((lp, save_path, hp, out_dict))

print(f"Enhanced {len(enhanced_records)} images -> {ENHANCED_DIR}")"""
        ),
        md_cell("## 9. Metric Calculation (PSNR, SSIM, LPIPS, NIQE)"),
        code_cell(
            """from utils import calc_PSNR, calc_SSIM
from scripts.run_ablation import _init_optional_metrics

lpips_fn, lpips_status, niqe_fn, niqe_status = _init_optional_metrics(DEVICE)
print("LPIPS:", lpips_status, "| NIQE:", niqe_status)

metric_rows = []
for lp, ep, hp, _ in enhanced_records:
    enh_t = torch.from_numpy(np.array(Image.open(ep).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    gt_t = torch.from_numpy(np.array(Image.open(hp).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    lpips_v = float(lpips_fn(enh_t.unsqueeze(0).to(DEVICE)*2-1, gt_t.unsqueeze(0).to(DEVICE)*2-1).item()) if lpips_fn else None
    niqe_v = float(niqe_fn(enh_t.unsqueeze(0).to(DEVICE)).item()) if niqe_fn else None
    metric_rows.append({
        "image": os.path.basename(lp),
        "psnr_db": calc_PSNR(gt_t, enh_t),
        "ssim": calc_SSIM(gt_t, enh_t),
        "lpips": lpips_v,
        "niqe": niqe_v,
    })

metrics_df = pd.DataFrame(metric_rows)
print(metrics_df.to_string(index=False))"""
        ),
        md_cell("## 10. Ablation Study (Configurations A, B, C, D)"),
        code_cell(
            """ablation_cmd = [
    sys.executable,
    os.path.join(PROJECT_ROOT, "scripts", "run_ablation.py"),
    "--test_low", VAL_LOW,
    "--test_high", VAL_HIGH,
    "--experiments_dir", EXPERIMENTS_DIR,
]
subprocess.check_call(ablation_cmd)

ab_df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "ablation_results.csv"))
print(ab_df[["config", "status", "params", "psnr_db", "ssim", "lpips", "niqe", "infer_time_ms"]].to_string(index=False))"""
        ),
        md_cell("## 11. Research Comparison"),
        code_cell(
            """subprocess.check_call([
    sys.executable,
    os.path.join(PROJECT_ROOT, "scripts", "research_comparison.py"),
    "--experiments_dir", EXPERIMENTS_DIR,
])

comp_df = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "published_research_comparison.csv"))
print(comp_df[["method", "source_type", "params_m", "psnr_db", "ssim"]].to_string(index=False))"""
        ),
        md_cell("## 12. Graph Generation"),
        code_cell(
            """from IPython.display import display

for plot_name in ["ablation_metrics_comparison.png", "ablation_efficiency_tradeoff.png", "research_comparison_chart.png"]:
    p = os.path.join(EXPERIMENTS_DIR, plot_name)
    if os.path.isfile(p):
        display(Image.open(p))

curve_p = os.path.join(EXPERIMENTS_DIR, "config_D", "loss_psnr_curves.png")
if os.path.isfile(curve_p):
    display(Image.open(curve_p))"""
        ),
        md_cell("## 13. Result Visualization"),
        code_cell(
            """VIS_DIR = os.path.join(EXPERIMENTS_DIR, "visualizations")
subprocess.check_call([
    sys.executable,
    os.path.join(PROJECT_ROOT, "scripts", "generate_visualizations.py"),
    "--test_low", VAL_LOW,
    "--test_high", VAL_HIGH,
    "--experiments_dir", EXPERIMENTS_DIR,
    "--out_dir", VIS_DIR,
    "--ckpt_D", best_d_ckpt,
])

for vis_file in ["difficult_cases_grid.png", "hybrid_internals_sample_1.png"]:
    vp = os.path.join(VIS_DIR, vis_file)
    if os.path.isfile(vp):
        display(Image.open(vp))"""
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "name": "Complete_Project_Implementation.ipynb",
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main():
    os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
    nb = build_notebook()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            ast.parse(src)
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated and AST-verified {NB_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
