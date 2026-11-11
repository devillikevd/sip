"""Phase 2: Baseline Verification Script for MIRNet Low-Light Enhancement.

Verifies dependencies, dataset paths, checkpoint availability, unmodified MIRNet
loading, parameter count, CPU/GPU inference timing, sample PSNR/SSIM metrics,
and the baseline training/checkpoint workflow without fabricating results.
"""

import contextlib
import glob
import io
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
from PIL import Image
import torch

from dataset import LoLDataset
from model import MIRNet
from utils import batch_psnr, batch_ssim, calc_PSNR, calc_SSIM, save_img


BASELINE_DIR = os.path.join(ROOT_DIR, "baseline_results")
SAMPLES_LOW_DIR = os.path.join(BASELINE_DIR, "samples", "low")
SAMPLES_HIGH_DIR = os.path.join(BASELINE_DIR, "samples", "high")
SAMPLES_AUTHOR_DIR = os.path.join(BASELINE_DIR, "samples", "author_published")
UNTRAINED_OUT_DIR = os.path.join(BASELINE_DIR, "untrained_outputs")
SMOKE_TEST_DIR = os.path.join(BASELINE_DIR, "smoke_test_run")


def check_dependencies():
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "PIL",
        "tqdm",
        "tensorboard",
        "skimage",
        "scipy",
        "matplotlib",
        "pandas",
        "streamlit",
    ]
    versions = {}
    missing = []
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError as exc:
            missing.append(f"{pkg}: {exc}")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": versions,
        "missing": missing,
    }


def check_checkpoints():
    patterns = ("*.pth", "*.pt", "*.ckpt", "*.safetensors")
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(ROOT_DIR, "**", pat), recursive=True))
    found = [
        os.path.relpath(p, ROOT_DIR)
        for p in sorted(set(found))
        if "smoke_test_run" not in p
    ]
    return {
        "pretrained_checkpoints_found": found,
        "has_pretrained_checkpoint": len(found) > 0,
        "note": (
            "Pretrained checkpoint(s) found."
            if found
            else "No pretrained .pth/.pt checkpoint exists in the upstream repository."
        ),
    }


def check_lol_dataset_paths():
    candidates = [
        os.path.join(ROOT_DIR, "data", "LOL"),
        os.path.join(ROOT_DIR, "data", "Loldataset"),
        "/content/Data/Loldataset",
    ]
    found = [p for p in candidates if os.path.isdir(p)]
    return {
        "checked_candidates": candidates,
        "existing_lol_dirs": found,
        "full_lol_dataset_present": len(found) > 0,
    }


def extract_sample_triptychs():
    """Extract clean 184x276 panels (divisible by 4) from Results/result1..7.png.

    Measured subplot bounding boxes in the upstream 851x201 figures:
      - rows: 8:192 (184 px)
      - left panel (low-light input): cols 9:285 (276 px)
      - middle panel (author MIRNet output): cols 288:564 (276 px)
      - right panel (ground truth): cols 567:843 (276 px)
    """
    os.makedirs(SAMPLES_LOW_DIR, exist_ok=True)
    os.makedirs(SAMPLES_HIGH_DIR, exist_ok=True)
    os.makedirs(SAMPLES_AUTHOR_DIR, exist_ok=True)

    result_files = sorted(glob.glob(os.path.join(ROOT_DIR, "Results", "result*.png")))
    extracted = []
    for idx, path in enumerate(result_files, start=1):
        arr = np.array(Image.open(path).convert("RGB"))
        low = arr[8:192, 9:285]
        author = arr[8:192, 288:564]
        high = arr[8:192, 567:843]

        name = f"sample_{idx}.png"
        low_path = os.path.join(SAMPLES_LOW_DIR, name)
        high_path = os.path.join(SAMPLES_HIGH_DIR, name)
        author_path = os.path.join(SAMPLES_AUTHOR_DIR, name)

        Image.fromarray(low).save(low_path)
        Image.fromarray(high).save(high_path)
        Image.fromarray(author).save(author_path)

        extracted.append(
            {
                "sample_name": name,
                "source_figure": os.path.relpath(path, ROOT_DIR),
                "low_path": low_path,
                "high_path": high_path,
                "author_path": author_path,
                "shape_hwc": list(low.shape),
            }
        )
    return extracted


def benchmark_model(model, device, height, width, warmup=2, runs=5):
    x = torch.randn(1, 3, height, width, device=device)
    model.eval()
    times_ms = []
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.perf_counter()
            y = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
    return {
        "input_shape": [1, 3, height, width],
        "output_shape": list(y.shape),
        "runs": runs,
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }


def _to_tensor_01(img_path):
    arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def run_sample_inference(model, device, samples):
    os.makedirs(UNTRAINED_OUT_DIR, exist_ok=True)
    model.eval()
    rows = []

    for item in samples:
        low_t = _to_tensor_01(item["low_path"])
        high_t = _to_tensor_01(item["high_path"])
        author_t = _to_tensor_01(item["author_path"])

        inp = low_t.unsqueeze(0).to(device)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            pred = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        pred_clamped = torch.clamp(pred.squeeze(0).cpu(), 0.0, 1.0)
        out_path = os.path.join(UNTRAINED_OUT_DIR, item["sample_name"])
        save_img(pred_clamped, out_path)

        rows.append(
            {
                "sample": item["sample_name"],
                "source_figure": item["source_figure"],
                "resolution": f"{low_t.shape[1]}x{low_t.shape[2]}",
                "raw_input_psnr_db": calc_PSNR(high_t, low_t),
                "raw_input_ssim": calc_SSIM(high_t, low_t),
                "untrained_mirnet_psnr_db": calc_PSNR(high_t, pred_clamped),
                "untrained_mirnet_ssim": calc_SSIM(high_t, pred_clamped),
                "author_panel_psnr_db": calc_PSNR(high_t, author_t),
                "author_panel_ssim": calc_SSIM(high_t, author_t),
                "untrained_mirnet_infer_ms": (t1 - t0) * 1000.0,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(BASELINE_DIR, "baseline_metrics.csv")
    df.to_csv(csv_path, index=False)
    return df, csv_path


def run_training_smoke_test():
    ckpt_dir = os.path.join(SMOKE_TEST_DIR, "checkpoints")
    log_dir = os.path.join(SMOKE_TEST_DIR, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "train_enhancement.py"),
        "--train_low",
        SAMPLES_LOW_DIR,
        "--train_high",
        SAMPLES_HIGH_DIR,
        "--val_low",
        SAMPLES_LOW_DIR,
        "--val_high",
        SAMPLES_HIGH_DIR,
        "--save_dir",
        ckpt_dir,
        "--log_dir",
        log_dir,
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--val_batch_size",
        "2",
        "--img_size",
        "64",
        "--transform_mode",
        "crop",
        "--max_steps",
        "2",
        "--device",
        "auto",
        "--seed",
        "42",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - t0

    ckpt_file = os.path.join(ckpt_dir, "Mirnet_enhance.pth")
    best_file = os.path.join(ckpt_dir, "Mirnet_enhance_best.pth")
    ckpt_reloaded = False
    if os.path.isfile(ckpt_file):
        loaded = torch.load(ckpt_file, map_location="cpu")
        test_model = MIRNet()
        test_model.load_state_dict(loaded["model_state_dict"])
        ckpt_reloaded = True

    return {
        "returncode": proc.returncode,
        "elapsed_seconds": float(elapsed_s),
        "checkpoint_saved": os.path.isfile(ckpt_file),
        "best_checkpoint_saved": os.path.isfile(best_file),
        "checkpoint_reloaded_into_MIRNet": ckpt_reloaded,
        "tensorboard_event_files": glob.glob(os.path.join(log_dir, "events.out.tfevents.*")),
        "stdout_tail": proc.stdout[-1200:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
    }


def write_report(summary, df):
    mean_row = {
        "raw_input_psnr_db": df["raw_input_psnr_db"].mean(),
        "raw_input_ssim": df["raw_input_ssim"].mean(),
        "untrained_mirnet_psnr_db": df["untrained_mirnet_psnr_db"].mean(),
        "untrained_mirnet_ssim": df["untrained_mirnet_ssim"].mean(),
        "author_panel_psnr_db": df["author_panel_psnr_db"].mean(),
        "author_panel_ssim": df["author_panel_ssim"].mean(),
        "untrained_mirnet_infer_ms": df["untrained_mirnet_infer_ms"].mean(),
    }

    report_path = os.path.join(BASELINE_DIR, "BASELINE_VERIFICATION_REPORT.md")
    lines = [
        "# Phase 2: Baseline Verification Report",
        "",
        "## 1. Environment & Dependency Status",
        f"- **Python:** `{summary['environment']['python_version']}`",
        f"- **PyTorch:** `{summary['environment']['packages'].get('torch')}` (`cuda_available={summary['environment']['cuda_available']}`)",
        f"- **TorchVision:** `{summary['environment']['packages'].get('torchvision')}`",
        f"- **Missing dependencies:** `{summary['environment']['missing']}`",
        "",
        "## 2. Dataset & Checkpoint Verification",
        f"- **Full LOL dataset present locally:** `{summary['dataset_check']['full_lol_dataset_present']}`",
        f"- **Checked dataset paths:** `{summary['dataset_check']['checked_candidates']}`",
        f"- **Pretrained checkpoints in repo:** `{summary['checkpoint_check']['pretrained_checkpoints_found']}`",
        f"- **Note:** {summary['checkpoint_check']['note']}",
        "- **Sample dataset check (`LoLDataset`):** Verified loading 7 paired `184x276` samples extracted from `Results/result1.png`–`result7.png` into `baseline_results/samples/`.",
        "",
        "## 3. Original MIRNet Model Verification (`model.py` unmodified)",
        f"- **Total parameters:** `{summary['model']['total_params']:,}`",
        f"- **Trainable parameters:** `{summary['model']['trainable_params']:,}`",
        f"- **Inference time (`1x3x64x64`, {summary['device']}):** `{summary['benchmarks']['64x64']['mean_ms']:.2f} ± {summary['benchmarks']['64x64']['std_ms']:.2f} ms`",
        f"- **Inference time (`1x3x184x276`, {summary['device']}):** `{summary['benchmarks']['184x276']['mean_ms']:.2f} ± {summary['benchmarks']['184x276']['std_ms']:.2f} ms`",
        "",
        "## 4. Sample Quantitative Verification (7 Extracted `Results/result*.png` Pairs)",
        "",
        "> **Honesty Notice:** Because no pretrained `.pth` checkpoint is distributed in the base repository, local `MIRNet()` execution uses random initialization (`seed=42`). Separately, the baseline visualization middle panels (`Results/result1..7.png`) are measured against the right ground-truth panels for transparent reference.",
        "",
        f"- **Raw Low-Light Input vs GT (mean over 7 samples):** PSNR = `{mean_row['raw_input_psnr_db']:.2f} dB`, SSIM = `{mean_row['raw_input_ssim']:.4f}`",
        f"- **Local Untrained `MIRNet` (random weights) vs GT:** PSNR = `{mean_row['untrained_mirnet_psnr_db']:.2f} dB`, SSIM = `{mean_row['untrained_mirnet_ssim']:.4f}`",
        f"- **Upstream Author Published Panels (`Results/*.png`) vs GT:** PSNR = `{mean_row['author_panel_psnr_db']:.2f} dB`, SSIM = `{mean_row['author_panel_ssim']:.4f}` (upstream README reports `22.97 dB` across the full 15-image LOL eval set)",
        "",
        "## 5. Baseline Training Workflow Verification (Smoke Test)",
        f"- **Exit code:** `{summary['training_smoke_test']['returncode']}`",
        f"- **Elapsed time (1 epoch, 2 steps, 64x64 crops on {summary['device']}):** `{summary['training_smoke_test']['elapsed_seconds']:.2f} s`",
        f"- **Checkpoint saved & reloaded into `MIRNet`:** `{summary['training_smoke_test']['checkpoint_reloaded_into_MIRNet']}`",
        f"- **TensorBoard logs created:** `{len(summary['training_smoke_test']['tensorboard_event_files']) > 0}`",
        "",
        "## 6. Full Baseline Training Requirements",
        "- **Dataset:** LOL (`our485` training: 485 paired images; `eval15` validation: 15 paired images, `600x400`).",
        "- **Compute:** `MIRNet` has 32.21M parameters; full 100-epoch training on CPU is impractical. Recommended execution is Google Colab T4 GPU (or local RTX 3050 6GB after installing CUDA-enabled PyTorch).",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    torch.manual_seed(42)

    env_info = check_dependencies()
    ckpt_info = check_checkpoints()
    data_info = check_lol_dataset_paths()
    samples = extract_sample_triptychs()

    # Verify LoLDataset directly on extracted samples
    crop_ds = LoLDataset(SAMPLES_LOW_DIR, SAMPLES_HIGH_DIR, img_size=64, mode="crop")
    full_ds = LoLDataset(SAMPLES_LOW_DIR, SAMPLES_HIGH_DIR, img_size=None, mode="full")
    sample_x, sample_y = crop_ds[0]
    full_x, full_y = full_ds[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MIRNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bench_64 = benchmark_model(model, device, 64, 64, warmup=2, runs=5)
    bench_184 = benchmark_model(model, device, 184, 276, warmup=1, runs=3)

    df, csv_path = run_sample_inference(model, device, samples)
    smoke_info = run_training_smoke_test()

    summary = {
        "device": str(device),
        "environment": env_info,
        "checkpoint_check": ckpt_info,
        "dataset_check": data_info,
        "lol_dataset_loader_check": {
            "num_samples": len(crop_ds),
            "crop_tensor_shape": list(sample_x.shape),
            "full_tensor_shape": list(full_x.shape),
        },
        "model": {
            "class_name": "MIRNet",
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
        },
        "benchmarks": {
            "64x64": bench_64,
            "184x276": bench_184,
        },
        "metrics_csv": os.path.relpath(csv_path, ROOT_DIR),
        "training_smoke_test": smoke_info,
    }

    summary_path = os.path.join(BASELINE_DIR, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = write_report(summary, df)
    print(f"Baseline verification complete.")
    print(f"Summary JSON: {summary_path}")
    print(f"Metrics CSV:  {csv_path}")
    print(f"Report MD:    {report_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

