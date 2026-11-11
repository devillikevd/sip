"""Phase 7: Ablation Study Runner for Configurations A, B, C, and D.

Evaluates all four configurations on an identical test split:
  A: Illumination branch only (IlluminationCurveEstimator)
  B: Original MIRNet only (Baseline)
  C: MIRNet + Illumination estimation (Sequential)
  D: MIRNet + Illumination estimation + Adaptive Feature Fusion (Proposed Hybrid)

Measures PSNR, SSIM, LPIPS (where feasible), NIQE (where feasible),
inference time (ms/image), and parameter count, generating CSVs and charts
automatically without fabricating any missing checkpoint or dataset results.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from dataset import _list_image_files
from models.hybrid_mirnet import CONFIG_DESCRIPTIONS, build_ablation_model
from utils import calc_PSNR, calc_SSIM, save_img


def _init_optional_metrics(device):
    lpips_fn = None
    lpips_status = "unavailable (lpips package not installed)"
    try:
        import lpips  # type: ignore

        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        lpips_status = "available (lpips AlexNet)"
    except Exception as exc:
        if "No module named" not in str(exc):
            lpips_status = f"unavailable ({type(exc).__name__})"

    niqe_fn = None
    niqe_status = "unavailable (pyiqa package not installed)"
    try:
        import pyiqa  # type: ignore

        niqe_fn = pyiqa.create_metric("niqe", device=device)
        niqe_status = "available (pyiqa NIQE)"
    except Exception as exc:
        if "No module named" not in str(exc):
            niqe_status = f"unavailable ({type(exc).__name__})"

    return lpips_fn, lpips_status, niqe_fn, niqe_status


def _load_tensor(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_checkpoint_if_available(model, ckpt_path: str | None, device: torch.device) -> tuple[bool, str]:
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        return True, ckpt_path
    return False, "none (untrained random init)"


def evaluate_configuration(
    config: str,
    low_files: list[str],
    high_files: list[str],
    device: torch.device,
    ckpt_path: str | None,
    outputs_dir: str,
    lpips_fn=None,
    niqe_fn=None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    model = build_ablation_model(config).to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    is_trained, resolved_ckpt = _load_checkpoint_if_available(model, ckpt_path, device)
    status = "trained" if is_trained else "untrained_verification"

    cfg_out_dir = os.path.join(outputs_dir, config)
    os.makedirs(cfg_out_dir, exist_ok=True)
    if config in ("C", "D"):
        os.makedirs(os.path.join(cfg_out_dir, "illumination_maps"), exist_ok=True)

    # Warmup pass on first image
    first_inp = _load_tensor(low_files[0]).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(first_inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

    per_image_rows = []
    for low_p, high_p in zip(low_files, high_files):
        name = os.path.basename(low_p)
        low_t = _load_tensor(low_p)
        high_t = _load_tensor(high_p)
        inp = low_t.unsqueeze(0).to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            out_dict = model(inp, return_dict=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        pred_t = torch.clamp(out_dict["enhanced"].squeeze(0).cpu(), 0.0, 1.0)
        save_img(pred_t, os.path.join(cfg_out_dir, name))

        if "illumination_map" in out_dict and config in ("C", "D"):
            illum = out_dict["illumination_map"].squeeze(0).squeeze(0).cpu().numpy()
            illum_rgb = np.stack([illum, illum, illum], axis=-1)
            save_img(illum_rgb, os.path.join(cfg_out_dir, "illumination_maps", name))

        psnr_val = calc_PSNR(high_t, pred_t)
        ssim_val = calc_SSIM(high_t, pred_t)

        lpips_val = None
        if lpips_fn is not None:
            with torch.no_grad():
                p_norm = pred_t.unsqueeze(0).to(device) * 2.0 - 1.0
                h_norm = high_t.unsqueeze(0).to(device) * 2.0 - 1.0
                lpips_val = float(lpips_fn(p_norm, h_norm).item())

        niqe_val = None
        if niqe_fn is not None:
            with torch.no_grad():
                niqe_val = float(niqe_fn(pred_t.unsqueeze(0).to(device)).item())

        per_image_rows.append(
            {
                "config": config,
                "status": status,
                "image": name,
                "resolution": f"{low_t.shape[1]}x{low_t.shape[2]}",
                "psnr_db": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "niqe": niqe_val,
                "infer_time_ms": (t1 - t0) * 1000.0,
            }
        )

    df_cfg = pd.DataFrame(per_image_rows)
    summary_row = {
        "config": config,
        "description": CONFIG_DESCRIPTIONS[config],
        "status": status,
        "checkpoint": resolved_ckpt,
        "num_images": len(df_cfg),
        "params": int(total_params),
        "params_millions": round(total_params / 1e6, 4),
        "psnr_db": float(df_cfg["psnr_db"].mean()),
        "ssim": float(df_cfg["ssim"].mean()),
        "lpips": float(df_cfg["lpips"].mean()) if df_cfg["lpips"].notna().all() else None,
        "niqe": float(df_cfg["niqe"].mean()) if df_cfg["niqe"].notna().all() else None,
        "infer_time_ms": float(df_cfg["infer_time_ms"].mean()),
    }
    return summary_row, per_image_rows


def generate_ablation_charts(summary_df: pd.DataFrame, out_dir: str):
    configs = summary_df["config"].tolist()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    status_note = (
        "Trained Checkpoints"
        if (summary_df["status"] == "trained").all()
        else "Local Untrained Architecture Verification (No Full LOL Weights)"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Ablation Study (Configs A–D) — {status_note}", fontsize=13, fontweight="bold")

    # 1. PSNR
    axes[0, 0].bar(configs, summary_df["psnr_db"], color=colors)
    axes[0, 0].set_title("Mean PSNR (dB) ↑")
    axes[0, 0].set_ylabel("PSNR (dB)")
    for i, v in enumerate(summary_df["psnr_db"]):
        axes[0, 0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. SSIM
    axes[0, 1].bar(configs, summary_df["ssim"], color=colors)
    axes[0, 1].set_title("Mean SSIM ↑")
    axes[0, 1].set_ylabel("SSIM")
    for i, v in enumerate(summary_df["ssim"]):
        axes[0, 1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Inference Time
    axes[1, 0].bar(configs, summary_df["infer_time_ms"], color=colors)
    axes[1, 0].set_title("Mean Inference Time (ms/image) ↓")
    axes[1, 0].set_ylabel("Latency (ms)")
    for i, v in enumerate(summary_df["infer_time_ms"]):
        axes[1, 0].text(i, v + max(summary_df["infer_time_ms"]) * 0.02, f"{v:.1f}", ha="center", fontsize=9)
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 4. Parameter Count (M)
    axes[1, 1].bar(configs, summary_df["params_millions"], color=colors)
    axes[1, 1].set_title("Parameter Count (Millions)")
    axes[1, 1].set_ylabel("Params (M)")
    for i, v in enumerate(summary_df["params_millions"]):
        axes[1, 1].text(i, v + 0.5, f"{v:.2f}M", ha="center", fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # Trade-off plot: Inference Time vs Parameter Count annotated by Config
    fig2, ax = plt.subplots(figsize=(7, 5))
    for i, row in summary_df.iterrows():
        ax.scatter(
            row["params_millions"],
            row["infer_time_ms"],
            s=140,
            color=colors[i % len(colors)],
            label=f"Config {row['config']} ({row['params_millions']:.2f}M)",
        )
        ax.annotate(
            f"Config {row['config']}\n({row['psnr_db']:.2f} dB)",
            (row["params_millions"], row["infer_time_ms"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )
    ax.set_title(f"Model Footprint vs. Latency ({status_note})")
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Inference Time (ms/image)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_efficiency_tradeoff.png"), dpi=150)
    plt.close(fig2)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7: Run Ablation Study (A, B, C, D)")
    parser.add_argument(
        "--test_low",
        type=str,
        default=os.path.join(ROOT_DIR, "baseline_results", "samples", "low"),
    )
    parser.add_argument(
        "--test_high",
        type=str,
        default=os.path.join(ROOT_DIR, "baseline_results", "samples", "high"),
    )
    parser.add_argument("--experiments_dir", type=str, default=os.path.join(ROOT_DIR, "experiments"))
    parser.add_argument("--ckpt_A", type=str, default=None)
    parser.add_argument("--ckpt_B", type=str, default=None)
    parser.add_argument("--ckpt_C", type=str, default=None)
    parser.add_argument("--ckpt_D", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.experiments_dir, exist_ok=True)
    outputs_dir = os.path.join(args.experiments_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    low_files = _list_image_files(args.test_low)
    high_files = _list_image_files(args.test_high)
    if len(low_files) != len(high_files):
        raise ValueError("Mismatched test split lengths.")

    lpips_fn, lpips_status, niqe_fn, niqe_status = _init_optional_metrics(device)
    print(f"Device: {device} | LPIPS: {lpips_status} | NIQE: {niqe_status}")

    ckpt_map = {
        "A": args.ckpt_A or os.path.join(args.experiments_dir, "config_A", "checkpoints", "best_model.pth"),
        "B": args.ckpt_B or os.path.join(args.experiments_dir, "config_B", "checkpoints", "best_model.pth"),
        "C": args.ckpt_C or os.path.join(args.experiments_dir, "config_C", "checkpoints", "best_model.pth"),
        "D": args.ckpt_D or os.path.join(args.experiments_dir, "config_D", "checkpoints", "best_model.pth"),
    }

    summaries = []
    all_per_image = []
    for cfg in ("A", "B", "C", "D"):
        s_row, per_rows = evaluate_configuration(
            config=cfg,
            low_files=low_files,
            high_files=high_files,
            device=device,
            ckpt_path=ckpt_map[cfg],
            outputs_dir=outputs_dir,
            lpips_fn=lpips_fn,
            niqe_fn=niqe_fn,
            seed=args.seed,
        )
        summaries.append(s_row)
        all_per_image.extend(per_rows)

    summary_df = pd.DataFrame(summaries)
    per_img_df = pd.DataFrame(all_per_image)

    summary_csv = os.path.join(args.experiments_dir, "ablation_results.csv")
    per_img_csv = os.path.join(args.experiments_dir, "ablation_per_image.csv")
    summary_df.to_csv(summary_csv, index=False)
    per_img_df.to_csv(per_img_csv, index=False)

    generate_ablation_charts(summary_df, args.experiments_dir)

    meta = {
        "device": str(device),
        "test_low": args.test_low,
        "test_high": args.test_high,
        "num_test_images": len(low_files),
        "lpips_status": lpips_status,
        "niqe_status": niqe_status,
    }
    with open(os.path.join(args.experiments_dir, "ablation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nAblation Summary:")
    print(summary_df[["config", "status", "params", "psnr_db", "ssim", "infer_time_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()

