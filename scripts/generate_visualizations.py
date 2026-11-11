"""Phase 9: Visualization Generator.

Produces 5-panel comparison figures saved to Results/comparisons/:
  1. Original Low-Light Input
  2. MIRNet Output (Config B)
  3. Illumination-Only Output (Config A)
  4. Proposed Hybrid Model Output (Config D)
  5. Ground Truth

Also generates an explainability figure for HybridMIRNet showing the learned
illumination map L(x) and spatially varying Adaptive Feature Fusion weights.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys

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
from models.hybrid_mirnet import HybridMIRNet


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def create_five_panel_figure(
    name: str,
    low_img: np.ndarray,
    mirnet_img: np.ndarray,
    illum_img: np.ndarray,
    hybrid_img: np.ndarray,
    gt_img: np.ndarray,
    metrics_lookup: dict,
    out_path: str,
    status_label: str,
):
    panels = [
        ("1. Low-Light Input", low_img, None),
        ("2. MIRNet (Config B)", mirnet_img, metrics_lookup.get(("B", name))),
        ("3. Illumination Only (Config A)", illum_img, metrics_lookup.get(("A", name))),
        ("4. Hybrid MIRNet (Config D)", hybrid_img, metrics_lookup.get(("D", name))),
        ("5. Ground Truth", gt_img, None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    fig.suptitle(f"{name} — Qualitative Comparison ({status_label})", fontsize=12, fontweight="bold")
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow(img)
        if m is not None:
            ax.set_title(f"{title}\nPSNR: {m['psnr_db']:.2f} dB | SSIM: {m['ssim']:.3f}", fontsize=9.5)
        else:
            ax.set_title(title, fontsize=9.5)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_difficult_cases_grid(cases, low_dir, out_dir_base, high_dir, metrics_lookup, out_path, status_label):
    fig, axes = plt.subplots(len(cases), 5, figsize=(20, 3.8 * len(cases)))
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(
        f"Difficult Low-Light Cases Comparison ({status_label})",
        fontsize=13,
        fontweight="bold",
    )
    col_titles = [
        "1. Low-Light Input",
        "2. MIRNet (Config B)",
        "3. Illumination Only (Config A)",
        "4. Hybrid MIRNet (Config D)",
        "5. Ground Truth",
    ]
    for r, name in enumerate(cases):
        imgs = [
            _load_rgb(os.path.join(low_dir, name)),
            _load_rgb(os.path.join(out_dir_base, "B", name)),
            _load_rgb(os.path.join(out_dir_base, "A", name)),
            _load_rgb(os.path.join(out_dir_base, "D", name)),
            _load_rgb(os.path.join(high_dir, name)),
        ]
        cfgs = [None, "B", "A", "D", None]
        for c, (img, cfg) in enumerate(zip(imgs, cfgs)):
            ax = axes[r, c]
            ax.imshow(img)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, fontweight="bold")
            if cfg is not None and (cfg, name) in metrics_lookup:
                m = metrics_lookup[(cfg, name)]
                ax.set_xlabel(f"PSNR: {m['psnr_db']:.2f} dB | SSIM: {m['ssim']:.3f}", fontsize=8.5)
            elif c == 0:
                ax.set_xlabel(name, fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_hybrid_internals_figure(sample_low_path: str, ckpt_D: str | None, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMIRNet().to(device).eval()
    status = "Untrained Random Init"
    if ckpt_D and os.path.isfile(ckpt_D):
        ckpt = torch.load(ckpt_D, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        status = "Trained Checkpoint"

    arr = np.array(Image.open(sample_low_path).convert("RGB"), dtype=np.float32) / 255.0
    inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp, return_dict=True)

    illum_map = out["illumination_map"].squeeze().cpu().numpy()
    w_spatial = out["fusion_weights"]["spatial_weights"].squeeze(0).cpu().numpy()  # (3, H, W)
    hybrid_img = np.clip(out["enhanced"].squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.0))
    fig.suptitle(
        f"HybridMIRNet Internal Representations — {os.path.basename(sample_low_path)} ({status})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].imshow(arr)
    axes[0].set_title("Input Low-Light")
    im1 = axes[1].imshow(illum_map, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Estimated Illumination L(x)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(w_spatial[1], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Fusion Weight: MIRNet")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(w_spatial[2], cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Fusion Weight: Illumination")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    axes[4].imshow(hybrid_img)
    axes[4].set_title("Hybrid Enhanced Output")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 9: Generate 5-panel comparison visualizations")
    parser.add_argument("--test_low", type=str, default=os.path.join(ROOT_DIR, "baseline_results", "samples", "low"))
    parser.add_argument("--test_high", type=str, default=os.path.join(ROOT_DIR, "baseline_results", "samples", "high"))
    parser.add_argument("--experiments_dir", type=str, default=os.path.join(ROOT_DIR, "experiments"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "Results", "comparisons"))
    parser.add_argument("--ckpt_D", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    outputs_base = os.path.join(args.experiments_dir, "outputs")

    per_img_csv = os.path.join(args.experiments_dir, "ablation_per_image.csv")
    metrics_lookup = {}
    status_label = "Untrained Verification"
    if os.path.isfile(per_img_csv):
        df = pd.read_csv(per_img_csv)
        if (df["status"] == "trained").all():
            status_label = "Trained Checkpoints"
        for _, row in df.iterrows():
            metrics_lookup[(row["config"], row["image"])] = {
                "psnr_db": float(row["psnr_db"]),
                "ssim": float(row["ssim"]),
            }

    low_files = _list_image_files(args.test_low)
    high_files = _list_image_files(args.test_high)
    names = [os.path.basename(p) for p in low_files]

    for low_p, high_p in zip(low_files, high_files):
        name = os.path.basename(low_p)
        create_five_panel_figure(
            name=name,
            low_img=_load_rgb(low_p),
            mirnet_img=_load_rgb(os.path.join(outputs_base, "B", name)),
            illum_img=_load_rgb(os.path.join(outputs_base, "A", name)),
            hybrid_img=_load_rgb(os.path.join(outputs_base, "D", name)),
            gt_img=_load_rgb(high_p),
            metrics_lookup=metrics_lookup,
            out_path=os.path.join(args.out_dir, f"comparison_{os.path.splitext(name)[0]}.png"),
            status_label=status_label,
        )

    # Select difficult low-light cases (lowest input mean luminance)
    luminances = []
    for p in low_files:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        luminances.append((float(arr.mean()), os.path.basename(p)))
    difficult_cases = [name for _, name in sorted(luminances)[: min(3, len(luminances))]]

    create_difficult_cases_grid(
        cases=difficult_cases,
        low_dir=args.test_low,
        out_dir_base=outputs_base,
        high_dir=args.test_high,
        metrics_lookup=metrics_lookup,
        out_path=os.path.join(args.out_dir, "difficult_cases_grid.png"),
        status_label=status_label,
    )

    ckpt_d = args.ckpt_D or os.path.join(args.experiments_dir, "config_D", "checkpoints", "best_model.pth")
    create_hybrid_internals_figure(
        sample_low_path=low_files[0],
        ckpt_D=ckpt_d,
        out_path=os.path.join(args.out_dir, f"hybrid_internals_{os.path.splitext(names[0])[0]}.png"),
    )
    print(f"Generated {len(names)} per-sample comparisons + difficult_cases_grid.png + internals figure in {args.out_dir}")


if __name__ == "__main__":
    main()

