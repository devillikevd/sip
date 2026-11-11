"""Phase 8: Published Research Comparison Script.

Compares published literature results on LOL (eval15) with clear citations
against locally verified measurements (read directly from ablation_results.csv,
never inventing or fabricating trained metrics).
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
import pandas as pd


PUBLISHED_LOL_BENCHMARKS = [
    {
        "method": "RetinexNet",
        "source_type": "Published (Wei et al., BMVC 2018)",
        "dataset_split": "LOL eval15 (paired)",
        "training_paradigm": "Paired Retinex Decomposition",
        "params_m": 0.84,
        "psnr_db": 16.77,
        "ssim": 0.560,
        "notes": "Standard benchmark score on LOL eval15 (various re-evaluations report 0.42–0.56 SSIM depending on RGB vs Y-channel metric).",
    },
    {
        "method": "EnlightenGAN",
        "source_type": "Published (Jiang et al., IEEE TIP 2021)",
        "dataset_split": "LOL eval15 (paired)",
        "training_paradigm": "Unpaired GAN",
        "params_m": 8.64,
        "psnr_db": 17.48,
        "ssim": 0.651,
        "notes": "Trained unpaired; evaluated on LOL eval15.",
    },
    {
        "method": "Zero-DCE",
        "source_type": "Published (Guo et al., CVPR 2020)",
        "dataset_split": "LOL eval15 (paired)",
        "training_paradigm": "Zero-Reference Curve Estimation",
        "params_m": 0.079,
        "psnr_db": 14.86,
        "ssim": 0.559,
        "notes": "Zero-reference training without ground-truth pairing (~79.4K params).",
    },
    {
        "method": "Zero-DCE++",
        "source_type": "Published (Li et al., IEEE TPAMI 2021)",
        "dataset_split": "LOL eval15 (paired)",
        "training_paradigm": "Zero-Reference Lightweight Curve",
        "params_m": 0.011,
        "psnr_db": 14.86,
        "ssim": 0.550,
        "notes": "Lightweight depthwise-separable variant (~10.6K params); literature reports ~14.8–15.2 dB PSNR depending on protocol.",
    },
    {
        "method": "MIRNet (Paper)",
        "source_type": "Published (Zamir et al., ECCV 2020)",
        "dataset_split": "LOL eval15 (paired)",
        "training_paradigm": "Paired Supervised Multi-Scale",
        "params_m": 31.79,
        "psnr_db": 24.14,
        "ssim": 0.830,
        "notes": "Official ECCV 2020 MIRNet benchmark on LOL eval15.",
    },
    {
        "method": "MIRNet (Baseline Reference)",
        "source_type": "Reference PyTorch README (2021)",
        "dataset_split": "LOL eval15 (64x64 patch setup)",
        "training_paradigm": "Paired Supervised (Charbonnier)",
        "params_m": 32.21,
        "psnr_db": 22.97,
        "ssim": None,
        "notes": "Reported in upstream README; no pretrained checkpoint included in repo.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 8: Published Research Comparison")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "experiments"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = args.experiments_dir
    os.makedirs(exp_dir, exist_ok=True)

    rows = list(PUBLISHED_LOL_BENCHMARKS)

    ablation_csv = os.path.join(exp_dir, "ablation_results.csv")
    if os.path.isfile(ablation_csv):
        ab_df = pd.read_csv(ablation_csv)
        for _, r in ab_df.iterrows():
            rows.append(
                {
                    "method": f"Local Config {r['config']}",
                    "source_type": f"Local Run ({r['status']})",
                    "dataset_split": f"Local test ({int(r['num_images'])} images)",
                    "training_paradigm": r["description"],
                    "params_m": float(r["params_millions"]),
                    "psnr_db": float(r["psnr_db"]),
                    "ssim": float(r["ssim"]),
                    "notes": f"Checkpoint: {r['checkpoint']}",
                }
            )

    comp_df = pd.DataFrame(rows)
    out_csv = os.path.join(exp_dir, "published_research_comparison.csv")
    comp_df.to_csv(out_csv, index=False)

    pub_df = comp_df[comp_df["source_type"].str.startswith("Published")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        "Published Research Benchmarks on LOL (eval15) — Strictly Literature Values",
        fontsize=12,
        fontweight="bold",
    )

    palette = ["#8C564B", "#9467BD", "#17BECF", "#BCBD22", "#1F77B4"]
    axes[0].barh(pub_df["method"], pub_df["psnr_db"], color=palette)
    axes[0].set_title("Published PSNR (dB) ↑")
    axes[0].set_xlabel("PSNR (dB)")
    for i, v in enumerate(pub_df["psnr_db"]):
        axes[0].text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(pub_df["psnr_db"]) + 3.5)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(pub_df["method"], pub_df["ssim"], color=palette)
    axes[1].set_title("Published SSIM ↑")
    axes[1].set_xlabel("SSIM")
    for i, v in enumerate(pub_df["ssim"]):
        axes[1].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9)
    axes[1].set_xlim(0, 1.0)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "research_comparison_chart.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {out_csv} and research_comparison_chart.png")
    print(comp_df[["method", "source_type", "params_m", "psnr_db", "ssim"]].to_string(index=False))


if __name__ == "__main__":
    main()
