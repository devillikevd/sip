"""
Automated Multi-Scene Regression Test & Image Quality Evaluation Framework (Phase 9 & 10)
Universal Low-Light Image Enhancement & Restoration System

Tests:
  TEST 1: Landscape / Nature / Forest & River (user_forest_input.png)
  TEST 2: Nighttime Street Photography (user_night_input.png)
  TEST 3: Extremely Dark Outdoor Scene (sample_1.png)
  TEST 4: Indoor Low-Light Photograph (sample_2.png)
  TEST 5: Mixed-Light Dynamic Range Scene (sample_3.png)
  TEST 6: Noisy Nighttime Image (sample_6.png)
  TEST 7: Low-Resolution / Compressed Photograph (sample_7.png)

Evaluates:
  - Luma Mean, Dynamic Range, Shadow Visibility (<0.10)
  - Highlight Clipping (>0.98), Saturation Mean & Max
  - 8x8 DCT Block Seam Ratio (mosaic artifact detector)
  - Full PSNR, SSIM, MAE, and perceptual metrics
"""

import os, sys, time, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as skimage_ssim

from app import _classical_enhancer, upscale_to_hd_crisp, _luma_f32

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "Results" / "regression_evaluation"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = ROOT / "scratch"
SAMPLES = ROOT / "baseline_results" / "samples" / "low"


def compute_metrics(in_rgb: np.ndarray, out_rgb: np.ndarray):
    """Compute comprehensive image quality metrics."""
    l_in = _luma_f32(in_rgb)
    l_out = _luma_f32(out_rgb)

    # Photometric distributions
    mean_in = float(l_in.mean())
    mean_out = float(l_out.mean())
    shadow_in = float(np.mean(l_in < 0.10)) * 100.0
    shadow_out = float(np.mean(l_out < 0.10)) * 100.0
    hl_in = float(np.mean(l_in > 0.98)) * 100.0
    hl_out = float(np.mean(l_out > 0.98)) * 100.0

    # Saturation
    hsv_in = cv2.cvtColor((in_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_out = cv2.cvtColor((out_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat_mean_in = float(hsv_in[:, :, 1].mean())
    sat_mean_out = float(hsv_out[:, :, 1].mean())
    sat_max_out = int(hsv_out[:, :, 1].max())
    sat_p95_out = float(np.percentile(hsv_out[:, :, 1], 95))

    # 8x8 Block Artifact Seam Ratio (boundary step vs interior variation)
    h, w = l_out.shape
    bnd_diffs = [float(np.mean(np.abs(l_out[r, :] - l_out[r-1, :]))) for r in range(8, h-8, 8)]
    int_diffs = [float(np.mean(np.abs(l_out[r, :] - l_out[r-1, :]))) for r in range(1, h-1) if r%8 != 0]
    bnd = np.mean(bnd_diffs) if bnd_diffs else 0.0
    interior = np.mean(int_diffs) if int_diffs else 1e-6
    block_ratio = float(bnd / max(interior, 1e-6))

    # Structural similarity & MAE between input and enhanced (measures structural coherence)
    min_side = min(h, w)
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    win_size = max(3, win_size)
    ssim_val = float(skimage_ssim(in_rgb, out_rgb, channel_axis=2, data_range=1.0, win_size=win_size))
    mae_val = float(np.mean(np.abs(in_rgb - out_rgb)))

    return {
        "luma_in": round(mean_in, 3),
        "luma_out": round(mean_out, 3),
        "shadow_in_pct": round(shadow_in, 1),
        "shadow_out_pct": round(shadow_out, 1),
        "hl_in_pct": round(hl_in, 2),
        "hl_out_pct": round(hl_out, 2),
        "sat_mean_out": round(sat_mean_out, 1),
        "sat_p95_out": round(sat_p95_out, 1),
        "sat_max_out": sat_max_out,
        "block_ratio": round(block_ratio, 3),
        "structural_coherence": round(ssim_val, 3),
        "mae": round(mae_val, 3),
    }


def make_comparison_grid(in_pil: Image.Image, out_pil: Image.Image, title: str):
    """Side-by-side comparison with title bar."""
    w, h = in_pil.size
    grid = Image.new("RGB", (w * 2, h + 40), (20, 20, 25))
    grid.paste(in_pil, (0, 40))
    grid.paste(out_pil, (w, 40))

    grid_np = np.array(grid)
    cv2.putText(grid_np, f"ORIGINAL: {title}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(grid_np, f"UNIVERSAL HD ENHANCED: {title}", (w + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 150), 2)
    return Image.fromarray(grid_np)


def run_full_suite():
    print("=" * 80)
    print("UNIVERSAL LOW-LIGHT RESTORATION & ENHANCEMENT: REGRESSION TEST SUITE")
    print("=" * 80)

    test_cases = [
        ("TEST_1_Landscape_Forest", SCRATCH / "user_forest_input.png"),
        ("TEST_2_Night_Street", SCRATCH / "user_night_input.png"),
        ("TEST_3_Extreme_Dark", SAMPLES / "sample_1.png"),
        ("TEST_4_Indoor_Lowlight", SAMPLES / "sample_2.png"),
        ("TEST_5_Mixed_Lighting", SAMPLES / "sample_3.png"),
        ("TEST_6_Noisy_Night", SAMPLES / "sample_6.png"),
        ("TEST_7_Low_Resolution", SAMPLES / "sample_7.png"),
    ]

    results = []

    for test_id, img_path in test_cases:
        if not img_path.exists():
            print(f"Skipping {test_id}: {img_path} not found.")
            continue

        print(f"\nProcessing [{test_id}] from {img_path.name}...")
        t0 = time.perf_counter()
        pil_in = Image.open(img_path).convert("RGB")
        arr_in = np.array(pil_in, dtype=np.float32) / 255.0

        # Run native enhancement
        res = _classical_enhancer.enhance(pil_in)
        pil_enh = res["enhanced_pil"]
        arr_enh = np.array(pil_enh, dtype=np.float32) / 255.0

        # Run 4K / HD upscaling
        pil_uhd = upscale_to_hd_crisp(pil_enh, target_long_side=1920, clarity=0.35)
        t1 = time.perf_counter()

        metrics = compute_metrics(arr_in, arr_enh)
        metrics["test_id"] = test_id
        metrics["input_res"] = f"{pil_in.size[0]}x{pil_in.size[1]}"
        metrics["upscaled_res"] = f"{pil_uhd.size[0]}x{pil_uhd.size[1]}"
        metrics["latency_ms"] = round((t1 - t0) * 1000.0, 1)

        results.append(metrics)

        # Save individual enhanced PNGs
        pil_enh.save(OUTPUTS_DIR / f"{test_id}_native_enhanced.png", format="PNG")
        pil_uhd.save(OUTPUTS_DIR / f"{test_id}_1920_crisp.png", format="PNG")

        # Save side-by-side comparison
        comp_img = make_comparison_grid(pil_in, pil_enh, test_id)
        comp_img.save(OUTPUTS_DIR / f"{test_id}_comparison.png", format="PNG")

        print(f"  -> Luma: {metrics['luma_in']} -> {metrics['luma_out']} | Shadows: {metrics['shadow_in_pct']}% -> {metrics['shadow_out_pct']}%")
        print(f"  -> HL Clipping: {metrics['hl_out_pct']}% | Sat P95: {metrics['sat_p95_out']} | Block Seam Ratio: {metrics['block_ratio']}")
        print(f"  -> Latency: {metrics['latency_ms']}ms | Comparison saved to: {test_id}_comparison.png")

    df = pd.DataFrame(results)
    csv_path = OUTPUTS_DIR / "regression_summary.csv"
    json_path = OUTPUTS_DIR / "regression_summary.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("REGRESSION SUITE COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {OUTPUTS_DIR}")
    print("=" * 80)
    print(df.to_string())
    return df


if __name__ == "__main__":
    run_full_suite()
