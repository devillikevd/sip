import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

def _luma(arr):
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)

def audit_quality_metrics(orig_rgb, enh_rgb, name=""):
    """Automated Quality Checks as requested by user."""
    l_orig = _luma(orig_rgb)
    l_enh = _luma(enh_rgb)
    
    # 1. Gray Haze Detection:
    # Gray haze manifests as an abnormally high black floor (p01 > 0.15) and compressed shadow variance
    p01_enh = float(np.percentile(l_enh, 1))
    p05_enh = float(np.percentile(l_enh, 5))
    p95_enh = float(np.percentile(l_enh, 95))
    p99_enh = float(np.percentile(l_enh, 99))
    
    gray_floor_ratio = p01_enh / (l_enh.mean() + 1e-6)
    # If p01 > 0.20, there is a severe gray haze
    is_gray_haze = p01_enh > 0.20 or gray_floor_ratio > 0.45
    
    # 2. Contrast Preservation:
    orig_dynamic_range = float(np.percentile(l_orig, 95) - np.percentile(l_orig, 5))
    enh_dynamic_range = float(p95_enh - p05_enh)
    contrast_ratio = enh_dynamic_range / max(orig_dynamic_range, 1e-6)
    
    # 3. Highlight Clipping:
    hl_clipped = float(np.mean(l_enh > 0.96)) * 100.0
    
    # 4. Shadow Recovery:
    # Measure detail gain in the dark areas of the original image (l_orig < 0.20)
    dark_mask = l_orig < 0.20
    dark_orig_mean = float(l_orig[dark_mask].mean()) if dark_mask.any() else 0.0
    dark_enh_mean = float(l_enh[dark_mask].mean()) if dark_mask.any() else 0.0
    shadow_lift_ratio = dark_enh_mean / max(dark_orig_mean, 1e-6)
    
    # 5. Color Fidelity:
    hsv_orig = cv2.cvtColor((orig_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_enh = cv2.cvtColor((enh_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat_orig = float(hsv_orig[:, :, 1].mean())
    sat_enh = float(hsv_enh[:, :, 1].mean())
    # Hue shift in non-dark pixels (sat > 20, luma > 0.10)
    valid_chroma = (hsv_orig[:, :, 1] > 20) & (l_orig > 0.10)
    if valid_chroma.any():
        hue_diff = np.abs(hsv_orig[:, :, 0].astype(np.float32) - hsv_enh[:, :, 0].astype(np.float32))
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
        mean_hue_shift = float(hue_diff[valid_chroma].mean())
    else:
        mean_hue_shift = 0.0
        
    # 6. Structural Similarity (SSIM):
    ssim_val = float(ssim_fn(l_orig, l_enh, data_range=1.0))
    
    print(f"=== Quality Audit: {name} ===")
    print(f"  Mean Luma: {l_orig.mean():.3f} -> {l_enh.mean():.3f}")
    print(f"  Black Floor (P01): {p01_enh:.3f} | Min: {l_enh.min():.3f} (Gray Haze: {is_gray_haze})")
    print(f"  Contrast (P95-P05): {orig_dynamic_range:.3f} -> {enh_dynamic_range:.3f} (Ratio: {contrast_ratio:.2f}x)")
    print(f"  Shadow Lift in darks: {dark_orig_mean:.3f} -> {dark_enh_mean:.3f} ({shadow_lift_ratio:.2f}x)")
    print(f"  Highlight Clipping (>0.96): {hl_clipped:.2f}%")
    print(f"  Color Fidelity: Sat {sat_orig:.1f} -> {sat_enh:.1f} | Mean Hue Shift: {mean_hue_shift:.2f} deg")
    print(f"  SSIM: {ssim_val:.4f}")
    return {
        "is_gray_haze": is_gray_haze,
        "contrast_ratio": contrast_ratio,
        "hl_clipped": hl_clipped,
        "shadow_lift_ratio": shadow_lift_ratio,
        "mean_hue_shift": mean_hue_shift,
        "ssim": ssim_val
    }

if __name__ == "__main__":
    pil_in = Image.open("scratch/user_forest_input.png")
    orig_rgb = np.array(pil_in.convert("RGB"), dtype=np.float32) / 255.0
    
    # 1. Audit old baseline
    old_enh = cv2.imread("scratch/stage_audit/Stage7_HDClarity.png")
    old_rgb = cv2.cvtColor(old_enh, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    audit_quality_metrics(orig_rgb, old_rgb, "Previous Baseline (With Gray Veil)")
