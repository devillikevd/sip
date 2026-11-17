"""
PHASE 1 DIAGNOSTIC: Print shapes, dtypes, min/max/NaN at every pipeline stage.
Run: python scripts/diagnose_pipeline.py
Saves ablation PNGs to scratch/ablation_*.png
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from PIL import Image
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent.parent / "scratch"
SCRATCH.mkdir(exist_ok=True)

# ── Load test image ──────────────────────────────────────────
# Try user night street image first, then any available sample
candidates = list((Path(__file__).resolve().parent.parent / "baseline_results" / "samples" / "low").glob("*.png"))
candidates += list((Path(__file__).resolve().parent.parent / "scratch").glob("*.jpg"))
candidates += list((Path(__file__).resolve().parent.parent / "scratch").glob("*.png"))

if not candidates:
    print("No test images found. Creating synthetic low-light test image.")
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[200:280, 300:340] = 255          # bright lamp core
    arr[150:350, 100:500] = np.clip(arr[150:350, 100:500] + 30, 0, 255)
    img_path = SCRATCH / "synthetic_night.png"
    Image.fromarray(arr).save(img_path)
else:
    img_path = candidates[0]

print(f"\n=== Using test image: {img_path} ===\n")

pil = Image.open(img_path).convert("RGB")
print(f"PIL size: {pil.size}  mode: {pil.mode}")

img_u8 = np.array(pil, dtype=np.uint8)
print(f"[A] uint8 input  shape={img_u8.shape}  dtype={img_u8.dtype}  min={img_u8.min()}  max={img_u8.max()}")

img_f = img_u8.astype(np.float32) / 255.0
print(f"[A] float32      shape={img_f.shape}  dtype={img_f.dtype}  min={img_f.min():.4f}  max={img_f.max():.4f}  NaN={np.isnan(img_f).any()}  Inf={np.isinf(img_f).any()}")

# Save A — original
Image.fromarray(img_u8).save(SCRATCH / "ablation_A_original.png")
print("Saved: ablation_A_original.png")

# ── Import pipeline internals ────────────────────────────────
from app import (
    ClassicalEnhancer, _luma_f32, _estimate_noise_mad,
    _apply_hd_clarity_tanh, _guided_filter_f32, _smoothstep_f32,
    _edge_confidence, _richardson_lucy_y, upscale_to_hd_crisp
)

enh = ClassicalEnhancer()
h0, w0 = img_f.shape[:2]
res_sc = min(max(max(h0, w0) / 480.0, 1.0), 3.0)

t0_raw = np.max(img_f, axis=2).astype(np.float32)
p99_raw = float(np.percentile(t0_raw, 99))
p25_raw = float(np.percentile(t0_raw, 25))
p02_raw = float(np.percentile(t0_raw, 2))
is_hdr_contrast = p99_raw > 0.70
is_backlit_night = is_hdr_contrast and (p25_raw < 0.14)

print(f"\n[Detect] p99={p99_raw:.3f}  p25={p25_raw:.3f}  p02={p02_raw:.3f}  is_hdr={is_hdr_contrast}  is_night={is_backlit_night}")

# Pedestal removal
img_work = img_f.copy()
if is_hdr_contrast and p02_raw > 0.025:
    ped = min(p02_raw * 0.82, 0.14)
    img_work = np.clip((img_work - np.float32(ped)) / np.float32(1.0 - ped), 0.0, 1.0).astype(np.float32)
    print(f"[Pedestal] removed {ped:.4f}  → min={img_work.min():.4f}  max={img_work.max():.4f}")

# Stage B — after normalization/pedestal
b_u8 = (np.clip(img_work, 0, 1) * 255).round().astype(np.uint8)
Image.fromarray(b_u8).save(SCRATCH / "ablation_B_after_pedestal.png")
print("Saved: ablation_B_after_pedestal.png")

# ── Illumination estimation ──────────────────────────────────
illum = enh._estimate_illumination(img_work)
print(f"\n[Illum] shape={illum.shape}  min={illum.min():.4f}  max={illum.max():.4f}  mean={illum.mean():.4f}  NaN={np.isnan(illum).any()}")

# ── Pre-denoise ───────────────────────────────────────────────
clean_f = enh._pre_denoise_float(img_work, illum, res_sc, is_backlit_night=is_backlit_night)
print(f"[PreDenoise] min={clean_f.min():.4f}  max={clean_f.max():.4f}  NaN={np.isnan(clean_f).any()}")

# Stage C — after preprocessing / denoise
c_u8 = (np.clip(clean_f, 0, 1) * 255).round().astype(np.uint8)
Image.fromarray(c_u8).save(SCRATCH / "ablation_C_pre_denoise.png")
print("Saved: ablation_C_pre_denoise.png")

# ── Retinex lift ─────────────────────────────────────────────
lifted_f = enh._retinex_lift(clean_f, illum, shadow_lift=1.2, light_boost=0.85, exposure_ev=0.50,
                              is_hdr_contrast=is_hdr_contrast, is_backlit_night=is_backlit_night)
print(f"\n[RetinexLift] min={lifted_f.min():.4f}  max={lifted_f.max():.4f}  mean={lifted_f.mean():.4f}  NaN={np.isnan(lifted_f).any()}")

# Check for overexposed regions
overexp_pct = float(np.mean(lifted_f > 0.95)) * 100
print(f"[RetinexLift] Overexposed pixels (>0.95): {overexp_pct:.1f}%")

# Stage D — after illumination correction
d_u8 = (np.clip(lifted_f, 0, 1) * 255).round().astype(np.uint8)
Image.fromarray(d_u8).save(SCRATCH / "ablation_D_after_retinex.png")
print("Saved: ablation_D_after_retinex.png")

# ── White balance ─────────────────────────────────────────────
wb_f = enh._white_balance_soft(lifted_f, strength=0.08)
lum_pre = _luma_f32(wb_f)
noise_est = _estimate_noise_mad(lum_pre)
print(f"\n[WhiteBalance] min={wb_f.min():.4f}  max={wb_f.max():.4f}  noise_est={noise_est:.5f}")

# ── Denoise ───────────────────────────────────────────────────
dn_f, edge_c = enh._denoise_float(wb_f, illum, res_sc, noise_est, light_boost=0.85, is_backlit_night=is_backlit_night)
print(f"[Denoise] min={dn_f.min():.4f}  max={dn_f.max():.4f}  edge_c mean={edge_c.mean():.4f}  NaN={np.isnan(dn_f).any()}")

# Check for block artifacts via variance in 8x8 blocks
def block_variance_uniformity(arr):
    """Detect if 8x8 blocks have abnormally uniform interior (artifact indicator)."""
    gray = _luma_f32(arr)
    h, w = gray.shape
    variances = []
    for r in range(0, h - 8, 8):
        for c in range(0, w - 8, 8):
            block = gray[r:r+8, c:c+8]
            variances.append(float(np.var(block)))
    variances = np.array(variances)
    very_flat = float(np.mean(variances < 0.0001)) * 100
    return very_flat, variances.mean()

flat_pct, avg_var = block_variance_uniformity(img_f)
print(f"\n[BlockArtifact Check] Original: {flat_pct:.1f}% flat 8x8 blocks, avg_var={avg_var:.6f}")
flat_pct2, avg_var2 = block_variance_uniformity(dn_f)
print(f"[BlockArtifact Check] Post-Denoise: {flat_pct2:.1f}% flat 8x8 blocks, avg_var={avg_var2:.6f}")

# ── CLAHE ─────────────────────────────────────────────────────
clahe_f = enh._clahe_l_float(dn_f, edge_c, res_sc=res_sc, clarity=0.40, clip_limit=1.6,
                               is_hdr_contrast=is_hdr_contrast, is_backlit_night=is_backlit_night)
print(f"\n[CLAHE] min={clahe_f.min():.4f}  max={clahe_f.max():.4f}  NaN={np.isnan(clahe_f).any()}")
flat_pct3, avg_var3 = block_variance_uniformity(clahe_f)
print(f"[BlockArtifact Check] Post-CLAHE: {flat_pct3:.1f}% flat 8x8 blocks, avg_var={avg_var3:.6f}")

# Stage E — after CLAHE
e_u8 = (np.clip(clahe_f, 0, 1) * 255).round().astype(np.uint8)
Image.fromarray(e_u8).save(SCRATCH / "ablation_E_after_clahe.png")
print("Saved: ablation_E_after_clahe.png")

# ── Tone & Color ──────────────────────────────────────────────
toned_f = enh._tone_and_color(clahe_f, saturation=1.05, warmth=0.0, contrast=0.14,
                               is_hdr_contrast=is_hdr_contrast, is_backlit_night=is_backlit_night)
# Chromaticity
toned_f = enh._ai_chromaticity_synthesis(toned_f, illum, color_fill=0.80,
                                          is_hdr_contrast=is_hdr_contrast, is_backlit_night=is_backlit_night)
res_noise = _estimate_noise_mad(_luma_f32(toned_f))
print(f"\n[ToneColor] min={toned_f.min():.4f}  max={toned_f.max():.4f}  res_noise={res_noise:.5f}")

# ── HD Clarity ────────────────────────────────────────────────
final_f = _apply_hd_clarity_tanh(toned_f, clarity=0.40, scale_factor=1.0, noise_level=res_noise)
print(f"[HDClarity] min={final_f.min():.4f}  max={final_f.max():.4f}  NaN={np.isnan(final_f).any()}")

# Stage F — final enhanced (native resolution, no upscale)
f_u8 = (np.clip(final_f, 0, 1) * 255).round().astype(np.uint8)
Image.fromarray(f_u8).save(SCRATCH / "ablation_F_enhanced_native.png")
print("Saved: ablation_F_enhanced_native.png")

# ── Upscale to 4K ─────────────────────────────────────────────
enh_pil = Image.fromarray(f_u8)
if max(enh_pil.size) < 1920:
    hd_pil = upscale_to_hd_crisp(enh_pil, target_long_side=1920, clarity=0.40, rl_iters=2, n_ibp=2)
    g_u8 = np.array(hd_pil)
    print(f"\n[Upscale→1920] output shape={g_u8.shape}  min={g_u8.min()}  max={g_u8.max()}")
    flat_pct4, avg_var4 = block_variance_uniformity(g_u8.astype(np.float32)/255.0)
    print(f"[BlockArtifact Check] Post-Upscale: {flat_pct4:.1f}% flat 8x8 blocks, avg_var={avg_var4:.6f}")
    hd_pil.save(SCRATCH / "ablation_G_upscaled.png")
    print("Saved: ablation_G_upscaled.png")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"Input shape:       {img_f.shape}")
print(f"is_hdr_contrast:   {is_hdr_contrast}")
print(f"is_backlit_night:  {is_backlit_night}")
print(f"noise_est:         {noise_est:.5f}")
print(f"res_noise:         {res_noise:.5f}")
print(f"Overexposed after Retinex: {overexp_pct:.1f}%")
print(f"Flat 8x8 blocks (original):   {flat_pct:.1f}%")
print(f"Flat 8x8 blocks (post-denoise): {flat_pct2:.1f}%")
print(f"Flat 8x8 blocks (post-clahe):   {flat_pct3:.1f}%")
print(f"\nAblation images saved to: {SCRATCH}")
print("\nCompare ablation_A through F to identify which stage causes degradation.")

