import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import _classical_enhancer, _luma_f32, _estimate_noise_mad, _edge_confidence, _apply_hd_clarity_tanh

audit_dir = Path("scratch/stage_audit")
audit_dir.mkdir(parents=True, exist_ok=True)

pil = Image.open("scratch/user_forest_input.png")
enh = _classical_enhancer

# Stage 0: Input
img_u8 = np.array(pil.convert("RGB"), dtype=np.uint8)
img_f0 = img_u8.astype(np.float32) / 255.0
h0, w0 = img_f0.shape[:2]
res_sc = min(max(max(h0, w0) / 480.0, 1.0), 3.0)

t0_raw = np.max(img_f0, axis=2).astype(np.float32)
p99_raw = float(np.percentile(t0_raw, 99))
p25_raw = float(np.percentile(t0_raw, 25))
p02_raw = float(np.percentile(t0_raw, 2))
is_backlit_night = p99_raw > 0.75 and p25_raw < 0.14

# Stage 1: Illumination & Pre-denoise
illum = enh._estimate_illumination(img_f0)
clean_f = enh._pre_denoise_float(img_f0, illum, res_sc, is_backlit_night=is_backlit_night)

# Stage 2: Retinex Lift
lifted_f = enh._retinex_lift(clean_f, illum, shadow_lift=1.2, light_boost=0.85, exposure_ev=0.50, is_backlit_night=is_backlit_night)

# Stage 3: Soft White Balance
wb_f = enh._white_balance_soft(lifted_f, strength=0.08)

# Stage 4: Denoise Float
lum_pre = _luma_f32(wb_f)
noise_est = _estimate_noise_mad(lum_pre)
dn_f, edge_c = enh._denoise_float(wb_f, illum, res_sc, noise_est, light_boost=0.85, is_backlit_night=is_backlit_night)

# Stage 5: CLAHE
clahe_f = enh._clahe_l_float(dn_f, edge_c, res_sc=res_sc, clarity=0.40, clip_limit=1.6, is_backlit_night=is_backlit_night)

# Stage 6: Tone & Color
toned_f = enh._tone_and_color(clahe_f, saturation=1.15, warmth=0.0, contrast=0.14, is_backlit_night=is_backlit_night)

# Stage 7: HD Clarity Tanh
res_noise = _estimate_noise_mad(_luma_f32(toned_f))
final_f = _apply_hd_clarity_tanh(toned_f, clarity=0.40, scale_factor=1.0, noise_level=res_noise)

stages = [
    ("Stage0_Input", img_f0),
    ("Stage1_PreDenoise", clean_f),
    ("Stage2_RetinexLift", lifted_f),
    ("Stage3_WhiteBalance", wb_f),
    ("Stage4_PostDenoise", dn_f),
    ("Stage5_CLAHE", clahe_f),
    ("Stage6_ToneColor", toned_f),
    ("Stage7_HDClarity", final_f)
]

print(f"is_backlit_night: {is_backlit_night}")
print(f"p99_raw: {p99_raw:.3f}, p25_raw: {p25_raw:.3f}, p02_raw: {p02_raw:.3f}")
print("-" * 105)
print(f"{'Stage':<22} | {'Mean':<6} | {'Min':<6} | {'P01':<6} | {'P05':<6} | {'P95':<6} | {'P99':<6} | {'Contrast':<10} | {'Sat Mean':<8}")
print("-" * 105)

for name, s_arr in stages:
    Image.fromarray((np.clip(s_arr, 0, 1) * 255).round().astype(np.uint8)).save(audit_dir / f"{name}.png")
    l = _luma_f32(s_arr)
    hsv = cv2.cvtColor((np.clip(s_arr, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2HSV)
    p01, p05, p95, p99 = np.percentile(l, [1, 5, 95, 99])
    contrast = p95 - p05
    sat = hsv[:, :, 1].mean()
    print(f"{name:<22} | {l.mean():<6.3f} | {l.min():<6.3f} | {p01:<6.3f} | {p05:<6.3f} | {p95:<6.3f} | {p99:<6.3f} | {contrast:<10.3f} | {sat:<8.1f}")
