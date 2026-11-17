"""
PHASE 5: ISOLATED TESTING PIPELINE
Evaluates each enhancement stage independently on:
  1. Landscape: user_forest_input.png
  2. Night Street: user_night_input.png

Generates:
  A. Original input
  B. Existing baseline model output (MIRNet deep model with checkpoint)
  C. Model output without additional postprocessing (raw Retinex illumination lift)
  D. Output with illumination correction only (Guided Retinex + rational curve)
  E. Output with denoising only (Illumination + edge-preserving bilateral denoising)
  F. Output with color correction only (Illumination + soft white balance + natural vibrance)
  G. Output with controlled sharpening only (Illumination + subtle guided acutance)
  H. Final output using ONLY validated stages (Native resolution, balanced contrast, natural color)
"""

import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from model import MIRNet

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "scratch"
ABLAT_DIR = ROOT / "Results" / "isolated_ablation"
ABLAT_DIR.mkdir(parents=True, exist_ok=True)


def _luma(arr: np.ndarray) -> np.ndarray:
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)


def guided_filter_f32(guide: np.ndarray, src: np.ndarray, radius: int = 15, eps: float = 0.04) -> np.ndarray:
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, k)
    corr_I = cv2.boxFilter(guide * guide, cv2.CV_32F, k)
    corr_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, k)
    var_I = np.maximum(corr_I - mean_I * mean_I, 0.0)
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, k)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, k)
    return np.clip(mean_a * guide + mean_b, 1e-4, 1.0).astype(np.float32)


def run_stage_ablation(img_path: Path, prefix: str):
    print("=" * 70)
    print(f"RUNNING ISOLATED ABLATION ON: {prefix} ({img_path.name})")
    print("=" * 70)

    # ── STAGE A: Original Input ──────────────────────────────────
    pil_in = Image.open(img_path).convert("RGB")
    arr_u8 = np.array(pil_in, dtype=np.uint8)
    arr_f = arr_u8.astype(np.float32) / 255.0
    h, w = arr_f.shape[:2]
    pil_in.save(ABLAT_DIR / f"{prefix}_Stage_A_original.png")
    luma_a = _luma(arr_f)
    print(f"Stage A (Original): shape={arr_f.shape}, Luma={luma_a.mean():.3f}, Min={arr_f.min():.3f}, Max={arr_f.max():.3f}")

    # ── STAGE B: Existing Baseline Model Output (Deep MIRNet) ─────
    ckpt_path = ROOT / "baseline_results" / "smoke_test_run" / "checkpoints" / "Mirnet_enhance_best.pth"
    if ckpt_path.is_file():
        model = MIRNet()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        t = torch.from_numpy(arr_f).permute(2, 0, 1).unsqueeze(0)
        if pad_h > 0 or pad_w > 0:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            out_deep = model(t)
        out_deep = out_deep[:, :, :h, :w].squeeze(0).permute(1, 2, 0).numpy()
        arr_b = np.clip(out_deep, 0.0, 1.0)
        Image.fromarray((arr_b * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_B_baseline_mirnet.png")
        luma_b = _luma(arr_b)
        print(f"Stage B (Baseline MIRNet): Luma={luma_b.mean():.3f}, Min={arr_b.min():.3f}, Max={arr_b.max():.3f}")
    else:
        print("Stage B: Checkpoint not found, skipping deep model run.")
        arr_b = arr_f

    # ── STAGE C: Raw Retinex without any postprocessing ──────────
    t0 = np.max(arr_f, axis=2)
    illum_raw = guided_filter_f32(luma_a, t0, radius=15, eps=0.04)
    illum_c = np.clip(np.maximum(illum_raw, 0.70 * t0), 0.015, 1.0)
    gamma_c = float(np.clip(np.log(0.48) / np.log(max(float(luma_a.mean()), 0.025)), 0.35, 0.85))
    gain_c = np.clip(np.power(illum_c, gamma_c) / np.maximum(illum_c, 0.015), 1.0, 5.0)
    arr_c = np.clip(arr_f * gain_c[:, :, None], 0.0, 1.0)
    Image.fromarray((arr_c * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_C_raw_retinex.png")
    luma_c = _luma(arr_c)
    print(f"Stage C (Raw Retinex): Luma={luma_c.mean():.3f}, Min={arr_c.min():.3f}, Max={arr_c.max():.3f}")

    # ── STAGE D: Illumination Correction Only (with Highlight Guard) ──
    # Prevents blown-out lights and washed-out highlights
    hl_guard = np.clip((0.92 - luma_a) / 0.25, 0.0, 1.0)
    gain_d = 1.0 + (gain_c - 1.0) * hl_guard
    arr_d = np.clip(arr_f * gain_d[:, :, None], 0.0, 1.0)
    # Rational exposure fill
    lum_d = _luma(arr_d)
    k_exp = 0.25
    lum_new = (1.0 + k_exp) * lum_d / (1.0 + k_exp * lum_d)
    arr_d = np.clip(arr_d * (lum_new / np.maximum(lum_d, 1e-4))[:, :, None], 0.0, 1.0)
    Image.fromarray((arr_d * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_D_illumination_only.png")
    luma_d = _luma(arr_d)
    print(f"Stage D (Illumination Only): Luma={luma_d.mean():.3f}, Min={arr_d.min():.3f}, Max={arr_d.max():.3f}")

    # ── STAGE E: Denoising Only (applied on Stage D) ─────────────
    # Bilateral edge-preserving chrominance and luminance filtering
    yc = cv2.cvtColor(arr_d, cv2.COLOR_RGB2YCrCb)
    yc[:, :, 1] = cv2.bilateralFilter(yc[:, :, 1], d=5, sigmaColor=0.03, sigmaSpace=2.0)
    yc[:, :, 2] = cv2.bilateralFilter(yc[:, :, 2], d=5, sigmaColor=0.03, sigmaSpace=2.0)
    # Luminance gentle bilateral
    y_base = cv2.bilateralFilter(yc[:, :, 0], d=5, sigmaColor=0.04, sigmaSpace=2.0)
    # Edge mask
    gx = cv2.Sobel(yc[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(yc[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.hypot(gx, gy) * 3.0 - 0.08, 0.0, 0.70)
    yc[:, :, 0] = y_base * (1.0 - edge) + yc[:, :, 0] * edge
    arr_e = np.clip(cv2.cvtColor(yc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
    Image.fromarray((arr_e * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_E_denoise_only.png")
    luma_e = _luma(arr_e)
    print(f"Stage E (Denoising Only): Luma={luma_e.mean():.3f}, Min={arr_e.min():.3f}, Max={arr_e.max():.3f}")

    # ── STAGE F: Color Correction Only (applied on Stage D) ───────
    # Soft gray-world white balance + natural vibrance without hue rotation
    m_r, m_g, m_b = arr_d[:, :, 0].mean(), arr_d[:, :, 1].mean(), arr_d[:, :, 2].mean()
    avg = (m_r + m_g + m_b) / 3.0 + 1e-6
    wb = np.stack([
        arr_d[:, :, 0] * (avg / (m_r + 1e-6)),
        arr_d[:, :, 1] * (avg / (m_g + 1e-6)),
        arr_d[:, :, 2] * (avg / (m_b + 1e-6)),
    ], axis=-1)
    arr_wb = np.clip(0.95 * arr_d + 0.05 * wb, 0.0, 1.0)

    # Natural vibrance (photometric ratio retention, no synthetic hue addition)
    hsv = cv2.cvtColor(arr_wb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    # Gentle boost only for muted mid-tones
    deficit = np.maximum(0.40 - s, 0.0)
    hsv[:, :, 1] = np.clip(s + deficit * 0.20, 0.0, 0.65)
    arr_f_out = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)
    Image.fromarray((arr_f_out * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_F_color_correction_only.png")
    luma_f = _luma(arr_f_out)
    print(f"Stage F (Color Correction Only): Luma={luma_f.mean():.3f}, Min={arr_f_out.min():.3f}, Max={arr_f_out.max():.3f}")

    # ── STAGE G: Controlled Sharpening Only (applied on Stage D) ──
    lum = _luma(arr_d)
    blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=0.85)
    detail = lum - blur
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge_g = np.clip(np.hypot(gx, gy) * 3.0 - 0.05, 0.0, 1.0)
    sharp_delta = np.clip(detail * 0.35 * edge_g, -0.035, 0.035)
    arr_g = np.clip(arr_d + sharp_delta[:, :, None], 0.0, 1.0)
    Image.fromarray((arr_g * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_G_sharpening_only.png")
    luma_g = _luma(arr_g)
    print(f"Stage G (Sharpening Only): Luma={luma_g.mean():.3f}, Min={arr_g.min():.3f}, Max={arr_g.max():.3f}")

    # ── STAGE H: Final Validated Clean Pipeline (Native Resolution) ─
    # Combines Illumination + Denoise + Color + Subtle USM with ZERO 8x upscaling and ZERO synthetic hues
    y_e = cv2.cvtColor(arr_e, cv2.COLOR_RGB2YCrCb)
    # Gentle LAB CLAHE for local dynamic range and depth
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    y_eq = clahe.apply((y_e[:, :, 0] * 255).astype(np.uint8)).astype(np.float32) / 255.0
    y_e[:, :, 0] = np.clip(0.75 * y_e[:, :, 0] + 0.25 * y_eq, 0.0, 1.0)
    arr_clahe = np.clip(cv2.cvtColor(y_e, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

    # Subtle contrast curve (recovers rich depth and prevents washed-out look)
    s_curve = 0.5 - 0.5 * np.cos(np.pi * arr_clahe)
    arr_contrast = np.clip(0.88 * arr_clahe + 0.12 * s_curve, 0.0, 1.0)

    # Natural vibrance without synthetic hue shifts
    hsv_h = cv2.cvtColor(arr_contrast, cv2.COLOR_RGB2HSV)
    s_h = hsv_h[:, :, 1]
    deficit_h = np.maximum(0.42 - s_h, 0.0)
    hsv_h[:, :, 1] = np.clip(s_h + deficit_h * 0.22, 0.0, 0.70)
    arr_col = np.clip(cv2.cvtColor(hsv_h, cv2.COLOR_HSV2RGB), 0.0, 1.0)

    # Subtle guided acutance
    lum_h = _luma(arr_col)
    blur_h = cv2.GaussianBlur(lum_h, (0, 0), sigmaX=0.85)
    detail_h = lum_h - blur_h
    gx_h = cv2.Sobel(lum_h, cv2.CV_32F, 1, 0, ksize=3)
    gy_h = cv2.Sobel(lum_h, cv2.CV_32F, 0, 1, ksize=3)
    edge_h = np.clip(np.hypot(gx_h, gy_h) * 2.5 - 0.06, 0.0, 0.80)
    # Dark gate prevents noise amplification in dark floor
    dark_gate = np.clip((lum_h - 0.03) / 0.10, 0.0, 1.0)
    delta_h = np.clip(detail_h * 0.30 * edge_h * dark_gate, -0.030, 0.030)
    arr_h = np.clip(arr_col + delta_h[:, :, None], 0.0, 1.0)

    Image.fromarray((arr_h * 255).astype(np.uint8)).save(ABLAT_DIR / f"{prefix}_Stage_H_final_validated.png")
    luma_h = _luma(arr_h)
    hsv_final = cv2.cvtColor((arr_h * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    print(f"Stage H (Final Validated): Luma={luma_h.mean():.3f}, Sat_mean={hsv_final[:,:,1].mean():.1f}, Sat_P95={np.percentile(hsv_final[:,:,1], 95):.1f}")
    print(f"Outputs successfully saved to: {ABLAT_DIR}")


if __name__ == "__main__":
    run_stage_ablation(SCRATCH / "user_forest_input.png", "Case1_Landscape")
    run_stage_ablation(SCRATCH / "user_night_input.png", "Case2_NightStreet")

