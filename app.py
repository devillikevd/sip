"""
Advanced Streamlit Interface — Low-Light Image Enhancement
Using MIRNet and Deep Illumination Curve Fusion

Run:
    streamlit run app.py

Features:
 • Single or Batch image upload (up to 20 images)
 • Config A/B/C/D model selector with descriptions
 • Checkpoint upload or path input
 • Max-side resize slider (for CPU friendliness)
 • Per-image PSNR / SSIM vs optional ground-truth upload
 • Interactive before/after split-slider comparison
 • Illumination map viewer (Config C/D only)
 • Download single PNG or full batch as ZIP
 • GPU / CPU device badge + RTX info
 • Inference time + parameter count metrics
 • Session history table (last N enhancements)
 • Dark-themed custom CSS
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import time
import base64
import zipfile
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import streamlit.components.v1 as components
import torch

from models.hybrid_mirnet import CONFIG_DESCRIPTIONS, build_ablation_model


# ──────────────────────────────────────────────────────────
# Advanced Multi-Stage Low-Light Enhancer
# (MSRCR + CLAHE + Shadow Lift + Denoise + Color Restore)
# Works 100% out-of-the-box without any trained checkpoint
# ──────────────────────────────────────────────────────────
def _luma_f32(arr_f: np.ndarray) -> np.ndarray:
    """Compute Rec.601 float32 luminance without implicit float64 promotion."""
    return (
        np.float32(0.299) * arr_f[:, :, 0]
        + np.float32(0.587) * arr_f[:, :, 1]
        + np.float32(0.114) * arr_f[:, :, 2]
    ).astype(np.float32)


def _guided_filter_f32(
    guide: np.ndarray, src: np.ndarray, radius: int = 15, eps: float = 0.04
) -> np.ndarray:
    """Fast O(N) edge-preserving float32 Guided Filter (zero halo overshoot)."""
    try:
        import cv2
        g = guide.astype(np.float32, copy=False)
        s = src.astype(np.float32, copy=False)
        k = (2 * radius + 1, 2 * radius + 1)
        mean_I = cv2.boxFilter(g, cv2.CV_32F, k)
        mean_p = cv2.boxFilter(s, cv2.CV_32F, k)
        corr_I = cv2.boxFilter(g * g, cv2.CV_32F, k)
        corr_Ip = cv2.boxFilter(g * s, cv2.CV_32F, k)
        var_I = np.maximum(corr_I - mean_I * mean_I, np.float32(0.0)).astype(np.float32)
        cov_Ip = (corr_Ip - mean_I * mean_p).astype(np.float32)
        a = cov_Ip / (var_I + np.float32(eps))
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_32F, k)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, k)
        return np.clip(mean_a * g + mean_b, 1e-4, 1.0).astype(np.float32)
    except ImportError:
        from PIL import ImageFilter
        src_pil = Image.fromarray((np.clip(src, 0, 1) * 255).astype(np.uint8))
        return np.array(src_pil.filter(ImageFilter.GaussianBlur(radius=12)), dtype=np.float32) / np.float32(255.0)


def _estimate_noise_mad(lum_f: np.ndarray) -> float:
    """Estimate high-frequency luminance noise floor robustly via Laplacian MAD."""
    try:
        import cv2
        lap = cv2.Laplacian(lum_f.astype(np.float32, copy=False), cv2.CV_32F, ksize=3)
        sigma = float(np.median(np.abs(lap))) / 0.6745 / 2.828
        return float(np.clip(sigma, 0.0, 0.08))
    except ImportError:
        return 0.01


def _smoothstep_f32(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth Hermite interpolation from 0 to 1 in float32."""
    denom = max(float(edge1) - float(edge0), 1e-5)
    t = np.clip((x.astype(np.float32, copy=False) - np.float32(edge0)) / np.float32(denom), 0.0, 1.0).astype(np.float32)
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)


def _edge_confidence(lum_f: np.ndarray, sigma: float = 0.8, gain: float = 5.0, threshold: float = 0.025) -> np.ndarray:
    """Compute noise-immune edge confidence mask [0, 1] from smoothed Sobel gradients."""
    try:
        import cv2
        sm = cv2.GaussianBlur(lum_f.astype(np.float32, copy=False), (0, 0), sigmaX=max(float(sigma), 0.4))
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        return np.clip(np.hypot(gx, gy) * np.float32(gain) - np.float32(threshold), 0.0, 1.0).astype(np.float32)
    except ImportError:
        return np.zeros_like(lum_f, dtype=np.float32)


def _apply_hd_clarity_tanh(
    img_f: np.ndarray,
    clarity: float = 0.40,
    scale_factor: float = 1.0,
    noise_level: float = 0.0,
) -> np.ndarray:
    """Apply single-pass, halo-free, tanh-bounded multi-scale DoG HD edge tightening."""
    if clarity <= 0.01:
        return img_f
    try:
        import cv2
        y = np.clip(_luma_f32(img_f), 1e-4, 1.0).astype(np.float32)
        r_sc = min(max(float(scale_factor), 1.0), 3.0)
        sig_sm = (0.72 + min(noise_level * 10.0, 0.65)) * r_sc
        noise_gate = 0.022 + min(noise_level * 1.8, 0.06)
        e_mask = _edge_confidence(y, sigma=sig_sm, gain=5.4 + 0.4 * r_sc, threshold=noise_gate)
        lit_tex = _smoothstep_f32(0.06, 0.18, y) * (np.float32(0.06) if noise_level <= 0.012 else np.float32(0.0))
        tex_gate = np.maximum(e_mask, lit_tex).astype(np.float32)
        dog = (
            np.float32(0.68) * (y - cv2.GaussianBlur(y, (0, 0), sigmaX=0.68 * r_sc))
            + np.float32(0.32) * (y - cv2.GaussianBlur(y, (0, 0), sigmaX=1.80 * r_sc))
        ).astype(np.float32)
        lim = np.float32(0.098)
        detail = lim * np.tanh((np.float32(2.5 + 2.7 * clarity) * dog) / lim) * tex_gate
        y_hd = np.clip(y + detail, 0.0, 1.0).astype(np.float32)
        ratio = np.clip(y_hd / (y + np.float32(1e-4)), 0.74, 1.38)[:, :, None].astype(np.float32)
        return np.clip(img_f * ratio, 0.0, 1.0).astype(np.float32)
    except ImportError:
        return img_f


class ClassicalEnhancer:
    """
    Float32 High-Clarity Low-Light Image Enhancement Pipeline:
      1. Backlit night-scene & pedestal detection (keeps night skies clean, stops lamp smog)
      2. Structure-aware illumination estimation via adaptive Guided Filter
      3. Pre-lift chroma smoothing + pitch-dark tint neutralization
      4. Highlight-safe LIME/Retinex illumination & rational exposure fill-light
      5. Texture-preserving 3x3 NLM + dead-flat DCT de-blocking (zero leaf/road blur)
      6. Smooth 16-bit local contrast, de-veil & halo-free tanh HD edge clarity
    """

    def _guided_filter_gray(self, guide: np.ndarray, src: np.ndarray, radius: int = 15, eps: float = 0.04) -> np.ndarray:
        return _guided_filter_f32(guide, src, radius=radius, eps=eps)

    def _estimate_illumination(self, img_f: np.ndarray) -> np.ndarray:
        """LIME-inspired structure-aware illumination map with adaptive window radius."""
        h, w = img_f.shape[:2]
        t0 = np.max(img_f, axis=2).astype(np.float32)
        guide = _luma_f32(img_f)
        radius = max(12, min(32, min(h, w) // 10))
        t_refined = _guided_filter_f32(guide, t0, radius=radius, eps=0.045)
        return np.clip(np.maximum(t_refined, np.float32(0.75) * t0), 0.015, 1.0).astype(np.float32)

    def _pre_denoise_float(
        self, img_f: np.ndarray, illum: np.ndarray, res_sc: float, is_backlit_night: bool = False
    ) -> np.ndarray:
        """Suppress dark chroma noise & pitch-dark RGB tint bias before Retinex gain."""
        try:
            import cv2
            yc = cv2.cvtColor(img_f.astype(np.float32, copy=False), cv2.COLOR_RGB2YCrCb)
            sig_c = 2.2 * res_sc if is_backlit_night else 1.4 * res_sc
            yc[:, :, 1] = cv2.GaussianBlur(yc[:, :, 1], (0, 0), sigmaX=sig_c)
            yc[:, :, 2] = cv2.GaussianBlur(yc[:, :, 2], (0, 0), sigmaX=sig_c)
            if is_backlit_night:
                cg = _smoothstep_f32(0.020, 0.095, illum)
                yc[:, :, 1] = np.float32(0.5) + (yc[:, :, 1] - np.float32(0.5)) * cg
                yc[:, :, 2] = np.float32(0.5) + (yc[:, :, 2] - np.float32(0.5)) * cg
            return np.clip(cv2.cvtColor(yc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0).astype(np.float32)
        except ImportError:
            return img_f

    def _retinex_lift(
        self,
        img_f: np.ndarray,
        illum: np.ndarray,
        shadow_lift: float = 1.2,
        light_boost: float = 0.85,
        exposure_ev: float = 0.50,
        is_backlit_night: bool = False,
    ) -> np.ndarray:
        """Highlight-safe LIME/Retinex illumination & rational exposure curve with night-sky protection."""
        base_target = 0.44 + 0.14 * light_boost + 0.05 * exposure_ev
        if is_backlit_night:
            base_target = min(base_target, 0.40 + 0.08 * light_boost)
        eff_target = float(np.clip(base_target, 0.32, 0.72))
        scene_level = float(0.65 * np.mean(illum) + 0.35 * np.percentile(illum, 65))
        if scene_level >= eff_target:
            auto_gamma = float(np.clip(0.95 - 0.12 * light_boost, 0.68, 1.0))
        else:
            min_gamma = 0.35 if is_backlit_night else 0.18
            auto_gamma = float(np.clip(np.log(eff_target) / np.log(max(scene_level, 0.025)), min_gamma, 0.92))

        t_enh = np.power(illum, np.float32(auto_gamma))
        max_gain = (2.0 + 5.8 * shadow_lift) if is_backlit_night else (2.0 + 9.0 * shadow_lift)
        raw_gain = np.clip(t_enh / np.maximum(illum, np.float32(0.02)), 1.0, max_gain).astype(np.float32)

        if is_backlit_night:
            sky_gate = _smoothstep_f32(0.020, 0.105, illum)
            hl_prot = np.clip((np.float32(0.82) - illum) / np.float32(0.45), 0.0, 1.0).astype(np.float32)
            gain = (
                np.float32(1.0)
                + (raw_gain - np.float32(1.0))
                * (np.float32(0.14) + np.float32(0.86) * sky_gate)
                * (np.float32(0.30) + np.float32(0.70) * hl_prot)
            ).astype(np.float32)
        else:
            gain = raw_gain

        lifted = np.clip(img_f * gain[:, :, None], 0.0, 1.0).astype(np.float32)

        if shadow_lift > 0:
            k_log = 3.5 + 3.8 * shadow_lift
            log_boost = np.log1p(k_log * img_f) / np.log1p(k_log)
            dm = np.clip((np.float32(0.30) - illum) / np.float32(0.30), 0.0, 1.0).astype(np.float32)
            if is_backlit_night:
                dm = dm * (np.float32(0.14) + np.float32(0.86) * _smoothstep_f32(0.020, 0.105, illum))
            dm = (dm[:, :, None] ** 1.6).astype(np.float32)
            lifted = (lifted * (np.float32(1.0) - np.float32(0.22) * dm) + log_boost * (np.float32(0.22) * dm)).astype(np.float32)

        if exposure_ev > 0.01:
            eff_ev = exposure_ev * (0.58 if is_backlit_night else 1.0)
            k_exp = np.float32(0.65 * eff_ev)
            lum = np.clip(_luma_f32(lifted), 1e-4, 1.0).astype(np.float32)
            lum_new = (np.float32(1.0) + k_exp) * lum / (np.float32(1.0) + k_exp * lum)
            if is_backlit_night:
                ev_mask = _smoothstep_f32(0.025, 0.12, lum) * _smoothstep_f32(0.92, 0.42, lum)
                lum_new = lum + (lum_new - lum) * (np.float32(0.15) + np.float32(0.85) * ev_mask)
            ratio = np.clip(lum_new / lum, 1.0, float(1.0 + k_exp))[:, :, None].astype(np.float32)
            lifted = np.clip(lifted * ratio, 0.0, 1.0).astype(np.float32)

        return np.clip(lifted, 0.0, 1.0).astype(np.float32)

    def _white_balance_soft(self, img_f: np.ndarray, strength: float = 0.08) -> np.ndarray:
        """Soft gray-world white balance so natural ambient lighting stays realistic."""
        m_r, m_g, m_b = img_f[:, :, 0].mean(), img_f[:, :, 1].mean(), img_f[:, :, 2].mean()
        avg = (m_r + m_g + m_b) / 3.0 + 1e-6
        wb = np.clip(
            np.stack(
                [
                    img_f[:, :, 0] * (avg / (m_r + 1e-6)),
                    img_f[:, :, 1] * (avg / (m_g + 1e-6)),
                    img_f[:, :, 2] * (avg / (m_b + 1e-6)),
                ],
                axis=-1,
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        return np.clip((1.0 - strength) * img_f + strength * wb, 0.0, 1.0).astype(np.float32)

    def _denoise_float(
        self,
        img_f: np.ndarray,
        illum: np.ndarray,
        res_sc: float,
        noise_est: float,
        light_boost: float = 0.85,
        is_backlit_night: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Texture-preserving denoiser: wide chroma cleaning + 3x3 NLM + dead-flat DCT clean."""
        try:
            import cv2
            yc = cv2.cvtColor(img_f.astype(np.float32, copy=False), cv2.COLOR_RGB2YCrCb)
            sig_c = 2.6 * res_sc if (noise_est > 0.012 or is_backlit_night) else 1.3 * res_sc
            yc[:, :, 1] = cv2.GaussianBlur(yc[:, :, 1], (0, 0), sigmaX=sig_c)
            yc[:, :, 2] = cv2.GaussianBlur(yc[:, :, 2], (0, 0), sigmaX=sig_c)
            yc[:, :, 1] = cv2.bilateralFilter(yc[:, :, 1], d=5, sigmaColor=0.06, sigmaSpace=1.6 * res_sc)
            yc[:, :, 2] = cv2.bilateralFilter(yc[:, :, 2], d=5, sigmaColor=0.06, sigmaSpace=1.6 * res_sc)
            if is_backlit_night:
                cg2 = _smoothstep_f32(0.025, 0.11, illum)
                yc[:, :, 1] = np.float32(0.5) + (yc[:, :, 1] - np.float32(0.5)) * cg2
                yc[:, :, 2] = np.float32(0.5) + (yc[:, :, 2] - np.float32(0.5)) * cg2

            y_u8 = np.clip(yc[:, :, 0] * 255.0, 0, 255).round().astype(np.uint8)
            h_nlm = float(np.clip(2.4 + 145.0 * noise_est + 1.2 * max(light_boost - 0.8, 0.0), 2.2, 11.0))
            y_nlm = (
                cv2.fastNlMeansDenoising(y_u8, None, h=h_nlm, templateWindowSize=3, searchWindowSize=9).astype(np.float32)
                / np.float32(255.0)
            )
            r_sm = max(3, int(round(3.5 * res_sc)))
            y_flat = _guided_filter_f32(y_nlm, y_nlm, radius=r_sm, eps=0.004 + 0.18 * noise_est)

            edge_c = _edge_confidence(
                y_nlm, sigma=max(0.65, 0.55 * res_sc), gain=5.2, threshold=0.02 + 0.45 * noise_est
            )
            # Clean ONLY smooth dark walls/sky blocks; never smear trees, foliage, van, or textures!
            dead_flat = (_smoothstep_f32(0.20, 0.02, edge_c) * _smoothstep_f32(0.32, 0.04, illum)).astype(np.float32)
            y_dn = (y_nlm * (np.float32(1.0) - dead_flat) + y_flat * dead_flat).astype(np.float32)
            edge_keep = (edge_c * (np.float32(0.82) if noise_est <= 0.012 else np.float32(0.58))).astype(np.float32)
            yc[:, :, 0] = np.clip(yc[:, :, 0] * edge_keep + y_dn * (np.float32(1.0) - edge_keep), 0.0, 1.0).astype(np.float32)
            return np.clip(cv2.cvtColor(yc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0).astype(np.float32), edge_c
        except ImportError:
            edge_c = _edge_confidence(_luma_f32(img_f))
            return img_f, edge_c

    def _clahe_l_float(
        self,
        img_f: np.ndarray,
        edge_c: np.ndarray,
        res_sc: float,
        clarity: float = 0.40,
        clip_limit: float = 1.6,
        is_backlit_night: bool = False,
    ) -> np.ndarray:
        """Apply structural glare de-veiling + smooth 16-bit CLAHE gated cleanly above night floor."""
        try:
            import cv2
            if is_backlit_night:
                lum_v = np.clip(_luma_f32(img_f), 1e-4, 1.0).astype(np.float32)
                base_l = _guided_filter_f32(lum_v, lum_v, radius=max(10, int(round(14 * res_sc))), eps=0.02)
                veil_gate = _smoothstep_f32(0.03, 0.14, lum_v) * _smoothstep_f32(0.96, 0.55, lum_v)
                lum_dv = np.clip(lum_v + np.float32(0.24 + 0.24 * clarity) * (lum_v - base_l) * veil_gate, 0.0, 1.0).astype(np.float32)
                img_f = np.clip(img_f * np.clip(lum_dv / (lum_v + np.float32(1e-4)), 0.76, 1.32)[:, :, None], 0.0, 1.0).astype(np.float32)

            if clip_limit <= 0.1:
                return img_f
            lum_f = np.clip(_luma_f32(img_f), 1e-4, 1.0).astype(np.float32)
            y16 = np.clip(lum_f * 65535.0, 0, 65535).round().astype(np.uint16)
            clahe = cv2.createCLAHE(clipLimit=max(1.0, min(float(clip_limit), 1.9)), tileGridSize=(4, 4))
            y16_eq = clahe.apply(y16).astype(np.float32) / np.float32(65535.0)
            sky_gate = _smoothstep_f32(0.04, 0.14, lum_f) if is_backlit_night else np.float32(1.0)
            cw = ((np.float32(0.16) + np.float32(0.22) * edge_c) * sky_gate).astype(np.float32)
            y_eq = (lum_f * (np.float32(1.0) - cw) + y16_eq * cw).astype(np.float32)
            ratio = np.clip(y_eq / (lum_f + np.float32(1e-4)), 0.84, 1.30)[:, :, None].astype(np.float32)
            return np.clip(img_f * ratio, 0.0, 1.0).astype(np.float32)
        except ImportError:
            return img_f

    def _tone_and_color(
        self,
        img_f: np.ndarray,
        saturation: float = 1.15,
        warmth: float = 0.0,
        contrast: float = 0.14,
        is_backlit_night: bool = False,
    ) -> np.ndarray:
        """Apply shadow-protected midtone S-curve contrast, warmth shift, and smart vibrance."""
        eff_con = contrast * (1.30 if is_backlit_night else 1.0)
        if abs(eff_con) > 0.01:
            lum_c = np.clip(_luma_f32(img_f), 1e-4, 1.0).astype(np.float32)
            s_c = np.float32(0.5) - np.float32(0.5) * np.cos(np.float32(np.pi) * lum_c)
            prot = np.clip((lum_c - np.float32(0.04)) / np.float32(0.16), 0.0, 1.0).astype(np.float32)
            lum_sc = (lum_c * (np.float32(1.0) - np.float32(eff_con) * prot) + s_c * (np.float32(eff_con) * prot)).astype(np.float32)
            ratio = np.clip(lum_sc / (lum_c + np.float32(1e-4)), 0.80, 1.25)[:, :, None].astype(np.float32)
            img_f = np.clip(img_f * ratio, 0.0, 1.0).astype(np.float32)

        if abs(warmth) > 0.01:
            w_r, w_b = np.float32(1.0 + 0.28 * warmth), np.float32(1.0 - 0.28 * warmth)
            lb = _luma_f32(img_f)
            warmed = np.stack([img_f[:, :, 0] * w_r, img_f[:, :, 1], img_f[:, :, 2] * w_b], axis=-1).astype(np.float32)
            la = _luma_f32(warmed) + np.float32(1e-5)
            img_f = np.clip(warmed * ((lb + np.float32(1e-5)) / la)[:, :, None], 0.0, 1.0).astype(np.float32)

        if abs(saturation - 1.0) > 0.01:
            mx = np.max(img_f, axis=2, keepdims=True)
            mn = np.min(img_f, axis=2, keepdims=True)
            cur_sat = (mx - mn) / np.maximum(mx, np.float32(1e-4))
            s_gate = _smoothstep_f32(0.025, 0.12, _luma_f32(img_f))[:, :, None] if is_backlit_night else np.float32(1.0)
            vib = (np.float32(1.0) + np.float32(saturation - 1.0) * (np.float32(1.0) - np.float32(0.45) * cur_sat) * s_gate).astype(np.float32)
            g = _luma_f32(img_f)[:, :, None]
            img_f = np.clip(g + (img_f - g) * vib, 0.0, 1.0).astype(np.float32)

        return np.clip(img_f, 0.0, 1.0).astype(np.float32)

    def enhance(
        self,
        pil_img: Image.Image,
        shadow_lift: float = 1.2,
        clahe_clip: float = 1.6,
        denoise: bool = True,
        saturation: float = 1.15,
        clarity: float = 0.40,
        light_boost: float = 0.85,
        exposure_ev: float = 0.50,
        warmth: float = 0.0,
        contrast: float = 0.14,
        gamma: Optional[float] = None,
        apply_sharpen: bool = True,
    ) -> dict:
        """Run float32 HD-clarity + Fill-Light + Noise-Adaptive low-light enhancement pipeline."""
        if gamma is not None:
            light_boost = float(np.clip((0.85 - gamma) * 2.2, 0.0, 2.0))
        t0 = time.perf_counter()
        img_u8 = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        img_f = (img_u8.astype(np.float32) / np.float32(255.0))
        h0, w0 = img_f.shape[:2]
        res_sc = min(max(max(h0, w0) / 480.0, 1.0), 3.0)

        t0_raw = np.max(img_f, axis=2).astype(np.float32)
        p99_raw = float(np.percentile(t0_raw, 99))
        p25_raw = float(np.percentile(t0_raw, 25))
        p02_raw = float(np.percentile(t0_raw, 2))
        is_backlit_night = p99_raw > 0.75 and p25_raw < 0.14

        if is_backlit_night and p02_raw > 0.025:
            ped = min(p02_raw * 0.72, 0.045)
            img_f = np.clip((img_f - np.float32(ped)) / np.float32(1.0 - ped), 0.0, 1.0).astype(np.float32)

        illum = self._estimate_illumination(img_f)
        clean_f = self._pre_denoise_float(img_f, illum, res_sc, is_backlit_night=is_backlit_night) if denoise else img_f

        lifted_f = self._retinex_lift(
            clean_f,
            illum,
            shadow_lift=shadow_lift,
            light_boost=light_boost,
            exposure_ev=exposure_ev,
            is_backlit_night=is_backlit_night,
        )
        wb_f = self._white_balance_soft(lifted_f, strength=0.08)

        lum_pre = _luma_f32(wb_f)
        noise_est = _estimate_noise_mad(lum_pre)
        if denoise:
            dn_f, edge_c = self._denoise_float(
                wb_f, illum, res_sc, noise_est, light_boost=light_boost, is_backlit_night=is_backlit_night
            )
        else:
            dn_f = wb_f
            edge_c = _edge_confidence(lum_pre)

        clahe_f = self._clahe_l_float(
            dn_f, edge_c, res_sc=res_sc, clarity=clarity, clip_limit=clahe_clip, is_backlit_night=is_backlit_night
        )
        toned_f = self._tone_and_color(
            clahe_f, saturation=saturation, warmth=warmth, contrast=contrast, is_backlit_night=is_backlit_night
        )
        res_noise = _estimate_noise_mad(_luma_f32(toned_f))

        final_f = _apply_hd_clarity_tanh(toned_f, clarity=clarity, scale_factor=1.0, noise_level=res_noise) if apply_sharpen else toned_f
        t1 = time.perf_counter()

        illum_u8 = (np.clip(illum, 0, 1) * 255).round().astype(np.uint8)
        enh_pil = Image.fromarray((np.clip(final_f, 0, 1) * 255).round().astype(np.uint8))

        return {
            "enhanced_pil": enh_pil,
            "illum_arr": illum_u8,
            "infer_ms": (t1 - t0) * 1000.0,
            "total_params": 0,
            "loaded_ckpt": "Retinex+CLAHE+HD Pipeline",
            "noise_level": res_noise,
        }


_classical_enhancer = ClassicalEnhancer()

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_LOW_DIR = ROOT_DIR / "baseline_results" / "samples" / "low"
RESULTS_DIR = ROOT_DIR / "Results" / "ui_outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_COLORS = {"A": "#F4D03F", "B": "#5DADE2", "C": "#58D68D", "D": "#E74C3C"}
CONFIG_ICONS  = {"A": "💡", "B": "🔬", "C": "⚡", "D": "🚀"}

HISTORY_KEY = "enhancement_history"
MAX_HISTORY = 10

# ──────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌙 Team - Vijendra, Vyom, Gautam | 4K Low-Light Studio",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Team - Vijendra, Vyom, Gautam | Low-Light Image Enhancement 4K AI Studio (2026)",
    },
)

# ──────────────────────────────────────────────────────────
# Custom CSS — Flagship VisionOS Prism-Glass 4K AI Studio Theme
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

/* ---- Animated Multi-Layer Aurora Backdrop ---- */
@keyframes auroraShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pillGlow {
    0%, 100% { box-shadow: 0 6px 24px rgba(56, 189, 248, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.45); }
    50%      { box-shadow: 0 6px 30px rgba(168, 85, 247, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.60); }
}

.stApp {
    background:
        radial-gradient(circle 720px at 8% 10%, rgba(56, 189, 248, 0.24), transparent 62%),
        radial-gradient(circle 660px at 90% 8%, rgba(168, 85, 247, 0.28), transparent 62%),
        radial-gradient(circle 580px at 52% 45%, rgba(251, 146, 60, 0.14), transparent 58%),
        radial-gradient(circle 700px at 84% 92%, rgba(14, 165, 233, 0.20), transparent 64%),
        radial-gradient(circle 600px at 14% 88%, rgba(244, 63, 94, 0.18), transparent 62%),
        linear-gradient(155deg, #040713 0%, #091026 50%, #050914 100%) !important;
    background-attachment: fixed !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    max-width: 100% !important;
    padding: 0.75rem 2.2rem 2.2rem 2.2rem !important;
}
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', 'Segoe UI', system-ui, sans-serif;
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.4); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.22); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(125, 211, 252, 0.45); }

/* ---- Floating Bento Glass Containers ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.085), rgba(255, 255, 255, 0.02)) !important;
    backdrop-filter: blur(26px) saturate(190%) !important;
    -webkit-backdrop-filter: blur(26px) saturate(190%) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.32) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 22px !important;
    padding: 18px 22px !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

/* ---- Frosted Crystal Hero Card ---- */
.hero-card {
    position: relative;
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.13) 0%,
        rgba(255, 255, 255, 0.04) 48%,
        rgba(125, 211, 252, 0.09) 100%
    );
    background-size: 200% 200%;
    animation: auroraShift 14s ease infinite;
    backdrop-filter: blur(32px) saturate(210%);
    -webkit-backdrop-filter: blur(32px) saturate(210%);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-top: 1.5px solid rgba(255, 255, 255, 0.45);
    border-left: 1.5px solid rgba(255, 255, 255, 0.32);
    border-radius: 26px;
    padding: 28px 34px 24px 34px;
    margin-bottom: 16px;
    box-shadow:
        0 28px 70px rgba(0, 0, 0, 0.52),
        0 0 40px rgba(56, 189, 248, 0.10),
        inset 0 1px 2px rgba(255, 255, 255, 0.38),
        inset 0 -1px 1px rgba(255, 255, 255, 0.08);
    text-align: center;
    overflow: hidden;
}
.hero-team-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.18), rgba(168, 85, 247, 0.22));
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.40);
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14.5px;
    font-weight: 700;
    letter-spacing: 0.9px;
    padding: 7px 24px;
    border-radius: 999px;
    margin-bottom: 14px;
    animation: pillGlow 4.5s ease-in-out infinite;
}
.hero-title {
    font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    line-height: 1.12;
    margin: 2px 0 16px 0;
    background: linear-gradient(92deg, #ffffff 0%, #bae6fd 28%, #fde68a 60%, #fbcfe8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 6px;
}
.feature-pill {
    background: rgba(255, 255, 255, 0.09);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-top: 1px solid rgba(255, 255, 255, 0.40);
    color: #f8fafc;
    font-size: 12.5px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 999px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}

/* ---- Glass Status Ribbon ---- */
.glass-ribbon {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.03));
    backdrop-filter: blur(24px) saturate(185%);
    -webkit-backdrop-filter: blur(24px) saturate(185%);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-top: 1px solid rgba(255, 255, 255, 0.34);
    border-radius: 18px;
    padding: 11px 20px;
    margin-bottom: 16px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.22);
}

/* ---- Photo Showcase Tag Headers ---- */
.photo-tag {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 16px;
    margin-bottom: 10px;
    border-radius: 13px;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-top: 1px solid rgba(255, 255, 255, 0.35);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.photo-tag-orig {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(148, 163, 184, 0.06));
    color: #e2e8f0;
}
.photo-tag-enh {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(168, 85, 247, 0.16));
    border-color: rgba(125, 211, 252, 0.45);
    color: #ffffff;
}
.photo-tag-meta {
    font-size: 12px;
    font-weight: 600;
    opacity: 0.88;
}

/* ---- Team Spotlight Cards ---- */
.team-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 14px 0 22px 0;
}
.team-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.03));
    backdrop-filter: blur(22px) saturate(185%);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-top: 1px solid rgba(255, 255, 255, 0.36);
    border-radius: 18px;
    padding: 20px 18px;
    text-align: center;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.team-avatar {
    width: 48px; height: 48px; line-height: 48px;
    margin: 0 auto 10px auto;
    border-radius: 50%;
    font-size: 20px; font-weight: 800; color: #fff;
    background: linear-gradient(135deg, #38bdf8, #8b5cf6);
    box-shadow: 0 8px 20px rgba(56, 189, 248, 0.35);
}
.team-name { font-size: 17px; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
.team-role { font-size: 12.5px; color: #cbd5e1; }

/* ---- Frosted Glass Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(12, 18, 36, 0.74) 0%, rgba(10, 15, 30, 0.62) 100%) !important;
    backdrop-filter: blur(28px) saturate(190%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.16) !important;
}
.sidebar-team-box {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.16), rgba(125, 211, 252, 0.14));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.32);
    border-radius: 14px;
    padding: 10px 12px;
    margin-top: 6px;
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.35);
}

/* ---- Liquid Glass Metric Cards ---- */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.03)) !important;
    backdrop-filter: blur(24px) saturate(190%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.36) !important;
    border-radius: 18px !important;
    padding: 16px 20px !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.24) !important;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-color: rgba(125, 211, 252, 0.65) !important;
    box-shadow: 0 18px 40px rgba(56, 189, 248, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.38) !important;
}

/* ---- Frosted Glass File Uploader Dropzone ---- */
section[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.025)) !important;
    backdrop-filter: blur(22px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(180%) !important;
    border: 1.5px dashed rgba(255, 255, 255, 0.32) !important;
    border-radius: 18px !important;
    padding: 16px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.22) !important;
    transition: all 0.25s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(125, 211, 252, 0.85) !important;
    background: linear-gradient(145deg, rgba(56, 189, 248, 0.16), rgba(168, 85, 247, 0.12)) !important;
    transform: translateY(-2px);
}

/* ---- Glossy Glass Buttons ---- */
div.stButton > button, div.stDownloadButton > button {
    border-radius: 14px;
    font-weight: 700;
    font-size: 14.5px;
    padding: 12px 22px;
    transition: all 0.22s ease;
    backdrop-filter: blur(18px);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.94), rgba(236, 72, 153, 0.90), rgba(139, 92, 246, 0.90));
    background-size: 180% 180%;
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.40);
    box-shadow:
        0 12px 32px rgba(236, 72, 153, 0.34),
        inset 0 1px 1px rgba(255, 255, 255, 0.50);
}
div.stButton > button[kind="primary"]:hover {
    background-position: right center;
    transform: translateY(-2px);
    box-shadow: 0 16px 38px rgba(249, 115, 22, 0.48), inset 0 1px 1px rgba(255, 255, 255, 0.60);
}
div.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.04));
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-top: 1px solid rgba(255, 255, 255, 0.40);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
}
div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(125, 211, 252, 0.72);
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.24), rgba(168, 85, 247, 0.18));
    transform: translateY(-2px);
}
div.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.90), rgba(99, 102, 241, 0.90));
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.38);
    box-shadow: 0 12px 28px rgba(59, 130, 246, 0.36), inset 0 1px 1px rgba(255, 255, 255, 0.48);
}
div.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.98), rgba(129, 140, 248, 0.98));
    transform: translateY(-2px);
}

/* ---- Glassy Showcase Image Frames ---- */
.stImage img {
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.26);
    box-shadow: 0 22px 52px rgba(0, 0, 0, 0.58);
}
.stImage > div > div > p {
    text-align: center;
    font-size: 12.5px;
    color: #cbd5e1;
    margin-top: 7px;
}

/* ---- Full-Width Glass Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.03));
    backdrop-filter: blur(24px) saturate(190%);
    -webkit-backdrop-filter: blur(24px) saturate(190%);
    padding: 8px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-top: 1px solid rgba(255, 255, 255, 0.34);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.34);
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    border-radius: 12px;
    padding: 11px 18px;
    font-weight: 600;
    color: #cbd5e1;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.22), rgba(125, 211, 252, 0.18)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.42) !important;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.45) !important;
}

/* ---- Glass Expanders & Headings ---- */
h2 { color: #f8fafc; border-bottom: 1px solid rgba(255, 255, 255, 0.16); padding-bottom: 8px; }
h3 { color: #f1f5f9; }
details {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.025)) !important;
    backdrop-filter: blur(22px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.32) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28) !important;
}
details summary { font-weight: 600; color: #f8fafc; }

/* ---- Frosted Status Badges ---- */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    backdrop-filter: blur(14px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.26);
}
.badge-green  { background:rgba(34, 197, 94, 0.22); color:#86efac; border:1px solid rgba(134, 239, 172, 0.48); }
.badge-blue   { background:rgba(56, 189, 248, 0.22); color:#7dd3fc; border:1px solid rgba(125, 211, 252, 0.48); }
.badge-red    { background:rgba(248, 113, 113, 0.22); color:#fca5a5; border:1px solid rgba(252, 165, 165, 0.48); }
.badge-yellow { background:rgba(251, 191, 36, 0.22); color:#fde68a; border:1px solid rgba(253, 230, 138, 0.48); }
.badge-purple { background:rgba(192, 132, 252, 0.22); color:#e9d5ff; border:1px solid rgba(233, 213, 255, 0.48); }

/* ---- Glass Tables ---- */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.16);
}
th { background: rgba(255, 255, 255, 0.11); color: #e2e8f0; padding: 11px 14px; text-align: left; }
td { padding: 10px 14px; border-top: 1px solid rgba(255, 255, 255, 0.08); color: #f8fafc; }
tr:hover td { background: rgba(255, 255, 255, 0.08); }

/* ---- Glass Footer ---- */
.studio-footer {
    text-align: center;
    color: #cbd5e1;
    font-size: 12.5px;
    padding: 15px 20px;
    margin-top: 28px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-top: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(config: str, checkpoint_path: Optional[str], device_str: str):
    device = torch.device(device_str)
    model = build_ablation_model(config).to(device).eval()
    loaded_ckpt = None
    if checkpoint_path and Path(checkpoint_path).is_file():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(sd)
        loaded_ckpt = checkpoint_path
    total_params = sum(p.numel() for p in model.parameters())
    return model, total_params, loaded_ckpt


def tensor_from_pil(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    arr = (
        torch.clamp(tensor.squeeze(0).detach().cpu(), 0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two HxWx3 uint8 arrays."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple mean-based SSIM estimate (full SSIM needs scikit-image)."""
    try:
        from skimage.metrics import structural_similarity as ssim_sk
        return float(ssim_sk(img1, img2, channel_axis=2, data_range=255))
    except ImportError:
        a = img1.astype(np.float32)
        b = img2.astype(np.float32)
        mu_a, mu_b = a.mean(), b.mean()
        s_a, s_b = a.std(), b.std()
        s_ab = np.mean((a - mu_a) * (b - mu_b))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return float(
            (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
            / ((mu_a**2 + mu_b**2 + c1) * (s_a**2 + s_b**2 + c2))
        )


def make_split_comparison(
    orig: Image.Image, enh: Image.Image, split_pct: int, divider_color=(255, 220, 60)
) -> Image.Image:
    orig_np = np.array(orig.convert("RGB"))
    enh_np  = np.array(enh.convert("RGB").resize(orig.size, Image.BICUBIC))
    h, w    = orig_np.shape[:2]
    cut     = int(round(w * split_pct / 100.0))
    merged  = enh_np.copy()
    merged[:, :cut] = orig_np[:, :cut]
    if 1 <= cut < w - 1:
        merged[:, cut - 2 : cut + 2] = divider_color
    # Add labels
    out_img = Image.fromarray(merged)
    draw = ImageDraw.Draw(out_img)
    label_h = max(18, h // 20)
    try:
        font = ImageFont.truetype("arial.ttf", label_h)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 4), "ORIGINAL", fill=(255, 220, 60), font=font)
    draw.text((cut + 6, 4), "ENHANCED", fill=(80, 220, 100), font=font)
    return out_img


def _pil_to_data_uri(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render_curtain_comparator(orig_pil: Image.Image, enh_pil: Image.Image, res_tag: str = "4K UHD"):
    """Render a smooth 60fps mouse/touch draggable Before/After curtain comparator via HTML5/CSS/JS."""
    disp_enh = resize_for_display(enh_pil, 1280)
    disp_orig = orig_pil.resize(disp_enh.size, Image.LANCZOS)
    w, h = disp_enh.size
    aspect_pct = max(35.0, min(85.0, (h / max(w, 1)) * 100.0))
    orig_uri = _pil_to_data_uri(disp_orig)
    enh_uri = _pil_to_data_uri(disp_enh)

    html_code = f"""
    <!DOCTYPE html><html><head><style>
        html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; width:100%; height:100%; }}
    </style></head><body>
    <div id="curtain-wrap" style="
        position:relative; width:100%; height:100%;
        overflow:hidden; border-radius:18px;
        border:1px solid rgba(255,255,255,0.28);
        box-shadow:0 20px 55px rgba(0,0,0,0.6), 0 0 30px rgba(56,189,248,0.14);
        user-select:none; touch-action:none; cursor:ew-resize; background:#050816;">
        <img src="{enh_uri}" draggable="false" style="
            position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none;" />
        <div id="orig-layer" style="
            position:absolute; inset:0; width:100%; height:100%; overflow:hidden;
            clip-path:inset(0 50% 0 0); pointer-events:none;">
            <img src="{orig_uri}" draggable="false" style="
                position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none;" />
        </div>
        <div style="position:absolute; top:14px; left:16px; padding:6px 14px; border-radius:999px;
            background:rgba(15,23,42,0.72); backdrop-filter:blur(12px);
            border:1px solid rgba(255,255,255,0.28); color:#f8fafc;
            font:700 12px 'Plus Jakarta Sans',sans-serif; letter-spacing:0.5px; pointer-events:none;">
            🌑 DARK ORIGINAL
        </div>
        <div style="position:absolute; top:14px; right:16px; padding:6px 14px; border-radius:999px;
            background:linear-gradient(135deg,rgba(56,189,248,0.35),rgba(168,85,247,0.35));
            backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,0.38); color:#ffffff;
            font:700 12px 'Plus Jakarta Sans',sans-serif; letter-spacing:0.5px; pointer-events:none;">
            ✨ {res_tag} ENHANCED
        </div>
        <div id="curtain-bar" style="
            position:absolute; top:0; bottom:0; left:50%; width:3px;
            background:linear-gradient(180deg,#38bdf8,#fde68a,#f472b6);
            box-shadow:0 0 18px rgba(125,211,252,0.95); transform:translateX(-50%); pointer-events:none;">
            <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                width:44px; height:44px; border-radius:50%;
                background:rgba(255,255,255,0.22); backdrop-filter:blur(16px);
                border:2px solid rgba(255,255,255,0.85); color:#ffffff;
                display:flex; align-items:center; justify-content:center;
                font:800 13px sans-serif; box-shadow:0 6px 24px rgba(0,0,0,0.45);">
                ◀ ▶
            </div>
        </div>
    </div>
    <script>
    (function() {{
        const wrap = document.getElementById('curtain-wrap');
        const orig = document.getElementById('orig-layer');
        const bar  = document.getElementById('curtain-bar');
        let dragging = false;
        function update(clientX) {{
            const rect = wrap.getBoundingClientRect();
            let pct = ((clientX - rect.left) / Math.max(rect.width, 1)) * 100;
            pct = Math.max(0, Math.min(100, pct));
            orig.style.clipPath = 'inset(0 ' + (100 - pct).toFixed(2) + '% 0 0)';
            bar.style.left = pct.toFixed(2) + '%';
        }}
        wrap.addEventListener('pointerdown', e => {{ dragging = true; wrap.setPointerCapture(e.pointerId); update(e.clientX); }});
        wrap.addEventListener('pointermove', e => {{ if (dragging) update(e.clientX); }});
        wrap.addEventListener('pointerup',   e => {{ dragging = false; }});
        wrap.addEventListener('pointercancel',() => {{ dragging = false; }});
    }})();
    </script>
    </body></html>
    """
    approx_px = int(min(620, max(360, 920 * (h / max(w, 1)) + 12)))
    components.html(html_code, height=approx_px)


def render_loupe_inspector(orig_pil: Image.Image, enh_pil: Image.Image):
    """60fps interactive 3.0x circular magnifying lens for inspecting 4K clarity."""
    disp_enh = resize_for_display(enh_pil, 1600)
    w, h = disp_enh.size
    enh_uri = _pil_to_data_uri(disp_enh)
    approx_px = int(min(620, max(360, 900 * (h / max(w, 1)) + 20)))

    html_code = f"""
    <!DOCTYPE html><html><head><style>
        html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; width:100%; height:100%;
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }}
    </style></head><body>
    <div id="loupe-box" style="position:relative; width:100%; height:100%; overflow:hidden; border-radius:18px;
        border:1px solid rgba(255,255,255,0.28); background:#050816; cursor:crosshair;">
        <img id="base-img" src="{enh_uri}" draggable="false" style="
            position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none;" />
        <div style="position:absolute; top:14px; left:16px; padding:6px 14px; border-radius:999px;
            background:rgba(15,23,42,0.75); backdrop-filter:blur(12px);
            border:1px solid rgba(255,255,255,0.28); color:#e2e8f0; font-size:12px; font-weight:700; pointer-events:none;">
            🔍 Move mouse over image to inspect 3× 4K Detail
        </div>
        <div id="lens" style="display:none; position:absolute; width:190px; height:190px; border-radius:50%;
            border:2.5px solid rgba(125,211,252,0.92); box-shadow:0 14px 40px rgba(0,0,0,0.7), 0 0 24px rgba(56,189,248,0.4);
            pointer-events:none; overflow:hidden; transform:translate(-50%,-50%);
            background-image:url('{enh_uri}'); background-repeat:no-repeat;">
            <div style="position:absolute; bottom:8px; left:50%; transform:translateX(-50%);
                background:rgba(15,23,42,0.82); color:#7dd3fc; font-size:10.5px; font-weight:800;
                padding:2px 9px; border-radius:999px; border:1px solid rgba(125,211,252,0.4);">
                3.0× 4K LOUPE
            </div>
        </div>
    </div>
    <script>
    (function() {{
        const box = document.getElementById('loupe-box');
        const img = document.getElementById('base-img');
        const lens = document.getElementById('lens');
        const zoom = 3.0;
        function move(cx, cy) {{
            const bRect = box.getBoundingClientRect();
            const iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
            const scale = Math.min(bRect.width / iw, bRect.height / ih);
            const rw = iw * scale, rh = ih * scale;
            const ox = (bRect.width - rw) / 2, oy = (bRect.height - rh) / 2;
            const x = cx - bRect.left, y = cy - bRect.top;
            if (x < ox || x > ox + rw || y < oy || y > oy + rh) {{ lens.style.display = 'none'; return; }}
            lens.style.display = 'block';
            lens.style.left = x + 'px';
            lens.style.top = y + 'px';
            const rx = (x - ox) / rw, ry = (y - oy) / rh;
            lens.style.backgroundSize = (rw * zoom) + 'px ' + (rh * zoom) + 'px';
            lens.style.backgroundPosition = (-(rx * rw * zoom - 95)) + 'px ' + (-(ry * rh * zoom - 95)) + 'px';
        }}
        box.addEventListener('pointermove', e => move(e.clientX, e.clientY));
        box.addEventListener('pointerdown', e => move(e.clientX, e.clientY));
        box.addEventListener('pointerleave', () => {{ lens.style.display = 'none'; }});
    }})();
    </script>
    </body></html>
    """
    components.html(html_code, height=approx_px)


def build_luminance_histogram_svg(orig_np: np.ndarray, enh_np: np.ndarray) -> str:
    """Fast frosted-glass inline SVG comparing Original vs Enhanced luminance distribution."""
    lum_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    orig_l = np.clip((orig_np @ lum_w), 0, 255)
    enh_l = np.clip((enh_np @ lum_w), 0, 255)
    bins = 48
    h_orig, _ = np.histogram(orig_l, bins=bins, range=(0, 255), density=True)
    h_enh, _ = np.histogram(enh_l, bins=bins, range=(0, 255), density=True)
    max_v = max(float(h_orig.max()), float(h_enh.max()), 1e-6)

    w_svg, h_svg, pad = 760, 230, 24
    plot_w, plot_h = w_svg - 2 * pad, h_svg - 2 * pad - 18

    def _pts(hist):
        pts = [f"{pad:.1f},{pad + plot_h:.1f}"]
        for idx, val in enumerate(hist):
            x = pad + (idx / (bins - 1)) * plot_w
            y = pad + plot_h - (float(val) / max_v) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        pts.append(f"{pad + plot_w:.1f},{pad + plot_h:.1f}")
        return " ".join(pts)

    return f"""
    <div style="background:linear-gradient(145deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02));
        backdrop-filter:blur(20px); border:1px solid rgba(255,255,255,0.18);
        border-radius:18px; padding:16px 20px; margin-top:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="color:#f8fafc;font-weight:700;font-size:14px;">📊 Luminance Distribution (Dark Shadows → Bright Daylight)</span>
            <span style="font-size:12px;color:#cbd5e1;">
                <span style="color:#f43f5e;">■ Original Low-Light</span> &nbsp;&nbsp;
                <span style="color:#38bdf8;">■ 4K Enhanced Output</span>
            </span>
        </div>
        <svg viewBox="0 0 {w_svg} {h_svg}" width="100%" style="display:block;">
            <defs>
                <linearGradient id="gOrig" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#f43f5e" stop-opacity="0.55"/>
                    <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.05"/>
                </linearGradient>
                <linearGradient id="gEnh" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.60"/>
                    <stop offset="100%" stop-color="#a855f7" stop-opacity="0.06"/>
                </linearGradient>
            </defs>
            <line x1="{pad}" y1="{pad + plot_h}" x2="{pad + plot_w}" y2="{pad + plot_h}" stroke="rgba(255,255,255,0.22)" stroke-width="1"/>
            <polygon points="{_pts(h_orig)}" fill="url(#gOrig)" stroke="#fb7185" stroke-width="2"/>
            <polygon points="{_pts(h_enh)}" fill="url(#gEnh)" stroke="#38bdf8" stroke-width="2.2"/>
            <text x="{pad}" y="{h_svg - 6}" fill="#94a3b8" font-size="11">0 (Pitch Black)</text>
            <text x="{w_svg // 2 - 28}" y="{h_svg - 6}" fill="#94a3b8" font-size="11">128 (Midtones)</text>
            <text x="{pad + plot_w - 75}" y="{h_svg - 6}" fill="#94a3b8" font-size="11">255 (Bright)</text>
        </svg>
    </div>
    """


def resize_for_display(pil: Image.Image, max_side: int = 800) -> Image.Image:
    w, h = pil.size
    if max(w, h) <= max_side:
        return pil
    scale = max_side / float(max(w, h))
    return pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def resize_for_processing(pil: Image.Image, max_side: int) -> Image.Image:
    w, h = pil.size
    if max(w, h) <= max_side:
        return pil
    scale = max_side / float(max(w, h))
    return pil.resize(
        (max(8, int(round(w * scale))), max(8, int(round(h * scale)))), Image.LANCZOS
    )


def upscale_to_hd_crisp(
    pil: Image.Image,
    target_long_side: int = 1920,
    clarity: float = 0.40,
    noise_level: Optional[float] = None,
) -> Image.Image:
    """Upscale images cleanly to HD/4K with zero source pre-blur, ringing, or grid tearing.

    Uses continuous Bicubic resampling followed by a single scale- and noise-adaptive
    halo-free tanh-bounded HD edge tightening pass at target resolution.
    """
    w, h = pil.size
    long_side = max(w, h)
    if target_long_side <= 0 or long_side >= target_long_side:
        return pil

    target_scale = target_long_side / float(long_side)
    target_size = (max(8, int(round(w * target_scale))), max(8, int(round(h * target_scale))))

    try:
        up_pil = pil.convert("RGB").resize(target_size, Image.BICUBIC)
        up_f = np.array(up_pil, dtype=np.float32) / np.float32(255.0)
        nl = _estimate_noise_mad(_luma_f32(up_f)) if noise_level is None else float(noise_level)

        hd_f = _apply_hd_clarity_tanh(
            up_f, clarity=clarity, scale_factor=target_scale, noise_level=nl
        )
        return Image.fromarray((np.clip(hd_f, 0, 1) * 255.0).round().astype(np.uint8))
    except ImportError:
        from PIL import ImageFilter
        up = pil.resize(target_size, Image.BICUBIC)
        return up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=75, threshold=3))


def pil_to_bytes(pil: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    fmt_u = fmt.upper()
    if fmt_u == "PNG":
        pil.save(buf, format="PNG", compress_level=1)
    elif fmt_u in ("JPG", "JPEG"):
        pil.convert("RGB").save(buf, format="JPEG", quality=98, subsampling=0)
    else:
        pil.save(buf, format=fmt)
    return buf.getvalue()


def make_zip(entries: list[tuple[str, bytes]]) -> bytes:
    """entries: list of (filename, bytes)"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def device_badge_html(device_str: str) -> str:
    if device_str == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        return f'<span class="badge badge-green">🟢 GPU · {name}</span>'
    return '<span class="badge badge-blue">🔵 CPU</span>'


def checkpoint_badge_html(loaded: bool) -> str:
    if loaded:
        return '<span class="badge badge-green">✅ Checkpoint Loaded</span>'
    return '<span class="badge badge-red">⚠️ No Checkpoint (init weights)</span>'


def file_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]


LIGHT_PRESETS = {
    "Balanced HD (Natural)": {
        "light_boost": 0.85,
        "exposure_ev": 0.50,
        "shadow_lift": 1.2,
        "clarity": 0.40,
        "clahe_clip": 1.6,
        "saturation": 1.15,
        "warmth": 0.0,
        "contrast": 0.14,
    },
    "Bright Studio Light ☀️": {
        "light_boost": 1.35,
        "exposure_ev": 0.95,
        "shadow_lift": 1.6,
        "clarity": 0.45,
        "clahe_clip": 1.8,
        "saturation": 1.18,
        "warmth": 0.05,
        "contrast": 0.16,
    },
    "Extreme Night Rescue 🔥": {
        "light_boost": 1.85,
        "exposure_ev": 1.45,
        "shadow_lift": 2.2,
        "clarity": 0.45,
        "clahe_clip": 2.0,
        "saturation": 1.20,
        "warmth": 0.05,
        "contrast": 0.18,
    },
}


OUTPUT_RES_MAP = {
    "4K UHD (3840px Crisp)": 3840,
    "Full HD (1920px)": 1920,
    "Native Resolution": 0,
}


@st.cache_data(show_spinner=False)
def run_inference_cached(
    image_bytes: bytes,
    config: str,
    ckpt_path: Optional[str],
    device_str: str,
    max_side: int,
    mode: str = "auto",   # "auto" | "classical" | "deep"
    light_boost: float = 0.85,
    exposure_ev: float = 0.50,
    shadow_lift: float = 1.2,
    clahe_clip: float = 1.6,
    denoise: bool = True,
    saturation: float = 1.15,
    clarity: float = 0.40,
    warmth: float = 0.0,
    contrast: float = 0.14,
    full_res: bool = True,
    output_res: str = "4K UHD (3840px Crisp)",
) -> dict:
    """Cached inference so repeated clicks don't re-run."""
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    target_px = OUTPUT_RES_MAP.get(output_res, 3840)

    # Decide effective mode:
    # 'auto' uses trained deep checkpoint if provided; otherwise uses
    # Retinex+FillLight+CLAHE+4K Clarity so dark photos come out ultra-bright and 4K crisp.
    effective_mode = mode
    if mode == "auto":
        effective_mode = "deep" if (ckpt_path and Path(ckpt_path).is_file()) else "classical"

    # Keep FULL native resolution up to 4096px in Classical/Auto unless user disables Preserve Full HD
    if effective_mode == "classical":
        proc = resize_for_processing(pil, 4096) if full_res else resize_for_processing(pil, max_side)
        will_upscale = target_px > 0 and max(proc.size) < target_px
        result = _classical_enhancer.enhance(
            proc,
            light_boost=light_boost,
            exposure_ev=exposure_ev,
            shadow_lift=shadow_lift,
            clahe_clip=clahe_clip,
            denoise=denoise,
            saturation=saturation,
            clarity=clarity,
            warmth=warmth,
            contrast=contrast,
            apply_sharpen=not will_upscale,
        )
        t_up0 = time.perf_counter()
        if will_upscale:
            result["enhanced_pil"] = upscale_to_hd_crisp(
                result["enhanced_pil"],
                target_long_side=target_px,
                clarity=clarity,
                noise_level=result.get("noise_level"),
            )
        t_up1 = time.perf_counter()
        result["infer_ms"] += (t_up1 - t_up0) * 1000.0
        result["original_pil"] = proc
        result["proc_size"]    = result["enhanced_pil"].size
        res_tag = "4K UHD" if target_px >= 3840 else ("Full HD" if target_px >= 1920 else "Native")
        result["mode_used"]    = f"Retinex + Fill-Light + {res_tag}"
        return result

    proc = resize_for_processing(pil, max_side)

    # ── Deep model inference (Hybrid MIRNet / A / B / C / D) ──────────
    model, total_params, loaded_ckpt = load_model(config, ckpt_path, device_str)
    inp = tensor_from_pil(proc).to(device_str)
    with torch.no_grad():
        t0 = time.perf_counter()
        out_dict = model(inp, return_dict=True)
        if device_str == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    enh_pil = pil_from_tensor(out_dict["enhanced"])
    if target_px > 0 and max(enh_pil.size) < target_px:
        enh_pil = upscale_to_hd_crisp(enh_pil, target_long_side=target_px)
    illum_arr = None
    if "illumination_map" in out_dict:
        illum = out_dict["illumination_map"].squeeze().cpu().numpy()
        illum_arr = (np.clip(illum, 0, 1) * 255).round().astype(np.uint8)
    return {
        "original_pil": proc,
        "enhanced_pil": enh_pil,
        "illum_arr":    illum_arr,
        "infer_ms":     (t1 - t0) * 1000.0,
        "total_params": total_params,
        "loaded_ckpt":  loaded_ckpt,
        "proc_size":    enh_pil.size,
        "mode_used":    f"Deep Config {config}" + (" (Trained)" if loaded_ckpt else " (Untrained)"),
    }


# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌙 4K Low-Light Studio")
        st.markdown(
            "<div class='sidebar-team-box'>👥 Team - Vijendra, Vyom, Gautam</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Enhancement Engine Mode
        st.subheader("⚡ Enhancement Engine")
        mode_choice = st.radio(
            "Pipeline Mode",
            options=["auto", "classical", "deep"],
            format_func=lambda m: {
                "auto": "✨ Auto Smart (Recommended)",
                "classical": "🌟 Retinex + CLAHE (Instant)",
                "deep": "🧠 Deep Neural Network (Config A-D)",
            }[m],
            help=(
                "Auto uses your trained .pth checkpoint when loaded, or our Multi-Scale Retinex "
                "+ CLAHE + Shadow Lift pipeline otherwise so dark photos are always cleanly enhanced."
            ),
        )

        # ── Enhancement Tuning Sliders (used in Auto/Classical)
        with st.expander("🎛️ Light & Clarity Tuning", expanded=(mode_choice != "deep")):
            for k_d, v_d in LIGHT_PRESETS["Balanced HD (Natural)"].items():
                st.session_state.setdefault(f"tune_{k_d}", v_d)

            pending_preset = st.session_state.pop("_pending_preset", None)
            if pending_preset and pending_preset in LIGHT_PRESETS:
                for k_p, v_p in LIGHT_PRESETS[pending_preset].items():
                    st.session_state[f"tune_{k_p}"] = v_p
                st.session_state["sidebar_preset"] = pending_preset
                st.session_state["_last_preset"] = pending_preset

            preset_name = st.selectbox(
                "☀️ Light Preset",
                options=list(LIGHT_PRESETS.keys()) + ["Custom"],
                index=0,
                key="sidebar_preset",
                help="Choose Bright Studio Light or Extreme Night Rescue for stronger illumination.",
            )
            if preset_name != "Custom" and st.session_state.get("_last_preset") != preset_name:
                for k_p, v_p in LIGHT_PRESETS[preset_name].items():
                    st.session_state[f"tune_{k_p}"] = v_p
                st.session_state["_last_preset"] = preset_name

            light_boost = st.slider(
                "☀️ Light Boost (More Right = Brighter)",
                min_value=0.0, max_value=2.0, step=0.05,
                key="tune_light_boost",
                help="Controls target scene brightness cleanly (drag RIGHT for stronger light).",
            )
            exposure_ev = st.slider(
                "💡 Fill Light / Exposure (+EV)",
                min_value=0.0, max_value=2.0, step=0.05,
                key="tune_exposure_ev",
                help="Adds studio fill light across dark & midtone areas without blowing out lamps.",
            )
            shadow_lift = st.slider(
                "🔦 Shadow Recovery",
                min_value=0.0, max_value=2.5, step=0.1,
                key="tune_shadow_lift",
                help="Boosts pitch-black shadows while protecting bright highlights.",
            )
            clarity = st.slider(
                "🔍 HD Clarity / Sharpness",
                min_value=0.0, max_value=1.5, step=0.05,
                key="tune_clarity",
                help="Multi-scale guided detail enhancement for crisp HD textures without noise grain.",
            )
            clahe_clip = st.slider(
                "Local Contrast (CLAHE)",
                min_value=0.0, max_value=4.0, step=0.2,
                key="tune_clahe_clip",
                help="Improves texture & clarity without color shift.",
            )
            saturation = st.slider(
                "Smart Color Vibrance",
                min_value=0.8, max_value=1.5, step=0.05,
                key="tune_saturation",
            )
            contrast = st.slider(
                "Filmic S-Curve Contrast",
                min_value=0.0, max_value=0.45, step=0.02,
                key="tune_contrast",
                help="Deepens clean blacks and adds rich studio contrast punch.",
            )
            warmth = st.slider(
                "Color Temp (Cool ↔ Warm)",
                min_value=-0.30, max_value=0.30, step=0.05,
                key="tune_warmth",
                help="Slide right for warm indoor glow, left for crisp cool moonlight.",
            )
            denoise = st.checkbox("Chroma Shadow Denoise (Zero Edge Blur)", value=True)

        st.divider()

        # ── Model config
        st.subheader("🧠 Model Architecture")
        config = st.selectbox(
            "Select Configuration",
            options=["D", "B", "A", "C"],
            format_func=lambda k: f"{CONFIG_ICONS[k]} Config {k}",
            help="D = Full Hybrid (best quality) | B = Baseline MIRNet | A = Illumination only | C = MIRNet + Illum",
        )
        col_badge, _ = st.columns([3, 1])
        with col_badge:
            clr = CONFIG_COLORS[config]
            st.markdown(
                f'<div style="border-left:3px solid {clr};padding:6px 10px;'
                f'background:#161b22;border-radius:4px;font-size:12px;color:#cdd9e5">'
                f"{CONFIG_DESCRIPTIONS[config]}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Checkpoint
        st.subheader("💾 Checkpoint")
        default_ckpt = str(
            ROOT_DIR / "experiments" / f"config_{config}" / "checkpoints" / "best_model.pth"
        )
        ckpt_text = st.text_input(
            "Path to .pth file",
            value=default_ckpt if Path(default_ckpt).is_file() else "",
            placeholder="experiments/config_D/checkpoints/best_model.pth",
        )
        uploaded_ckpt_file = st.file_uploader(
            "Or upload checkpoint (.pth / .pt)", type=["pth", "pt"]
        )
        ckpt_path: Optional[str] = None
        if uploaded_ckpt_file is not None:
            tmp_ckpt = ROOT_DIR / "experiments" / "_ui_uploaded_ckpt.pth"
            tmp_ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp_ckpt.write_bytes(uploaded_ckpt_file.getbuffer())
            ckpt_path = str(tmp_ckpt)
        elif ckpt_text.strip() and Path(ckpt_text.strip()).is_file():
            ckpt_path = ckpt_text.strip()

        st.divider()

        # ── Processing options
        st.subheader("⚙️ Output Quality & Resolution")
        output_res = st.selectbox(
            "🖥️ Output Target Resolution",
            options=list(OUTPUT_RES_MAP.keys()),
            index=0,
            help="4K UHD upscales smaller images cleanly with edge-directed de-staircasing & 4K micro-detail.",
        )
        full_res = st.checkbox(
            "Preserve Original Camera Input (up to 4K)",
            value=True,
            help="Keeps full native camera resolution (no downscale blur) before enhancement.",
        )
        max_side = st.slider(
            "Max side if resizing / Deep mode (px)",
            min_value=256,
            max_value=2560,
            value=1024,
            step=128,
            help="Used for Deep Neural Net mode or when Preserve Original Input is unchecked.",
        )
        save_to_disk = st.checkbox("Auto-save outputs to Results/ui_outputs/", value=False)

        st.divider()

        # ── Device info
        st.subheader("🖥️ Hardware")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        st.markdown(device_badge_html(device_str), unsafe_allow_html=True)
        if device_str == "cuda":
            props = torch.cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024**3)
            st.caption(f"VRAM: {mem_gb:.1f} GB | SM: {props.multi_processor_count}")
        else:
            import platform
            st.caption(f"CPU · {platform.processor()[:40]}")

        st.divider()
        st.caption(
            "Team - Vijendra, Vyom, Gautam · College Research Project (2026) · MIT License"
        )

    tune = {
        "mode": mode_choice,
        "light_boost": light_boost,
        "exposure_ev": exposure_ev,
        "shadow_lift": shadow_lift,
        "clarity": clarity,
        "clahe_clip": clahe_clip,
        "saturation": saturation,
        "contrast": contrast,
        "warmth": warmth,
        "denoise": denoise,
        "full_res": full_res,
        "output_res": output_res,
    }
    return config, ckpt_path, device_str, max_side, save_to_disk, tune


# ──────────────────────────────────────────────────────────
# Single Image Tab
# ──────────────────────────────────────────────────────────
def tab_single(config, ckpt_path, device_str, max_side, save_to_disk, tune):
    with st.container(border=True):
        st.markdown("### 📸 Upload Low-Light Photo Studio")
        col_up, col_gt = st.columns(2)
        with col_up:
            uploaded = st.file_uploader(
                "Drop your dark/night image here",
                type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
                key="single_upload",
                help="Upload any dark/night photograph to enhance in 4K UHD.",
            )
        with col_gt:
            gt_upload = st.file_uploader(
                "Reference Ground-Truth (optional · computes PSNR/SSIM)",
                type=["png", "jpg", "jpeg", "bmp"],
                key="gt_upload",
                help="If you have the reference image, PSNR and SSIM will be computed.",
            )

        sample_files = (
            sorted(f for f in os.listdir(SAMPLE_LOW_DIR) if f.lower().endswith(".png"))
            if SAMPLE_LOW_DIR.is_dir()
            else []
        )
        if not uploaded and sample_files:
            sel = st.selectbox("✨ Or test immediately with a bundled LOL low-light sample", sample_files, key="sample_sel")
            sample_path = SAMPLE_LOW_DIR / sel
            with open(sample_path, "rb") as f:
                img_bytes = f.read()
            image_name = sel
        elif uploaded:
            img_bytes = uploaded.getvalue()
            image_name = uploaded.name
        else:
            st.info("📂 Upload a low-light image above to begin.")
            return

        preview_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        in_lum = float(
            np.mean(
                np.array(preview_pil, dtype=np.float32) @ np.array([0.299, 0.587, 0.114])
            )
        )
        st.markdown(
            f"**Selected Input:** `{image_name}` · `{preview_pil.size[0]}×{preview_pil.size[1]} px` · "
            f"`{len(img_bytes)/1024:.1f} KB` · Avg Brightness: `{in_lum:.1f}/255`"
        )
        with st.expander("👁️ Preview Input Image", expanded=False):
            st.image(resize_for_display(preview_pil, 700), width="content")

    # Effective engine info before running
    will_use_classical = tune["mode"] == "classical" or (
        tune["mode"] == "auto" and not (ckpt_path and Path(ckpt_path).is_file())
    )
    btn_label = (
        "🚀 Run Enhancement (Light Boost + HD Clarity)"
        if will_use_classical
        else f"🚀 Run Enhancement (Deep Config {config})"
    )

    # Quick inline lighting buttons when in Auto/Classical so user can boost light in 1 click
    if will_use_classical:
        qb1, qb2, qb3 = st.columns(3)
        with qb1:
            if st.button("🌗 Balanced HD Light", width="stretch", key="q_bal"):
                st.session_state["_pending_preset"] = "Balanced HD (Natural)"
                st.session_state["_force_auto_run"] = True
                st.rerun()
        with qb2:
            if st.button("☀️ More Light (+Bright)", width="stretch", key="q_bright"):
                st.session_state["_pending_preset"] = "Bright Studio Light ☀️"
                st.session_state["_force_auto_run"] = True
                st.rerun()
        with qb3:
            if st.button("🔥 Max Night Light Boost", width="stretch", key="q_max"):
                st.session_state["_pending_preset"] = "Extreme Night Rescue 🔥"
                st.session_state["_force_auto_run"] = True
                st.rerun()

    run_btn = st.button(
        btn_label,
        type="primary",
        width="stretch",
        key="single_run",
    )

    # Track current input+settings signature so classical slider tweaks update live once run
    img_sig = hashlib.md5(img_bytes).hexdigest()[:10]
    already_ran_on_this_img = st.session_state.get("single_img_sig") == img_sig
    force_run = st.session_state.pop("_force_auto_run", False)
    auto_refresh = (already_ran_on_this_img or force_run) and will_use_classical

    if run_btn or auto_refresh:
        if run_btn:
            prog = st.progress(0, text="Preparing enhancement pipeline…")
            prog.progress(30, text="Estimating illumination & lifting light…")

        result = run_inference_cached(
            img_bytes,
            config,
            ckpt_path,
            device_str,
            max_side,
            mode=tune["mode"],
            light_boost=tune["light_boost"],
            exposure_ev=tune["exposure_ev"],
            shadow_lift=tune["shadow_lift"],
            clahe_clip=tune["clahe_clip"],
            denoise=tune["denoise"],
            saturation=tune["saturation"],
            clarity=tune["clarity"],
            warmth=tune["warmth"],
            contrast=tune["contrast"],
            full_res=tune["full_res"],
            output_res=tune["output_res"],
        )

        if run_btn:
            prog.progress(100, text="Done!")
            time.sleep(0.15)
            prog.empty()

        # Store in session
        st.session_state["single_result"] = result
        st.session_state["single_name"] = image_name
        st.session_state["single_img_sig"] = img_sig

        if run_btn:
            hist_entry = {
                "name": image_name,
                "config": result.get("mode_used", f"Config {config}"),
                "size": f"{result['proc_size'][0]}×{result['proc_size'][1]}",
                "ms": f"{result['infer_ms']:.1f}",
                "params": f"{result['total_params']:,}" if result["total_params"] else "Classical",
                "ckpt": "✅" if result["loaded_ckpt"] else "⚠️",
            }
            if HISTORY_KEY not in st.session_state:
                st.session_state[HISTORY_KEY] = []
            st.session_state[HISTORY_KEY].insert(0, hist_entry)
            st.session_state[HISTORY_KEY] = st.session_state[HISTORY_KEY][:MAX_HISTORY]

    # ── Results panel
    if "single_result" in st.session_state:
        result = st.session_state["single_result"]
        orig_pil = result["original_pil"]
        enh_pil = result["enhanced_pil"]
        illum_arr = result["illum_arr"]
        infer_ms = result["infer_ms"]
        params = result["total_params"]
        mode_used = result.get("mode_used", f"Config {config}")
        name = st.session_state.get("single_name", "image")

        orig_np = np.array(orig_pil, dtype=np.float32)
        enh_np = np.array(enh_pil, dtype=np.float32)
        lum_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        orig_mean = float(np.mean(orig_np @ lum_w))
        enh_mean = float(np.mean(enh_np @ lum_w))
        bright_ratio = enh_mean / max(orig_mean, 1.0)

        st.divider()
        st.markdown(f"## ✨ Enhancement Results — `{mode_used}`")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⏱ Inference", f"{infer_ms:.1f} ms")
        m2.metric("☀️ Brightness Lift", f"{enh_mean:.1f} / 255", delta=f"{bright_ratio:.2f}× brighter")
        res_delta = (
            "4K UHD" if max(enh_pil.size) >= 3840 else ("Full HD" if max(enh_pil.size) >= 1920 else "Native")
        )
        m3.metric(
            "📐 Output Size",
            f"{enh_pil.size[0]}×{enh_pil.size[1]}",
            delta=f"{res_delta} (from {orig_pil.size[0]}×{orig_pil.size[1]})",
        )
        m4.metric(
            "🧠 Engine",
            f"{params/1e6:.2f}M params" if params > 0 else res_delta,
        )

        # No-reference automatic quality telemetry row (fast strided sampling even for 4K images)
        orig_l_s = (orig_np[::max(1, orig_np.shape[0] // 256), ::max(1, orig_np.shape[1] // 256)] @ lum_w)
        enh_l_s = (enh_np[::max(1, enh_np.shape[0] // 256), ::max(1, enh_np.shape[1] // 256)] @ lum_w)
        dark_frac_in = float(np.mean(orig_l_s < 35.0) * 100.0)
        dark_frac_out = float(np.mean(enh_l_s < 35.0) * 100.0)
        rescued_pct = max(0.0, dark_frac_in - dark_frac_out)
        orig_span = float(np.percentile(orig_l_s, 95) - np.percentile(orig_l_s, 5))
        enh_span = float(np.percentile(enh_l_s, 95) - np.percentile(enh_l_s, 5))
        span_gain = ((enh_span - orig_span) / max(orig_span, 1.0)) * 100.0

        t1, t2, t3 = st.columns(3)
        t1.metric("🌑 Dark Shadows Rescued", f"{rescued_pct:.1f}%", delta=f"Dark pixels {dark_frac_in:.0f}% → {dark_frac_out:.0f}%")
        t2.metric("🎚️ Dynamic Range Span", f"{enh_span:.0f} / 255", delta=f"{span_gain:+.0f}% contrast")
        t3.metric("☀️ Mean Brightness Gain", f"{enh_mean - orig_mean:+.1f} Lum", delta="Balanced Highlights")

        # PSNR / SSIM if GT provided
        if gt_upload is not None:
            gt_pil = Image.open(gt_upload).convert("RGB").resize(orig_pil.size, Image.BICUBIC)
            orig_arr = np.array(orig_pil)
            enh_arr = np.array(enh_pil.resize(orig_pil.size, Image.BICUBIC))
            gt_arr = np.array(gt_pil)
            psnr_enh = compute_psnr(enh_arr, gt_arr)
            ssim_enh = compute_ssim(enh_arr, gt_arr)
            psnr_orig = compute_psnr(orig_arr, gt_arr)
            ssim_orig = compute_ssim(orig_arr, gt_arr)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("📊 PSNR (input)", f"{psnr_orig:.2f} dB")
            mc2.metric(
                "📊 PSNR (enhanced)",
                f"{psnr_enh:.2f} dB",
                delta=f"{psnr_enh - psnr_orig:+.2f} dB",
            )
            mc3.metric("📐 SSIM (input)", f"{ssim_orig:.4f}")
            mc4.metric(
                "📐 SSIM (enhanced)",
                f"{ssim_enh:.4f}",
                delta=f"{ssim_enh - ssim_orig:+.4f}",
            )

        # Tabs: Side-by-side | Interactive 60fps Curtain | 3x 4K Loupe | Illumination Map | Histogram | 3-Preset Compare | Download
        tab_sb, tab_split, tab_zoom, tab_illum, tab_hist_plot, tab_dl = st.tabs(
            [
                "🖼️ Showcase",
                "🎚️ Drag Wipe",
                "🔍 4K Live Zoom",
                "💡 Illumination Map",
                "📊 Tone Histogram",
                "⬇️ 4K Download",
            ]
        )

        with tab_sb:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""<div class="photo-tag photo-tag-orig">
                        <span>🌑 DARK INPUT</span>
                        <span class="photo-tag-meta">{orig_pil.size[0]}×{orig_pil.size[1]} · Lum {orig_mean:.1f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.image(resize_for_display(orig_pil, 1600), width="stretch")
            with c2:
                st.markdown(
                    f"""<div class="photo-tag photo-tag-enh">
                        <span>✨ {res_delta} ENHANCED</span>
                        <span class="photo-tag-meta">{enh_pil.size[0]}×{enh_pil.size[1]} · Lum {enh_mean:.1f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.image(resize_for_display(enh_pil, 1600), width="stretch")

            if will_use_classical:
                with st.expander("🎨 Compare All 3 Light Presets Side-by-Side", expanded=False):
                    thumb_orig = resize_for_display(orig_pil, 560)
                    p_cols = st.columns(3)
                    for idx_p, (p_label, p_cfg) in enumerate(LIGHT_PRESETS.items()):
                        with p_cols[idx_p]:
                            p_res = _classical_enhancer.enhance(thumb_orig, **p_cfg)
                            st.image(
                                p_res["enhanced_pil"],
                                caption=p_label,
                                width="stretch",
                            )

        with tab_split:
            st.caption("Drag the glowing divider handle directly on the photo left ↔ right to compare in real time.")
            render_curtain_comparator(orig_pil, enh_pil, res_tag=res_delta)
            with st.expander("🎛️ Precision Numeric Split Exporter", expanded=False):
                split_pct = st.slider(
                    "Split % (Left: Original | Right: Enhanced)",
                    min_value=0, max_value=100, value=50, key="split_sl",
                )
                disp_enh = resize_for_display(enh_pil, 1600)
                disp_orig = orig_pil.resize(disp_enh.size, Image.LANCZOS)
                st.image(make_split_comparison(disp_orig, disp_enh, split_pct), width="stretch")

        with tab_zoom:
            st.caption("Move your cursor over any corner or fine detail to inspect 3.0× magnified 4K UHD clarity.")
            render_loupe_inspector(orig_pil, enh_pil)

        with tab_illum:
            if illum_arr is not None:
                st.markdown(
                    "**Estimated Illumination Map L(x)** "
                    "— darker areas receive stronger adaptive brightness & shadow recovery"
                )
                col_il1, col_il2 = st.columns(2)
                with col_il1:
                    st.image(illum_arr, caption="L(x) · Grayscale Map", width="stretch")
                with col_il2:
                    h_arr = illum_arr.astype(np.float32) / 255.0
                    r = np.clip(h_arr * 3.0, 0, 1)
                    g = np.clip(h_arr * 3.0 - 1.0, 0, 1)
                    b = np.clip(h_arr * 3.0 - 2.0, 0, 1)
                    hot_arr = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
                    st.image(hot_arr, caption="L(x) · Luminance Heat Map", width="stretch")
            else:
                st.info(
                    f"{mode_used} does not expose an illumination map. "
                    "Use Auto/Classical or Config C/D to inspect L(x)."
                )

        with tab_hist_plot:
            st.markdown(
                build_luminance_histogram_svg(
                    orig_np[::max(1, orig_np.shape[0] // 256), ::max(1, orig_np.shape[1] // 256)],
                    enh_np[::max(1, enh_np.shape[0] // 256), ::max(1, enh_np.shape[1] // 256)],
                ),
                unsafe_allow_html=True,
            )

        with tab_dl:
            st.markdown("### ⬇️ Export Your Enhanced Studio Image")
            stem_name = Path(name).stem
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                enh_bytes = pil_to_bytes(enh_pil, fmt="PNG")
                st.download_button(
                    f"⬇️ Download {res_delta} Lossless PNG ({enh_pil.size[0]}×{enh_pil.size[1]})",
                    data=enh_bytes,
                    file_name=f"enhanced_{res_delta.replace(' ', '')}_{stem_name}.png",
                    mime="image/png",
                    width="stretch",
                )
            with dl_c2:
                jpg_bytes = pil_to_bytes(enh_pil, fmt="JPEG")
                st.download_button(
                    f"⬇️ Download {res_delta} Shareable JPG 98% ({enh_pil.size[0]}×{enh_pil.size[1]})",
                    data=jpg_bytes,
                    file_name=f"enhanced_{res_delta.replace(' ', '')}_{stem_name}.jpg",
                    mime="image/jpeg",
                    width="stretch",
                )
            if max(enh_pil.size) < 3840:
                uhd_pil = upscale_to_hd_crisp(
                    enh_pil, target_long_side=3840, clarity=min(0.28, tune["clarity"])
                )
                st.download_button(
                    f"⬇️ Download 4K UHD Upscaled PNG ({uhd_pil.size[0]}×{uhd_pil.size[1]})",
                    data=pil_to_bytes(uhd_pil, fmt="PNG"),
                    file_name=f"enhanced_4K_{stem_name}.png",
                    mime="image/png",
                    width="stretch",
                )
            disp_enh_dl = resize_for_display(enh_pil, 1920)
            disp_orig_dl = orig_pil.resize(disp_enh_dl.size, Image.LANCZOS)
            split_bytes = pil_to_bytes(make_split_comparison(disp_orig_dl, disp_enh_dl, 50), fmt="PNG")
            st.download_button(
                "⬇️ Download 50/50 Split Comparison PNG",
                data=split_bytes,
                file_name=f"comparison_{stem_name}.png",
                mime="image/png",
                width="stretch",
            )

        if save_to_disk:
            save_path = RESULTS_DIR / f"enhanced_{config}_{name}"
            enh_pil.save(str(save_path))
            st.caption(f"✅ Saved to `{save_path}`")


# ──────────────────────────────────────────────────────────
# Batch Tab
# ──────────────────────────────────────────────────────────
def tab_batch(config, ckpt_path, device_str, max_side, save_to_disk, tune):
    st.markdown("### 📦 Batch Image Enhancement")
    st.caption("Upload up to **20** images at once. Each will be enhanced and downloadable as a ZIP.")

    batch_files = st.file_uploader(
        "Upload multiple low-light images",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if not batch_files:
        st.info("📂 Upload multiple images above to begin batch processing.")
        return

    if len(batch_files) > 20:
        st.warning("⚠️ Max 20 images per batch. Only the first 20 will be processed.")
        batch_files = batch_files[:20]

    st.markdown(f"**{len(batch_files)} image(s) queued for enhancement.**")

    if st.button(
        f"🚀 Enhance All {len(batch_files)} Images",
        type="primary",
        width="stretch",
        key="batch_run",
    ):
        results = []
        prog_bar = st.progress(0, text="Starting batch…")
        status_area = st.empty()

        for i, f in enumerate(batch_files):
            pct = int((i / len(batch_files)) * 100)
            prog_bar.progress(pct, text=f"Processing {f.name} ({i+1}/{len(batch_files)})…")
            status_area.info(f"🔄 Enhancing `{f.name}`…")

            result = run_inference_cached(
                f.getvalue(),
                config,
                ckpt_path,
                device_str,
                max_side,
                mode=tune["mode"],
                light_boost=tune["light_boost"],
                exposure_ev=tune["exposure_ev"],
                shadow_lift=tune["shadow_lift"],
                clahe_clip=tune["clahe_clip"],
                denoise=tune["denoise"],
                saturation=tune["saturation"],
                clarity=tune["clarity"],
                warmth=tune["warmth"],
                contrast=tune["contrast"],
                full_res=tune["full_res"],
                output_res=tune["output_res"],
            )
            results.append({"name": f.name, "result": result})

        prog_bar.progress(100, text="All done!")
        status_area.success(f"✅ {len(results)} images enhanced!")
        st.session_state["batch_results"] = results

    if "batch_results" in st.session_state and st.session_state["batch_results"]:
        results = st.session_state["batch_results"]
        st.divider()
        st.markdown("## ✨ Batch Results")

        rows_html = ""
        zip_entries = []
        total_ms = 0.0
        for entry in results:
            name = entry["name"]
            res = entry["result"]
            ms = res["infer_ms"]
            total_ms += ms
            w, h = res["proc_size"]
            rows_html += (
                f"<tr><td>{name}</td><td>{w}×{h}</td>"
                f"<td>{ms:.1f} ms</td><td>{res.get('mode_used', config)}</td>"
                f"<td>{'✅' if res['loaded_ckpt'] else '⚠️'}</td></tr>"
            )
            zip_entries.append((f"enhanced_{config}_{name}", pil_to_bytes(res["enhanced_pil"])))

        st.markdown(
            f"<table><thead><tr><th>File</th><th>Size</th><th>Inference</th>"
            f"<th>Engine</th><th>Status</th></tr></thead><tbody>{rows_html}</tbody></table>"
            f"<p style='color:#8b949e;font-size:12px;margin-top:6px'>"
            f"Total processing time: {total_ms:.0f} ms · "
            f"Avg per image: {total_ms/len(results):.0f} ms</p>",
            unsafe_allow_html=True,
        )

        st.markdown("### 🖼️ Enhanced Image Grid")
        cols_per_row = 3
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for ci, entry in enumerate(results[row_start : row_start + cols_per_row]):
                with cols[ci]:
                    enh = entry["result"]["enhanced_pil"]
                    st.image(
                        resize_for_display(enh, 400),
                        caption=entry["name"],
                        width="stretch",
                    )

        st.divider()
        zip_bytes = make_zip(zip_entries)
        st.download_button(
            f"⬇️ Download All {len(results)} Enhanced Images as ZIP",
            data=zip_bytes,
            file_name=f"enhanced_batch_config{config}.zip",
            mime="application/zip",
            width="stretch",
        )

        if save_to_disk:
            for entry in results:
                sp = RESULTS_DIR / f"enhanced_{config}_{entry['name']}"
                entry["result"]["enhanced_pil"].save(str(sp))
            st.caption(f"✅ All saved to `{RESULTS_DIR}`")


# ──────────────────────────────────────────────────────────
# History Tab
# ──────────────────────────────────────────────────────────
def tab_history():
    st.markdown("### 📜 Enhancement History")
    history = st.session_state.get(HISTORY_KEY, [])
    if not history:
        st.info("No enhancements run yet in this session.")
        return
    rows_html = ""
    for i, h in enumerate(history):
        rows_html += (
            f"<tr><td>{i+1}</td><td>{h['name']}</td><td>{h['config']}</td>"
            f"<td>{h['size']}</td><td>{h['ms']} ms</td><td>{h['params']}</td>"
            f"<td>{h['ckpt']}</td></tr>"
        )
    st.markdown(
        f"<table><thead><tr><th>#</th><th>File</th><th>Engine</th><th>Size</th>"
        f"<th>Time</th><th>Params</th><th>Status</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Clear History"):
        st.session_state[HISTORY_KEY] = []
        st.rerun()


# ──────────────────────────────────────────────────────────
# About Tab
# ──────────────────────────────────────────────────────────
def tab_about():
    st.markdown("## 👥 Project Team & Architecture")
    st.markdown(
        """
        <div class="team-grid">
            <div class="team-card">
                <div class="team-avatar">V</div>
                <div class="team-name">Vijendra</div>
                <div class="team-role">Deep Learning & Hybrid MIRNet Engineering</div>
            </div>
            <div class="team-card">
                <div class="team-avatar" style="background:linear-gradient(135deg,#f97316,#ec4899);">V</div>
                <div class="team-name">Vyom</div>
                <div class="team-role">Illumination Curves, Ablation & Evaluation</div>
            </div>
            <div class="team-card">
                <div class="team-avatar" style="background:linear-gradient(135deg,#22c55e,#0ea5e9);">G</div>
                <div class="team-name">Gautam</div>
                <div class="team-role">4K UHD Enhancement Pipeline & Studio UI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
**Project Title:** *Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion* (Submission: November 2026)

This research project extends the MIRNet architecture with two novel modules plus a real-time 4K UHD LIME/Retinex Studio pipeline:

| Module | Description |
|---|---|
| **IlluminationCurveEstimator** | Learnable pixel-wise curve estimation with iterative enhancement |
| **AdaptiveFeatureFusion** | Spatio-channel attention that blends MIRNet + illumination features |
| **ClassicalEnhancer + 4K UHD** | Real-time exposure-safe Retinex fill-light + edge-directed `3840px` super-res |

### Model Configurations

| Config | Architecture | Params |
|---|---|---|
| A | Illumination branch only | 74,317 |
| B | Original MIRNet (baseline) | 32.2 M |
| C | MIRNet + Illumination (sequential) | 32.3 M |
| D | Full Hybrid (Proposed) | 32.5 M |

### Pipeline

```
Input Low-Light Image
    ↓
Illumination Curve Estimation  (Config A, C, D)
    ↓
MIRNet Multi-Scale Restoration (Config B, C, D)
    ↓
Adaptive Feature Fusion        (Config D only)
    ↓
Final Refined Output
```

### References & License
- **MIRNet Architecture:** S. W. Zamir et al., *Learning Enriched Features for Real Image Restoration and Enhancement*, ECCV 2020 (base PyTorch backbone preserved under MIT `LICENSE`).
- **Project Extensions & 4K Studio:** Team - Vijendra, Vyom, Gautam (2026).

### Training
Train the model using:
```bash
python train_hybrid.py --config D --epochs 200 --batch_size 4 --patch_size 256
```
Or open `notebooks/Complete_Project_Implementation.ipynb` in Google Colab.
""")

    st.markdown("### 🏗️ Architecture Overview")
    if (ROOT_DIR / "experiments" / "ablation_metrics_comparison.png").exists():
        st.image(
            str(ROOT_DIR / "experiments" / "ablation_metrics_comparison.png"),
            caption="Ablation Study Results",
            width="stretch",
        )


# ──────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-team-pill">👥 Team - Vijendra, Vyom, Gautam</div>
            <div class="hero-title">🌙 Low-Light Image Enhancer · 4K AI Studio</div>
            <div class="hero-pills">
                <span class="feature-pill">✨ 4K UHD (3840px) Output</span>
                <span class="feature-pill">☀️ Highlight-Safe Fill-Light</span>
                <span class="feature-pill">🔍 Guided Edge Clarity</span>
                <span class="feature-pill">🧠 Hybrid MIRNet & Curve Fusion</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config, ckpt_path, device_str, max_side, save_to_disk, tune = render_sidebar()

    # Frosted Glass Studio Status Ribbon
    ckpt_badge = (
        checkpoint_badge_html(True)
        if ckpt_path is not None
        else (
            '<span class="badge badge-yellow">✨ Retinex + 4K Engine Active</span>'
            if tune["mode"] in ("auto", "classical")
            else checkpoint_badge_html(False)
        )
    )
    st.markdown(
        f"""
        <div class="glass-ribbon">
            <div>{device_badge_html(device_str)}</div>
            <div>{ckpt_badge}</div>
            <div><span class="badge badge-purple">🖥️ Output: {tune["output_res"]}</span></div>
            <div><span class="badge badge-blue">🧠 Config {config} Ready</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    t_single, t_batch, t_history, t_about = st.tabs(
        ["📸 Single Image Studio", "📦 Batch 4K Processing", "📜 Session History", "📖 Architecture & Team"]
    )

    with t_single:
        tab_single(config, ckpt_path, device_str, max_side, save_to_disk, tune)
    with t_batch:
        tab_batch(config, ckpt_path, device_str, max_side, save_to_disk, tune)
    with t_history:
        tab_history()
    with t_about:
        tab_about()

    st.markdown(
        """
        <div class="studio-footer">
            <b>Team - Vijendra, Vyom, Gautam</b> &nbsp;|&nbsp;
            Low-Light Image Enhancement Using MIRNet and Deep Illumination Curve Fusion &nbsp;|&nbsp;
            Research Project (2026)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
