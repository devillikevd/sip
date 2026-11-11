import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim


class MixUp:
    """Beta-distribution MixUp augmentation compatible with CPU and CUDA tensors."""

    def __init__(self, alpha=1.2):
        self.distribution = torch.distributions.beta.Beta(
            torch.tensor([alpha]), torch.tensor([alpha])
        )

    def augment(self, gt, ip):
        batch_size = gt.size(0)
        idx = torch.randperm(batch_size, device=gt.device)
        gt1 = gt[idx]
        ip1 = ip[idx]

        lam = (
            self.distribution.rsample((batch_size, 1))
            .view(-1, 1, 1, 1)
            .to(device=gt.device, dtype=gt.dtype)
        )
        gt = lam * gt + (1.0 - lam) * gt1
        ip = lam * ip + (1.0 - lam) * ip1
        return gt, ip


def calc_PSNR(target, pred, eps=1e-8):
    """Compute PSNR (dB) for tensors scaled in [0, 1]."""
    with torch.no_grad():
        diff = torch.clamp(pred, 0.0, 1.0) - torch.clamp(target, 0.0, 1.0)
        rmse = torch.sqrt(torch.clamp((diff ** 2).mean(), min=eps * eps))
        res = 20.0 * torch.log10(1.0 / rmse)
        return float(res.item())


def batch_psnr(img1, img2, data_range=None):
    """Compute mean per-image PSNR across a batch (N, C, H, W)."""
    if img1.dim() == 3:
        return calc_PSNR(img1, img2)
    psnr_vals = [calc_PSNR(tgt, pred) for tgt, pred in zip(img1, img2)]
    return sum(psnr_vals) / len(psnr_vals)


def calc_SSIM(target, pred):
    """Compute SSIM for a single (C, H, W) tensor scaled in [0, 1]."""
    with torch.no_grad():
        tgt_np = (
            torch.clamp(target.detach().cpu(), 0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
            .astype(np.float32)
        )
        pred_np = (
            torch.clamp(pred.detach().cpu(), 0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
            .astype(np.float32)
        )
        min_side = min(tgt_np.shape[0], tgt_np.shape[1])
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        win_size = max(3, win_size)
        return float(
            skimage_ssim(tgt_np, pred_np, channel_axis=2, data_range=1.0, win_size=win_size)
        )


def batch_ssim(img1, img2):
    """Compute mean per-image SSIM across a batch (N, C, H, W)."""
    if img1.dim() == 3:
        return calc_SSIM(img1, img2)
    ssim_vals = [calc_SSIM(tgt, pred) for tgt, pred in zip(img1, img2)]
    return sum(ssim_vals) / len(ssim_vals)


def load_img(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def save_img(img, path):
    arr = img
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
