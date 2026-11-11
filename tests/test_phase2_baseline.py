"""Unit and integration verification tests for Phase 2 baseline components."""

import contextlib
import io
import os
import sys
import tempfile
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image
import torch

from dataset import LoLDataset
from model import MIRNet
from train_enhancement import Loss_fn
from utils import MixUp, batch_psnr, batch_ssim, calc_PSNR, calc_SSIM, load_img, save_img


class TestPhase2Baseline(unittest.TestCase):
    def test_mirnet_parameter_count_and_forward(self):
        model = MIRNet()
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 32212944)

        x = torch.rand(1, 3, 64, 64)
        model.eval()
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 3, 64, 64))

    def test_lol_dataset_sorted_pairing_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            low_dir = os.path.join(tmp, "low")
            high_dir = os.path.join(tmp, "high")
            os.makedirs(low_dir)
            os.makedirs(high_dir)

            for name in ("b.png", "a.png"):
                arr = np.full((80, 100, 3), 120 if name == "a.png" else 200, dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(low_dir, name))
                Image.fromarray(arr).save(os.path.join(high_dir, name))

            ds_crop = LoLDataset(low_dir, high_dir, img_size=64, mode="crop")
            self.assertEqual(len(ds_crop), 2)
            self.assertTrue(ds_crop.lol_fnames[0].endswith("a.png"))
            x, y = ds_crop[0]
            self.assertEqual(tuple(x.shape), (3, 64, 64))
            self.assertEqual(tuple(y.shape), (3, 64, 64))

            ds_full = LoLDataset(low_dir, high_dir, img_size=None, mode="full")
            xf, yf = ds_full[1]
            self.assertEqual(tuple(xf.shape), (3, 80, 100))

    def test_utils_metrics_mixup_and_io(self):
        a = torch.rand(2, 3, 32, 32)
        b = a.clone()
        self.assertGreater(batch_psnr(a, b), 70.0)
        self.assertAlmostEqual(batch_ssim(a, b), 1.0, places=5)
        self.assertAlmostEqual(calc_SSIM(a[0], b[0]), 1.0, places=5)
        self.assertGreater(calc_PSNR(a[0], b[0]), 70.0)

        mix = MixUp()
        gt_m, ip_m = mix.augment(a, b)
        self.assertEqual(tuple(gt_m.shape), tuple(a.shape))
        self.assertEqual(tuple(ip_m.shape), tuple(b.shape))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "img.png")
            save_img(a[0], path)
            loaded = load_img(path)
            self.assertEqual(loaded.shape, (32, 32, 3))

    def test_charbonnier_loss_backward(self):
        criterion = Loss_fn()
        pred = torch.rand(2, 3, 16, 16, requires_grad=True)
        tgt = torch.rand(2, 3, 16, 16)
        loss = criterion(pred, tgt)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()

