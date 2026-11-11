"""Unit and checkpoint-resume tests for Phase 6: losses.py and train_hybrid.py."""

import os
import subprocess
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

from dataset import PairedRandomPatchDataset
from losses import HybridEnhancementLoss, SSIMLoss
from models.hybrid_mirnet import build_ablation_model


class TestPhase6Training(unittest.TestCase):
    def test_ssim_and_composite_loss(self):
        ssim_loss = SSIMLoss()
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        self.assertAlmostEqual(float(ssim_loss(x, x.detach()).item()), 0.0, places=5)

        criterion = HybridEnhancementLoss(perceptual_weight=0.0)
        model = build_ablation_model("A")
        out = model(x, return_dict=True)
        target = torch.rand(2, 3, 32, 32)
        total, terms = criterion(out, target, x)
        total.backward()

        self.assertGreater(float(total.item()), 0.0)
        self.assertIn("l1", terms)
        self.assertIn("ssim_loss", terms)
        self.assertIn("color", terms)
        self.assertIn("smooth", terms)

    def test_random_patch_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            low_dir = os.path.join(tmp, "low")
            high_dir = os.path.join(tmp, "high")
            os.makedirs(low_dir)
            os.makedirs(high_dir)
            arr = np.full((96, 128, 3), 80, dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(low_dir, "1.png"))
            Image.fromarray(arr + 40).save(os.path.join(high_dir, "1.png"))

            ds = PairedRandomPatchDataset(low_dir, high_dir, patch_size=64, augment=True)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (3, 64, 64))
            self.assertEqual(tuple(y.shape), (3, 64, 64))

    def test_train_and_resume_end_to_end(self):
        samples_low = os.path.join(ROOT_DIR, "baseline_results", "samples", "low")
        samples_high = os.path.join(ROOT_DIR, "baseline_results", "samples", "high")
        if not os.path.isdir(samples_low):
            self.skipTest("Sample images unavailable.")

        with tempfile.TemporaryDirectory() as out_dir:
            # Train Config A for 1 epoch (2 steps)
            cmd1 = [
                sys.executable,
                os.path.join(ROOT_DIR, "train_hybrid.py"),
                "--config", "A",
                "--train_low", samples_low,
                "--train_high", samples_high,
                "--val_low", samples_low,
                "--val_high", samples_high,
                "--output_dir", out_dir,
                "--epochs", "1",
                "--batch_size", "2",
                "--patch_size", "64",
                "--val_mode", "resize",
                "--val_size", "64",
                "--max_steps", "2",
                "--seed", "42",
            ]
            p1 = subprocess.run(cmd1, cwd=ROOT_DIR, capture_output=True, text=True)
            self.assertEqual(p1.returncode, 0, msg=p1.stderr)

            latest_ckpt = os.path.join(out_dir, "checkpoints", "latest_checkpoint.pth")
            best_ckpt = os.path.join(out_dir, "checkpoints", "best_model.pth")
            self.assertTrue(os.path.isfile(latest_ckpt))
            self.assertTrue(os.path.isfile(best_ckpt))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "training_history.csv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "loss_psnr_curves.png")))

            # Resume for epoch 2
            cmd2 = [
                sys.executable,
                os.path.join(ROOT_DIR, "train_hybrid.py"),
                "--config", "A",
                "--train_low", samples_low,
                "--train_high", samples_high,
                "--val_low", samples_low,
                "--val_high", samples_high,
                "--output_dir", out_dir,
                "--epochs", "2",
                "--batch_size", "2",
                "--patch_size", "64",
                "--val_mode", "resize",
                "--val_size", "64",
                "--max_steps", "2",
                "--resume", latest_ckpt,
                "--seed", "42",
            ]
            p2 = subprocess.run(cmd2, cwd=ROOT_DIR, capture_output=True, text=True)
            self.assertEqual(p2.returncode, 0, msg=p2.stderr)

            import pandas as pd
            df = pd.read_csv(os.path.join(out_dir, "training_history.csv"))
            self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()

