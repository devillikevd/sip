"""Integration and gradient verification tests for Phase 5: HybridMIRNet & Configs A-D."""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from models.hybrid_mirnet import CONFIG_DESCRIPTIONS, HybridMIRNet, build_ablation_model


class TestPhase5HybridMIRNet(unittest.TestCase):
    def test_all_ablation_configs_forward_and_odd_dims(self):
        x_even = torch.rand(1, 3, 64, 64)
        x_odd = torch.rand(1, 3, 67, 73)

        param_counts = {}
        for cfg in ("A", "B", "C", "D"):
            self.assertIn(cfg, CONFIG_DESCRIPTIONS)
            model = build_ablation_model(cfg)
            model.eval()
            param_counts[cfg] = sum(p.numel() for p in model.parameters())

            with torch.no_grad():
                out_even = model(x_even)
                out_odd = model(x_odd)
                out_dict = model(x_even, return_dict=True)

            self.assertEqual(tuple(out_even.shape), (1, 3, 64, 64))
            self.assertEqual(tuple(out_odd.shape), (1, 3, 67, 73))
            self.assertGreaterEqual(float(out_even.min().item()), 0.0)
            self.assertLessEqual(float(out_even.max().item()), 1.0)
            self.assertIn("enhanced", out_dict)

        self.assertEqual(param_counts["B"], 32212944)
        self.assertGreater(param_counts["D"], param_counts["C"])
        self.assertGreater(param_counts["C"], param_counts["B"])
        self.assertLess(param_counts["A"], 200_000)

    def test_hybrid_full_backward_pass(self):
        # Lightweight config for rapid unit backward check verifying full HybridMIRNet wiring
        model = HybridMIRNet(num_features=16, illum_base_channels=16, num_MSRB=1, num_RRG=1)
        x = torch.rand(1, 3, 32, 32, requires_grad=True)
        target = torch.rand(1, 3, 32, 32)

        out = model(x, return_dict=True)
        loss = torch.nn.functional.l1_loss(out["enhanced"], target)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite grad for {name}")


if __name__ == "__main__":
    unittest.main()

