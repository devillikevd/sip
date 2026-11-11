"""Unit and gradient tests for Phase 4: AdaptiveFeatureFusion."""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from models.adaptive_fusion import AdaptiveFeatureFusion


class TestPhase4AdaptiveFusion(unittest.TestCase):
    def test_fusion_shapes_and_simplex_weights(self):
        fusion = AdaptiveFeatureFusion(in_channels=3, feature_channels=64)
        raw_img = torch.rand(2, 3, 64, 64)
        f_mir = torch.randn(2, 64, 64, 64)
        f_illum = torch.randn(2, 64, 64, 64)

        fused, weights = fusion(raw_img, f_mir, f_illum, return_weights=True)
        self.assertEqual(tuple(fused.shape), (2, 64, 64, 64))
        self.assertEqual(tuple(weights["spatial_weights"].shape), (2, 3, 64, 64))
        self.assertEqual(tuple(weights["channel_weights"].shape), (2, 3, 64, 1, 1))

        spatial_sum = weights["spatial_weights"].sum(dim=1)
        channel_sum = weights["channel_weights"].sum(dim=1)
        self.assertTrue(torch.allclose(spatial_sum, torch.ones_like(spatial_sum), atol=1e-5))
        self.assertTrue(torch.allclose(channel_sum, torch.ones_like(channel_sum), atol=1e-5))

    def test_preextracted_original_features_and_gradients(self):
        fusion = AdaptiveFeatureFusion(in_channels=3, feature_channels=64)
        raw_img = torch.rand(2, 3, 48, 52, requires_grad=True)
        f_mir = torch.randn(2, 64, 48, 52, requires_grad=True)
        f_illum = torch.randn(2, 64, 48, 52, requires_grad=True)

        fused = fusion(raw_img, f_mir, f_illum)
        loss = (fused ** 2).mean()
        loss.backward()

        for inp in (raw_img, f_mir, f_illum):
            self.assertIsNotNone(inp.grad)
            self.assertTrue(torch.isfinite(inp.grad).all())
            self.assertGreater(float(inp.grad.abs().sum().item()), 0.0)

        for name, param in fusion.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing grad for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite grad for {name}")

    def test_shape_mismatch_raises(self):
        fusion = AdaptiveFeatureFusion(in_channels=3, feature_channels=64)
        with self.assertRaises(ValueError):
            fusion(torch.rand(1, 3, 32, 32), torch.randn(1, 64, 32, 32), torch.randn(1, 64, 16, 16))


if __name__ == "__main__":
    unittest.main()

