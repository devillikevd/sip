"""Unit and mathematical verification tests for Phase 3: IlluminationCurveEstimator."""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image
import torch

from models.illumination_curve import (
    IlluminationCurveEstimator,
    color_consistency_loss,
    illumination_smoothness_loss,
)


class TestPhase3IlluminationCurve(unittest.TestCase):
    def test_parameter_efficiency_and_output_contract(self):
        model = IlluminationCurveEstimator(
            in_channels=3, base_channels=32, feature_channels=64, num_iters=8
        )
        total_params = sum(p.numel() for p in model.parameters())
        self.assertLess(total_params, 200_000)

        x = torch.rand(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 3, 64, 64))
        self.assertGreaterEqual(float(out.min().item()), 0.0)
        self.assertLessEqual(float(out.max().item()), 1.0)

        out_dict = model(x, return_dict=True)
        expected_keys = {
            "enhanced",
            "curve_enhanced",
            "illumination_map",
            "curve_maps",
            "brightness_gain",
            "delta_lift",
            "color_blend_weight",
            "features",
        }
        self.assertEqual(set(out_dict.keys()), expected_keys)
        self.assertEqual(tuple(out_dict["enhanced"].shape), (2, 3, 64, 64))
        self.assertEqual(tuple(out_dict["illumination_map"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(out_dict["curve_maps"].shape), (2, 8, 3, 64, 64))
        self.assertEqual(tuple(out_dict["features"].shape), (2, 64, 64, 64))
        self.assertGreater(float(out_dict["illumination_map"].min().item()), 0.0)
        self.assertLessEqual(float(out_dict["illumination_map"].max().item()), 1.0)
        self.assertGreaterEqual(float(out_dict["curve_maps"].min().item()), -1.0)
        self.assertLessEqual(float(out_dict["curve_maps"].max().item()), 1.0)

    def test_arbitrary_and_odd_resolutions(self):
        model = IlluminationCurveEstimator()
        model.eval()
        for h, w in [(64, 64), (185, 277), (184, 276)]:
            x = torch.rand(1, 3, h, w)
            with torch.no_grad():
                out = model(x, return_dict=True)
            self.assertEqual(tuple(out["enhanced"].shape), (1, 3, h, w))
            self.assertEqual(tuple(out["features"].shape), (1, 64, h, w))

    def test_iterative_curve_mathematical_properties(self):
        model = IlluminationCurveEstimator(num_iters=8)
        x = torch.linspace(0.05, 0.95, 16).view(1, 1, 4, 4).repeat(1, 3, 1, 1)

        # Zero curves must be an exact identity transform
        zero_curves = torch.zeros(1, 8, 3, 4, 4)
        y_zero = model.apply_iterative_curves(x, zero_curves)
        self.assertTrue(torch.allclose(x, y_zero, atol=1e-6))

        # Positive curves must strictly increase interior intensities while remaining <= 1
        pos_curves = torch.full((1, 8, 3, 4, 4), 0.5)
        y_pos = model.apply_iterative_curves(x, pos_curves)
        self.assertTrue(torch.all(y_pos > x))
        self.assertTrue(torch.all(y_pos <= 1.0))

    def test_regularization_losses_and_full_gradient_flow(self):
        model = IlluminationCurveEstimator()
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        target = torch.rand(2, 3, 32, 32)

        out = model(x, return_dict=True)
        recon_loss = torch.nn.functional.l1_loss(out["enhanced"], target)
        feat_loss = 0.01 * (out["features"] ** 2).mean()
        reg_loss, reg_terms = model.compute_regularization_loss(out, x)

        self.assertIn("illumination_smoothness", reg_terms)
        self.assertIn("color_consistency", reg_terms)

        total_loss = recon_loss + feat_loss + reg_loss
        total_loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")
            self.assertGreater(float(param.grad.abs().sum().item()), 0.0, f"Zero gradient for {name}")

    def test_smoothness_and_color_loss_edge_cases(self):
        const_illum = torch.full((2, 1, 16, 16), 0.4)
        const_curves = torch.zeros(2, 8, 3, 16, 16)
        self.assertAlmostEqual(
            float(illumination_smoothness_loss(const_illum, const_curves).item()), 0.0, places=6
        )

        gray_img = torch.full((2, 3, 16, 16), 0.5)
        self.assertAlmostEqual(
            float(color_consistency_loss(gray_img, gray_img).item()), 0.0, places=3
        )

    def test_real_sample_image_forward(self):
        sample_path = os.path.join(
            ROOT_DIR, "baseline_results", "samples", "low", "sample_1.png"
        )
        if not os.path.isfile(sample_path):
            self.skipTest("Sample image not extracted yet.")
        arr = np.array(Image.open(sample_path).convert("RGB"), dtype=np.float32) / 255.0
        inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        model = IlluminationCurveEstimator()
        model.eval()
        with torch.no_grad():
            out = model(inp, return_dict=True)
        self.assertEqual(tuple(out["enhanced"].shape), tuple(inp.shape))
        self.assertTrue(torch.isfinite(out["enhanced"]).all())


if __name__ == "__main__":
    unittest.main()

