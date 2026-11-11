"""Verification tests for Phase 10 (Streamlit UI) and Phase 11 (Colab Notebook)."""

import ast
import json
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

import app as ui_app


class TestPhase10UIAndNotebook(unittest.TestCase):
    def test_app_helpers_and_model_loader(self):
        pil_in = Image.fromarray(np.full((48, 64, 3), 30, dtype=np.uint8))
        # New app uses public names (no underscore prefix)
        tensor = ui_app.tensor_from_pil(pil_in)
        self.assertEqual(tuple(tensor.shape), (1, 3, 48, 64))

        pil_out = ui_app.pil_from_tensor(tensor)
        self.assertEqual(pil_out.size, (64, 48))

        split_img = ui_app.make_split_comparison(pil_in, pil_out, 50)
        self.assertEqual(split_img.size, (64, 48))

        model, total_params, ckpt = ui_app.load_model("A", None, "cpu")
        self.assertGreater(total_params, 10_000)
        self.assertIsNone(ckpt)
        with torch.no_grad():
            out = model(tensor)
        self.assertEqual(tuple(out.shape), (1, 3, 48, 64))

        # Verify ClassicalEnhancer cleanly brightens dark images and produces illumination map
        sample_path = os.path.join(ROOT_DIR, "baseline_results", "samples", "low", "sample_1.png")
        dark_pil = (
            Image.open(sample_path).convert("RGB")
            if os.path.isfile(sample_path)
            else Image.fromarray(np.full((64, 64, 3), 15, dtype=np.uint8))
        )
        cls_out = ui_app._classical_enhancer.enhance(dark_pil)
        self.assertEqual(cls_out["enhanced_pil"].size, dark_pil.size)
        self.assertGreater(
            np.array(cls_out["enhanced_pil"]).mean(),
            np.array(dark_pil).mean() * 2.0,
            msg="Classical Retinex+CLAHE enhancer should substantially brighten dark images",
        )
        self.assertEqual(cls_out["illum_arr"].shape, (dark_pil.size[1], dark_pil.size[0]))

        bright_out = ui_app._classical_enhancer.enhance(
            dark_pil, **ui_app.LIGHT_PRESETS["Bright Studio Light ☀️"]
        )
        self.assertGreater(
            np.array(bright_out["enhanced_pil"]).mean(),
            np.array(cls_out["enhanced_pil"]).mean(),
            msg="Higher Light Boost preset should produce strictly brighter illumination",
        )

        uhd_pil = ui_app.upscale_to_hd_crisp(cls_out["enhanced_pil"], target_long_side=3840)
        self.assertEqual(max(uhd_pil.size), 3840)

    def test_streamlit_apptest_render_and_enhance_click(self):
        try:
            from streamlit.testing.v1 import AppTest
        except Exception:
            self.skipTest("streamlit.testing.v1.AppTest unavailable")

        at = AppTest.from_file(os.path.join(ROOT_DIR, "app.py"), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, msg=f"Initial Streamlit render error: {at.exception}")
        self.assertTrue(
            any("Team - Vijendra, Vyom, Gautam" in m.value for m in at.markdown),
            msg="Hero header should display Team - Vijendra, Vyom, Gautam",
        )

        # Switch Architecture selectbox (index 1 after Light Preset) to Config A and enhance
        arch_box = [sb for sb in at.sidebar.selectbox if "Configuration" in sb.label][0]
        arch_box.set_value("A").run()
        self.assertFalse(at.exception, msg=f"Config switch error: {at.exception}")

        run_btns = [b for b in at.button if "Run Enhancement" in b.label]
        self.assertEqual(len(run_btns), 1)
        run_btns[0].click().run()
        self.assertFalse(at.exception, msg=f"Run Enhancement error: {at.exception}")
        self.assertGreaterEqual(len(at.metric), 4)

        # Also verify clicking 1-click inline More Light button updates cleanly with 0 exceptions
        more_light_btns = [b for b in at.button if "More Light" in b.label]
        self.assertEqual(len(more_light_btns), 1)
        more_light_btns[0].click().run()
        self.assertFalse(at.exception, msg=f"More Light preset click error: {at.exception}")
        self.assertGreaterEqual(len(at.metric), 4)
        self.assertTrue(any("3840" in m.value for m in at.metric), msg="Default output should reach 4K (3840px)")

    def test_colab_notebook_structure_and_syntax(self):
        nb_path = os.path.join(ROOT_DIR, "notebooks", "Complete_Project_Implementation.ipynb")
        self.assertTrue(os.path.isfile(nb_path))
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb["nbformat"], 4)
        cells = nb["cells"]
        self.assertGreaterEqual(len(cells), 26)

        md_text = "\n".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "markdown"
        )
        required_sections = [
            "1. Environment Setup",
            "2. Dependency Installation",
            "3. Dataset Preparation",
            "4. Model Architecture",
            "5. Training",
            "6. Validation",
            "7. Checkpoint Loading",
            "8. Image Enhancement",
            "9. Metric Calculation",
            "10. Ablation Study",
            "11. Research Comparison",
            "12. Graph Generation",
            "13. Result Visualization",
        ]
        for sec in required_sections:
            self.assertIn(sec, md_text)

        for idx, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                ast.parse(src, filename=f"cell_{idx}")


if __name__ == "__main__":
    unittest.main()
