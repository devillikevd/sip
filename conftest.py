"""
conftest.py — Pytest configuration for the Low-Light Enhancement project.

Sets KMP_DUPLICATE_LIB_OK before any test imports torch/MKL so that the
OpenMP duplicate-library crash is avoided on Windows (Intel MKL + libomp).
Also seeds torch/numpy/random for reproducibility.
"""
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def pytest_configure(config):
    """Called once before test collection; safest place for env + seed setup."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

