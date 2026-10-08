"""GPU tests of the GLM-5.3-Flash patches: one GPU (any DGX Spark), the recipe's image (the patched TensorFold
installed; TF_SRC points at another patched source tree instead). Small: single kernels on synthetic tensors at a
rank's real shapes, a few GiB at most. scripts/test-gpu.sh runs them."""

import os
import sys

import pytest

if os.environ.get("TF_SRC"):
    sys.path.insert(0, os.environ["TF_SRC"])
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cpu"))     # the tiny checkpoint (flash_fakes)


def pytest_configure(config):
    """TEST_GPU_GIB: the most GPU memory the tests may take (a Spark that serves something else meanwhile)."""
    gib = os.environ.get("TEST_GPU_GIB")
    if gib:
        import torch

        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(gib) * 2**30 / total))


def pytest_collection_modifyitems(config, items):
    import torch

    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA GPU")
        for item in items:
            item.add_marker(skip)
