"""CUDA-gated 3D wrappers must import safely and fail with a clear message on
a CPU-only machine."""

import pytest
import torch

import image_analytics.detection_3d  # noqa: F401  (register models)
from image_analytics.core.registry import MODELS

GATED = ["second", "centerpoint", "bevformer", "mask3d"]


def test_all_registered():
    for name in GATED:
        assert name in MODELS


@pytest.mark.skipif(torch.cuda.is_available(), reason="gate only triggers without CUDA")
@pytest.mark.parametrize("name", GATED)
def test_gated_raises_without_cuda(name):
    with pytest.raises((RuntimeError, ImportError), match="CUDA|3d-cuda"):
        MODELS.build(name, num_classes=3)
