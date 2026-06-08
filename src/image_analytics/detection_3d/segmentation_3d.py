"""3D instance segmentation (Mask3D) — CUDA-gated wrapper.

Mask3D builds on MinkowskiEngine sparse convolutions (no CPU/macOS wheels). On
this machine the factory gates with an actionable error; importing is safe.
"""

from __future__ import annotations

import torch.nn as nn

from image_analytics.core.registry import MODELS
from image_analytics.detection_3d._cuda_gate import require_cuda_packages


@MODELS.register("mask3d")
def build_mask3d(num_classes: int, **kwargs) -> nn.Module:
    """Mask3D (Schult 2023): transformer 3D instance segmentation — the 3D
    analogue of Mask2Former, on MinkowskiEngine sparse features."""
    require_cuda_packages("Mask3D", ["MinkowskiEngine"])
    raise NotImplementedError(  # pragma: no cover - GPU only
        "Mask3D is built from its reference implementation on a CUDA box; "
        "this wrapper only gates availability."
    )
