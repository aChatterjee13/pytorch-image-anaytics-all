"""Camera-only BEV detector (BEVFormer) — CUDA-gated wrapper.

BEVFormer's deformable BEV attention is built through mmdet3d on a Linux/CUDA
box. On this machine the factory gates with an actionable error.
"""

from __future__ import annotations

import torch.nn as nn

from image_analytics.core.registry import MODELS
from image_analytics.detection_3d._cuda_gate import require_cuda_packages


@MODELS.register("bevformer")
def build_bevformer(num_classes: int, model_cfg: dict | None = None, **kwargs) -> nn.Module:
    """BEVFormer (Li 2022): spatiotemporal transformer over BEV queries."""
    require_cuda_packages("BEVFormer", ["mmdet3d"])
    if model_cfg is None:  # pragma: no cover - GPU only
        raise ValueError("build_bevformer requires an mmdet3d 'model_cfg' dict")
    from mmdet3d.registry import MODELS as MMDET3D_MODELS  # pragma: no cover - GPU only

    return MMDET3D_MODELS.build(model_cfg)  # pragma: no cover - GPU only
