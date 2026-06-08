"""Voxel-based 3D detectors (SECOND, CenterPoint) — CUDA-gated wrappers.

Both rely on spconv's sparse 3D convolutions (no CPU/macOS wheels), so they are
built through mmdet3d on a Linux/CUDA box. On this machine the factories gate
with an actionable error; importing the module is always safe.
"""

from __future__ import annotations

import torch.nn as nn

from image_analytics.core.registry import MODELS
from image_analytics.detection_3d._cuda_gate import require_cuda_packages


def _build_mmdet3d(model_cfg: dict) -> nn.Module:  # pragma: no cover - GPU only
    from mmdet3d.registry import MODELS as MMDET3D_MODELS

    return MMDET3D_MODELS.build(model_cfg)


@MODELS.register("second")
def build_second(num_classes: int, model_cfg: dict | None = None, **kwargs) -> nn.Module:
    """SECOND (Yan 2018): sparse-conv voxel detector. Provide an mmdet3d
    ``model_cfg`` (or use mmdet3d's config zoo)."""
    require_cuda_packages("SECOND", ["spconv", "mmdet3d"])
    if model_cfg is None:  # pragma: no cover - GPU only
        raise ValueError("build_second requires an mmdet3d 'model_cfg' dict")
    return _build_mmdet3d(model_cfg)  # pragma: no cover - GPU only


@MODELS.register("centerpoint")
def build_centerpoint(num_classes: int, model_cfg: dict | None = None, **kwargs) -> nn.Module:
    """CenterPoint (Yin 2021): anchor-free center-based voxel detector."""
    require_cuda_packages("CenterPoint", ["spconv", "mmdet3d"])
    if model_cfg is None:  # pragma: no cover - GPU only
        raise ValueError("build_centerpoint requires an mmdet3d 'model_cfg' dict")
    return _build_mmdet3d(model_cfg)  # pragma: no cover - GPU only
