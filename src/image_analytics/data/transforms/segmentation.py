"""Mask-aware transform pipelines (torchvision v2 + tv_tensors).

Semantic samples are ``(image, mask)`` and instance samples are
``(image, target_dict)`` carrying ``boxes`` and per-instance ``masks``. v2
geometric transforms resample masks with nearest-neighbour automatically, and
the dtype conversion is targeted at the image so class-index masks keep their
integer labels (a blanket ``ToDtype(float32, scale=True)`` would divide masks
by 255).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from image_analytics.data.transforms.augmentations import IMAGENET_MEAN, IMAGENET_STD


def build_segmentation_transforms(
    image_size: int,
    train: bool = True,
    hflip: float = 0.5,
    normalize: str = "imagenet",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    instance: bool = False,
) -> v2.Compose:
    """Segmentation pipeline: flip (train) -> square resize -> normalize.

    Args:
        instance: append ``SanitizeBoundingBoxes`` for instance segmentation
            (drops degenerate boxes and their labels/masks after resizing).
    """
    ops: list[v2.Transform] = []
    if train and hflip > 0:
        ops.append(v2.RandomHorizontalFlip(p=hflip))
    ops.append(v2.Resize((image_size, image_size), antialias=True))
    ops.append(v2.ToImage())
    # Scale only the image to float [0, 1]; masks keep their integer labels.
    ops.append(
        v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True)
    )
    if normalize == "imagenet" or (mean is not None and std is not None):
        ops.append(
            v2.Normalize(
                mean=list(mean) if mean is not None else list(IMAGENET_MEAN),
                std=list(std) if std is not None else list(IMAGENET_STD),
            )
        )
    if instance:
        ops.append(v2.SanitizeBoundingBoxes())
    return v2.Compose(ops)
