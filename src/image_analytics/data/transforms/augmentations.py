"""Augmentation pipelines built on torchvision transforms v2.

The pipelines accept both PIL images (standard RGB datasets) and float
tensors (multispectral datasets, which normalize at load time — pass
``normalize="none"`` so no ImageNet statistics are applied).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUGMENT_POLICIES = ("none", "default", "strong")


def build_transforms(
    image_size: int,
    train: bool = True,
    augment: str = "default",
    normalize: str = "imagenet",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    hflip: float = 0.5,
) -> v2.Compose:
    """Build a transform pipeline.

    Args:
        image_size: output spatial size (square).
        train: training (random crops/flips) vs eval (resize + center crop).
        augment: ``none`` | ``default`` (RRC + flip) | ``strong`` (adds
            RandAugment + random erasing; RGB inputs only).
        normalize: ``imagenet`` applies ImageNet stats (or ``mean``/``std``
            when given); anything else skips normalization here (multispectral
            datasets normalize at load time, see ``transforms/spectral.py``).
        mean/std: custom normalization statistics.
    """
    if augment not in AUGMENT_POLICIES:
        raise ValueError(f"augment must be one of {AUGMENT_POLICIES}, got {augment!r}")

    ops: list[v2.Transform] = []
    if train and augment != "none":
        ops.append(
            v2.RandomResizedCrop(image_size, scale=(0.5, 1.0), antialias=True)
        )
        if hflip > 0:
            ops.append(v2.RandomHorizontalFlip(p=hflip))
        if augment == "strong":
            ops.append(v2.RandAugment())
    else:
        ops.append(v2.Resize(image_size, antialias=True))
        ops.append(v2.CenterCrop(image_size))

    # PIL -> tensor; uint8 -> float in [0, 1]. No-op for float tensor inputs.
    ops.append(v2.ToImage())
    ops.append(v2.ToDtype(torch.float32, scale=True))

    if normalize == "imagenet" or (mean is not None and std is not None):
        ops.append(
            v2.Normalize(
                mean=list(mean) if mean is not None else list(IMAGENET_MEAN),
                std=list(std) if std is not None else list(IMAGENET_STD),
            )
        )

    if train and augment == "strong":
        ops.append(v2.RandomErasing(p=0.25))

    return v2.Compose(ops)
