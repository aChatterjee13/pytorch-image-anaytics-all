"""Procedural detection dataset: colored shapes on a noisy background.

Fully offline and deterministic per (split, index) — a detector trainable on
CPU in minutes, used for tests, smoke configs, and the detection notebook.
The shape drawing lives in :mod:`image_analytics.data.datasets._shapes` so the
segmentation fixtures share the exact same images (the drawn pixels are also
their masks).

Classes (0-based labels): 0=rectangle, 1=circle, 2=triangle.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from image_analytics.core.registry import DATASETS
from image_analytics.data.datasets._shapes import (
    SHAPE_CLASSES,
    draw_shape,
    random_background,
)

_SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}


@DATASETS.register("synthetic_shapes")
class SyntheticShapesDetection(Dataset):
    """Generates images with 1..max_shapes random shapes and tight boxes.

    Args:
        size: number of samples in the split.
        image_size: square image side in pixels.
        max_shapes: maximum shapes per image.
    """

    CLASSES = SHAPE_CLASSES

    def __init__(
        self,
        root: str | None = None,  # unused; registry protocol compatibility
        split: str = "train",
        transform: Callable | None = None,
        size: int = 256,
        image_size: int = 96,
        max_shapes: int = 3,
    ) -> None:
        self.split = split
        self.transform = transform
        self.size = size
        self.image_size = image_size
        self.max_shapes = max_shapes
        self._offset = _SPLIT_OFFSETS.get(split, 3) * 1_000_003

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(self._offset + index)
        s = self.image_size

        image = random_background(s, generator)

        num_shapes = int(torch.randint(1, self.max_shapes + 1, (1,), generator=generator))
        boxes, labels = [], []
        for _ in range(num_shapes):
            label = int(torch.randint(0, len(self.CLASSES), (1,), generator=generator))
            box, _mask = draw_shape(image, label, generator, s)
            if box is not None:
                boxes.append(box)
                labels.append(label)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.stack(boxes), format="XYXY", canvas_size=(s, s)
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
