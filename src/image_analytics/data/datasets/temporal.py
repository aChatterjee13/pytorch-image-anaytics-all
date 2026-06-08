"""Temporal / change-detection datasets (synthetic, offline).

* ``synthetic_change`` — a shapes scene and a mutated copy (one shape added or
  removed); the changed pixels are the GT mask. Sample:
  ``(image (2C, H, W), mask (H, W) int64)`` — t0 and t1 channel-concatenated so
  it flows through the segmentation pipeline; ``SiameseUNet`` splits it back.
* ``synthetic_temporal`` — a shape translating across ``T`` frames; the class is
  its motion direction. Sample: ``(clip (C, T, H, W), label)`` — feeds temporal
  encoders (Prithvi, ``temporal_classifier``).

Both reuse the Phase 2 shape rasterizer, so the fixtures stay consistent.
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


@DATASETS.register("synthetic_change")
class SyntheticChangeDetection(Dataset):
    """Bi-temporal change detection over the synthetic shapes scene."""

    CLASSES = ("no_change", "change")

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
        return 2  # no-change / change

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(self._offset + index)
        s = self.image_size

        img0 = random_background(s, gen)
        masks = []
        num_shapes = int(torch.randint(1, self.max_shapes + 1, (1,), generator=gen))
        for _ in range(num_shapes):
            label = int(torch.randint(0, len(SHAPE_CLASSES), (1,), generator=gen))
            box, mask = draw_shape(img0, label, gen, s)
            if box is not None:
                masks.append(mask)

        img1 = img0.clone()
        change = torch.zeros(s, s, dtype=torch.bool)
        add = bool(torch.randint(0, 2, (1,), generator=gen)) or not masks
        if add:  # a new shape appears in t1
            label = int(torch.randint(0, len(SHAPE_CLASSES), (1,), generator=gen))
            box, mask = draw_shape(img1, label, gen, s)
            if mask is not None:
                change = mask
        else:  # a shape disappears in t1 (repainted with fresh background)
            idx = int(torch.randint(0, len(masks), (1,), generator=gen))
            mask = masks[idx]
            patch = random_background(s, gen)
            img1[:, mask] = patch[:, mask]
            change = mask

        image = torch.cat([img0, img1], dim=0)              # (2C, H, W)
        target = tv_tensors.Mask(change.long())
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


@DATASETS.register("synthetic_temporal")
class TemporalStackDataset(Dataset):
    """A shape translating across ``num_frames``; label = motion direction
    (0:+x, 1:-x, 2:+y, 3:-y). Returns a ``(C, T, H, W)`` clip."""

    CLASSES = ("right", "left", "down", "up")
    _STEPS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        transform: Callable | None = None,
        size: int = 256,
        image_size: int = 64,
        num_frames: int = 4,
    ) -> None:
        self.split = split
        self.transform = transform
        self.size = size
        self.image_size = image_size
        self.num_frames = num_frames
        self._offset = _SPLIT_OFFSETS.get(split, 3) * 1_000_003

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        gen = torch.Generator().manual_seed(self._offset + index)
        s = self.image_size
        direction = int(torch.randint(0, 4, (1,), generator=gen))
        dx, dy = self._STEPS[direction]
        label = int(torch.randint(0, len(SHAPE_CLASSES), (1,), generator=gen))
        r = s // 8
        cx = int(torch.randint(2 * r, s - 2 * r, (1,), generator=gen))
        cy = int(torch.randint(2 * r, s - 2 * r, (1,), generator=gen))
        step = max(s // (3 * self.num_frames), 1)

        ys, xs = torch.meshgrid(
            torch.arange(s, dtype=torch.float32), torch.arange(s, dtype=torch.float32), indexing="ij"
        )
        color = torch.rand(3, generator=gen) * 0.6 + 0.4
        frames = []
        for t in range(self.num_frames):
            frame = random_background(s, gen)
            ox = min(max(cx + dx * step * t, r), s - r)
            oy = min(max(cy + dy * step * t, r), s - r)
            mask = (ys - oy) ** 2 + (xs - ox) ** 2 <= r**2
            frame[:, mask] = color.unsqueeze(1)
            frames.append(frame)

        clip = torch.stack(frames, dim=1)                   # (C, T, H, W)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip, direction
