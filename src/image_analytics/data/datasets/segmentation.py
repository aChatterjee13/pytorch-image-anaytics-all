"""Segmentation datasets: semantic (class-index masks) and instance.

Sample protocols
----------------
* **Semantic**: ``(image (C, H, W) float, mask (H, W) int64)`` where the mask
  holds class indices and ``ignore_index`` (default 255) marks pixels excluded
  from the loss and metrics. The mask is a ``tv_tensors.Mask`` so v2 geometric
  transforms resample it with nearest-neighbour automatically.
* **Instance**: the Phase 2 detection target dict extended with
  ``"masks": tv_tensors.Mask (N, H, W) uint8`` — one binary mask per box.

The synthetic fixtures reuse the Phase 2 shape rasterizer
(:mod:`image_analytics.data.datasets._shapes`), so ``synthetic_shapes[i]`` and
``synthetic_shapes_seg[i]`` are the *same image* with consistent boxes/masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from image_analytics.core.registry import DATASETS
from image_analytics.data.datasets._shapes import (
    SHAPE_CLASSES,
    draw_shape,
    random_background,
)

_SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}

# Semantic shape classes reserve index 0 for background.
SEMANTIC_SHAPE_CLASSES = ("background", *SHAPE_CLASSES)


@DATASETS.register("segmentation_folder")
class SemanticSegmentationDataset(Dataset):
    """Class-index mask dataset in an images/masks folder layout::

        root/
            images/{split}/<name>.jpg
            masks/{split}/<name>.png      # L-mode, pixel value = class index

    Images and masks are paired by filename stem. Mask pixels equal to
    ``ignore_index`` are ignored downstream (loss + metrics).

    Args:
        classes: optional class names; ``num_classes`` is derived from them.
        image_subdir / mask_subdir: top-level folders for images and masks.
        mask_suffix: mask file extension (masks are single-channel PNGs).
        ignore_index: pixel value marking ignored regions.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        classes: Sequence[str] | None = None,
        image_subdir: str = "images",
        mask_subdir: str = "masks",
        mask_suffix: str = ".png",
        ignore_index: int = 255,
        extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    ) -> None:
        self.transform = transform
        self.ignore_index = ignore_index
        self.classes = list(classes) if classes is not None else None

        image_dir = Path(root) / image_subdir / split
        mask_dir = Path(root) / mask_subdir / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        exts = {e.lower() for e in extensions}
        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            mask_path = mask_dir / (img_path.stem + mask_suffix)
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found under {image_dir} + {mask_dir}"
            )

    @property
    def num_classes(self) -> int:
        if self.classes is None:
            raise AttributeError(
                "num_classes is unknown; pass classes=[...] to "
                "SemanticSegmentationDataset or set model.num_classes in the config"
            )
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, mask_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path), dtype=np.int64)
        mask = tv_tensors.Mask(torch.from_numpy(mask))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


@DATASETS.register("synthetic_shapes_seg")
class SyntheticShapesSegmentation(Dataset):
    """Semantic-segmentation twin of ``synthetic_shapes``.

    Returns ``(image, mask)`` with ``mask[pixel] = shape_label + 1`` (0 is
    background). Deterministic and offline; trains a U-Net on CPU in minutes.
    """

    CLASSES = SEMANTIC_SHAPE_CLASSES

    def __init__(
        self,
        root: str | None = None,  # unused; registry protocol compatibility
        split: str = "train",
        transform: Callable | None = None,
        size: int = 256,
        image_size: int = 96,
        max_shapes: int = 3,
        ignore_index: int = 255,
    ) -> None:
        self.split = split
        self.transform = transform
        self.size = size
        self.image_size = image_size
        self.max_shapes = max_shapes
        self.ignore_index = ignore_index
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
        mask = torch.zeros(s, s, dtype=torch.int64)

        num_shapes = int(torch.randint(1, self.max_shapes + 1, (1,), generator=generator))
        for _ in range(num_shapes):
            label = int(torch.randint(0, len(SHAPE_CLASSES), (1,), generator=generator))
            _box, shape_mask = draw_shape(image, label, generator, s)
            if shape_mask is not None:
                mask[shape_mask] = label + 1  # later shapes occlude earlier

        mask = tv_tensors.Mask(mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask


@DATASETS.register("synthetic_shapes_instance")
class SyntheticShapesInstanceSegmentation(Dataset):
    """Instance-segmentation twin of ``synthetic_shapes``.

    Returns ``(image, target)`` with the Phase 2 detection target dict plus a
    per-instance ``masks`` (N, H, W) uint8 ``tv_tensors.Mask``. Labels are
    0-based foreground classes (Mask R-CNN adds the internal background class).
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
        boxes, labels, masks = [], [], []
        for _ in range(num_shapes):
            label = int(torch.randint(0, len(SHAPE_CLASSES), (1,), generator=generator))
            box, shape_mask = draw_shape(image, label, generator, s)
            if box is not None:
                boxes.append(box)
                labels.append(label)
                masks.append(shape_mask.to(torch.uint8))

        boxes_t = torch.stack(boxes) if boxes else torch.zeros(0, 4)
        masks_t = (
            torch.stack(masks) if masks else torch.zeros(0, s, s, dtype=torch.uint8)
        )
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_t, format="XYXY", canvas_size=(s, s)
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": tv_tensors.Mask(masks_t),
            "image_id": torch.tensor([index]),
        }
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
