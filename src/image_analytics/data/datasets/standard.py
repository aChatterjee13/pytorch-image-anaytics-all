"""Standard 8-bit RGB datasets (torchvision-backed).

All factories follow the registry protocol:

    factory(root, split="train", transform=None, **kwargs) -> Dataset
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision.datasets as tvd
from PIL import Image
from torch.utils.data import Dataset

from image_analytics.core.registry import DATASETS


@DATASETS.register("cifar10")
def cifar10(
    root: str, split: str = "train", transform: Callable | None = None, download: bool = True
) -> Dataset:
    return tvd.CIFAR10(root, train=split == "train", transform=transform, download=download)


@DATASETS.register("cifar100")
def cifar100(
    root: str, split: str = "train", transform: Callable | None = None, download: bool = True
) -> Dataset:
    return tvd.CIFAR100(root, train=split == "train", transform=transform, download=download)


@DATASETS.register("image_folder")
def image_folder(
    root: str, split: str = "train", transform: Callable | None = None
) -> Dataset:
    """Class-per-subdirectory layout: ``root/{split}/{class_name}/*.jpg``."""
    return tvd.ImageFolder(str(Path(root) / split), transform=transform)


@DATASETS.register("fake")
def fake_data(
    root: str | None = None,
    split: str = "train",
    transform: Callable | None = None,
    size: int = 256,
    image_size: tuple[int, int, int] = (3, 224, 224),
    num_classes: int = 10,
) -> Dataset:
    """Synthetic dataset for smoke tests and offline demos (no download)."""
    # Different seed offset per split so train/val are not identical.
    offset = {"train": 0, "val": 1, "test": 2}.get(split, 3)
    return tvd.FakeData(
        size=size,
        image_size=tuple(image_size),
        num_classes=num_classes,
        transform=transform,
        random_offset=offset * size,
    )


@DATASETS.register("multilabel_csv")
class MultiLabelImageDataset(Dataset):
    """Multi-label dataset driven by per-split CSV files.

    Layout::

        root/
            train.csv          # header: filepath,label_a,label_b,...
            val.csv            # rows:   images/img1.jpg,1,0,...
            images/...

    Image paths are relative to ``root``; label columns are 0/1. Targets are
    float vectors, ready for ``BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        csv_name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        csv_path = self.root / (csv_name or f"{split}.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Label file not found: {csv_path}")

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.classes: list[str] = header[1:]
            if not self.classes:
                raise ValueError(f"{csv_path} must have label columns after 'filepath'")
            self.samples: list[tuple[str, torch.Tensor]] = []
            for row in reader:
                if not row:
                    continue
                labels = torch.tensor([float(v) for v in row[1:]], dtype=torch.float32)
                if labels.numel() != len(self.classes):
                    raise ValueError(
                        f"Row {row[0]!r} has {labels.numel()} labels, "
                        f"expected {len(self.classes)}"
                    )
                self.samples.append((row[0], labels))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        relpath, target = self.samples[index]
        image = Image.open(self.root / relpath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target
