"""TorchGeo dataset adapter (``[geo]`` extra).

TorchGeo handles CRS-aware geospatial datasets and spatial sampling. This thin
adapter maps a torchgeo (map-style) dataset's sample dicts — ``{"image", ...}``
with ``"label"`` (classification) or ``"mask"`` (segmentation) — to this
platform's ``(image, target)`` protocol, so any of our heads/pipelines consume
torchgeo data. Geo-aware samplers for raster ``GeoDataset``s live in
``data/samplers.py``. CRS handling stays inside torchgeo.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

from image_analytics.core.registry import DATASETS


def _load_torchgeo():
    try:
        import torchgeo  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "torchgeo is required for geospatial datasets. "
            "Install it with: pip install 'image-analytics[geo]'"
        ) from exc
    import torchgeo

    return torchgeo


class TorchGeoAdapter(Dataset):
    """Wrap a map-style torchgeo dataset -> ``(image, target)``."""

    def __init__(
        self, dataset, task: str = "classification", transform: Callable | None = None
    ) -> None:
        self.dataset = dataset
        self.task = task
        self.transform = transform
        self.classes = getattr(dataset, "classes", None)

    @property
    def num_classes(self) -> int:
        if self.classes is None:
            raise AttributeError("underlying torchgeo dataset exposes no `classes`")
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        image = sample["image"].float()
        if self.task == "segmentation":
            target = sample["mask"].long()
        else:
            target = int(sample["label"]) if "label" in sample else sample.get("mask")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


@DATASETS.register("torchgeo")
def build_torchgeo(
    root: str,
    split: str = "train",
    transform: Callable | None = None,
    dataset: str = "EuroSAT",
    task: str = "classification",
    download: bool = False,
    **kwargs,
) -> TorchGeoAdapter:
    """Construct a named torchgeo dataset (e.g. ``EuroSAT``, ``BigEarthNet``)
    and adapt it. Extra kwargs pass through to the torchgeo dataset."""
    _load_torchgeo()
    from torchgeo import datasets as tg_datasets

    if not hasattr(tg_datasets, dataset):
        raise ValueError(f"Unknown torchgeo dataset {dataset!r}")
    inner = getattr(tg_datasets, dataset)(root=root, split=split, download=download, **kwargs)
    return TorchGeoAdapter(inner, task=task, transform=transform)
