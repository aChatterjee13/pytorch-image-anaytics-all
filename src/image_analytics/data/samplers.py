"""Samplers: class-balanced sampling for imbalanced datasets."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def extract_targets(dataset: Dataset) -> Sequence[int]:
    """Best-effort extraction of integer class targets from a dataset.

    Uses the ``targets`` attribute when available (torchvision datasets,
    MultispectralDataset); falls back to iterating the dataset.
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(dataset[i][1]) for i in range(len(dataset))]
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return [int(t) for t in targets]


def build_balanced_sampler(
    dataset: Dataset,
    num_samples: int | None = None,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler: each class is drawn with roughly
    equal probability regardless of its share of the dataset."""
    targets = extract_targets(dataset)
    counts = Counter(targets)
    weights = torch.tensor(
        [1.0 / counts[t] for t in targets], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights,
        num_samples=num_samples or len(targets),
        replacement=True,
        generator=generator,
    )


def build_geo_sampler(
    dataset,
    kind: str = "random",
    size: int = 256,
    length: int = 1000,
    stride: int | None = None,
    **kwargs,
):
    """Geo-aware sampler over a torchgeo ``GeoDataset`` (``[geo]`` extra).

    ``random`` -> ``RandomGeoSampler`` (training); ``grid`` -> ``GridGeoSampler``
    (tiled inference). These yield CRS-aware bounding-box queries and pass
    straight through as DataLoader samplers (use torchgeo's ``stack_samples``
    collate with raster GeoDatasets).
    """
    try:
        from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "torchgeo is required for geo-aware samplers. "
            "Install it with: pip install 'image-analytics[geo]'"
        ) from exc

    if kind == "random":
        return RandomGeoSampler(dataset, size=size, length=length, **kwargs)
    if kind == "grid":
        return GridGeoSampler(dataset, size=size, stride=stride or size, **kwargs)
    raise ValueError(f"kind must be 'random' or 'grid', got {kind!r}")
