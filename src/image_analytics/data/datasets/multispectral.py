"""Multispectral / 16-bit imagery datasets (rasterio-backed).

Handles GeoTIFF inputs with arbitrary channel counts and bit depths —
e.g. 13-band Sentinel-2 uint16 tiles. Normalization happens at load time
(per-image percentile/min-max, or dataset-level z-score), so the downstream
transform pipeline should be built with ``normalize="none"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from image_analytics.core.registry import DATASETS
from image_analytics.data.transforms.spectral import (
    minmax_normalize,
    percentile_normalize,
    zscore_normalize,
)

NORMALIZE_MODES = ("percentile", "minmax", "zscore", "none")


def _load_rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "rasterio is required for multispectral datasets. "
            "Install it with: pip install 'image-analytics[geo]'"
        ) from exc
    return rasterio


@DATASETS.register("multispectral")
class MultispectralDataset(Dataset):
    """Classification dataset over GeoTIFF tiles in a class-per-subdirectory
    layout::

        root/
            train/
                forest/tile_001.tif
                water/tile_042.tif
            val/...

    Args:
        bands: 0-based band indices to select (None = all bands).
        normalize: ``percentile`` (default, robust for satellite data) |
            ``minmax`` | ``zscore`` (requires ``mean``/``std``) | ``none``.
        percentiles: (low, high) clip percentiles for ``percentile`` mode.
        mean/std: per-selected-band statistics for ``zscore`` mode.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        bands: Sequence[int] | None = None,
        normalize: str = "percentile",
        percentiles: tuple[float, float] = (2.0, 98.0),
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        extensions: Sequence[str] = (".tif", ".tiff"),
    ) -> None:
        if normalize not in NORMALIZE_MODES:
            raise ValueError(
                f"normalize must be one of {NORMALIZE_MODES} for multispectral "
                f"data, got {normalize!r} (ImageNet statistics do not apply to "
                f"multi-band imagery — set data.normalize accordingly)"
            )
        if normalize == "zscore" and (mean is None or std is None):
            raise ValueError("normalize='zscore' requires mean and std")

        self.transform = transform
        self.bands = list(bands) if bands is not None else None
        self.normalize = normalize
        self.percentiles = percentiles
        self.mean = list(mean) if mean is not None else None
        self.std = list(std) if std is not None else None

        split_dir = Path(root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if not class_dirs:
            raise FileNotFoundError(f"No class subdirectories under {split_dir}")
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        exts = {e.lower() for e in extensions}
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in exts:
                    self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No raster files with extensions {sorted(exts)} under {split_dir}"
            )

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    @property
    def num_bands(self) -> int:
        if self.bands is not None:
            return len(self.bands)
        rasterio = _load_rasterio()
        with rasterio.open(self.samples[0][0]) as src:
            return src.count

    def _read(self, path: Path) -> torch.Tensor:
        rasterio = _load_rasterio()
        with rasterio.open(path) as src:
            if self.bands is not None:
                # rasterio band indexes are 1-based
                array = src.read([b + 1 for b in self.bands])
            else:
                array = src.read()
        return torch.from_numpy(array.astype(np.float32))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "percentile":
            return percentile_normalize(x, *self.percentiles)
        if self.normalize == "minmax":
            return minmax_normalize(x)
        if self.normalize == "zscore":
            return zscore_normalize(x, self.mean, self.std)
        return x

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        x = self._normalize(self._read(path))
        if self.transform is not None:
            x = self.transform(x)
        return x, label
