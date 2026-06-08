"""Spectral transforms: band normalization and spectral indices.

All functions operate on float tensors of shape (C, H, W). Indices follow the
standard remote-sensing definitions:

    NDVI = (NIR - Red) / (NIR + Red)                          vegetation
    NDWI = (Green - NIR) / (Green + NIR)                      water
    NDBI = (SWIR - NIR) / (SWIR + NIR)                        built-up areas
    EVI  = 2.5 (NIR - Red) / (NIR + 6 Red - 7.5 Blue + 1)     enhanced vegetation
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Normalization (per-channel, for 16-bit / multispectral inputs)
# ---------------------------------------------------------------------------


def percentile_normalize(
    x: torch.Tensor, low: float = 2.0, high: float = 98.0
) -> torch.Tensor:
    """Per-channel percentile clipping to [0, 1] — robust to sensor outliers;
    the standard choice for satellite imagery."""
    flat = x.reshape(x.shape[0], -1).float()
    qs = torch.quantile(
        flat, torch.tensor([low / 100.0, high / 100.0]), dim=1
    )
    lo = qs[0].reshape(-1, 1, 1)
    hi = qs[1].reshape(-1, 1, 1)
    return ((x.float() - lo) / (hi - lo + _EPS)).clamp(0.0, 1.0)


def minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-channel min-max scaling to [0, 1]."""
    flat = x.reshape(x.shape[0], -1).float()
    lo = flat.min(dim=1).values.reshape(-1, 1, 1)
    hi = flat.max(dim=1).values.reshape(-1, 1, 1)
    return (x.float() - lo) / (hi - lo + _EPS)


def zscore_normalize(
    x: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """Per-channel standardization with dataset-level statistics."""
    mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1, 1, 1)
    if mean_t.shape[0] != x.shape[0] or std_t.shape[0] != x.shape[0]:
        raise ValueError(
            f"mean/std have {mean_t.shape[0]}/{std_t.shape[0]} entries "
            f"but input has {x.shape[0]} channels"
        )
    return (x.float() - mean_t) / (std_t + _EPS)


class PercentileNormalize:
    def __init__(self, low: float = 2.0, high: float = 98.0) -> None:
        self.low, self.high = low, high

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return percentile_normalize(x, self.low, self.high)


class MinMaxNormalize:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return minmax_normalize(x)


class ZScoreNormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean, self.std = mean, std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return zscore_normalize(x, self.mean, self.std)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------


def normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(a - b) / (a + b), numerically safe."""
    return (a - b) / (a + b + _EPS)


def compute_index(
    x: torch.Tensor, index: str, band_map: Mapping[str, int]
) -> torch.Tensor:
    """Compute a named spectral index from a (C, H, W) tensor.

    ``band_map`` maps semantic band names (``nir``, ``red``, ``green``,
    ``blue``, ``swir``) to channel positions in ``x``.
    """
    x = x.float()

    def band(name: str) -> torch.Tensor:
        if name not in band_map:
            raise KeyError(
                f"Index {index!r} requires band {name!r}; band_map has {sorted(band_map)}"
            )
        return x[band_map[name]]

    index = index.lower()
    if index == "ndvi":
        return normalized_difference(band("nir"), band("red"))
    if index == "ndwi":
        return normalized_difference(band("green"), band("nir"))
    if index == "ndbi":
        return normalized_difference(band("swir"), band("nir"))
    if index == "evi":
        nir, red, blue = band("nir"), band("red"), band("blue")
        return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + _EPS)
    raise ValueError(f"Unknown spectral index {index!r}; expected ndvi|ndwi|ndbi|evi")


class AppendIndex:
    """Append a computed spectral index as an extra channel.

    Example (Sentinel-2 L2A band order, 0-based):

        transform = AppendIndex("ndvi", band_map={"nir": 7, "red": 3})
    """

    def __init__(self, index: str, band_map: Mapping[str, int]) -> None:
        self.index = index
        self.band_map = dict(band_map)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        idx = compute_index(x, self.index, self.band_map)
        return torch.cat([x.float(), idx.unsqueeze(0)], dim=0)
