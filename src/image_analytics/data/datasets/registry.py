"""Dataset registry and config-driven construction.

Registered factories follow the protocol:

    factory(root, split="train", transform=None, **kwargs) -> Dataset

``build_dataset`` forwards optional DataConfig fields (``bands``,
``normalize``) only to factories whose signature accepts them, so standard
RGB datasets are not polluted with multispectral arguments.
"""

from __future__ import annotations

import inspect
from typing import Any

from torch.utils.data import Dataset

from image_analytics.core.config import DataConfig
from image_analytics.core.registry import DATASETS


def _accepted_params(factory: Any) -> tuple[set[str], bool]:
    """Return (explicit parameter names, accepts **kwargs) for a factory."""
    target = factory.__init__ if inspect.isclass(factory) else factory
    sig = inspect.signature(target)
    names = set(sig.parameters)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return names, has_var_kw


def build_dataset(
    config: DataConfig, split: str = "train", transform: Any = None
) -> Dataset:
    """Build a dataset for one split from a :class:`DataConfig`."""
    factory = DATASETS.get(config.dataset)
    accepted, has_var_kw = _accepted_params(factory)

    kwargs: dict[str, Any] = dict(config.kwargs)
    if config.bands is not None and ("bands" in accepted or has_var_kw):
        kwargs.setdefault("bands", config.bands)
    # normalize is overloaded: "imagenet" configures the transform pipeline,
    # while multispectral datasets normalize at load time. Only forward it to
    # factories that explicitly declare the parameter.
    if "normalize" in accepted:
        kwargs.setdefault("normalize", config.normalize)
    if "mean" in accepted and config.mean is not None:
        kwargs.setdefault("mean", config.mean)
    if "std" in accepted and config.std is not None:
        kwargs.setdefault("std", config.std)

    return factory(root=config.root, split=split, transform=transform, **kwargs)
