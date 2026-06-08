import importlib.util

import pytest

import image_analytics.data.datasets  # noqa: F401  (register torchgeo)
from image_analytics.core.registry import DATASETS

_HAS_TORCHGEO = importlib.util.find_spec("torchgeo") is not None


def test_registered():
    assert "torchgeo" in DATASETS


@pytest.mark.skipif(_HAS_TORCHGEO, reason="torchgeo installed; lazy-error path n/a")
def test_dataset_raises_without_torchgeo():
    from image_analytics.data.datasets.torchgeo_adapter import build_torchgeo

    with pytest.raises(ImportError, match="torchgeo"):
        build_torchgeo(root="data", dataset="EuroSAT")


@pytest.mark.skipif(_HAS_TORCHGEO, reason="torchgeo installed; lazy-error path n/a")
def test_geo_sampler_raises_without_torchgeo():
    from image_analytics.data.samplers import build_geo_sampler

    with pytest.raises(ImportError, match="torchgeo"):
        build_geo_sampler(object(), kind="random")
