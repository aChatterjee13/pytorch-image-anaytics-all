"""Dataset implementations. Importing this package populates DATASETS."""

from image_analytics.data.datasets import (  # noqa: F401  (registration side effects)
    coco,
    multispectral,
    pointcloud,
    segmentation,
    standard,
    synthetic_shapes,
    temporal,
    torchgeo_adapter,
)
from image_analytics.data.datasets.registry import DATASETS, build_dataset

__all__ = ["DATASETS", "build_dataset"]
