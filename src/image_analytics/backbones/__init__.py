"""Feature extractors. Importing this package populates BACKBONES."""

from image_analytics.backbones import (  # noqa: F401  (registration side effects)
    convnext,
    efficientnet,
    resnet,
    swin,
    vit,
)
from image_analytics.foundation import (  # noqa: F401  (register satellite backbones)
    prithvi,
    satmae,
)
from image_analytics.backbones.base import TimmBackbone
from image_analytics.backbones.multichannel import (
    ChannelAttentionInput,
    GroupedBandStem,
    GroupedStemBackbone,
    MultiChannelBackbone,
    adapt_first_conv,
)
from image_analytics.backbones.registry import BACKBONES, build_backbone

__all__ = [
    "BACKBONES",
    "TimmBackbone",
    "ChannelAttentionInput",
    "GroupedBandStem",
    "GroupedStemBackbone",
    "MultiChannelBackbone",
    "adapt_first_conv",
    "build_backbone",
]
