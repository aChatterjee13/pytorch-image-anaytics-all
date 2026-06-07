"""ConvNeXt / V2 backbones (Liu 2022/2023) — modernized CNN design."""

from image_analytics.backbones.base import register_timm_backbones

register_timm_backbones(
    {
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
        "convnextv2_tiny": "convnextv2_tiny",
        "convnextv2_base": "convnextv2_base",
    }
)
