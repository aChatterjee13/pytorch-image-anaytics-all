"""Swin Transformer backbones (Liu 2021) — shifted window attention,
hierarchical features; strong default for detection/segmentation necks."""

from image_analytics.backbones.base import register_timm_backbones

register_timm_backbones(
    {
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
        "swinv2_tiny": "swinv2_tiny_window8_256",
        "swinv2_base": "swinv2_base_window8_256",
    }
)
