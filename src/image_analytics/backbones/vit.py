"""ViT-family backbones: ViT, DeiT, and DINOv2/v3 self-supervised encoders.

DINOv2/v3 are strong few-shot feature extractors — linear probing lands
within ~2% of full fine-tuning on many tasks.
"""

from image_analytics.backbones.base import register_timm_backbones

register_timm_backbones(
    {
        "vit_tiny": "vit_tiny_patch16_224",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
        "deit3_small": "deit3_small_patch16_224",
        "deit3_base": "deit3_base_patch16_224",
        "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
        "dinov3_base": "vit_base_patch16_dinov3.lvd1689m",
    }
)
