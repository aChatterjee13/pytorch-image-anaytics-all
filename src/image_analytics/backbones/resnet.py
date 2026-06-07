"""ResNet family backbones (He 2015) — residual skip connections."""

from image_analytics.backbones.base import register_timm_backbones

register_timm_backbones(
    {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "resnet152": "resnet152",
        "resnext50": "resnext50_32x4d",
        "wide_resnet50": "wide_resnet50_2",
    }
)
