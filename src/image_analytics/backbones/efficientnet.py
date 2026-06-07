"""EfficientNet / V2 backbones (Tan & Le 2019/2021) — compound scaling."""

from image_analytics.backbones.base import register_timm_backbones

register_timm_backbones(
    {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "efficientnet_b2": "efficientnet_b2",
        "efficientnet_b3": "efficientnet_b3",
        "efficientnetv2_s": "tf_efficientnetv2_s",
        "efficientnetv2_m": "tf_efficientnetv2_m",
        "mobilenetv3_small": "mobilenetv3_small_100",
        "mobilenetv3_large": "mobilenetv3_large_100",
    }
)
