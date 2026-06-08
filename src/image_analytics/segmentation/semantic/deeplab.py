"""DeepLabv3+ (Chen 2018) from scratch.

ASPP (atrous rates 1/6/12/18 + image-level pooling) on the output-stride-16
high-level features, then a lightweight decoder that fuses the stride-4
low-level features through a 48-channel projection before the final upsample.

The encoder is a timm backbone built with ``output_stride=16`` exposing two
levels — low-level (C2, stride 4) and high-level (C5, stride 16). Interface:
``model(images) -> logits (B, num_classes, H, W)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import MODELS


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 1, dilation: int = 1) -> nn.Sequential:
    padding = dilation if kernel > 1 else 0
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: parallel atrous convs + image pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: tuple[int, ...] = (1, 6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            _conv_bn_relu(in_channels, out_channels, kernel=1 if r == 1 else 3, dilation=r)
            for r in rates
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 1) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


@MODELS.register("deeplabv3plus")
class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        aspp_channels: int = 256,
        aspp_rates: tuple[int, ...] = (1, 6, 12, 18),
        low_level_channels: int = 48,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "DeepLabV3Plus requires a pyramid backbone in output_stride=16 "
                "mode exposing [low-level (stride 4), high-level (stride 16)]; "
                "set backbone.features_only: true with out_indices and "
                "kwargs.output_stride: 16"
            )
        channels = list(backbone.feature_channels)
        if len(channels) < 2:
            raise ValueError(
                f"DeepLabV3Plus needs at least 2 encoder levels (low + high), "
                f"got {len(channels)}"
            )
        self.backbone = backbone
        self.num_classes = num_classes

        self.aspp = ASPP(channels[-1], aspp_channels, aspp_rates, dropout)
        self.low_level_proj = _conv_bn_relu(channels[0], low_level_channels, kernel=1)
        self.decoder = nn.Sequential(
            _conv_bn_relu(aspp_channels + low_level_channels, decoder_channels, kernel=3),
            _conv_bn_relu(decoder_channels, decoder_channels, kernel=3),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        low, high = features[0], features[-1]

        x = self.aspp(high)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_level_proj(low)], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        return F.interpolate(
            x, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
