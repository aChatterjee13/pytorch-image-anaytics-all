"""U-Net (Ronneberger 2015) over any registered pyramid backbone.

Encoder = a timm backbone in ``features_only`` mode (multi-channel capable, so
a 13-band U-Net comes for free). Decoder = repeated ``upsample x2 -> concat
skip -> double conv`` blocks, one per encoder level, deepest to shallowest;
the final block upsamples to input resolution and a 1x1 conv produces the
class logits. Decoder widths are configurable (default 256/128/64/32/16).

Interface: ``model(images) -> logits (B, num_classes, H, W)`` at the input
resolution — trained with a pixel-wise criterion (see ``segmentation/losses``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import MODELS


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    """Upsample x2, optionally concat a skip connection, then double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _double_conv(in_ch + skip_ch, out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


@MODELS.register("unet")
class UNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "UNet requires a pyramid backbone; set backbone.features_only: "
                "true (out_indices covering all encoder levels)"
            )
        self.backbone = backbone
        self.num_classes = num_classes

        enc_channels = list(backbone.feature_channels)  # shallow -> deep
        num_levels = len(enc_channels)
        if len(decoder_channels) != num_levels:
            raise ValueError(
                f"decoder_channels has {len(decoder_channels)} entries but the "
                f"encoder produces {num_levels} feature levels; pass one decoder "
                f"width per level (e.g. out_indices and decoder_channels must align)"
            )

        # Skips are all encoder features except the deepest (the decoder input),
        # consumed deep -> shallow; the final block upsamples with no skip.
        skip_channels = list(reversed(enc_channels[:-1])) + [0]
        decoder_blocks = []
        in_ch = enc_channels[-1]
        for skip_ch, out_ch in zip(skip_channels, decoder_channels):
            decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(decoder_blocks)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)            # shallow -> deep
        skips = list(reversed(features[:-1])) + [None]

        x = features[-1]
        for block, skip in zip(self.decoder, skips):
            x = block(x, skip)

        x = self.head(self.dropout(x))
        if x.shape[-2:] != images.shape[-2:]:
            x = F.interpolate(
                x, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        return x
