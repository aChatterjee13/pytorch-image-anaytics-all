"""Change detection and temporal pooling for satellite time series.

* :class:`SiameseUNet` — a shared-weight pyramid encoder runs on both dates of
  a bi-temporal pair; per-level absolute feature differences feed a U-Net
  decoder (Phase 3 reuse) to a change-mask. Input is the two dates
  channel-concatenated ``(B, 2C, H, W)`` (what ``synthetic_change`` yields), so
  it trains through the segmentation pipeline with a CE / CE+Dice criterion.
* :class:`TemporalPoolingClassifier` — runs a 2D backbone per frame of a
  ``(B, C, T, H, W)`` clip and pools features over time (mean / max / attention)
  before a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import MODELS
from image_analytics.segmentation.semantic.unet import DecoderBlock


@MODELS.register("siamese_unet")
class SiameseUNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        in_channels_per_image: int | None = None,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "SiameseUNet requires a pyramid backbone; set backbone.features_only: true"
            )
        self.backbone = backbone
        self.in_ch = in_channels_per_image or getattr(backbone, "in_channels", 3)

        enc_channels = list(backbone.feature_channels)
        if len(decoder_channels) != len(enc_channels):
            raise ValueError(
                f"decoder_channels has {len(decoder_channels)} entries but the "
                f"encoder produces {len(enc_channels)} levels"
            )
        skip_channels = list(reversed(enc_channels[:-1])) + [0]
        blocks, in_ch = [], enc_channels[-1]
        for skip_ch, out_ch in zip(skip_channels, decoder_channels):
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.in_ch
        f0 = self.backbone(x[:, :c])
        f1 = self.backbone(x[:, c : 2 * c])
        diffs = [(a - b).abs() for a, b in zip(f0, f1)]      # per-level change signal

        skips = list(reversed(diffs[:-1])) + [None]
        out = diffs[-1]
        for block, skip in zip(self.decoder, skips):
            out = block(out, skip)
        out = self.head(self.dropout(out))
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


@MODELS.register("temporal_classifier")
class TemporalPoolingClassifier(nn.Module):
    """Per-frame 2D backbone + temporal pooling over a ``(B, C, T, H, W)`` clip."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        pool: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if getattr(backbone, "features_only", False):
            raise ValueError("TemporalPoolingClassifier requires pooled backbone features")
        if pool not in ("mean", "max", "attention"):
            raise ValueError(f"pool must be mean|max|attention, got {pool!r}")
        self.backbone = backbone
        self.pool = pool
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if pool == "attention":
            self.attn = nn.Linear(backbone.feature_dim, 1)
        self.head = nn.Linear(backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feats = self.backbone(frames).reshape(b, t, -1)      # (B, T, F)
        if self.pool == "mean":
            pooled = feats.mean(dim=1)
        elif self.pool == "max":
            pooled = feats.max(dim=1).values
        else:
            weights = torch.softmax(self.attn(feats), dim=1)
            pooled = (feats * weights).sum(dim=1)
        return self.head(self.dropout(pooled))
