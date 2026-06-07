"""Multi-channel input support for >3-band imagery (satellite, multispectral).

Three strategies (see EXPLORATION.md section 4.1):

1. Modified first conv — ``adapt_first_conv`` transplants pretrained RGB
   filters and initializes the extra channels. (timm models get this for free
   via ``in_chans``; this utility covers non-timm models.)
2. Channel-wise attention — ``ChannelAttentionInput`` learns per-band
   importance before the stem; ``MultiChannelBackbone`` composes it with any
   backbone (enable via ``backbone.channel_attention: true`` in config).
3. Band grouping with separate stems (SatMAE-style) — planned for Phase 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def adapt_first_conv(
    conv: nn.Conv2d, in_channels: int, init: str = "kaiming"
) -> nn.Conv2d:
    """Return a copy of ``conv`` accepting ``in_channels`` inputs.

    Pretrained weights are transplanted for the first ``min(3, in_channels)``
    channels; extra channels are initialized with ``kaiming`` noise or the
    ``mean`` of the RGB filters.
    """
    if init not in ("kaiming", "mean"):
        raise ValueError(f"init must be 'kaiming' or 'mean', got {init!r}")

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        copied = min(conv.in_channels, in_channels)
        new_conv.weight[:, :copied] = conv.weight[:, :copied]
        if in_channels > copied:
            if init == "mean":
                mean_filter = conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, copied:] = mean_filter.expand(
                    -1, in_channels - copied, -1, -1
                )
            else:
                nn.init.kaiming_normal_(new_conv.weight[:, copied:])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class ChannelAttentionInput(nn.Module):
    """SE-style attention over input channels: learns which spectral bands
    matter before features are extracted."""

    def __init__(self, num_channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(num_channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(self.pool(x).view(b, c))
        return x * weights.view(b, c, 1, 1)


class MultiChannelBackbone(nn.Module):
    """Compose channel-input attention with any backbone; forwards the
    backbone's feature interface (``feature_dim`` / ``feature_channels``)."""

    def __init__(self, backbone: nn.Module, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.attention = ChannelAttentionInput(in_channels, reduction=reduction)
        self.backbone = backbone
        self.in_channels = in_channels

    @property
    def feature_dim(self) -> int:
        return self.backbone.feature_dim

    @property
    def feature_channels(self) -> list[int]:
        return self.backbone.feature_channels

    @property
    def features_only(self) -> bool:
        return getattr(self.backbone, "features_only", False)

    def forward(self, x: torch.Tensor):
        return self.backbone(self.attention(x))
