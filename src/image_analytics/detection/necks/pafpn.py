"""PAFPN — Path Aggregation FPN (Liu 2018, PANet).

FPN's top-down pathway propagates strong semantics from coarse to fine levels;
PAFPN adds a second, **bottom-up** pathway so precise localisation signals from
fine levels also reach the coarse ones. It reuses :class:`FPN` for the
top-down stage, then for each level (fine→coarse) downsamples the previous
output with a stride-2 conv, adds the FPN map, and smooths with a 3×3 conv.

Same I/O contract as ``FPN`` (``list[Tensor] -> list[Tensor]`` with
``out_channels`` everywhere and the same ``num_levels``), so it is a drop-in
neck wherever an FPN is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import NECKS
from image_analytics.detection.necks.fpn import FPN


@NECKS.register("pafpn")
class PAFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        extra_levels: str | None = None,
    ) -> None:
        super().__init__()
        self.fpn = FPN(in_channels_list, out_channels, extra_levels)
        self.out_channels = out_channels
        self.extra_levels = extra_levels

        num_levels = self.fpn.num_levels
        self.downsample_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        )
        self.pa_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(num_levels - 1)
        )
        for module in (*self.downsample_convs, *self.pa_convs):
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.zeros_(module.bias)

    @property
    def num_levels(self) -> int:
        return self.fpn.num_levels

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = self.fpn(features)  # top-down, finest first

        outputs = [laterals[0]]
        for i, (down_conv, pa_conv) in enumerate(
            zip(self.downsample_convs, self.pa_convs)
        ):
            down = down_conv(outputs[-1])
            if down.shape[-2:] != laterals[i + 1].shape[-2:]:
                down = F.interpolate(down, size=laterals[i + 1].shape[-2:], mode="nearest")
            outputs.append(pa_conv(laterals[i + 1] + down))
        return outputs
