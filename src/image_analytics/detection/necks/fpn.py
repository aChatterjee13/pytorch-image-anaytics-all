"""Feature Pyramid Network (Lin 2017).

Top-down pathway with lateral connections: every backbone level is projected
to a common channel width (1x1 lateral conv), enriched with upsampled
higher-level semantics, and smoothed with a 3x3 conv. Optional extra levels:

    extra_levels="pool"   P6 = max-pool(P5)            (Faster R-CNN flavor)
    extra_levels="p6p7"   P6/P7 = strided convs on P5  (RetinaNet flavor)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import NECKS


@NECKS.register("fpn")
class FPN(nn.Module):
    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int = 256,
        extra_levels: str | None = None,
    ) -> None:
        super().__init__()
        if extra_levels not in (None, "pool", "p6p7"):
            raise ValueError(
                f"extra_levels must be None, 'pool', or 'p6p7', got {extra_levels!r}"
            )
        if any(c <= 0 for c in in_channels_list):
            raise ValueError(f"invalid in_channels_list: {in_channels_list}")

        self.in_channels_list = list(in_channels_list)
        self.out_channels = out_channels
        self.extra_levels = extra_levels

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        )
        if extra_levels == "p6p7":
            self.p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.zeros_(module.bias)

    @property
    def num_levels(self) -> int:
        extra = {"pool": 1, "p6p7": 2}.get(self.extra_levels or "", 0)
        return len(self.in_channels_list) + extra

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.lateral_convs):
            raise ValueError(
                f"FPN expects {len(self.lateral_convs)} feature maps, got {len(features)}"
            )

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down: coarsest level untouched, others accumulate upsampled context
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )

        outputs = [conv(lateral) for conv, lateral in zip(self.output_convs, laterals)]

        if self.extra_levels == "pool":
            outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))
        elif self.extra_levels == "p6p7":
            p6 = self.p6(outputs[-1])
            outputs.append(p6)
            outputs.append(self.p7(F.relu(p6)))
        return outputs
