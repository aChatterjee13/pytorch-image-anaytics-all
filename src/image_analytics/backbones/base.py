"""Generic timm-backed feature extractor.

timm natively supports ``in_chans != 3`` and adapts pretrained stem weights
(repeating/averaging RGB filters), so multi-channel support comes for free
for any timm architecture.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from image_analytics.core.registry import BACKBONES


class TimmBackbone(nn.Module):
    """Wrap any timm model as a feature extractor.

    Two modes:
      * pooled (default): ``forward`` returns (B, feature_dim) embeddings —
        what classification heads consume.
      * ``features_only=True``: ``forward`` returns a list of pyramid feature
        maps — what detection/segmentation necks consume; per-level channel
        counts are exposed via ``feature_channels``.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        features_only: bool = False,
        out_indices: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        import timm  # deferred: keep package import light

        self.model_name = model_name
        self.in_channels = in_channels
        self.features_only = features_only

        if features_only:
            extra = {"out_indices": tuple(out_indices)} if out_indices is not None else {}
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                features_only=True,
                **extra,
                **kwargs,
            )
            self.feature_channels: list[int] = self.model.feature_info.channels()
            self.feature_dim: int = self.feature_channels[-1]
        else:
            # num_classes=0 strips the classifier but keeps global pooling.
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0,
                **kwargs,
            )
            self.feature_dim = self.model.num_features
            self.feature_channels = [self.feature_dim]

    def forward(self, x: torch.Tensor):
        return self.model(x)


def register_timm_backbones(mapping: dict[str, str]) -> None:
    """Register registry-name -> timm-name factories for a model family."""
    for key, timm_name in mapping.items():

        def factory(_timm_name: str = timm_name, **kwargs) -> TimmBackbone:
            return TimmBackbone(_timm_name, **kwargs)

        factory.__name__ = key
        factory.__doc__ = f"TimmBackbone factory for timm model {timm_name!r}."
        BACKBONES.register(key)(factory)
