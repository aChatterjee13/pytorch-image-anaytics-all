"""Backbone registry and config-driven construction."""

from __future__ import annotations

import dataclasses

import torch.nn as nn

from image_analytics.backbones.base import TimmBackbone
from image_analytics.backbones.multichannel import MultiChannelBackbone
from image_analytics.core.config import BackboneConfig
from image_analytics.core.registry import BACKBONES


def build_backbone(config: BackboneConfig | str, **overrides) -> nn.Module:
    """Build a backbone from a :class:`BackboneConfig` or bare name.

    Resolution order: the BACKBONES registry first, then any valid timm model
    name as a fallback — so all 900+ timm architectures are usable without
    explicit registration. ``overrides`` replace BackboneConfig fields, e.g.
    ``build_backbone("resnet50", pretrained=False, in_channels=13)``.
    """
    if isinstance(config, str):
        config = BackboneConfig(name=config, **overrides)
    elif overrides:
        config = dataclasses.replace(config, **overrides)

    kwargs = dict(
        pretrained=config.pretrained,
        in_channels=config.in_channels,
        features_only=config.features_only,
        **config.kwargs,
    )

    if config.name in BACKBONES:
        backbone = BACKBONES.build(config.name, **kwargs)
    else:
        try:
            backbone = TimmBackbone(config.name, **kwargs)
        except RuntimeError as exc:
            available = ", ".join(sorted(BACKBONES.keys()))
            raise KeyError(
                f"{config.name!r} is neither a registered backbone nor a valid "
                f"timm model name. Registered: {available}"
            ) from exc

    if config.channel_attention:
        backbone = MultiChannelBackbone(backbone, config.in_channels)
    return backbone
