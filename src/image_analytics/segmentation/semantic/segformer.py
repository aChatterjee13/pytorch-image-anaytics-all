"""SegFormer (Xie 2021) via HuggingFace transformers (``[seg]`` extra).

HF's ``SegformerForSemanticSegmentation`` emits logits at 1/4 input
resolution; this wrapper upsamples them back to the input size so SegFormer
presents the same ``model(images) -> logits (B, C, H, W)`` interface as the
from-scratch models and trains through the base Trainer with an external
criterion (CrossEntropy / Dice / CE+Dice) — no special-casing needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import MODELS


def _load_transformers():
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for SegFormer. "
            "Install it with: pip install 'image-analytics[seg]'"
        ) from exc
    return transformers


@MODELS.register("segformer")
class SegFormerWrapper(nn.Module):
    """Fine-tunable SegFormer.

    Args:
        model_name: HF hub id (e.g. ``nvidia/mit-b0``) for the pretrained
            encoder; ignored when ``pretrained`` is False.
        pretrained: load hub weights (downloads) vs. random init (offline).
        in_channels: input bands; values other than 3 force random init since
            the pretrained patch-embed stem is RGB.
        config_kwargs: forwarded to ``SegformerConfig`` for random-init models
            (e.g. shrink ``hidden_sizes``/``depths`` for CPU smoke runs).
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "nvidia/mit-b0",
        pretrained: bool = True,
        in_channels: int = 3,
        backbone: nn.Module | None = None,  # registry-protocol compat; unused
        config_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        _load_transformers()
        from transformers import (
            SegformerConfig,
            SegformerForSemanticSegmentation,
        )

        if pretrained and in_channels == 3:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                num_channels=in_channels,
                **(config_kwargs or {}),
            )
            self.model = SegformerForSemanticSegmentation(config)
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=images).logits  # (B, C, H/4, W/4)
        return F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
