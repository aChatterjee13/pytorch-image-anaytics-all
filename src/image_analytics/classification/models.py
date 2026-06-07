"""Classification models: backbone + linear head."""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.backbones.registry import build_backbone
from image_analytics.core.config import BackboneConfig, ModelConfig
from image_analytics.core.registry import MODELS


@MODELS.register("classifier")
class ImageClassifier(nn.Module):
    """Single-label image classifier: pooled backbone features -> dropout ->
    linear head. Trained with cross-entropy."""

    is_multilabel = False

    def __init__(
        self,
        backbone: nn.Module | BackboneConfig | str,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            backbone = build_backbone(backbone)
        if getattr(backbone, "features_only", False):
            raise ValueError(
                "ImageClassifier requires pooled backbone features; "
                "set backbone.features_only: false"
            )
        self.backbone = backbone
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(backbone.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.forward_features(x)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities (softmax)."""
        return self.forward(x).softmax(dim=1)

    def freeze_backbone(self) -> None:
        """Linear probing: train only the classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(config: ModelConfig) -> nn.Module:
    """Build a classification model from a :class:`ModelConfig`."""
    backbone = build_backbone(config.backbone)
    return MODELS.build(
        config.name,
        backbone=backbone,
        num_classes=config.num_classes,
        dropout=config.dropout,
        **config.kwargs,
    )
