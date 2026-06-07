"""Multi-label classification: same architecture, BCEWithLogitsLoss-based.

Targets are float vectors of shape (B, num_labels) with values in {0, 1}
(see ``data/datasets/standard.py::MultiLabelImageDataset``).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from image_analytics.classification.models import ImageClassifier
from image_analytics.core.registry import LOSSES, MODELS


@MODELS.register("multilabel_classifier")
class MultiLabelImageClassifier(ImageClassifier):
    """Per-label binary classification over shared backbone features."""

    is_multilabel = True

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Per-label probabilities (sigmoid)."""
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict_labels(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary label predictions at ``threshold``."""
        return (self.predict(x) >= threshold).long()


@LOSSES.register("bce")
def build_multilabel_criterion(
    pos_weight: Sequence[float] | None = None,
) -> nn.BCEWithLogitsLoss:
    """Per-label binary cross-entropy; ``pos_weight`` counteracts label
    imbalance (typically num_negatives / num_positives per label)."""
    weight = (
        torch.as_tensor(pos_weight, dtype=torch.float32)
        if pos_weight is not None
        else None
    )
    return nn.BCEWithLogitsLoss(pos_weight=weight)
