"""Segmentation losses: Dice, CE+Dice combo, and a multiclass focal variant.

All operate on logits ``(B, C, H, W)`` and integer targets ``(B, H, W)`` and
honour ``ignore_index`` (pixels excluded from both region overlap and the
pixel-wise terms). Registered in ``LOSSES`` so configs select them by name
(``training.loss``); ``CombinedLoss`` (CE+Dice) is the practical default.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import LOSSES


def _class_weights(weights: Sequence[float] | None) -> torch.Tensor | None:
    return None if weights is None else torch.as_tensor(weights, dtype=torch.float32)


@LOSSES.register("cross_entropy")
def cross_entropy_loss(
    ignore_index: int = 255,
    class_weights: Sequence[float] | None = None,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    """Pixel-wise cross-entropy honouring ``ignore_index`` — the seg baseline."""
    return nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        weight=_class_weights(class_weights),
        label_smoothing=label_smoothing,
    )


@LOSSES.register("dice")
class DiceLoss(nn.Module):
    """Soft (multiclass) Dice loss, averaged over classes.

    ``1 - mean_c Dice_c`` where Dice is computed from softmax probabilities and
    one-hot targets over non-ignored pixels. ``smooth`` stabilizes empty
    classes (their Dice is ~1, contributing ~0 loss).
    """

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = logits.softmax(dim=1)                      # (B, C, H, W)

        valid = (target != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
        safe_target = target.clone()
        safe_target[target == self.ignore_index] = 0
        one_hot = (
            F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
        )

        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)  # sum over batch + spatial -> per-class
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


@LOSSES.register("ce_dice")
class CombinedLoss(nn.Module):
    """Weighted sum of cross-entropy and Dice — the recommended default."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        class_weights: Sequence[float] | None = None,
        label_smoothing: float = 0.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=_class_weights(class_weights),
            label_smoothing=label_smoothing,
        )
        self.dice = DiceLoss(ignore_index=ignore_index, smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(
            logits, target
        )


@LOSSES.register("seg_focal")
class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin 2017) over softmax probabilities.

    ``FL = (1 - p_t)^gamma * CE`` — down-weights easy pixels, useful for the
    severe class imbalance of dense prediction. ``alpha`` is an optional
    per-class weight vector.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer(
            "alpha",
            _class_weights(alpha) if alpha is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W); ignored pixels are 0
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        valid = target != self.ignore_index
        denom = valid.sum().clamp(min=1)
        return loss.sum() / denom
