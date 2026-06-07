"""Detection losses: sigmoid focal loss, GIoU/DIoU, smooth L1.

Implemented from first principles (tests assert parity with the torchvision
reference ops). All box arguments are XYXY tensors of shape (N, 4).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from image_analytics.core.registry import LOSSES

_REDUCTIONS = ("none", "mean", "sum")


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")


@LOSSES.register("sigmoid_focal")
def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Focal loss for dense detection (Lin 2017, RetinaNet).

    FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t), addressing extreme
    foreground/background imbalance. ``targets`` are float {0, 1} of the same
    shape as ``logits``.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * (1.0 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return _reduce(loss, reduction)


def _box_areas(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def paired_box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise IoU and union for paired boxes (N, 4) vs (N, 4).

    (torchvision's ``box_iou`` computes the full NxM matrix; losses need the
    paired diagonal without materializing it.)
    """
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    union = _box_areas(boxes1) + _box_areas(boxes2) - intersection
    return intersection / union.clamp(min=1e-7), union


@LOSSES.register("giou")
def giou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """Generalized IoU loss (Rezatofighi 2019): 1 - GIoU.

    GIoU = IoU - |C \\ (A u B)| / |C| where C is the smallest enclosing box;
    provides gradient signal even for non-overlapping boxes.
    """
    iou, union = paired_box_iou(pred_boxes, target_boxes)
    lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosure = (wh[:, 0] * wh[:, 1]).clamp(min=1e-7)
    giou = iou - (enclosure - union) / enclosure
    return _reduce(1.0 - giou, reduction)


@LOSSES.register("diou")
def diou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """Distance IoU loss (Zheng 2020): 1 - IoU + d^2(centers) / diagonal^2."""
    iou, _ = paired_box_iou(pred_boxes, target_boxes)
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    target_centers = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
    center_dist = (pred_centers - target_centers).pow(2).sum(dim=1)

    lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
    diagonal = (rb - lt).pow(2).sum(dim=1).clamp(min=1e-7)
    return _reduce(1.0 - iou + center_dist / diagonal, reduction)


@LOSSES.register("smooth_l1")
def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0 / 9,
    reduction: str = "none",
) -> torch.Tensor:
    """Huber-style smooth L1 (Fast R-CNN); ``beta`` is the L2->L1 transition."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)
