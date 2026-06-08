"""Faster R-CNN (Ren 2015) with FPN: RPN proposals -> RoIAlign -> box head.

Two-stage pipeline:
  1. RPN generates class-agnostic proposals over P2-P6.
  2. Each proposal is pooled from its FPN level (k = 4 + log2(sqrt(area)/224))
     via ``torchvision.ops.roi_align``, classified (softmax over background +
     K classes) and refined (class-specific box deltas, (10,10,5,5) weights).

External interface identical to the one-stage detectors; dataset labels stay
0-based foreground (the background class exists only internally as 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

from image_analytics.core.registry import MODELS
from image_analytics.detection.anchors.matcher import (
    BalancedPositiveNegativeSampler,
    Matcher,
)
from image_analytics.detection.anchors.rpn import RegionProposalNetwork
from image_analytics.detection.box_coder import BoxCoder
from image_analytics.detection.losses import smooth_l1_loss
from image_analytics.detection.necks.fpn import FPN


def assign_boxes_to_fpn_levels(
    boxes: torch.Tensor, num_levels: int, canonical_level: int = 4,
    canonical_size: float = 224.0, min_level: int = 2,
) -> torch.Tensor:
    """FPN paper level assignment: k = floor(k0 + log2(sqrt(area)/224)).

    Returns 0-based indices into the RoI feature list (P2 -> 0).
    """
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=1e-6)
    k = torch.floor(canonical_level + torch.log2(areas.sqrt() / canonical_size))
    return (k - min_level).clamp(0, num_levels - 1).long()


class TwoMLPHead(nn.Module):
    """Flatten pooled RoI features -> two FC layers."""

    def __init__(self, in_features: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


@MODELS.register("faster_rcnn")
class FasterRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,                      # foreground classes
        fpn_channels: int = 256,
        # RPN
        rpn_anchor_sizes: tuple[tuple[float, ...], ...] = (
            (32,), (64,), (128,), (256,), (512,),
        ),
        rpn_pre_nms_topk: tuple[int, int] = (2000, 1000),
        rpn_post_nms_topk: tuple[int, int] = (2000, 1000),
        rpn_nms_thresh: float = 0.7,
        # RoI head
        roi_output_size: int = 7,
        roi_sampling_ratio: int = 2,
        box_head_dim: int = 1024,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "FasterRCNN requires a pyramid backbone; set "
                "backbone.features_only: true (out_indices for C2-C5)"
            )
        self.backbone = backbone
        self.num_classes = num_classes

        # P2-P5 + pooled P6 (P6 feeds the RPN only)
        self.fpn = FPN(backbone.feature_channels, fpn_channels, extra_levels="pool")
        if len(rpn_anchor_sizes) != self.fpn.num_levels:
            raise ValueError(
                f"Need one rpn_anchor_sizes entry per pyramid level "
                f"({self.fpn.num_levels}), got {len(rpn_anchor_sizes)}"
            )
        self.rpn = RegionProposalNetwork(
            fpn_channels,
            anchor_sizes=rpn_anchor_sizes,
            pre_nms_topk=rpn_pre_nms_topk,
            post_nms_topk=rpn_post_nms_topk,
            nms_thresh=rpn_nms_thresh,
        )

        self.roi_output_size = roi_output_size
        self.roi_sampling_ratio = roi_sampling_ratio
        self.box_head = TwoMLPHead(
            fpn_channels * roi_output_size**2, box_head_dim
        )
        self.cls_predictor = nn.Linear(box_head_dim, num_classes + 1)
        self.reg_predictor = nn.Linear(box_head_dim, num_classes * 4)
        nn.init.normal_(self.cls_predictor.weight, std=0.01)
        nn.init.normal_(self.reg_predictor.weight, std=0.001)
        nn.init.zeros_(self.cls_predictor.bias)
        nn.init.zeros_(self.reg_predictor.bias)

        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.box_matcher = Matcher(
            box_fg_iou_thresh, box_bg_iou_thresh, allow_low_quality_matches=False
        )
        self.box_sampler = BalancedPositiveNegativeSampler(
            box_batch_size_per_image, box_positive_fraction
        )
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    # -- shared ------------------------------------------------------------

    def _pool_rois(
        self, features: list[torch.Tensor], proposals: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Multiscale RoIAlign: route each proposal to its FPN level."""
        device = features[0].device
        # (K, 5) rois with batch index
        rois = torch.cat(
            [
                torch.cat(
                    [torch.full((len(p), 1), b, dtype=p.dtype, device=device), p],
                    dim=1,
                )
                for b, p in enumerate(proposals)
            ]
        )
        levels = assign_boxes_to_fpn_levels(rois[:, 1:], num_levels=len(features))

        h_img = image_size[0]
        output = rois.new_zeros(
            len(rois), features[0].shape[1], self.roi_output_size, self.roi_output_size
        )
        for level, feature in enumerate(features):
            idx = torch.where(levels == level)[0]
            if not len(idx):
                continue
            scale = feature.shape[-2] / h_img
            output[idx] = tvops.roi_align(
                feature,
                rois[idx],
                output_size=self.roi_output_size,
                spatial_scale=scale,
                sampling_ratio=self.roi_sampling_ratio,
                aligned=True,
            )
        return output

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        image_size = tuple(images.shape[-2:])
        pyramid = self.fpn(self.backbone(images))
        roi_features = pyramid[:-1]  # P2-P5; P6 is RPN-only

        proposals, rpn_losses = self.rpn(pyramid, image_size, targets)

        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            return self._forward_train(
                roi_features, proposals, targets, image_size, rpn_losses
            )
        return self._forward_eval(roi_features, proposals, image_size)

    # -- training ------------------------------------------------------------

    def _select_training_samples(
        self, proposals: list[torch.Tensor], targets: list[dict]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Match proposals to GT and subsample per image.

        Returns ``(sampled_proposals, labels, reg_targets, matched_gt)`` where
        ``labels`` are internal (0 = background, foreground = dataset label + 1)
        and ``matched_gt`` is the GT index per sample (>= 0 for positives, -1
        otherwise) — the mask branch (Mask R-CNN) reuses it to fetch GT masks.
        """
        sampled_proposals, labels, reg_targets, matched_gt = [], [], [], []
        for b, target in enumerate(targets):
            gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
            gt_labels = target["labels"]

            # Standard trick: append GT to proposals so positives exist early
            props = torch.cat([proposals[b], gt_boxes]) if len(gt_boxes) else proposals[b]

            if len(gt_boxes) == 0:
                matches = torch.full(
                    (len(props),), Matcher.BACKGROUND,
                    dtype=torch.int64, device=props.device,
                )
            else:
                matches = self.box_matcher(tvops.box_iou(gt_boxes, props))

            pos_idx, neg_idx = self.box_sampler(matches)
            keep = torch.cat([pos_idx, neg_idx])

            sample_labels = torch.zeros(len(keep), dtype=torch.int64, device=props.device)
            # internal labels: 0 = background, foreground = dataset label + 1
            sample_labels[: len(pos_idx)] = gt_labels[matches[pos_idx]] + 1

            sample_reg = torch.zeros(len(keep), 4, device=props.device)
            if len(pos_idx):
                sample_reg[: len(pos_idx)] = self.box_coder.encode(
                    gt_boxes[matches[pos_idx]], props[pos_idx]
                )

            sample_gt = torch.full((len(keep),), -1, dtype=torch.int64, device=props.device)
            sample_gt[: len(pos_idx)] = matches[pos_idx]

            sampled_proposals.append(props[keep])
            labels.append(sample_labels)
            reg_targets.append(sample_reg)
            matched_gt.append(sample_gt)
        return sampled_proposals, labels, reg_targets, matched_gt

    def _box_head_losses(
        self,
        roi_features: list[torch.Tensor],
        sampled_proposals: list[torch.Tensor],
        labels: list[torch.Tensor],
        reg_targets: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        pooled = self._pool_rois(roi_features, sampled_proposals, image_size)
        box_features = self.box_head(pooled)
        cls_logits = self.cls_predictor(box_features)               # (K, C+1)
        reg_deltas = self.reg_predictor(box_features)               # (K, C*4)

        labels_cat = torch.cat(labels)
        reg_targets_cat = torch.cat(reg_targets)

        loss_cls = F.cross_entropy(cls_logits, labels_cat)

        fg = torch.where(labels_cat > 0)[0]
        if len(fg):
            fg_class = labels_cat[fg] - 1
            reg_fg = reg_deltas.view(-1, self.num_classes, 4)[fg, fg_class]
            loss_reg = smooth_l1_loss(
                reg_fg, reg_targets_cat[fg], beta=1.0 / 9, reduction="sum"
            ) / labels_cat.numel()
        else:
            loss_reg = reg_deltas.sum() * 0.0

        return {"loss_cls": loss_cls, "loss_reg": loss_reg}

    def _forward_train(
        self,
        roi_features: list[torch.Tensor],
        proposals: list[torch.Tensor],
        targets: list[dict],
        image_size: tuple[int, int],
        rpn_losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        sampled_proposals, labels, reg_targets, _matched_gt = (
            self._select_training_samples(proposals, targets)
        )
        box_losses = self._box_head_losses(
            roi_features, sampled_proposals, labels, reg_targets, image_size
        )
        losses = {**rpn_losses, **box_losses}
        losses["loss"] = sum(losses.values())
        return losses

    # -- inference -------------------------------------------------------------

    @torch.no_grad()
    def _forward_eval(
        self,
        roi_features: list[torch.Tensor],
        proposals: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        h_img, w_img = image_size
        device = roi_features[0].device
        empty = {
            "boxes": torch.empty(0, 4, device=device),
            "scores": torch.empty(0, device=device),
            "labels": torch.empty(0, dtype=torch.int64, device=device),
        }
        if sum(len(p) for p in proposals) == 0:
            return [dict(empty) for _ in proposals]

        pooled = self._pool_rois(roi_features, proposals, image_size)
        box_features = self.box_head(pooled)
        cls_scores = F.softmax(self.cls_predictor(box_features), dim=1)  # (K, C+1)
        reg_deltas = self.reg_predictor(box_features).view(-1, self.num_classes, 4)

        results = []
        offset = 0
        for props in proposals:
            n = len(props)
            if n == 0:
                results.append(dict(empty))
                continue
            scores = cls_scores[offset : offset + n, 1:]            # drop background
            deltas = reg_deltas[offset : offset + n]                # (n, C, 4)
            offset += n

            # Class-specific decode, vectorized over (n*C)
            boxes = self.box_coder.decode(
                deltas.reshape(-1, 4),
                props.unsqueeze(1).expand(-1, self.num_classes, -1).reshape(-1, 4),
            )
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w_img)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h_img)

            scores_flat = scores.reshape(-1)
            labels_flat = (
                torch.arange(self.num_classes, device=device)
                .repeat(n)
            )
            keep = scores_flat > self.score_thresh
            boxes, scores_flat, labels_flat = (
                boxes[keep], scores_flat[keep], labels_flat[keep],
            )
            keep = tvops.batched_nms(boxes, scores_flat, labels_flat, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            results.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores_flat[keep],
                    "labels": labels_flat[keep],
                }
            )
        return results
