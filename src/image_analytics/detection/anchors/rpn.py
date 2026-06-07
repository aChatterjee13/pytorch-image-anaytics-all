"""Region Proposal Network (Ren 2015): class-agnostic objectness + box deltas
over anchors, producing proposals for the second stage.

Per FPN level: shared 3x3 conv -> 1x1 objectness (A) and 1x1 deltas (A*4).
Proposals are decoded, clipped, NMS-ed across levels, and capped. During
training, anchors are matched at (0.7 / 0.3) with low-quality forcing and
sampled 256 per image at 50% positives for the objectness/regression losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

from image_analytics.detection.anchors.generator import AnchorGenerator
from image_analytics.detection.anchors.matcher import (
    BalancedPositiveNegativeSampler,
    Matcher,
)
from image_analytics.detection.box_coder import BoxCoder
from image_analytics.detection.losses import smooth_l1_loss


class RPNHead(nn.Module):
    """Shared conv head: objectness + box deltas per anchor."""

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, 1)
        self.deltas = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for module in (self.conv, self.objectness, self.deltas):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, features: list[torch.Tensor]):
        objectness, deltas = [], []
        for feature in features:
            b, _, h, w = feature.shape
            x = F.relu(self.conv(feature))
            # grid-major reshape, matching AnchorGenerator layout
            obj = (
                self.objectness(x).view(b, -1, 1, h, w)
                .permute(0, 3, 4, 1, 2).reshape(b, -1)
            )
            reg = (
                self.deltas(x).view(b, -1, 4, h, w)
                .permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
            )
            objectness.append(obj)
            deltas.append(reg)
        return objectness, deltas


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        anchor_sizes: tuple[tuple[float, ...], ...] = (
            (32,), (64,), (128,), (256,), (512,),
        ),
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
        pre_nms_topk: tuple[int, int] = (2000, 1000),    # (train, eval) per level
        post_nms_topk: tuple[int, int] = (2000, 1000),   # (train, eval) per image
        nms_thresh: float = 0.7,
        min_size: float = 1.0,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location)
        self.box_coder = BoxCoder()
        self.matcher = Matcher(high_threshold, low_threshold, allow_low_quality_matches=True)
        self.sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_size = min_size

    def _select_topk(self, count: int) -> int:
        return self.pre_nms_topk[0] if self.training else self.pre_nms_topk[1]

    def forward(
        self,
        features: list[torch.Tensor],
        image_size: tuple[int, int],
        targets: list[dict] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        """Return (proposals per image, losses)."""
        objectness_per_level, deltas_per_level = self.head(features)

        h_img, w_img = image_size
        shapes = [tuple(f.shape[-2:]) for f in features]
        strides = [max(round(h_img / h), 1) for h, _ in shapes]
        anchors_per_level = self.anchor_generator(
            shapes, strides, device=features[0].device
        )

        proposals = self._generate_proposals(
            objectness_per_level, deltas_per_level, anchors_per_level, image_size
        )

        losses: dict[str, torch.Tensor] = {}
        if self.training:
            if targets is None:
                raise ValueError("RPN requires targets in training mode")
            losses = self._compute_losses(
                torch.cat(objectness_per_level, dim=1),
                torch.cat(deltas_per_level, dim=1),
                torch.cat(anchors_per_level, dim=0),
                targets,
            )
        return proposals, losses

    # -- proposal generation ----------------------------------------------

    @torch.no_grad()
    def _generate_proposals(
        self,
        objectness_per_level: list[torch.Tensor],
        deltas_per_level: list[torch.Tensor],
        anchors_per_level: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        batch_size = objectness_per_level[0].shape[0]
        h_img, w_img = image_size
        post_nms = self.post_nms_topk[0] if self.training else self.post_nms_topk[1]

        proposals = []
        for b in range(batch_size):
            boxes_all, scores_all, level_ids = [], [], []
            for level, (obj, deltas, anchors) in enumerate(
                zip(objectness_per_level, deltas_per_level, anchors_per_level)
            ):
                scores = obj[b].sigmoid()
                topk = min(self._select_topk(len(scores)), len(scores))
                scores, order = scores.topk(topk)
                boxes = self.box_coder.decode(deltas[b][order], anchors[order])
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w_img)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h_img)
                keep = tvops.remove_small_boxes(boxes, self.min_size)
                boxes_all.append(boxes[keep])
                scores_all.append(scores[keep])
                level_ids.append(
                    torch.full((len(keep),), level, dtype=torch.int64, device=boxes.device)
                )

            boxes = torch.cat(boxes_all)
            scores = torch.cat(scores_all)
            levels = torch.cat(level_ids)
            # NMS within each level (batched by level id), then global top-k
            keep = tvops.batched_nms(boxes, scores, levels, self.nms_thresh)
            keep = keep[:post_nms]
            proposals.append(boxes[keep])
        return proposals

    # -- training -----------------------------------------------------------

    def _compute_losses(
        self,
        objectness: torch.Tensor,   # (B, M)
        deltas: torch.Tensor,       # (B, M, 4)
        anchors: torch.Tensor,      # (M, 4)
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        total_obj = objectness.sum() * 0.0
        total_reg = total_obj.clone()
        total_samples = 0

        for b, target in enumerate(targets):
            gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)

            if len(gt_boxes) == 0:
                matches = torch.full(
                    (anchors.shape[0],), Matcher.BACKGROUND,
                    dtype=torch.int64, device=anchors.device,
                )
            else:
                matches = self.matcher(tvops.box_iou(gt_boxes, anchors))

            pos_idx, neg_idx = self.sampler(matches)
            sampled = torch.cat([pos_idx, neg_idx])
            sample_targets = torch.zeros(len(sampled), device=objectness.device)
            sample_targets[: len(pos_idx)] = 1.0

            total_obj = total_obj + F.binary_cross_entropy_with_logits(
                objectness[b][sampled], sample_targets, reduction="sum"
            )
            if len(pos_idx):
                reg_target = self.box_coder.encode(
                    gt_boxes[matches[pos_idx]], anchors[pos_idx]
                )
                total_reg = total_reg + smooth_l1_loss(
                    deltas[b][pos_idx], reg_target, beta=1.0 / 9, reduction="sum"
                )
            total_samples += len(sampled)

        norm = max(total_samples, 1)
        return {
            "loss_rpn_cls": total_obj / norm,
            "loss_rpn_reg": total_reg / norm,
        }
