"""FCOS (Tian 2019): fully convolutional one-stage anchor-free detector.

Per feature-map location: K class logits, 4 box-side distances (in stride
units, through a per-level learnable scale + ReLU), and a centerness logit.
Losses: focal (classification), GIoU on decoded boxes weighted by centerness
targets (regression), BCE (centerness). Same external interface as RetinaNet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

from image_analytics.core.registry import MODELS
from image_analytics.detection.anchors.anchor_free import (
    BACKGROUND,
    INF,
    assign_fcos_targets,
    boxes_from_distances,
    centerness_from_targets,
    pyramid_locations,
)
from image_analytics.detection.losses import giou_loss, sigmoid_focal_loss
from image_analytics.detection.necks.fpn import FPN


class Scale(nn.Module):
    """Per-level learnable scalar for regression outputs."""

    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _conv_tower(channels: int, num_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        layers.append(nn.GroupNorm(32 if channels % 32 == 0 else 8, channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@MODELS.register("fcos")
class FCOS(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        fpn_channels: int = 256,
        num_convs: int = 4,
        regress_ranges: tuple[tuple[float, float], ...] = (
            (0, 64), (64, 128), (128, 256), (256, 512), (512, INF),
        ),
        center_radius: float = 1.5,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.6,
        topk_candidates: int = 1000,
        detections_per_img: int = 100,
        prior_prob: float = 0.01,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "FCOS requires a pyramid backbone; set backbone.features_only: true"
            )
        self.backbone = backbone
        self.num_classes = num_classes

        self.fpn = FPN(backbone.feature_channels, fpn_channels, extra_levels="p6p7")
        if len(regress_ranges) != self.fpn.num_levels:
            raise ValueError(
                f"Need one regress_range per pyramid level ({self.fpn.num_levels}), "
                f"got {len(regress_ranges)}"
            )
        self.regress_ranges = tuple((float(lo), float(hi)) for lo, hi in regress_ranges)

        self.cls_tower = _conv_tower(fpn_channels, num_convs)
        self.reg_tower = _conv_tower(fpn_channels, num_convs)
        self.cls_head = nn.Conv2d(fpn_channels, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(fpn_channels, 4, 3, padding=1)
        self.centerness_head = nn.Conv2d(fpn_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(Scale(1.0) for _ in range(self.fpn.num_levels))

        self.center_radius = center_radius
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topk_candidates = topk_candidates
        self.detections_per_img = detections_per_img
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.cls_head.bias, -math.log((1.0 - prior_prob) / prior_prob))

    # -- forward ---------------------------------------------------------

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        pyramid = self.fpn(self.backbone(images))

        cls_per_level, reg_per_level, ctr_per_level = [], [], []
        for level, feature in enumerate(pyramid):
            b, _, h, w = feature.shape
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)

            cls = self.cls_head(cls_feat)
            # Distances in stride units: ReLU keeps them non-negative
            reg = F.relu(self.scales[level](self.reg_head(reg_feat)))
            ctr = self.centerness_head(reg_feat)

            cls_per_level.append(
                cls.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            )
            reg_per_level.append(reg.permute(0, 2, 3, 1).reshape(b, -1, 4))
            ctr_per_level.append(ctr.permute(0, 2, 3, 1).reshape(b, -1))

        image_h, image_w = images.shape[-2:]
        shapes = [tuple(f.shape[-2:]) for f in pyramid]
        strides = [max(round(image_h / h), 1) for h, _ in shapes]
        locations_per_level = pyramid_locations(shapes, strides, device=images.device)

        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            return self._compute_losses(
                cls_per_level, reg_per_level, ctr_per_level,
                locations_per_level, strides, targets,
            )
        return self._inference(
            cls_per_level, reg_per_level, ctr_per_level,
            locations_per_level, strides, (image_h, image_w),
        )

    # -- training --------------------------------------------------------

    def _compute_losses(
        self,
        cls_per_level: list[torch.Tensor],
        reg_per_level: list[torch.Tensor],
        ctr_per_level: list[torch.Tensor],
        locations_per_level: list[torch.Tensor],
        strides: list[int],
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        device = cls_per_level[0].device
        locations = torch.cat(locations_per_level, dim=0)        # (M, 2)
        strides_per_loc = torch.cat(
            [
                torch.full((len(locs),), float(stride), device=device)
                for locs, stride in zip(locations_per_level, strides)
            ]
        )
        ranges_per_loc = torch.cat(
            [
                torch.tensor(rng, device=device).expand(len(locs), 2)
                for locs, rng in zip(locations_per_level, self.regress_ranges)
            ]
        )

        cls_logits = torch.cat(cls_per_level, dim=1)             # (B, M, K)
        reg_preds = torch.cat(reg_per_level, dim=1)              # (B, M, 4)
        ctr_logits = torch.cat(ctr_per_level, dim=1)             # (B, M)

        total_cls = cls_logits.sum() * 0.0
        total_reg = total_cls.clone()
        total_ctr = total_cls.clone()
        total_fg = 0
        total_ctr_weight = 0.0

        for b, target in enumerate(targets):
            gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32).to(device)
            gt_labels = target["labels"].to(device)

            labels, reg_targets = assign_fcos_targets(
                locations, strides_per_loc, ranges_per_loc,
                gt_boxes, gt_labels, self.center_radius,
            )
            foreground = labels != BACKGROUND
            num_fg = int(foreground.sum())
            total_fg += num_fg

            cls_target = torch.zeros_like(cls_logits[b])
            if num_fg:
                cls_target[foreground, labels[foreground]] = 1.0
            total_cls = total_cls + sigmoid_focal_loss(
                cls_logits[b], cls_target,
                alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum",
            )

            if num_fg:
                fg_locations = locations[foreground]
                fg_strides = strides_per_loc[foreground].unsqueeze(1)
                pred_boxes = boxes_from_distances(
                    fg_locations, reg_preds[b][foreground] * fg_strides
                )
                target_boxes = boxes_from_distances(
                    fg_locations, reg_targets[foreground]
                )
                ctr_targets = centerness_from_targets(reg_targets[foreground])
                total_reg = total_reg + (
                    giou_loss(pred_boxes, target_boxes, reduction="none") * ctr_targets
                ).sum()
                total_ctr_weight += float(ctr_targets.sum())
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr_logits[b][foreground], ctr_targets, reduction="sum"
                )

        norm = max(total_fg, 1)
        loss_cls = total_cls / norm
        loss_reg = total_reg / max(total_ctr_weight, 1e-6)
        loss_ctr = total_ctr / norm
        return {
            "loss": loss_cls + loss_reg + loss_ctr,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_centerness": loss_ctr,
        }

    # -- inference -------------------------------------------------------

    @torch.no_grad()
    def _inference(
        self,
        cls_per_level: list[torch.Tensor],
        reg_per_level: list[torch.Tensor],
        ctr_per_level: list[torch.Tensor],
        locations_per_level: list[torch.Tensor],
        strides: list[int],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        batch_size = cls_per_level[0].shape[0]
        h, w = image_size
        results = []
        for b in range(batch_size):
            boxes_all, scores_all, labels_all = [], [], []
            for cls, reg, ctr, locations, stride in zip(
                cls_per_level, reg_per_level, ctr_per_level,
                locations_per_level, strides,
            ):
                # Geometric mean of class prob and centerness
                scores = (cls[b].sigmoid() * ctr[b].sigmoid().unsqueeze(1)).sqrt()
                scores = scores.flatten()  # (M*K,)
                keep = scores > self.score_thresh
                if not keep.any():
                    continue
                idxs = keep.nonzero(as_tuple=True)[0]
                kept_scores = scores[idxs]
                if len(idxs) > self.topk_candidates:
                    kept_scores, order = kept_scores.topk(self.topk_candidates)
                    idxs = idxs[order]
                loc_idx = torch.div(idxs, self.num_classes, rounding_mode="floor")
                labels = idxs % self.num_classes
                boxes = boxes_from_distances(
                    locations[loc_idx], reg[b][loc_idx] * stride
                )
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
                boxes_all.append(boxes)
                scores_all.append(kept_scores)
                labels_all.append(labels)

            if boxes_all:
                boxes = torch.cat(boxes_all)
                scores = torch.cat(scores_all)
                labels = torch.cat(labels_all)
                keep = tvops.batched_nms(boxes, scores, labels, self.nms_thresh)
                keep = keep[: self.detections_per_img]
                results.append(
                    {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
                )
            else:
                device = cls_per_level[0].device
                results.append(
                    {
                        "boxes": torch.empty(0, 4, device=device),
                        "scores": torch.empty(0, device=device),
                        "labels": torch.empty(0, dtype=torch.int64, device=device),
                    }
                )
        return results
