"""RetinaNet (Lin 2017): one-stage detector with FPN and focal loss.

Architecture: pyramid backbone (C3-C5) -> FPN with P6/P7 -> shared conv
towers -> per-anchor classification (sigmoid, K foreground classes) and box
regression. Focal loss handles the extreme foreground/background imbalance
that one-stage detectors face.

Interface (shared by all detectors in this package):
    train mode:  model(images, targets) -> {"loss", "loss_cls", "loss_reg"}
    eval mode:   model(images) -> list[{"boxes", "scores", "labels"}]
Labels are 0-based foreground classes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.ops as tvops

from image_analytics.core.registry import MODELS, NECKS
from image_analytics.detection.anchors import AnchorGenerator, Matcher
from image_analytics.detection.anchors.generator import retinanet_sizes
from image_analytics.detection.box_coder import BoxCoder
from image_analytics.detection.losses import sigmoid_focal_loss, smooth_l1_loss
from image_analytics.detection.necks import fpn as _fpn  # noqa: F401  (register fpn)
from image_analytics.detection.necks import pafpn as _pafpn  # noqa: F401  (register pafpn)


def _conv_tower(channels: int, num_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@MODELS.register("retinanet")
class RetinaNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        fpn_channels: int = 256,
        neck: str = "fpn",                 # "fpn" | "pafpn" (NECKS registry key)
        num_convs: int = 4,
        anchor_base_sizes: tuple[int, ...] = (32, 64, 128, 256, 512),
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
        high_threshold: float = 0.5,
        low_threshold: float = 0.4,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk_candidates: int = 1000,
        detections_per_img: int = 100,
        prior_prob: float = 0.01,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0 / 9,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "RetinaNet requires a pyramid backbone; set "
                "backbone.features_only: true (with out_indices for C3-C5)"
            )
        self.backbone = backbone
        self.num_classes = num_classes

        # FPN (or PAFPN) over C3-C5 plus P6/P7 -> 5 pyramid levels
        self.fpn = NECKS.build(
            neck,
            in_channels_list=backbone.feature_channels,
            out_channels=fpn_channels,
            extra_levels="p6p7",
        )
        num_levels = self.fpn.num_levels
        sizes = retinanet_sizes(tuple(anchor_base_sizes))
        if len(sizes) != num_levels:
            raise ValueError(
                f"Need one anchor base size per pyramid level "
                f"({num_levels}), got {len(anchor_base_sizes)}"
            )
        self.anchor_generator = AnchorGenerator(sizes, aspect_ratios)
        num_anchors = self.anchor_generator.num_anchors_per_location

        self.cls_tower = _conv_tower(fpn_channels, num_convs)
        self.reg_tower = _conv_tower(fpn_channels, num_convs)
        self.cls_head = nn.Conv2d(fpn_channels, num_anchors * num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(fpn_channels, num_anchors * 4, 3, padding=1)

        self.box_coder = BoxCoder()
        self.matcher = Matcher(high_threshold, low_threshold, allow_low_quality_matches=True)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topk_candidates = topk_candidates
        self.detections_per_img = detections_per_img
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth_l1_beta = smooth_l1_beta

        for module in [*self.cls_tower, *self.reg_tower, self.cls_head, self.reg_head]:
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        # Prior so that initial foreground probability ~ prior_prob — keeps
        # the focal loss from being swamped by background at step 0.
        nn.init.constant_(self.cls_head.bias, -math.log((1.0 - prior_prob) / prior_prob))

    # -- forward ---------------------------------------------------------

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        pyramid = self.fpn(self.backbone(images))

        cls_per_level, reg_per_level = [], []
        for feature in pyramid:
            b, _, h, w = feature.shape
            cls = self.cls_head(self.cls_tower(feature))
            reg = self.reg_head(self.reg_tower(feature))
            # (B, A*K, H, W) -> (B, H*W*A, K): grid-major to match anchors
            cls = (
                cls.view(b, -1, self.num_classes, h, w)
                .permute(0, 3, 4, 1, 2)
                .reshape(b, -1, self.num_classes)
            )
            reg = reg.view(b, -1, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
            cls_per_level.append(cls)
            reg_per_level.append(reg)

        image_h, image_w = images.shape[-2:]
        shapes = [tuple(f.shape[-2:]) for f in pyramid]
        strides = [max(round(image_h / h), 1) for h, _ in shapes]
        anchors_per_level = self.anchor_generator(shapes, strides, device=images.device)

        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            return self._compute_losses(
                torch.cat(cls_per_level, dim=1),
                torch.cat(reg_per_level, dim=1),
                torch.cat(anchors_per_level, dim=0),
                targets,
            )
        return self._inference(
            cls_per_level, reg_per_level, anchors_per_level, (image_h, image_w)
        )

    # -- training --------------------------------------------------------

    def _compute_losses(
        self,
        cls_logits: torch.Tensor,   # (B, M, K)
        reg_deltas: torch.Tensor,   # (B, M, 4)
        anchors: torch.Tensor,      # (M, 4)
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        total_cls = cls_logits.sum() * 0.0  # keeps graph + dtype/device
        total_reg = total_cls.clone()
        total_fg = 0

        for b, target in enumerate(targets):
            gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
            gt_labels = target["labels"]

            if len(gt_boxes) == 0:
                matches = torch.full(
                    (anchors.shape[0],), Matcher.BACKGROUND,
                    dtype=torch.int64, device=anchors.device,
                )
            else:
                iou = tvops.box_iou(gt_boxes, anchors)
                matches = self.matcher(iou)

            foreground = matches >= 0
            valid = matches != Matcher.IGNORE
            num_fg = int(foreground.sum())
            total_fg += num_fg

            cls_target = torch.zeros_like(cls_logits[b])
            if num_fg:
                cls_target[foreground, gt_labels[matches[foreground]]] = 1.0
            total_cls = total_cls + sigmoid_focal_loss(
                cls_logits[b][valid],
                cls_target[valid],
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )

            if num_fg:
                reg_target = self.box_coder.encode(
                    gt_boxes[matches[foreground]], anchors[foreground]
                )
                total_reg = total_reg + smooth_l1_loss(
                    reg_deltas[b][foreground],
                    reg_target,
                    beta=self.smooth_l1_beta,
                    reduction="sum",
                )

        norm = max(total_fg, 1)
        loss_cls = total_cls / norm
        loss_reg = total_reg / norm
        return {"loss": loss_cls + loss_reg, "loss_cls": loss_cls, "loss_reg": loss_reg}

    # -- inference -------------------------------------------------------

    @torch.no_grad()
    def _inference(
        self,
        cls_per_level: list[torch.Tensor],
        reg_per_level: list[torch.Tensor],
        anchors_per_level: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        batch_size = cls_per_level[0].shape[0]
        h, w = image_size
        results = []
        for b in range(batch_size):
            boxes_all, scores_all, labels_all = [], [], []
            for cls, reg, anchors in zip(cls_per_level, reg_per_level, anchors_per_level):
                scores = cls[b].sigmoid().flatten()  # (M*K,)
                keep = scores > self.score_thresh
                if not keep.any():
                    continue
                idxs = keep.nonzero(as_tuple=True)[0]
                kept_scores = scores[idxs]
                if len(idxs) > self.topk_candidates:
                    kept_scores, order = kept_scores.topk(self.topk_candidates)
                    idxs = idxs[order]
                anchor_idx = torch.div(idxs, self.num_classes, rounding_mode="floor")
                labels = idxs % self.num_classes
                boxes = self.box_coder.decode(reg[b][anchor_idx], anchors[anchor_idx])
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
