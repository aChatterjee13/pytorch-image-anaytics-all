"""Cascade R-CNN (Cai 2018): a cascade of box heads at rising IoU thresholds.

Built on the Phase 2 :class:`FasterRCNN` — same backbone / FPN / RPN /
multiscale RoIAlign — but the single box head is replaced by ``num_stages``
heads trained at progressively stricter IoU thresholds (0.5 → 0.6 → 0.7). Each
stage classifies and regresses (class-agnostic) the proposals, then the
*regressed* boxes become the next stage's proposals, so later heads see
higher-quality, better-localised inputs. At inference the per-stage
classification scores are averaged and the final stage's boxes are kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

from image_analytics.core.registry import MODELS
from image_analytics.detection.anchors.matcher import Matcher
from image_analytics.detection.box_coder import BoxCoder
from image_analytics.detection.heads.faster_rcnn import FasterRCNN, TwoMLPHead
from image_analytics.detection.losses import smooth_l1_loss


@MODELS.register("cascade_rcnn")
class CascadeRCNN(FasterRCNN):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_stages: int = 3,
        stage_iou_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7),
        stage_reg_weights: tuple[tuple[float, float, float, float], ...] = (
            (10.0, 10.0, 5.0, 5.0),
            (20.0, 20.0, 10.0, 10.0),
            (30.0, 30.0, 15.0, 15.0),
        ),
        stage_loss_weights: tuple[float, ...] = (1.0, 0.5, 0.25),
        **frcnn_kwargs,
    ) -> None:
        if not (num_stages == len(stage_iou_thresholds) == len(stage_reg_weights) == len(stage_loss_weights)):
            raise ValueError("num_stages must match the per-stage threshold/weight tuples")

        fpn_channels = frcnn_kwargs.get("fpn_channels", 256)
        roi_output_size = frcnn_kwargs.get("roi_output_size", 7)
        box_head_dim = frcnn_kwargs.get("box_head_dim", 1024)
        super().__init__(backbone, num_classes, **frcnn_kwargs)

        # Replace the parent's single box head with per-stage heads.
        del self.box_head, self.cls_predictor, self.reg_predictor
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights

        self.box_heads = nn.ModuleList(
            TwoMLPHead(fpn_channels * roi_output_size**2, box_head_dim)
            for _ in range(num_stages)
        )
        self.cls_predictors = nn.ModuleList(
            nn.Linear(box_head_dim, num_classes + 1) for _ in range(num_stages)
        )
        # Class-agnostic box regression (Cascade R-CNN convention).
        self.reg_predictors = nn.ModuleList(
            nn.Linear(box_head_dim, 4) for _ in range(num_stages)
        )
        for cls_p, reg_p in zip(self.cls_predictors, self.reg_predictors):
            nn.init.normal_(cls_p.weight, std=0.01)
            nn.init.normal_(reg_p.weight, std=0.001)
            nn.init.zeros_(cls_p.bias)
            nn.init.zeros_(reg_p.bias)

        self.stage_box_coders = [BoxCoder(weights=w) for w in stage_reg_weights]
        self.stage_matchers = [
            Matcher(thr, thr, allow_low_quality_matches=False)
            for thr in stage_iou_thresholds
        ]

    # -- shared helpers ----------------------------------------------------

    def _stage_head(self, stage: int, roi_features, proposals, image_size):
        pooled = self._pool_rois(roi_features, proposals, image_size)
        feat = self.box_heads[stage](pooled)
        return self.cls_predictors[stage](feat), self.reg_predictors[stage](feat)

    def _refine_proposals(self, proposals_list, reg_deltas, box_coder, image_size):
        """Decode class-agnostic deltas onto each image's proposals (clipped)."""
        h_img, w_img = image_size
        refined, offset = [], 0
        for props in proposals_list:
            n = len(props)
            boxes = box_coder.decode(reg_deltas[offset : offset + n], props)
            offset += n
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w_img)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h_img)
            refined.append(boxes)
        return refined

    # -- training ----------------------------------------------------------

    def _sample_stage(self, proposals, targets, matcher, box_coder):
        sampled, labels, reg_targets = [], [], []
        for b, target in enumerate(targets):
            gt_boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
            gt_labels = target["labels"]
            props = torch.cat([proposals[b], gt_boxes]) if len(gt_boxes) else proposals[b]

            if len(gt_boxes) == 0:
                matches = torch.full(
                    (len(props),), Matcher.BACKGROUND, dtype=torch.int64, device=props.device
                )
            else:
                matches = matcher(tvops.box_iou(gt_boxes, props))

            pos_idx, neg_idx = self.box_sampler(matches)
            keep = torch.cat([pos_idx, neg_idx])
            sample_labels = torch.zeros(len(keep), dtype=torch.int64, device=props.device)
            sample_labels[: len(pos_idx)] = gt_labels[matches[pos_idx]] + 1
            sample_reg = torch.zeros(len(keep), 4, device=props.device)
            if len(pos_idx):
                sample_reg[: len(pos_idx)] = box_coder.encode(
                    gt_boxes[matches[pos_idx]], props[pos_idx]
                )
            sampled.append(props[keep])
            labels.append(sample_labels)
            reg_targets.append(sample_reg)
        return sampled, labels, reg_targets

    def _forward_train(self, roi_features, proposals, targets, image_size, rpn_losses):
        losses = dict(rpn_losses)
        stage_proposals = proposals
        for s in range(self.num_stages):
            sampled, labels, reg_targets = self._sample_stage(
                stage_proposals, targets, self.stage_matchers[s], self.stage_box_coders[s]
            )
            cls_logits, reg_deltas = self._stage_head(s, roi_features, sampled, image_size)

            labels_cat = torch.cat(labels)
            reg_targets_cat = torch.cat(reg_targets)
            loss_cls = F.cross_entropy(cls_logits, labels_cat)
            fg = torch.where(labels_cat > 0)[0]
            if len(fg):
                loss_reg = smooth_l1_loss(
                    reg_deltas[fg], reg_targets_cat[fg], beta=1.0 / 9, reduction="sum"
                ) / labels_cat.numel()
            else:
                loss_reg = reg_deltas.sum() * 0.0

            w = self.stage_loss_weights[s]
            losses[f"loss_cls_s{s}"] = w * loss_cls
            losses[f"loss_reg_s{s}"] = w * loss_reg

            if s < self.num_stages - 1:
                with torch.no_grad():
                    stage_proposals = self._refine_proposals(
                        sampled, reg_deltas.detach(), self.stage_box_coders[s], image_size
                    )
        losses["loss"] = sum(losses.values())
        return losses

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def _forward_eval(self, roi_features, proposals, image_size):
        h_img, w_img = image_size
        device = roi_features[0].device
        empty = {
            "boxes": torch.empty(0, 4, device=device),
            "scores": torch.empty(0, device=device),
            "labels": torch.empty(0, dtype=torch.int64, device=device),
        }
        if sum(len(p) for p in proposals) == 0:
            return [dict(empty) for _ in proposals]

        stage_proposals = proposals
        cls_accum = None
        for s in range(self.num_stages):
            cls_logits, reg_deltas = self._stage_head(s, roi_features, stage_proposals, image_size)
            scores = F.softmax(cls_logits, dim=1)
            cls_accum = scores if cls_accum is None else cls_accum + scores
            stage_proposals = self._refine_proposals(
                stage_proposals, reg_deltas, self.stage_box_coders[s], image_size
            )
        cls_accum = cls_accum / self.num_stages  # ensemble over stages

        results, offset = [], 0
        for props in stage_proposals:
            n = len(props)
            if n == 0:
                results.append(dict(empty))
                continue
            scores = cls_accum[offset : offset + n, 1:]  # drop background
            offset += n

            scores_flat = scores.reshape(-1)
            labels_flat = torch.arange(self.num_classes, device=device).repeat(n)
            boxes = props.unsqueeze(1).expand(-1, self.num_classes, -1).reshape(-1, 4)

            keep = scores_flat > self.score_thresh
            boxes, scores_flat, labels_flat = boxes[keep], scores_flat[keep], labels_flat[keep]
            keep = tvops.batched_nms(boxes, scores_flat, labels_flat, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            results.append(
                {"boxes": boxes[keep], "scores": scores_flat[keep], "labels": labels_flat[keep]}
            )
        return results
