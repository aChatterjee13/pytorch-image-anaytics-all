"""Mask R-CNN (He 2017): Faster R-CNN + a parallel per-RoI mask branch.

The box pipeline is inherited unchanged from the Phase 2 :class:`FasterRCNN`.
The mask branch adds, on top of it:

* **Train** — RoIAlign (14x14) the *positive* sampled proposals from the
  shared FPN features, 4 conv layers + a 2x deconv to 28x28, a per-class
  1x1 conv, and a BCE loss against GT instance masks cropped to each proposal
  (``project_masks_on_boxes`` via ``roi_align`` at scale 1).
* **Eval** — re-pool the final detected boxes, predict their masks, select the
  predicted-class channel, and paste each 28x28 mask back to image resolution.

Target dicts carry ``masks`` (N, H, W) alongside ``boxes``/``labels``; eval
predictions gain a ``masks`` (D, H, W) uint8 field for mask-mAP evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

from image_analytics.core.registry import MODELS
from image_analytics.detection.heads.faster_rcnn import (
    FasterRCNN,
    assign_boxes_to_fpn_levels,
)


def project_masks_on_boxes(
    gt_masks: torch.Tensor, boxes: torch.Tensor, output_size: int
) -> torch.Tensor:
    """Crop each GT mask to its proposal box and resample to ``M x M``.

    ``gt_masks`` is (P, H, W); ``boxes`` is (P, 4) in image coordinates. Uses
    ``roi_align`` at spatial scale 1 (masks are at image resolution), the same
    construction torchvision uses for Mask R-CNN targets.
    """
    rois = torch.cat(
        [torch.arange(len(boxes), device=boxes.device, dtype=boxes.dtype)[:, None], boxes],
        dim=1,
    )
    targets = tvops.roi_align(
        gt_masks[:, None].float(), rois, output_size, spatial_scale=1.0, aligned=True
    )
    return (targets[:, 0] >= 0.5).float()


def paste_masks_in_image(
    masks: torch.Tensor, boxes: torch.Tensor, image_size: tuple[int, int],
    threshold: float = 0.5,
) -> torch.Tensor:
    """Resize each (M, M) soft mask to its box and paste into a full canvas.

    Returns (D, H, W) uint8 binary masks at image resolution.
    """
    h_img, w_img = image_size
    out = torch.zeros(len(masks), h_img, w_img, dtype=torch.uint8, device=masks.device)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.round().to(torch.int64).tolist()
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, w_img), min(y2, h_img)
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1 or x2c <= x1c or y2c <= y1c:
            continue
        resized = F.interpolate(
            masks[i][None, None], size=(bh, bw), mode="bilinear", align_corners=False
        )[0, 0]
        # Crop the resized mask to the visible (clipped) region of the box.
        resized = resized[y1c - y1 : y2c - y1, x1c - x1 : x2c - x1]
        out[i, y1c:y2c, x1c:x2c] = (resized >= threshold).to(torch.uint8)
    return out


@MODELS.register("mask_rcnn")
class MaskRCNN(FasterRCNN):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        mask_roi_output_size: int = 14,
        mask_head_dim: int = 256,
        mask_num_convs: int = 4,
        mask_thresh: float = 0.5,
        **frcnn_kwargs,
    ) -> None:
        super().__init__(backbone, num_classes, **frcnn_kwargs)
        self.mask_roi_output_size = mask_roi_output_size
        self.mask_thresh = mask_thresh

        fpn_channels = self.fpn.out_channels
        convs: list[nn.Module] = []
        in_ch = fpn_channels
        for _ in range(mask_num_convs):
            convs.append(nn.Conv2d(in_ch, mask_head_dim, 3, padding=1))
            convs.append(nn.ReLU(inplace=True))
            in_ch = mask_head_dim
        self.mask_head = nn.Sequential(*convs)
        self.mask_deconv = nn.ConvTranspose2d(mask_head_dim, mask_head_dim, 2, stride=2)
        self.mask_predictor = nn.Conv2d(mask_head_dim, num_classes, 1)

        for module in [*self.mask_head, self.mask_deconv, self.mask_predictor]:
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # -- mask head (shared by train/eval) ----------------------------------

    def _pool_mask_rois(
        self, roi_features: list[torch.Tensor], proposals: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Multiscale RoIAlign at the mask resolution (14x14)."""
        device = roi_features[0].device
        rois = torch.cat(
            [
                torch.cat(
                    [torch.full((len(p), 1), b, dtype=p.dtype, device=device), p], dim=1
                )
                for b, p in enumerate(proposals)
            ]
        )
        levels = assign_boxes_to_fpn_levels(rois[:, 1:], num_levels=len(roi_features))
        output = rois.new_zeros(
            len(rois), roi_features[0].shape[1],
            self.mask_roi_output_size, self.mask_roi_output_size,
        )
        h_img = image_size[0]
        for level, feature in enumerate(roi_features):
            idx = torch.where(levels == level)[0]
            if not len(idx):
                continue
            output[idx] = tvops.roi_align(
                feature, rois[idx], output_size=self.mask_roi_output_size,
                spatial_scale=feature.shape[-2] / h_img,
                sampling_ratio=self.roi_sampling_ratio, aligned=True,
            )
        return output

    def _predict_masks(
        self, roi_features: list[torch.Tensor], proposals: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Per-RoI per-class mask logits (P, num_classes, 2M, 2M)."""
        pooled = self._pool_mask_rois(roi_features, proposals, image_size)
        x = self.mask_head(pooled)
        x = F.relu(self.mask_deconv(x))
        return self.mask_predictor(x)

    # -- training ----------------------------------------------------------

    def _forward_train(self, roi_features, proposals, targets, image_size, rpn_losses):
        sampled_proposals, labels, reg_targets, matched_gt = (
            self._select_training_samples(proposals, targets)
        )
        box_losses = self._box_head_losses(
            roi_features, sampled_proposals, labels, reg_targets, image_size
        )
        loss_mask = self._mask_loss(
            roi_features, sampled_proposals, labels, matched_gt, targets, image_size
        )
        losses = {**rpn_losses, **box_losses, "loss_mask": loss_mask}
        losses["loss"] = sum(losses.values())
        return losses

    def _mask_loss(
        self, roi_features, sampled_proposals, labels, matched_gt, targets, image_size
    ) -> torch.Tensor:
        pos_proposals: list[torch.Tensor] = []
        pos_classes: list[torch.Tensor] = []
        pos_targets: list[torch.Tensor] = []
        out_size = self.mask_roi_output_size * 2  # after the 2x deconv

        for b in range(len(sampled_proposals)):
            fg = torch.where(labels[b] > 0)[0]
            if not len(fg):
                pos_proposals.append(sampled_proposals[b].new_zeros(0, 4))
                continue
            props = sampled_proposals[b][fg]
            gt_idx = matched_gt[b][fg]
            gt_masks = torch.as_tensor(targets[b]["masks"], device=props.device)
            pos_proposals.append(props)
            pos_classes.append(labels[b][fg] - 1)  # 0-based class
            pos_targets.append(project_masks_on_boxes(gt_masks[gt_idx], props, out_size))

        if not pos_classes:  # no positives in the whole batch
            return self.mask_predictor.weight.sum() * 0.0

        mask_logits = self._predict_masks(roi_features, pos_proposals, image_size)
        classes = torch.cat(pos_classes)
        target_masks = torch.cat(pos_targets)
        idx = torch.arange(len(classes), device=mask_logits.device)
        pred = mask_logits[idx, classes]  # (P, 2M, 2M)
        return F.binary_cross_entropy_with_logits(pred, target_masks)

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def _forward_eval(self, roi_features, proposals, image_size):
        detections = super()._forward_eval(roi_features, proposals, image_size)

        boxes_per_image = [d["boxes"] for d in detections]
        if sum(len(b) for b in boxes_per_image) == 0:
            for d in detections:
                d["masks"] = torch.zeros(0, *image_size, dtype=torch.uint8, device=d["boxes"].device)
            return detections

        mask_logits = self._predict_masks(roi_features, boxes_per_image, image_size)
        offset = 0
        for d in detections:
            n = len(d["boxes"])
            if n == 0:
                d["masks"] = torch.zeros(0, *image_size, dtype=torch.uint8, device=d["boxes"].device)
                continue
            logits = mask_logits[offset : offset + n]
            offset += n
            probs = logits[torch.arange(n), d["labels"]].sigmoid()  # (n, 2M, 2M)
            d["masks"] = paste_masks_in_image(
                probs, d["boxes"], image_size, threshold=self.mask_thresh
            )
        return detections
