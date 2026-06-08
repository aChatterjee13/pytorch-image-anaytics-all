"""PointPillars (Lang 2019) from scratch — the mainstream 3D detector that
needs no sparse convolution.

Pipeline: points → **pillar feature net** (decorate each point with offsets to
its pillar's centroid/center, a per-point MLP, max-pool per pillar) → scatter
to a BEV pseudo-image → 2D conv backbone → an anchor-free per-cell head
(focal classification + smooth-L1 regression with sin/cos yaw encoding). Boxes
are axis-aligned in the synthetic regime; decode + greedy 3D NMS at inference.

Interface matches the 2D detectors:
    train:  model(points, targets) -> {"loss", ...}
    eval:   model(points)          -> [{"boxes_3d", "scores", "labels"}]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import MODELS
from image_analytics.detection_3d.box3d import nms_3d
from image_analytics.detection.losses import sigmoid_focal_loss, smooth_l1_loss


class PillarFeatureNet(nn.Module):
    """Per-point MLP over decorated pillar features, max-pooled per pillar and
    scattered to a BEV canvas ``(B, C, H, W)``."""

    def __init__(self, out_channels: int, point_cloud_range, voxel_size) -> None:
        super().__init__()
        self.pc_range = point_cloud_range
        self.vx, self.vy = voxel_size
        self.W = int(round((point_cloud_range[3] - point_cloud_range[0]) / self.vx))
        self.H = int(round((point_cloud_range[4] - point_cloud_range[1]) / self.vy))
        self.out_channels = out_channels
        # decorated features: x,y,z + (xc,yc,zc to centroid) + (xp,yp to center)
        self.linear = nn.Linear(8, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        b = points.shape[0]
        xmin, ymin = self.pc_range[0], self.pc_range[1]
        hw = self.H * self.W
        canvases = []

        for i in range(b):
            base = points.new_zeros(self.out_channels, hw)
            p = points[i]
            inside = (
                (p[:, 0] >= xmin) & (p[:, 0] < self.pc_range[3])
                & (p[:, 1] >= ymin) & (p[:, 1] < self.pc_range[4])
            )
            p = p[inside]
            if len(p) == 0:
                canvases.append(base)
                continue
            ix = ((p[:, 0] - xmin) / self.vx).long().clamp(0, self.W - 1)
            iy = ((p[:, 1] - ymin) / self.vy).long().clamp(0, self.H - 1)
            pid = iy * self.W + ix

            # per-pillar centroid (for the xc,yc,zc offsets)
            count = torch.zeros(hw, device=p.device).index_add_(
                0, pid, torch.ones(len(p), device=p.device)
            ).clamp(min=1)
            mean = torch.zeros(hw, 3, device=p.device).index_add_(0, pid, p)[pid] / count[pid, None]
            center_x = xmin + (ix.float() + 0.5) * self.vx
            center_y = ymin + (iy.float() + 0.5) * self.vy

            decorated = torch.cat(
                [p, p - mean, (p[:, 0] - center_x)[:, None], (p[:, 1] - center_y)[:, None]],
                dim=1,
            )  # (Nv, 8)
            feat = F.relu(self.bn(self.linear(decorated)))   # (Nv, C); >= 0

            # Out-of-place scatter-max so autograd sees no in-place edits.
            index = pid[None, :].expand(self.out_channels, -1)
            canvases.append(base.scatter_reduce(1, index, feat.t(), reduce="amax", include_self=True))
        return torch.stack(canvases).view(b, self.out_channels, self.H, self.W)


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


@MODELS.register("pointpillars")
class PointPillars(nn.Module):
    def __init__(
        self,
        num_classes: int,
        point_cloud_range: tuple[float, ...] = (-5.0, -5.0, -1.0, 5.0, 5.0, 3.0),
        voxel_size: tuple[float, float] = (0.25, 0.25),
        pillar_channels: int = 64,
        backbone_channels: int = 128,
        score_thresh: float = 0.2,
        nms_iou: float = 0.1,
        max_detections: int = 50,
        anchor_size: tuple[float, float, float] = (1.5, 1.5, 1.5),
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pc_range = point_cloud_range
        self.vx, self.vy = voxel_size
        self.anchor_size = anchor_size
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma

        self.pfn = PillarFeatureNet(pillar_channels, point_cloud_range, voxel_size)
        self.H, self.W = self.pfn.H, self.pfn.W
        self.backbone = nn.Sequential(
            _conv_block(pillar_channels, backbone_channels),
            _conv_block(backbone_channels, backbone_channels),
            _conv_block(backbone_channels, backbone_channels),
        )
        self.cls_head = nn.Conv2d(backbone_channels, num_classes, 1)
        self.reg_head = nn.Conv2d(backbone_channels, 8, 1)  # dx,dy,z,ldx,ldy,ldz,sin,cos
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior_prob) / prior_prob))

    # -- geometry ----------------------------------------------------------

    def _cell_centers(self, device):
        xs = self.pc_range[0] + (torch.arange(self.W, device=device) + 0.5) * self.vx
        ys = self.pc_range[1] + (torch.arange(self.H, device=device) + 0.5) * self.vy
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        return cx, cy                                        # (H, W) each

    def forward(self, points: torch.Tensor, targets: list[dict] | None = None):
        canvas = self.pfn(points)
        feat = self.backbone(canvas)
        cls_logits = self.cls_head(feat)                     # (B, K, H, W)
        reg = self.reg_head(feat)                            # (B, 8, H, W)
        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            return self._compute_losses(cls_logits, reg, targets)
        return self._inference(cls_logits, reg)

    # -- training ----------------------------------------------------------

    def _encode_targets(self, targets, device):
        """Dense per-cell classification + regression targets and a positive mask."""
        b = len(targets)
        cls_t = torch.zeros(b, self.num_classes, self.H, self.W, device=device)
        reg_t = torch.zeros(b, 8, self.H, self.W, device=device)
        pos = torch.zeros(b, self.H, self.W, dtype=torch.bool, device=device)
        for i, target in enumerate(targets):
            boxes = torch.as_tensor(target["boxes_3d"], dtype=torch.float32, device=device)
            labels = target["labels"].to(device)
            for box, label in zip(boxes, labels):
                ix = int(((box[0] - self.pc_range[0]) / self.vx).clamp(0, self.W - 1))
                iy = int(((box[1] - self.pc_range[1]) / self.vy).clamp(0, self.H - 1))
                cx = self.pc_range[0] + (ix + 0.5) * self.vx
                cy = self.pc_range[1] + (iy + 0.5) * self.vy
                cls_t[i, label, iy, ix] = 1.0
                reg_t[i, :, iy, ix] = torch.tensor([
                    (box[0] - cx) / self.vx, (box[1] - cy) / self.vy, box[2],
                    torch.log(box[3] / self.anchor_size[0]),
                    torch.log(box[4] / self.anchor_size[1]),
                    torch.log(box[5] / self.anchor_size[2]),
                    torch.sin(box[6]), torch.cos(box[6]),
                ], device=device)
                pos[i, iy, ix] = True
        return cls_t, reg_t, pos

    def _compute_losses(self, cls_logits, reg, targets):
        device = cls_logits.device
        cls_t, reg_t, pos = self._encode_targets(targets, device)
        num_pos = max(int(pos.sum()), 1)

        loss_cls = sigmoid_focal_loss(
            cls_logits, cls_t, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum"
        ) / num_pos

        if pos.any():
            reg_pred = reg.permute(0, 2, 3, 1)[pos]          # (P, 8)
            reg_tgt = reg_t.permute(0, 2, 3, 1)[pos]
            loss_reg = smooth_l1_loss(reg_pred, reg_tgt, beta=1.0 / 9, reduction="sum") / num_pos
        else:
            loss_reg = reg.sum() * 0.0

        return {"loss": loss_cls + loss_reg, "loss_cls": loss_cls, "loss_reg": loss_reg}

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def _inference(self, cls_logits, reg):
        device = cls_logits.device
        b = cls_logits.shape[0]
        cx, cy = self._cell_centers(device)
        scores_map = cls_logits.sigmoid()                    # (B, K, H, W)

        results = []
        for i in range(b):
            scores_k = scores_map[i]                          # (K, H, W)
            best_score, best_label = scores_k.max(dim=0)      # (H, W)
            keep = best_score > self.score_thresh
            if not keep.any():
                results.append({
                    "boxes_3d": torch.zeros(0, 7, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=device),
                })
                continue
            r = reg[i].permute(1, 2, 0)[keep]                 # (P, 8)
            boxes = torch.stack([
                cx[keep] + r[:, 0] * self.vx,
                cy[keep] + r[:, 1] * self.vy,
                r[:, 2],
                torch.exp(r[:, 3]) * self.anchor_size[0],
                torch.exp(r[:, 4]) * self.anchor_size[1],
                torch.exp(r[:, 5]) * self.anchor_size[2],
                torch.atan2(r[:, 6], r[:, 7]),
            ], dim=1)
            scores = best_score[keep]
            labels = best_label[keep]
            order = nms_3d(boxes, scores, self.nms_iou)[: self.max_detections]
            results.append({"boxes_3d": boxes[order], "scores": scores[order], "labels": labels[order]})
        return results
