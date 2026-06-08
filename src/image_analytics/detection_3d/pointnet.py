"""PointNet (Qi 2017) and PointNet++ (Qi 2017) from scratch.

Both consume ``points (B, N, 3)`` and present one interface:
    classification -> logits ``(B, num_classes)``
    segmentation   -> logits ``(B, num_classes, N)``  (channels-second, like
                      image segmentation, so the base losses/evaluators reuse).

PointNet uses input + feature T-Nets (the feature one is regularised toward
orthogonality via :meth:`regularization_loss`). PointNet++ stacks Set
Abstraction layers (FPS → ball query → mini-PointNet) and, for segmentation,
Feature Propagation layers (inverse-distance interpolation + unit PointNet).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_analytics.core.registry import MODELS
from image_analytics.detection_3d.ops import (
    ball_query,
    farthest_point_sample,
    index_points,
    square_distance,
)


# ---------------------------------------------------------------------------
# PointNet
# ---------------------------------------------------------------------------


class TNet(nn.Module):
    """Predict a ``k×k`` input/feature alignment matrix (identity-initialised)."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, k * k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, k, N)
        b = x.shape[0]
        x = self.conv(x).max(dim=2).values
        return self.fc(x).view(b, self.k, self.k)


@MODELS.register("pointnet")
class PointNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        task: str = "classification",
        dropout: float = 0.3,
        reg_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if task not in ("classification", "segmentation"):
            raise ValueError(f"task must be classification|segmentation, got {task!r}")
        self.task = task
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self._reg = torch.zeros(())

        self.input_tnet = TNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.feature_tnet = TNet(64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )

        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:  # per-point: concat point feature (64) + global (1024)
            self.seg_head = nn.Sequential(
                nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.Conv1d(128, num_classes, 1),
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2)                       # (B, 3, N)
        n = x.shape[2]

        in_trans = self.input_tnet(x)
        x = torch.bmm(in_trans, x)                        # align input
        x = self.mlp1(x)                                  # (B, 64, N)

        feat_trans = self.feature_tnet(x)
        x = torch.bmm(feat_trans, x)
        point_feat = x                                   # (B, 64, N)

        # Orthogonality regularizer on the feature transform.
        eye = torch.eye(feat_trans.shape[1], device=feat_trans.device)
        self._reg = ((torch.bmm(feat_trans, feat_trans.transpose(1, 2)) - eye) ** 2).sum(
            dim=(1, 2)
        ).mean()

        x = self.mlp2(x)                                 # (B, 1024, N)
        global_feat = x.max(dim=2).values                # (B, 1024)

        if self.task == "classification":
            return self.head(global_feat)
        concat = torch.cat(
            [point_feat, global_feat.unsqueeze(2).expand(-1, -1, n)], dim=1
        )
        return self.seg_head(concat)                     # (B, num_classes, N)

    def regularization_loss(self) -> torch.Tensor:
        return self.reg_weight * self._reg


# ---------------------------------------------------------------------------
# PointNet++ (single-scale grouping)
# ---------------------------------------------------------------------------


class SetAbstraction(nn.Module):
    """FPS → ball query → mini-PointNet. ``npoint=None`` groups all points."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp) -> None:
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, last = [], in_channel + 3
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        b, n, _ = xyz.shape
        if self.npoint is None:                          # group-all
            new_xyz = torch.zeros(b, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)               # (B, 1, N, 3)
            grouped = (
                torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
                if points is not None else grouped_xyz
            )
        else:
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)             # (B, npoint, 3)
            group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
            grouped = (
                torch.cat([grouped_xyz, index_points(points, group_idx)], dim=-1)
                if points is not None else grouped_xyz
            )
        grouped = grouped.permute(0, 3, 2, 1)            # (B, C+3, nsample, npoint)
        new_points = self.mlp(grouped).max(dim=2).values  # (B, mlp[-1], npoint)
        return new_xyz, new_points.transpose(1, 2)       # (B, npoint, mlp[-1])


class FeaturePropagation(nn.Module):
    """Inverse-distance interpolation from a coarse level + unit PointNet."""

    def __init__(self, in_channel, mlp) -> None:
        super().__init__()
        layers, last = [], in_channel
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        # xyz1 (B,N,3) dense target; xyz2 (B,S,3) coarse source; points2 (B,S,C2)
        dists = square_distance(xyz1, xyz2)              # (B, N, S)
        k = min(3, xyz2.shape[1])                        # group-all levels have S=1
        dists, idx = dists.topk(k, dim=-1, largest=False)
        weight = 1.0 / (dists + 1e-8)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        interp = (index_points(points2, idx) * weight.unsqueeze(-1)).sum(dim=2)
        new_points = torch.cat([points1, interp], dim=-1) if points1 is not None else interp
        return self.mlp(new_points.transpose(1, 2)).transpose(1, 2)


@MODELS.register("pointnet2")
class PointNetPlusPlus(nn.Module):
    def __init__(
        self,
        num_classes: int,
        task: str = "classification",
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if task not in ("classification", "segmentation"):
            raise ValueError(f"task must be classification|segmentation, got {task!r}")
        self.task = task
        self.num_classes = num_classes

        self.sa1 = SetAbstraction(512, 0.2, 32, 0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256, [256, 512, 1024])

        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.fp3 = FeaturePropagation(1024 + 256, [256, 256])
            self.fp2 = FeaturePropagation(256 + 128, [256, 128])
            self.fp1 = FeaturePropagation(128, [128, 128])
            self.seg_head = nn.Sequential(
                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(128, num_classes, 1),
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        l0_xyz = points[:, :, :3]
        l1_xyz, l1_points = self.sa1(l0_xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 1, 1024)

        if self.task == "classification":
            return self.head(l3_points.squeeze(1))

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        return self.seg_head(l0_points.transpose(1, 2))  # (B, num_classes, N)
