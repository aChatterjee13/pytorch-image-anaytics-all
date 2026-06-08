"""DGCNN (Wang 2019): EdgeConv on dynamically recomputed kNN graphs.

Each EdgeConv builds a kNN graph **in the current feature space** (so the
graph adapts layer to layer), forms edge features ``[x_i, x_j - x_i]``, runs a
shared MLP, and max-aggregates over neighbours. Same interface as PointNet:
classification -> ``(B, num_classes)``, segmentation -> ``(B, num_classes, N)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import MODELS
from image_analytics.detection_3d.ops import index_points


def _edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """``(B, C, N)`` -> edge features ``(B, 2C, N, k)`` over a feature-space kNN."""
    xt = x.transpose(1, 2)                               # (B, N, C)
    idx = torch.cdist(xt, xt).topk(k, dim=-1, largest=False).indices
    neighbors = index_points(xt, idx)                    # (B, N, k, C)
    xi = xt.unsqueeze(2).expand_as(neighbors)
    edge = torch.cat([xi, neighbors - xi], dim=-1)       # (B, N, k, 2C)
    return edge.permute(0, 3, 1, 2).contiguous()         # (B, 2C, N, k)


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in, N)
        return self.conv(_edge_features(x, self.k)).max(dim=-1).values


@MODELS.register("dgcnn")
class DGCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        task: str = "classification",
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if task not in ("classification", "segmentation"):
            raise ValueError(f"task must be classification|segmentation, got {task!r}")
        self.task = task
        self.num_classes = num_classes

        self.ec1 = EdgeConv(3, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.ec4 = EdgeConv(128, 256, k)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(emb_dims * 2, 512, bias=False),
                nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.seg_head = nn.Sequential(
                nn.Conv1d(emb_dims + 512, 256, 1, bias=False),
                nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout),
                nn.Conv1d(256, 128, 1, bias=False),
                nn.BatchNorm1d(128), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(128, num_classes, 1),
            )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        n = points.shape[1]
        x = points.transpose(1, 2)                       # (B, 3, N)
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)
        multi = torch.cat([x1, x2, x3, x4], dim=1)       # (B, 512, N)
        emb = self.conv5(multi)                          # (B, emb_dims, N)

        if self.task == "classification":
            pooled = torch.cat([emb.max(dim=2).values, emb.mean(dim=2)], dim=1)
            return self.head(pooled)

        global_feat = emb.max(dim=2, keepdim=True).values.expand(-1, -1, n)
        return self.seg_head(torch.cat([multi, global_feat], dim=1))
