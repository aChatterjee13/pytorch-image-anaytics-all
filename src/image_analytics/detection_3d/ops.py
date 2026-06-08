"""Pure-PyTorch point-cloud ops: FPS, ball query, kNN, gather.

Batched and CPU-friendly at test scale, with the same signatures CUDA kernels
(``pointnet2_ops`` etc.) expose — so a later GPU build can swap in faster
implementations without touching the models. All take ``xyz`` of shape
``(B, N, 3)``.
"""

from __future__ import annotations

import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance: ``(B, N, M)`` from ``(B, N, 3)`` and
    ``(B, M, 3)``."""
    return torch.cdist(src, dst) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``points`` (B, N, C) at ``idx`` (B, S) or (B, S, K) -> (B, S, C)
    or (B, S, K, C)."""
    batch = torch.arange(points.shape[0], device=points.device)
    view = [points.shape[0]] + [1] * (idx.dim() - 1)
    batch = batch.view(view).expand_as(idx)
    return points[batch, idx]


def farthest_point_sample(xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Iterative farthest-point sampling -> ``(B, num_samples)`` indices.

    Greedily picks the point maximising the minimum distance to the points
    already chosen (starting deterministically from index 0), yielding a
    near-uniform spatial cover.
    """
    b, n, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(b, num_samples, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch = torch.arange(b, device=device)
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)         # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)           # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def ball_query(
    radius: float, max_samples: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Group up to ``max_samples`` points within ``radius`` of each query point.

    Returns ``(B, S, max_samples)`` indices into ``xyz``; query neighbourhoods
    with fewer than ``max_samples`` points are padded by repeating the closest.
    """
    b, n, _ = xyz.shape
    s = new_xyz.shape[1]
    device = xyz.device

    sqr = square_distance(new_xyz, xyz)                       # (B, S, N)
    group_idx = torch.arange(n, device=device).view(1, 1, n).expand(b, s, n).clone()
    group_idx[sqr > radius**2] = n                           # sentinel for out-of-ball
    group_idx = group_idx.sort(dim=-1).values[:, :, :max_samples]
    # Replace padding sentinels with the nearest in-ball point (column 0).
    first = group_idx[:, :, 0:1].expand(-1, -1, max_samples)
    mask = group_idx == n
    group_idx[mask] = first[mask]
    return group_idx


def knn(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """k nearest neighbours of each query point -> ``(B, S, k)`` indices."""
    return square_distance(new_xyz, xyz).topk(k, dim=-1, largest=False).indices
