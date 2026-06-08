"""Point-cloud transforms.

Operate on a single cloud ``(N, 3+F)`` float tensor — only the XYZ columns are
transformed geometrically; any trailing features (intensity, normals) pass
through. ``build_pointcloud_transforms`` assembles the standard train/eval
pipeline (normalize to the unit sphere, plus jitter/rotation/scale for train).
"""

from __future__ import annotations

import torch

AUGMENT_POLICIES = ("none", "default", "strong")


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            points = t(points)
        return points


class NormalizePoints:
    """Center at the centroid and scale to the unit sphere (XYZ only)."""

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        points = points.clone()
        xyz = points[:, :3]
        xyz = xyz - xyz.mean(dim=0, keepdim=True)
        scale = xyz.norm(dim=1).max().clamp(min=1e-6)
        points[:, :3] = xyz / scale
        return points


class RandomRotateZ:
    """Random rotation about the vertical (z) axis — a label-preserving
    augmentation for upright objects."""

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        theta = torch.rand(1) * 2 * torch.pi
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).reshape(3, 3)
        points = points.clone()
        points[:, :3] = points[:, :3] @ rot.t()
        return points


class RandomJitter:
    """Per-point Gaussian noise, clipped (XYZ only)."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05) -> None:
        self.sigma, self.clip = sigma, clip

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        noise = (torch.randn_like(points[:, :3]) * self.sigma).clamp(-self.clip, self.clip)
        points = points.clone()
        points[:, :3] = points[:, :3] + noise
        return points


class RandomScale:
    def __init__(self, low: float = 0.8, high: float = 1.25) -> None:
        self.low, self.high = low, high

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.low, self.high)
        points = points.clone()
        points[:, :3] = points[:, :3] * scale
        return points


def build_pointcloud_transforms(
    train: bool = True, augment: str = "default", normalize: bool = True
) -> Compose:
    if augment not in AUGMENT_POLICIES:
        raise ValueError(f"augment must be one of {AUGMENT_POLICIES}, got {augment!r}")
    ops: list = []
    if normalize:
        ops.append(NormalizePoints())
    if train and augment != "none":
        ops.append(RandomRotateZ())
        ops.append(RandomScale())
        ops.append(RandomJitter())
        if augment == "strong":
            ops.append(RandomJitter(sigma=0.02, clip=0.08))
    return Compose(ops)
