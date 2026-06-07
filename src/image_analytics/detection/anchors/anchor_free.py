"""Anchor-free target assignment (FCOS-style, Tian 2019).

Instead of anchor boxes, every feature-map location regresses distances to
the four box sides (l, t, r, b). Assignment rules:

1. A location is a candidate for a GT box if it falls inside the box AND
   within ``center_radius * stride`` of the box center (center sampling).
2. Each pyramid level handles a size band: a location only matches a GT if
   ``max(l,t,r,b)`` lies inside the level's regression range — small objects
   to fine levels, large objects to coarse levels.
3. Ambiguity (location inside several GTs) resolves to the smallest-area GT.

Centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b)) downweights
low-quality predictions far from object centers.
"""

from __future__ import annotations

import torch

INF = 1e8

BACKGROUND = -1


def pyramid_locations(
    feature_shapes: list[tuple[int, int]],
    strides: list[int],
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Per-level (H*W, 2) location centers (x, y) in image coordinates."""
    locations = []
    for (h, w), stride in zip(feature_shapes, strides):
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        locations.append(torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1))
    return locations


def assign_fcos_targets(
    locations: torch.Tensor,        # (M, 2) all levels concatenated
    strides_per_loc: torch.Tensor,  # (M,)
    ranges_per_loc: torch.Tensor,   # (M, 2) regression range per location
    gt_boxes: torch.Tensor,         # (N, 4) XYXY
    gt_labels: torch.Tensor,        # (N,)
    center_radius: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (labels (M,), reg_targets (M, 4) raw pixel distances).

    ``labels`` is the matched GT class or ``BACKGROUND`` (-1).
    """
    num_locations = locations.shape[0]
    if len(gt_boxes) == 0:
        return (
            torch.full((num_locations,), BACKGROUND, dtype=torch.int64,
                       device=locations.device),
            torch.zeros(num_locations, 4, device=locations.device),
        )

    xs = locations[:, 0].unsqueeze(1)  # (M, 1)
    ys = locations[:, 1].unsqueeze(1)

    left = xs - gt_boxes[:, 0].unsqueeze(0)     # (M, N)
    top = ys - gt_boxes[:, 1].unsqueeze(0)
    right = gt_boxes[:, 2].unsqueeze(0) - xs
    bottom = gt_boxes[:, 3].unsqueeze(0) - ys
    reg = torch.stack([left, top, right, bottom], dim=2)  # (M, N, 4)

    inside_box = reg.min(dim=2).values > 0

    centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    radius = center_radius * strides_per_loc.unsqueeze(1)  # (M, 1)
    inside_center = ((xs - centers_x.unsqueeze(0)).abs() <= radius) & (
        (ys - centers_y.unsqueeze(0)).abs() <= radius
    )

    max_reg = reg.max(dim=2).values
    in_range = (max_reg >= ranges_per_loc[:, 0].unsqueeze(1)) & (
        max_reg <= ranges_per_loc[:, 1].unsqueeze(1)
    )

    candidate = inside_box & inside_center & in_range

    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    areas = areas.unsqueeze(0).expand(num_locations, -1).clone()
    areas[~candidate] = INF
    min_area, matched = areas.min(dim=1)

    labels = gt_labels[matched].clone()
    labels[min_area == INF] = BACKGROUND
    reg_targets = reg[torch.arange(num_locations, device=locations.device), matched]
    return labels, reg_targets


def centerness_from_targets(reg_targets: torch.Tensor) -> torch.Tensor:
    """Centerness in [0, 1] from (K, 4) l/t/r/b regression targets."""
    lr = reg_targets[:, [0, 2]]
    tb = reg_targets[:, [1, 3]]
    ratio = (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-7)) * (
        tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-7)
    )
    return ratio.clamp(min=0).sqrt()


def boxes_from_distances(
    locations: torch.Tensor, distances: torch.Tensor
) -> torch.Tensor:
    """Decode (K, 4) l/t/r/b distances at (K, 2) locations into XYXY boxes."""
    return torch.stack(
        [
            locations[:, 0] - distances[:, 0],
            locations[:, 1] - distances[:, 1],
            locations[:, 0] + distances[:, 2],
            locations[:, 1] + distances[:, 3],
        ],
        dim=1,
    )
