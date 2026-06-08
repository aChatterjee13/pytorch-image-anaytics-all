"""3D bounding-box utilities.

Boxes are ``(x, y, z, dx, dy, dz, yaw)`` — center, dimensions, and heading
about the vertical (z) axis. Provides axis-aligned 3D IoU (exact, ignoring
yaw — the tested path for the axis-aligned synthetic fixture), rotated BEV IoU
(Sutherland–Hodgman polygon clipping), and full 3D IoU (BEV ∩ × z-overlap).
"""

from __future__ import annotations

import torch


def box3d_volume(boxes: torch.Tensor) -> torch.Tensor:
    return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]


def _to_minmax(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center, half = boxes[:, :3], boxes[:, 3:6] / 2
    return center - half, center + half


def axis_aligned_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Exact 3D IoU treating boxes as axis-aligned (yaw ignored) -> ``(N, M)``."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return boxes1.new_zeros(len(boxes1), len(boxes2))
    min1, max1 = _to_minmax(boxes1)
    min2, max2 = _to_minmax(boxes2)
    lo = torch.maximum(min1[:, None, :], min2[None, :, :])
    hi = torch.minimum(max1[:, None, :], max2[None, :, :])
    inter = (hi - lo).clamp(min=0).prod(dim=2)              # (N, M)
    v1 = box3d_volume(boxes1)[:, None]
    v2 = box3d_volume(boxes2)[None, :]
    return inter / (v1 + v2 - inter).clamp(min=1e-9)


def boxes_to_bev_corners(boxes: torch.Tensor) -> torch.Tensor:
    """``(N, 7)`` -> BEV corner coordinates ``(N, 4, 2)`` (CCW, with yaw)."""
    x, y = boxes[:, 0], boxes[:, 1]
    dx, dy, yaw = boxes[:, 3], boxes[:, 4], boxes[:, 6]
    unit = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], device=boxes.device
    )
    cx = unit[:, 0][None, :] * dx[:, None]                  # (N, 4)
    cy = unit[:, 1][None, :] * dy[:, None]
    cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    rx = cx * cos - cy * sin + x[:, None]
    ry = cx * sin + cy * cos + y[:, None]
    return torch.stack([rx, ry], dim=2)


def _polygon_area(poly: torch.Tensor) -> torch.Tensor:
    if len(poly) < 3:
        return poly.new_zeros(())
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * torch.abs(torch.dot(x, y.roll(-1)) - torch.dot(y, x.roll(-1)))


def _clip_polygon(subject: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Sutherland–Hodgman: clip convex ``subject`` by convex ``clip`` (CCW)."""
    output = subject
    cn = len(clip)
    for i in range(cn):
        a, b = clip[i], clip[(i + 1) % cn]
        edge = b - a
        if len(output) == 0:
            break
        inputs = output
        output = []
        for j in range(len(inputs)):
            cur, prv = inputs[j], inputs[j - 1]
            cur_in = edge[0] * (cur[1] - a[1]) - edge[1] * (cur[0] - a[0]) >= 0
            prv_in = edge[0] * (prv[1] - a[1]) - edge[1] * (prv[0] - a[0]) >= 0
            if cur_in:
                if not prv_in:
                    output.append(_line_intersect(prv, cur, a, b))
                output.append(cur)
            elif prv_in:
                output.append(_line_intersect(prv, cur, a, b))
        output = torch.stack(output) if output else subject.new_zeros(0, 2)
    return output


def _line_intersect(p1, p2, p3, p4) -> torch.Tensor:
    d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    d = d if torch.abs(d) > 1e-9 else d + 1e-9
    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / d
    return p1 + t * (p2 - p1)


def bev_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Rotated bird's-eye-view IoU -> ``(N, M)`` (looped polygon clipping)."""
    n, m = len(boxes1), len(boxes2)
    if n == 0 or m == 0:
        return boxes1.new_zeros(n, m)
    corners1 = boxes_to_bev_corners(boxes1)
    corners2 = boxes_to_bev_corners(boxes2)
    area1 = boxes1[:, 3] * boxes1[:, 4]
    area2 = boxes2[:, 3] * boxes2[:, 4]
    iou = boxes1.new_zeros(n, m)
    for i in range(n):
        for j in range(m):
            inter = _polygon_area(_clip_polygon(corners1[i], corners2[j]))
            iou[i, j] = inter / (area1[i] + area2[j] - inter).clamp(min=1e-9)
    return iou


def nms_3d(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.1,
    rotated: bool = False,
) -> torch.Tensor:
    """Greedy 3D NMS -> kept indices (by descending score). Uses axis-aligned
    IoU by default; ``rotated=True`` uses rotated BEV IoU."""
    iou_fn = bev_iou if rotated else axis_aligned_iou_3d
    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_fn(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Rotated 3D IoU = BEV intersection × z-overlap / union -> ``(N, M)``."""
    n, m = len(boxes1), len(boxes2)
    if n == 0 or m == 0:
        return boxes1.new_zeros(n, m)
    corners1 = boxes_to_bev_corners(boxes1)
    corners2 = boxes_to_bev_corners(boxes2)
    z1_lo, z1_hi = boxes1[:, 2] - boxes1[:, 5] / 2, boxes1[:, 2] + boxes1[:, 5] / 2
    z2_lo, z2_hi = boxes2[:, 2] - boxes2[:, 5] / 2, boxes2[:, 2] + boxes2[:, 5] / 2
    vol1, vol2 = box3d_volume(boxes1), box3d_volume(boxes2)
    iou = boxes1.new_zeros(n, m)
    for i in range(n):
        for j in range(m):
            bev_inter = _polygon_area(_clip_polygon(corners1[i], corners2[j]))
            z_overlap = (torch.minimum(z1_hi[i], z2_hi[j]) - torch.maximum(z1_lo[i], z2_lo[j])).clamp(min=0)
            inter = bev_inter * z_overlap
            iou[i, j] = inter / (vol1[i] + vol2[j] - inter).clamp(min=1e-9)
    return iou
