"""Box delta encoding/decoding (Faster R-CNN parameterization).

Regression targets are expressed relative to anchors/proposals:

    dx = (x_gt - x_a) / w_a        dw = log(w_gt / w_a)
    dy = (y_gt - y_a) / h_a        dh = log(h_gt / h_a)

so predictions are scale-invariant. All boxes are XYXY.
"""

from __future__ import annotations

import math

import torch


class BoxCoder:
    """Encode boxes as deltas w.r.t. reference boxes and decode them back.

    Args:
        weights: (wx, wy, ww, wh) scaling applied to the deltas — Faster R-CNN
            RoI heads traditionally use (10, 10, 5, 5); RPN uses (1, 1, 1, 1).
        clip: maximum dw/dh before exp() to prevent overflow for extreme
            predictions early in training.
    """

    def __init__(
        self,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        clip: float = math.log(1000.0 / 16),
    ) -> None:
        self.weights = weights
        self.clip = clip

    def encode(
        self, reference_boxes: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Compute regression targets mapping ``anchors`` -> ``reference_boxes``.

        Both inputs are (N, 4) XYXY; returns (N, 4) deltas.
        """
        wx, wy, ww, wh = self.weights

        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_w
        anchor_cy = anchors[:, 1] + 0.5 * anchor_h

        gt_w = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_h = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_cx = reference_boxes[:, 0] + 0.5 * gt_w
        gt_cy = reference_boxes[:, 1] + 0.5 * gt_h

        dx = wx * (gt_cx - anchor_cx) / anchor_w
        dy = wy * (gt_cy - anchor_cy) / anchor_h
        dw = ww * torch.log(gt_w / anchor_w)
        dh = wh * torch.log(gt_h / anchor_h)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def decode(self, deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Apply predicted ``deltas`` (N, 4) to ``anchors`` (N, 4) -> XYXY boxes."""
        wx, wy, ww, wh = self.weights

        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + 0.5 * anchor_w
        anchor_cy = anchors[:, 1] + 0.5 * anchor_h

        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = torch.clamp(deltas[:, 2] / ww, max=self.clip)
        dh = torch.clamp(deltas[:, 3] / wh, max=self.clip)

        cx = dx * anchor_w + anchor_cx
        cy = dy * anchor_h + anchor_cy
        w = torch.exp(dw) * anchor_w
        h = torch.exp(dh) * anchor_h

        return torch.stack(
            [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=1
        )
