"""IoU-based assignment of ground-truth boxes to anchors/proposals."""

from __future__ import annotations

import torch


class Matcher:
    """Assign each anchor the best-overlapping GT, with an ignore band.

    Given an IoU matrix (num_gt, num_anchors), returns a (num_anchors,) index
    tensor: matched GT index, ``BACKGROUND`` (-1) below ``low_threshold``, or
    ``IGNORE`` (-2) inside [low, high) — excluded from the classification loss.

    ``allow_low_quality_matches`` forces every GT to claim its best-IoU
    anchor(s) even below the threshold, so no ground truth goes unassigned
    (essential for small objects between anchor grid points).
    """

    BACKGROUND = -1
    IGNORE = -2

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.4,
        allow_low_quality_matches: bool = True,
    ) -> None:
        if low_threshold > high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be <= high_threshold "
                f"({high_threshold})"
            )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou: torch.Tensor) -> torch.Tensor:
        if iou.numel() == 0:
            # No GT: everything is background
            return torch.full(
                (iou.shape[1],), self.BACKGROUND, dtype=torch.int64, device=iou.device
            )

        best_iou, matches = iou.max(dim=0)  # per anchor

        below = best_iou < self.low_threshold
        between = (best_iou >= self.low_threshold) & (best_iou < self.high_threshold)
        matches = matches.clone()
        matches[below] = self.BACKGROUND
        matches[between] = self.IGNORE

        if self.allow_low_quality_matches:
            # For each GT, find anchor(s) tied at its maximum IoU and force-match
            best_per_gt = iou.max(dim=1).values  # (num_gt,)
            gt_idx, anchor_idx = torch.where(iou == best_per_gt[:, None])
            keep = best_per_gt[gt_idx] > 0  # GT with zero overlap stays unmatched
            matches[anchor_idx[keep]] = gt_idx[keep]

        return matches


class BalancedPositiveNegativeSampler:
    """Subsample matches to a fixed minibatch with a target positive fraction.

    Used by RPN (256 @ 0.5) and RoI heads (512 @ 0.25) where, unlike focal
    loss, plain BCE/CE needs explicit foreground/background balancing.
    """

    def __init__(self, batch_size_per_image: int = 256, positive_fraction: float = 0.5) -> None:
        if not 0.0 < positive_fraction <= 1.0:
            raise ValueError(f"positive_fraction must be in (0, 1], got {positive_fraction}")
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (positive indices, negative indices) sampled from a match
        vector (>=0 positive, BACKGROUND negative, IGNORE excluded)."""
        positive = torch.where(matches >= 0)[0]
        negative = torch.where(matches == Matcher.BACKGROUND)[0]

        num_pos = min(len(positive), int(self.batch_size_per_image * self.positive_fraction))
        num_neg = min(len(negative), self.batch_size_per_image - num_pos)

        pos_perm = torch.randperm(len(positive), device=matches.device)[:num_pos]
        neg_perm = torch.randperm(len(negative), device=matches.device)[:num_neg]
        return positive[pos_perm], negative[neg_perm]
