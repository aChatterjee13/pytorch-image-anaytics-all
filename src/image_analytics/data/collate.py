"""Collate functions for tasks with ragged targets."""

from __future__ import annotations

import torch


def detection_collate(batch):
    """Stack images (all resized to a common shape by the transforms) and
    keep per-image target dicts as a list."""
    images = torch.stack([sample[0] for sample in batch])
    targets = [sample[1] for sample in batch]
    return images, targets
