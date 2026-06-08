"""Box-aware transform pipelines (torchvision v2 + tv_tensors).

Detection samples are ``(image, target)`` pairs where ``target`` is a dict:

    {"boxes": tv_tensors.BoundingBoxes (N, 4) XYXY,
     "labels": LongTensor (N,),          # 0-based foreground classes
     "image_id": LongTensor (1,)}

v2 geometric transforms update the boxes automatically; degenerate boxes
(and their labels) are dropped by ``SanitizeBoundingBoxes``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2._utils import query_size

from image_analytics.data.transforms.augmentations import IMAGENET_MEAN, IMAGENET_STD


class LetterboxResize(v2.Transform):
    """Aspect-preserving resize onto a square ``size``×``size`` canvas.

    The longer side is scaled to ``size`` and the shorter side is padded
    (bottom/right) so boxes keep their aspect ratio — unlike a plain square
    ``Resize``, which distorts non-square images. Box/mask geometry is updated
    by the underlying v2 functional ops; the top-left origin means padding adds
    no coordinate shift.
    """

    def __init__(self, size: int, fill: float = 0.0) -> None:
        super().__init__()
        self.size = int(size)
        self.fill = fill

    def _get_params(self, flat_inputs: list) -> dict:
        h, w = query_size(flat_inputs)
        scale = self.size / max(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        # pad = [left, top, right, bottom]
        return {
            "size": [new_h, new_w],
            "padding": [0, 0, self.size - new_w, self.size - new_h],
        }

    def _transform(self, inpt, params: dict):
        inpt = TF.resize(inpt, params["size"], antialias=True)
        return TF.pad(inpt, params["padding"], fill=self.fill)


def build_detection_transforms(
    image_size: int,
    train: bool = True,
    hflip: float = 0.5,
    normalize: str = "imagenet",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    letterbox: bool = False,
) -> v2.Compose:
    """Detection pipeline: flip (train) -> resize -> normalize -> sanitize.

    Images are sized to a fixed square so the collate function can stack them
    into one batch tensor. ``letterbox=True`` preserves aspect ratio with
    padding (recommended for non-square imagery); the default square resize is
    simpler and matches the synthetic fixtures.
    """
    ops: list[v2.Transform] = []
    if train and hflip > 0:
        ops.append(v2.RandomHorizontalFlip(p=hflip))
    if letterbox:
        ops.append(LetterboxResize(image_size))
    else:
        ops.append(v2.Resize((image_size, image_size), antialias=True))
    ops.append(v2.ToImage())
    ops.append(v2.ToDtype(torch.float32, scale=True))
    if normalize == "imagenet" or (mean is not None and std is not None):
        ops.append(
            v2.Normalize(
                mean=list(mean) if mean is not None else list(IMAGENET_MEAN),
                std=list(std) if std is not None else list(IMAGENET_STD),
            )
        )
    ops.append(v2.SanitizeBoundingBoxes())
    return v2.Compose(ops)
