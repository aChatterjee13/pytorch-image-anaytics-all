"""segmentation_models_pytorch breadth via a thin wrapper (``[seg]`` extra).

smp contributes architectures we don't hand-roll (U-Net++, MAnet, Linknet,
PSPNet, PAN, DeepLabV3) over 400+ timm encoders, all behind the same
``model(images) -> logits (B, C, H, W)`` interface as the from-scratch models.
Use ``encoder_name="tu-<timm_model>"`` to pull any timm backbone as encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import MODELS


def _load_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "segmentation-models-pytorch is required for smp models. "
            "Install it with: pip install 'image-analytics[seg]'"
        ) from exc
    return smp


@MODELS.register("smp")
class SMPWrapper(nn.Module):
    """Wrap any ``smp.create_model`` architecture.

    Args:
        arch: ``unet`` | ``unetplusplus`` | ``manet`` | ``linknet`` | ``fpn`` |
            ``pspnet`` | ``pan`` | ``deeplabv3`` | ``deeplabv3plus``.
        encoder_name: smp/timm encoder (e.g. ``resnet34`` or ``tu-convnext_tiny``).
        encoder_weights: ``imagenet`` for pretrained, ``None`` for offline.
        in_channels: input bands (multi-channel capable via the encoder stem).
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "unet",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        backbone: nn.Module | None = None,  # registry-protocol compat; unused
        **kwargs,
    ) -> None:
        super().__init__()
        smp = _load_smp()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
