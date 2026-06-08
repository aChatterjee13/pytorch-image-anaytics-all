"""Mask2Former (Cheng 2022) via HuggingFace (``[seg]`` extra).

A universal (semantic / instance / panoptic) segmenter behind a thin inference
wrapper: ``predict(images, task=...)`` runs the model and applies HF's
task-specific post-processing, returning either a class-index map (semantic) or
per-image ``{"segmentation", "segments_info"}`` (instance/panoptic), which feed
the :class:`PanopticQualityEvaluator`.

Pretrained weights download from the hub on first use (network required), so
this is exercised manually rather than in the offline test suite.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import MODELS

_DEFAULT_MODELS = {
    "panoptic": "facebook/mask2former-swin-tiny-coco-panoptic",
    "instance": "facebook/mask2former-swin-tiny-coco-instance",
    "semantic": "facebook/mask2former-swin-tiny-ade-semantic",
}


def _load_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for Mask2Former. "
            "Install it with: pip install 'image-analytics[seg]'"
        ) from exc


@MODELS.register("mask2former")
class Mask2FormerWrapper(nn.Module):
    """Promptless universal segmentation via Mask2Former.

    Args:
        task: ``panoptic`` | ``instance`` | ``semantic`` — selects the default
            checkpoint and the post-processing applied in ``predict``.
        model_name: override the HF checkpoint id.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        task: str = "panoptic",
        model_name: str | None = None,
        pretrained: bool = True,
        backbone: nn.Module | None = None,  # registry-protocol compat; unused
    ) -> None:
        super().__init__()
        if task not in _DEFAULT_MODELS:
            raise ValueError(f"task must be one of {sorted(_DEFAULT_MODELS)}, got {task!r}")
        _load_transformers()
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )

        self.task = task
        name = model_name or _DEFAULT_MODELS[task]
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(name)

    def forward(self, images, **inputs):
        """Raw HF forward (pass ``pixel_values`` / labels for fine-tuning)."""
        if not inputs:
            inputs = {"pixel_values": images}
        return self.model(**inputs)

    @torch.no_grad()
    def predict(self, images, target_sizes=None):
        """Run inference and post-process per ``self.task``.

        ``images`` may be PIL images / numpy arrays (preprocessed by the HF
        image processor). Returns the processor's task-specific output.
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if self.task == "semantic":
            return self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        if self.task == "instance":
            return self.processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )
        return self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes
        )
