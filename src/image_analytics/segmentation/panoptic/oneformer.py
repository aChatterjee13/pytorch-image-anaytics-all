"""OneFormer (Jain 2023) via HuggingFace (``[seg]`` extra).

A task-conditioned universal segmenter: a single model handles semantic,
instance, and panoptic segmentation, selected by a text task token. Thin
inference wrapper — ``predict(images, task=...)`` builds the task token, runs
the model, and applies HF's post-processing; outputs feed the
:class:`PanopticQualityEvaluator`.

Pretrained weights download from the hub on first use (network required).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import MODELS

_TASKS = ("panoptic", "instance", "semantic")


def _load_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for OneFormer. "
            "Install it with: pip install 'image-analytics[seg]'"
        ) from exc


@MODELS.register("oneformer")
class OneFormerWrapper(nn.Module):
    """Task-conditioned universal segmentation via OneFormer.

    Args:
        task: default task (``panoptic`` | ``instance`` | ``semantic``); can be
            overridden per ``predict`` call.
        model_name: HF checkpoint id (default COCO-pretrained Swin-tiny).
    """

    def __init__(
        self,
        num_classes: int | None = None,
        task: str = "panoptic",
        model_name: str = "shi-labs/oneformer_coco_swin_large",
        pretrained: bool = True,
        backbone: nn.Module | None = None,  # registry-protocol compat; unused
    ) -> None:
        super().__init__()
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        _load_transformers()
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        self.default_task = task
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)

    @torch.no_grad()
    def predict(self, images, task: str | None = None, target_sizes=None):
        task = task or self.default_task
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
        inputs = self.processor(
            images=images, task_inputs=[task], return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        if task == "semantic":
            return self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        if task == "instance":
            return self.processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )
        return self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=target_sizes
        )
