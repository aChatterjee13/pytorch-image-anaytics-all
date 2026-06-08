"""Production transformer detectors via HuggingFace (``[seg]`` extra shares
``transformers``).

From-scratch DETR (Phase 2) is educational and converges slowly; Deformable
DETR and RT-DETR are the production paths. This generic wrapper adapts any HF
object-detection model to the platform's detector interface:

    train:  model(images, targets) -> {"loss", ...}   (HF's own set criterion)
    eval:   model(images)          -> [{"boxes", "scores", "labels"}]

so they fine-tune through ``DetectionTrainer`` and score through
``DetectionEvaluator`` unchanged. Our detection transforms already apply
ImageNet normalization (what these models expect), so pixel tensors are fed
straight in; the HF image processor is used only to post-process boxes.

Weights download from the hub on first use (network required), so these are
exercised manually rather than in the offline test suite.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.ops as tvops

from image_analytics.core.registry import MODELS


def _load_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for Deformable-DETR / RT-DETR. "
            "Install it with: pip install 'image-analytics[seg]'"
        ) from exc


class HFDetectionWrapper(nn.Module):
    """Adapter around ``AutoModelForObjectDetection`` (DETR family)."""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        pretrained: bool = True,
        score_thresh: float = 0.3,
        backbone: nn.Module | None = None,  # registry-protocol compat; unused
    ) -> None:
        super().__init__()
        _load_transformers()
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def _to_hf_labels(self, images: torch.Tensor, targets: list[dict]) -> list[dict]:
        h, w = images.shape[-2:]
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=images.device)
        labels = []
        for target in targets:
            boxes = torch.as_tensor(target["boxes"], dtype=torch.float32, device=images.device)
            norm = (
                tvops.box_convert(boxes, "xyxy", "cxcywh") / scale
                if len(boxes)
                else boxes.reshape(0, 4)
            )
            labels.append({"class_labels": target["labels"].to(images.device), "boxes": norm})
        return labels

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        if self.training or targets is not None:
            if targets is None:
                raise ValueError("targets are required in training mode")
            outputs = self.model(pixel_values=images, labels=self._to_hf_labels(images, targets))
            losses = dict(outputs.loss_dict) if outputs.loss_dict else {}
            losses["loss"] = outputs.loss
            return losses

        outputs = self.model(pixel_values=images)
        sizes = [tuple(images.shape[-2:])] * images.shape[0]
        processed = self.processor.post_process_object_detection(
            outputs, threshold=self.score_thresh, target_sizes=sizes
        )
        return [
            {"boxes": r["boxes"], "scores": r["scores"], "labels": r["labels"]}
            for r in processed
        ]


@MODELS.register("deformable_detr")
def build_deformable_detr(
    num_classes: int, backbone: nn.Module | None = None,
    model_name: str = "SenseTime/deformable-detr", **kwargs,
) -> HFDetectionWrapper:
    """Deformable DETR (Zhu 2020): deformable attention, ~10x faster
    convergence than vanilla DETR."""
    return HFDetectionWrapper(num_classes, model_name=model_name, **kwargs)


@MODELS.register("rt_detr")
def build_rt_detr(
    num_classes: int, backbone: nn.Module | None = None,
    model_name: str = "PekingU/rtdetr_r50vd", **kwargs,
) -> HFDetectionWrapper:
    """RT-DETR (Zhao 2023): real-time DETR, NMS-free."""
    return HFDetectionWrapper(num_classes, model_name=model_name, **kwargs)
