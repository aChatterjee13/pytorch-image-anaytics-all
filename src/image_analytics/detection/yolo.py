"""YOLO family adapter (ultralytics — optional ``[detection]`` extra).

YOLO ships with its own highly-tuned training loop, mosaic augmentation, and
loss; re-plumbing it through our Trainer would discard most of its value.
The integration is therefore:

* **Inference/eval adapter** — ``YOLOWrapper`` converts ultralytics results
  into this platform's prediction dicts, so YOLO models drop into
  ``DetectionEvaluator`` and downstream tooling unchanged.
* **Native training passthrough** — ``YOLOWrapper.train_native`` delegates to
  ultralytics' ``model.train()`` with its dataset-YAML convention.

Model names: ``yolo11n.pt`` (downloads pretrained weights) or
``yolo11n.yaml`` (random-init architecture, fully offline).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from image_analytics.core.registry import MODELS


def _load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ultralytics is required for YOLO models. "
            "Install it with: pip install 'image-analytics[detection]'"
        ) from exc
    return YOLO


@MODELS.register("yolo")
class YOLOWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        backbone: nn.Module | None = None,   # registry protocol compat; unused
        num_classes: int | None = None,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.7,
    ) -> None:
        super().__init__()
        YOLO = _load_ultralytics()
        self.yolo = YOLO(model_name)
        self.model_name = model_name
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def train(self, mode: bool = True) -> "YOLOWrapper":
        """Set training flag WITHOUT recursing into children.

        ultralytics' ``Model`` overrides ``.train()`` as its training API, so
        the default ``nn.Module.train()`` recursion would call it with
        ``mode`` as a trainer argument and crash.
        """
        self.training = mode
        return self

    def forward(
        self, images: torch.Tensor, targets: list[dict] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        if self.training or targets is not None:
            raise RuntimeError(
                "YOLO trains through its own loop — use YOLOWrapper.train_native"
                "(data_yaml, epochs=...) instead of the platform Trainer."
            )
        results = self.yolo.predict(
            images, conf=self.score_thresh, iou=self.nms_thresh, verbose=False
        )
        predictions = []
        for result in results:
            boxes = result.boxes
            predictions.append(
                {
                    "boxes": boxes.xyxy.detach().cpu(),
                    "scores": boxes.conf.detach().cpu(),
                    "labels": boxes.cls.detach().cpu().long(),
                }
            )
        return predictions

    def train_native(self, data_yaml: str, **kwargs: Any):
        """Train with the ultralytics loop (dataset described by a YOLO data
        YAML); returns the ultralytics results object."""
        return self.yolo.train(data=data_yaml, **kwargs)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self.yolo.names)
