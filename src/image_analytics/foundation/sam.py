"""Segment Anything (Kirillov 2023) via HuggingFace (``[seg]`` extra).

SAM v1 is inference-only and promptable: ``predict(image, points=…, boxes=…)``
returns one or more candidate masks per prompt with IoU confidence scores. It
runs on torch 2.2 (this Intel-Mac ceiling). SAM 2 needs ``torch>=2.3.1`` and is
lazy-gated behind :func:`load_sam2` with an actionable error.

Weights download from the hub on first use (network required), so SAM is
exercised manually rather than in the offline test suite.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import MODELS


def _load_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for SAM. "
            "Install it with: pip install 'image-analytics[seg]'"
        ) from exc


@MODELS.register("sam")
class SAMWrapper(nn.Module):
    """Promptable segmentation with SAM v1.

    Prompts follow the HF ``SamProcessor`` convention:
      * ``points``: nested list ``[[[x, y], ...]]`` (per image, per object).
      * ``labels``: matching foreground(1)/background(0) flags.
      * ``boxes``: nested list ``[[[x1, y1, x2, y2], ...]]``.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        backbone: nn.Module | None = None,  # registry-protocol compat; unused
        num_classes: int | None = None,     # unused; SAM is class-agnostic
    ) -> None:
        super().__init__()
        _load_transformers()
        from transformers import SamModel, SamProcessor

        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name)

    @torch.no_grad()
    def predict(
        self,
        image,
        points=None,
        labels=None,
        boxes=None,
        multimask_output: bool = True,
    ):
        """Return ``(masks, iou_scores)`` for the given prompts.

        ``masks`` is a list (per image) of ``(num_objects, num_masks, H, W)``
        boolean tensors at the original image resolution; ``iou_scores`` are the
        model's per-mask confidence estimates.
        """
        inputs = self.processor(
            image,
            input_points=points,
            input_labels=labels,
            input_boxes=boxes,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**inputs, multimask_output=multimask_output)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks, outputs.iou_scores.cpu()


def load_sam2(model_name: str = "facebook/sam2-hiera-tiny"):
    """Lazy gate for SAM 2 (video-capable). Requires ``torch>=2.3.1``, which is
    unavailable on the x86_64 macOS torch 2.2.x ceiling — raises with guidance
    rather than failing obscurely."""
    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:3])
    if version < (2, 3, 1):
        raise RuntimeError(
            f"SAM 2 requires torch>=2.3.1 but found {torch.__version__}. "
            "SAM 2 is unavailable on this machine (x86_64 macOS caps torch at "
            "2.2.x); use SAMWrapper (SAM v1) here, or run SAM 2 on a "
            "Linux/CUDA box."
        )
    try:  # pragma: no cover - requires torch>=2.3.1
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sam2 is required for SAM 2. Install it with: pip install sam2"
        ) from exc
    return SAM2ImagePredictor.from_pretrained(model_name)
