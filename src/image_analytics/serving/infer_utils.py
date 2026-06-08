"""Shared inference helpers: task-dispatched model construction and the eval
preprocessing transform rebuilt from a config (used by the inference CLI and
the TorchServe handler, so they never diverge)."""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from image_analytics.core.config import ExperimentConfig
from image_analytics.data.transforms.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_transforms


def build_task_model(config: ExperimentConfig) -> torch.nn.Module:
    """Build the model for ``config.task`` (untrained — load a checkpoint next)."""
    if config.task == "classification":
        from image_analytics.classification.models import build_model

        return build_model(config.model)
    if config.task == "detection":
        from image_analytics.detection.train import build_detection_model

        return build_detection_model(config.model)
    if config.task == "segmentation":
        from image_analytics.segmentation.train import build_segmentation_model

        return build_segmentation_model(config.model)
    if config.task == "pointcloud":
        from image_analytics.detection_3d.train import build_pointcloud_model

        return build_pointcloud_model(config.model)
    raise ValueError(f"Unknown task {config.task!r}")


def build_eval_transform(config: ExperimentConfig):
    """A PIL-image -> normalized tensor eval transform matching training.

    Classification reuses the classification eval pipeline (resize + center
    crop); detection/segmentation use a square resize (matching their training
    transform) so predicted coordinates map back cleanly.
    """
    data = config.data
    if config.task == "classification":
        return build_transforms(
            data.image_size, train=False, augment="none",
            normalize=data.normalize, mean=data.mean, std=data.std,
        )

    ops: list = [
        v2.Resize((data.image_size, data.image_size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if data.normalize == "imagenet" or (data.mean is not None and data.std is not None):
        ops.append(
            v2.Normalize(
                mean=list(data.mean) if data.mean is not None else list(IMAGENET_MEAN),
                std=list(data.std) if data.std is not None else list(IMAGENET_STD),
            )
        )
    return v2.Compose(ops)


def load_checkpoint_into(model: torch.nn.Module, checkpoint: str, device: str = "cpu") -> torch.nn.Module:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
    return model


# ---------------------------------------------------------------------------
# Task-dispatched post-processing (shared by the CLI and the TorchServe handler)
# ---------------------------------------------------------------------------


def classification_topk(logits: torch.Tensor, topk: int = 5) -> dict:
    probs = logits.softmax(dim=1)[0]
    top = probs.topk(min(topk, probs.numel()))
    return {
        "labels": top.indices.tolist(),
        "scores": [round(s, 5) for s in top.values.tolist()],
    }


def detection_to_dict(
    predictions: list, orig_size: tuple[int, int], image_size: int, score_thresh: float = 0.3
) -> dict:
    """Filter by score and rescale boxes from the square model input back to the
    original ``(width, height)``."""
    pred = predictions[0]
    w, h = orig_size
    sx, sy = w / image_size, h / image_size
    keep = pred["scores"] >= score_thresh
    boxes = pred["boxes"][keep].clone()
    boxes[:, 0::2] *= sx
    boxes[:, 1::2] *= sy
    return {
        "boxes": [[round(v, 2) for v in b] for b in boxes.tolist()],
        "scores": [round(s, 5) for s in pred["scores"][keep].tolist()],
        "labels": pred["labels"][keep].tolist(),
    }


def segmentation_summary(logits: torch.Tensor) -> dict:
    mask = logits.argmax(dim=1)[0]
    classes, counts = mask.unique(return_counts=True)
    return {"pixel_counts": {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}}
