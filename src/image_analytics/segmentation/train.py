"""Segmentation training pipeline: config -> data -> model -> Trainer.

    python scripts/train.py --config configs/segmentation/unet_shapes.yaml

Semantic models (``unet``, ``deeplabv3plus``, ``smp``, ``segformer``) present a
``model(images) -> logits (B, C, H, W)`` interface and train through the base
Trainer with a pixel-wise criterion (``training.loss``; default CE+Dice) and a
:class:`SegmentationEvaluator`. Instance segmentation (``mask_rcnn``) reuses the
detection stack (DetectionTrainer, detection collate, mask-mAP evaluator).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from image_analytics.backbones.registry import build_backbone
from image_analytics.core.callbacks import (
    CheckpointCallback,
    EarlyStopping,
    LoggingCallback,
)
from image_analytics.core.config import ExperimentConfig, ModelConfig, save_config
from image_analytics.core.evaluator import MaskMAPEvaluator, SegmentationEvaluator
from image_analytics.core.mlflow import maybe_mlflow_callback
from image_analytics.core.registry import LOSSES, MODELS
from image_analytics.core.trainer import (
    Trainer,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    resolve_device,
    seed_everything,
)
from image_analytics.data.collate import detection_collate
from image_analytics.data.datasets.registry import build_dataset
from image_analytics.data.transforms.segmentation import build_segmentation_transforms
from image_analytics.detection.trainer import DetectionTrainer
from image_analytics.segmentation import change_detection  # noqa: F401  (register siamese_unet)
from image_analytics.segmentation import instance  # noqa: F401  (register mask_rcnn/mask2former)
from image_analytics.segmentation import losses  # noqa: F401  (register losses)
from image_analytics.segmentation import panoptic  # noqa: F401  (register oneformer)
from image_analytics.segmentation import semantic  # noqa: F401  (register models)

logger = logging.getLogger("image_analytics")

IGNORE_INDEX = 255

# Instance-segmentation models route through the detection stack.
_INSTANCE_MODELS = {"mask_rcnn"}
# Wrapper models build their own encoder (no pyramid backbone passed in).
_WRAPPER_MODELS = {"smp", "segformer"}
# Per-model pyramid configuration for the from-scratch semantic/instance models.
_SCRATCH_OUT_INDICES = {
    "unet": (0, 1, 2, 3, 4),       # all encoder levels (decoder per level)
    "siamese_unet": (0, 1, 2, 3, 4),  # U-Net decoder over per-level feature diffs
    "deeplabv3plus": (1, 4),       # low-level (stride 4) + high-level (stride 16)
    "mask_rcnn": (1, 2, 3, 4),     # C2-C5, as Faster R-CNN
}
_SCRATCH_BACKBONE_KWARGS = {
    "deeplabv3plus": {"output_stride": 16},
}


def _is_instance(model_name: str) -> bool:
    return model_name in _INSTANCE_MODELS


def build_segmentation_model(config: ModelConfig) -> nn.Module:
    """Build a segmentation model, dispatching on whether it owns its encoder."""
    bcfg = config.backbone
    kwargs = dict(config.kwargs)

    if config.name in _WRAPPER_MODELS:
        kwargs.setdefault("in_channels", bcfg.in_channels)
        if config.name == "smp":
            kwargs.setdefault("encoder_name", bcfg.name)
            kwargs.setdefault("encoder_weights", "imagenet" if bcfg.pretrained else None)
        elif config.name == "segformer":
            kwargs.setdefault("pretrained", bcfg.pretrained)
        return MODELS.build(config.name, num_classes=config.num_classes, **kwargs)

    # From-scratch: force pyramid mode and per-model out_indices/output_stride.
    if not bcfg.features_only:
        bcfg = dataclasses.replace(bcfg, features_only=True)
    if "out_indices" not in bcfg.kwargs:
        out_indices = _SCRATCH_OUT_INDICES.get(config.name, (0, 1, 2, 3, 4))
        extra = _SCRATCH_BACKBONE_KWARGS.get(config.name, {})
        bcfg = dataclasses.replace(
            bcfg, kwargs={**bcfg.kwargs, "out_indices": out_indices, **extra}
        )
    backbone = build_backbone(bcfg)

    if config.neck is not None:
        kwargs.setdefault("fpn_channels", config.neck.out_channels)
    return MODELS.build(
        config.name, backbone=backbone, num_classes=config.num_classes, **kwargs
    )


def build_dataloaders(
    config: ExperimentConfig, distributed: bool
) -> tuple[DataLoader, DataLoader]:
    data = config.data
    instance = _is_instance(config.model.name)

    def make_tf(train: bool):
        return build_segmentation_transforms(
            data.image_size, train=train, normalize=data.normalize,
            mean=data.mean, std=data.std, instance=instance,
        )

    train_ds = build_dataset(data, split="train", transform=make_tf(True))
    val_ds = build_dataset(data, split="val", transform=make_tf(False))
    collate = detection_collate if instance else None

    if distributed:
        train_sampler, shuffle = DistributedSampler(train_ds), False
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler, shuffle = None, True
        val_sampler = None

    pin_memory = resolve_device(config.training.device).type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=data.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=data.num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=len(train_ds) > data.batch_size,
        persistent_workers=data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=data.num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        persistent_workers=data.num_workers > 0,
    )
    return train_loader, val_loader


def _build_criterion(config: ExperimentConfig) -> nn.Module:
    tc = config.training
    loss_name = tc.loss or "ce_dice"  # CE+Dice is the practical default
    loss_kwargs = {"ignore_index": IGNORE_INDEX, **tc.loss_kwargs}
    return LOSSES.build(loss_name, **loss_kwargs)


def run(config: ExperimentConfig) -> dict[str, float]:
    """Train a segmentation model end-to-end from an ExperimentConfig."""
    distributed = init_distributed()
    seed_everything(config.seed)

    if is_main_process() and not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    tc = config.training
    output_dir = Path(config.output_dir) / config.experiment_name
    is_instance = _is_instance(config.model.name)

    train_loader, val_loader = build_dataloaders(config, distributed)
    model = build_segmentation_model(config.model)

    if is_instance:
        criterion = None
        evaluator = MaskMAPEvaluator(num_classes=config.model.num_classes)
        trainer_cls = DetectionTrainer
        default_monitor = "val/mAP"
    else:
        criterion = _build_criterion(config)
        evaluator = SegmentationEvaluator(
            num_classes=config.model.num_classes, ignore_index=IGNORE_INDEX
        )
        trainer_cls = Trainer
        default_monitor = "val/mIoU"

    monitor = tc.monitor if tc.monitor != "val/accuracy" else default_monitor

    optimizer = build_optimizer(model.parameters(), tc)
    scheduler = build_scheduler(optimizer, tc)

    callbacks = [
        LoggingCallback(log_interval=tc.log_interval),
        CheckpointCallback(
            dirpath=output_dir / "checkpoints", monitor=monitor, mode=tc.monitor_mode
        ),
    ]
    if tc.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=monitor, mode=tc.monitor_mode,
                patience=tc.early_stopping_patience,
            )
        )
    callbacks += maybe_mlflow_callback(config)

    trainer = trainer_cls(
        model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        evaluator=evaluator,
        device=tc.device,
        amp=tc.amp,
        grad_clip=tc.grad_clip,
        callbacks=callbacks,
        output_dir=output_dir,
        log_interval=tc.log_interval,
    )
    if tc.resume:
        trainer.load_checkpoint(tc.resume, resume=True)

    if is_main_process():
        save_config(config, output_dir / "config.yaml")

    try:
        metrics = trainer.fit(train_loader, val_loader, epochs=tc.epochs)
    finally:
        cleanup_distributed()
    return metrics
