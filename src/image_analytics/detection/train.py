"""Detection training pipeline: config -> data -> model -> DetectionTrainer.

    python scripts/train.py --config configs/detection/retinanet_shapes.yaml
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
from image_analytics.core.evaluator import DetectionEvaluator
from image_analytics.core.registry import MODELS
from image_analytics.core.trainer import (
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
from image_analytics.data.transforms.detection import build_detection_transforms
from image_analytics.detection import heads  # noqa: F401  (register detectors)
from image_analytics.detection.trainer import DetectionTrainer

logger = logging.getLogger("image_analytics")

# Per-detector pyramid levels (timm resnet-style feature_info indices):
# one-stage detectors take C3-C5 (strides 8/16/32) and extend with P6/P7;
# two-stage takes C2-C5 (strides 4/8/16/32) for fine RoI pooling.
_DEFAULT_OUT_INDICES = {
    "faster_rcnn": (1, 2, 3, 4),
}
_FALLBACK_OUT_INDICES = (2, 3, 4)


def build_detection_model(config: ModelConfig) -> nn.Module:
    """Build a detector; the backbone is forced into pyramid mode."""
    backbone_cfg = config.backbone
    if not backbone_cfg.features_only:
        backbone_cfg = dataclasses.replace(backbone_cfg, features_only=True)
    if "out_indices" not in backbone_cfg.kwargs:
        out_indices = _DEFAULT_OUT_INDICES.get(config.name, _FALLBACK_OUT_INDICES)
        backbone_cfg = dataclasses.replace(
            backbone_cfg,
            kwargs={**backbone_cfg.kwargs, "out_indices": out_indices},
        )
    backbone = build_backbone(backbone_cfg)

    kwargs = dict(config.kwargs)
    if config.neck is not None:
        kwargs.setdefault("fpn_channels", config.neck.out_channels)
    return MODELS.build(
        config.name, backbone=backbone, num_classes=config.num_classes, **kwargs
    )


def build_dataloaders(
    config: ExperimentConfig, distributed: bool
) -> tuple[DataLoader, DataLoader]:
    data = config.data
    train_tf = build_detection_transforms(
        data.image_size, train=True, normalize=data.normalize,
        mean=data.mean, std=data.std,
    )
    val_tf = build_detection_transforms(
        data.image_size, train=False, normalize=data.normalize,
        mean=data.mean, std=data.std,
    )
    train_ds = build_dataset(data, split="train", transform=train_tf)
    val_ds = build_dataset(data, split="val", transform=val_tf)

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
        collate_fn=detection_collate,
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
        collate_fn=detection_collate,
        pin_memory=pin_memory,
        persistent_workers=data.num_workers > 0,
    )
    return train_loader, val_loader


def run(config: ExperimentConfig) -> dict[str, float]:
    """Train a detector end-to-end from an ExperimentConfig."""
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

    train_loader, val_loader = build_dataloaders(config, distributed)
    model = build_detection_model(config.model)

    optimizer = build_optimizer(model.parameters(), tc)
    scheduler = build_scheduler(optimizer, tc)
    evaluator = DetectionEvaluator(num_classes=config.model.num_classes)

    callbacks = [
        LoggingCallback(log_interval=tc.log_interval),
        CheckpointCallback(
            dirpath=output_dir / "checkpoints",
            monitor=tc.monitor,
            mode=tc.monitor_mode,
        ),
    ]
    if tc.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=tc.monitor, mode=tc.monitor_mode,
                patience=tc.early_stopping_patience,
            )
        )

    trainer = DetectionTrainer(
        model,
        optimizer=optimizer,
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
