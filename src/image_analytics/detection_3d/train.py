"""Point-cloud training pipeline: config -> data -> model -> Trainer.

    python scripts/train.py --config configs/pointcloud/pointnet_primitives.yaml

Dispatches on the model: classification / part-segmentation models
(``pointnet``, ``pointnet2``, ``dgcnn``) present ``model(points) -> logits`` and
train through a :class:`PointCloudTrainer` (which adds PointNet's orthogonality
regularizer); ``pointpillars`` trains through the detection stack (loss-dict +
3D-mAP evaluator).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from image_analytics.core.callbacks import (
    CheckpointCallback,
    EarlyStopping,
    LoggingCallback,
)
from image_analytics.core.config import ExperimentConfig, ModelConfig, save_config
from image_analytics.core.evaluator import (
    ClassificationEvaluator,
    Detection3DEvaluator,
    SegmentationEvaluator,
)
from image_analytics.core.mlflow import maybe_mlflow_callback
from image_analytics.core.registry import MODELS
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
from image_analytics.data.transforms.pointcloud import build_pointcloud_transforms
from image_analytics.detection.trainer import DetectionTrainer
from image_analytics.detection_3d import (  # noqa: F401  (register point models)
    dgcnn,
    pointnet,
    pointpillars,
)

logger = logging.getLogger("image_analytics")

_DETECTION_MODELS = {"pointpillars"}


def _mode(config: ModelConfig) -> str:
    if config.name in _DETECTION_MODELS:
        return "detection"
    return config.kwargs.get("task", "classification")


class PointCloudTrainer(Trainer):
    """Base loop + PointNet's orthogonality regularizer (if the model has one)."""

    def training_step(self, batch) -> torch.Tensor:
        points, targets = self._move_batch(batch)
        with self._autocast():
            outputs = self.model(points)
            loss = self.criterion(outputs, targets)
            reg = getattr(self.module, "regularization_loss", None)
            if reg is not None:
                loss = loss + reg()
        return loss


def build_pointcloud_model(config: ModelConfig) -> nn.Module:
    return MODELS.build(config.name, num_classes=config.num_classes, **config.kwargs)


def build_dataloaders(
    config: ExperimentConfig, distributed: bool
) -> tuple[DataLoader, DataLoader]:
    data = config.data
    mode = _mode(config.model)
    collate = detection_collate if mode == "detection" else None

    # Point detection keeps raw coordinates; cls/seg normalize to the unit sphere.
    train_tf = (
        None if mode == "detection"
        else build_pointcloud_transforms(train=True, augment=data.augment)
    )
    val_tf = (
        None if mode == "detection"
        else build_pointcloud_transforms(train=False, augment=data.augment)
    )
    train_ds = build_dataset(data, split="train", transform=train_tf)
    val_ds = build_dataset(data, split="val", transform=val_tf)

    if distributed:
        train_sampler, shuffle = DistributedSampler(train_ds), False
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler, shuffle, val_sampler = None, True, None

    pin_memory = resolve_device(config.training.device).type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=data.batch_size, shuffle=shuffle, sampler=train_sampler,
        num_workers=data.num_workers, collate_fn=collate, pin_memory=pin_memory,
        drop_last=len(train_ds) > data.batch_size, persistent_workers=data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=data.num_workers, collate_fn=collate, pin_memory=pin_memory,
        persistent_workers=data.num_workers > 0,
    )
    return train_loader, val_loader


def run(config: ExperimentConfig) -> dict[str, float]:
    distributed = init_distributed()
    seed_everything(config.seed)
    if is_main_process() and not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
        )

    tc = config.training
    output_dir = Path(config.output_dir) / config.experiment_name
    mode = _mode(config.model)

    train_loader, val_loader = build_dataloaders(config, distributed)
    model = build_pointcloud_model(config.model)

    if mode == "detection":
        criterion, trainer_cls = None, DetectionTrainer
        evaluator = Detection3DEvaluator(num_classes=config.model.num_classes)
        default_monitor = "val/mAP_3d"
    elif mode == "segmentation":
        criterion = nn.CrossEntropyLoss()
        trainer_cls = PointCloudTrainer
        evaluator = SegmentationEvaluator(num_classes=config.model.num_classes)
        default_monitor = "val/mIoU"
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
        trainer_cls = PointCloudTrainer
        topk = (1, 5) if config.model.num_classes > 5 else (1,)
        evaluator = ClassificationEvaluator(config.model.num_classes, topk=topk)
        default_monitor = "val/accuracy"

    monitor = tc.monitor if tc.monitor != "val/accuracy" else default_monitor

    optimizer = build_optimizer(model.parameters(), tc)
    scheduler = build_scheduler(optimizer, tc)
    callbacks = [
        LoggingCallback(log_interval=tc.log_interval),
        CheckpointCallback(dirpath=output_dir / "checkpoints", monitor=monitor, mode=tc.monitor_mode),
    ]
    if tc.early_stopping_patience is not None:
        callbacks.append(EarlyStopping(monitor=monitor, mode=tc.monitor_mode, patience=tc.early_stopping_patience))
    callbacks += maybe_mlflow_callback(config)

    trainer = trainer_cls(
        model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        evaluator=evaluator, device=tc.device, amp=tc.amp, grad_clip=tc.grad_clip,
        callbacks=callbacks, output_dir=output_dir, log_interval=tc.log_interval,
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
