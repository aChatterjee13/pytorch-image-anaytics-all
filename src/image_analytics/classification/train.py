"""Classification training pipeline: config -> data -> model -> Trainer.

Invoked via the unified entry point:

    python scripts/train.py --config configs/classification/cifar10_resnet18.yaml

Distributed:

    torchrun --nproc_per_node=4 scripts/train.py --config ...
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from image_analytics.classification.models import build_model
from image_analytics.core.callbacks import (
    CheckpointCallback,
    EarlyStopping,
    LoggingCallback,
)
from image_analytics.core.config import ExperimentConfig, save_config
from image_analytics.core.evaluator import ClassificationEvaluator, MultiLabelEvaluator
from image_analytics.core.mlflow import maybe_mlflow_callback
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
from image_analytics.data.datasets.registry import build_dataset
from image_analytics.data.samplers import build_balanced_sampler
from image_analytics.data.transforms.augmentations import build_transforms

logger = logging.getLogger("image_analytics")


def build_dataloaders(
    config: ExperimentConfig, distributed: bool
) -> tuple[DataLoader, DataLoader]:
    data = config.data
    train_tf = build_transforms(
        data.image_size,
        train=True,
        augment=data.augment,
        normalize=data.normalize,
        mean=data.mean,
        std=data.std,
    )
    val_tf = build_transforms(
        data.image_size,
        train=False,
        augment=data.augment,
        normalize=data.normalize,
        mean=data.mean,
        std=data.std,
    )

    train_ds = build_dataset(data, split="train", transform=train_tf)
    val_ds = build_dataset(data, split="val", transform=val_tf)

    if distributed:
        train_sampler, shuffle = DistributedSampler(train_ds), False
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    elif data.balanced_sampling:
        train_sampler, shuffle = build_balanced_sampler(train_ds), False
        val_sampler = None
    else:
        train_sampler, shuffle = None, True
        val_sampler = None

    device = resolve_device(config.training.device)
    pin_memory = device.type == "cuda"
    # drop_last avoids BatchNorm failures on a trailing batch of size 1, but
    # would produce zero batches when the dataset is smaller than one batch.
    drop_last = len(train_ds) > data.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=data.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=data.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=data.num_workers > 0,
    )
    return train_loader, val_loader


def run(config: ExperimentConfig) -> dict[str, float]:
    """Train a classification model end-to-end from an ExperimentConfig."""
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
    model = build_model(config.model)

    multilabel = getattr(model, "is_multilabel", False)
    if multilabel:
        criterion: nn.Module = nn.BCEWithLogitsLoss()
        evaluator = MultiLabelEvaluator(num_labels=config.model.num_classes)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
        topk = (1, 5) if config.model.num_classes > 5 else (1,)
        evaluator = ClassificationEvaluator(config.model.num_classes, topk=topk)

    optimizer = build_optimizer(model.parameters(), tc)
    scheduler = build_scheduler(optimizer, tc)

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
                monitor=tc.monitor,
                mode=tc.monitor_mode,
                patience=tc.early_stopping_patience,
            )
        )
    callbacks += maybe_mlflow_callback(config)

    trainer = Trainer(
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
