"""Unified training loop, DDP-aware.

Single-process:

    trainer = Trainer(model, optimizer, criterion, evaluator=evaluator)
    trainer.fit(train_loader, val_loader, epochs=10)

Distributed (launched via ``torchrun --nproc_per_node=N scripts/train.py ...``):
``init_distributed()`` picks up the torchrun environment, and the Trainer
wraps the model in DDP automatically when a process group is initialized.

The default ``training_step`` assumes ``(inputs, targets)`` batches and an
external criterion; task packages with richer batch structures (detection,
segmentation) subclass and override ``training_step`` / ``eval_step``.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from image_analytics.core.callbacks import Callback
from image_analytics.core.evaluator import Evaluator

logger = logging.getLogger("image_analytics")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> bool:
    """Initialize the process group from a torchrun environment.

    Returns True when distributed training is active. No-op (False) when not
    launched via torchrun.
    """
    if dist.is_initialized():
        return True
    if not dist.is_available() or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Optimizer / scheduler construction (shared by all task train pipelines)
# ---------------------------------------------------------------------------


def build_optimizer(params, config) -> torch.optim.Optimizer:
    """Build an optimizer from a :class:`TrainingConfig`."""
    name = config.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer {config.optimizer!r}; expected adamw|adam|sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Epoch-stepped LR schedule (cosine | step | none) with optional warmup."""
    name = (config.scheduler or "none").lower()
    if name == "none":
        return None

    if name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs - config.warmup_epochs, 1)
        )
    elif name == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    else:
        raise ValueError(f"Unknown scheduler {config.scheduler!r}; expected cosine|step|none")

    if config.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(config.warmup_epochs, 1) / 10,
            total_iters=config.warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, main], milestones=[config.warmup_epochs]
        )
    return main


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Task-agnostic training loop with AMP, gradient clipping, callbacks,
    checkpointing, and automatic DDP wrapping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        evaluator: Evaluator | None = None,
        device: str | torch.device = "auto",
        amp: bool = False,
        grad_clip: float | None = None,
        callbacks: Sequence[Callback] = (),
        output_dir: str | Path = "outputs",
        log_interval: int = 50,
    ) -> None:
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.distributed = dist.is_available() and dist.is_initialized()

        model = model.to(self.device)
        if self.distributed:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion.to(self.device) if criterion is not None else None
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.grad_clip = grad_clip
        self.callbacks = list(callbacks)
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval

        # AMP is enabled for CUDA only; CPU/MPS run full precision.
        self.amp_enabled = bool(amp) and self.device.type == "cuda"
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:  # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # Mutable state read by callbacks.
        self.epoch = 0
        self.start_epoch = 0
        self.global_step = 0
        self.last_loss = 0.0
        self.metrics: dict[str, float] = {}
        self.should_stop = False

        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- properties ---------------------------------------------------------

    @property
    def module(self) -> nn.Module:
        """The underlying model, unwrapped from DDP."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    @property
    def is_main_process(self) -> bool:
        return is_main_process()

    # -- hooks (override in task-specific trainers) --------------------------

    def training_step(self, batch) -> torch.Tensor:
        """Compute the loss for one batch. Default: (inputs, targets) + criterion."""
        inputs, targets = self._move_batch(batch)
        with self._autocast():
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, targets)
        return loss

    def eval_step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (outputs, targets, loss) for one validation batch."""
        inputs, targets = self._move_batch(batch)
        with self._autocast():
            outputs = self.model(inputs)
            loss = self._compute_loss(outputs, targets) if self.criterion is not None else None
        return outputs, targets, loss

    # -- internals -----------------------------------------------------------

    def _autocast(self):
        if self.amp_enabled:
            return torch.autocast(self.device.type)
        return nullcontext()

    def _move_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch
        return inputs.to(self.device, non_blocking=True), targets.to(
            self.device, non_blocking=True
        )

    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        if self.criterion is not None:
            return self.criterion(outputs, targets)
        # Forward-compat: models that return their own loss dict (detection).
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        raise RuntimeError(
            "No criterion provided and the model did not return a loss; "
            "pass criterion= or override training_step()."
        )

    def _call(self, hook: str) -> None:
        for callback in self.callbacks:
            getattr(callback, hook)(self)

    # -- main loop -----------------------------------------------------------

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Iterable | None = None,
        epochs: int = 1,
    ) -> dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Trainer.fit requires an optimizer")

        self._call("on_fit_start")
        for epoch in range(self.start_epoch, epochs):
            self.epoch = epoch
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            self._call("on_epoch_start")
            self.metrics = self._train_epoch(train_loader)
            if val_loader is not None:
                self.metrics.update(self.validate(val_loader))
            if self.scheduler is not None:
                self.scheduler.step()
            self._call("on_epoch_end")
            if self.should_stop:
                break
        self._call("on_fit_end")
        return self.metrics

    def _train_epoch(self, loader: Iterable) -> dict[str, float]:
        self.model.train()
        total_loss, batches = 0.0, 0
        for batch in loader:
            loss = self.training_step(batch)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            self.last_loss = float(loss.detach())
            total_loss += self.last_loss
            batches += 1
            self._call("on_batch_end")
        return {"train/loss": total_loss / max(batches, 1)}

    @torch.no_grad()
    def validate(self, loader: Iterable) -> dict[str, float]:
        self.model.eval()
        if self.evaluator is not None:
            self.evaluator.reset()

        total_loss, batches = 0.0, 0
        for batch in loader:
            outputs, targets, loss = self.eval_step(batch)
            if loss is not None:
                total_loss += float(loss.detach())
                batches += 1
            if self.evaluator is not None:
                if torch.is_tensor(outputs):
                    outputs = outputs.float()
                self.evaluator.update(outputs, targets)

        metrics: dict[str, float] = {}
        if batches:
            # Sync the loss so DDP ranks agree on monitored values.
            stats = torch.tensor([total_loss, float(batches)])
            if self.distributed:
                backend_device = "cuda" if dist.get_backend() == "nccl" else "cpu"
                stats = stats.to(backend_device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats = stats.cpu()
            metrics["val/loss"] = float(stats[0] / stats[1])
        if self.evaluator is not None:
            metrics.update(
                {f"val/{k}": v for k, v in self.evaluator.compute().items()}
            )
        return metrics

    # -- checkpointing --------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        if not self.is_main_process:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch + 1,
            "global_step": self.global_step,
            "metrics": dict(self.metrics),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path, resume: bool = True) -> None:
        """Load weights; with ``resume=True`` also restore optimizer/scheduler
        state and continue from the saved epoch."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.module.load_state_dict(state["model"])
        if resume:
            if self.optimizer is not None and state.get("optimizer"):
                self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler"):
                self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.start_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)
        logger.info("Loaded checkpoint %s (epoch %d)", path, state.get("epoch", 0))
