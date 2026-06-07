"""Trainer callbacks: logging, checkpointing, early stopping.

Callbacks receive the trainer instance and read its state
(``trainer.epoch``, ``trainer.global_step``, ``trainer.metrics``, ...).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

logger = logging.getLogger("image_analytics")


class Callback:
    """Base callback; subclass and override any hook."""

    def on_fit_start(self, trainer) -> None: ...
    def on_fit_end(self, trainer) -> None: ...
    def on_epoch_start(self, trainer) -> None: ...
    def on_epoch_end(self, trainer) -> None: ...
    def on_batch_end(self, trainer) -> None: ...


class LoggingCallback(Callback):
    """Console logging of batch loss / lr and epoch metric summaries."""

    def __init__(self, log_interval: int = 50) -> None:
        self.log_interval = log_interval
        self._epoch_start = 0.0

    def on_fit_start(self, trainer) -> None:
        if trainer.is_main_process:
            params = sum(p.numel() for p in trainer.module.parameters())
            logger.info(
                "Starting training: %s parameters, device=%s, distributed=%s",
                f"{params:,}", trainer.device, trainer.distributed,
            )

    def on_epoch_start(self, trainer) -> None:
        self._epoch_start = time.perf_counter()

    def on_batch_end(self, trainer) -> None:
        if not trainer.is_main_process:
            return
        if self.log_interval and trainer.global_step % self.log_interval == 0:
            lr = trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer else 0.0
            logger.info(
                "epoch %d step %d | loss %.4f | lr %.2e",
                trainer.epoch, trainer.global_step, trainer.last_loss, lr,
            )

    def on_epoch_end(self, trainer) -> None:
        if not trainer.is_main_process:
            return
        elapsed = time.perf_counter() - self._epoch_start
        summary = " | ".join(f"{k} {v:.4f}" for k, v in sorted(trainer.metrics.items()))
        logger.info("epoch %d done in %.1fs | %s", trainer.epoch, elapsed, summary)


class CheckpointCallback(Callback):
    """Save ``last.pt`` every epoch and ``best.pt`` on monitored improvement."""

    def __init__(
        self,
        dirpath: str | Path | None = None,
        monitor: str = "val/accuracy",
        mode: str = "max",
        save_last: bool = True,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.dirpath = Path(dirpath) if dirpath else None
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best = -math.inf if mode == "max" else math.inf
        self._warned = False

    def _improved(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def on_epoch_end(self, trainer) -> None:
        if not trainer.is_main_process:
            return
        dirpath = self.dirpath or (Path(trainer.output_dir) / "checkpoints")
        dirpath.mkdir(parents=True, exist_ok=True)

        if self.save_last:
            trainer.save_checkpoint(dirpath / "last.pt")

        value = trainer.metrics.get(self.monitor)
        if value is None:
            if not self._warned and trainer.metrics:
                logger.warning(
                    "CheckpointCallback: monitor %r not in metrics %s; "
                    "only saving last.pt", self.monitor, sorted(trainer.metrics),
                )
                self._warned = True
            return
        if self._improved(value):
            self.best = value
            trainer.save_checkpoint(dirpath / "best.pt")
            logger.info("New best %s: %.4f -> saved best.pt", self.monitor, value)


class EarlyStopping(Callback):
    """Stop training when the monitored metric stops improving.

    Metrics are rank-synchronized by the evaluators, so all DDP ranks make
    the same stop decision.
    """

    def __init__(
        self,
        monitor: str = "val/accuracy",
        mode: str = "max",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf if mode == "max" else math.inf
        self.counter = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def on_epoch_end(self, trainer) -> None:
        value = trainer.metrics.get(self.monitor)
        if value is None:
            return
        if self._improved(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True
                if trainer.is_main_process:
                    logger.info(
                        "Early stopping at epoch %d: %s did not improve for %d epochs "
                        "(best %.4f)", trainer.epoch, self.monitor, self.patience, self.best,
                    )
