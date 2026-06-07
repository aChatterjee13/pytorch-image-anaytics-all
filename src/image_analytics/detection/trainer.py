"""Detection trainer: loss-dict training, prediction-list evaluation.

Detectors follow the shared interface (train mode: ``model(images, targets)
-> loss dict``; eval mode: ``model(images) -> prediction list``), so only the
batch handling and step hooks differ from the base Trainer — the loop, AMP,
DDP, callbacks, and checkpointing are inherited unchanged.
"""

from __future__ import annotations

import torch

from image_analytics.core.trainer import Trainer


class DetectionTrainer(Trainer):
    def _move_batch(self, batch):
        images, targets = batch
        images = images.to(self.device, non_blocking=True)
        targets = [
            {
                key: value.to(self.device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in target.items()
            }
            for target in targets
        ]
        return images, targets

    def training_step(self, batch) -> torch.Tensor:
        images, targets = self._move_batch(batch)
        with self._autocast():
            losses = self.model(images, targets)
        return losses["loss"]

    def eval_step(self, batch):
        images, targets = self._move_batch(batch)
        with self._autocast():
            predictions = self.model(images)
        return predictions, targets, None
