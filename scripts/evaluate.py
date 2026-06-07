#!/usr/bin/env python
"""Evaluate a trained checkpoint on the validation split.

Usage:
    python scripts/evaluate.py --config outputs/<exp>/config.yaml \\
        --checkpoint outputs/<exp>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging

from image_analytics.classification.models import build_model
from image_analytics.classification.train import build_dataloaders
from image_analytics.core.config import load_config
from image_analytics.core.evaluator import ClassificationEvaluator, MultiLabelEvaluator
from image_analytics.core.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument(
        "overrides", nargs="*", help="Dotted config overrides, e.g. data.batch_size=64"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config, overrides=args.overrides)
    if config.task != "classification":
        raise SystemExit(f"Unknown task {config.task!r}")

    _, val_loader = build_dataloaders(config, distributed=False)
    model = build_model(config.model)

    if getattr(model, "is_multilabel", False):
        evaluator = MultiLabelEvaluator(num_labels=config.model.num_classes)
    else:
        topk = (1, 5) if config.model.num_classes > 5 else (1,)
        evaluator = ClassificationEvaluator(config.model.num_classes, topk=topk)

    trainer = Trainer(model, evaluator=evaluator, device=config.training.device)
    trainer.load_checkpoint(args.checkpoint, resume=False)

    metrics = trainer.validate(val_loader)
    print("\nValidation metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
