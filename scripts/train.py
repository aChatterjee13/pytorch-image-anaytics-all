#!/usr/bin/env python
"""Unified training entry point.

Usage:
    python scripts/train.py --config configs/classification/cifar10_resnet18.yaml
    python scripts/train.py --config <path> training.lr=1e-4 data.batch_size=64

Distributed:
    torchrun --nproc_per_node=4 scripts/train.py --config <path>
"""

from __future__ import annotations

import argparse

from image_analytics.core.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted config overrides, e.g. training.lr=1e-4",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)

    # Tasks are imported lazily so unrelated heavy dependencies don't load.
    if config.task == "classification":
        from image_analytics.classification.train import run
    else:
        raise SystemExit(
            f"Unknown task {config.task!r}. Available tasks: classification "
            f"(detection/segmentation/3d arrive in later phases — see EXPLORATION.md)"
        )

    metrics = run(config)
    print("\nFinal metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
