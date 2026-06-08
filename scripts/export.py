#!/usr/bin/env python
"""Export a trained checkpoint to ONNX with a mandatory parity gate.

    python scripts/export.py --config outputs/<exp>/config.yaml \\
        --checkpoint outputs/<exp>/checkpoints/best.pt --output model.onnx

Classifiers/segmenters export whole; RetinaNet-style detectors export the raw
heads graph (decode/NMS stay in Python — see serving/onnx_export.py).
"""

from __future__ import annotations

import argparse

import torch

from image_analytics.core.config import load_config
from image_analytics.serving.infer_utils import build_task_model, load_checkpoint_into
from image_analytics.serving.onnx_export import build_exportable, export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument("--output", "-o", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.task == "pointcloud":
        raise SystemExit("ONNX export for point-cloud models is not supported via this CLI")

    model = build_task_model(config)
    load_checkpoint_into(model, args.checkpoint)
    exportable = build_exportable(model, config.task)

    in_channels = config.model.backbone.in_channels
    size = config.data.image_size
    example = torch.randn(1, in_channels, size, size)

    path = export_onnx(exportable, example, args.output, opset=args.opset, atol=args.atol)
    print(f"Exported and parity-checked (|Δ| < {args.atol:g}): {path}")


if __name__ == "__main__":
    main()
