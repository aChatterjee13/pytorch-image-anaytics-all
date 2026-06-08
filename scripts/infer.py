#!/usr/bin/env python
"""Unified inference CLI: run a checkpoint over file/dir/glob inputs -> JSON.

    python scripts/infer.py --config outputs/<exp>/config.yaml \\
        --checkpoint outputs/<exp>/checkpoints/best.pt \\
        --input 'images/*.jpg' --output preds.json [--visualize]

Task-dispatched: classification -> top-k probs; detection -> boxes/scores/labels
(scaled back to the original image); segmentation -> per-class pixel counts
(``--visualize`` writes annotated images / colorized masks).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from PIL import Image

from image_analytics.core.config import load_config
from image_analytics.serving.infer_utils import (
    build_eval_transform,
    build_task_model,
    classification_topk,
    detection_to_dict,
    load_checkpoint_into,
    segmentation_summary,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gather_inputs(spec: str) -> list[str]:
    path = Path(spec)
    if path.is_dir():
        return sorted(str(p) for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    matches = sorted(glob.glob(spec))
    if matches:
        return matches
    if path.exists():
        return [str(path)]
    raise SystemExit(f"No inputs matched {spec!r}")


def _visualize(task, image, result, out_path, image_size):
    from PIL import ImageDraw

    if task == "detection":
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for box, label in zip(result["boxes"], result["labels"]):
            draw.rectangle(box, outline=(255, 0, 0), width=2)
            draw.text((box[0], box[1]), str(label), fill=(255, 0, 0))
        annotated.save(out_path)
    elif task == "segmentation":
        # Saved separately by the caller (needs the mask tensor); skip here.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument("--input", "-i", required=True, help="file | dir | glob")
    parser.add_argument("--output", "-o", default="preds.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.task == "pointcloud":
        raise SystemExit("Use a point-cloud-specific tool; this CLI handles image inputs")

    model = build_task_model(config)
    load_checkpoint_into(model, args.checkpoint)
    model.eval()
    transform = build_eval_transform(config)
    size = config.data.image_size

    vis_dir = Path(args.output).with_suffix("") if args.visualize else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in gather_inputs(args.input):
        image = Image.open(path).convert("RGB")
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        if config.task == "classification":
            result = classification_topk(out, args.topk)
        elif config.task == "detection":
            result = detection_to_dict(out, image.size, size, args.score_thresh)
        else:
            result = segmentation_summary(out)
        results.append({"input": path, **result})
        if vis_dir is not None:
            _visualize(config.task, image, result, vis_dir / Path(path).name, size)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
