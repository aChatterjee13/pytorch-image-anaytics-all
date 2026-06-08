#!/usr/bin/env python
"""Extract backbone embeddings over a dataset's val split -> npz/parquet.

    python scripts/embed.py --config outputs/<exp>/config.yaml \\
        --checkpoint outputs/<exp>/checkpoints/best.pt \\
        --output embeddings.parquet [--faiss index.faiss]

The feature-store on-ramp: vectors are ``forward_features`` outputs; an optional
FAISS inner-product index enables similarity search.
"""

from __future__ import annotations

import argparse

from image_analytics.classification.train import build_dataloaders
from image_analytics.core.config import load_config
from image_analytics.core.trainer import resolve_device
from image_analytics.serving.embeddings import (
    build_faiss_index,
    extract_embeddings,
    save_embeddings,
)
from image_analytics.serving.infer_utils import build_task_model, load_checkpoint_into


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument("--output", "-o", default="embeddings.npz")
    parser.add_argument("--faiss", default=None, help="optional path to write a FAISS index")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.task != "classification":
        raise SystemExit("embed.py extracts classification backbone features")

    device = str(resolve_device(config.training.device))
    model = build_task_model(config)
    load_checkpoint_into(model, args.checkpoint, device=device)

    _, val_loader = build_dataloaders(config, distributed=False)
    vectors, labels = extract_embeddings(model, val_loader, device=device)
    save_embeddings(args.output, vectors, labels=labels)
    print(f"Saved {vectors.shape[0]} x {vectors.shape[1]} embeddings to {args.output}")

    if args.faiss:
        import faiss

        index = build_faiss_index(vectors)
        faiss.write_index(index, args.faiss)
        print(f"Wrote FAISS index ({index.ntotal} vectors) to {args.faiss}")


if __name__ == "__main__":
    main()
