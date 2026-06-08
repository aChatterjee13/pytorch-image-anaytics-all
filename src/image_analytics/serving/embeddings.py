"""Embedding extraction — the feature-store on-ramp.

Batches ``forward_features`` (backbone embeddings) over a dataloader and saves
``(ids, labels, vectors)`` to ``.npz`` or ``.parquet``; optionally builds a
FAISS inner-product index for similarity search (``faiss-cpu``, optional).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def extract_embeddings(model, loader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vectors (N, D), labels (N,))`` of backbone features.

    Uses ``model.forward_features`` when available (our classifiers), else the
    ``backbone``, else the model itself; multi-dim features are flattened.
    """
    model.eval().to(device)
    if hasattr(model, "forward_features"):
        feature_fn = model.forward_features
    elif hasattr(model, "backbone"):
        feature_fn = model.backbone
    else:
        feature_fn = model

    vectors, labels = [], []
    for inputs, targets in loader:
        feat = feature_fn(inputs.to(device))
        if feat.dim() > 2:
            feat = feat.flatten(1)
        vectors.append(feat.cpu())
        labels.append(targets if torch.is_tensor(targets) else torch.as_tensor(targets))
    return torch.cat(vectors).numpy(), torch.cat(labels).numpy()


def save_embeddings(path, vectors: np.ndarray, labels=None, ids=None) -> Path:
    """Save embeddings to ``.npz`` or ``.parquet`` (inferred from the suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        np.savez(
            path, vectors=vectors,
            labels=np.asarray(labels) if labels is not None else np.array([]),
            ids=np.asarray(ids) if ids is not None else np.array([]),
        )
    elif path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pandas is required for parquet output") from exc
        df = pd.DataFrame(vectors, columns=[f"f{i}" for i in range(vectors.shape[1])])
        if labels is not None:
            df.insert(0, "label", np.asarray(labels))
        if ids is not None:
            df.insert(0, "id", np.asarray(ids))
        df.to_parquet(path)
    else:
        raise ValueError(f"Unsupported embeddings format {path.suffix!r}; use .npz or .parquet")
    return path


def build_faiss_index(vectors: np.ndarray, normalize: bool = True):
    """Build a ``faiss.IndexFlatIP`` (cosine similarity when ``normalize``)."""
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "faiss is required to build a similarity index. "
            "Install it with: pip install faiss-cpu"
        ) from exc
    vectors = np.ascontiguousarray(vectors.astype("float32"))
    if normalize:
        faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index
