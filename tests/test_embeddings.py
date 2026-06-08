import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from image_analytics.backbones.registry import build_backbone
from image_analytics.classification.models import ImageClassifier
from image_analytics.core.config import BackboneConfig
from image_analytics.serving.embeddings import extract_embeddings, save_embeddings


def _model_and_loader(n=8):
    bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
    model = ImageClassifier(bb, num_classes=4)
    ds = TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 4, (n,)))
    return model, DataLoader(ds, batch_size=4)


def test_extract_shapes():
    model, loader = _model_and_loader(8)
    vectors, labels = extract_embeddings(model, loader)
    assert vectors.shape == (8, model.backbone.feature_dim)
    assert labels.shape == (8,)


def test_save_npz(tmp_path):
    model, loader = _model_and_loader(8)
    vectors, labels = extract_embeddings(model, loader)
    path = save_embeddings(tmp_path / "emb.npz", vectors, labels=labels)
    assert path.exists()
    data = np.load(path)
    assert data["vectors"].shape == vectors.shape
    assert data["labels"].shape == (8,)


def test_save_parquet(tmp_path):
    import importlib.util

    if importlib.util.find_spec("pyarrow") is None:
        import pytest

        pytest.skip("pyarrow not installed")
    model, loader = _model_and_loader(8)
    vectors, labels = extract_embeddings(model, loader)
    path = save_embeddings(tmp_path / "emb.parquet", vectors, labels=labels)
    assert path.exists()
    import pandas as pd

    df = pd.read_parquet(path)
    assert len(df) == 8 and "label" in df.columns
