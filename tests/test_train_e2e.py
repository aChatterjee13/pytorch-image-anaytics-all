"""End-to-end smoke test: full config -> data -> model -> trainer pipeline."""

import pytest

from image_analytics.classification.train import run
from image_analytics.core.config import config_from_dict, load_config


def smoke_config(tmp_path, **model_overrides):
    model = {
        "name": "classifier",
        "num_classes": 4,
        "backbone": {"name": "resnet18", "pretrained": False},
    }
    model.update(model_overrides)
    return config_from_dict(
        {
            "task": "classification",
            "experiment_name": "e2e_smoke",
            "seed": 0,
            "output_dir": str(tmp_path),
            "model": model,
            "data": {
                "dataset": "fake",
                "root": str(tmp_path / "data"),
                "image_size": 32,
                "batch_size": 16,
                "num_workers": 0,
                "augment": "default",
                "normalize": "imagenet",
                "kwargs": {"size": 32, "image_size": [3, 32, 32], "num_classes": 4},
            },
            "training": {
                "epochs": 1,
                "optimizer": "adamw",
                "lr": 1e-3,
                "scheduler": "cosine",
                "warmup_epochs": 0,
                "device": "cpu",
                "log_interval": 0,
            },
        }
    )


def test_end_to_end_run(tmp_path):
    metrics = run(smoke_config(tmp_path))
    assert "train/loss" in metrics
    assert "val/accuracy" in metrics

    exp_dir = tmp_path / "e2e_smoke"
    assert (exp_dir / "config.yaml").exists()
    assert (exp_dir / "checkpoints" / "last.pt").exists()
    assert (exp_dir / "checkpoints" / "best.pt").exists()

    # The saved config is itself loadable (reproducibility contract)
    reloaded = load_config(exp_dir / "config.yaml")
    assert reloaded.model.num_classes == 4


def test_end_to_end_balanced_sampling(tmp_path):
    config = smoke_config(tmp_path)
    config.data.balanced_sampling = True
    metrics = run(config)
    assert "val/accuracy" in metrics


def test_smoke_config_file_parses():
    """The checked-in smoke config must stay valid."""
    config = load_config("configs/classification/smoke_fake.yaml")
    assert config.task == "classification"
    assert config.model.backbone.pretrained is False


def test_checked_in_configs_parse():
    for path in (
        "configs/classification/cifar10_resnet18.yaml",
        "configs/classification/multispectral_resnet50.yaml",
    ):
        config = load_config(path)
        assert config.task == "classification"
