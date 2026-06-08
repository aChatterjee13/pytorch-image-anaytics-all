import pytest

from image_analytics.core.config import config_from_dict
from image_analytics.core.mlflow import _flatten, maybe_mlflow_callback


def _config(tmp_path, mlflow=False):
    return config_from_dict(
        {
            "task": "classification",
            "experiment_name": "mlflow_smoke",
            "seed": 0,
            "output_dir": str(tmp_path),
            "model": {"name": "classifier", "num_classes": 4,
                      "backbone": {"name": "resnet18", "pretrained": False}},
            "data": {
                "dataset": "fake", "root": str(tmp_path / "data"),
                "image_size": 32, "batch_size": 16, "num_workers": 0,
                "kwargs": {"size": 32, "image_size": [3, 32, 32], "num_classes": 4},
            },
            "training": {
                "epochs": 1, "lr": 1e-3, "scheduler": "none",
                "device": "cpu", "log_interval": 0, "mlflow": mlflow,
            },
        }
    )


def test_flatten_dotted_keys():
    flat = _flatten({"training": {"lr": 0.1, "epochs": 3}, "task": "x"})
    assert flat["training.lr"] == 0.1
    assert flat["training.epochs"] == 3
    assert flat["task"] == "x"


def test_callback_off_by_default(tmp_path):
    assert maybe_mlflow_callback(_config(tmp_path, mlflow=False)) == []
    assert len(maybe_mlflow_callback(_config(tmp_path, mlflow=True))) == 1


def test_run_round_trip(tmp_path):
    """A training run with training.mlflow logs params + metrics to a file store."""
    mlflow = pytest.importorskip("mlflow")
    from image_analytics.classification.train import run

    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    run(_config(tmp_path, mlflow=True))

    exp = mlflow.get_experiment_by_name("mlflow_smoke")
    assert exp is not None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.task"] == "classification"
    assert "metrics.train/loss" in row.index
    assert "metrics.val/accuracy" in row.index
