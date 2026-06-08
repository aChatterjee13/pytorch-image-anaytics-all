import pytest
import torch

import image_analytics.detection_3d  # noqa: F401  (register models)
from image_analytics.core.config import config_from_dict
from image_analytics.core.registry import MODELS
from image_analytics.detection_3d.train import run


def _model(**kwargs):
    return MODELS.build("pointpillars", num_classes=1, **kwargs)


def _batch(bs=2, n=2048):
    points = torch.rand(bs, n, 3) * 10 - 5
    targets = [
        {"boxes_3d": torch.tensor([[0.0, 0, 0.5, 1.5, 1.5, 1.0, 0.0]]),
         "labels": torch.tensor([0])}
        for _ in range(bs)
    ]
    return points, targets


class TestPointPillars:
    def test_training_loss_dict(self):
        model = _model().train()
        losses = model(*_batch())
        assert set(losses) == {"loss", "loss_cls", "loss_reg"}
        for v in losses.values():
            assert torch.isfinite(v)
        assert losses["loss"].requires_grad

    def test_eval_predictions(self):
        model = _model().eval()
        points, _ = _batch()
        with torch.no_grad():
            preds = model(points)
        assert len(preds) == 2
        for p in preds:
            assert set(p) == {"boxes_3d", "scores", "labels"}
            assert p["boxes_3d"].shape[1] == 7

    def test_empty_targets(self):
        model = _model().train()
        points, _ = _batch()
        empty = [{"boxes_3d": torch.zeros(0, 7), "labels": torch.zeros(0, dtype=torch.int64)}] * 2
        assert torch.isfinite(model(points, empty)["loss"])

    def test_config_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/pointcloud/pointpillars_synthetic.yaml")
        assert config.model.name == "pointpillars" and config.task == "pointcloud"


class TestPointCloudEndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict({
            "task": "pointcloud", "experiment_name": "pp_smoke",
            "seed": 0, "output_dir": str(tmp_path),
            "model": {"name": "pointpillars", "num_classes": 1,
                      "kwargs": {"point_cloud_range": [-5, -5, -1, 5, 5, 3], "voxel_size": [0.25, 0.25]}},
            "data": {"dataset": "synthetic_pointcloud_det", "batch_size": 8, "num_workers": 0,
                     "kwargs": {"size": 16, "num_points": 1024}},
            "training": {"epochs": 1, "lr": 1e-3, "scheduler": "none",
                         "device": "cpu", "log_interval": 0, "monitor": "val/mAP_3d"},
        })
        metrics = run(config)
        assert "train/loss" in metrics and "val/mAP_3d" in metrics
        assert (tmp_path / "pp_smoke" / "checkpoints" / "last.pt").exists()


@pytest.mark.slow
class TestPointPillarsLearns:
    def test_overfit_single_batch(self):
        torch.manual_seed(0)
        model = _model().train()
        points, targets = _batch()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        first = None
        for _ in range(20):
            losses = model(points, targets)
            opt.zero_grad(); losses["loss"].backward(); opt.step()
            first = first if first is not None else float(losses["loss"])
        assert float(losses["loss"]) < first * 0.5
