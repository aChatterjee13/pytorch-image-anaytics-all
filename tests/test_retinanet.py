import pytest
import torch

from image_analytics.core.config import (
    BackboneConfig,
    ModelConfig,
    NeckConfig,
    config_from_dict,
)
from image_analytics.detection.train import build_detection_model, run


def tiny_retinanet_config(**kwargs):
    """Smallest practical RetinaNet for CPU tests."""
    defaults = dict(
        fpn_channels=32,
        num_convs=1,
        anchor_base_sizes=(16, 32, 64, 128, 256),
        score_thresh=0.05,
    )
    defaults.update(kwargs)
    return ModelConfig(
        name="retinanet",
        num_classes=3,
        backbone=BackboneConfig(name="resnet18", pretrained=False, features_only=True),
        kwargs=defaults,
    )


def make_batch(batch_size=2, image_size=64, device="cpu"):
    images = torch.rand(batch_size, 3, image_size, image_size, device=device)
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8, 30, 30], [35.0, 35, 60, 58]], device=device),
            "labels": torch.tensor([0, 2], device=device),
        }
        for _ in range(batch_size)
    ]
    return images, targets


class TestRetinaNetForward:
    def test_training_returns_finite_loss_dict(self):
        model = build_detection_model(tiny_retinanet_config()).train()
        images, targets = make_batch()
        losses = model(images, targets)
        assert set(losses) == {"loss", "loss_cls", "loss_reg"}
        for value in losses.values():
            assert torch.isfinite(value)
        assert losses["loss"].requires_grad

    def test_training_without_targets_raises(self):
        model = build_detection_model(tiny_retinanet_config()).train()
        images, _ = make_batch()
        with pytest.raises(ValueError, match="targets"):
            model(images)

    def test_empty_targets_supported(self):
        model = build_detection_model(tiny_retinanet_config()).train()
        images, _ = make_batch()
        empty = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(2)
        ]
        losses = model(images, empty)
        assert torch.isfinite(losses["loss"])
        assert losses["loss_reg"].item() == pytest.approx(0.0)

    def test_eval_returns_prediction_dicts(self):
        model = build_detection_model(tiny_retinanet_config()).eval()
        images, _ = make_batch()
        with torch.no_grad():
            predictions = model(images)
        assert len(predictions) == 2
        for pred in predictions:
            assert set(pred) == {"boxes", "scores", "labels"}
            n = len(pred["scores"])
            assert pred["boxes"].shape == (n, 4)
            assert pred["labels"].shape == (n,)
            assert n <= 100  # detections_per_img cap
            if n:
                # Boxes clipped to image bounds
                assert pred["boxes"].min() >= 0
                assert pred["boxes"].max() <= 64

    def test_neck_config_controls_fpn_width(self):
        config = tiny_retinanet_config()
        config.kwargs.pop("fpn_channels")
        config.neck = NeckConfig(name="fpn", out_channels=48)
        model = build_detection_model(config)
        assert model.fpn.out_channels == 48

    def test_pooled_backbone_rejected(self):
        config = tiny_retinanet_config()
        config.backbone = BackboneConfig(name="resnet18", pretrained=False)
        # build_detection_model forces pyramid mode, so this must still work
        model = build_detection_model(config)
        assert model.backbone.features_only


class TestRetinaNetLearns:
    def test_overfit_single_batch(self):
        """The full loss pipeline must be optimizable end-to-end."""
        torch.manual_seed(0)
        config = tiny_retinanet_config()
        config.backbone = BackboneConfig(
            name="resnet10t", pretrained=False, features_only=True,
            kwargs={"out_indices": (2, 3, 4)},
        )
        model = build_detection_model(config).train()
        images, targets = make_batch(batch_size=2, image_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        first = None
        for _ in range(30):
            losses = model(images, targets)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            if first is None:
                first = float(losses["loss"])
        last = float(losses["loss"])
        assert last < first * 0.5, f"loss did not halve: {first:.4f} -> {last:.4f}"


class TestDetectionEndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict(
            {
                "task": "detection",
                "experiment_name": "det_smoke",
                "seed": 0,
                "output_dir": str(tmp_path),
                "model": {
                    "name": "retinanet",
                    "num_classes": 3,
                    "backbone": {
                        "name": "resnet18",
                        "pretrained": False,
                        "features_only": True,
                    },
                    "neck": {"name": "fpn", "out_channels": 32},
                    "kwargs": {
                        "num_convs": 1,
                        "anchor_base_sizes": [16, 32, 64, 128, 256],
                    },
                },
                "data": {
                    "dataset": "synthetic_shapes",
                    "image_size": 64,
                    "batch_size": 8,
                    "num_workers": 0,
                    "kwargs": {"size": 24, "image_size": 64},
                },
                "training": {
                    "epochs": 1,
                    "optimizer": "adamw",
                    "lr": 1e-3,
                    "scheduler": "none",
                    "device": "cpu",
                    "log_interval": 0,
                    "monitor": "val/mAP",
                },
            }
        )
        metrics = run(config)
        assert "train/loss" in metrics
        assert "val/mAP" in metrics and "val/mAP50" in metrics
        assert (tmp_path / "det_smoke" / "checkpoints" / "last.pt").exists()

    def test_checked_in_detection_configs_parse(self):
        from image_analytics.core.config import load_config

        for path in (
            "configs/detection/retinanet_shapes.yaml",
            "configs/detection/retinanet_resnet50_coco.yaml",
        ):
            config = load_config(path)
            assert config.task == "detection"
            assert config.model.backbone.features_only
