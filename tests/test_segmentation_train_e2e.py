import torch

from image_analytics.core.config import (
    BackboneConfig,
    ModelConfig,
    config_from_dict,
    load_config,
)
from image_analytics.segmentation.train import build_segmentation_model, run


def _smoke_config(tmp_path, name="unet", **model_kwargs):
    return config_from_dict(
        {
            "task": "segmentation",
            "experiment_name": "seg_smoke",
            "seed": 0,
            "output_dir": str(tmp_path),
            "model": {
                "name": name,
                "num_classes": 4,
                "backbone": {"name": "resnet18", "pretrained": False},
                "kwargs": model_kwargs,
            },
            "data": {
                "dataset": "synthetic_shapes_seg",
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
                "loss": "ce_dice",
            },
        }
    )


class TestBuildModel:
    def test_unet_forces_pyramid_and_levels(self):
        model = build_segmentation_model(
            ModelConfig(
                name="unet", num_classes=4,
                backbone=BackboneConfig(name="resnet18", pretrained=False),
            )
        )
        assert model.backbone.features_only
        assert len(model.decoder) == 5  # one decoder block per encoder level

    def test_deeplab_sets_output_stride(self):
        model = build_segmentation_model(
            ModelConfig(
                name="deeplabv3plus", num_classes=4,
                backbone=BackboneConfig(name="resnet18", pretrained=False),
            )
        )
        # output_stride=16 encoder exposes [low (stride4), high (stride16)]
        assert model.backbone.features_only
        with torch.no_grad():
            y = model.eval()(torch.randn(2, 3, 64, 64))
        assert y.shape == (2, 4, 64, 64)


class TestEndToEnd:
    def test_run_smoke(self, tmp_path):
        metrics = run(_smoke_config(tmp_path, decoder_channels=[64, 32, 16, 8, 8]))
        assert "train/loss" in metrics
        assert "val/mIoU" in metrics and "val/dice" in metrics
        assert "val/pixel_accuracy" in metrics
        assert (tmp_path / "seg_smoke" / "checkpoints" / "last.pt").exists()
        assert (tmp_path / "seg_smoke" / "config.yaml").exists()

    def test_checked_in_configs_parse(self):
        for path in (
            "configs/segmentation/unet_shapes.yaml",
            "configs/segmentation/deeplabv3plus_shapes.yaml",
            "configs/segmentation/smoke.yaml",
            "configs/segmentation/segformer_shapes.yaml",
        ):
            config = load_config(path)
            assert config.task == "segmentation"
            assert config.model.num_classes == 4
