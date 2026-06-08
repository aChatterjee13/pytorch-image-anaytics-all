import pytest
import torch

import image_analytics.segmentation.train  # noqa: F401  (register siamese_unet / temporal)
from image_analytics.backbones.registry import build_backbone
from image_analytics.core.config import BackboneConfig, config_from_dict
from image_analytics.core.registry import MODELS
from image_analytics.data.datasets import build_dataset
from image_analytics.core.config import DataConfig


def _pyramid_backbone(in_channels=3):
    return build_backbone(BackboneConfig(
        name="resnet18", pretrained=False, in_channels=in_channels, features_only=True,
        kwargs={"out_indices": (0, 1, 2, 3, 4)},
    ))


class TestSyntheticChangeDataset:
    def test_sample_shape(self):
        ds = build_dataset(DataConfig(dataset="synthetic_change", kwargs={"size": 8, "image_size": 64}), split="train")
        img, mask = ds[0]
        assert img.shape == (6, 64, 64)        # t0 + t1 channel-concatenated
        assert mask.shape == (64, 64)
        assert set(int(v) for v in mask.unique()) <= {0, 1}
        assert ds.num_classes == 2

    def test_deterministic(self):
        ds = build_dataset(DataConfig(dataset="synthetic_change", kwargs={"size": 4, "image_size": 48}), split="train")
        assert torch.equal(ds[1][0], ds[1][0]) and torch.equal(ds[1][1], ds[1][1])


class TestSiameseUNet:
    def test_forward_shape(self):
        model = MODELS.build("siamese_unet", backbone=_pyramid_backbone(3), num_classes=2).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 6, 64, 64))
        assert out.shape == (2, 2, 64, 64)

    def test_pooled_backbone_rejected(self):
        bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
        with pytest.raises(ValueError, match="pyramid"):
            MODELS.build("siamese_unet", backbone=bb, num_classes=2)

    def test_gradients_flow(self):
        model = MODELS.build("siamese_unet", backbone=_pyramid_backbone(3), num_classes=2).train()
        model(torch.randn(2, 6, 64, 64)).mean().backward()
        assert all(p.grad is not None for p in model.head.parameters())


class TestTemporal:
    def test_dataset_clip_shape(self):
        ds = build_dataset(DataConfig(dataset="synthetic_temporal", kwargs={"size": 8, "image_size": 48, "num_frames": 4}), split="train")
        clip, label = ds[0]
        assert clip.shape == (3, 4, 48, 48)
        assert isinstance(label, int) and 0 <= label < 4

    @pytest.mark.parametrize("pool", ["mean", "max", "attention"])
    def test_temporal_classifier(self, pool):
        bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
        model = MODELS.build("temporal_classifier", backbone=bb, num_classes=4, pool=pool).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 4, 48, 48))
        assert out.shape == (2, 4)


class TestChangeDetectionEndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict({
            "task": "segmentation", "experiment_name": "cd_smoke",
            "seed": 0, "output_dir": str(tmp_path),
            "model": {"name": "siamese_unet", "num_classes": 2,
                      "backbone": {"name": "resnet18", "pretrained": False, "in_channels": 3,
                                   "features_only": True},
                      "kwargs": {"decoder_channels": [64, 32, 16, 8, 8]}},
            "data": {"dataset": "synthetic_change", "image_size": 64, "batch_size": 8,
                     "num_workers": 0, "normalize": "none", "kwargs": {"size": 16, "image_size": 64}},
            "training": {"epochs": 1, "lr": 1e-3, "scheduler": "none", "loss": "ce_dice",
                         "device": "cpu", "log_interval": 0, "monitor": "val/mIoU"},
        })
        from image_analytics.segmentation.train import run

        metrics = run(config)
        assert "train/loss" in metrics and "val/mIoU" in metrics
        assert (tmp_path / "cd_smoke" / "checkpoints" / "last.pt").exists()

    def test_config_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/segmentation/change_detection_shapes.yaml")
        assert config.model.name == "siamese_unet"


@pytest.mark.slow
class TestSiameseLearns:
    def test_overfit(self):
        torch.manual_seed(0)
        ds = build_dataset(DataConfig(dataset="synthetic_change", kwargs={"size": 8, "image_size": 64}), split="train")
        from image_analytics.data.transforms.segmentation import build_segmentation_transforms

        tf = build_segmentation_transforms(64, train=False, normalize="none")
        imgs = torch.stack([tf(*ds[i])[0] for i in range(8)])
        masks = torch.stack([tf(*ds[i])[1] for i in range(8)])
        model = MODELS.build("siamese_unet", backbone=_pyramid_backbone(3), num_classes=2,
                             decoder_channels=(64, 32, 16, 8, 8)).train()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        crit = torch.nn.CrossEntropyLoss()
        first = None
        for _ in range(60):
            loss = crit(model(imgs), masks)
            opt.zero_grad(); loss.backward(); opt.step()
            first = first if first is not None else float(loss)
        assert float(loss) < first * 0.5
