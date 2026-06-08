"""Phase 2 leftovers: letterbox transform, PAFPN neck, HF detector wrappers."""

import pytest
import torch
from torchvision import tv_tensors

from image_analytics.core.config import BackboneConfig, ModelConfig
from image_analytics.core.registry import MODELS, NECKS
from image_analytics.data.transforms.detection import build_detection_transforms
from image_analytics.detection.train import build_detection_model


class TestLetterbox:
    def test_square_canvas_preserves_aspect(self):
        tf = build_detection_transforms(64, train=False, normalize="none", letterbox=True)
        image = torch.rand(3, 40, 80)  # 1:2 aspect
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor([[10.0, 5, 70, 35]]), format="XYXY", canvas_size=(40, 80)
            ),
            "labels": torch.tensor([0]),
        }
        out_img, out_t = tf(image, target)
        assert out_img.shape == (3, 64, 64)
        # longer side (80) scales by 64/80 = 0.8; box scales accordingly
        assert out_t["boxes"][0].tolist() == pytest.approx([8.0, 4.0, 56.0, 28.0])

    def test_square_input_matches_plain_resize_shape(self):
        tf = build_detection_transforms(48, train=False, normalize="none", letterbox=True)
        image = torch.rand(3, 32, 32)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.tensor([[4.0, 4, 20, 20]]), format="XYXY", canvas_size=(32, 32)
            ),
            "labels": torch.tensor([0]),
        }
        out_img, _ = tf(image, target)
        assert out_img.shape == (3, 48, 48)


class TestPAFPN:
    def test_output_levels_and_channels(self):
        neck = NECKS.build(
            "pafpn", in_channels_list=[64, 128, 256], out_channels=32, extra_levels="p6p7"
        )
        feats = [torch.rand(2, 64, 32, 32), torch.rand(2, 128, 16, 16), torch.rand(2, 256, 8, 8)]
        outs = neck(feats)
        assert len(outs) == 5  # 3 + P6 + P7
        assert all(o.shape[1] == 32 for o in outs)
        assert neck.num_levels == 5

    def test_retinanet_with_pafpn(self):
        from image_analytics.detection.necks.pafpn import PAFPN

        model = build_detection_model(
            ModelConfig(
                name="retinanet", num_classes=3,
                backbone=BackboneConfig(name="resnet18", pretrained=False, features_only=True),
                kwargs=dict(fpn_channels=32, num_convs=1, neck="pafpn",
                            anchor_base_sizes=(16, 32, 64, 128, 256)),
            )
        )
        assert isinstance(model.fpn, PAFPN)
        images = torch.rand(2, 3, 64, 64)
        targets = [{"boxes": torch.tensor([[8.0, 8, 30, 30]]), "labels": torch.tensor([1])}] * 2
        assert torch.isfinite(model.train()(images, targets)["loss"])


class TestHFDetectorWrappers:
    def test_registered(self):
        assert "deformable_detr" in MODELS
        assert "rt_detr" in MODELS

    def test_build_skips_backbone(self, monkeypatch):
        """Wrapper detectors must not build a (timm) pyramid backbone."""
        import image_analytics.detection.train as dt

        built = {"backbone": False}

        def _fail(*a, **k):
            built["backbone"] = True
            raise AssertionError("backbone should not be built for wrapper detectors")

        captured = {}

        def fake_build(name, /, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setattr(dt, "build_backbone", _fail)
        monkeypatch.setattr(dt.MODELS, "build", fake_build)

        dt.build_detection_model(
            ModelConfig(name="deformable_detr", num_classes=5,
                        backbone=BackboneConfig(name="resnet50"))
        )
        assert built["backbone"] is False
        assert captured["name"] == "deformable_detr"
        assert captured["kwargs"]["num_classes"] == 5
