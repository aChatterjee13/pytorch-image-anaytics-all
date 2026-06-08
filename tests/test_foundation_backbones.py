import pytest
import torch

import image_analytics.backbones  # noqa: F401  (register satellite backbones)
from image_analytics.backbones import BACKBONES


def test_registered():
    assert "satmae_base" in BACKBONES
    assert "prithvi_100m" in BACKBONES


class TestSatMAE:
    def test_forward_pooled(self):
        model = BACKBONES.build(
            "satmae_base", img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=2,
        ).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 10, 32, 32))  # 10-band default grouping
        assert out.shape == (2, 64)
        assert model.feature_dim == 64

    def test_custom_band_groups(self):
        model = BACKBONES.build(
            "satmae_base", img_size=32, patch_size=16, embed_dim=64, depth=1, num_heads=2,
            band_groups=[[0, 1], [2, 3]],
        ).eval()
        with torch.no_grad():
            out = model(torch.randn(1, 4, 32, 32))
        assert out.shape == (1, 64)

    def test_features_only_rejected(self):
        with pytest.raises(ValueError, match="pooled"):
            BACKBONES.build("satmae_base", features_only=True)

    def test_classifier_integration(self):
        from image_analytics.classification.models import ImageClassifier

        bb = BACKBONES.build("satmae_base", img_size=32, patch_size=16,
                             embed_dim=64, depth=1, num_heads=2)
        model = ImageClassifier(bb, num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 10, 32, 32))
        assert out.shape == (2, 10)


class TestPrithvi:
    def test_forward_temporal(self):
        model = BACKBONES.build(
            "prithvi_100m", in_channels=6, img_size=32, patch_size=16,
            num_frames=3, embed_dim=64, depth=2, num_heads=2,
        ).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 6, 3, 32, 32))
        assert out.shape == (2, 64)

    def test_rejects_non_temporal_input(self):
        model = BACKBONES.build(
            "prithvi_100m", in_channels=6, img_size=32, patch_size=16,
            num_frames=3, embed_dim=64, depth=1, num_heads=2,
        )
        with pytest.raises(ValueError, match="B, C, T, H, W"):
            model(torch.randn(2, 6, 32, 32))
