import torch

from image_analytics.backbones.multichannel import GroupedBandStem, GroupedStemBackbone
from image_analytics.backbones.registry import build_backbone
from image_analytics.core.config import BackboneConfig

GROUPS = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]


class TestGroupedBandStem:
    def test_fuses_to_out_channels(self):
        stem = GroupedBandStem(GROUPS, out_channels=3, stem_channels=8)
        out = stem(torch.randn(2, 10, 32, 32))
        assert out.shape == (2, 3, 32, 32)

    def test_gradients_flow(self):
        stem = GroupedBandStem(GROUPS, out_channels=3)
        stem(torch.randn(2, 10, 32, 32)).sum().backward()
        assert all(p.grad is not None for p in stem.fuse.parameters())


class TestBuildBackboneWiring:
    def test_pooled_wrap(self):
        bb = build_backbone(BackboneConfig(
            name="resnet18", pretrained=False, in_channels=10,
            kwargs={"stem_band_groups": GROUPS},
        ))
        assert isinstance(bb, GroupedStemBackbone)
        out = bb(torch.randn(2, 10, 64, 64))
        assert out.dim() == 2 and out.shape[1] == bb.feature_dim

    def test_pyramid_wrap_preserves_feature_channels(self):
        bb = build_backbone(BackboneConfig(
            name="resnet18", pretrained=False, in_channels=8, features_only=True,
            kwargs={"stem_band_groups": [[0, 1, 2, 3], [4, 5, 6, 7]], "out_indices": (2, 3, 4)},
        ))
        assert isinstance(bb, GroupedStemBackbone) and bb.features_only
        feats = bb(torch.randn(1, 8, 64, 64))
        assert len(feats) == 3
        assert bb.feature_channels == bb.backbone.feature_channels

    def test_classifier_over_grouped_stem(self):
        from image_analytics.classification.models import ImageClassifier

        bb = build_backbone(BackboneConfig(
            name="resnet18", pretrained=False, in_channels=10,
            kwargs={"stem_band_groups": GROUPS},
        ))
        model = ImageClassifier(bb, num_classes=7).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 10, 64, 64))
        assert out.shape == (2, 7)
