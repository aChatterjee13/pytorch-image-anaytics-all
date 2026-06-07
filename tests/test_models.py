import pytest
import torch

from image_analytics.classification import (
    ImageClassifier,
    MultiLabelImageClassifier,
    build_model,
)
from image_analytics.core.config import BackboneConfig, ModelConfig


def small_model_config(name="classifier", num_classes=10, **backbone_kwargs):
    backbone_kwargs.setdefault("name", "resnet18")
    backbone_kwargs.setdefault("pretrained", False)
    return ModelConfig(
        name=name,
        num_classes=num_classes,
        dropout=0.1,
        backbone=BackboneConfig(**backbone_kwargs),
    )


class TestImageClassifier:
    def test_forward_shape(self):
        model = build_model(small_model_config())
        assert isinstance(model, ImageClassifier)
        logits = model(torch.rand(2, 3, 64, 64))
        assert logits.shape == (2, 10)

    def test_predict_is_distribution(self):
        model = build_model(small_model_config(num_classes=4)).eval()
        probs = model.predict(torch.rand(2, 3, 64, 64))
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(2))

    def test_forward_features(self):
        model = build_model(small_model_config())
        feats = model.forward_features(torch.rand(2, 3, 64, 64))
        assert feats.shape == (2, 512)

    def test_freeze_backbone(self):
        model = build_model(small_model_config())
        model.freeze_backbone()
        assert not any(p.requires_grad for p in model.backbone.parameters())
        assert all(p.requires_grad for p in model.head.parameters())
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_features_only_backbone_rejected(self):
        config = small_model_config(features_only=True)
        with pytest.raises(ValueError, match="pooled"):
            build_model(config)

    def test_multichannel_classifier(self):
        config = small_model_config(in_channels=13, channel_attention=True)
        model = build_model(config)
        assert model(torch.rand(2, 13, 64, 64)).shape == (2, 10)


class TestMultiLabelClassifier:
    def test_build_and_flags(self):
        model = build_model(small_model_config(name="multilabel_classifier", num_classes=5))
        assert isinstance(model, MultiLabelImageClassifier)
        assert model.is_multilabel is True

    def test_predict_probabilities(self):
        model = build_model(
            small_model_config(name="multilabel_classifier", num_classes=5)
        ).eval()
        probs = model.predict(torch.rand(2, 3, 64, 64))
        assert probs.shape == (2, 5)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_labels_binary(self):
        model = build_model(
            small_model_config(name="multilabel_classifier", num_classes=5)
        ).eval()
        labels = model.predict_labels(torch.rand(2, 3, 64, 64))
        assert set(labels.unique().tolist()) <= {0, 1}
