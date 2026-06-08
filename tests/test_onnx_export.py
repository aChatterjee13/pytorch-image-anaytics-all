import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from image_analytics.backbones.registry import build_backbone
from image_analytics.classification.models import ImageClassifier
from image_analytics.core.config import BackboneConfig, ModelConfig
from image_analytics.serving.onnx_export import (
    RetinaNetRawHeads,
    build_exportable,
    check_parity,
    export_onnx,
)


def test_classifier_export_and_parity(tmp_path):
    bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
    model = ImageClassifier(bb, num_classes=10).eval()
    path = export_onnx(model, torch.randn(1, 3, 64, 64), tmp_path / "clf.onnx", atol=1e-3)
    assert path.exists()
    # parity also holds at a different batch size (dynamic axis)
    assert check_parity(model, torch.randn(4, 3, 64, 64), path, atol=1e-3) < 1e-3


def test_parity_failure_raises(tmp_path):
    bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
    model = ImageClassifier(bb, num_classes=10).eval()
    path = export_onnx(model, torch.randn(1, 3, 64, 64), tmp_path / "clf.onnx", atol=1e-3)
    # an unrealistically tight tolerance must fail the gate
    with pytest.raises(RuntimeError, match="parity"):
        check_parity(model, torch.randn(1, 3, 64, 64), path, atol=1e-12)


def test_retinanet_raw_heads_export(tmp_path):
    from image_analytics.detection.train import build_detection_model

    model = build_detection_model(ModelConfig(
        name="retinanet", num_classes=3,
        backbone=BackboneConfig(name="resnet18", pretrained=False, features_only=True),
        kwargs={"fpn_channels": 32, "num_convs": 1, "anchor_base_sizes": (16, 32, 64, 128, 256)},
    )).eval()
    wrapped = build_exportable(model, "detection")
    assert isinstance(wrapped, RetinaNetRawHeads)
    path = export_onnx(wrapped, torch.randn(1, 3, 64, 64), tmp_path / "rn.onnx", atol=1e-3)
    assert path.exists()


def test_build_exportable_classifier_is_whole_model():
    bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
    model = ImageClassifier(bb, num_classes=4)
    assert build_exportable(model, "classification") is model
