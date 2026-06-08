import pytest
import torch

from image_analytics.core.config import BackboneConfig, ModelConfig, config_from_dict
from image_analytics.detection.train import build_detection_model, run


def tiny_cascade_config(**kwargs):
    defaults = dict(
        fpn_channels=32,
        box_head_dim=64,
        rpn_anchor_sizes=[[16], [32], [64], [128], [256]],
        rpn_post_nms_topk=[200, 100],
    )
    defaults.update(kwargs)
    return ModelConfig(
        name="cascade_rcnn",
        num_classes=3,
        backbone=BackboneConfig(name="resnet18", pretrained=False, features_only=True),
        kwargs=defaults,
    )


def make_batch(batch_size=2, image_size=64):
    images = torch.rand(batch_size, 3, image_size, image_size)
    targets = [
        {"boxes": torch.tensor([[8.0, 8, 30, 30], [35.0, 35, 60, 58]]),
         "labels": torch.tensor([0, 2])}
        for _ in range(batch_size)
    ]
    return images, targets


class TestCascadeRCNN:
    def test_three_stage_loss_dict(self):
        model = build_detection_model(tiny_cascade_config()).train()
        losses = model(*make_batch())
        assert "loss" in losses
        for s in range(3):
            assert f"loss_cls_s{s}" in losses
            assert f"loss_reg_s{s}" in losses
        assert {"loss_rpn_cls", "loss_rpn_reg"} <= set(losses)
        for v in losses.values():
            assert torch.isfinite(v)
        assert losses["loss"].requires_grad

    def test_per_stage_heads_built(self):
        model = build_detection_model(tiny_cascade_config())
        assert len(model.box_heads) == 3
        assert len(model.cls_predictors) == 3
        assert len(model.reg_predictors) == 3
        # class-agnostic regression: 4 outputs, not num_classes*4
        assert model.reg_predictors[0].out_features == 4

    def test_empty_targets(self):
        model = build_detection_model(tiny_cascade_config()).train()
        images, _ = make_batch()
        empty = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(2)
        ]
        assert torch.isfinite(model(images, empty)["loss"])

    def test_eval_predictions(self):
        model = build_detection_model(tiny_cascade_config()).eval()
        images, _ = make_batch()
        with torch.no_grad():
            preds = model(images)
        assert len(preds) == 2
        for p in preds:
            assert set(p) == {"boxes", "scores", "labels"}
            if len(p["labels"]):
                assert int(p["labels"].max()) < 3
                assert p["boxes"].min() >= 0 and p["boxes"].max() <= 64

    def test_config_file_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/detection/cascade_rcnn_shapes.yaml")
        assert config.model.name == "cascade_rcnn"


@pytest.mark.slow
class TestCascadeLearns:
    def test_overfit_single_batch(self):
        torch.manual_seed(0)
        config = tiny_cascade_config()
        config.backbone = BackboneConfig(
            name="resnet10t", pretrained=False, features_only=True,
            kwargs={"out_indices": (1, 2, 3, 4)},
        )
        model = build_detection_model(config).train()
        images, targets = make_batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for _ in range(30):
            losses = model(images, targets)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            first = first if first is not None else float(losses["loss"])
        assert float(losses["loss"]) < first * 0.6
