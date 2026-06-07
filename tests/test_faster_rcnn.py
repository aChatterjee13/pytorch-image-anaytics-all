import pytest
import torch

from image_analytics.core.config import BackboneConfig, ModelConfig, config_from_dict
from image_analytics.detection.anchors.matcher import (
    BalancedPositiveNegativeSampler,
    Matcher,
)
from image_analytics.detection.heads.faster_rcnn import assign_boxes_to_fpn_levels
from image_analytics.detection.train import build_detection_model, run


class TestBalancedSampler:
    def test_respects_positive_fraction(self):
        torch.manual_seed(0)
        matches = torch.cat([torch.zeros(50, dtype=torch.int64),          # 50 positives
                             torch.full((500,), Matcher.BACKGROUND)])     # 500 negatives
        sampler = BalancedPositiveNegativeSampler(256, 0.5)
        pos, neg = sampler(matches)
        assert len(pos) == 50            # fewer positives than budget -> take all
        assert len(neg) == 256 - 50
        assert (matches[pos] >= 0).all()
        assert (matches[neg] == Matcher.BACKGROUND).all()

    def test_caps_positives(self):
        matches = torch.zeros(1000, dtype=torch.int64)  # all positive
        sampler = BalancedPositiveNegativeSampler(256, 0.25)
        pos, neg = sampler(matches)
        assert len(pos) == 64
        assert len(neg) == 0

    def test_ignore_never_sampled(self):
        matches = torch.full((100,), Matcher.IGNORE)
        pos, neg = BalancedPositiveNegativeSampler(64, 0.5)(matches)
        assert len(pos) == 0 and len(neg) == 0


class TestFPNLevelAssignment:
    def test_small_boxes_to_fine_level(self):
        boxes = torch.tensor([[0.0, 0, 32, 32]])  # sqrt(area)=32 -> k=4+log2(32/224)<2
        assert assign_boxes_to_fpn_levels(boxes, num_levels=4).item() == 0

    def test_canonical_box_to_canonical_level(self):
        boxes = torch.tensor([[0.0, 0, 224, 224]])  # k = 4 -> index 2 (P4)
        assert assign_boxes_to_fpn_levels(boxes, num_levels=4).item() == 2

    def test_huge_boxes_clamped_to_coarsest(self):
        boxes = torch.tensor([[0.0, 0, 2000, 2000]])
        assert assign_boxes_to_fpn_levels(boxes, num_levels=4).item() == 3


def tiny_frcnn_config(**kwargs):
    defaults = dict(
        fpn_channels=32,
        box_head_dim=64,
        rpn_anchor_sizes=[[16], [32], [64], [128], [256]],
        rpn_post_nms_topk=[200, 100],
        score_thresh=0.05,
    )
    defaults.update(kwargs)
    return ModelConfig(
        name="faster_rcnn",
        num_classes=3,
        backbone=BackboneConfig(name="resnet18", pretrained=False, features_only=True),
        kwargs=defaults,
    )


def make_batch(batch_size=2, image_size=64):
    images = torch.rand(batch_size, 3, image_size, image_size)
    targets = [
        {
            "boxes": torch.tensor([[8.0, 8, 30, 30], [35.0, 35, 60, 58]]),
            "labels": torch.tensor([0, 2]),
        }
        for _ in range(batch_size)
    ]
    return images, targets


class TestFasterRCNN:
    def test_backbone_gets_four_levels(self):
        model = build_detection_model(tiny_frcnn_config())
        assert len(model.backbone.feature_channels) == 4  # C2-C5
        assert model.fpn.num_levels == 5                  # + pooled P6

    def test_training_loss_dict(self):
        model = build_detection_model(tiny_frcnn_config()).train()
        losses = model(*make_batch())
        assert set(losses) == {
            "loss", "loss_rpn_cls", "loss_rpn_reg", "loss_cls", "loss_reg",
        }
        for v in losses.values():
            assert torch.isfinite(v)
        assert losses["loss"].requires_grad

    def test_empty_targets(self):
        model = build_detection_model(tiny_frcnn_config()).train()
        images, _ = make_batch()
        empty = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(2)
        ]
        losses = model(images, empty)
        assert torch.isfinite(losses["loss"])

    def test_eval_predictions_zero_based_labels(self):
        model = build_detection_model(tiny_frcnn_config()).eval()
        images, _ = make_batch()
        with torch.no_grad():
            predictions = model(images)
        assert len(predictions) == 2
        for p in predictions:
            assert set(p) == {"boxes", "scores", "labels"}
            if len(p["labels"]):
                assert int(p["labels"].max()) < 3   # background never leaks out
                assert p["boxes"].min() >= 0 and p["boxes"].max() <= 64

    def test_overfit_single_batch(self):
        torch.manual_seed(0)
        config = tiny_frcnn_config()
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
            if first is None:
                first = float(losses["loss"])
        assert float(losses["loss"]) < first * 0.6


class TestFasterRCNNEndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict(
            {
                "task": "detection",
                "experiment_name": "frcnn_smoke",
                "output_dir": str(tmp_path),
                "model": {
                    "name": "faster_rcnn",
                    "num_classes": 3,
                    "backbone": {
                        "name": "resnet18", "pretrained": False, "features_only": True,
                    },
                    "neck": {"name": "fpn", "out_channels": 32},
                    "kwargs": {
                        "box_head_dim": 64,
                        "rpn_anchor_sizes": [[16], [32], [64], [128], [256]],
                        "rpn_post_nms_topk": [200, 100],
                    },
                },
                "data": {
                    "dataset": "synthetic_shapes",
                    "image_size": 64,
                    "batch_size": 8,
                    "num_workers": 0,
                    "kwargs": {"size": 16, "image_size": 64},
                },
                "training": {
                    "epochs": 1, "lr": 1e-3, "scheduler": "none",
                    "device": "cpu", "log_interval": 0, "monitor": "val/mAP",
                },
            }
        )
        metrics = run(config)
        assert "val/mAP" in metrics
        assert (tmp_path / "frcnn_smoke" / "checkpoints" / "last.pt").exists()

    def test_config_file_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/detection/faster_rcnn_shapes.yaml")
        assert config.model.name == "faster_rcnn"
