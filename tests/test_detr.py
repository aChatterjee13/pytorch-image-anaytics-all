import pytest
import torch

from image_analytics.core.config import BackboneConfig, ModelConfig, config_from_dict
from image_analytics.detection.heads.detr import (
    HungarianMatcher,
    SetCriterion,
    sine_position_encoding,
)
from image_analytics.detection.train import build_detection_model, run


class TestPositionEncoding:
    def test_shape_and_range(self):
        pos = sine_position_encoding(4, 6, 64)
        assert pos.shape == (24, 64)
        assert pos.min() >= -1.0 and pos.max() <= 1.0

    def test_positions_distinct(self):
        pos = sine_position_encoding(8, 8, 64)
        # No two grid positions share an encoding
        dists = torch.cdist(pos, pos)
        dists.fill_diagonal_(1.0)
        assert dists.min() > 1e-3

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            sine_position_encoding(4, 4, 30)


class TestHungarianMatcher:
    def test_perfect_prediction_matched(self):
        matcher = HungarianMatcher()
        # Query 1 predicts target's class+box perfectly; query 0 predicts junk
        logits = torch.tensor([[[5.0, 0, 0, 0], [0, 5.0, 0, 0]]])  # K=3 + no-obj
        boxes = torch.tensor([[[0.9, 0.9, 0.05, 0.05], [0.3, 0.3, 0.2, 0.2]]])
        targets = [
            {
                "boxes_norm": torch.tensor([[0.3, 0.3, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
            }
        ]
        indices = matcher(logits, boxes, targets)
        query_idx, tgt_idx = indices[0]
        assert query_idx.tolist() == [1]
        assert tgt_idx.tolist() == [0]

    def test_one_to_one_assignment(self):
        torch.manual_seed(0)
        matcher = HungarianMatcher()
        logits = torch.randn(1, 10, 4)
        boxes = torch.rand(1, 10, 4) * 0.4 + 0.1
        targets = [
            {
                "boxes_norm": torch.rand(3, 4) * 0.4 + 0.1,
                "labels": torch.tensor([0, 1, 2]),
            }
        ]
        query_idx, tgt_idx = matcher(logits, boxes, targets)[0]
        assert len(query_idx) == 3
        assert len(set(query_idx.tolist())) == 3   # distinct queries
        assert sorted(tgt_idx.tolist()) == [0, 1, 2]

    def test_empty_targets(self):
        matcher = HungarianMatcher()
        indices = matcher(
            torch.randn(1, 5, 4), torch.rand(1, 5, 4),
            [{"boxes_norm": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}],
        )
        assert len(indices[0][0]) == 0


class TestSetCriterion:
    def test_perfect_predictions_near_zero_box_loss(self):
        criterion = SetCriterion(num_classes=3, matcher=HungarianMatcher())
        logits = torch.full((1, 4, 4), -10.0)
        logits[0, :, 3] = 10.0           # all queries: no-object...
        logits[0, 0, 3] = -10.0
        logits[0, 0, 1] = 10.0           # ...except query 0: class 1
        boxes = torch.rand(1, 4, 4) * 0.3 + 0.2
        targets = [
            {"boxes_norm": boxes[0, 0:1].clone(), "labels": torch.tensor([1])}
        ]
        losses = criterion(logits, boxes, targets)
        assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-5)
        assert losses["loss_giou"].item() == pytest.approx(0.0, abs=1e-4)
        assert losses["loss_class"].item() < 0.01

    def test_no_objects_image(self):
        criterion = SetCriterion(num_classes=3, matcher=HungarianMatcher())
        losses = criterion(
            torch.randn(1, 4, 4), torch.rand(1, 4, 4),
            [{"boxes_norm": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}],
        )
        assert torch.isfinite(losses["loss_class"])
        assert losses["loss_bbox"].item() == 0.0


def tiny_detr_config(**kwargs):
    defaults = dict(
        hidden_dim=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        num_queries=16,
        dropout=0.0,
        score_thresh=0.05,
    )
    defaults.update(kwargs)
    return ModelConfig(
        name="detr",
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


class TestDETRModel:
    def test_backbone_single_scale(self):
        model = build_detection_model(tiny_detr_config())
        assert len(model.backbone.feature_channels) == 1  # C5 only

    def test_training_loss_dict_with_aux(self):
        model = build_detection_model(tiny_detr_config()).train()
        losses = model(*make_batch())
        assert {"loss", "loss_class", "loss_bbox", "loss_giou"} <= set(losses)
        # 2 decoder layers -> one set of aux losses
        assert "loss_class_aux0" in losses
        for v in losses.values():
            assert torch.isfinite(v)

    def test_aux_disabled(self):
        model = build_detection_model(tiny_detr_config(aux_loss=False)).train()
        losses = model(*make_batch())
        assert not any("aux" in k for k in losses)

    def test_eval_predictions_capped_by_queries(self):
        model = build_detection_model(tiny_detr_config()).eval()
        images, _ = make_batch()
        with torch.no_grad():
            predictions = model(images)
        for p in predictions:
            assert set(p) == {"boxes", "scores", "labels"}
            assert len(p["scores"]) <= 16   # one prediction per query max
            if len(p["labels"]):
                assert int(p["labels"].max()) < 3
                assert p["boxes"].min() >= 0 and p["boxes"].max() <= 64

    def test_empty_targets(self):
        model = build_detection_model(tiny_detr_config()).train()
        images, _ = make_batch()
        empty = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(2)
        ]
        losses = model(images, empty)
        assert torch.isfinite(losses["loss"])

    def test_overfit_single_batch(self):
        torch.manual_seed(0)
        config = tiny_detr_config()
        config.backbone = BackboneConfig(
            name="resnet10t", pretrained=False, features_only=True,
            kwargs={"out_indices": (4,)},
        )
        model = build_detection_model(config).train()
        images, targets = make_batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        first = None
        for _ in range(50):
            losses = model(images, targets)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            if first is None:
                first = float(losses["loss"])
        assert float(losses["loss"]) < first * 0.6


class TestDETREndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict(
            {
                "task": "detection",
                "experiment_name": "detr_smoke",
                "output_dir": str(tmp_path),
                "model": {
                    "name": "detr",
                    "num_classes": 3,
                    "backbone": {
                        "name": "resnet18", "pretrained": False, "features_only": True,
                    },
                    "kwargs": {
                        "hidden_dim": 64, "nhead": 4,
                        "num_encoder_layers": 2, "num_decoder_layers": 2,
                        "dim_feedforward": 128, "num_queries": 16, "dropout": 0.0,
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
                    "epochs": 1, "lr": 1e-4, "scheduler": "none",
                    "device": "cpu", "log_interval": 0, "monitor": "val/mAP",
                },
            }
        )
        metrics = run(config)
        assert "val/mAP" in metrics

    def test_config_file_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/detection/detr_shapes.yaml")
        assert config.model.name == "detr"
        assert config.training.grad_clip == pytest.approx(0.1)
