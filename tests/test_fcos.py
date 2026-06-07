import math

import pytest
import torch

from image_analytics.core.config import BackboneConfig, ModelConfig, config_from_dict
from image_analytics.detection.anchors.anchor_free import (
    BACKGROUND,
    INF,
    assign_fcos_targets,
    boxes_from_distances,
    centerness_from_targets,
    pyramid_locations,
)
from image_analytics.detection.train import build_detection_model, run


class TestPyramidLocations:
    def test_centers(self):
        locs = pyramid_locations([(2, 2)], strides=[8])[0]
        assert locs.shape == (4, 2)
        torch.testing.assert_close(locs[0], torch.tensor([4.0, 4.0]))
        torch.testing.assert_close(locs[1], torch.tensor([12.0, 4.0]))  # x advances first
        torch.testing.assert_close(locs[3], torch.tensor([12.0, 12.0]))


class TestAssignment:
    def _single_level(self, gt_boxes, gt_labels, **kwargs):
        locations = pyramid_locations([(8, 8)], strides=[8])[0]  # 96px coverage
        strides = torch.full((64,), 8.0)
        ranges = torch.tensor([0.0, 64.0]).expand(64, 2)
        return assign_fcos_targets(
            locations, strides, ranges, gt_boxes, gt_labels, **kwargs
        )

    def test_location_inside_center_matched(self):
        gt = torch.tensor([[8.0, 8, 40, 40]])  # center (24, 24)
        labels, reg = self._single_level(gt, torch.tensor([2]))
        # Location (20, 20) is inside box and within 1.5*8=12 of the center
        idx = 2 * 8 + 2  # grid (row 2, col 2) -> (20, 20)
        assert labels[idx] == 2
        torch.testing.assert_close(reg[idx], torch.tensor([12.0, 12.0, 20.0, 20.0]))

    def test_location_outside_box_background(self):
        gt = torch.tensor([[8.0, 8, 40, 40]])
        labels, _ = self._single_level(gt, torch.tensor([0]))
        idx = 7 * 8 + 7  # (60, 60) — far outside
        assert labels[idx] == BACKGROUND

    def test_center_sampling_excludes_box_edge(self):
        # Inside the box but > radius from center -> background
        gt = torch.tensor([[0.0, 0, 64, 64]])  # center (32,32), radius 12
        labels, _ = self._single_level(gt, torch.tensor([1]))
        idx = 0  # location (4, 4): inside box, 28px from center in x and y
        assert labels[idx] == BACKGROUND

    def test_regress_range_filters_large_objects(self):
        locations = pyramid_locations([(8, 8)], strides=[8])[0]
        strides = torch.full((64,), 8.0)
        narrow = torch.tensor([0.0, 16.0]).expand(64, 2)  # max side dist <= 16
        gt = torch.tensor([[0.0, 0, 64, 64]])  # center loc has max dist 32 > 16
        labels, _ = assign_fcos_targets(
            locations, strides, narrow, gt, torch.tensor([0])
        )
        assert (labels == BACKGROUND).all()

    def test_ambiguity_resolves_to_smaller_box(self):
        # Two concentric boxes; shared candidate locations take the smaller GT
        gt = torch.tensor([[16.0, 16, 48, 48], [24.0, 24, 40, 40]])
        labels, _ = self._single_level(gt, torch.tensor([0, 1]))
        center_idx = 4 * 8 + 4  # (36, 36): near both centers
        assert labels[center_idx] == 1

    def test_no_gt(self):
        labels, reg = self._single_level(
            torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64)
        )
        assert (labels == BACKGROUND).all()
        assert reg.shape == (64, 4)


class TestCenterness:
    def test_perfect_center_is_one(self):
        reg = torch.tensor([[10.0, 10, 10, 10]])
        assert centerness_from_targets(reg).item() == pytest.approx(1.0)

    def test_known_value(self):
        reg = torch.tensor([[10.0, 5, 40, 20]])
        expected = math.sqrt((10 / 40) * (5 / 20))
        assert centerness_from_targets(reg).item() == pytest.approx(expected, rel=1e-5)

    def test_edge_location_near_zero(self):
        reg = torch.tensor([[0.5, 10, 63.5, 10]])
        assert centerness_from_targets(reg).item() < 0.15


class TestBoxDecode:
    def test_roundtrip(self):
        locations = torch.tensor([[30.0, 40.0]])
        distances = torch.tensor([[10.0, 20, 15, 5]])
        boxes = boxes_from_distances(locations, distances)
        torch.testing.assert_close(boxes, torch.tensor([[20.0, 20, 45, 45]]))


def tiny_fcos_config(**kwargs):
    defaults = dict(
        fpn_channels=32,
        num_convs=1,
        regress_ranges=[[0, 32], [32, 64], [64, 128], [128, 256], [256, INF]],
        score_thresh=0.05,
    )
    defaults.update(kwargs)
    return ModelConfig(
        name="fcos",
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


class TestFCOSModel:
    def test_training_loss_dict(self):
        model = build_detection_model(tiny_fcos_config()).train()
        losses = model(*make_batch())
        assert set(losses) == {"loss", "loss_cls", "loss_reg", "loss_centerness"}
        for v in losses.values():
            assert torch.isfinite(v)

    def test_empty_targets(self):
        model = build_detection_model(tiny_fcos_config()).train()
        images, _ = make_batch()
        empty = [
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            for _ in range(2)
        ]
        losses = model(images, empty)
        assert torch.isfinite(losses["loss"])

    def test_eval_predictions(self):
        model = build_detection_model(tiny_fcos_config()).eval()
        images, _ = make_batch()
        with torch.no_grad():
            predictions = model(images)
        assert len(predictions) == 2
        for p in predictions:
            assert set(p) == {"boxes", "scores", "labels"}
            if len(p["scores"]):
                assert p["boxes"].min() >= 0 and p["boxes"].max() <= 64

    def test_overfit_single_batch(self):
        torch.manual_seed(0)
        config = tiny_fcos_config()
        config.backbone = BackboneConfig(
            name="resnet10t", pretrained=False, features_only=True,
            kwargs={"out_indices": (2, 3, 4)},
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


class TestFCOSEndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict(
            {
                "task": "detection",
                "experiment_name": "fcos_smoke",
                "output_dir": str(tmp_path),
                "model": {
                    "name": "fcos",
                    "num_classes": 3,
                    "backbone": {
                        "name": "resnet18", "pretrained": False, "features_only": True,
                    },
                    "neck": {"name": "fpn", "out_channels": 32},
                    "kwargs": {
                        "num_convs": 1,
                        "regress_ranges": [
                            [0, 32], [32, 64], [64, 128], [128, 256], [256, 1.0e8],
                        ],
                    },
                },
                "data": {
                    "dataset": "synthetic_shapes",
                    "image_size": 64,
                    "batch_size": 8,
                    "num_workers": 0,
                    "kwargs": {"size": 24, "image_size": 64},
                },
                "training": {
                    "epochs": 1, "lr": 1e-3, "scheduler": "none",
                    "device": "cpu", "log_interval": 0, "monitor": "val/mAP",
                },
            }
        )
        metrics = run(config)
        assert "val/mAP" in metrics

    def test_config_file_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/detection/fcos_shapes.yaml")
        assert config.model.name == "fcos"
        assert len(config.model.kwargs["regress_ranges"]) == 5
