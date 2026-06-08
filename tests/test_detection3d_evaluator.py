import pytest
import torch

from image_analytics.core.evaluator import Detection3DEvaluator


def pred(boxes, scores, labels):
    return {
        "boxes_3d": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def gt(boxes, labels):
    return {
        "boxes_3d": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TestDetection3DEvaluator:
    def test_perfect(self):
        ev = Detection3DEvaluator(num_classes=1)
        boxes = [[0, 0, 0, 2, 2, 2, 0], [5, 5, 5, 1, 1, 1, 0]]
        ev.update([pred(boxes, [0.9, 0.8], [0, 0])], [gt(boxes, [0, 0])])
        m = ev.compute()
        assert m["mAP_3d"] == pytest.approx(1.0)
        assert m["AP_3d@0.7"] == pytest.approx(1.0)

    def test_localization_quality_affects_strict_threshold(self):
        ev = Detection3DEvaluator(num_classes=1)
        # offset 0.5 in x: IoU 1/3 -> counts at 0.25 but not 0.5/0.7
        ev.update(
            [pred([[0.5, 0, 0, 1, 1, 1, 0]], [0.9], [0])],
            [gt([[0, 0, 0, 1, 1, 1, 0]], [0])],
        )
        m = ev.compute()
        assert m["AP_3d@0.25"] == pytest.approx(1.0)
        assert m["AP_3d@0.5"] == pytest.approx(0.0)

    def test_no_predictions(self):
        ev = Detection3DEvaluator(num_classes=1)
        ev.update([pred([], [], [])], [gt([[0, 0, 0, 1, 1, 1, 0]], [0])])
        assert ev.compute()["mAP_3d"] == pytest.approx(0.0)

    def test_reset(self):
        ev = Detection3DEvaluator(num_classes=1)
        ev.update([pred([[0, 0, 0, 1, 1, 1, 0]], [0.9], [0])], [gt([[0, 0, 0, 1, 1, 1, 0]], [0])])
        ev.reset()
        assert ev.compute() == {}
