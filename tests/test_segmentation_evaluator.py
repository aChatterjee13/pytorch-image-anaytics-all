import math

import pytest
import torch

from image_analytics.core.evaluator import SegmentationEvaluator


def _one_hot_logits(target, num_classes):
    """Build logits that argmax to `target` exactly."""
    b, h, w = target.shape
    logits = torch.full((b, num_classes, h, w), -10.0)
    for c in range(num_classes):
        logits[:, c][target == c] = 10.0
    return logits


class TestSegmentationEvaluator:
    def test_perfect_prediction(self):
        ev = SegmentationEvaluator(num_classes=4)
        target = torch.tensor([[[0, 1], [2, 3]]])
        ev.update(_one_hot_logits(target, 4), target)
        m = ev.compute()
        assert m["mIoU"] == 1.0
        assert m["dice"] == 1.0
        assert m["pixel_accuracy"] == 1.0

    def test_class_index_predictions_path(self):
        """update accepts (B,H,W) preds as well as (B,C,H,W) logits."""
        ev = SegmentationEvaluator(num_classes=2)
        target = torch.tensor([[0, 1, 0, 1]])
        preds = torch.tensor([[0, 1, 0, 1]])
        ev.update(preds, target)
        assert ev.compute()["mIoU"] == 1.0

    def test_ignore_index_excluded(self):
        # target [0, 1, 255], preds [0, 0, 0]; the 255 pixel is dropped, so
        # class 0 IoU = 1/2, class 1 IoU = 0 -> mIoU 0.25, pixel acc 0.5.
        ev = SegmentationEvaluator(num_classes=2, ignore_index=255)
        ev.update(torch.tensor([[0, 0, 0]]), torch.tensor([[0, 1, 255]]))
        m = ev.compute()
        assert m["pixel_accuracy"] == pytest.approx(0.5)
        assert m["mIoU"] == pytest.approx(0.25)
        assert m["iou_0"] == pytest.approx(0.5)
        assert m["iou_1"] == pytest.approx(0.0)

    def test_partial_overlap_hand_computed(self):
        # target [[1,1],[2,2]], preds [[1,0],[2,2]]
        #   class1 IoU = 1/2, class2 IoU = 1.0, class0 absent -> mIoU 0.75
        ev = SegmentationEvaluator(num_classes=3)
        target = torch.tensor([[[1, 1], [2, 2]]])
        preds = torch.tensor([[[1, 0], [2, 2]]])
        ev.update(preds, target)
        m = ev.compute()
        assert m["iou_1"] == pytest.approx(0.5)
        assert m["iou_2"] == pytest.approx(1.0)
        assert math.isnan(m["iou_0"])  # class 0 has no ground-truth support
        assert m["mIoU"] == pytest.approx(0.75)
        assert m["pixel_accuracy"] == pytest.approx(0.75)

    def test_streaming_accumulation(self):
        """Two updates accumulate like one combined batch."""
        ev = SegmentationEvaluator(num_classes=2)
        ev.update(torch.tensor([[0, 0]]), torch.tensor([[0, 1]]))
        ev.update(torch.tensor([[1, 1]]), torch.tensor([[0, 1]]))
        m = ev.compute()
        # confusion: class0 {tp1, fn1}, class1 {tp1, fn1} -> each IoU 1/3
        assert m["pixel_accuracy"] == pytest.approx(0.5)
        assert m["mIoU"] == pytest.approx(1 / 3)

    def test_empty_returns_empty_dict(self):
        ev = SegmentationEvaluator(num_classes=2)
        assert ev.compute() == {}

    def test_reset(self):
        ev = SegmentationEvaluator(num_classes=2)
        ev.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
        ev.reset()
        assert ev.compute() == {}
