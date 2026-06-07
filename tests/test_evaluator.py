import math

import pytest
import torch

from image_analytics.core.evaluator import (
    ClassificationEvaluator,
    MultiLabelEvaluator,
    average_precision,
)


class TestClassificationEvaluator:
    def test_perfect_predictions(self):
        ev = ClassificationEvaluator(num_classes=3)
        logits = torch.tensor(
            [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        )
        targets = torch.tensor([0, 1, 2])
        ev.update(logits, targets)
        metrics = ev.compute()
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    def test_known_confusion(self):
        # 4 samples, 2 classes: predictions [0, 0, 1, 1], targets [0, 1, 1, 1]
        # -> accuracy 3/4; class0: P=1/2, R=1; class1: P=1, R=2/3
        ev = ClassificationEvaluator(num_classes=2)
        logits = torch.tensor([[2.0, 0], [2.0, 0], [0, 2.0], [0, 2.0]])
        targets = torch.tensor([0, 1, 1, 1])
        ev.update(logits, targets)
        m = ev.compute()
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["macro_precision"] == pytest.approx((0.5 + 1.0) / 2)
        assert m["macro_recall"] == pytest.approx((1.0 + 2 / 3) / 2)
        f1_0 = 2 * 0.5 * 1.0 / (0.5 + 1.0)
        f1_1 = 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3)
        assert m["macro_f1"] == pytest.approx((f1_0 + f1_1) / 2)

    def test_topk(self):
        ev = ClassificationEvaluator(num_classes=10, topk=(1, 5))
        logits = torch.zeros(2, 10)
        logits[0, 3] = 5.0  # target 3 -> top1 hit
        logits[1, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])  # target 4 -> top5 hit only
        targets = torch.tensor([3, 4])
        ev.update(logits, targets)
        m = ev.compute()
        assert m["accuracy"] == pytest.approx(0.5)
        assert m["top5_accuracy"] == pytest.approx(1.0)

    def test_streaming_updates_match_single_batch(self):
        torch.manual_seed(1)
        logits = torch.randn(64, 5)
        targets = torch.randint(0, 5, (64,))
        ev_one = ClassificationEvaluator(5)
        ev_one.update(logits, targets)
        ev_stream = ClassificationEvaluator(5)
        for chunk in range(4):
            sl = slice(chunk * 16, (chunk + 1) * 16)
            ev_stream.update(logits[sl], targets[sl])
        assert ev_one.compute() == ev_stream.compute()

    def test_reset(self):
        ev = ClassificationEvaluator(2)
        ev.update(torch.tensor([[1.0, 0]]), torch.tensor([0]))
        ev.reset()
        assert ev.compute() == {}


class TestAveragePrecision:
    def test_perfect_ranking(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        targets = torch.tensor([1, 1, 0, 0])
        assert average_precision(scores, targets) == pytest.approx(1.0)

    def test_known_value(self):
        # Ranking: pos, neg, pos -> AP = (1/1 + 2/3) / 2
        scores = torch.tensor([0.9, 0.5, 0.3])
        targets = torch.tensor([1, 0, 1])
        assert average_precision(scores, targets) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_no_positives_is_nan(self):
        ap = average_precision(torch.tensor([0.5]), torch.tensor([0]))
        assert math.isnan(ap)


class TestMultiLabelEvaluator:
    def test_perfect(self):
        ev = MultiLabelEvaluator(num_labels=3)
        logits = torch.tensor([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])
        targets = torch.tensor([[1.0, 0, 1], [0, 1, 0]])
        ev.update(logits, targets)
        m = ev.compute()
        assert m["micro_f1"] == pytest.approx(1.0)
        assert m["mAP"] == pytest.approx(1.0)
        assert m["subset_accuracy"] == pytest.approx(1.0)

    def test_partial(self):
        ev = MultiLabelEvaluator(num_labels=2)
        # Sample 0: predicts [1, 1], target [1, 0]; sample 1: [0, 1] vs [0, 1]
        logits = torch.tensor([[5.0, 5.0], [-5.0, 5.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ev.update(logits, targets)
        m = ev.compute()
        # micro: TP=2, FP=1, FN=0 -> F1 = 4/5
        assert m["micro_f1"] == pytest.approx(0.8)
        assert m["subset_accuracy"] == pytest.approx(0.5)
        assert m["accuracy"] == m["micro_f1"]  # alias for the default monitor

    def test_empty(self):
        assert MultiLabelEvaluator(num_labels=3).compute() == {}
