import pytest
import torch

import image_analytics.segmentation.losses  # noqa: F401  (registers losses)
from image_analytics.core.registry import LOSSES
from image_analytics.segmentation.losses import CombinedLoss, DiceLoss, FocalLoss


def _perfect_logits(target, num_classes):
    b, h, w = target.shape
    logits = torch.full((b, num_classes, h, w), -10.0)
    for c in range(num_classes):
        logits[:, c][target == c] = 10.0
    return logits


class TestDiceLoss:
    def test_perfect_prediction_is_zero(self):
        target = torch.randint(0, 3, (2, 8, 8))
        loss = DiceLoss()(_perfect_logits(target, 3), target)
        assert loss.item() < 0.02

    def test_worst_prediction_is_near_one(self):
        target = torch.zeros(1, 4, 4, dtype=torch.int64)  # all class 0
        logits = torch.full((1, 2, 4, 4), -10.0)
        logits[:, 1] = 10.0  # predict everything as class 1
        loss = DiceLoss(smooth=1e-6)(logits, target)
        assert loss.item() > 0.9

    def test_ignore_index_excluded(self):
        # All non-ignored pixels predicted correctly -> ~0 loss despite a
        # deliberately wrong prediction on the ignored pixel.
        target = torch.tensor([[[0, 1], [255, 1]]])
        logits = _perfect_logits(target.clamp(max=1), 2)
        logits[0, :, 1, 0] = torch.tensor([-10.0, 10.0])  # wrong where ignored
        loss = DiceLoss(ignore_index=255)(logits, target)
        assert loss.item() < 0.05

    def test_gradients_flow(self):
        target = torch.randint(0, 3, (2, 8, 8))
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        DiceLoss()(logits, target).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


class TestCombinedLoss:
    def test_equals_weighted_ce_plus_dice(self):
        target = torch.randint(0, 3, (2, 8, 8))
        logits = torch.randn(2, 3, 8, 8)
        ce = torch.nn.CrossEntropyLoss(ignore_index=255)(logits, target)
        dice = DiceLoss()(logits, target)
        combined = CombinedLoss(ce_weight=1.0, dice_weight=0.5)(logits, target)
        assert combined.item() == pytest.approx((ce + 0.5 * dice).item(), rel=1e-5)


class TestFocalLoss:
    def test_downweights_easy_pixels(self):
        target = torch.randint(0, 3, (2, 8, 8))
        logits = torch.randn(2, 3, 8, 8)
        focal = FocalLoss(gamma=2.0)(logits, target)
        ce = torch.nn.CrossEntropyLoss()(logits, target)
        assert focal.item() < ce.item()  # focal modulation reduces the loss

    def test_finite_and_differentiable(self):
        target = torch.randint(0, 3, (2, 8, 8))
        target[0, 0, 0] = 255
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        loss = FocalLoss(ignore_index=255)(logits, target)
        loss.backward()
        assert torch.isfinite(loss) and logits.grad is not None


def test_registered_in_losses():
    for name in ("cross_entropy", "dice", "ce_dice", "seg_focal"):
        assert name in LOSSES
        crit = LOSSES.build(name, ignore_index=255)
        out = crit(torch.randn(1, 3, 4, 4), torch.randint(0, 3, (1, 4, 4)))
        assert torch.isfinite(out)
