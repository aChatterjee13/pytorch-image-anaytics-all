import pytest
import torch
import torchvision.ops as tvops

from image_analytics.detection.box_coder import BoxCoder
from image_analytics.detection.losses import (
    diou_loss,
    giou_loss,
    paired_box_iou,
    sigmoid_focal_loss,
)


class TestFocalLoss:
    def test_parity_with_torchvision(self):
        torch.manual_seed(0)
        logits = torch.randn(64, 10)
        targets = (torch.rand(64, 10) > 0.9).float()
        ours = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        ref = tvops.sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0)
        torch.testing.assert_close(ours, ref)

    def test_easy_examples_downweighted(self):
        # Confident-correct prediction must contribute far less than a hard one
        easy = sigmoid_focal_loss(torch.tensor([8.0]), torch.tensor([1.0]))
        hard = sigmoid_focal_loss(torch.tensor([-8.0]), torch.tensor([1.0]))
        assert hard.item() / easy.item() > 1e4

    def test_reductions(self):
        logits, targets = torch.randn(4, 3), torch.zeros(4, 3)
        none = sigmoid_focal_loss(logits, targets, reduction="none")
        assert none.shape == (4, 3)
        torch.testing.assert_close(
            sigmoid_focal_loss(logits, targets, reduction="sum"), none.sum()
        )
        torch.testing.assert_close(
            sigmoid_focal_loss(logits, targets, reduction="mean"), none.mean()
        )

    def test_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            sigmoid_focal_loss(torch.zeros(1), torch.zeros(1), reduction="bogus")


class TestPairedIoU:
    def test_matches_torchvision_diagonal(self):
        torch.manual_seed(1)
        xy = torch.rand(32, 2) * 50
        wh = torch.rand(32, 2) * 30 + 1
        boxes1 = torch.cat([xy, xy + wh], dim=1)
        xy2 = torch.rand(32, 2) * 50
        boxes2 = torch.cat([xy2, xy2 + wh], dim=1)
        ours, _ = paired_box_iou(boxes1, boxes2)
        ref = tvops.box_iou(boxes1, boxes2).diagonal()
        torch.testing.assert_close(ours, ref)


class TestGIoULoss:
    def test_identical_boxes_zero_loss(self):
        boxes = torch.tensor([[0.0, 0, 10, 10], [5, 5, 20, 30]])
        loss = giou_loss(boxes, boxes)
        torch.testing.assert_close(loss, torch.zeros(2))

    def test_parity_with_torchvision(self):
        torch.manual_seed(2)
        xy = torch.rand(32, 2) * 50
        boxes1 = torch.cat([xy, xy + torch.rand(32, 2) * 30 + 1], dim=1)
        xy2 = torch.rand(32, 2) * 50
        boxes2 = torch.cat([xy2, xy2 + torch.rand(32, 2) * 30 + 1], dim=1)
        ours = giou_loss(boxes1, boxes2)
        ref = 1.0 - tvops.generalized_box_iou(boxes1, boxes2).diagonal()
        torch.testing.assert_close(ours, ref)

    def test_disjoint_boxes_have_gradient_signal(self):
        pred = torch.tensor([[0.0, 0, 10, 10]], requires_grad=True)
        target = torch.tensor([[100.0, 100, 110, 110]])
        loss = giou_loss(pred, target, reduction="sum")
        assert loss.item() > 1.0  # IoU=0 and enclosure penalty active
        loss.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0


class TestDIoULoss:
    def test_identical_boxes_zero_loss(self):
        boxes = torch.tensor([[0.0, 0, 10, 10]])
        torch.testing.assert_close(diou_loss(boxes, boxes), torch.zeros(1))

    def test_parity_with_torchvision(self):
        torch.manual_seed(3)
        xy = torch.rand(16, 2) * 50
        boxes1 = torch.cat([xy, xy + torch.rand(16, 2) * 30 + 1], dim=1)
        xy2 = torch.rand(16, 2) * 50
        boxes2 = torch.cat([xy2, xy2 + torch.rand(16, 2) * 30 + 1], dim=1)
        ours = diou_loss(boxes1, boxes2)
        ref = tvops.distance_box_iou_loss(boxes1, boxes2)
        torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)


class TestBoxCoder:
    def test_encode_decode_roundtrip(self):
        torch.manual_seed(4)
        coder = BoxCoder()
        xy = torch.rand(64, 2) * 100
        anchors = torch.cat([xy, xy + torch.rand(64, 2) * 50 + 4], dim=1)
        xy2 = torch.rand(64, 2) * 100
        gt = torch.cat([xy2, xy2 + torch.rand(64, 2) * 50 + 4], dim=1)
        decoded = coder.decode(coder.encode(gt, anchors), anchors)
        torch.testing.assert_close(decoded, gt, rtol=1e-4, atol=1e-3)

    def test_zero_deltas_recover_anchor(self):
        coder = BoxCoder()
        anchors = torch.tensor([[10.0, 20, 50, 80]])
        decoded = coder.decode(torch.zeros(1, 4), anchors)
        torch.testing.assert_close(decoded, anchors)

    def test_known_shift(self):
        # Anchor 0,0,10,10 (cx=5, w=10); gt shifted +10 in x, same size
        coder = BoxCoder()
        anchors = torch.tensor([[0.0, 0, 10, 10]])
        gt = torch.tensor([[10.0, 0, 20, 10]])
        deltas = coder.encode(gt, anchors)
        torch.testing.assert_close(deltas, torch.tensor([[1.0, 0, 0, 0]]))

    def test_weights_scale_deltas(self):
        coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        anchors = torch.tensor([[0.0, 0, 10, 10]])
        gt = torch.tensor([[10.0, 0, 20, 10]])
        deltas = coder.encode(gt, anchors)
        torch.testing.assert_close(deltas, torch.tensor([[10.0, 0, 0, 0]]))
        torch.testing.assert_close(coder.decode(deltas, anchors), gt)

    def test_clip_prevents_overflow(self):
        coder = BoxCoder()
        anchors = torch.tensor([[0.0, 0, 10, 10]])
        wild = torch.tensor([[0.0, 0, 100.0, 100.0]])  # huge dw/dh
        decoded = coder.decode(wild, anchors)
        assert torch.isfinite(decoded).all()
