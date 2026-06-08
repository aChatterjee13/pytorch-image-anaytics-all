import math

import pytest
import torch

from image_analytics.detection_3d import box3d


def box(x, y, z, dx, dy, dz, yaw=0.0):
    return torch.tensor([[x, y, z, dx, dy, dz, yaw]], dtype=torch.float32)


class TestAxisAlignedIoU3D:
    def test_identical(self):
        b = box(0, 0, 0, 2, 2, 2)
        assert box3d.axis_aligned_iou_3d(b, b)[0, 0] == pytest.approx(1.0)

    def test_disjoint(self):
        assert box3d.axis_aligned_iou_3d(box(0, 0, 0, 1, 1, 1), box(5, 5, 5, 1, 1, 1))[0, 0] == 0.0

    def test_half_overlap_hand_computed(self):
        # unit cubes offset 0.5 in x: inter 0.5, union 1.5 -> IoU 1/3
        iou = box3d.axis_aligned_iou_3d(box(0, 0, 0, 1, 1, 1), box(0.5, 0, 0, 1, 1, 1))
        assert iou[0, 0] == pytest.approx(1 / 3)

    def test_empty(self):
        assert box3d.axis_aligned_iou_3d(torch.zeros(0, 7), box(0, 0, 0, 1, 1, 1)).shape == (0, 1)


class TestBEVIoU:
    def test_identical(self):
        b = box(0, 0, 0, 2, 2, 2)
        assert box3d.bev_iou(b, b)[0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_half_overlap(self):
        iou = box3d.bev_iou(box(0, 0, 0, 1, 1, 1), box(0.5, 0, 0, 1, 1, 1))
        assert iou[0, 0] == pytest.approx(1 / 3, abs=1e-3)

    def test_rotation_45_degrees(self):
        # A box vs itself rotated 45° about center: overlap < 1.
        b = box(0, 0, 0, 2, 2, 2, 0.0)
        r = box(0, 0, 0, 2, 2, 2, math.pi / 4)
        iou = box3d.bev_iou(b, r)[0, 0]
        assert 0.5 < iou < 1.0


class TestIoU3D:
    def test_z_offset_reduces_iou(self):
        # Same BEV footprint, boxes offset in z by half their height -> IoU 1/3.
        iou = box3d.iou_3d(box(0, 0, 0, 1, 1, 1), box(0, 0, 0.5, 1, 1, 1))
        assert iou[0, 0] == pytest.approx(1 / 3, abs=1e-3)


class TestNMS3D:
    def test_suppresses_overlap(self):
        boxes = torch.cat([box(0, 0, 0, 2, 2, 2), box(0.1, 0, 0, 2, 2, 2), box(9, 9, 9, 1, 1, 1)])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = box3d.nms_3d(boxes, scores, iou_threshold=0.3)
        assert keep.tolist() == [0, 2]  # second box suppressed by the first
