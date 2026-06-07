import math

import pytest
import torch

from image_analytics.detection.anchors import AnchorGenerator, Matcher
from image_analytics.detection.anchors.generator import retinanet_sizes
from image_analytics.detection.necks.fpn import FPN


class TestFPN:
    def test_output_shapes_and_channels(self):
        fpn = FPN([64, 128, 256], out_channels=32)
        features = [
            torch.rand(2, 64, 32, 32),
            torch.rand(2, 128, 16, 16),
            torch.rand(2, 256, 8, 8),
        ]
        outputs = fpn(features)
        assert len(outputs) == 3
        assert [o.shape for o in outputs] == [
            torch.Size([2, 32, 32, 32]),
            torch.Size([2, 32, 16, 16]),
            torch.Size([2, 32, 8, 8]),
        ]

    def test_p6p7_extra_levels(self):
        fpn = FPN([64, 128], out_channels=32, extra_levels="p6p7")
        outputs = fpn([torch.rand(1, 64, 16, 16), torch.rand(1, 128, 8, 8)])
        assert fpn.num_levels == 4
        assert len(outputs) == 4
        assert outputs[2].shape[-2:] == (4, 4)
        assert outputs[3].shape[-2:] == (2, 2)

    def test_pool_extra_level(self):
        fpn = FPN([64, 128], out_channels=32, extra_levels="pool")
        outputs = fpn([torch.rand(1, 64, 16, 16), torch.rand(1, 128, 8, 8)])
        assert len(outputs) == 3
        assert outputs[2].shape[-2:] == (4, 4)

    def test_top_down_propagates_semantics(self):
        # Zeroing the coarsest input must change the finest output
        fpn = FPN([8, 8], out_channels=8).eval()
        fine = torch.rand(1, 8, 16, 16)
        coarse = torch.rand(1, 8, 8, 8)
        with torch.no_grad():
            out_with = fpn([fine, coarse])[0]
            out_without = fpn([fine, torch.zeros_like(coarse)])[0]
        assert not torch.allclose(out_with, out_without)

    def test_wrong_level_count_raises(self):
        fpn = FPN([64, 128], out_channels=32)
        with pytest.raises(ValueError, match="feature maps"):
            fpn([torch.rand(1, 64, 16, 16)])


class TestAnchorGenerator:
    def test_counts(self):
        gen = AnchorGenerator(sizes=((32,), (64,)), aspect_ratios=(0.5, 1.0, 2.0))
        anchors = gen([(8, 8), (4, 4)], strides=[8, 16])
        assert gen.num_anchors_per_location == 3
        assert anchors[0].shape == (8 * 8 * 3, 4)
        assert anchors[1].shape == (4 * 4 * 3, 4)

    def test_first_anchor_centered_on_first_cell(self):
        gen = AnchorGenerator(sizes=((32,),), aspect_ratios=(1.0,), offset=0.5)
        anchors = gen([(4, 4)], strides=[8])[0]
        cx = (anchors[0, 0] + anchors[0, 2]) / 2
        cy = (anchors[0, 1] + anchors[0, 3]) / 2
        assert cx.item() == pytest.approx(4.0)  # 0.5 * stride
        assert cy.item() == pytest.approx(4.0)

    def test_aspect_ratios_and_area(self):
        gen = AnchorGenerator(sizes=((32,),), aspect_ratios=(0.5, 1.0, 2.0))
        anchors = gen([(1, 1)], strides=[8])[0]
        for i, ratio in enumerate((0.5, 1.0, 2.0)):
            w = anchors[i, 2] - anchors[i, 0]
            h = anchors[i, 3] - anchors[i, 1]
            assert (h / w).item() == pytest.approx(ratio, rel=1e-5)
            assert (h * w).item() == pytest.approx(32 * 32, rel=1e-5)

    def test_grid_major_layout(self):
        # Layout contract: anchors grouped per grid cell (A consecutive rows)
        gen = AnchorGenerator(sizes=((16,),), aspect_ratios=(0.5, 1.0), offset=0.5)
        anchors = gen([(2, 2)], strides=[16])[0]
        first_cell = anchors[:2]
        c1 = (first_cell[:, :2] + first_cell[:, 2:]) / 2
        torch.testing.assert_close(c1[0], c1[1])  # same center, different ratio
        second_cell = anchors[2:4]
        c2 = (second_cell[:, :2] + second_cell[:, 2:]) / 2
        assert c2[0, 0] > c1[0, 0]  # x advances along the row first

    def test_retinanet_sizes_octave_scales(self):
        sizes = retinanet_sizes((32, 64), scales_per_octave=3)
        assert sizes[0][0] == pytest.approx(32)
        assert sizes[0][1] == pytest.approx(32 * 2 ** (1 / 3))
        assert sizes[0][2] == pytest.approx(32 * 2 ** (2 / 3))
        assert sizes[1][0] == pytest.approx(64)

    def test_mismatched_levels_raise(self):
        gen = AnchorGenerator(sizes=((32,), (64,)))
        with pytest.raises(ValueError, match="feature shapes"):
            gen([(8, 8)], strides=[8])


class TestMatcher:
    def test_assignment_table(self):
        matcher = Matcher(high_threshold=0.5, low_threshold=0.4,
                          allow_low_quality_matches=False)
        # 2 GT x 5 anchors
        iou = torch.tensor(
            [
                [0.9, 0.45, 0.1, 0.0, 0.55],
                [0.1, 0.30, 0.2, 0.0, 0.60],
            ]
        )
        matches = matcher(iou)
        assert matches.tolist() == [0, Matcher.IGNORE, Matcher.BACKGROUND,
                                    Matcher.BACKGROUND, 1]

    def test_low_quality_forcing(self):
        matcher = Matcher(0.5, 0.4, allow_low_quality_matches=True)
        # GT 1's best anchor (idx 2) has IoU 0.3 — below low threshold,
        # but must be force-matched so the GT is not unassigned.
        iou = torch.tensor(
            [
                [0.8, 0.1, 0.0],
                [0.0, 0.1, 0.3],
            ]
        )
        matches = matcher(iou)
        assert matches[0] == 0
        assert matches[2] == 1

    def test_no_gt_all_background(self):
        matcher = Matcher()
        matches = matcher(torch.zeros(0, 7))
        assert matches.shape == (7,)
        assert (matches == Matcher.BACKGROUND).all()

    def test_zero_overlap_gt_not_forced(self):
        matcher = Matcher(0.5, 0.4, allow_low_quality_matches=True)
        iou = torch.tensor([[0.0, 0.0], [0.6, 0.0]])
        matches = matcher(iou)
        assert matches[0] == 1
        assert matches[1] == Matcher.BACKGROUND  # GT 0 has no overlap anywhere

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="low_threshold"):
            Matcher(high_threshold=0.4, low_threshold=0.5)
