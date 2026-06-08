import pytest
import torch

from image_analytics.core.evaluator import PanopticQualityEvaluator


def seg(category_id, x1, y1, x2, y2, h=16, w=16):
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask[y1:y2, x1:x2] = True
    return {"category_id": category_id, "mask": mask}


class TestPanopticQuality:
    def test_perfect_match_pq_one(self):
        ev = PanopticQualityEvaluator(num_classes=2)
        segs = [seg(0, 0, 0, 8, 8), seg(1, 8, 8, 14, 14)]
        ev.update([segs], [segs])
        m = ev.compute()
        assert m["PQ"] == pytest.approx(1.0)
        assert m["SQ"] == pytest.approx(1.0)
        assert m["RQ"] == pytest.approx(1.0)

    def test_false_positive_and_negative(self):
        # GT has one class-0 segment; prediction misses it (FN) and adds a
        # class-1 segment that has no GT (FP). Both classes present -> PQ 0.
        ev = PanopticQualityEvaluator(num_classes=2)
        gt = [seg(0, 0, 0, 8, 8)]
        pred = [seg(1, 0, 0, 8, 8)]
        ev.update([pred], [gt])
        m = ev.compute()
        assert m["PQ"] == pytest.approx(0.0)
        assert m["RQ"] == pytest.approx(0.0)

    def test_partial_overlap_below_threshold_is_unmatched(self):
        # IoU = 1/3 < 0.5 -> not a TP: one FP + one FN.
        ev = PanopticQualityEvaluator(num_classes=1)
        ev.update([[seg(0, 0, 0, 4, 4)]], [[seg(0, 2, 0, 6, 4)]])
        m = ev.compute()
        assert m["PQ"] == pytest.approx(0.0)

    def test_sq_reflects_match_iou(self):
        # One matched pair with IoU 0.6 (>0.5): RQ=1, SQ=0.6, PQ=0.6.
        # 10-wide vs shifted: build masks with known IoU.
        ev = PanopticQualityEvaluator(num_classes=1)
        a = torch.zeros(1, 10, dtype=torch.bool); a[0, :6] = True   # 6 px
        b = torch.zeros(1, 10, dtype=torch.bool); b[0, :6] = True
        # identical -> IoU 1; instead make a deterministic 0.6 IoU pair:
        gt = torch.zeros(10, dtype=torch.bool); gt[:10] = True       # 10 px
        pr = torch.zeros(10, dtype=torch.bool); pr[:6] = True        # 6 px, inter 6, union 10
        ev.update(
            [[{"category_id": 0, "mask": pr}]],
            [[{"category_id": 0, "mask": gt}]],
        )
        m = ev.compute()
        assert m["SQ"] == pytest.approx(0.6)
        assert m["RQ"] == pytest.approx(1.0)
        assert m["PQ"] == pytest.approx(0.6)

    def test_empty(self):
        ev = PanopticQualityEvaluator(num_classes=2)
        assert ev.compute() == {}


class TestWrappersRegistered:
    def test_universal_models_registered(self):
        import image_analytics.segmentation.instance  # noqa: F401
        import image_analytics.segmentation.panoptic  # noqa: F401
        from image_analytics.core.registry import MODELS

        assert "mask2former" in MODELS
        assert "oneformer" in MODELS
