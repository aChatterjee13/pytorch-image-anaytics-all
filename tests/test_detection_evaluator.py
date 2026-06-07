import contextlib
import io

import pytest
import torch

from image_analytics.core.evaluator import DetectionEvaluator


def pred(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def gt(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TestDetectionEvaluator:
    def test_perfect_predictions(self):
        ev = DetectionEvaluator(num_classes=2)
        boxes = [[10, 10, 30, 30], [50, 50, 80, 90]]
        ev.update([pred(boxes, [0.9, 0.8], [0, 1])], [gt(boxes, [0, 1])])
        m = ev.compute()
        assert m["mAP"] == pytest.approx(1.0)
        assert m["mAP50"] == pytest.approx(1.0)

    def test_no_predictions_zero_map(self):
        ev = DetectionEvaluator(num_classes=2)
        ev.update([pred([], [], [])], [gt([[10, 10, 30, 30]], [0])])
        assert ev.compute()["mAP"] == pytest.approx(0.0)

    def test_wrong_class_zero_map(self):
        ev = DetectionEvaluator(num_classes=2)
        boxes = [[10, 10, 30, 30]]
        ev.update([pred(boxes, [0.9], [1])], [gt(boxes, [0])])
        assert ev.compute()["mAP"] == pytest.approx(0.0)

    def test_localization_quality_affects_high_iou_thresholds(self):
        ev = DetectionEvaluator(num_classes=1)
        # IoU vs GT = 20*20/(2*30*20-20*20) ≈ 0.5: counts at 0.5 only with shift
        ev.update(
            [pred([[10, 10, 40, 30]], [0.9], [0])],
            [gt([[20, 10, 50, 30]], [0])],
        )
        m = ev.compute()
        assert m["mAP50"] > 0.9
        assert m["mAP75"] == pytest.approx(0.0)

    def test_duplicate_detections_penalized(self):
        ev = DetectionEvaluator(num_classes=1, iou_thresholds=(0.5,))
        box = [[10, 10, 40, 40]]
        # Two near-identical predictions for one GT: second is FP
        ev.update(
            [pred([[10, 10, 40, 40], [11, 11, 41, 41]], [0.9, 0.8], [0, 0])],
            [gt(box, [0])],
        )
        m = ev.compute()
        assert 0.5 < m["mAP"] <= 1.0  # precision drops after the duplicate

    def test_score_ranking_matters(self):
        # FP outranking the TP lowers AP below the reverse ordering
        def run(fp_score):
            ev = DetectionEvaluator(num_classes=1, iou_thresholds=(0.5,))
            ev.update(
                [
                    pred(
                        [[10, 10, 40, 40], [200, 200, 240, 240]],
                        [0.9, fp_score],
                        [0, 0],
                    )
                ],
                [gt([[10, 10, 40, 40]], [0])],
            )
            return ev.compute()["mAP"]

        assert run(fp_score=0.5) > run(fp_score=0.95) - 1e-9
        assert run(fp_score=0.5) == pytest.approx(1.0)

    def test_reset(self):
        ev = DetectionEvaluator(num_classes=1)
        ev.update([pred([[0, 0, 10, 10]], [0.9], [0])], [gt([[0, 0, 10, 10]], [0])])
        ev.reset()
        assert ev.compute() == {}


class TestPycocotoolsParity:
    def test_map_matches_cocoeval(self):
        """Our mAP must match pycocotools COCOeval on a mixed fixture."""
        pycoco = pytest.importorskip("pycocotools.coco")
        from pycocotools.cocoeval import COCOeval

        torch.manual_seed(0)
        num_images, num_classes = 8, 3
        all_preds, all_gts = [], []
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [{"id": c + 1, "name": str(c)} for c in range(num_classes)],
        }
        coco_dt = []
        ann_id = 1
        for img_id in range(num_images):
            coco_gt["images"].append({"id": img_id, "width": 200, "height": 200})
            n_gt = int(torch.randint(1, 4, (1,)))
            gboxes, glabels = [], []
            for _ in range(n_gt):
                xy = torch.rand(2) * 120
                wh = torch.rand(2) * 50 + 10
                box = [float(xy[0]), float(xy[1]), float(xy[0] + wh[0]), float(xy[1] + wh[1])]
                label = int(torch.randint(0, num_classes, (1,)))
                gboxes.append(box)
                glabels.append(label)
                coco_gt["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": label + 1,
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "area": (box[2] - box[0]) * (box[3] - box[1]),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
            # Predictions: jittered GT (some good, some bad) + random noise boxes
            pboxes, pscores, plabels = [], [], []
            for box, label in zip(gboxes, glabels):
                jitter = (torch.rand(4) - 0.5) * float(torch.randint(0, 30, (1,)))
                jb = torch.tensor(box) + jitter
                pboxes.append(jb.tolist())
                pscores.append(float(torch.rand(1) * 0.5 + 0.5))
                plabels.append(label)
            for _ in range(int(torch.randint(0, 3, (1,)))):
                xy = torch.rand(2) * 150
                wh = torch.rand(2) * 40 + 5
                pboxes.append(
                    [float(xy[0]), float(xy[1]), float(xy[0] + wh[0]), float(xy[1] + wh[1])]
                )
                pscores.append(float(torch.rand(1) * 0.6))
                plabels.append(int(torch.randint(0, num_classes, (1,))))
            for box, score, label in zip(pboxes, pscores, plabels):
                coco_dt.append(
                    {
                        "image_id": img_id,
                        "category_id": label + 1,
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": score,
                    }
                )
            all_preds.append(pred(pboxes, pscores, plabels))
            all_gts.append(gt(gboxes, glabels))

        ev = DetectionEvaluator(num_classes=num_classes)
        ev.update(all_preds, all_gts)
        ours = ev.compute()

        with contextlib.redirect_stdout(io.StringIO()):
            coco = pycoco.COCO()
            coco.dataset = coco_gt
            coco.createIndex()
            coco_pred = coco.loadRes(coco_dt)
            coco_eval = COCOeval(coco, coco_pred, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        assert ours["mAP"] == pytest.approx(coco_eval.stats[0], abs=0.01)
        assert ours["mAP50"] == pytest.approx(coco_eval.stats[1], abs=0.01)
        assert ours["mAP75"] == pytest.approx(coco_eval.stats[2], abs=0.01)
