import contextlib
import io

import numpy as np
import pytest
import torch

from image_analytics.core.evaluator import MaskMAPEvaluator, mask_iou


def _box_mask(h, w, x1, y1, x2, y2):
    m = torch.zeros(h, w, dtype=torch.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def pred(masks, scores, labels):
    return {
        "masks": torch.stack(masks) if masks else torch.zeros(0, 16, 16, dtype=torch.uint8),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def gt(masks, labels):
    return {
        "masks": torch.stack(masks) if masks else torch.zeros(0, 16, 16, dtype=torch.uint8),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class TestMaskIoU:
    def test_identical_masks(self):
        m = _box_mask(16, 16, 2, 2, 8, 8)[None]
        assert mask_iou(m, m)[0, 0] == pytest.approx(1.0)

    def test_disjoint_masks(self):
        a = _box_mask(16, 16, 0, 0, 4, 4)[None]
        b = _box_mask(16, 16, 8, 8, 12, 12)[None]
        assert mask_iou(a, b)[0, 0] == pytest.approx(0.0)

    def test_half_overlap_hand_computed(self):
        # 4x4 and 4x4 sharing a 4x2 overlap -> IoU = 8 / (16 + 16 - 8) = 1/3
        a = _box_mask(16, 16, 0, 0, 4, 4)[None]
        b = _box_mask(16, 16, 2, 0, 6, 4)[None]
        assert mask_iou(a, b)[0, 0] == pytest.approx(1 / 3)

    def test_empty_inputs(self):
        a = torch.zeros(0, 16, 16, dtype=torch.uint8)
        b = _box_mask(16, 16, 0, 0, 4, 4)[None]
        assert mask_iou(a, b).shape == (0, 1)


class TestMaskMAPEvaluator:
    def test_perfect_predictions(self):
        ev = MaskMAPEvaluator(num_classes=2)
        masks = [_box_mask(16, 16, 2, 2, 8, 8), _box_mask(16, 16, 9, 9, 14, 14)]
        ev.update([pred(masks, [0.9, 0.8], [0, 1])], [gt(masks, [0, 1])])
        m = ev.compute()
        assert m["mask_mAP"] == pytest.approx(1.0)
        assert m["mAP"] == pytest.approx(1.0)  # alias

    def test_wrong_class_zero(self):
        ev = MaskMAPEvaluator(num_classes=2)
        masks = [_box_mask(16, 16, 2, 2, 8, 8)]
        ev.update([pred(masks, [0.9], [1])], [gt(masks, [0])])
        assert ev.compute()["mask_mAP"] == pytest.approx(0.0)

    def test_no_predictions(self):
        ev = MaskMAPEvaluator(num_classes=2)
        ev.update([pred([], [], [])], [gt([_box_mask(16, 16, 2, 2, 8, 8)], [0])])
        assert ev.compute()["mask_mAP"] == pytest.approx(0.0)

    def test_reset(self):
        ev = MaskMAPEvaluator(num_classes=1)
        masks = [_box_mask(16, 16, 0, 0, 8, 8)]
        ev.update([pred(masks, [0.9], [0])], [gt(masks, [0])])
        ev.reset()
        assert ev.compute() == {}


class TestPycocotoolsSegmParity:
    def test_mask_map_matches_cocoeval(self):
        """Our mask mAP must match pycocotools COCOeval(iouType='segm')."""
        pycoco = pytest.importorskip("pycocotools.coco")
        from pycocotools import mask as mask_utils
        from pycocotools.cocoeval import COCOeval

        torch.manual_seed(0)
        H = W = 64
        num_images, num_classes = 6, 3

        def encode(binary):
            rle = mask_utils.encode(np.asfortranarray(binary.numpy().astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            return rle

        all_preds, all_gts = [], []
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [{"id": c + 1, "name": str(c)} for c in range(num_classes)],
        }
        coco_dt = []
        ann_id = 1
        for img_id in range(num_images):
            coco_gt["images"].append({"id": img_id, "width": W, "height": H})
            n_gt = int(torch.randint(1, 4, (1,)))
            gmasks, glabels = [], []
            for _ in range(n_gt):
                x1 = int(torch.randint(0, W - 16, (1,)))
                y1 = int(torch.randint(0, H - 16, (1,)))
                w = int(torch.randint(8, 16, (1,)))
                h = int(torch.randint(8, 16, (1,)))
                m = _box_mask(H, W, x1, y1, x1 + w, y1 + h)
                label = int(torch.randint(0, num_classes, (1,)))
                gmasks.append(m)
                glabels.append(label)
                rle = encode(m)
                coco_gt["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": label + 1,
                        "segmentation": rle,
                        "area": float(m.sum()),
                        "bbox": [x1, y1, w, h],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            pmasks, pscores, plabels = [], [], []
            for m, label in zip(gmasks, glabels):
                # jitter: shift the mask by a few pixels
                shift = int(torch.randint(0, 5, (1,)))
                jm = torch.zeros_like(m)
                jm[shift:, shift:] = m[: H - shift, : W - shift]
                pmasks.append(jm)
                pscores.append(float(torch.rand(1) * 0.5 + 0.5))
                plabels.append(label)
            for _ in range(int(torch.randint(0, 3, (1,)))):
                x1 = int(torch.randint(0, W - 16, (1,)))
                y1 = int(torch.randint(0, H - 16, (1,)))
                pmasks.append(_box_mask(H, W, x1, y1, x1 + 10, y1 + 10))
                pscores.append(float(torch.rand(1) * 0.6))
                plabels.append(int(torch.randint(0, num_classes, (1,))))
            for m, score, label in zip(pmasks, pscores, plabels):
                coco_dt.append(
                    {
                        "image_id": img_id,
                        "category_id": label + 1,
                        "segmentation": encode(m),
                        "score": score,
                    }
                )
            all_preds.append(pred(pmasks, pscores, plabels))
            all_gts.append(gt(gmasks, glabels))

        ev = MaskMAPEvaluator(num_classes=num_classes)
        ev.update(all_preds, all_gts)
        ours = ev.compute()

        with contextlib.redirect_stdout(io.StringIO()):
            coco = pycoco.COCO()
            coco.dataset = coco_gt
            coco.createIndex()
            coco_pred = coco.loadRes(coco_dt)
            coco_eval = COCOeval(coco, coco_pred, iouType="segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        assert ours["mask_mAP"] == pytest.approx(coco_eval.stats[0], abs=0.02)
        assert ours["mask_mAP50"] == pytest.approx(coco_eval.stats[1], abs=0.02)
