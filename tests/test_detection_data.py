import json

import numpy as np
import pytest
import torch
from PIL import Image

from image_analytics.core.config import DataConfig
from image_analytics.data.collate import detection_collate
from image_analytics.data.datasets import DATASETS, build_dataset
from image_analytics.data.transforms.detection import build_detection_transforms


class TestSyntheticShapes:
    def test_sample_structure(self):
        ds = DATASETS.build("synthetic_shapes", size=8, image_size=96)
        image, target = ds[0]
        assert image.shape == (3, 96, 96)
        assert 0.0 <= image.min() and image.max() <= 1.0
        n = len(target["labels"])
        assert 1 <= n <= 3
        assert target["boxes"].shape == (n, 4)
        assert all(0 <= int(l) < 3 for l in target["labels"])

    def test_boxes_valid_and_inside_canvas(self):
        ds = DATASETS.build("synthetic_shapes", size=32, image_size=64)
        for i in range(len(ds)):
            _, target = ds[i]
            boxes = torch.as_tensor(target["boxes"])
            assert (boxes[:, 2] > boxes[:, 0]).all()
            assert (boxes[:, 3] > boxes[:, 1]).all()
            assert (boxes >= 0).all() and (boxes <= 64).all()

    def test_deterministic_per_index(self):
        ds1 = DATASETS.build("synthetic_shapes", size=4)
        ds2 = DATASETS.build("synthetic_shapes", size=4)
        img1, t1 = ds1[2]
        img2, t2 = ds2[2]
        torch.testing.assert_close(img1, img2)
        torch.testing.assert_close(torch.as_tensor(t1["boxes"]), torch.as_tensor(t2["boxes"]))

    def test_splits_differ(self):
        train = DATASETS.build("synthetic_shapes", size=4, split="train")
        val = DATASETS.build("synthetic_shapes", size=4, split="val")
        assert not torch.allclose(train[0][0], val[0][0])

    def test_shapes_brighter_than_background(self):
        # The drawn shape pixels must stand out for the task to be learnable
        ds = DATASETS.build("synthetic_shapes", size=8, image_size=64)
        image, target = ds[0]
        box = torch.as_tensor(target["boxes"])[0].long()
        inside = image[:, box[1] : box[3], box[0] : box[2]].mean()
        assert inside > image.mean() - 0.05


class TestDetectionTransforms:
    def test_resize_updates_boxes(self):
        ds = DATASETS.build("synthetic_shapes", size=4, image_size=96)
        tf = build_detection_transforms(48, train=False)
        _, raw_target = ds[0]
        image, target = tf(*ds[0])
        assert image.shape == (3, 48, 48)
        boxes = torch.as_tensor(target["boxes"])
        raw_boxes = torch.as_tensor(raw_target["boxes"])
        torch.testing.assert_close(boxes, raw_boxes / 2, rtol=0.05, atol=1.0)

    def test_hflip_keeps_box_count(self):
        ds = DATASETS.build("synthetic_shapes", size=4)
        tf = build_detection_transforms(96, train=True, hflip=1.0)
        _, raw = ds[1]
        _, target = tf(*ds[1])
        assert len(target["labels"]) == len(raw["labels"])

    def test_collate(self):
        ds = DATASETS.build("synthetic_shapes", size=4)
        tf = build_detection_transforms(64, train=False)
        samples = [tf(*ds[i]) for i in range(3)]
        images, targets = detection_collate(samples)
        assert images.shape == (3, 3, 64, 64)
        assert len(targets) == 3 and "boxes" in targets[0]


class TestCocoDataset:
    @pytest.fixture
    def coco_root(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "annotations").mkdir()
        rng = np.random.default_rng(0)
        images, annotations = [], []
        ann_id = 1
        for img_id in (1, 2):
            name = f"img{img_id}.jpg"
            Image.fromarray(
                rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
            ).save(tmp_path / "train" / name)
            images.append(
                {"id": img_id, "file_name": name, "width": 100, "height": 80}
            )
            for _ in range(2):
                x, y = int(rng.integers(0, 50)), int(rng.integers(0, 40))
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(rng.choice([1, 7])),  # sparse ids
                        "bbox": [x, y, 20, 15],
                        "area": 300,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        # One crowd annotation that must be filtered out
        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 1,
            }
        )
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 7, "name": "dog"},
            ],
        }
        (tmp_path / "annotations" / "instances_train.json").write_text(json.dumps(coco))
        return tmp_path

    def test_loads_and_remaps_labels(self, coco_root):
        ds = DATASETS.build("coco_detection", root=str(coco_root), split="train")
        assert len(ds) == 2
        assert ds.classes == ["cat", "dog"]
        image, target = ds[0]
        assert target["boxes"].shape[1] == 4
        assert set(target["labels"].tolist()) <= {0, 1}  # contiguous 0-based

    def test_crowd_excluded(self, coco_root):
        ds = DATASETS.build("coco_detection", root=str(coco_root), split="train")
        _, target = ds[0]  # image 1 has 2 normal + 1 crowd ann
        assert len(target["labels"]) == 2

    def test_via_build_dataset(self, coco_root):
        config = DataConfig(dataset="coco_detection", root=str(coco_root))
        tf = build_detection_transforms(64, train=False)
        ds = build_dataset(config, split="train", transform=tf)
        image, target = ds[0]
        assert image.shape == (3, 64, 64)

    def test_missing_annotations_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="annotation"):
            DATASETS.build("coco_detection", root=str(tmp_path), split="train")
