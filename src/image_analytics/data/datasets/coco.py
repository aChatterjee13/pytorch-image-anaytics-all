"""COCO-format object detection dataset (pycocotools-backed).

Default layout (override via ``ann_file`` / ``image_dir`` kwargs):

    root/
        annotations/instances_{split}.json
        {split}/  *.jpg

Category ids are remapped to contiguous 0-based labels (COCO ids are sparse:
1..90 with gaps). Crowd annotations are excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from image_analytics.core.registry import DATASETS


@DATASETS.register("coco_detection")
class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        ann_file: str | None = None,
        image_dir: str | None = None,
        filter_empty: bool = True,
    ) -> None:
        from pycocotools.coco import COCO  # deferred: heavy import

        root_path = Path(root)
        self.image_dir = Path(image_dir) if image_dir else root_path / split
        ann_path = (
            Path(ann_file)
            if ann_file
            else root_path / "annotations" / f"instances_{split}.json"
        )
        if not ann_path.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")

        self.coco = COCO(str(ann_path))
        self.transform = transform

        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.classes = [self.coco.cats[cid]["name"] for cid in cat_ids]

        self.ids = sorted(self.coco.imgs.keys())
        if filter_empty:
            self.ids = [
                img_id
                for img_id in self.ids
                if any(
                    not ann.get("iscrowd", 0)
                    for ann in self.coco.loadAnns(self.coco.getAnnIds(img_id))
                )
            ]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        info = self.coco.imgs[img_id]
        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")

        boxes, labels = [], []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(img_id)):
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]  # COCO is XYWH
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                format="XYXY",
                canvas_size=(info["height"], info["width"]),
            ),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target
