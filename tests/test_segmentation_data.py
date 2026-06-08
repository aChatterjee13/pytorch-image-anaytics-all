import numpy as np
import torch
from PIL import Image

from image_analytics.core.config import DataConfig
from image_analytics.data.datasets import build_dataset
from image_analytics.data.transforms.segmentation import build_segmentation_transforms


def _seg_cfg(**kwargs):
    return DataConfig(dataset="synthetic_shapes_seg", kwargs=kwargs)


class TestSyntheticSemantic:
    def test_sample_shapes_and_dtype(self):
        tf = build_segmentation_transforms(64, train=True, normalize="imagenet")
        ds = build_dataset(_seg_cfg(size=8, image_size=80), split="train", transform=tf)
        image, mask = ds[0]
        assert image.shape == (3, 64, 64) and image.dtype == torch.float32
        assert mask.shape == (64, 64) and mask.dtype == torch.int64
        assert ds.num_classes == 4  # background + 3 shapes

    def test_mask_labels_in_range(self):
        ds = build_dataset(_seg_cfg(size=8, image_size=72), split="train")
        for i in range(len(ds)):
            _, mask = ds[i]
            assert int(mask.min()) >= 0 and int(mask.max()) <= 3

    def test_deterministic(self):
        ds = build_dataset(_seg_cfg(size=4, image_size=64), split="train")
        a_img, a_mask = ds[1]
        b_img, b_mask = ds[1]
        assert torch.equal(a_img, b_img) and torch.equal(a_mask, b_mask)

    def test_pixel_consistent_with_detection_fixture(self):
        """The semantic fixture shares images with the detection fixture."""
        det = build_dataset(
            DataConfig(dataset="synthetic_shapes", kwargs={"size": 6, "image_size": 80}),
            split="train",
        )
        seg = build_dataset(_seg_cfg(size=6, image_size=80), split="train")
        for i in range(6):
            assert torch.equal(det[i][0], seg[i][0])

    def test_splits_differ(self):
        train = build_dataset(_seg_cfg(size=4, image_size=64), split="train")
        val = build_dataset(_seg_cfg(size=4, image_size=64), split="val")
        assert not torch.equal(train[0][0], val[0][0])


class TestSyntheticInstance:
    def test_target_structure(self):
        tf = build_segmentation_transforms(64, train=False, instance=True)
        ds = build_dataset(
            DataConfig(dataset="synthetic_shapes_instance", kwargs={"size": 8, "image_size": 80}),
            split="train",
            transform=tf,
        )
        image, target = ds[1]
        assert image.shape == (3, 64, 64)
        n = target["boxes"].shape[0]
        assert target["masks"].shape == (n, 64, 64)
        assert target["masks"].dtype == torch.uint8
        assert target["labels"].shape == (n,)
        assert n >= 1


class TestSemanticFolderDataset:
    def _make_dataset(self, root, split="train", n=3, size=16):
        img_dir = root / "images" / split
        mask_dir = root / "masks" / split
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(
                np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            ).save(img_dir / f"img_{i}.jpg")
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[: size // 2] = 1
            mask[size // 2 :, : size // 2] = 255  # ignore region
            Image.fromarray(mask, mode="L").save(mask_dir / f"img_{i}.png")

    def test_pairs_and_transform(self, tmp_path):
        self._make_dataset(tmp_path)
        tf = build_segmentation_transforms(8, train=False, normalize="none")
        ds = build_dataset(
            DataConfig(
                dataset="segmentation_folder",
                root=str(tmp_path),
                kwargs={"classes": ["background", "fg"]},
            ),
            split="train",
            transform=tf,
        )
        assert len(ds) == 3 and ds.num_classes == 2
        image, mask = ds[0]
        assert image.shape == (3, 8, 8) and image.dtype == torch.float32
        assert mask.shape == (8, 8)
        assert 255 in set(mask.flatten().tolist())  # ignore label preserved
