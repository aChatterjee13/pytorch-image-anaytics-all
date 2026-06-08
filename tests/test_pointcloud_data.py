import numpy as np
import torch

from image_analytics.core.config import DataConfig
from image_analytics.data.datasets import build_dataset
from image_analytics.data.datasets.pointcloud import load_point_cloud
from image_analytics.data.transforms.pointcloud import (
    NormalizePoints,
    build_pointcloud_transforms,
)


class TestSyntheticPointCloud:
    def test_classification_sample(self):
        ds = build_dataset(
            DataConfig(dataset="synthetic_pointcloud", kwargs={"size": 8, "num_points": 256}),
            split="train",
        )
        points, label = ds[0]
        assert points.shape == (256, 3)
        assert isinstance(label, int) and 0 <= label < 4
        assert ds.num_classes == 4

    def test_segmentation_sample(self):
        ds = build_dataset(
            DataConfig(
                dataset="synthetic_pointcloud",
                kwargs={"size": 8, "num_points": 256, "task": "segmentation"},
            ),
            split="train",
        )
        points, seg = ds[0]
        assert points.shape == (256, 3) and seg.shape == (256,)
        assert int(seg.min()) >= 0 and int(seg.max()) < 4

    def test_deterministic(self):
        ds = build_dataset(DataConfig(dataset="synthetic_pointcloud", kwargs={"size": 4, "num_points": 128}), split="train")
        assert torch.equal(ds[1][0], ds[1][0])

    def test_detection_sample(self):
        ds = build_dataset(
            DataConfig(dataset="synthetic_pointcloud_det", kwargs={"size": 8, "num_points": 1024}),
            split="train",
        )
        points, target = ds[0]
        assert points.shape == (1024, 3)
        m = target["boxes_3d"].shape[0]
        assert target["boxes_3d"].shape == (m, 7)
        assert target["labels"].shape == (m,) and m >= 1


class TestTransforms:
    def test_normalize_to_unit_sphere(self):
        points = torch.rand(100, 3) * 10 + 5
        out = NormalizePoints()(points)
        assert out.norm(dim=1).max() <= 1.0 + 1e-5
        assert out.mean(dim=0).abs().max() < 1e-5  # centered

    def test_pipeline_runs(self):
        tf = build_pointcloud_transforms(train=True, augment="strong")
        out = tf(torch.rand(64, 3))
        assert out.shape == (64, 3)


class TestLoaders:
    def test_npy_roundtrip(self, tmp_path):
        arr = np.random.rand(50, 3).astype(np.float32)
        np.save(tmp_path / "cloud.npy", arr)
        loaded = load_point_cloud(tmp_path / "cloud.npy")
        assert loaded.shape == (50, 3)
        assert torch.allclose(loaded, torch.from_numpy(arr))

    def test_off_loader(self, tmp_path):
        path = tmp_path / "cube.off"
        path.write_text("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
        loaded = load_point_cloud(path)
        assert loaded.shape == (3, 3)

    def test_folder_dataset(self, tmp_path):
        for cls in ("a", "b"):
            d = tmp_path / "train" / cls
            d.mkdir(parents=True)
            for i in range(2):
                np.save(d / f"{i}.npy", np.random.rand(80, 3).astype(np.float32))
        ds = build_dataset(
            DataConfig(dataset="pointcloud_folder", root=str(tmp_path), kwargs={"num_points": 64}),
            split="train",
        )
        assert len(ds) == 4 and ds.num_classes == 2
        points, label = ds[0]
        assert points.shape == (64, 3)
