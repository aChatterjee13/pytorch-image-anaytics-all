import numpy as np
import pytest
import torch
from PIL import Image

from image_analytics.core.config import DataConfig
from image_analytics.data.datasets import DATASETS, build_dataset
from image_analytics.data.samplers import build_balanced_sampler, extract_targets


class TestBuildDataset:
    def test_fake_dataset_with_kwargs(self):
        config = DataConfig(
            dataset="fake",
            kwargs={"size": 32, "image_size": [3, 32, 32], "num_classes": 4},
        )
        ds = build_dataset(config, split="train")
        assert len(ds) == 32
        img, label = ds[0]
        assert 0 <= int(label) < 4

    def test_train_val_splits_differ(self):
        config = DataConfig(dataset="fake", kwargs={"size": 16, "image_size": [3, 16, 16]})
        train = build_dataset(config, split="train")
        val = build_dataset(config, split="val")
        t0 = np.asarray(train[0][0], dtype=np.float32)
        v0 = np.asarray(val[0][0], dtype=np.float32)
        assert not np.allclose(t0, v0)

    def test_multispectral_args_not_leaked_to_rgb_datasets(self):
        # bands/normalize must not reach factories that don't accept them
        config = DataConfig(
            dataset="fake",
            bands=[0, 1, 2],
            normalize="imagenet",
            kwargs={"size": 8, "image_size": [3, 16, 16]},
        )
        ds = build_dataset(config, split="train")  # would TypeError if leaked
        assert len(ds) == 8


class TestMultiLabelCsv:
    @pytest.fixture
    def csv_root(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(4):
            Image.new("RGB", (32, 32), color=(i * 60, 0, 0)).save(img_dir / f"img{i}.jpg")
        rows = ["filepath,cat,dog,bird"]
        labels = [(1, 0, 1), (0, 1, 0), (1, 1, 0), (0, 0, 0)]
        for i, lab in enumerate(labels):
            rows.append(f"images/img{i}.jpg,{lab[0]},{lab[1]},{lab[2]}")
        (tmp_path / "train.csv").write_text("\n".join(rows))
        return tmp_path

    def test_loads_and_targets_float(self, csv_root):
        ds = DATASETS.build("multilabel_csv", root=str(csv_root), split="train")
        assert len(ds) == 4
        assert ds.classes == ["cat", "dog", "bird"]
        img, target = ds[0]
        assert target.dtype == torch.float32
        torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 1.0]))

    def test_missing_split_raises(self, csv_root):
        with pytest.raises(FileNotFoundError):
            DATASETS.build("multilabel_csv", root=str(csv_root), split="val")


class TestMultispectral:
    @pytest.fixture
    def geotiff_root(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            for cls in ("forest", "water"):
                cls_dir = tmp_path / split / cls
                cls_dir.mkdir(parents=True)
                for i in range(3):
                    data = rng.integers(0, 10000, size=(6, 32, 32), dtype=np.uint16)
                    with rasterio.open(
                        cls_dir / f"tile_{i}.tif",
                        "w",
                        driver="GTiff",
                        height=32,
                        width=32,
                        count=6,
                        dtype="uint16",
                    ) as dst:
                        dst.write(data)
        return tmp_path

    def test_loads_16bit_all_bands(self, geotiff_root):
        ds = DATASETS.build("multispectral", root=str(geotiff_root), split="train")
        assert len(ds) == 6
        assert ds.classes == ["forest", "water"]
        x, label = ds[0]
        assert x.shape == (6, 32, 32)
        assert x.dtype == torch.float32
        assert 0.0 <= x.min() and x.max() <= 1.0  # percentile-normalized
        assert label in (0, 1)

    def test_band_selection(self, geotiff_root):
        ds = DATASETS.build(
            "multispectral", root=str(geotiff_root), split="train", bands=[0, 3, 5]
        )
        x, _ = ds[0]
        assert x.shape == (3, 32, 32)
        assert ds.num_bands == 3

    def test_via_build_dataset_config(self, geotiff_root):
        config = DataConfig(
            dataset="multispectral",
            root=str(geotiff_root),
            normalize="percentile",
            bands=[0, 1],
        )
        ds = build_dataset(config, split="val")
        x, _ = ds[0]
        assert x.shape == (2, 32, 32)

    def test_zscore_requires_stats(self, geotiff_root):
        with pytest.raises(ValueError, match="zscore"):
            DATASETS.build(
                "multispectral", root=str(geotiff_root), split="train", normalize="zscore"
            )

    def test_imagenet_normalize_rejected(self, geotiff_root):
        with pytest.raises(ValueError, match="multispectral"):
            DATASETS.build(
                "multispectral", root=str(geotiff_root), split="train", normalize="imagenet"
            )

    def test_targets_property(self, geotiff_root):
        ds = DATASETS.build("multispectral", root=str(geotiff_root), split="train")
        assert ds.targets == [0, 0, 0, 1, 1, 1]


class TestBalancedSampler:
    def test_balances_skewed_classes(self):
        class Skewed(torch.utils.data.Dataset):
            targets = [0] * 90 + [1] * 10

            def __len__(self):
                return 100

            def __getitem__(self, i):
                return torch.zeros(1), self.targets[i]

        generator = torch.Generator().manual_seed(0)
        sampler = build_balanced_sampler(Skewed(), num_samples=2000, generator=generator)
        targets = Skewed.targets
        drawn = [targets[i] for i in sampler]
        minority_share = sum(drawn) / len(drawn)
        assert 0.4 < minority_share < 0.6  # ~uniform despite 9:1 imbalance

    def test_extract_targets_fallback_iterates(self):
        ds = torch.utils.data.TensorDataset(
            torch.zeros(6, 1), torch.tensor([0, 0, 1, 1, 2, 2])
        )
        assert extract_targets(ds) == [0, 0, 1, 1, 2, 2]
