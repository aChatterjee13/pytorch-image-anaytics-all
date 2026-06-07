import pytest
import torch

from image_analytics.data.transforms.augmentations import build_transforms
from image_analytics.data.transforms.spectral import (
    AppendIndex,
    compute_index,
    minmax_normalize,
    normalized_difference,
    percentile_normalize,
    zscore_normalize,
)


class TestSpectralIndices:
    def test_normalized_difference(self):
        a = torch.tensor([[0.8]])
        b = torch.tensor([[0.2]])
        assert normalized_difference(a, b).item() == pytest.approx(0.6, abs=1e-6)

    def test_ndvi(self):
        x = torch.zeros(4, 2, 2)
        x[3] = 0.8  # nir
        x[2] = 0.2  # red
        ndvi = compute_index(x, "ndvi", {"nir": 3, "red": 2})
        assert ndvi.shape == (2, 2)
        assert ndvi[0, 0].item() == pytest.approx(0.6, abs=1e-6)

    def test_evi(self):
        x = torch.zeros(3, 1, 1)
        x[0] = 0.1  # blue
        x[1] = 0.2  # red
        x[2] = 0.8  # nir
        evi = compute_index(x, "evi", {"nir": 2, "red": 1, "blue": 0})
        expected = 2.5 * (0.8 - 0.2) / (0.8 + 6 * 0.2 - 7.5 * 0.1 + 1)
        assert evi.item() == pytest.approx(expected, rel=1e-4)

    def test_missing_band_raises(self):
        x = torch.zeros(2, 2, 2)
        with pytest.raises(KeyError, match="nir"):
            compute_index(x, "ndvi", {"red": 0})

    def test_unknown_index_raises(self):
        with pytest.raises(ValueError, match="Unknown spectral index"):
            compute_index(torch.zeros(2, 2, 2), "bogus", {})

    def test_append_index(self):
        x = torch.rand(4, 8, 8)
        out = AppendIndex("ndvi", {"nir": 3, "red": 2})(x)
        assert out.shape == (5, 8, 8)
        torch.testing.assert_close(out[:4], x)


class TestNormalization:
    def test_percentile_range(self):
        x = torch.rand(3, 32, 32) * 10000  # 16-bit-ish dynamic range
        out = percentile_normalize(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_percentile_clips_outliers(self):
        x = torch.ones(1, 10, 10)
        x[0, 0, 0] = 1e6  # extreme outlier
        out = percentile_normalize(x, 2, 98)
        assert out[0, 0, 0] == pytest.approx(1.0)  # clipped, not stretched

    def test_minmax(self):
        x = torch.rand(5, 16, 16) * 100 + 50
        out = minmax_normalize(x)
        for c in range(5):
            assert out[c].min().item() == pytest.approx(0.0, abs=1e-5)
            assert out[c].max().item() == pytest.approx(1.0, abs=1e-5)

    def test_zscore(self):
        x = torch.rand(2, 8, 8)
        mean = [0.5, 0.5]
        std = [0.25, 0.25]
        out = zscore_normalize(x, mean, std)
        torch.testing.assert_close(
            out, (x - 0.5) / (0.25 + 1e-8), rtol=1e-5, atol=1e-6
        )

    def test_zscore_channel_mismatch_raises(self):
        with pytest.raises(ValueError, match="channels"):
            zscore_normalize(torch.rand(3, 4, 4), [0.5], [0.2])


class TestAugmentations:
    def test_train_pipeline_pil(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        out = build_transforms(64, train=True)(img)
        assert out.shape == (3, 64, 64)
        assert out.dtype == torch.float32

    def test_eval_pipeline_deterministic(self):
        x = torch.rand(3, 100, 80)
        tf = build_transforms(64, train=False)
        torch.testing.assert_close(tf(x), tf(x))

    def test_multispectral_tensor_no_imagenet_stats(self):
        # 13-band float tensor through the pipeline with normalize="none"
        x = torch.rand(13, 72, 72)
        out = build_transforms(64, train=True, normalize="none")(x)
        assert out.shape == (13, 64, 64)

    def test_invalid_augment_raises(self):
        with pytest.raises(ValueError, match="augment"):
            build_transforms(64, augment="bogus")
