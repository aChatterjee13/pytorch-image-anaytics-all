import pytest
import torch

import image_analytics.backbones  # noqa: F401  (register backbones)
import image_analytics.segmentation.semantic  # noqa: F401  (register seg models)
from image_analytics.backbones.registry import build_backbone
from image_analytics.core.config import BackboneConfig
from image_analytics.core.registry import MODELS


def _unet_backbone():
    return build_backbone(
        BackboneConfig(
            name="resnet18", pretrained=False, features_only=True,
            kwargs={"out_indices": (0, 1, 2, 3, 4)},
        )
    )


def _deeplab_backbone():
    return build_backbone(
        BackboneConfig(
            name="resnet18", pretrained=False, features_only=True,
            kwargs={"out_indices": (1, 4), "output_stride": 16},
        )
    )


class TestUNet:
    @pytest.mark.parametrize("size", [64, 96, 100])
    def test_logits_match_input_resolution(self, size):
        model = MODELS.build("unet", backbone=_unet_backbone(), num_classes=4).eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, size, size))
        assert y.shape == (2, 4, size, size)

    def test_decoder_channels_must_match_levels(self):
        with pytest.raises(ValueError, match="decoder_channels"):
            MODELS.build(
                "unet", backbone=_unet_backbone(), num_classes=4,
                decoder_channels=(64, 32),  # only 2 for a 5-level encoder
            )

    def test_pooled_backbone_rejected(self):
        bb = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
        with pytest.raises(ValueError, match="pyramid"):
            MODELS.build("unet", backbone=bb, num_classes=4)

    def test_gradients_flow(self):
        model = MODELS.build("unet", backbone=_unet_backbone(), num_classes=4).train()
        loss = model(torch.randn(2, 3, 64, 64)).mean()
        loss.backward()
        assert all(
            p.grad is not None for p in model.head.parameters()
        )


class TestDeepLab:
    @pytest.mark.parametrize("size", [64, 96, 128])
    def test_logits_match_input_resolution(self, size):
        model = MODELS.build(
            "deeplabv3plus", backbone=_deeplab_backbone(), num_classes=4
        ).eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, size, size))
        assert y.shape == (2, 4, size, size)

    def test_gradients_flow(self):
        model = MODELS.build(
            "deeplabv3plus", backbone=_deeplab_backbone(), num_classes=4
        ).train()
        model(torch.randn(2, 3, 64, 64)).mean().backward()
        assert all(p.grad is not None for p in model.classifier.parameters())


class TestSMPWrapper:
    def test_offline_forward(self):
        model = MODELS.build(
            "smp", num_classes=4, arch="unet",
            encoder_name="resnet18", encoder_weights=None, in_channels=3,
        ).eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 64, 64))
        assert y.shape == (2, 4, 64, 64)

    def test_multichannel(self):
        model = MODELS.build(
            "smp", num_classes=3, arch="fpn",
            encoder_name="resnet18", encoder_weights=None, in_channels=13,
        ).eval()
        with torch.no_grad():
            y = model(torch.randn(1, 13, 64, 64))
        assert y.shape == (1, 3, 64, 64)


def _shapes_batch(n=4, image_size=64):
    """A fixed batch of structured, image-derived masks (overfittable)."""
    from image_analytics.core.config import DataConfig
    from image_analytics.data.datasets import build_dataset
    from image_analytics.data.transforms.segmentation import build_segmentation_transforms

    tf = build_segmentation_transforms(image_size, train=False, normalize="none")
    ds = build_dataset(
        DataConfig(dataset="synthetic_shapes_seg", kwargs={"size": n, "image_size": image_size}),
        split="train", transform=tf,
    )
    images = torch.stack([ds[i][0] for i in range(n)])
    masks = torch.stack([ds[i][1] for i in range(n)])
    return images, masks


@pytest.mark.slow
class TestSemanticLearns:
    """Overfit a fixed synthetic shapes batch — run on a GPU box (slow on CPU)."""

    def _overfit(self, model, iters=80):
        torch.manual_seed(0)
        images, masks = _shapes_batch()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        crit = torch.nn.CrossEntropyLoss()
        first = None
        for _ in range(iters):
            loss = crit(model(images), masks)
            opt.zero_grad(); loss.backward(); opt.step()
            first = first if first is not None else float(loss)
        assert float(loss) < first * 0.5, f"loss did not halve: {first:.4f} -> {float(loss):.4f}"

    def test_unet_overfits(self):
        self._overfit(MODELS.build("unet", backbone=_unet_backbone(), num_classes=4).train())

    def test_deeplab_overfits(self):
        self._overfit(
            MODELS.build("deeplabv3plus", backbone=_deeplab_backbone(), num_classes=4).train()
        )
