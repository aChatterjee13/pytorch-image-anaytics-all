import pytest
import torch
import torch.nn as nn

from image_analytics.backbones import (
    BACKBONES,
    ChannelAttentionInput,
    MultiChannelBackbone,
    TimmBackbone,
    adapt_first_conv,
    build_backbone,
)
from image_analytics.core.config import BackboneConfig


class TestTimmBackbone:
    def test_pooled_features(self):
        backbone = TimmBackbone("resnet18", pretrained=False)
        assert backbone.feature_dim == 512
        out = backbone(torch.rand(2, 3, 64, 64))
        assert out.shape == (2, 512)

    def test_multichannel_input(self):
        backbone = TimmBackbone("resnet18", pretrained=False, in_channels=13)
        out = backbone(torch.rand(2, 13, 64, 64))
        assert out.shape == (2, 512)

    def test_features_only_pyramid(self):
        backbone = TimmBackbone("resnet18", pretrained=False, features_only=True)
        features = backbone(torch.rand(1, 3, 64, 64))
        assert isinstance(features, list)
        assert len(features) == len(backbone.feature_channels)
        for fmap, channels in zip(features, backbone.feature_channels):
            assert fmap.shape[1] == channels


class TestRegistry:
    def test_families_registered(self):
        for name in ("resnet50", "efficientnet_b0", "convnext_tiny", "swin_tiny", "vit_base"):
            assert name in BACKBONES

    def test_build_from_config(self):
        backbone = build_backbone(BackboneConfig(name="resnet18", pretrained=False))
        assert backbone(torch.rand(1, 3, 64, 64)).shape == (1, 512)

    def test_build_from_string_with_overrides(self):
        backbone = build_backbone("resnet18", pretrained=False, in_channels=4)
        assert backbone.feature_dim == 512
        assert backbone(torch.rand(1, 4, 64, 64)).shape == (1, 512)

    def test_timm_fallback_for_unregistered_name(self):
        # resnet10t is a valid timm model not in our registry
        assert "resnet10t" not in BACKBONES
        backbone = build_backbone(BackboneConfig(name="resnet10t", pretrained=False))
        assert backbone(torch.rand(1, 3, 64, 64)).shape[0] == 1

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="neither a registered backbone"):
            build_backbone(BackboneConfig(name="not_a_real_model_xyz", pretrained=False))


class TestMultiChannel:
    def test_channel_attention_shape_preserved(self):
        attn = ChannelAttentionInput(num_channels=13)
        x = torch.rand(2, 13, 32, 32)
        out = attn(x)
        assert out.shape == x.shape

    def test_channel_attention_bounded(self):
        attn = ChannelAttentionInput(num_channels=4)
        x = torch.rand(2, 4, 8, 8)
        out = attn(x)
        # Sigmoid gating: output is element-wise scaled down, never amplified
        assert (out.abs() <= x.abs() + 1e-6).all()

    def test_multichannel_backbone_wrapper(self):
        config = BackboneConfig(
            name="resnet18", pretrained=False, in_channels=6, channel_attention=True
        )
        backbone = build_backbone(config)
        assert isinstance(backbone, MultiChannelBackbone)
        assert backbone.feature_dim == 512
        out = backbone(torch.rand(2, 6, 64, 64))
        assert out.shape == (2, 512)

    def test_adapt_first_conv_kaiming(self):
        conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv = adapt_first_conv(conv, in_channels=13)
        assert new_conv.in_channels == 13
        torch.testing.assert_close(new_conv.weight[:, :3], conv.weight)
        out = new_conv(torch.rand(1, 13, 32, 32))
        assert out.shape[1] == 64

    def test_adapt_first_conv_mean(self):
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        new_conv = adapt_first_conv(conv, in_channels=5, init="mean")
        mean_filter = conv.weight.mean(dim=1)
        torch.testing.assert_close(new_conv.weight[:, 3], mean_filter)
        torch.testing.assert_close(new_conv.weight[:, 4], mean_filter)
        torch.testing.assert_close(new_conv.bias, conv.bias)

    def test_adapt_first_conv_reduce_channels(self):
        conv = nn.Conv2d(3, 8, kernel_size=3)
        new_conv = adapt_first_conv(conv, in_channels=1)
        torch.testing.assert_close(new_conv.weight[:, :1], conv.weight[:, :1])
