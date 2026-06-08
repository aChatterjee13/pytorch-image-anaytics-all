import pytest
import torch

from image_analytics.core.config import BackboneConfig, ModelConfig, config_from_dict
from image_analytics.segmentation.instance.mask_rcnn import (
    paste_masks_in_image,
    project_masks_on_boxes,
)
from image_analytics.segmentation.train import build_segmentation_model, run


def tiny_mask_rcnn_config(**kwargs):
    defaults = dict(
        fpn_channels=32,
        box_head_dim=64,
        mask_head_dim=32,
        rpn_anchor_sizes=[[16], [32], [64], [128], [256]],
        rpn_post_nms_topk=[200, 100],
    )
    defaults.update(kwargs)
    return ModelConfig(
        name="mask_rcnn",
        num_classes=3,
        backbone=BackboneConfig(name="resnet18", pretrained=False, features_only=True),
        kwargs=defaults,
    )


def make_batch(batch_size=2, image_size=64):
    images = torch.rand(batch_size, 3, image_size, image_size)
    targets = []
    for _ in range(batch_size):
        masks = torch.zeros(2, image_size, image_size, dtype=torch.uint8)
        masks[0, 8:30, 8:30] = 1
        masks[1, 35:58, 35:60] = 1
        targets.append(
            {
                "boxes": torch.tensor([[8.0, 8, 30, 30], [35.0, 35, 60, 58]]),
                "labels": torch.tensor([0, 2]),
                "masks": masks,
            }
        )
    return images, targets


class TestMaskRCNN:
    def test_training_adds_mask_loss(self):
        model = build_segmentation_model(tiny_mask_rcnn_config()).train()
        losses = model(*make_batch())
        assert set(losses) == {
            "loss", "loss_rpn_cls", "loss_rpn_reg", "loss_cls", "loss_reg", "loss_mask",
        }
        for v in losses.values():
            assert torch.isfinite(v)
        assert losses["loss"].requires_grad

    def test_empty_targets(self):
        model = build_segmentation_model(tiny_mask_rcnn_config()).train()
        images, _ = make_batch()
        empty = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros(0, 64, 64, dtype=torch.uint8),
            }
            for _ in range(2)
        ]
        losses = model(images, empty)
        assert torch.isfinite(losses["loss"])

    def test_eval_returns_masks(self):
        model = build_segmentation_model(tiny_mask_rcnn_config()).eval()
        images, _ = make_batch()
        with torch.no_grad():
            preds = model(images)
        for p in preds:
            assert set(p) == {"boxes", "scores", "labels", "masks"}
            n = len(p["boxes"])
            assert p["masks"].shape == (n, 64, 64)
            assert p["masks"].dtype == torch.uint8


class TestMaskUtils:
    def test_project_masks_on_boxes_full_image_box(self):
        # A box covering the whole image should round-trip the mask (resampled).
        mask = torch.zeros(1, 16, 16, dtype=torch.uint8)
        mask[0, 4:12, 4:12] = 1
        box = torch.tensor([[0.0, 0, 16, 16]])
        out = project_masks_on_boxes(mask, box, output_size=16)
        assert out.shape == (1, 16, 16)
        assert out[0, 8, 8] == 1.0 and out[0, 0, 0] == 0.0

    def test_paste_masks_in_image(self):
        soft = torch.ones(1, 8, 8)  # fully-on mask
        boxes = torch.tensor([[10.0, 20, 30, 50]])
        out = paste_masks_in_image(soft, boxes, image_size=(64, 64))
        assert out.shape == (1, 64, 64) and out.dtype == torch.uint8
        assert out[0, 35, 20] == 1          # inside the box
        assert out[0, 5, 5] == 0            # outside the box

    def test_paste_clips_to_image_bounds(self):
        soft = torch.ones(1, 8, 8)
        boxes = torch.tensor([[-10.0, -10, 20, 20]])  # partially off-canvas
        out = paste_masks_in_image(soft, boxes, image_size=(32, 32))
        assert out.shape == (1, 32, 32)
        assert out[0, 10, 10] == 1


class TestMaskRCNNEndToEnd:
    def test_run_smoke(self, tmp_path):
        config = config_from_dict(
            {
                "task": "segmentation",
                "experiment_name": "mrcnn_smoke",
                "output_dir": str(tmp_path),
                "model": {
                    "name": "mask_rcnn",
                    "num_classes": 3,
                    "backbone": {"name": "resnet18", "pretrained": False, "features_only": True},
                    "neck": {"name": "fpn", "out_channels": 32},
                    "kwargs": {
                        "box_head_dim": 64,
                        "mask_head_dim": 32,
                        "rpn_anchor_sizes": [[16], [32], [64], [128], [256]],
                        "rpn_post_nms_topk": [100, 50],
                    },
                },
                "data": {
                    "dataset": "synthetic_shapes_instance",
                    "image_size": 64,
                    "batch_size": 4,
                    "num_workers": 0,
                    "kwargs": {"size": 12, "image_size": 64},
                },
                "training": {
                    "epochs": 1, "lr": 1e-3, "scheduler": "none",
                    "device": "cpu", "log_interval": 0, "monitor": "val/mask_mAP50",
                },
            }
        )
        metrics = run(config)
        assert "train/loss" in metrics
        assert "val/mask_mAP" in metrics and "val/mask_mAP50" in metrics
        assert (tmp_path / "mrcnn_smoke" / "checkpoints" / "last.pt").exists()

    def test_config_file_parses(self):
        from image_analytics.core.config import load_config

        config = load_config("configs/segmentation/mask_rcnn_shapes.yaml")
        assert config.model.name == "mask_rcnn"
        assert config.task == "segmentation"


@pytest.mark.slow
class TestMaskRCNNLearns:
    def test_overfit_single_batch(self):
        torch.manual_seed(0)
        config = tiny_mask_rcnn_config()
        config.backbone = BackboneConfig(
            name="resnet10t", pretrained=False, features_only=True,
            kwargs={"out_indices": (1, 2, 3, 4)},
        )
        model = build_segmentation_model(config).train()
        images, targets = make_batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for _ in range(30):
            losses = model(images, targets)
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            if first is None:
                first = float(losses["loss"])
        assert float(losses["loss"]) < first * 0.6
