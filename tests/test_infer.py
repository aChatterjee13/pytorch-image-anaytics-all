import numpy as np
import torch
from PIL import Image

from image_analytics.core.config import config_from_dict
from image_analytics.serving.infer_utils import (
    build_eval_transform,
    build_task_model,
    classification_topk,
    detection_to_dict,
    load_checkpoint_into,
    segmentation_summary,
)


def _clf_config():
    return config_from_dict({
        "task": "classification", "experiment_name": "infer",
        "model": {"name": "classifier", "num_classes": 4,
                  "backbone": {"name": "resnet18", "pretrained": False}},
        "data": {"dataset": "fake", "image_size": 32, "normalize": "imagenet"},
        "training": {"device": "cpu"},
    })


class TestPostprocess:
    def test_classification_topk(self):
        out = classification_topk(torch.tensor([[0.1, 5.0, 0.2, 0.3]]), topk=2)
        assert out["labels"][0] == 1
        assert len(out["labels"]) == 2 and len(out["scores"]) == 2

    def test_detection_rescales_boxes(self):
        preds = [{
            "boxes": torch.tensor([[0.0, 0, 32, 32], [0.0, 0, 16, 16]]),
            "scores": torch.tensor([0.9, 0.1]),
            "labels": torch.tensor([1, 2]),
        }]
        out = detection_to_dict(preds, orig_size=(64, 128), image_size=32, score_thresh=0.3)
        # sx = 64/32 = 2, sy = 128/32 = 4; low-score box filtered out
        assert out["boxes"] == [[0.0, 0.0, 64.0, 128.0]]
        assert out["labels"] == [1]

    def test_segmentation_summary(self):
        logits = torch.full((1, 3, 4, 4), -10.0)
        logits[0, 1] = 10.0  # everything predicted class 1
        assert segmentation_summary(logits) == {"pixel_counts": {1: 16}}


class TestInferencePipeline:
    def test_build_eval_transform_classification(self):
        tf = build_eval_transform(_clf_config())
        img = Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8"))
        assert tf(img).shape == (3, 32, 32)

    def test_full_single_image_inference(self, tmp_path):
        config = _clf_config()
        model = build_task_model(config).eval()
        torch.save({"model": model.state_dict()}, tmp_path / "ckpt.pt")

        loaded = build_task_model(config)
        load_checkpoint_into(loaded, str(tmp_path / "ckpt.pt"))
        loaded.eval()

        tf = build_eval_transform(config)
        img = Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8"))
        with torch.no_grad():
            out = loaded(tf(img).unsqueeze(0))
        result = classification_topk(out, topk=3)
        assert len(result["labels"]) == 3
        assert all(0 <= label < 4 for label in result["labels"])
