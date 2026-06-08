import pytest
import torch

pytest.importorskip("ultralytics")

from image_analytics.detection.yolo import YOLOWrapper  # noqa: E402


@pytest.fixture(scope="module")
def wrapper():
    # .yaml = random-init architecture: no weight download, fully offline
    return YOLOWrapper("yolo11n.yaml", score_thresh=0.01).eval()


class TestYOLOWrapper:
    def test_registered(self):
        from image_analytics.core.registry import MODELS

        assert "yolo" in MODELS

    def test_eval_prediction_protocol(self, wrapper):
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            predictions = wrapper(images)
        assert len(predictions) == 2
        for p in predictions:
            assert set(p) == {"boxes", "scores", "labels"}
            assert p["boxes"].shape[1] == 4 if len(p["boxes"]) else True
            assert p["labels"].dtype == torch.int64

    def test_training_mode_redirects_to_native(self, wrapper):
        with pytest.raises(RuntimeError, match="train_native"):
            wrapper.train()(torch.rand(1, 3, 64, 64))
        wrapper.eval()

    def test_class_names_exposed(self, wrapper):
        names = wrapper.class_names
        assert isinstance(names, dict) and len(names) > 0
