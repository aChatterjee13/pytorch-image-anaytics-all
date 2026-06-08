import pytest
import torch

import image_analytics.detection_3d  # noqa: F401  (register models)
from image_analytics.core.registry import MODELS

MODEL_NAMES = ["pointnet", "pointnet2", "dgcnn"]


@pytest.mark.parametrize("name", MODEL_NAMES)
class TestForward:
    def test_classification_shape(self, name):
        model = MODELS.build(name, num_classes=4, task="classification").eval()
        with torch.no_grad():
            out = model(torch.rand(2, 1024, 3))
        assert out.shape == (2, 4)

    def test_segmentation_shape(self, name):
        model = MODELS.build(name, num_classes=5, task="segmentation").eval()
        with torch.no_grad():
            out = model(torch.rand(2, 1024, 3))
        assert out.shape == (2, 5, 1024)  # channels-second, per point

    def test_gradients_flow(self, name):
        model = MODELS.build(name, num_classes=4, task="classification").train()
        model(torch.rand(2, 1024, 3)).sum().backward()
        assert any(p.grad is not None for p in model.parameters())


def test_pointnet_orthogonality_regularizer():
    model = MODELS.build("pointnet", num_classes=4, task="classification").train()
    model(torch.rand(2, 512, 3))
    reg = model.regularization_loss()
    assert torch.is_tensor(reg) and reg.item() >= 0


def test_invalid_task_rejected():
    with pytest.raises(ValueError, match="task"):
        MODELS.build("pointnet", num_classes=4, task="bogus")


@pytest.mark.slow
class TestPointModelsLearn:
    def _overfit(self, model, iters=60):
        from image_analytics.core.config import DataConfig
        from image_analytics.data.datasets import build_dataset

        torch.manual_seed(0)
        ds = build_dataset(
            DataConfig(dataset="synthetic_pointcloud", kwargs={"size": 8, "num_points": 512}),
            split="train",
        )
        points = torch.stack([ds[i][0] for i in range(8)])
        labels = torch.tensor([ds[i][1] for i in range(8)])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for _ in range(iters):
            out = model(points)
            loss = torch.nn.functional.cross_entropy(out, labels)
            reg = getattr(model, "regularization_loss", None)
            if reg is not None:
                loss = loss + reg()
            opt.zero_grad(); loss.backward(); opt.step()
            first = first if first is not None else float(loss)
        assert float(loss) < first * 0.5

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_overfits(self, name):
        self._overfit(MODELS.build(name, num_classes=4, task="classification").train())
