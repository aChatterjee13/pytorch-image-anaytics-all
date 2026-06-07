import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from image_analytics.core.callbacks import Callback, CheckpointCallback, EarlyStopping
from image_analytics.core.evaluator import ClassificationEvaluator
from image_analytics.core.trainer import Trainer, resolve_device, seed_everything


class TinyNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(2 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def loaders():
    """Linearly separable 2-class problem: class = sign of channel-0 mean."""
    seed_everything(7)
    x = torch.randn(128, 2, 8, 8)
    y = (x[:, 0].mean(dim=(1, 2)) > 0).long()
    ds = TensorDataset(x, y)
    train = DataLoader(ds, batch_size=32, shuffle=True)
    val = DataLoader(ds, batch_size=32)
    return train, val


def make_trainer(tmp_path, **kwargs):
    model = TinyNet()
    kwargs.setdefault("optimizer", torch.optim.Adam(model.parameters(), lr=1e-2))
    kwargs.setdefault("criterion", nn.CrossEntropyLoss())
    kwargs.setdefault("evaluator", ClassificationEvaluator(num_classes=2))
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("output_dir", tmp_path)
    return Trainer(model, **kwargs)


def test_fit_learns_separable_problem(tmp_path, loaders):
    train, val = loaders
    trainer = make_trainer(tmp_path)
    metrics = trainer.fit(train, val, epochs=5)
    assert metrics["val/accuracy"] > 0.9
    assert "train/loss" in metrics
    assert "val/loss" in metrics


def test_checkpoint_save_load(tmp_path, loaders):
    train, val = loaders
    trainer = make_trainer(tmp_path)
    trainer.fit(train, val, epochs=1)
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)
    assert path.exists()

    fresh = make_trainer(tmp_path)
    fresh.load_checkpoint(path, resume=True)
    assert fresh.start_epoch == 1
    assert fresh.global_step == trainer.global_step
    for p1, p2 in zip(fresh.module.parameters(), trainer.module.parameters()):
        torch.testing.assert_close(p1, p2)


def test_resume_skips_completed_epochs(tmp_path, loaders):
    train, val = loaders
    trainer = make_trainer(tmp_path)
    trainer.fit(train, val, epochs=2)
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    resumed = make_trainer(tmp_path)
    resumed.load_checkpoint(path, resume=True)
    resumed.fit(train, val, epochs=2)  # already at epoch 2 -> no extra steps
    assert resumed.global_step == trainer.global_step


def test_checkpoint_callback_writes_best_and_last(tmp_path, loaders):
    train, val = loaders
    ckpt_dir = tmp_path / "checkpoints"
    trainer = make_trainer(
        tmp_path,
        callbacks=[CheckpointCallback(dirpath=ckpt_dir, monitor="val/accuracy")],
    )
    trainer.fit(train, val, epochs=2)
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()


def test_early_stopping(tmp_path, loaders):
    train, val = loaders

    class NeverImproves(Callback):
        # Overwrite the monitored metric after validation so it cannot improve.
        def on_epoch_end(self, trainer):
            trainer.metrics["val/accuracy"] = 0.5

    stopper = EarlyStopping(monitor="val/accuracy", patience=2)
    trainer = make_trainer(tmp_path, callbacks=[NeverImproves(), stopper])
    trainer.fit(train, val, epochs=20)
    assert trainer.should_stop
    assert trainer.epoch < 19  # stopped well before max epochs


def test_grad_clip_runs(tmp_path, loaders):
    train, val = loaders
    trainer = make_trainer(tmp_path, grad_clip=0.5)
    metrics = trainer.fit(train, epochs=1)
    assert metrics["train/loss"] > 0


def test_fit_without_optimizer_raises(tmp_path, loaders):
    train, _ = loaders
    trainer = Trainer(TinyNet(), device="cpu", output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="optimizer"):
        trainer.fit(train, epochs=1)


def test_validate_standalone(tmp_path, loaders):
    _, val = loaders
    trainer = make_trainer(tmp_path)
    metrics = trainer.validate(val)
    assert set(metrics) >= {"val/loss", "val/accuracy"}


def test_callback_hooks_fire_in_order(tmp_path, loaders):
    train, val = loaders
    calls = []

    class Recorder(Callback):
        def on_fit_start(self, trainer):
            calls.append("fit_start")

        def on_epoch_start(self, trainer):
            calls.append("epoch_start")

        def on_epoch_end(self, trainer):
            calls.append("epoch_end")

        def on_fit_end(self, trainer):
            calls.append("fit_end")

    trainer = make_trainer(tmp_path, callbacks=[Recorder()])
    trainer.fit(train, val, epochs=2)
    assert calls == ["fit_start", "epoch_start", "epoch_end", "epoch_start", "epoch_end", "fit_end"]


def test_resolve_device_explicit():
    assert resolve_device("cpu").type == "cpu"
