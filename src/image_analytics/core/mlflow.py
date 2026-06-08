"""MLflow experiment tracking as a Trainer callback (``[serve]`` extra).

Opt in with ``training.mlflow: true``. The callback opens a run named after the
experiment, logs the flattened config as params, the per-epoch
``trainer.metrics``, and (on completion) the checkpoints + resolved config as
artifacts — optionally registering the final model. It is a no-op when mlflow
is not installed or on non-main DDP ranks, so it never blocks training. The
tracking URI comes from the standard ``MLFLOW_TRACKING_URI`` environment
variable (defaults to a local ``./mlruns`` file store).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from image_analytics.core.callbacks import Callback
from image_analytics.core.config import ExperimentConfig, to_dict

logger = logging.getLogger("image_analytics")


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config dict to dotted keys for ``mlflow.log_params``."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


class MLflowCallback(Callback):
    """Track an experiment to MLflow. Safe no-op without mlflow / off main rank."""

    def __init__(
        self,
        config: ExperimentConfig,
        registered_model_name: str | None = None,
        log_model: bool = False,
    ) -> None:
        self.config = config
        self.registered_model_name = registered_model_name
        self.log_model = log_model or registered_model_name is not None
        self._mlflow = None
        self._active = False

    def _load(self):
        if self._mlflow is None:
            try:
                import mlflow

                self._mlflow = mlflow
            except ImportError:
                logger.warning(
                    "training.mlflow is set but mlflow is not installed; "
                    "skipping tracking (pip install 'image-analytics[serve]')"
                )
                self._mlflow = False
        return self._mlflow or None

    def on_fit_start(self, trainer) -> None:
        if not trainer.is_main_process:
            return
        mlflow = self._load()
        if mlflow is None:
            return
        mlflow.set_experiment(self.config.experiment_name)
        mlflow.start_run(run_name=self.config.experiment_name)
        params = _flatten(to_dict(self.config))
        # mlflow rejects params over 500 chars and None-valued keys are noise.
        mlflow.log_params(
            {k: str(v)[:500] for k, v in params.items() if v is not None}
        )
        self._active = True

    def on_epoch_end(self, trainer) -> None:
        if not self._active:
            return
        metrics = {k: float(v) for k, v in trainer.metrics.items()}
        if metrics:
            self._mlflow.log_metrics(metrics, step=trainer.epoch)

    def on_fit_end(self, trainer) -> None:
        if not self._active:
            return
        mlflow = self._mlflow
        output_dir = Path(trainer.output_dir)
        for artifact in ("config.yaml", "checkpoints/best.pt", "checkpoints/last.pt"):
            path = output_dir / artifact
            if path.exists():
                mlflow.log_artifact(str(path))
        if self.log_model:
            try:
                mlflow.pytorch.log_model(
                    trainer.module,
                    artifact_path="model",
                    registered_model_name=self.registered_model_name,
                )
            except Exception as exc:  # pragma: no cover - registry/env dependent
                logger.warning("MLflow log_model failed: %s", exc)
        mlflow.end_run()
        self._active = False


def maybe_mlflow_callback(config: ExperimentConfig) -> list[Callback]:
    """Return ``[MLflowCallback]`` when ``training.mlflow`` is set, else ``[]``."""
    if not getattr(config.training, "mlflow", False):
        return []
    return [
        MLflowCallback(
            config,
            registered_model_name=config.training.mlflow_registered_model,
        )
    ]
