"""TorchServe handler — preprocessing rebuilt from the archived config.

A single thin handler for all tasks: ``initialize`` loads the archived
``config.yaml`` + checkpoint and rebuilds the *eval* transform pipeline from the
config, so serving preprocessing can never drift from training. It does not
subclass ``ts``' ``BaseHandler`` (so the module imports without TorchServe
installed); TorchServe drives it through the module-level ``handle`` entrypoint.

TorchServe has been in maintenance mode since 2024 — for new deployments BentoML
is a good alternative (wrap ``scripts/infer.py``'s predict path in a Service).

Package a model::

    torch-model-archiver --model-name clf --version 1.0 \\
        --serialized-file best.pt --handler image_analytics/serving/torchserve_handler.py \\
        --extra-files config.yaml
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path


class ImageAnalyticsHandler:
    def __init__(self) -> None:
        self.initialized = False
        self.model = None
        self.transform = None
        self.task = None
        self.device = "cpu"

    def initialize(self, context) -> None:
        import torch

        from image_analytics.core.config import load_config

        model_dir = Path(context.system_properties.get("model_dir", "."))
        config = load_config(model_dir / "config.yaml")
        self.task = config.task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = _build_model(config).to(self.device).eval()
        ckpt = next(model_dir.glob("*.pt"), None)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.transform = _build_eval_transform(config)
        self.initialized = True

    def preprocess(self, data):
        import torch
        from PIL import Image

        images = []
        for row in data:
            raw = row.get("body") or row.get("data")
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append(self.transform(image))
        return torch.stack(images).to(self.device)

    def inference(self, x):
        import torch

        with torch.no_grad():
            return self.model(x)

    def postprocess(self, outputs):
        import torch

        if self.task == "classification":
            probs = outputs.softmax(dim=1)
            top = probs.topk(min(5, probs.shape[1]), dim=1)
            return [
                {"labels": labels.tolist(), "scores": scores.tolist()}
                for scores, labels in zip(top.values, top.indices)
            ]
        if self.task == "segmentation":
            return [{"mask": m.tolist()} for m in outputs.argmax(dim=1)]
        # detection: model already returns prediction dicts
        return [
            {k: v.tolist() for k, v in pred.items() if torch.is_tensor(v)}
            for pred in outputs
        ]

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)
        return self.postprocess(self.inference(self.preprocess(data)))


def _build_model(config):
    from image_analytics.serving.infer_utils import build_task_model

    return build_task_model(config)


def _build_eval_transform(config):
    from image_analytics.serving.infer_utils import build_eval_transform

    return build_eval_transform(config)


_service = ImageAnalyticsHandler()


def handle(data, context):
    """TorchServe entrypoint."""
    if data is None:
        return None
    return json.loads(json.dumps(_service.handle(data, context)))  # ensure JSON-safe
