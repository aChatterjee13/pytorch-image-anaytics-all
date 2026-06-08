"""ONNX export with a mandatory onnxruntime parity gate (``[serve]`` extra).

Export is only considered successful once the ONNX graph reproduces the eager
model's outputs on CPU within ``atol`` (default 1e-4). Classifiers export whole;
detectors export the **raw heads** graph (backbone + neck + towers + heads),
keeping decode/NMS in Python — NMS-in-graph export is brittle across opsets, so
the recommended serving shape is a Triton ensemble (ONNX heads + a Python
postprocess step).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class RetinaNetRawHeads(nn.Module):
    """Wrap a RetinaNet-style detector (cls/reg towers + heads over an FPN) to
    return raw ``(cls_logits, box_deltas)`` per anchor — the ONNX-exportable
    part; decode + NMS stay in Python."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        for attr in ("backbone", "fpn", "cls_tower", "reg_tower", "cls_head", "reg_head"):
            if not hasattr(detector, attr):
                raise ValueError(
                    f"RetinaNetRawHeads expects a RetinaNet-style detector; missing {attr!r}"
                )
        self.detector = detector

    def forward(self, images: torch.Tensor):
        m = self.detector
        cls_per_level, reg_per_level = [], []
        for feature in m.fpn(m.backbone(images)):
            b, _, h, w = feature.shape
            cls = (
                m.cls_head(m.cls_tower(feature))
                .view(b, -1, m.num_classes, h, w).permute(0, 3, 4, 1, 2)
                .reshape(b, -1, m.num_classes)
            )
            reg = m.reg_head(m.reg_tower(feature)).view(b, -1, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
            cls_per_level.append(cls)
            reg_per_level.append(reg)
        return torch.cat(cls_per_level, dim=1), torch.cat(reg_per_level, dim=1)


def _load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "onnxruntime is required for the ONNX parity gate. "
            "Install it with: pip install 'image-analytics[serve]'"
        ) from exc
    return ort


def check_parity(
    model: nn.Module, example_input: torch.Tensor, onnx_path: str | Path, atol: float = 1e-4
) -> float:
    """Run torch vs onnxruntime on ``example_input`` and return max |Δ|; raise
    if it exceeds ``atol``."""
    ort = _load_onnxruntime()
    model.eval()
    with torch.no_grad():
        torch_out = model(example_input)
    torch_outs = torch_out if isinstance(torch_out, (tuple, list)) else (torch_out,)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {sess.get_inputs()[0].name: example_input.cpu().numpy()})

    max_diff = 0.0
    for t, o in zip(torch_outs, ort_outs):
        max_diff = max(max_diff, float(abs(t.detach().cpu().numpy() - o).max()))
    if max_diff > atol:
        raise RuntimeError(
            f"ONNX parity failed: max|Δ|={max_diff:.2e} exceeds atol={atol:.0e} "
            f"({onnx_path})"
        )
    return max_diff


def export_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
    input_name: str = "input",
    output_names: list[str] | None = None,
    dynamic_batch: bool = True,
    opset: int = 17,
    parity: bool = True,
    atol: float = 1e-4,
) -> Path:
    """Export ``model`` to ONNX (dynamic batch axis) and verify parity.

    Returns the output path only after the onnxruntime parity check passes.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        sample = model(example_input)
    n_out = len(sample) if isinstance(sample, (tuple, list)) else 1
    output_names = output_names or (["output"] if n_out == 1 else [f"output_{i}" for i in range(n_out)])

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {input_name: {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})

    torch.onnx.export(
        model, example_input, str(output_path),
        input_names=[input_name], output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset,
    )
    if parity:
        check_parity(model, example_input, output_path, atol=atol)
    return output_path


def build_exportable(model: nn.Module, task: str) -> nn.Module:
    """Return the ONNX-exportable module for a task: classifiers/segmenters
    export whole; RetinaNet-style detectors export raw heads."""
    if task == "detection" and hasattr(model, "cls_tower"):
        return RetinaNetRawHeads(model)
    return model
