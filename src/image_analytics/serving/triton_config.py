"""NVIDIA Triton ``config.pbtxt`` generator — pure templating, no deps.

Emits a model-repository config (onnxruntime backend, dynamic batching,
instance groups) from a :class:`TritonModelConfig` dataclass, so it is fully
unit-testable and never drifts from a hand-written file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TritonModelConfig:
    name: str
    input_dims: list[int]                 # without the batch axis, e.g. [3, 224, 224]
    output_dims: list[int]
    input_name: str = "input"
    output_name: str = "output"
    backend: str = "onnxruntime"
    data_type: str = "TYPE_FP32"
    max_batch_size: int = 8
    dynamic_batching: bool = True
    preferred_batch_sizes: list[int] = field(default_factory=lambda: [4, 8])
    max_queue_delay_us: int = 100
    instance_count: int = 1
    instance_kind: str = "KIND_CPU"        # or KIND_GPU


def _dims(dims: list[int]) -> str:
    return "[ " + ", ".join(str(d) for d in dims) + " ]"


def generate_triton_config(config: TritonModelConfig) -> str:
    """Render a Triton ``config.pbtxt`` for an ONNX model."""
    lines = [
        f'name: "{config.name}"',
        f'backend: "{config.backend}"',
        f"max_batch_size: {config.max_batch_size}",
        "input [",
        "  {",
        f'    name: "{config.input_name}"',
        f"    data_type: {config.data_type}",
        f"    dims: {_dims(config.input_dims)}",
        "  }",
        "]",
        "output [",
        "  {",
        f'    name: "{config.output_name}"',
        f"    data_type: {config.data_type}",
        f"    dims: {_dims(config.output_dims)}",
        "  }",
        "]",
    ]
    if config.dynamic_batching:
        lines += [
            "dynamic_batching {",
            f"  preferred_batch_size: {_dims(config.preferred_batch_sizes)}",
            f"  max_queue_delay_microseconds: {config.max_queue_delay_us}",
            "}",
        ]
    lines += [
        "instance_group [",
        "  {",
        f"    count: {config.instance_count}",
        f"    kind: {config.instance_kind}",
        "  }",
        "]",
    ]
    return "\n".join(lines) + "\n"


def write_triton_config(config: TritonModelConfig, path) -> str:
    """Write ``config.pbtxt`` and return the rendered text."""
    from pathlib import Path

    text = generate_triton_config(config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return text
