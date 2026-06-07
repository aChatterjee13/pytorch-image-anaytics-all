"""Config parsing: typed dataclasses loaded from YAML.

Every experiment is reproducible via a YAML config:

    config = load_config("configs/classification/cifar10_resnet18.yaml")
    config = load_config(path, overrides=["training.lr=0.01", "data.batch_size=64"])

Unknown keys raise immediately (catches typos), and scalar values are coerced
to the annotated field type (YAML parses ``1e-4`` as a string, not a float).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackboneConfig:
    """Feature extractor configuration (resolved via the BACKBONES registry,
    falling back to any valid ``timm`` model name)."""

    name: str = "resnet50"
    pretrained: bool = True
    in_channels: int = 3
    features_only: bool = False          # pyramid features (detection/segmentation necks)
    channel_attention: bool = False      # SE-style attention over input channels (>3 band imagery)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "classifier"             # MODELS registry key
    num_classes: int = 1000
    dropout: float = 0.0
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    dataset: str = "cifar10"             # DATASETS registry key
    root: str = "data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    augment: str = "default"             # none | default | strong
    normalize: str = "imagenet"          # imagenet | percentile | minmax | zscore | none
    mean: list[float] | None = None      # custom normalization stats (override imagenet)
    std: list[float] | None = None
    bands: list[int] | None = None       # band selection for multispectral datasets (0-based)
    balanced_sampling: bool = False      # inverse-frequency weighted sampling
    kwargs: dict[str, Any] = field(default_factory=dict)  # dataset-specific passthrough


@dataclass
class TrainingConfig:
    epochs: int = 10
    optimizer: str = "adamw"             # adamw | adam | sgd
    lr: float = 1e-3
    weight_decay: float = 0.05
    momentum: float = 0.9                # sgd only
    scheduler: str = "cosine"            # cosine | step | none
    warmup_epochs: int = 0
    step_size: int = 10                  # step scheduler only
    gamma: float = 0.1                   # step scheduler only
    label_smoothing: float = 0.0
    amp: bool = False                    # mixed precision (CUDA only)
    grad_clip: float | None = None
    device: str = "auto"                 # auto | cuda | mps | cpu | cuda:N
    log_interval: int = 50
    monitor: str = "val/accuracy"        # metric for checkpointing / early stopping
    monitor_mode: str = "max"            # max | min
    early_stopping_patience: int | None = None
    resume: str | None = None            # checkpoint path to resume from


@dataclass
class ExperimentConfig:
    task: str = "classification"
    experiment_name: str = "experiment"
    seed: int = 42
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# Dict -> dataclass construction with validation and scalar coercion
# ---------------------------------------------------------------------------

_NoneType = type(None)


def _unwrap_optional(hint: Any) -> Any:
    """Return the non-None member of ``X | None`` hints, else the hint itself."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not _NoneType]
        if len(args) == 1:
            return args[0]
    return hint


def _coerce(value: Any, hint: Any, path: str) -> Any:
    """Coerce YAML scalars to the annotated type where unambiguous."""
    hint = _unwrap_optional(hint)
    if value is None:
        return None
    if is_dataclass(hint):
        return _build_dataclass(hint, value, path)
    if hint is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value!r} to float for {path!r}") from None
    if hint is int and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value!r} to int for {path!r}") from None
    return value


def _build_dataclass(cls: type, data: Any, path: str = "config") -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {path!r}, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {path!r}. "
            f"Valid keys: {sorted(field_names)}"
        )

    kwargs = {
        name: _coerce(value, hints[name], f"{path}.{name}")
        for name, value in data.items()
    }
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# YAML I/O and CLI overrides
# ---------------------------------------------------------------------------


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted ``key.path=value`` overrides; values are YAML-parsed."""
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid override {override!r}; expected dotted 'key.path=value'"
            )
        value = yaml.safe_load(raw) if raw else None
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                node[part] = nxt = {}
            node = nxt
        node[parts[-1]] = value
    return data


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    return _build_dataclass(ExperimentConfig, data)


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file with optional
    dotted-key overrides (e.g. ``["training.lr=0.01"]``)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = apply_overrides(data, list(overrides))
    return config_from_dict(data)


def to_dict(config: Any) -> dict[str, Any]:
    return dataclasses.asdict(config)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(config), f, default_flow_style=False, sort_keys=False)
