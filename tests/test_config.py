import pytest
import yaml

from image_analytics.core.config import (
    ExperimentConfig,
    config_from_dict,
    load_config,
    save_config,
)

MINIMAL_YAML = """
task: classification
experiment_name: test_exp
model:
  name: classifier
  num_classes: 7
  backbone:
    name: resnet18
    pretrained: false
data:
  dataset: fake
  batch_size: 8
training:
  epochs: 3
  lr: 1e-4
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML)
    return path


def test_load_config(config_path):
    config = load_config(config_path)
    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "test_exp"
    assert config.model.num_classes == 7
    assert config.model.backbone.name == "resnet18"
    assert config.model.backbone.pretrained is False
    assert config.data.batch_size == 8
    assert config.training.epochs == 3


def test_scientific_notation_coerced_to_float(config_path):
    # YAML 1.1 parses bare "1e-4" as a string; the loader must coerce it.
    config = load_config(config_path)
    assert isinstance(config.training.lr, float)
    assert config.training.lr == pytest.approx(1e-4)


def test_defaults_fill_missing_sections():
    config = config_from_dict({"task": "classification"})
    assert config.model.backbone.name == "resnet50"
    assert config.training.optimizer == "adamw"
    assert config.data.normalize == "imagenet"


def test_unknown_key_raises(config_path):
    data = yaml.safe_load(config_path.read_text())
    data["training"]["leraning_rate"] = 0.1  # typo
    with pytest.raises(ValueError, match="leraning_rate"):
        config_from_dict(data)


def test_overrides(config_path):
    config = load_config(
        config_path,
        overrides=["training.lr=0.01", "data.batch_size=64", "model.backbone.name=resnet50"],
    )
    assert config.training.lr == pytest.approx(0.01)
    assert config.data.batch_size == 64
    assert config.model.backbone.name == "resnet50"


def test_invalid_override_format(config_path):
    with pytest.raises(ValueError, match="key.path=value"):
        load_config(config_path, overrides=["training.lr"])


def test_save_roundtrip(config_path, tmp_path):
    config = load_config(config_path)
    out = tmp_path / "saved" / "config.yaml"
    save_config(config, out)
    reloaded = load_config(out)
    assert reloaded == config


def test_optional_fields():
    config = config_from_dict(
        {"training": {"grad_clip": 1.0, "early_stopping_patience": 5}}
    )
    assert config.training.grad_clip == pytest.approx(1.0)
    assert config.training.early_stopping_patience == 5
    assert config.training.resume is None
