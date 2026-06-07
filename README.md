# pytorch-image-analytics

Modular PyTorch platform for image analytics: classification, object
detection, segmentation, multispectral/satellite imagery, and 3D — built
phase by phase per [EXPLORATION.md](EXPLORATION.md).

**Status: Phase 1 (Foundation)** — core infrastructure, data pipeline,
timm-backed backbones with multi-channel support, and end-to-end
classification training.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev,geo,notebooks]"   # geo = rasterio for multispectral
```

## Train

```bash
# Fine-tune ResNet-18 on CIFAR-10 (downloads the dataset on first run)
python scripts/train.py --config configs/classification/cifar10_resnet18.yaml

# Override any config key from the CLI
python scripts/train.py --config configs/classification/cifar10_resnet18.yaml \
    training.lr=1e-4 data.batch_size=64 training.epochs=5

# Offline smoke test (synthetic data, CPU, ~30s)
python scripts/train.py --config configs/classification/smoke_fake.yaml

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py --config <path>
```

Checkpoints (`last.pt`, `best.pt`) and the resolved config land in
`outputs/<experiment_name>/`.

## Evaluate

```bash
python scripts/evaluate.py \
    --config outputs/cifar10_resnet18/config.yaml \
    --checkpoint outputs/cifar10_resnet18/checkpoints/best.pt
```

## Design

- **Registry pattern** — backbones, models, and datasets register via
  decorator (`core/registry.py`); new architectures plug in without touching
  existing code. Any of timm's 900+ model names also works as a backbone
  without registration.
- **Config-driven** — every experiment is a YAML file parsed into typed
  dataclasses (`core/config.py`); unknown keys fail fast, dotted CLI
  overrides (`training.lr=1e-4`) are YAML-parsed.
- **Task-agnostic Trainer** — DDP-aware loop with AMP, gradient clipping,
  checkpointing, and callbacks (logging / best-checkpoint / early stopping).
  Detection and segmentation tasks will override `training_step`.
- **Multi-channel first** — backbones accept `in_channels != 3` (pretrained
  stem weights adapted), optional per-band channel attention, and the
  multispectral dataset handles 16-bit GeoTIFFs with percentile/min-max/
  z-score normalization plus spectral indices (NDVI, NDWI, NDBI, EVI).

## Tests

```bash
.venv/bin/python -m pytest
```

## Roadmap

Phase 2 (detection: FPN, Faster R-CNN, RetinaNet, FCOS, DETR), Phase 3
(segmentation), Phase 4 (satellite/foundation models), Phase 5 (3D),
Phase 6 (serving/MLOps) — see [EXPLORATION.md](EXPLORATION.md#9-phased-build-roadmap).
