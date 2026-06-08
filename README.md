# pytorch-image-analytics

Modular PyTorch platform for image analytics: classification, object
detection, segmentation, multispectral/satellite imagery, and 3D — built
phase by phase per [EXPLORATION.md](EXPLORATION.md) and
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Status: Phase 2 (Detection)** — Phase 1 (core infra, data pipeline,
timm backbones, classification) plus four from-scratch detectors sharing one
interface: RetinaNet, FCOS, Faster R-CNN, and DETR, with YOLO available via
an ultralytics wrapper.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev,geo,notebooks]"   # geo = rasterio for multispectral
```

## Train

```bash
# Fine-tune ResNet-18 on CIFAR-10 (downloads the dataset on first run)
python scripts/train.py --config configs/classification/cifar10_resnet18.yaml

# Detection on the offline synthetic-shapes dataset (CPU, minutes)
python scripts/train.py --config configs/detection/faster_rcnn_shapes.yaml
python scripts/train.py --config configs/detection/retinanet_shapes.yaml
python scripts/train.py --config configs/detection/fcos_shapes.yaml

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
  `DetectionTrainer` overrides only the step hooks (loss-dict training,
  prediction-list evaluation); the loop is shared.
- **One detection interface** — every detector (anchor-based, anchor-free,
  two-stage, transformer) trains as `model(images, targets) -> loss dict` and
  predicts as `model(images) -> [{boxes, scores, labels}]`; the COCO-protocol
  mAP evaluator is parity-tested against pycocotools.
- **Multi-channel first** — backbones accept `in_channels != 3` (pretrained
  stem weights adapted), optional per-band channel attention, and the
  multispectral dataset handles 16-bit GeoTIFFs with percentile/min-max/
  z-score normalization plus spectral indices (NDVI, NDWI, NDBI, EVI).

## Tests

```bash
.venv/bin/python -m pytest
```

## Roadmap

Phase 3 (segmentation: U-Net, DeepLabv3+, Mask R-CNN, SegFormer, SAM),
Phase 4 (satellite/foundation models), Phase 5 (3D), Phase 6 (serving/MLOps)
— design specs and acceptance criteria in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
