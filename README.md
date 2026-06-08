# pytorch-image-analytics

Modular PyTorch platform for image analytics: classification, object
detection, segmentation, multispectral/satellite imagery, and 3D — built
phase by phase per [EXPLORATION.md](EXPLORATION.md) and
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Status: all phases (1–6) implemented.** Phase 1 (core infra, data pipeline,
timm backbones, classification); Phase 2 detection (RetinaNet, FCOS, Faster
R-CNN, Cascade R-CNN, DETR; PAFPN/letterbox; YOLO + Deformable-DETR / RT-DETR
wrappers); Phase 3 segmentation (from-scratch **U-Net** / **DeepLabv3+**,
**Mask R-CNN**, plus SegFormer / Mask2Former / OneFormer / SAM / smp wrappers);
Phase 4 satellite (band-group stems, **SatMAE / Prithvi** backbones, TorchGeo,
**Siamese change detection**, temporal pooling); Phase 5 3D (**PointNet /
PointNet++ / DGCNN** and **PointPillars** on pure-PyTorch point ops, CUDA
methods gated); Phase 6 serving (ONNX export + parity gate, unified inference
CLI, TorchServe/Triton, embeddings, drift monitoring, CI). MLflow tracking is
wired into every task.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev,geo,notebooks]"   # geo = rasterio for multispectral
uv pip install -e ".[seg,serve]"           # seg = smp + transformers; serve = mlflow
```

## Train

```bash
# Fine-tune ResNet-18 on CIFAR-10 (downloads the dataset on first run)
python scripts/train.py --config configs/classification/cifar10_resnet18.yaml

# Detection on the offline synthetic-shapes dataset (CPU, minutes)
python scripts/train.py --config configs/detection/faster_rcnn_shapes.yaml
python scripts/train.py --config configs/detection/retinanet_shapes.yaml
python scripts/train.py --config configs/detection/fcos_shapes.yaml

# Segmentation on synthetic shapes-with-masks (CPU, minutes)
python scripts/train.py --config configs/segmentation/unet_shapes.yaml
python scripts/train.py --config configs/segmentation/deeplabv3plus_shapes.yaml
python scripts/train.py --config configs/segmentation/mask_rcnn_shapes.yaml   # instance

# 3D point clouds on synthetic primitives (CPU, minutes)
python scripts/train.py --config configs/pointcloud/pointnet_primitives.yaml
python scripts/train.py --config configs/pointcloud/pointpillars_synthetic.yaml   # 3D detection

# Bi-temporal change detection on synthetic shapes (CPU, minutes)
python scripts/train.py --config configs/segmentation/change_detection_shapes.yaml

# Track any run with MLflow (uses MLFLOW_TRACKING_URI)
python scripts/train.py --config configs/segmentation/unet_shapes.yaml training.mlflow=true

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
- **Segmentation shares the core** — semantic models output
  `logits (B, C, H, W)` and train through the *base* Trainer with a pixel-wise
  criterion (CE / Dice / CE+Dice) and a streaming mIoU/Dice evaluator;
  instance (Mask R-CNN) reuses the detection Trainer and a mask-mAP evaluator
  (pycocotools `segm` parity-tested). One synthetic shapes rasterizer feeds the
  detection, semantic, and instance fixtures so they stay pixel-consistent.
- **Multi-channel first** — backbones accept `in_channels != 3` (pretrained
  stem weights adapted), optional per-band channel attention, and the
  multispectral dataset handles 16-bit GeoTIFFs with percentile/min-max/
  z-score normalization plus spectral indices (NDVI, NDWI, NDBI, EVI).

## Serve & deploy

```bash
# Export to ONNX (mandatory onnxruntime parity gate before the file is written)
python scripts/export.py --config outputs/<exp>/config.yaml \
    --checkpoint outputs/<exp>/checkpoints/best.pt --output model.onnx

# Run inference over file/dir/glob -> JSON (optionally annotated images)
python scripts/infer.py --config outputs/<exp>/config.yaml \
    --checkpoint outputs/<exp>/checkpoints/best.pt --input 'images/*.jpg' --visualize

# Extract backbone embeddings (+ optional FAISS index)
python scripts/embed.py --config outputs/<exp>/config.yaml \
    --checkpoint outputs/<exp>/checkpoints/best.pt --output embeddings.parquet
```

`serving/` also generates Triton `config.pbtxt`, provides a config-driven
TorchServe handler, and a pure-numpy `DriftMonitor` (PSI/KS/cosine-shift).

## Tests

```bash
.venv/bin/python -m pytest            # add -m "not slow" to skip overfit/training tests
```

## Roadmap

All six phases are implemented (design specs and acceptance criteria in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)). What remains is hardware-bound:
real-data GPU runs (CIFAR-10, COCO), verifying the CUDA-gated 3D wrappers
(SECOND/CenterPoint/BEVFormer/Mask3D) on a GPU box, and the backlog items in
[TODO.md](TODO.md).
