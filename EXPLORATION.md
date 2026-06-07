# Image Analytics Platform — Exploration & Architecture Reference

> Research compiled: 2026-06-02
> Status: Exploration phase — implementation pending

---

## Table of Contents

1. [Repository Architecture](#repository-architecture)
2. [Classification](#1-image-classification)
3. [Object Detection](#2-object-detection)
4. [Segmentation](#3-segmentation)
5. [Multi-Channel / 16-bit / Satellite Imagery](#4-multi-channel--16-bit--satellite-imagery)
6. [3D Object Detection & Segmentation](#5-3d-object-detection--segmentation)
7. [Enterprise Architecture Patterns](#6-enterprise-architecture-patterns)
8. [Ecosystem Reference Matrix](#7-ecosystem-reference-matrix)
9. [Architecture Selection Guide](#8-architecture-selection-guide)
10. [Phased Build Roadmap](#9-phased-build-roadmap)
11. [Core Dependencies](#10-core-dependencies)

---

## Repository Architecture

```
pytorch-image-analytics-all/
├── pyproject.toml
├── configs/
│   ├── classification/
│   ├── detection/
│   ├── segmentation/
│   └── 3d/
│
├── src/
│   └── image_analytics/
│       ├── core/                     # shared infrastructure
│       │   ├── config.py             # config parsing (hydra or dataclasses)
│       │   ├── registry.py           # model/backbone/head registry pattern
│       │   ├── trainer.py            # unified training loop (DDP/FSDP aware)
│       │   ├── evaluator.py          # metric computation
│       │   └── callbacks.py          # logging, checkpointing, early stopping
│       │
│       ├── data/
│       │   ├── datasets/
│       │   │   ├── standard.py       # 8-bit RGB datasets (COCO, VOC, ImageNet)
│       │   │   ├── multispectral.py  # >3 channel, 16-bit (rasterio-based)
│       │   │   ├── pointcloud.py     # 3D point cloud datasets
│       │   │   └── registry.py
│       │   ├── transforms/
│       │   │   ├── augmentations.py  # albumentations / torchvision v2
│       │   │   ├── spectral.py       # band normalization, NDVI, indices
│       │   │   └── pointcloud.py     # 3D augmentations
│       │   └── samplers.py           # geo-aware, balanced, etc.
│       │
│       ├── backbones/                # feature extractors (wrappers around timm)
│       │   ├── resnet.py
│       │   ├── efficientnet.py
│       │   ├── swin.py
│       │   ├── convnext.py
│       │   ├── vit.py                # ViT, DeiT, DINOv2/v3
│       │   ├── multichannel.py       # modified stems for >3 channels
│       │   └── registry.py
│       │
│       ├── classification/
│       │   ├── models.py             # classifier heads
│       │   ├── multilabel.py         # BCEWithLogitsLoss-based
│       │   └── train.py
│       │
│       ├── detection/
│       │   ├── anchors/
│       │   │   ├── rpn.py            # Region Proposal Network
│       │   │   └── anchor_free.py    # FCOS-style, CenterNet-style
│       │   ├── necks/
│       │   │   ├── fpn.py            # Feature Pyramid Network
│       │   │   └── pafpn.py          # Path Aggregation FPN
│       │   ├── heads/
│       │   │   ├── faster_rcnn.py
│       │   │   ├── cascade_rcnn.py
│       │   │   ├── retinanet.py
│       │   │   └── detr.py           # DETR / Deformable DETR
│       │   ├── losses.py             # focal loss, GIoU, etc.
│       │   └── train.py
│       │
│       ├── segmentation/
│       │   ├── semantic/
│       │   │   ├── unet.py           # U-Net, U-Net++, Attention U-Net
│       │   │   ├── deeplab.py        # DeepLabv3, DeepLabv3+
│       │   │   └── segformer.py
│       │   ├── instance/
│       │   │   ├── mask_rcnn.py
│       │   │   └── mask2former.py
│       │   ├── panoptic/
│       │   │   └── oneformer.py
│       │   └── train.py
│       │
│       ├── detection_3d/
│       │   ├── pointnet.py           # PointNet, PointNet++
│       │   ├── voxel.py              # VoxelNet, SECOND, CenterPoint
│       │   ├── bev.py                # BEVFormer, BEVDet
│       │   ├── segmentation_3d.py    # Mask3D, PointGroup
│       │   └── train.py
│       │
│       ├── foundation/               # foundation model wrappers
│       │   ├── dinov2.py             # DINOv2/v3 feature extraction
│       │   ├── clip.py               # CLIP embeddings
│       │   ├── sam.py                # SAM/SAM2 promptable segmentation
│       │   └── satmae.py             # satellite foundation models
│       │
│       └── serving/                  # inference & deployment
│           ├── torchserve_handler.py
│           ├── onnx_export.py
│           └── triton_config.py
│
├── notebooks/
│   ├── 01_classification_quickstart.ipynb
│   ├── 02_detection_faster_rcnn.ipynb
│   ├── 03_segmentation_unet.ipynb
│   ├── 04_satellite_multispectral.ipynb
│   └── 05_3d_pointcloud.ipynb
│
├── scripts/
│   ├── train.py                      # unified: python scripts/train.py --config configs/...
│   ├── evaluate.py
│   ├── infer.py
│   └── export.py
│
└── tests/
```

---

## Key Design Patterns

### Registry Pattern

All models, backbones, datasets, and losses register via decorator so new architectures plug in without modifying existing code:

```python
# core/registry.py
BACKBONE_REGISTRY = {}

def register_backbone(name):
    def decorator(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator

# backbones/resnet.py
@register_backbone("resnet50")
class ResNet50Backbone(nn.Module): ...
```

### Config-Driven Training

Every experiment is reproducible via YAML config (Hydra or dataclasses):

```yaml
# configs/detection/faster_rcnn_fpn_satellite.yaml
backbone:
  name: resnet50
  in_channels: 13
  pretrained: true
neck:
  name: fpn
  out_channels: 256
head:
  name: faster_rcnn
  num_classes: 15
data:
  dataset: multispectral
  bands: [0, 1, 2, 3, 7, 9, 10]
  normalize: percentile
training:
  epochs: 50
  optimizer: adamw
  lr: 1e-4
  distributed: fsdp
```

---

## 1. Image Classification

### 1.1 Classic CNNs

| Architecture | Key Contribution | Params (typical) | PyTorch Source |
|---|---|---|---|
| **ResNet** (He 2015) | Residual skip connections enabling 100+ layer training | 25M (ResNet-50) | `torchvision`, `timm` |
| **VGG** (Simonyan 2014) | Uniform 3x3 convolutions; depth matters | 138M (VGG-16) | `torchvision`, `timm` |
| **Inception / GoogLeNet** (Szegedy 2014) | Multi-scale parallel convolutions (Inception modules) | 23M (InceptionV3) | `torchvision`, `timm` |
| **DenseNet** (Huang 2017) | Dense connectivity — each layer receives all preceding feature maps | 8M (DenseNet-121) | `torchvision`, `timm` |
| **EfficientNet / V2** (Tan & Le 2019/2021) | Compound scaling (depth, width, resolution); Fused-MBConv in V2 | 5.3M (B0) to 66M (B7) | `torchvision`, `timm` |
| **MobileNetV2/V3/V4** (Sandler 2018+) | Inverted residuals, squeeze-excite, hardware-aware NAS | 3.4M (V3-Small) | `torchvision`, `timm` |
| **ShuffleNetV2** (Ma 2018) | Channel shuffle for efficient group convolutions | 2.3M | `torchvision`, `timm` |
| **ConvNeXt / V2** (Liu 2022/2023) | CNN modernized with Transformer design principles; 87.8% ImageNet top-1 | 28M-350M | `torchvision`, `timm` |

**Production guidance:** EfficientNetV2 and ConvNeXt remain strong production choices for accuracy-latency-cost balance. MobileNetV3/V4 is the go-to for mobile/edge due to superior quantization properties.

### 1.2 Vision Transformers

| Architecture | Key Innovation | Notes |
|---|---|---|
| **ViT** (Dosovitskiy 2020) | Patch tokenization + standard Transformer encoder | Requires large-scale pretraining; `timm` `vit_base_patch16_224` etc. |
| **DeiT** (Touvron 2021) | Data-efficient training with distillation token | Makes ViT viable on ImageNet-1k alone |
| **Swin Transformer** (Liu 2021) | Shifted window attention for hierarchical features | Linear complexity; excellent backbone for detection/segmentation |
| **BEiT / BEiT-3** (Bao 2021) | BERT-style pretraining with visual tokenizer | Strong for fine-tuning; `timm`, HuggingFace |
| **CaiT** (Touvron 2021) | Class-Attention decoupled from patch attention | Improved training stability at depth |

### 1.3 Hybrid CNN-Transformer

| Architecture | Design | Availability |
|---|---|---|
| **CoAtNet** (Dai 2021) | MBConv blocks (local) then relative-attention Transformer blocks (global) | `timm` — `coatnet_0_rw_224` through `coatnet_rmlp_2_rw_384` |
| **MaxViT** (Tu 2022) | MBConv + blocked local attention + dilated global attention per stage | `timm` — `maxvit_tiny_tf_224` etc. |
| **CoAtNeXt** | ConvNeXt blocks replacing MBConv in CoAtNet; all LayerNorm | `timm` |

### 1.4 Self-Supervised / Foundation Models

| Model | Paradigm | Key Details |
|---|---|---|
| **DINO** (Caron 2021) | Self-distillation with no labels (ViT student-teacher) | Emergent object segmentation in attention maps |
| **DINOv2** (Oquab 2023) | Scaled self-supervised ViT; 1B params distilled down | 142M curated images; linear probing within ~2% of fine-tuning |
| **DINOv3** (Meta 2025) | Latest DINO series | 1.7B curated images, 1B parameter model; available in `timm` |
| **MAE** (He 2022) | Masked Autoencoder: reconstruct 75% masked patches | Excellent pre-training for ViT; efficient training |
| **CLIP** (Radford 2021) | Contrastive language-image pretraining on 400M pairs | Zero-shot transfer; `openai/clip`, `open_clip` |
| **SigLIP / SigLIP 2** (Google 2023-2025) | Sigmoid loss replacing softmax in CLIP-style training | Weights in `timm` |

### 1.5 Transfer Learning & Fine-Tuning Best Practices

```python
import timm
import torch.nn as nn

# timm: central hub for classification backbones (900+ architectures)
model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=10)

# Fine-tuning strategies:
# 1. Linear probing: freeze backbone, train only classifier head
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

# 2. Gradual unfreezing: unfreeze last N layers progressively
# 3. Discriminative learning rates: lower LR for early layers, higher for head
# 4. Layer-wise LR decay (timm supports via param_groups_layer_decay)

# Multi-label classification:
model.head = nn.Linear(model.head.in_features, num_labels)
criterion = nn.BCEWithLogitsLoss()  # per-label binary cross-entropy
```

**Key `timm` features:**
- `timm.create_model()` — unified API for 900+ architectures
- `timm.data.create_transform()` — model-specific preprocessing
- `timm.optim` — LAMB, LARS, AdaFactor, Lookahead
- `timm.scheduler` — cosine with warmup, step, plateau
- `timm.data.Mixup` — Mixup, CutMix, label smoothing
- Feature extraction: `model.forward_features(x)` returns spatial features before classifier

---

## 2. Object Detection

### 2.1 Two-Stage Detectors

| Architecture | Mechanism | Framework |
|---|---|---|
| **Faster R-CNN** (Ren 2015) | RPN generates proposals -> RoI pooling -> per-region classification + regression | `torchvision`, Detectron2, MMDetection |
| **Cascade R-CNN** (Cai 2018) | Multi-stage with progressively higher IoU thresholds (0.5 -> 0.6 -> 0.7) | Detectron2, MMDetection |
| **Feature Pyramid Network (FPN)** (Lin 2017) | Top-down pathway with lateral connections for multi-scale feature fusion | Integral component in most modern detectors |
| **RPN** | Sliding window with multi-scale anchors for region proposals | Built into Faster R-CNN and derivatives |

### 2.2 One-Stage Detectors

| Architecture | Key Design | Status (2026) |
|---|---|---|
| **YOLOv5** (Ultralytics 2020) | CSP-Darknet backbone, PANet neck, anchor-based | Mature, widely deployed; `ultralytics` |
| **YOLOv8** (Ultralytics 2023) | Anchor-free decoupled head, C2f modules | Production standard |
| **YOLO11** (Ultralytics 2024) | Attention in backbone/neck; 22% fewer params than v8m at higher mAP | `ultralytics` |
| **YOLOv12** (Feb 2025) | Attention-centric architecture with global context at real-time speed | `ultralytics` |
| **YOLO26** (Jan 2026) | Eliminates NMS; end-to-end; 43% faster CPU; multi-task (det, seg, pose, OBB) | `ultralytics` |
| **SSD** (Liu 2016) | Multi-scale feature maps with default boxes | `torchvision` |
| **RetinaNet** (Lin 2017) | Focal Loss addressing class imbalance; FPN backbone | `torchvision`, Detectron2, MMDetection |
| **FCOS** (Tian 2019) | Fully convolutional anchor-free; per-pixel (l,t,r,b) regression + centerness | `torchvision`, MMDetection |
| **CenterNet** (Zhou 2019) | Objects as points; keypoint center detection + size regression | MMDetection, Detectron2 |

### 2.3 Transformer-Based Detectors

| Architecture | Innovation | Notes |
|---|---|---|
| **DETR** (Carion 2020) | End-to-end set prediction via Transformer; eliminates NMS/anchors; bipartite matching | Slow convergence (500 epochs); Detectron2, HuggingFace |
| **Deformable DETR** (Zhu 2020) | Deformable attention — sparse key sampling; 10x faster convergence | MMDetection, HuggingFace |
| **DINO (detection)** (Zhang 2022) | Denoising anchor boxes + contrastive denoising + mixed query selection | SOTA COCO; MMDetection |
| **RT-DETR** (Zhao 2023) | Real-time DETR; 53.0% AP at 114 FPS on T4 | `ultralytics`, PaddleDetection |
| **Co-DETR** (Zong 2023) | Collaborative hybrid assignments training with auxiliary heads | MMDetection |
| **RF-DETR** (Roboflow Mar 2025) | DINOv2 backbone; 54.7% mAP <5ms; anchor-free, NMS-free; ICLR 2026 | `rf-detr` pip package |

### 2.4 Anchor-Free vs Anchor-Based

| Aspect | Anchor-Based | Anchor-Free |
|---|---|---|
| Examples | Faster R-CNN, SSD, RetinaNet, YOLOv5 | FCOS, CenterNet, YOLOv8+, DETR family |
| Proposals | Predefined anchor boxes at multiple scales/ratios | Per-pixel or per-point regression; no priors |
| Trend (2025-2026) | Declining in new architectures | Dominant paradigm |

### 2.5 Detection Ecosystem

```python
# torchvision (built-in)
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn

# Detectron2 (Meta) — Faster/Mask/Cascade R-CNN, RetinaNet, Panoptic FPN
from detectron2 import model_zoo
cfg = model_zoo.get_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# MMDetection (OpenMMLab) — broadest model zoo: 300+ configs
# Supports: Faster R-CNN, Cascade R-CNN, DETR, Deformable DETR, DINO, Co-DETR, FCOS, etc.

# Ultralytics — YOLO family + RT-DETR
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # or yolov8, yolo26, rtdetr-l
```

---

## 3. Segmentation

### 3.1 Instance Segmentation

| Architecture | Approach | Availability |
|---|---|---|
| **Mask R-CNN** (He 2017) | Faster R-CNN + parallel mask branch per RoI | `torchvision`, Detectron2, MMDetection |
| **Cascade Mask R-CNN** | Cascade R-CNN with mask heads at each stage | Detectron2, MMDetection |
| **YOLO11-seg / YOLO26-seg** | Real-time instance segmentation in YOLO | `ultralytics` |

### 3.2 Semantic Segmentation

| Architecture | Design | Source |
|---|---|---|
| **U-Net** (Ronneberger 2015) | Encoder-decoder with skip connections | `segmentation_models_pytorch` (smp) |
| **U-Net++** (Zhou 2018) | Nested dense skip connections | `smp` |
| **Attention U-Net** (Oktay 2018) | Attention gates on skip connections | `smp`, custom |
| **DeepLabv3** (Chen 2017) | Atrous Spatial Pyramid Pooling (ASPP); multi-scale context | `torchvision`, `smp` |
| **DeepLabv3+** (Chen 2018) | DeepLabv3 encoder + lightweight decoder with low-level fusion | `smp`, MMSegmentation |
| **SegFormer** (Xie 2021) | Hierarchical Transformer encoder + All-MLP decoder; no positional encoding | HuggingFace, MMSegmentation |
| **Segmenter** (Strudel 2021) | Pure ViT encoder + mask transformer decoder | MMSegmentation |

### 3.3 Panoptic Segmentation

| Architecture | Approach | Performance |
|---|---|---|
| **Panoptic FPN** (Kirillov 2019) | FPN + semantic branch alongside instance branch | Detectron2 |
| **Mask2Former** (Cheng 2022) | Masked attention + universal arch; 57.8 PQ COCO | Detectron2, HuggingFace, MMDetection |
| **OneFormer** (Jain 2023) | Task-conditioned queries; single model outperforms task-specific Mask2Former | HuggingFace |

### 3.4 Foundation / Promptable Segmentation

| Model | Capabilities | Notes |
|---|---|---|
| **SAM** (Kirillov 2023) | Segment anything with point/box/text prompts; zero-shot | `segment-anything` (Meta); ViT-H backbone |
| **SAM 2** (Ravi 2024) | Extends to video; memory attention across frames; near real-time | `sam2` (Meta); HuggingFace |
| **Grounded SAM** | SAM + Grounded DINO for open-vocabulary detection + segmentation | Community |

### 3.5 Segmentation Library Overview

```python
# segmentation_models_pytorch (smp) — most popular dedicated library
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet50",        # 400+ encoders from timm
    encoder_weights="imagenet",
    in_channels=3,                  # can change for multi-channel
    classes=10,
)
# Architectures: Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN,
#                DeepLabV3, DeepLabV3Plus

# torchvision
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet50

# MMSegmentation — broadest research coverage
# HuggingFace Transformers — SegFormer, Mask2Former, OneFormer, SAM2
```

---

## 4. Multi-Channel / 16-bit / Satellite Imagery

### 4.1 Handling >3 Channel Inputs

**Strategy 1: Modify first convolutional layer**

```python
import timm
import torch
import torch.nn as nn

model = timm.create_model('resnet50', pretrained=True)
old_conv = model.conv1
new_conv = nn.Conv2d(
    in_channels=13,              # e.g., Sentinel-2 bands
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)
with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    nn.init.kaiming_normal_(new_conv.weight[:, 3:])
model.conv1 = new_conv

# timm shortcut: many models support in_chans parameter
model = timm.create_model('resnet50', pretrained=True, in_chans=13)
```

**Strategy 2: Channel-wise attention**

```python
class ChannelAttentionInput(nn.Module):
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c))
        return x * w.view(b, c, 1, 1)
```

**Strategy 3: Band grouping with separate stems**
- Process spectral band groups through separate conv stems, then fuse
- Used in SatMAE — groups of bands with distinct spectral positional encodings

### 4.2 16-bit Image Handling

```python
import rasterio
import numpy as np
import torch

# Loading with rasterio (standard for geospatial)
with rasterio.open('image.tif') as src:
    data = src.read()  # shape: (C, H, W), dtype: uint16 or float32
    profile = src.profile

tensor = torch.from_numpy(data.astype(np.float32))

# Normalization strategies:

# 1. Per-channel min-max to [0, 1]
for c in range(tensor.shape[0]):
    cmin, cmax = tensor[c].min(), tensor[c].max()
    tensor[c] = (tensor[c] - cmin) / (cmax - cmin + 1e-8)

# 2. Per-channel z-score (using dataset-level statistics)
# tensor[c] = (tensor[c] - means[c]) / stds[c]

# 3. Percentile clipping (robust to outliers — common for satellite)
for c in range(tensor.shape[0]):
    p2, p98 = np.percentile(data[c], [2, 98])
    tensor[c] = torch.clamp(tensor[c], p2, p98)
    tensor[c] = (tensor[c] - p2) / (p98 - p2 + 1e-8)

# 4. Physical reflectance scaling (if calibration metadata available)
# reflectance = DN * gain + offset
```

### 4.3 Spectral Indices

```python
# Common indices computed from multi-band imagery:
# NDVI = (NIR - Red) / (NIR + Red)        — vegetation
# NDWI = (Green - NIR) / (Green + NIR)    — water
# NDBI = (SWIR - NIR) / (SWIR + NIR)      — built-up areas
# EVI  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)  — enhanced vegetation
```

### 4.4 Multi-Temporal Analysis

- Stack temporal observations as additional channels or as a sequence dimension
- Temporal attention (used by SatMAE) or 3D convolutions over time
- Change detection: Siamese networks comparing bi-temporal pairs

### 4.5 Libraries for Geospatial Deep Learning

| Library | Purpose | Key Features |
|---|---|---|
| **TorchGeo** | PyTorch domain library for geospatial data | CRS-aware datasets, spatial samplers, multispectral transforms, pretrained RS backbones; official PyTorch ecosystem library |
| **rasterio** | Read/write geotiff, COG, raster formats | Built on GDAL; windowed reading for large images |
| **GDAL / osgeo** | Comprehensive geospatial data abstraction | Reprojection, warping, virtual rasters (VRT) |
| **rio-tiler** | Dynamic tile serving from COGs | Cloud-optimized GeoTIFF reading |

### 4.6 Pre-trained Models for Remote Sensing

| Model | Architecture | Training Data | Capabilities |
|---|---|---|---|
| **SatMAE** (ICML 2023) | Masked Autoencoder on ViT | 1M+ satellite images | Grouped band encoding with spectral positional encodings; SOTA fMoW, EuroSAT |
| **SatMAE++** (CVPR 2024) | Improved SatMAE | Rethought pre-training | Scale-aware, multi-scale positional encodings |
| **SpectralGPT** (IEEE TPAMI 2024) | 3D generative pretrained transformer | 1M Sentinel-2; 600M+ params | 3D token generation for spatial-spectral coupling; handles varying sizes, resolutions, time series |
| **SSL4EO-S12** | Various SSL (MoCo, DINO, MAE, data2vec) | Sentinel-1/2 | Available via TorchGeo |
| **Prithvi** (IBM/NASA 2023) | Temporal ViT with MAE pretraining | HLS Sentinel-2/Landsat | Multi-temporal, multi-spectral; HuggingFace |

---

## 5. 3D Object Detection & Segmentation

### 5.1 Point Cloud Processing

| Architecture | Approach | Details |
|---|---|---|
| **PointNet** (Qi 2017) | Direct point set processing via shared MLPs + max-pool | Pioneering; permutation invariant; lacks local structure |
| **PointNet++** (Qi 2017) | Hierarchical set abstraction with ball query grouping | Captures local geometry at multiple scales |
| **Point Transformer v1/v2/v3** (Zhao 2021+) | Self-attention on point clouds | v3 is SOTA; serialization-based attention |
| **DGCNN** (Wang 2019) | Dynamic graph convolution in feature space | EdgeConv operator; captures point relationships |
| **PointNeXt** (Qian 2022) | Modernized PointNet++ with improved training recipes | Training strategy matters as much as architecture |

### 5.2 Voxel-Based Methods

| Architecture | Key Innovation | Use Case |
|---|---|---|
| **VoxelNet** (Zhou 2018) | End-to-end: voxelization -> VFE layers -> 3D conv -> RPN | Autonomous driving (KITTI) |
| **SECOND** (Yan 2018) | Sparse 3D convolutions to accelerate VoxelNet | 3-4x faster than VoxelNet |
| **PointPillars** (Lang 2019) | Pseudo-images from pillar features; 2D conv only | Very fast; good for real-time |
| **CenterPoint** (Yin 2021) | Anchor-free center-based; keypoint on BEV + refinement | SOTA nuScenes/Waymo; supports tracking |

### 5.3 BEV (Bird's Eye View) Methods

| Architecture | Design | Key Advantage |
|---|---|---|
| **BEVFormer** (Li 2022) | Spatiotemporal transformer: BEV queries, spatial cross-attention, temporal self-attention | Camera-only; no LiDAR needed |
| **BEVDet** (Huang 2022) | Explicit view transformation to BEV; BEV augmentation | Simpler; extensible (BEVDet4D adds temporal) |
| **BEVFusion** (Liu 2023) | Unified BEV space for LiDAR + camera fusion | Multi-modal; strong on nuScenes |

### 5.4 3D Instance Segmentation

| Architecture | Method | Notes |
|---|---|---|
| **3D-BoNet** (Yang 2019) | Direct bbox regression + per-point mask; single-stage, anchor-free | No NMS/clustering needed |
| **PointGroup** (Jiang 2020) | Dual-set point grouping: original + shifted coordinates; bottom-up clustering | Strong on ScanNet |
| **Mask3D** (Schult 2023) | Transformer: object queries from voxel features; end-to-end | SOTA ScanNet, ScanNet200, S3DIS; 3D analog of Mask2Former |
| **SoftGroup** (Vu 2022) | Soft semantic scores for bottom-up grouping | Avoids hard semantic prediction errors |

### 5.5 3D Libraries and Frameworks

| Library | Focus | Key Features |
|---|---|---|
| **PyTorch3D** (Meta) | Differentiable 3D operators | Differentiable rendering, 3D transforms, Chamfer distance, GPU-accelerated |
| **Open3D** | 3D data processing and visualization | Point cloud I/O, filtering, registration; `Open3D-ML` for DL |
| **MinkowskiEngine** (NVIDIA) | Sparse convolutions on 4D spatiotemporal data | Used by many SOTA 3D segmentation methods |
| **MMDetection3D** (OpenMMLab) | Comprehensive 3D detection toolbox | VoxelNet, PointPillars, CenterPoint, SECOND, BEVFormer; KITTI/nuScenes/Waymo/ScanNet |
| **Torch-Points3D** | Unified framework for point cloud DL | MinkowskiEngine + torchsparse backends |
| **torchsparse** | Fast sparse convolutions | Lighter alternative to MinkowskiEngine |
| **spconv** | Spatially sparse convolutions | Used by SECOND, CenterPoint, VoxelNeXt |

---

## 6. Enterprise Architecture Patterns

### 6.1 Model Registry and Versioning

| Tool | Capabilities |
|---|---|
| **MLflow Model Registry** | Centralized store with versioning, staging (None->Staging->Production->Archived); `mlflow.pytorch.log_model()` |
| **W&B Registry** | Model artifacts with lineage tracking, linked to experiments |
| **DVC** | Git-like versioning for models and datasets; S3/GCS/Azure backends |
| **HuggingFace Hub** | Model cards, versioning, access control, inference API |

### 6.2 Feature Stores for Embeddings

```python
# Pre-compute embeddings and store for downstream tasks
import timm, torch

model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True)
model.eval()

with torch.no_grad():
    embeddings = model.forward_features(batch)  # (B, N, D) or (B, D)

# Storage options:
# - Vector databases: FAISS, Milvus, Pinecone, Weaviate, Qdrant
# - Feature stores: Feast, Tecton, Hopsworks
# - Simple: HDF5, Parquet with numpy arrays, LanceDB
```

### 6.3 Inference Serving

| Framework | Strengths | Best For |
|---|---|---|
| **TorchServe** (AWS+Meta) | Native PyTorch; `.mar` archiving, batching, A/B testing, model versioning, DeepSpeed support | PyTorch-centric; `torch.compile`, `torch.export` |
| **NVIDIA Triton** | Multi-framework; dynamic batching, model ensembles, GPU scheduling | Multi-framework at scale |
| **BentoML** | Pythonic API; easy containerization; adaptive batching; MLflow integration | Rapid prototyping to production |
| **TorchScript / torch.export** | AOT compilation; no Python runtime | Edge/mobile (ExecuTorch) |
| **ONNX Runtime** | Cross-platform; graph optimizations | Diverse hardware (CPU, GPU, NPU) |

### 6.4 MLOps Pipeline

| Component | Tools |
|---|---|
| **Experiment Tracking** | MLflow (`mlflow.pytorch.autolog()`), W&B, ClearML, Neptune |
| **Pipeline Orchestration** | Kubeflow Pipelines, Airflow, Prefect, Metaflow, ZenML |
| **Model Monitoring** | Evidently AI, Seldon Alibi Detect, WhyLabs, Arize |
| **CI/CD for ML** | GitHub Actions + DVC/CML, GitLab CI |
| **Data Labeling** | Label Studio, CVAT, Labelbox, Scale AI, Roboflow |

### 6.5 Scalable Training

| Strategy | Mechanism | When to Use |
|---|---|---|
| **DDP** | Replicates model on each GPU; synchronizes gradients via all-reduce | Default for multi-GPU; model fits on one GPU |
| **FSDP** | Shards params, gradients, optimizer states across GPUs; gathers on-demand | Native PyTorch; 100M-1B params; best default for 2-8 GPU fine-tuning |
| **DeepSpeed ZeRO** | Stage 1/2/3 progressive sharding; CPU/NVMe offloading | 10B+ params; extreme memory constraints |
| **Tensor Parallelism** | Splits individual ops across GPUs (Megatron-LM) | Very large single layers |
| **Pipeline Parallelism** | Splits model layers across GPUs with micro-batching | Complements data/tensor parallelism |

```python
# FSDP (PyTorch native)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=size_based_auto_wrap_policy,
    mixed_precision=MixedPrecision(param_dtype=torch.float16),
)

# Launch: torchrun --nproc_per_node=4 train.py
```

---

## 7. Ecosystem Reference Matrix

| Library | Domain | Maintained By | Key Strengths |
|---|---|---|---|
| **`timm`** | Classification backbones | HuggingFace | 900+ architectures; pretrained weights; DINOv3, SigLIP 2 |
| **`torchvision`** | Det, seg, classification | PyTorch Core | Official; Faster/Mask R-CNN, FCOS, RetinaNet, DeepLabv3, Swin |
| **Detectron2** | Det, instance/panoptic seg | Meta (FAIR) | Mask2Former, Cascade R-CNN, Panoptic FPN; config-driven |
| **MMDetection** | Detection | OpenMMLab | 300+ model configs; DINO, Co-DETR, Deformable DETR |
| **MMDetection3D** | 3D detection | OpenMMLab | VoxelNet, SECOND, CenterPoint, BEVFormer; KITTI/nuScenes/Waymo |
| **MMSegmentation** | Semantic segmentation | OpenMMLab | SegFormer, Segmenter, DeepLabv3+, 50+ architectures |
| **`ultralytics`** | YOLO + RT-DETR | Ultralytics | YOLOv5/v8/11/12/26; det, seg, pose, OBB, classification |
| **`segmentation_models_pytorch`** | Semantic segmentation | qubvel | U-Net, U-Net++, FPN, DeepLabv3+; 400+ encoders via `timm` |
| **HuggingFace Transformers** | Multi-task | HuggingFace | ViT, Swin, DeiT, DETR, Mask2Former, OneFormer, SAM2, SegFormer, CLIP, DINOv2 |
| **`torchgeo`** | Geospatial/RS | TorchGeo Org | CRS-aware datasets, spatial samplers, pretrained RS backbones |
| **`rf-detr`** | Real-time detection | Roboflow | DINOv2 backbone; SOTA COCO mAP; ICLR 2026 |

---

## 8. Architecture Selection Guide

| Use Case | Recommended Architecture(s) | Library |
|---|---|---|
| Classification (general) | ConvNeXtV2 or EfficientNetV2 (production); Swin/ViT (max accuracy) | `timm` |
| Classification (mobile/edge) | MobileNetV3/V4, ShuffleNetV2 | `timm`, `torchvision` |
| Classification (few-shot/zero-shot) | DINOv2/v3 (linear probe), CLIP (zero-shot) | `dinov2`, `open_clip`, `timm` |
| Detection (real-time) | YOLO26 (edge), RF-DETR (accuracy-speed) | `ultralytics`, `rf-detr` |
| Detection (accuracy-first) | Co-DETR, DINO, Cascade R-CNN | MMDetection |
| Instance segmentation | Mask2Former (accuracy), YOLO26-seg (real-time) | Detectron2, `ultralytics` |
| Semantic segmentation | SegFormer (accuracy+efficiency), DeepLabv3+ (proven) | HuggingFace, `smp` |
| Panoptic segmentation | OneFormer (unified SOTA), Mask2Former | HuggingFace |
| Promptable segmentation | SAM 2 | Meta `sam2`, HuggingFace |
| Satellite/multi-spectral | SatMAE++ backbone + smp decoder; SpectralGPT | `torchgeo`, custom |
| 3D detection (LiDAR) | CenterPoint (anchor-free), SECOND (fast) | MMDetection3D, OpenPCDet |
| 3D detection (camera-only) | BEVFormer | MMDetection3D |
| 3D instance segmentation | Mask3D (SOTA), PointGroup | MinkowskiEngine |
| Serving | TorchServe (PyTorch-native), Triton (multi-framework) | — |
| Distributed training (<=1B params) | FSDP | PyTorch native |
| Distributed training (>10B params) | DeepSpeed ZeRO Stage 2/3 | `deepspeed` |

---

## 9. Phased Build Roadmap

### Phase 1 — Foundation (start here)
- `core/` infrastructure: config, registry, trainer with DDP support
- `data/` pipeline: standard datasets + multispectral loader with rasterio
- `backbones/` via `timm` wrappers with multi-channel stem support
- Classification module (simplest end-to-end validation of infra)
- First notebook demonstrating classification on standard dataset

### Phase 2 — Detection
- FPN neck implementation
- Faster R-CNN (full RPN -> RoI -> head pipeline)
- RetinaNet with focal loss (anchor-based one-stage reference)
- FCOS (anchor-free reference)
- YOLO integration via ultralytics wrapper
- DETR / Deformable DETR (transformer-based reference)

### Phase 3 — Segmentation
- U-Net family via `smp` with custom encoders
- Mask R-CNN (instance)
- DeepLabv3+ (semantic)
- Mask2Former / OneFormer (unified panoptic)
- SAM2 integration for promptable segmentation

### Phase 4 — Satellite / Multi-Spectral
- 16-bit data pipeline with percentile normalization
- Spectral index computation (NDVI, NDWI, etc.)
- TorchGeo dataset integration
- SatMAE / Prithvi foundation model wrappers
- Multi-temporal stacking support

### Phase 5 — 3D
- Point cloud data loading (Open3D, PLY/LAS formats)
- PointNet / PointNet++ implementations
- Voxel-based: SECOND / CenterPoint via spconv
- BEV: BEVFormer wrapper
- 3D instance segmentation: Mask3D via MinkowskiEngine

### Phase 6 — Enterprise / Serving
- MLflow integration for experiment tracking + model registry
- TorchServe handlers for each task type
- ONNX export pipeline
- Embedding extraction + vector store integration
- Monitoring hooks (drift detection)

---

## 10. Core Dependencies

```toml
[project]
dependencies = [
    "torch>=2.3",
    "torchvision>=0.18",
    "timm>=1.0",                        # 900+ classification backbones
    "segmentation-models-pytorch>=0.4",  # U-Net, DeepLab, FPN
    "albumentations>=2.0",               # augmentations
    "ultralytics>=8.3",                  # YOLO family
    "rasterio>=1.3",                     # geospatial raster I/O
    "open3d>=0.18",                      # 3D point cloud processing
    "pytorch3d>=0.7",                    # differentiable 3D ops
    "mlflow>=2.15",                      # experiment tracking
    "hydra-core>=1.3",                   # config management
    "pycocotools",                       # COCO evaluation
]

[project.optional-dependencies]
geo = ["torchgeo>=0.6"]
3d = ["MinkowskiEngine>=0.5", "spconv-cu120>=2.3"]
serve = ["torchserve", "bentoml>=1.2"]
```

---

## Key Trends (mid-2026)

1. **Anchor-free, NMS-free detectors** are now standard (YOLO26, RF-DETR)
2. **Foundation models** (DINOv2/v3, SAM2, CLIP) replacing task-specific pretraining
3. **Transformer-based segmentation** dominating via Mask2Former/OneFormer
4. **FSDP** becoming default distributed training for most workloads
5. **BEV representations** unifying multi-camera 3D perception for autonomous driving
