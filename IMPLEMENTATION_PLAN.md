# Implementation Plan — Phases 2–6

> Companion to [EXPLORATION.md](EXPLORATION.md). Phase 1 (Foundation) shipped:
> registry, config, DDP-aware Trainer, evaluators, callbacks, data pipeline
> (incl. multispectral), timm backbones, classification end-to-end.
>
> Decisions locked in: core detectors **from scratch** on `torchvision.ops`
> primitives; 3D scoped to **pure-PyTorch core + CUDA-gated wrappers**;
> phases land sequentially with tests green at every step.

---

## Cross-cutting conventions (established in Phase 1, carried forward)

1. **Registry everything** — new component types get registries in
   `core/registry.py` (`NECKS`, `HEADS` added in Phase 2). Wrappers and
   from-scratch models register identically; configs don't care which is which.
2. **Config-driven** — each task gets a `run(config)` in its `train.py`;
   `scripts/train.py` dispatches on `config.task`. New config dataclasses
   (e.g. `NeckConfig`, `HeadConfig`) extend `core/config.py`.
3. **Trainer subclassing** — tasks with richer batches override
   `training_step`/`eval_step` only. The Phase 1 loop (AMP, DDP, callbacks,
   checkpointing) is never reimplemented. Models returning `{"loss": ...}`
   dicts already work (`_compute_loss` handles them) — this is how HF
   wrapper models fine-tune through our Trainer.
4. **Synthetic-data-first testing** — every phase ships an offline synthetic
   dataset (Phase 1: `fake` + generated GeoTIFFs; Phase 2: shapes-with-boxes;
   Phase 3: shapes-with-masks; Phase 5: sampled point-cloud primitives).
   Tests never download. Real-data configs are provided but exercised manually.
5. **Optional deps stay optional** — heavy/platform-bound libraries are
   lazy-imported with actionable error messages and live in pyproject extras.
6. **Per-phase definition of done** — tests green; CPU smoke config runs via
   `scripts/train.py`; notebook added; README + this file updated.

### Local environment constraints (Intel Mac, CPU-only)

| Constraint | Consequence |
|---|---|
| torch capped at 2.2.2 (last x86_64 macOS wheel), numpy<2 | No torch>=2.3 APIs without fallback; SAM2 package (needs ≥2.3.1) is gated |
| No CUDA/MPS | spconv, MinkowskiEngine, mmdet3d unavailable → `[3d-cuda]` extra, tests skipped locally |
| CPU training only | Smoke runs use tiny synthetic datasets; real training targets a future GPU box |

---

## Phase 2 — Detection

**Goal:** from-scratch FPN, RetinaNet, FCOS, Faster R-CNN, and DETR over the
existing pyramid-mode backbones (`features_only=True`), plus YOLO via an
ultralytics wrapper. `torchvision.ops` supplies primitives only
(`roi_align`, `nms`, `box_iou`, `box_convert`).

**New config surface:** `model.neck` (`NeckConfig`), detection-specific
`model.head` settings, `task: detection`.

**Data interface:** batches are `(images, targets)` where `targets` is a list
of dicts (`boxes` XYXY, `labels`); collate keeps lists ragged. Transforms use
torchvision v2 `tv_tensors.BoundingBoxes` (box-aware flips/crops/resize work
out of the box in tv 0.17).

### Build order (each step independently committable)

| Step | Deliverable | Key files |
|---|---|---|
| 2.1 | Box/loss toolkit: focal loss, smooth-L1, GIoU/DIoU loss, `BoxCoder` (encode/decode deltas) | `detection/losses.py`, `detection/box_coder.py` |
| 2.2 | Data + eval: COCO-format dataset, **synthetic shapes dataset** (colored rects/circles with boxes — CPU-trainable in minutes), detection collate, box-aware transform builder, mAP evaluator (own AP@[.5:.95] for tests + pycocotools for real COCO) | `data/datasets/coco.py`, `data/datasets/synthetic_shapes.py`, `data/transforms/detection.py`, `core/evaluator.py` (+`DetectionEvaluator`) |
| 2.3 | FPN neck (lateral 1×1 + top-down + 3×3 smoothing, optional P6/P7), `NECKS` registry | `detection/necks/fpn.py` |
| 2.4 | Anchor machinery: `AnchorGenerator` (per-level scales/ratios), IoU `Matcher` (pos/neg/ignore thresholds), balanced sampler | `detection/anchors/generator.py`, `detection/anchors/matcher.py` |
| 2.5 | **RetinaNet** end-to-end (first full detector): cls/reg towers, focal training, decode+NMS inference; `DetectionTrainer`; `detection/train.py`; smoke config | `detection/heads/retinanet.py`, `detection/trainer.py`, `detection/train.py` |
| 2.6 | **FCOS** (anchor-free reference): per-pixel (l,t,r,b) + centerness, center sampling | `detection/anchors/anchor_free.py`, `detection/heads/fcos.py` |
| 2.7 | **Faster R-CNN** (two-stage): RPN (objectness + proposals + NMS), RoIAlign via `torchvision.ops`, TwoMLP box head | `detection/anchors/rpn.py`, `detection/heads/faster_rcnn.py` |
| 2.8 | **DETR** (transformer reference): sine pos-encoding, encoder/decoder, Hungarian matcher (scipy), set criterion. Educational — sanity-trained on synthetic shapes only (real DETR needs 300+ epochs) | `detection/heads/detr.py` |
| 2.9 | YOLO wrapper (ultralytics, `[detection]` extra) + Cascade R-CNN (stretch, reuses 2.7 box heads) | `detection/yolo.py`, `detection/heads/cascade_rcnn.py` |
| 2.10 | Notebook 02 (train RetinaNet on synthetic shapes, visualize predictions; optional COCO128 section), README/EXPLORATION updates | `notebooks/02_detection.ipynb` |

**New deps:** `pycocotools`, `scipy` (core); `ultralytics` (`[detection]` extra).

**Tests:** IoU/GIoU/focal/box-coder vs hand-computed values; anchor counts &
coverage; matcher assignment tables; FPN shape contracts; per-model forward
shapes + loss-decreases-on-overfit (single synthetic batch); e2e
`run(config)` smoke with RetinaNet on shapes; pycocotools parity check of our
mAP on a small fixture.

---

## Phase 3 — Segmentation

**Goal:** semantic from scratch + smp breadth; instance by extending Phase 2
Faster R-CNN; panoptic/SOTA via HF wrappers; promptable via SAM.

### Build order

| Step | Deliverable | Key files |
|---|---|---|
| 3.1 | Data + eval: image/mask dataset (index masks), `tv_tensors.Mask` transforms, synthetic shapes-with-masks; `SegmentationEvaluator` (streaming confusion → mIoU/Dice/pixel-acc, same pattern as Phase 1) | `data/datasets/segmentation.py`, `data/transforms/segmentation.py`, `core/evaluator.py` |
| 3.2 | Seg losses: Dice, CE+Dice combo, boundary-aware focal | `segmentation/losses.py` |
| 3.3 | **U-Net from scratch** — decoder over our timm `features_only` encoders (multi-channel capable for Phase 4 synergy) + **smp wrapper** for the family (U-Net++, MAnet, Linknet, PSPNet…, 400+ encoders) | `segmentation/semantic/unet.py`, `segmentation/semantic/smp_wrapper.py` |
| 3.4 | **DeepLabv3+ from scratch**: ASPP (atrous pyramid) + low-level-fusion decoder | `segmentation/semantic/deeplab.py` |
| 3.5 | SegFormer via HF `transformers` (fine-tunes through our Trainer via loss-dict outputs) | `segmentation/semantic/segformer.py` |
| 3.6 | `SegmentationTrainer` + `segmentation/train.py` + configs + smoke test | `segmentation/train.py` |
| 3.7 | **Mask R-CNN**: mask branch (RoIAlign 14×14 → deconv → per-class masks) on Phase 2 Faster R-CNN; mask AP in evaluator | `segmentation/instance/mask_rcnn.py` |
| 3.8 | Mask2Former + OneFormer HF wrappers; basic PQ (panoptic quality) metric | `segmentation/instance/mask2former.py`, `segmentation/panoptic/oneformer.py` |
| 3.9 | **SAM** promptable segmentation via HF (works on torch 2.2); **SAM2 gated** (lazy import, clear "requires torch>=2.3.1" error locally) | `foundation/sam.py` |
| 3.10 | Notebook 03 (U-Net on synthetic masks; SegFormer fine-tune; SAM prompting demo) | `notebooks/03_segmentation.ipynb` |

**New deps:** `segmentation-models-pytorch`, `transformers` (`[seg]` extra; `transformers` shared with `[foundation]`).

---

## Phase 4 — Satellite / Multi-Spectral

Phase 1 already shipped the 16-bit pipeline, percentile normalization,
spectral indices, multispectral dataset, and channel attention. Remaining:

| Step | Deliverable | Key files |
|---|---|---|
| 4.1 | Band-group stems (SatMAE strategy 3): separate conv stems per spectral group, fused | `backbones/multichannel.py` |
| 4.2 | **SatMAE / Prithvi backbone wrappers**: HF-hub weight loading, grouped-band patch embeds, registered into `BACKBONES` (usable by every task incl. U-Net/FPN) | `foundation/satmae.py`, `foundation/prithvi.py` |
| 4.3 | TorchGeo integration: dataset adapters (torchgeo → our protocol), **geo-aware samplers** (random/grid geo-sampling wrappers) | `data/datasets/torchgeo_adapter.py`, `data/samplers.py` |
| 4.4 | Multi-temporal: temporal stacking dataset wrapper (T,C,H,W), temporal pooling heads (mean/attention), **Siamese change detection** (bi-temporal diff features → U-Net decoder → change mask) | `data/datasets/temporal.py`, `segmentation/change_detection.py` |
| 4.5 | EuroSAT-MS configs (scratch CNN vs SatMAE linear probe), notebook 04 | `configs/classification/eurosat_*.yaml`, `notebooks/04_satellite_multispectral.ipynb` |

**New deps:** `torchgeo` (grows the `[geo]` extra).
**Tests:** synthetic GeoTIFF time-series (Phase 1 fixture pattern + T dim);
band-group stem shape tests; change-detection loss-decreases test.

---

## Phase 5 — 3D (pure-PyTorch core, CUDA wrappers gated)

| Step | Deliverable | Key files |
|---|---|---|
| 5.1 | Point-cloud data: PLY/NPZ/OFF loaders (`plyfile`, no Open3D hard dep), **synthetic primitives dataset** (sampled cubes/spheres/planes), transforms (rotate/jitter/scale/dropout), FPS + ball-query ops in pure torch | `data/datasets/pointcloud.py`, `data/transforms/pointcloud.py`, `detection_3d/ops.py` |
| 5.2 | **PointNet** (with T-Net) + **PointNet++** (SA/FP layers): classification + part-seg heads | `detection_3d/pointnet.py` |
| 5.3 | **DGCNN** (EdgeConv, dynamic kNN graphs) | `detection_3d/dgcnn.py` |
| 5.4 | **PointPillars from scratch** (the one mainstream 3D detector that needs no sparse conv: pillar VFE → scatter to BEV → 2D CNN → SSD head, simplified rotated NMS) + 3D box utils/IoU | `detection_3d/pointpillars.py`, `detection_3d/box3d.py` |
| 5.5 | CUDA-gated wrappers behind `[3d-cuda]`: SECOND/CenterPoint (spconv), BEVFormer (mmdet3d), Mask3D (MinkowskiEngine). Lazy imports, tests `skipif(not cuda)` — verified later on a GPU box | `detection_3d/voxel.py`, `detection_3d/bev.py`, `detection_3d/segmentation_3d.py` |
| 5.6 | 3D evaluator (cls metrics reuse; BEV/3D IoU mAP for detection), notebook 05 (PointNet++ on synthetic/ModelNet sample) | `core/evaluator.py`, `notebooks/05_3d_pointcloud.ipynb` |

**New deps:** `plyfile` (`[3d]`); `spconv-cu1xx`, `mmdet3d` (`[3d-cuda]`, Linux/CUDA only).

---

## Phase 6 — Enterprise / Serving

| Step | Deliverable | Key files |
|---|---|---|
| 6.1 | **MLflowCallback** (params/metrics/artifacts + `mlflow.pytorch.log_model` registry staging). *Pull-forward candidate: ~100 lines, could land during Phase 2 to track all subsequent experiments* | `core/callbacks.py` or `core/mlflow.py` |
| 6.2 | **ONNX export** + onnxruntime parity tests (CPU-testable locally): classifier first, then exportable detectors; dynamic batch axes | `serving/onnx_export.py`, `scripts/export.py` |
| 6.3 | Unified inference CLI: image/dir → JSON/CSV predictions, task-dispatched | `scripts/infer.py` |
| 6.4 | TorchServe handlers per task + `.mar` packaging script (kept thin — TorchServe is in maintenance mode since 2024; BentoML noted as alternative) | `serving/torchserve_handler.py` |
| 6.5 | Triton `config.pbtxt` generator (pure templating, fully testable) | `serving/triton_config.py` |
| 6.6 | Embedding extraction: batch `forward_features` → parquet/npz; optional FAISS index | `scripts/embed.py`, `serving/embeddings.py` |
| 6.7 | Drift monitoring: PSI/KS statistics over embeddings & prediction distributions (pure numpy core, testable; evidently optional) | `serving/monitoring.py` |
| 6.8 | CI: GitHub Actions (CPU test matrix, lint) | `.github/workflows/ci.yml` |

**New deps:** `[serve]`: `onnx`, `onnxruntime`, `mlflow`; optional `faiss-cpu`, `bentoml`.

---

## Dependency extras (target state)

```toml
[project.optional-dependencies]
detection = ["ultralytics>=8.3"]
seg = ["segmentation-models-pytorch>=0.4", "transformers>=4.45"]
geo = ["rasterio>=1.3", "torchgeo>=0.6"]
3d = ["plyfile>=1.0"]
3d-cuda = ["spconv-cu120>=2.3", "mmdet3d>=1.4"]      # Linux/CUDA only
serve = ["onnx>=1.16", "onnxruntime>=1.18", "mlflow>=2.15"]
dev = ["pytest>=8.0"]
```
`scipy` + `pycocotools` move into core deps at Phase 2 (Hungarian matching, COCO eval).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| DETR from scratch converges too slowly to validate | Sanity-train on synthetic shapes (loss decreases + overfit-one-batch test); document; HF Deformable-DETR wrapper as production path |
| Hand-rolled Faster R-CNN correctness | Step-wise unit tests (RPN proposal recall on synthetic data, matcher tables); compare against torchvision outputs on a fixed input |
| COCO too large for local work | Synthetic shapes for tests/demos; `coco128` (~7MB) for optional realism |
| SAM2 / torch≥2.3 locally unavailable | HF SAM (v1) works on 2.2; SAM2 lazy-gated with actionable error |
| Open3D wheel availability on Intel Mac | `plyfile`+numpy for I/O; Open3D never a hard dep |
| pyproject extras with CUDA-only packages break resolution | `[3d-cuda]` documented as Linux-only; never in default install |

## Sequencing

Phases land in order 2 → 3 → 4 → 5 → 6 (3 depends on 2's Faster R-CNN; 4 reuses
3's U-Net decoder for change detection; 6 exports models from all phases).
The only intentional deviation: MLflowCallback (6.1) may land early.
Each phase is broken into independently committable steps so work can pause/resume
at any boundary with tests green.
