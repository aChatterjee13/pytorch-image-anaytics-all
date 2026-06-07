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

### Design contracts

- **Semantic samples**: `(image (C,H,W) float, mask (H,W) int64)` with `255`
  as the ignore index. Mask-aware augmentation via `tv_tensors.Mask` (nearest
  interpolation preserved automatically by v2 transforms).
- **Instance samples**: Phase 2 detection target dicts extended with
  `"masks": (N,H,W) uint8`.
- **Synthetic fixture**: shapes-with-masks reuses the Phase 2 shape rasterizer
  — the drawn pixels *are* the mask, so detection and segmentation fixtures
  stay consistent.
- **`SegmentationEvaluator`**: streaming C×C confusion matrix via bincount of
  `target*C + pred` over non-ignored pixels (same pattern/sync as Phase 1
  evaluators). Metrics: `mIoU` (mean over classes with support), per-class
  IoU, `dice`, `pixel_accuracy`.
- **No new trainer for semantic seg**: the base Trainer already handles
  `(inputs, targets) + criterion` — `CrossEntropyLoss(ignore_index=255)`
  works as-is; the evaluator argmaxes logits in `update`. HF models fine-tune
  through the existing loss-dict path (`_compute_loss` accepts
  `{"loss": ...}`); a thin wrapper upsamples their 1/4-resolution logits for
  evaluation. Instance seg reuses `DetectionTrainer`.

### Architecture decisions

- **U-Net (scratch)**: encoder = any registered backbone in pyramid mode
  (5 levels); decoder block = upsample ×2 → concat skip → double conv;
  decoder widths (256,128,64,32,16) configurable. Multi-channel capable out
  of the box (Phase 4 synergy: 13-band U-Net for free).
- **DeepLabv3+ (scratch)**: ASPP rates (1,6,12,18) + image-level pooling on
  C5 at output-stride 16 (timm `output_stride=16` dilation support); decoder
  fuses C2 via 48-channel projection.
- **Losses**: `DiceLoss` (soft, per-class averaged), `CombinedLoss`
  (weighted CE+Dice, the practical default), focal variant; all honor
  ignore_index.
- **Mask R-CNN**: mask branch on Phase 2 Faster R-CNN — RoIAlign 14×14 with
  FPN level assignment `k = 4 + log2(sqrt(area)/224)`, 4 convs + deconv →
  28×28 per-class masks, BCE on positive RoIs vs cropped GT masks; mask mAP
  via pycocotools RLE (`iouType="segm"`).
- **Panoptic quality**: `PQ = Σ IoU(TP) / (|TP| + |FP|/2 + |FN|/2)` with
  matching at IoU > 0.5; reported as PQ/SQ/RQ.
- **SAM (v1)** via HF `SamModel`/`SamProcessor` — inference-only promptable
  API `predict(image, points=…, boxes=…)`; SAM2 lazy-gated
  ("requires torch>=2.3.1").

### Build order

| Step | Deliverable | Key files |
|---|---|---|
| 3.1 | Data + eval: mask dataset (class-index PNGs), `tv_tensors.Mask` transforms, synthetic shapes-with-masks; `SegmentationEvaluator` | `data/datasets/segmentation.py`, `data/transforms/segmentation.py`, `core/evaluator.py` |
| 3.2 | Seg losses: Dice, CE+Dice combo, focal variant | `segmentation/losses.py` |
| 3.3 | **U-Net from scratch** over timm pyramid encoders + **smp wrapper** (U-Net++, MAnet, Linknet, PSPNet…, 400+ encoders) | `segmentation/semantic/unet.py`, `segmentation/semantic/smp_wrapper.py` |
| 3.4 | **DeepLabv3+ from scratch** (ASPP + low-level fusion) | `segmentation/semantic/deeplab.py` |
| 3.5 | SegFormer via HF (loss-dict fine-tuning, logit-upsampling wrapper) | `segmentation/semantic/segformer.py` |
| 3.6 | `segmentation/train.py` + configs + smoke test | `segmentation/train.py` |
| 3.7 | **Mask R-CNN** + mask mAP | `segmentation/instance/mask_rcnn.py` |
| 3.8 | Mask2Former + OneFormer HF wrappers; PQ metric | `segmentation/instance/mask2former.py`, `segmentation/panoptic/oneformer.py` |
| 3.9 | **SAM** promptable wrapper; SAM2 gated | `foundation/sam.py` |
| 3.10 | Notebook 03 (U-Net on synthetic masks; SegFormer fine-tune; SAM prompting) | `notebooks/03_segmentation.ipynb` |

**New deps:** `segmentation-models-pytorch`, `transformers` (`[seg]` extra; `transformers` shared with `[foundation]`).

**Acceptance:** Dice/IoU metrics vs hand-computed values; U-Net +
DeepLabv3+ overfit a synthetic batch; e2e `run(config)` smoke on
shapes-with-masks reaches mIoU > 0.5 in a 2-minute CPU run; mask mAP
parity-checked against pycocotools `segm` on a fixture.

---

## Phase 4 — Satellite / Multi-Spectral

Phase 1 already shipped the 16-bit pipeline, percentile normalization,
spectral indices, multispectral dataset, and channel attention.

### Design contracts

- **Band-group stems** (SatMAE strategy 3): `GroupedBandStem(band_groups)` —
  each spectral group (e.g. Sentinel-2 RGB / red-edge / SWIR) gets its own
  conv stem; outputs concatenated before the backbone body. Configured via
  `backbone.kwargs.band_groups: [[0,1,2],[3,4,5,6],[7,8,9]]`.
- **Foundation backbones register like any other**: `satmae_base` and
  `prithvi_100m` land in `BACKBONES`, so any task head (classifier, U-Net,
  FPN) consumes them through the existing config path. SatMAE = ViT with
  grouped-band patch embed + spectral positional encodings, weights pulled
  from HF hub with key remapping. Prithvi = temporal ViT (3D patch embed over
  T×H×W) accepting `(B, C, T, H, W)`.
- **Temporal samples**: `TemporalStackDataset` wraps co-registered raster
  time series → `(T, C, H, W)` float tensors; heads choose `mean` / `max` /
  attention pooling over per-frame features.
- **Change detection**: Siamese shared-weight pyramid encoder on
  `(img_t0, img_t1)`; per-level feature differences feed a U-Net-style
  decoder (Phase 3 reuse) → binary change mask; BCE+Dice loss. Synthetic
  fixture: a shapes scene and a mutated copy — the mutation mask is the GT.
- **TorchGeo adapter**: converts torchgeo sample dicts (`image`/`mask`,
  CRS-aware) to our protocols; geo-aware samplers (Random/GridGeoSampler)
  pass through as DataLoader samplers. CRS handling stays inside torchgeo.

### Build order

| Step | Deliverable | Key files |
|---|---|---|
| 4.1 | Band-group stems + tests | `backbones/multichannel.py` |
| 4.2 | SatMAE + Prithvi backbone wrappers (HF hub weights) | `foundation/satmae.py`, `foundation/prithvi.py` |
| 4.3 | TorchGeo dataset adapter + geo samplers | `data/datasets/torchgeo_adapter.py`, `data/samplers.py` |
| 4.4 | Temporal stacking + pooling heads + Siamese change detection | `data/datasets/temporal.py`, `segmentation/change_detection.py` |
| 4.5 | EuroSAT-MS configs (scratch 13-band CNN vs SatMAE linear probe), BigEarthNet multilabel template, notebook 04 | `configs/classification/eurosat_*.yaml`, `notebooks/04_satellite_multispectral.ipynb` |

**New deps:** `torchgeo` (grows the `[geo]` extra).

**Acceptance:** synthetic GeoTIFF time-series round-trip (Phase 1 fixture +
T dim); band-group stem shape/gradient tests; change detection overfits a
synthetic pair; foundation wrappers load and forward (weight download marked
slow/manual); EuroSAT configs parse and build.

---

## Phase 5 — 3D (pure-PyTorch core, CUDA wrappers gated)

### Design contracts

- **Point-cloud samples**: `(points (N, 3+F) float, target)` — XYZ plus
  optional features (intensity, normals). Classification targets are ints;
  per-point segmentation targets are `(N,)` int64; detection targets are 3D
  box dicts.
- **3D boxes**: `(x, y, z, dx, dy, dz, yaw)` — center, dimensions, heading.
  `box3d.py` provides axis-aligned 3D IoU (exact), BEV rotated IoU, and
  3D IoU (BEV ∩ × z-overlap). 3D mAP reuses the Phase 2 evaluator pattern
  with these IoU kernels.
- **Pure-torch point ops** (`detection_3d/ops.py`): farthest-point sampling
  (iterative argmax of running min-distance), ball query (cdist + radius
  mask, capped at K), kNN, gather/index utilities — all batched, CPU-fine at
  test scale; later swappable for CUDA kernels behind the same signatures.
- **Synthetic fixtures**: (a) classification — points sampled from cube /
  sphere / plane surfaces with jitter + random pose; (b) detection — ground
  plane with cuboid point clusters and their GT boxes. Both deterministic
  per index, same pattern as Phase 2 shapes.
- **CUDA gating**: `voxel.py` / `bev.py` / `segmentation_3d.py` lazy-import
  spconv / mmdet3d / MinkowskiEngine with actionable errors; tests
  `skipif(not torch.cuda.is_available())`. Verified on a GPU box later —
  code review only, locally.

### Architecture decisions

- **PointNet**: input + feature T-Nets (with orthogonality regularizer),
  shared MLPs (64,64,128,1024) + max-pool; classification and per-point
  segmentation heads.
- **PointNet++**: SetAbstraction = FPS → ball query → mini-PointNet (SSG;
  MSG optional); FeaturePropagation = inverse-distance interpolation + unit
  PointNet for segmentation.
- **DGCNN**: EdgeConv on dynamic kNN graphs recomputed per layer in feature
  space; edge features `[x_i, x_j - x_i]`, max aggregation.
- **PointPillars** (the one mainstream 3D detector needing no sparse conv):
  pillar feature net (per-point decorated features → linear+BN+ReLU → max
  per pillar) → scatter to BEV canvas → 2D conv backbone (down/up blocks,
  concat) → SSD-style head with sin/cos yaw encoding, focal cls + smooth-L1
  reg; axis-aligned simplification first (synthetic tests), rotated NMS after.

### Build order

| Step | Deliverable | Key files |
|---|---|---|
| 5.1 | Point ops (FPS, ball query, kNN) + loaders (PLY/NPZ/OFF via `plyfile`) + transforms + synthetic primitives dataset | `detection_3d/ops.py`, `data/datasets/pointcloud.py`, `data/transforms/pointcloud.py` |
| 5.2 | PointNet + PointNet++ (cls + seg heads) | `detection_3d/pointnet.py` |
| 5.3 | DGCNN | `detection_3d/dgcnn.py` |
| 5.4 | box3d utils + PointPillars + synthetic 3D detection fixture | `detection_3d/box3d.py`, `detection_3d/pointpillars.py` |
| 5.5 | CUDA-gated wrappers: SECOND/CenterPoint (spconv), BEVFormer (mmdet3d), Mask3D (MinkowskiEngine) | `detection_3d/voxel.py`, `detection_3d/bev.py`, `detection_3d/segmentation_3d.py` |
| 5.6 | 3D evaluators + notebook 05 (PointNet++ on synthetic / ModelNet sample) | `core/evaluator.py`, `notebooks/05_3d_pointcloud.ipynb` |

**New deps:** `plyfile` (`[3d]`); `spconv-cu1xx`, `mmdet3d` (`[3d-cuda]`, Linux/CUDA only).

**Acceptance:** FPS/ball-query unit tests vs brute-force reference; PointNet
and PointNet++ overfit synthetic primitives to >90% accuracy on CPU;
axis-aligned 3D IoU vs hand-computed values; PointPillars overfits a
synthetic scene; gated wrappers import-error cleanly without CUDA.

---

## Phase 6 — Enterprise / Serving

### Design contracts

- **MLflowCallback**: `on_fit_start` opens a run under
  `experiment_name` and logs the flattened config as params; `on_epoch_end`
  logs `trainer.metrics` at `step=epoch`; `on_fit_end` logs artifacts
  (best/last checkpoints, `config.yaml`) and optionally
  `mlflow.pytorch.log_model(..., registered_model_name=…)` for registry
  staging. No-op when mlflow is absent or `training.mlflow: false`; tracking
  URI via standard `MLFLOW_TRACKING_URI`. Main-process only under DDP.
- **ONNX export**: `torch.onnx.export` (dynamic batch axis), then a
  mandatory **onnxruntime parity check** (CPU, |Δ| < 1e-4) before the file is
  considered exported. Classifiers export whole; detectors export the
  backbone+heads graph with decode/NMS kept in Python (documented Triton
  ensemble pattern) since NMS-in-graph export is brittle across opsets.
- **Inference CLI**: `scripts/infer.py --config … --checkpoint … --input
  <file|dir|glob> --output preds.json` — task-dispatched (classification:
  top-k probs; detection: boxes/scores/labels; later: masks), optional
  `--visualize` writing annotated images.
- **TorchServe handler**: one `BaseHandler` subclass per task; preprocess
  rebuilds the transform pipeline from the archived `config.yaml`, so
  serving preprocessing can never drift from training. Kept thin —
  TorchServe has been in maintenance mode since 2024; BentoML documented as
  the alternative path.
- **Triton config generator**: emits `config.pbtxt` (onnxruntime backend,
  dynamic batching, instance groups) from a model-metadata dataclass — pure
  templating, fully unit-testable.
- **Embeddings**: `scripts/embed.py` batches `forward_features` over a
  dataset → parquet/npz (paths, labels, vectors); optional
  `faiss.IndexFlatIP` build. This is the feature-store on-ramp.
- **Drift monitoring** (`serving/monitoring.py`, pure numpy):
  `DriftMonitor.fit(reference)` stores per-dimension histograms +
  prediction distributions; `score(batch)` reports PSI per feature
  (`Σ (p-q)·ln(p/q)`, alert > 0.2), KS statistics, and embedding cosine
  shift. Evidently optional, never required.

### Build order

| Step | Deliverable | Key files |
|---|---|---|
| 6.1 | **MLflowCallback** — *pull-forward candidate: ~100 lines, could land mid-Phase 3 to track all subsequent experiments* | `core/mlflow.py` |
| 6.2 | ONNX export + parity gate + `scripts/export.py` | `serving/onnx_export.py` |
| 6.3 | Unified inference CLI | `scripts/infer.py` |
| 6.4 | TorchServe handlers + `.mar` packaging script | `serving/torchserve_handler.py` |
| 6.5 | Triton `config.pbtxt` generator | `serving/triton_config.py` |
| 6.6 | Embedding extraction + optional FAISS index | `scripts/embed.py`, `serving/embeddings.py` |
| 6.7 | Drift monitoring (PSI/KS/cosine-shift) | `serving/monitoring.py` |
| 6.8 | CI: GitHub Actions (CPU test matrix, ruff lint) | `.github/workflows/ci.yml` |

**New deps:** `[serve]`: `onnx`, `onnxruntime`, `mlflow`; optional `faiss-cpu`, `bentoml`.

**Acceptance:** ONNX parity test green for classifier + RetinaNet raw-heads
export; infer CLI produces correct JSON on the synthetic fixtures; Triton
configs validated against the reference schema; PSI/KS vs hand-computed
values; MLflow run round-trip against a local file store; CI green on a
clean clone.

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
