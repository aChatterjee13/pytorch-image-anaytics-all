# TODO — Pending Implementation

> Working checklist. Design specs and acceptance criteria for every item live
> in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md); architecture reference
> in [EXPLORATION.md](EXPLORATION.md). Update this file as items land.

## Status

- ✅ **Phase 1 — Foundation** (core infra, config, trainer, evaluators,
  data pipeline incl. multispectral, timm backbones, classification)
- ✅ **Phase 2 — Detection** (losses/box coder, synthetic shapes + COCO
  datasets, pycocotools-parity mAP evaluator, FPN, anchors/matcher/sampler,
  RetinaNet, FCOS, Faster R-CNN+RPN, DETR, YOLO wrapper, notebook 02)
- ✅ **Phase 3 — Segmentation** (mask data pipeline + evaluators, seg losses,
  U-Net, DeepLabv3+, smp + SegFormer wrappers, Mask R-CNN + mask-mAP,
  Mask2Former/OneFormer + PQ, SAM, notebook 03; MLflow pulled forward)
- ✅ **Phase 4 — Satellite / Multi-Spectral** (band-group stems, SatMAE/Prithvi
  backbones, TorchGeo adapter + geo samplers, temporal stacking + pooling,
  Siamese change detection, EuroSAT/BigEarthNet templates, notebook 04)
- ✅ **Phase 5 — 3D** (point ops, PointNet/PointNet++/DGCNN cls+seg, box3d +
  3D IoU/NMS, PointPillars + 3D-mAP, CUDA-gated SECOND/CenterPoint/BEVFormer/
  Mask3D, notebook 05) — CUDA wrappers verified by review only
- ✅ **Phase 6 — Enterprise / Serving** (MLflow, ONNX export + parity gate,
  inference CLI, TorchServe handler, Triton config generator, embeddings +
  FAISS, drift monitoring, GitHub Actions CI)

**All phases (1–6) implemented.** Remaining: real-data GPU runs, CUDA-3D
verification, and the backlog/nice-to-have items below.
- ⬜ Phase 5 — 3D
- ⬜ Phase 6 — Enterprise / Serving

## Phase 2 leftovers (done)

- [x] Cascade R-CNN (`heads/cascade_rcnn.py`) — 3 box heads at IoU 0.5/0.6/0.7,
      class-agnostic regression, stage-averaged inference; subclasses FasterRCNN
- [x] PAFPN neck (`detection/necks/pafpn.py`) — bottom-up path augmentation;
      drop-in for FPN (RetinaNet `neck: pafpn`)
- [x] Letterbox resize (`LetterboxResize` in `transforms/detection.py`,
      `data.letterbox: true`) — aspect-preserving resize + pad
- [x] Deformable-DETR / RT-DETR (`detection/hf_detr.py`) — HF wrappers that
      train through DetectionTrainer (loss-dict) and eval to prediction dicts

## Phase 3 — Segmentation (done)

- [x] 3.1 Mask data pipeline: `data/datasets/segmentation.py` (class-index
      masks, 255=ignore) + instance fixture, `tv_tensors.Mask` transforms,
      synthetic shapes-with-masks (shares the Phase 2 rasterizer via
      `data/datasets/_shapes.py`)
- [x] 3.1 `SegmentationEvaluator` (streaming confusion → mIoU / Dice /
      pixel-acc; same sync pattern as existing evaluators)
- [x] 3.2 `segmentation/losses.py`: Dice, CE+Dice combo, focal variant
      (+ `cross_entropy` factory); `training.loss` config selector added
- [x] 3.3 U-Net from scratch over timm pyramid encoders (multi-channel
      capable) + smp wrapper (`[seg]` extra)
- [x] 3.4 DeepLabv3+ from scratch (ASPP rates 1/6/12/18, output-stride 16,
      C2 low-level fusion)
- [x] 3.5 SegFormer via HF transformers (logit-upsampling wrapper → trains
      through base Trainer + external criterion)
- [x] 3.6 `segmentation/train.py` + configs (unet/deeplab/segformer/smoke)
      + dispatch in `scripts/train.py` + e2e smoke
- [x] 3.7 Mask R-CNN: mask branch (RoIAlign 14×14 → deconv → 28×28
      per-class masks) on Phase 2 Faster R-CNN; `MaskMAPEvaluator` mask mAP
      (pycocotools segm parity test green)
- [x] 3.8 Mask2Former + OneFormer HF wrappers; `PanopticQualityEvaluator`
      (PQ/SQ/RQ)
- [x] 3.9 SAM promptable wrapper (HF, works on torch 2.2); SAM2 gated
      (`load_sam2` raises on torch<2.3.1)
- [x] 3.10 `notebooks/03_segmentation.ipynb`
- [x] **Pull-forward 6.1: MLflowCallback** (`core/mlflow.py`) wired into all
      three task pipelines (`training.mlflow: true`); `[serve]` extra adds mlflow

## Phase 4 — Satellite / Multi-Spectral (done)

- [x] 4.1 Band-group stems (`GroupedBandStem`/`GroupedStemBackbone` in
      `backbones/multichannel.py`); wired via `backbone.kwargs.stem_band_groups`
- [x] 4.2 SatMAE + Prithvi backbones (`foundation/satmae.py`, `prithvi.py`,
      registered in BACKBONES; offline-instantiable, best-effort HF weight load)
- [x] 4.3 TorchGeo dataset adapter (`data/datasets/torchgeo_adapter.py`) +
      geo-aware samplers (`build_geo_sampler` in `data/samplers.py`)
- [x] 4.4 Temporal stacking (`synthetic_temporal`) + pooling head
      (`temporal_classifier`) + Siamese change detection (`siamese_unet` on
      `synthetic_change`, reuses the U-Net decoder; wired into the seg pipeline)
- [x] 4.5 EuroSAT-MS configs (scratch 13-band + SatMAE linear probe),
      BigEarthNet multilabel template, change-detection config, notebook 04

> Note (key naming): the generic grouped-stem wrapper uses
> `backbone.kwargs.stem_band_groups` (not `band_groups`), so it doesn't collide
> with `satmae_base`'s native `band_groups` patch-embed argument.

## Phase 5 — 3D (pure-PyTorch core; CUDA wrappers gated) — done

- [x] 5.1 Point ops (FPS, ball query, kNN — pure torch, `detection_3d/ops.py`)
      + PLY/NPZ/OFF loaders + point transforms + synthetic primitives/det datasets
- [x] 5.2 PointNet (T-Nets + orthogonality reg) + PointNet++ (SA/FP layers);
      classification + per-point seg heads
- [x] 5.3 DGCNN (EdgeConv, dynamic kNN graphs)
- [x] 5.4 `box3d.py` (xyz+dims+yaw, axis-aligned + rotated-BEV + 3D IoU, NMS) +
      PointPillars from scratch + synthetic 3D detection fixture
- [x] 5.5 CUDA-gated wrappers (`[3d-cuda]`): SECOND/CenterPoint (`voxel.py`),
      BEVFormer (`bev.py`), Mask3D (`segmentation_3d.py`) — gate on CPU, review-only
- [x] 5.6 `Detection3DEvaluator` (3D mAP) + `detection_3d/train.py`
      (cls/seg/detection dispatch) + configs + `notebooks/05_3d_pointcloud.ipynb`

## Phase 6 — Enterprise / Serving (done)

- [x] 6.1 MLflowCallback (pulled forward in Phase 3 — `core/mlflow.py`)
- [x] 6.2 ONNX export + onnxruntime parity gate (`serving/onnx_export.py`,
      `scripts/export.py`); detectors export raw heads (`RetinaNetRawHeads`)
- [x] 6.3 `scripts/infer.py` unified CLI (file/dir/glob → JSON, `--visualize`);
      postprocess in `serving/infer_utils.py`
- [x] 6.4 TorchServe handler (`serving/torchserve_handler.py`, preprocess rebuilt
      from archived config; `.mar` command documented; BentoML noted)
- [x] 6.5 Triton `config.pbtxt` generator (`serving/triton_config.py`)
- [x] 6.6 Embedding extraction (`serving/embeddings.py`, `scripts/embed.py`) +
      optional FAISS index
- [x] 6.7 Drift monitoring (PSI/KS/cosine-shift, pure numpy — `serving/monitoring.py`)
- [x] 6.8 GitHub Actions CI (`.github/workflows/ci.yml` — CPU matrix, ruff
      error-lint, `pytest -m "not slow"`)

## Backlog / nice-to-have (no phase)

- [ ] Hydra config swap-in if dataclass+YAML setup is outgrown
- [ ] Albumentations as alternative augmentation backend
- [ ] CheckpointCallback best-value persistence across resume
      (currently `best` resets when resuming a run)
- [ ] Per-parameter-group LRs (backbone vs head; DETR convention) in
      `build_optimizer`
- [ ] Real-data sanity runs on GPU hardware: CIFAR-10 full fine-tune,
      COCO RetinaNet/Faster R-CNN, then Phase 5 CUDA wrappers

## Environment reminders (this machine)

- Intel Mac: torch capped at **2.2.2**, `numpy<2`, CPU-only — keep smoke
  runs synthetic/small; SAM2 and `[3d-cuda]` cannot run locally
- Tests must stay offline (synthetic fixtures, `pretrained=false`,
  `.yaml`-built YOLO models)
