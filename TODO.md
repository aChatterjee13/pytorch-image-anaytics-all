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
- ⏳ **Phase 3 — Segmentation** ← next
- ⬜ Phase 4 — Satellite / Multi-Spectral
- ⬜ Phase 5 — 3D
- ⬜ Phase 6 — Enterprise / Serving

## Phase 2 leftovers (low priority)

- [ ] Cascade R-CNN (stretch goal — cascade of box heads at IoU 0.5/0.6/0.7
      reusing `heads/faster_rcnn.py` components)
- [ ] PAFPN neck (`detection/necks/pafpn.py` — bottom-up path augmentation)
- [ ] Letterbox resize for detection transforms (current: square resize with
      aspect distortion)
- [ ] Deformable-DETR / RT-DETR wrappers (production transformer detection;
      from-scratch DETR is educational, converges slowly by design)

## Phase 3 — Segmentation (next up)

- [ ] 3.1 Mask data pipeline: `data/datasets/segmentation.py` (class-index
      masks, 255=ignore), `tv_tensors.Mask` transforms, synthetic
      shapes-with-masks fixture (reuse the Phase 2 shape rasterizer)
- [ ] 3.1 `SegmentationEvaluator` (streaming confusion → mIoU / Dice /
      pixel-acc; same sync pattern as existing evaluators)
- [ ] 3.2 `segmentation/losses.py`: Dice, CE+Dice combo, focal variant
- [ ] 3.3 U-Net from scratch over timm pyramid encoders (multi-channel
      capable) + smp wrapper (`[seg]` extra)
- [ ] 3.4 DeepLabv3+ from scratch (ASPP rates 1/6/12/18, output-stride 16,
      C2 low-level fusion)
- [ ] 3.5 SegFormer via HF transformers (loss-dict fine-tune; upsample
      1/4-res logits for eval)
- [ ] 3.6 `segmentation/train.py` + configs + smoke (base Trainer suffices
      for semantic seg — no subclass needed)
- [ ] 3.7 Mask R-CNN: mask branch (RoIAlign 14×14 → deconv → 28×28
      per-class masks) on Phase 2 Faster R-CNN; mask mAP (pycocotools segm
      parity test)
- [ ] 3.8 Mask2Former + OneFormer HF wrappers; PQ/SQ/RQ metric
- [ ] 3.9 SAM promptable wrapper (HF, works on torch 2.2); SAM2 gated
      (needs torch>=2.3.1 — unavailable on this Intel Mac)
- [ ] 3.10 `notebooks/03_segmentation.ipynb`
- [ ] **Pull-forward 6.1: MLflowCallback** (~100 lines in `core/mlflow.py`)
      so all Phase 3+ experiments get tracked

## Phase 4 — Satellite / Multi-Spectral

- [ ] 4.1 Band-group stems (SatMAE strategy 3) in `backbones/multichannel.py`
- [ ] 4.2 SatMAE + Prithvi backbone wrappers (HF hub weights, registered
      into BACKBONES so any head can consume them)
- [ ] 4.3 TorchGeo dataset adapter + geo-aware samplers (`[geo]` extra grows)
- [ ] 4.4 Temporal stacking dataset `(T,C,H,W)` + pooling heads + Siamese
      change detection (reuses Phase 3 U-Net decoder)
- [ ] 4.5 EuroSAT-MS configs (scratch 13-band CNN vs SatMAE linear probe),
      BigEarthNet multilabel template, `notebooks/04_satellite_multispectral.ipynb`

## Phase 5 — 3D (pure-PyTorch core; CUDA wrappers gated)

- [ ] 5.1 Point ops (FPS, ball query, kNN — pure torch) + PLY/NPZ/OFF
      loaders (`plyfile`) + point transforms + synthetic primitives dataset
- [ ] 5.2 PointNet (T-Nets + orthogonality reg) + PointNet++ (SA/FP layers);
      classification + per-point seg heads
- [ ] 5.3 DGCNN (EdgeConv, dynamic kNN graphs)
- [ ] 5.4 `box3d.py` (xyz+dims+yaw format, BEV/3D IoU, 3D mAP) +
      PointPillars from scratch + synthetic 3D detection fixture
- [ ] 5.5 CUDA-gated wrappers (`[3d-cuda]`, needs a GPU box to verify):
      SECOND/CenterPoint (spconv), BEVFormer (mmdet3d), Mask3D (MinkowskiEngine)
- [ ] 5.6 3D evaluators + `notebooks/05_3d_pointcloud.ipynb`

## Phase 6 — Enterprise / Serving

- [ ] 6.1 MLflowCallback (see Phase 3 pull-forward)
- [ ] 6.2 ONNX export + mandatory onnxruntime parity gate +
      `scripts/export.py` (detectors: raw-heads graph, decode/NMS in Python)
- [ ] 6.3 `scripts/infer.py` unified CLI (file/dir/glob → JSON/CSV,
      optional --visualize)
- [ ] 6.4 TorchServe handlers (preprocess rebuilt from archived config) +
      .mar packaging; BentoML documented as alternative
- [ ] 6.5 Triton config.pbtxt generator
- [ ] 6.6 Embedding extraction (`scripts/embed.py` → parquet/npz, optional
      FAISS index)
- [ ] 6.7 Drift monitoring (PSI/KS/cosine-shift, pure numpy)
- [ ] 6.8 GitHub Actions CI (CPU test matrix, ruff lint)

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
