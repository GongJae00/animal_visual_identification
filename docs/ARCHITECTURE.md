# CVI Architecture — Target and Current Runtime

> Status: this document contains a target architecture, not measured system
> capability. The canonical runtime is currently image-crop appearance
> enrollment/search. MiewID-msv3 is a 440×440 general wildlife ReID benchmark
> candidate, not a nose-print model. Random TinyViT, DNP mask, landmark graph,
> and fabricated uncertainty paths are disabled. No nose, landmark, fusion,
> open-set, video, edge, or service performance claim is supported yet.

## Design Principles

1. **Evidence earns inclusion**: appearance is the baseline; every added
   channel requires calibrated ablation evidence and compute justification
2. **Search-space reduction**: species → breed → color → individual
3. **Invariance-first**: each layer uses biologically invariant features
4. **Uncertainty-gated fusion**: low-confidence channels are suppressed, not averaged
5. **Signal maximization**: every preprocessing step maximizes SNR (super-resolution, denoising, segmentation)
6. **Continuous improvement**: every module is independently benchmarkable and tunable

```
Input Frame
  │
  ├─[1] Hierarchical Classifier──────── species / breed / color filter
  │
  ├─[2] Nose Print Pipeline──────────── 1st biometric (99.8% Rank-1)
  │     YOLO-Nose → SuperRes → UNet Mask → TinyViT+MagFace → 512-d
  │
  ├─[3] Skeletal Landmark Graph──────── 2nd biometric (pose-invariant)
  │     HRNet 17-pt → DGCNN → 256-d
  │
  ├─[4] Appearance (Uncertainty)──────── 3rd biometric (conditioned)
  │     DINOV2/ConvNeXt ArcFace → 384-d → evidential head → uncertainty
  │
  └─[5] Learned Fusion ──────────────── uncertainty-gated weighted sum
        → FAISS GPU Index → evidence breakdown
        → Open-Set (evidential threshold) → Temporal Aggregation
```

## Directory Layout

```
src/cvi/
├── evidence/                  # NEW: per-modality subpackage
│   ├── __init__.py
│   ├── base.py                # AbstractEvidencer
│   ├── nose_print.py          # YOLO-Nose + TinyViT + MagFace
│   ├── landmark_graph.py      # HRNet + DGCNN
│   ├── appearance.py          # DINOV2/ConvNeXt + uncertainty
│   └── quality.py             # Blur/lighting/occlusion estimator
├── classifier/                # NEW: hierarchical classification
│   ├── __init__.py
│   ├── species.py             # Canidae-level
│   ├── breed.py               # Breed-level
│   └── color.py               # Coat color/pattern
├── fusion/                    # NEW: evidence fusion
│   ├── __init__.py
│   ├── calibrator.py          # Per-channel ScoreCalibrator
│   ├── fuser.py               # LearnedWeightFuser + UncertaintyFuser
│   ├── open_set.py            # EvidentialOpenSet
│   └── temporal.py            # TemporalAggregator (improved)
├── index/                     # NEW: search index
│   ├── __init__.py
│   ├── base.py                # AbstractIdentityIndex
│   ├── faiss_gpu.py           # GpuIdentityIndex (existing)
│   └── hierarchical.py        # Species-filtered index shards
├── train/                     # NEW: training
│   ├── __init__.py
│   ├── config.py              # TrainConfig (extended)
│   ├── model.py               # ArcFaceModel + MagFaceModel
│   ├── dataset.py             # PetFaceDataset + OracleCropDataset
│   ├── augment.py             # Environmental augmentation
│   └── callbacks.py           # FAR/FRR/CMC eval during training
├── pipeline/                  # NEW: orchestration
│   ├── __init__.py
│   ├── enroll.py              # Multi-evidence enrollment
│   ├── search.py              # Multi-evidence search
│   └── explain.py             # Evidence explanation
└── post_search.py             # KEPT: legacy wrappers call new modules
```

## Phase Implementation Plan

### Phase 1 — Hierarchical Classifier
Goal: species → breed → color → filter search space 10x–100x
- PetFace (257K individuals, 13 families, 319 breeds) fine-tune ConvNeXt ArcFace
- Hierarchical softmax: family → genus → species → breed → color
- Inference: top-3 breed filter → downstream only searches within those breeds

### Phase 2 — Nose Print Candidate
Goal: determine whether a licensed, trained nose-specific channel improves the
frozen appearance baseline under the same leakage-controlled protocol
- YOLOv8-nose detector (from Nose-to-ID, 2025)
- Super-resolution enhancer for low-res crops
- UNet DNPMask segmentation (noise reduction)
- TinyViT backbone + MagFace loss + ArcFace head
- Multi-resolution ensemble (3 scales)
- Data: MiewID dataset + DNND (196K samples) + self-collected

### Phase 3 — Skeletal Landmark Graph
Goal: pose-invariant 2nd biometric
- DogFLW 17-point → HRNet-W32 heatmap
- Pose normalization → distance matrix → DGCNN (EdgeConv) → 256-d
- Synthetic pose augmentation (rotation, perspective, partial occlusion)

### Phase 4 — Appearance with Uncertainty
Goal: 3rd biometric, conditionally used when nose/landmark fail
- DINOv2/ConvNeXt fine-tune ArcFace (existing trainer.py)
- Evidential head: output evidence → Dirichlet → aleatoric + epistemic uncertainty
- Quality gate: blur/lighting/occlusion pre-filter (from quality.py)

### Phase 5 — Learned Fusion + Open-Set
Goal: uncertainty-gated, threshold-calibrated, temporally stable
- LearnedWeightFuser: softmax(confidence × attention)
- EvidentialOpenSet: reject if epistemic uncertainty > threshold
- TemporalAggregator: learned weighted median

### Phase 6 — E2E Evaluation
Goal: FAR/FRR/CMC + longitudinal invariance + domain shift
- Holdout: known-ID dataset with time-separated pairs
- Metrics: Rank-1, Rank-5, mAP, AUC, EER, TPR@FPR=1e-3
- Ablation: remove each evidence layer → measure impact
- Per-breed breakdown
- Domain shift: indoor/outdoor, day/night, summer/winter
