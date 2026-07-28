# CVI Canid-First ReID — Current Architecture Audit

- Starting commit: `bacdb5c`
- Ending commit: `bacdb5c`
- Date: 2026-07-28
- All assertions verified against actual code at the above commit.

## Implemented Path (production — callable from `CVI.enroll/search`)

```text
PIL.Image (dog crop)
  │
  ├─► MultiEvidencePipeline
  │     evidence["appearance"] ← ReceiptBoundDinov2Small
  │       backboone: local DINOv2-small (Safetensors, receipt-bound)
  │       preprocessing: shortest-edge 256, center-crop 224, bicubic
  │       output: float32 384D, L2-normalized
  │
  ├─► [optional channels — if configured]
  │     evidence["nose_print"] ← NosePrintExtractor (ONNX)
  │     evidence["landmark"]   ← LandmarkEvidencer (ONNX)
  │     evidence["*"]         ← generic OnnxExtractor (dogfacenet/convnext/petreid)
  │
  ▼
SpeciesFilteredIndex
  enrollment → FAISS IndexFlatIP + NPZ sidecars + gallery_manifest.v4
  search     → exact_available_intersection_weighted_cosine.v1
               per-channel cosine → weight sum → renormalize
               identity aggregation: max
  output     → Match(dog_id, similarity, evidence, ...)

Gallery format:
  gallery_manifest.json        cvi.gallery_manifest.v4
  required.idx                 FAISS binary index (uint8)
  optional_vectors.npz         sparse per-channel (deflate)
  ids.json                     template IDs + metadata
```

## Research-Only Path (implemented, not connected to CVI API)

```text
PIL.Image (dog crop, oracle nose ROI/keypoints/masks)
  │
  ├─► NoseID-v1 (src/cvi/nose_id/)
  │     NoseIDSample → alignment(448) → FactorizedNoseSegmenter(frozen)
  │     → DINOv2 patch tokens(RGB 336) + FixedFrequencyBank(448)
  │     → TextureConvNeXtS + ShapeStream + QualityHead
  │     → gated fusion 512D L2-norm embedding
  │
  ├─► Stage-C training (tools/train_nose_identity.py)
  │     P=16,K=4 cross-session + hard-neighbor sampler
  │     SubCenter ArcFace + SupCon + triplet + consistency + quality aux
  │     gradient-cache (64 logical, view-sequential backward)
  │     persistent scaler, scheduler, checkpoint v1
  │
  ├─► DEV evaluation (tools/evaluate_noseid_oracle.py)
  │     capture-disjoint fold protocol (≤3 folds, one gallery capture/id)
  │     A0/N0/NT/N3/F0_FIXED/F0_QUALITY metrics
  │
  └─► Hard-negative mining (tools/mine_nose_hard_negatives.py)

Quality fusion (src/cvi/fusion/quality_fuser.py)
  PositiveAffineCalibrator + QualityFusionMLP(27D→3)
  Not wired to gallery search — research-only
```

## Missing Path (not yet implemented)

```text
Dog detector → full-to-face crop bridge → CVI pipeline  | Step 03-04
Face bbox detector → face crop → face ReID                | Step 04-05
Dog detector → nose bbox → weak nose ReID                 | Step 06
Quality-aware fusion → gallery scorer connection           | Step 07
NoseIDExtractor → api.py channel integration               | Step 07
ONNX export → NoseID-v1 runtime deployment                 | Step 08
Open-set calibrated decision                               | Step 07-08
```

## Component Status Table

| Component | Status | Location |
|---|---|---|
| CVI.enroll/search | IMPLEMENTED | src/cvi/api.py |
| MultiEvidencePipeline | IMPLEMENTED | src/cvi/pipeline/enroll.py |
| IdentitySearchPipeline | IMPLEMENTED | src/cvi/pipeline/search.py |
| ReceiptBoundDinov2Small (appearance) | IMPLEMENTED | src/cvi/evidence/appearance.py |
| DINOv2 local artifact contract | IMPLEMENTED | src/cvi/evidence/dinov2_contract.py |
| NosePrintExtractor (nose_print_onnx) | IMPLEMENTED | src/cvi/evidence/nose_print.py |
| YoloNoseDetector (ONNX) | IMPLEMENTED | src/cvi/evidence/nose_print.py |
| DNPMask (nose mask, optional) | IMPLEMENTED | src/cvi/evidence/nose_print.py |
| LandmarkEvidencer | IMPLEMENTED | src/cvi/evidence/landmark_graph.py |
| HRNetHeatmap (ONNX) | IMPLEMENTED | src/cvi/evidence/landmark_graph.py |
| LandmarkGraphEmbedder (ONNX) | IMPLEMENTED | src/cvi/evidence/landmark_graph.py |
| MiewIDReIDExtractor | IMPLEMENTED | src/cvi/evidence/miewid.py |
| DogFaceNetExtractor (ONNX) | IMPLEMENTED | src/cvi/evidence_extractor.py |
| ConvNeXtExtractor (ONNX) | IMPLEMENTED | src/cvi/evidence_extractor.py |
| PetReIDExtractor (ONNX) | IMPLEMENTED | src/cvi/evidence_extractor.py |
| SpeciesFilteredIndex (gallery) | IMPLEMENTED | src/cvi/index/hierarchical.py |
| IdentityRegistry (UUIDv5) | IMPLEMENTED | src/cvi/identity_registry.py |
| RetriievalEvaluator | IMPLEMENTED | src/cvi/evaluation/retrieval.py |
| OpenSetEvaluator | IMPLEMENTED | src/cvi/evaluation/open_set.py |
| EvidentialOpenSet (fusion) | IMPLEMENTED | src/cvi/fusion/open_set.py |
| LearnedWeightFuser | RESEARCH_ONLY | src/cvi/fusion/fuser.py |
| PerChannelCalibrator | RESEARCH_ONLY | src/cvi/fusion/calibrator.py |
| QualityFusionMLP | RESEARCH_ONLY | src/cvi/fusion/quality_fuser.py |
| TemporalAggregator | RESEARCH_ONLY | src/cvi/fusion/temporal.py |
| DogDetector (YOLO) | IMPLEMENTED | src/cvi/detection.py |
| DogFLWLandmarkDetector | IMPLEMENTED | src/cvi/face_aligner.py |
| FaceAligner | IMPLEMENTED | src/cvi/face_aligner.py |
| TinyViTBackbone | DISABLED | src/cvi/evidence/nose_print.py |
| MagFaceNoseHead | DISABLED | src/cvi/evidence/nose_print.py |
| NoseEnhancer | DISABLED | src/cvi/evidence/nose_print.py |
| MiewIDNoseExtractor | DISABLED | src/cvi/evidence/nose_print.py |
| SuperAnimalExtractor | DISABLED | src/cvi/evidence_extractor.py |
| NoseID-v1 oracle core | RESEARCH_ONLY | src/cvi/nose_id/ |
| NoseIDDataset | RESEARCH_ONLY | src/cvi/nose_id/dataset.py |
| NoseIDModel | RESEARCH_ONLY | src/cvi/nose_id/model.py |
| NoseIDObjective | RESEARCH_ONLY | src/cvi/nose_id/losses.py |
| NoseID trainer/evaluator | RESEARCH_ONLY | src/cvi/nose_id/trainer.py |
| NoseID checkpoint | RESEARCH_ONLY | src/cvi/nose_id/checkpoint.py |
| NoseID protocol (DEV folds) | RESEARCH_ONLY | src/cvi/nose_id/protocol.py |
| NoseID training CLI | RESEARCH_ONLY | tools/train_nose_identity.py |
| NoseID eval CLI | RESEARCH_ONLY | tools/evaluate_noseid_oracle.py |
| NoseID hard-negative mine | RESEARCH_ONLY | tools/mine_nose_hard_negatives.py |
| Face bbox detector | MISSING | — |
| Face ReID dedicated model | MISSING | — |
| NoseID Extractor (api channel) | MISSING | — |
| Quality fusion → gallery | MISSING | — |
| Calibrated open-set | MISSING | — |
| ONNX NoseID export | MISSING | — |
| Video tracking/temporal | OUT_OF_SCOPE | — |

## Architectural Conflicts / Misleading Names

| Issue | Resolution |
|---|---|
| `nose_print_onnx` is bbox-only, not keypoint-aligned nor anatomically segmented | Rename to `nose_bbox_onnx` or keep legacy; NoseID-v1 is the canonical "nose_id" channel |
| `LearnedWeightFuser` exists but `IdentitySearchPipeline.search()` never calls `fuse()` | Quality/uncertainty from this class is dead code in the production path |
| `multi_head.py` and `train_multi_head_model.py` sound like NoseID but use CLS visual/patch attention on 224 without nose ROI | Already documented as NoseID-unrelated; keep as archived research |
| `TinyViTBackbone` is a disabled alias, not TinyViT | Already disabled with clear RuntimeError |
| `MagFaceNoseHead` is a disabled alias | Already disabled with clear RuntimeError |
| `MiewIDNoseExtractor` is a disabled alias | Already disabled with clear RuntimeError |
| `Dinov2WithUncertainty` has uncertainty disabled ("no strict calibrated checkpoint-loading contract") | The class is importable but `evidential_checkpoint` always raises RuntimeError |

## Reusable Interfaces for Phase 2+

| Interface | Can be extended for |
|---|---|
| `AbstractEvidencer` | face_id_v1, nose_id_v1 evidence channels |
| `EvidenceObservation` | availability, embedding, details — already supports optional channels |
| `NoseIDSample` (dataclass) | unified canid manifest with optional nose/keypoint fields |
| `IdentityRegistry` (UUIDv5) | all datasets share this namespace |
| `SpeciesFilteredIndex._score_template()` | add face/nose channels as new Channel entries |
| `compute_cosine_score_matrix` + `evaluate_multi_template_closed_set` | A/N/F ablation (already used by NoseID evaluator) |
| `NoseIDConfig` / `NoseIDTrainConfig` | frozen config contract, can be reused for FaceID config |

## Tests

- Full suite: 876 passed, 27 skipped, 0 failures (bacdb5c)
- Public runtime contracts: 44 passed (import surface + CVI API)
- Evidence lazy imports: torch/transformers not loaded by `import cvi`
- NoseID-v1 focused: 27 passed (alignment, frequency, model, losses, sampler, checkpoint, protocol, padding, signed_gem, augmentation, gradient cache, training smoke)

## Blockers

1. **NoseID data**: no oracle annotation manifest exists → blocks Step 06
2. **Face detector**: no face bbox model admitted → blocks Step 05 face-reid training
3. **Dog detector integration**: DogDetector exists but not wired to CVI pipeline → blocks Step 03
4. **Quality fusion**: research-only, not connected to gallery search → blocks Step 07
