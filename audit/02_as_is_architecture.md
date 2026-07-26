# As-Is Architecture (Phase 0)

> Historical pre-hardening architecture snapshot. It is not the current runtime
> contract; see `docs/ARCHITECTURE.md` and `audit/08_model_contract_hardening.md`.

## Actual Runtime Data Flow

```
PIL Image
  │
  ├─ CVI._build_evidence()
  │    ├─ MiewIDNoseExtractor (type="miewid")
  │    │    └─ ONNX Runtime — input 160×160 (WRONG: should be 440×440)
  │    │    └─ AvgPool backbone (WRONG: should be GeM)
  │    │    └─ Output: 2152-d L2-normalized (correct dim, wrong features)
  │    │
  │    ├─ LandmarkEvidencer (type="landmark")
  │    │    └─ HRNetHeatmap — 5-layer random CNN (NOT real HRNet)
  │    │    └─ LandmarkGraphEmbedder — EdgeConv GNN (random)
  │    │    └─ Output: 256-d L2-normalized (random noise)
  │    │
  │    └─ Dinov2WithUncertainty (type="dinov2" or "appearance")
  │         └─ ONNX Runtime — dinov2-small 224×224
  │         └─ Output: 384-d L2-normalized (MEANINGFUL)
  │         └─ Also returns hardcoded epistemic=0.05, aleatoric=0.1
  │
  ├─ MultiEvidencePipeline (pipeline/enroll.py)
  │    └─ extract_all(image) → dict of channel outputs
  │
  ├─ SpeciesFilteredIndex (index/hierarchical.py)
  │    └─ FAISS IndexFlatIP — exact search (no IVF)
  │    └─ Breed prefilter — if breed is provided, shards by breed
  │    └─ Save/load: separate .faiss file per breed
  │
  ├─ LearnedWeightFuser (fusion/fuser.py)
  │    └─ Weighted sum with optional quality/uncertainty modulation
  │    └─ Weights: self._weights (private), no public getter
  │    └─ update_weights: logistic regression (requires training)
  │
  ├─ EvidentialOpenSet (fusion/open_set.py)
  │    └─ Fixed thresholds: min_sim=0.4, epi_thresh=0.3, ratio_thresh=0.15
  │    └─ Uncertainty values are always 0.05 → never triggers epi reject
  │
  └─ IdentitySearchPipeline (pipeline/search.py)
       └─ enroll: extract → index.add
       └─ search: extract → index.search → open_set.reject → Match[]
```

## Legacy Flat Stack (still active)

```
search_engine.py:
  SearchEngine — wraps EvidenceExtractorRegistry + IdentityIndex
  FeatureExtractor — legacy single-ONNX wrapper

evidence_extractor.py:
  EvidenceExtractor + OnnxExtractor + DogFaceNetExtractor
  ConvNeXtExtractor + SuperAnimalExtractor + PetReIDExtractor
  EvidenceExtractorRegistry

identity_index.py:
  IdentityIndex — visual/texture/structural slice fusion + FAISS
  Has EMBEDDING_DIM, TEXTURE_SLICE, STRUCTURAL_SLICE constants

gpu_index.py:
  GpuIdentityIndex — FAISS GPU wrapper (identical to identity/gpu_index.py)

identity/__init__.py.bak:
  Stale backup file tracked in git
```

## What Actually Works End-to-End

1. `uv sync` — Python environment setup
2. Dataset downloads — 4 datasets present
3. Model downloads — MiewID, DINOv2, ConvNeXt ONNX models present
4. `from cvi import CVI; cvi = CVI(); cvi.enroll(img, "test"); cvi.search(img)` — runs without error
5. DINOv2 appearance channel — produces semantically meaningful embeddings
6. FAISS index read/write — round-trip works
7. Core tests (20 tests in test_evidence_extractor) — pass

## What Does NOT Work (or Produces Garbage)

1. MiewID/Nose channel — wrong input size (160 vs 440), AvgPool instead of GeM
2. Landmark channel — random CNN + random GNN (no weights)
3. SuperAnimal ONNX — 9 KB dummy wrapper
4. Open-set uncertainty — hardcoded 0.05/0.1 (not computed)
5. Evaluator rank1 — always 0 (missing key)
6. Video pipeline — not connected (Image-only API)
7. Dog detection — not wired into any pipeline
8. Temporal aggregation — not connected

## Deployment Backend Reality

| Class | Runner | Index | Actual GPU Use | Status |
|-------|--------|-------|----------------|--------|
| CVIDeploymentCPU | None (class only) | FAISS CPU IndexFlatIP | No | Class exists, unused by any tool |
| CVIDeploymentCUDA | None (class only) | FAISS CPU IndexFlatIP | No (uses CPU FAISS via SpeciesFilteredIndex) | Misleading name — no GPU use |
| GpuIdentityIndex | gpu_index.py | FAISS GPU GpuIndexFlatIP | Yes | Isolated class, not connected to CVI API |
| CVI (api.py) | api.py | SpeciesFilteredIndex → FAISS CPU | No | Main entry point, CPU only |

## Training Infrastructure Reality

| Component | Status | Notes |
|-----------|--------|-------|
| trainer.ArcFaceModel | EXISTS | forward() always through head, no encode() for inference |
| backbones.Dinov2Backbone | EXISTS | torch.hub load, 384-d, L2 norm |
| backbones.ConvNeXtBackbone | EXISTS | transformers load, 1024→embedding_dim project |
| backbones.TinyViTBackbone | EXISTS | 3-layer random CNN — named "TinyViT" |
| heads.ArcFaceHead | EXISTS | Standard ArcFace with cos(theta+m) |
| heads.MagFaceHead | EXISTS | MagFace with norm penalty |
| heads.EvidentialHead | EXISTS | Random weights — never trained |
| backbones._BACKBONE_REGISTRY | EXISTS | Factory dict for all backbones |
| train.config.TrainConfig | EXISTS | Configuration dataclass |
| train.dataset | EXISTS | Dataset loaders |
| train.augment | EXISTS | CutMix + environmental aug |

## Key Architectural Decisions (Undeclared)

1. No canonical index implementation — 3 competing implementations
2. No canonical evidence extraction — new package + legacy flat module coexist
3. No canonical evaluator — evaluate_multichannel.py has broken rank1
4. No canonical deployment — CPU/CUDA classes are disconnected from CVI API
5. Public API (CVI class) uses pipeline/ package, but legacy code uses SearchEngine directly
