# Repository Inventory — Phase 0

> Historical Phase 0 snapshot at `0ba3b1b`. Current state is recorded in
> `audit/STATUS.md`, `audit/findings.json`, and later receipts.

## Overview

- **Repository:** `https://github.com/GongJae00/canine_video_identity`
- **Pinned commit:** `0ba3b1bef4ad6bd18ee516260cf938e9e43ca659`
- **Environment:** WSL2, Python 3.12.13, uv 0.11.25, RTX 5080 16 GB
- **Total Python:** 115 modules (src) + 65 tools + 66 tests = 246 Python files
- **Total lines:** 47,261 (src + tools) + 22,519 (tests) = 69,780
- **Test methods:** 605 test methods across 66 test files
- **Config files:** 48 (JSON examples + README)

## Directory Structure

```
src/cvi/            115 .py files (core package)
├── __init__.py         Public API re-export (flat + package)
├── api.py              CVI entry point (new public API)
├── backbones/          3 backbone classes (DINOv2, ConvNeXt, TinyViT)
├── heads/              5 head classes (ArcFace, MagFace, Evidential, FullModel)
├── evidence/           5 files (base, appearance, nose_print, landmark_graph, quality)
├── fusion/             4 files (fuser, calibrator, open_set, temporal)
├── index/              2 files (base, hierarchical)
├── pipeline/           2 files (enroll, search)
├── deployment/         2 files (cpu, cuda)
├── classifier/         3 files (species, breed, color)
├── identity/           2 files + 1 stale .bak (gpu_index, registry)
├── train/              3 files (config, dataset, augment)
├── utils/              2 files (metrics, model_paths legacy)
├── 55 flat modules     (trainer.py, detection.py, search_engine.py, etc.)
```

## CLI Entry Points (65 tools)

All in `tools/` directory. Key ones:
- `download_datasets.py` — dataset download
- `download_models.py` — model download
- `train_embedding_model.py` — backbone training
- `evaluate_multichannel.py` — multi-channel evaluation
- `export_onnx.py` — ONNX export
- `search_identity.py` — identity search CLI
- `probe_video.py` — video probing
- `build_identity_registry.py` — registry construction

## Test Layout (66 files, 605 tests)

All in `tests/` directory. Key test files:
- `test_evidence_extractor.py` — 20 tests (core evidence tests)
- `test_integration_pipeline.py` — pipeline integration tests
- `test_identity_index.py` — index tests
- `test_trainer.py` — training tests
- `test_evaluation.py` — evaluation tests

## Duplicate Implementations

| Concept | Count | Locations |
|---------|-------|-----------|
| FAISS index | 3 | identity_index.py, gpu_index.py, identity/gpu_index.py |
| GPU index | 2 (identical) | gpu_index.py, identity/gpu_index.py |
| TinyViTBackbone | 2 (identical) | backbones/__init__.py, evidence/nose_print.py |
| Evidence extraction | 2 stacks | evidence_extractor.py (flat), evidence/ (new package) |
| Model paths | 2 | model_paths.py (canonical), utils/model_paths.py (legacy) |

## Key Anti-Patterns Found

### broad except BaseException (23 sites)
```
src/cvi/pdq_native.py:651, protected_publication.py:75,
public_dataset_extraction.py:284, acquisition.py:678,
protected_public_split.py:515, public_image_content_audit.py:466,712,
embedding_producer.py:1001,1060, protected_io.py:104,
crop_export.py:554, public_canine_phash_audit.py:567,1160,1261,
embedding_production_runner.py:771,1322, control_transform.py:1127,
pdq_source_intake.py:619
tools/: execute_visual_control_transforms.py:94,
produce_embedding_cache.py:115, evaluate_multichannel.py:147,290,
export_oracle_crops.py:83
```

### strict=False (3 sites)
```
tools/download_models.py:182 — MiewID backbone loading
tools/build_calibration_pairs.py:74
tools/export_onnx.py:28
```

### pickle (2 files)
```
src/cvi/fusion/calibrator.py:35,41 — calibrator serialization
src/cvi/post_search.py:38,45 — post-search calibrator serialization
```

### TODO/FIXME/NotImplemented (1 site)
```
src/cvi/index/hierarchical.py:120 — NotImplementedError for remove()
```

### Random/Temporal seed usage
```
src/cvi/trainer.py:156 — manual_seed per identity (g.manual_seed)
src/cvi/multi_head.py:236 — torch.randn for dummy input
```

## Model/Data Paths

### Checkpoints exists
- DINOv2-Small torch.hub: 84.2 MB
- ConvNeXt-Base HF Hub: cached
- MiewID-msv3 safetensors: 196.3 MB
- MiewID ONNX: 2.1 MB + 196.1 MB data
- DINOv2 ONNX: 1.0 MB + 86.3 MB data
- MobileNetV4 ONNX: 173 KB + 4.8 MB data
- SuperAnimal PT: 112.6 MB
- SuperAnimal ONNX: 9.2 KB (BROKEN — dummy wrapper)

### Datasets
- DogFaceNet: 8,363 images, 1,393 identities
- MPDD: 1,657 images
- SiBeTan: 1,755 images, 59 identities
- YT-BB-dog v1+v2+v3: 27,036 each (81,108 total)

### Registries
- identity_registry.db: 1.7 MB
- binding.json: 5.3 MB

## Summary of Severity Distribution

| Severity | Count | Key Items |
|----------|-------|-----------|
| P0 | 15 | Wrong input size, random models, misleading naming, broken evaluator |
| P1 | TBD | License gaps, legacy cruft, missing tests |
| P2 | TBD | Documentation, cleanup |
