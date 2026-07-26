# Consolidated Findings — Phase 0

> Evaluator-specific findings (CVI-EVAL-*) were created during G1 review.
> Evaluator fixes are recorded as FIXED; remaining P0 model-level findings below.

## G1 Evaluator Findings

| ID | Title | Severity | Status | File |
|----|-------|----------|--------|------|
| CVI-EVAL-001 | TAR@FAR selection uses `valid[-1]` from ascending linspace → always TAR=0 | P0 | FIXED | evaluation/verification.py |
| CVI-EVAL-002 | Sequential train/cal/test split (file-order-dependent) | P0 | FIXED | evaluate_multichannel.py → split files + validation |
| CVI-EVAL-003 | Raw similarity treated as calibrated probability | P0 | FIXED | evaluation/calibration.py |
| CVI-EVAL-004 | Hardcoded pinned commit | P0 | FIXED | evaluate_multichannel.py → dynamic git metadata |
| CVI-EVAL-005 | No open-set evaluation protocol | P1 | FIXED | evaluation/open_set.py |
| CVI-EVAL-006 | No retrieval mINP metric | P1 | FIXED | evaluation/retrieval.py |
| CVI-EVAL-007 | No self-match exclusion for retrieval | P1 | FIXED | evaluation/retrieval.py |
| CVI-EVAL-008 | No confidence intervals on FAR/TAR | P1 | FIXED | evaluate_multichannel.py |
| CVI-EVAL-009 | Circular regression test oracles | P1 | FIXED | tests/test_evaluation_metrics.py |
| CVI-EVAL-010 | Retrieval rank_ks parameter ignored | P1 | FIXED | evaluation/retrieval.py |
| CVI-EVAL-011 | Typed exceptions replaced with error-dict returns | P1 | FIXED | evaluation/verification.py |
| CVI-EVAL-012 | Rounding inside core metric functions | P1 | FIXED | evaluation/verification.py |
| CVI-EVAL-013 | select_threshold_at_far returns sentinel -1 values | P1 | OPEN | evaluation/verification.py |
| CVI-EVAL-014 | OSCR not implemented in open_set.py | P1 | OPEN | evaluation/open_set.py |
| CVI-EVAL-015 | CLI report schema v2 needs formal validation file | P2 | FIXED | evaluate_multichannel.py |
| CVI-EVAL-024 | Report Schema v2 with formal JSON Schema validation | P2 | IMPLEMENTED | schemas/cvi.evaluation.report.v2.schema.json |
| CVI-EVAL-025 | CLI smoke test framework for evaluation pipeline | P2 | IMPLEMENTED | tests/test_evaluate_multichannel_cli.py |
| CVI-EVAL-026 | Full provenance tracking with file/config SHA256 hashes | P2 | IMPLEMENTED | tools/evaluate_multichannel.py |
| CVI-EVAL-027 | Split leakage enforced as fatal error (not warning) | P1 | IMPLEMENTED | tools/evaluate_multichannel.py |
| CVI-EVAL-028 | Bootstrap confidence intervals for FAR/TAR and retrieval | P2 | IMPLEMENTED | tools/evaluate_multichannel.py |
| CVI-EVAL-029 | Metric invariant enforcement in retrieval (mINP/invariance guards) | P1 | IMPLEMENTED | evaluation/retrieval.py |
| CVI-EVAL-030 | Open-set DIR with detection vs identification separation | P1 | IMPLEMENTED | evaluation/open_set.py |

## P0 (Must Fix Before Any Experiment)

| ID | Title | Severity | Status | File |
|----|-------|----------|--------|------|
| CVI-P0-001 | MiewID runtime input 160×160 vs ONNX 440×440 | P0 | FIXED | evidence/wildlife_reid.py |
| CVI-P0-002 | MiewID ONNX uses AvgPool instead of official GeM | P0 | FIXED | evidence/wildlife_reid.py |
| CVI-P0-003 | MiewID misclassified as "nose" — it's wildlife re-ID | P0 | FIXED | evidence/wildlife_reid.py |
| CVI-P0-004 | TinyViTBackbone is random 3-layer CNN named "ViT" | P0 | FIXED | backbones/__init__.py |
| CVI-P0-005 | DNPMask UNet is fully random — destroys signal | P0 | FIXED | evidence/nose_print.py |
| CVI-P0-006 | LandmarkEvidencer uses random CNN+GNN | P0 | FIXED | evidence/landmark_graph.py |
| CVI-P0-007 | SuperAnimal ONNX is 9 KB dummy (needs re-export) | P0 | OPEN | tools/download_models.py:219-220 |
| CVI-P0-009 | pickle serialization in calibrator | P0 | OPEN | fusion/calibrator.py:35 |
| CVI-P0-010 | Fake uncertainty (epistemic=0.05 hardcoded) | P0 | FIXED | pipeline/enroll.py |
| CVI-P0-011 | Video API not supported despite README claims | P0 | OPEN | api.py:108-119 |
| CVI-P0-012 | MiewID-msv3 has no license (production risk) | P0 | OPEN | HF model card |
| CVI-P0-013 | 3 duplicate FAISS index implementations | P0 | OPEN | identity_index.py, gpu_index.py, index/hierarchical.py |
| CVI-P0-015 | trainer validation used inference embedding as CE logits | P0 | FIXED | trainer.py |

## P1 (Should Fix Before Multi-Channel)

| ID | Title | File | Evidence |
|----|-------|------|----------|
| CVI-P1-001 | 23 broad except BaseException sites | 12 files in src/cvi/ + 4 tools | Line numbers documented in 00_inventory.md |
| CVI-P1-002 | strict=False in 3 locations (model weight loading) | tools/download_models.py:182, build_calibration_pairs.py:74, export_onnx.py:28 | Masks architectural mismatches |
| CVI-P1-003 | _weights private attribute accessed directly | fusion/fuser.py:12-48 | No public getter |
| CVI-P1-004 | _calibrators pickle serialization | fusion/calibrator.py:35,41 | Security risk |
| CVI-P1-005 | identity/gpu_index.py is byte-for-byte duplicate of gpu_index.py | identity/gpu_index.py | Maintenance burden |
| CVI-P1-006 | identity/__init__.py.bak is tracked in git | identity/__init__.py.bak | Stale artifact |
| CVI-P1-007 | utils/model_paths.py vs model_paths.py — two path modules | utils/model_paths.py, model_paths.py | Different path conventions |
| CVI-P1-008 | No breed classifier ONNX exists | classifier/breed.py | Class exists but no model downloaded |
| CVI-P1-009 | crop_export decoder is untestable, silently swallows errors | crop_export.py:554 | Broad except |

## P2 (Should Fix for Clean Release)

| ID | Title | File |
|----|-------|------|
| CVI-P2-001 | ARCHITECTURE.md describes planned features as existing | docs/ARCHITECTURE.md |
| CVI-P2-002 | README.md claims 77 tests (actual: 605) | README.md:100 |
| CVI-P2-003 | No LICENSE file in repo root | N/A |
| CVI-P2-004 | configs/ have only .example.json files, no actual configs | configs/ |
| CVI-P2-005 | CI workflow missing (no .github/workflows/) | N/A |
| CVI-P2-006 | No uv.lock committed | N/A |
