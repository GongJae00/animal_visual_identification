# CVI Audit — STATUS

## Current Phase
**Phase 0: G1 EVALUATION PASS** — all evaluation gate fixes verified; P0 model-level findings remain open

## Gate Progress
- **G0** (Truth and reproducibility): FAIL — corrections in progress
- **G1** (Evaluator): **PASS** — all evaluation fixes verified, 83/83 evaluation tests pass, 12/12 CLI smoke tests pass

## Pinned (Baseline) Commit
`0ba3b1bef4ad6bd18ee516260cf938e9e43ca659`

## Working Branch
`audit/e2e-hardening`

## Follow-Up Commits
1. `331c1f411ce8552b87469e127700a8ef0935299c` — first evaluator fix (package migration, channel fusion fix, fake pair removal, NaN serialization fix)
2. `49a9e7ca6131571f0eb383c684b5315fa4a41a30` — third evaluator fix (mINP invariant, self-match removal, open-set DIR protocol, calibration error propagation, split enforcement, provenance, schema, CLI smoke tests)

## G1 Findings Status (All G1 Items Complete)

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| CVI-EVAL-013 | select_threshold_at_far returns sentinel -1 counts | P1 | FIXED (exact counts now tracked) |
| CVI-EVAL-015 | CLI report schema v2 needs formal validation file | P2 | FIXED (schemas/cvi.evaluation.report.v2.schema.json) |
| CVI-EVAL-016 | mINP formula computes rank/n_relevant instead of n_relevant/last_positive_rank | P1 | FIXED |
| CVI-EVAL-017 | Self-match exclusion does not remove candidate from is_positive mask | P1 | FIXED |
| CVI-EVAL-018 | Open-set DIR does not require correct top-1 identity match | P1 | FIXED |
| CVI-EVAL-019 | Label validation casts to int64 before checking {0,1} | P1 | FIXED |
| CVI-EVAL-020 | Calibration silently catches all exceptions | P2 | FIXED |
| CVI-EVAL-021 | Report provenance missing file/config hashes | P2 | FIXED |
| CVI-EVAL-022 | Split leakage produces warnings not fatal errors | P1 | FIXED |
| CVI-EVAL-023 | Confidence intervals incomplete (no bootstrap) | P2 | PARTIAL (basic bootstrap CI added) |
| CVI-EVAL-014 | OSCR not implemented in open_set.py | P1 | DEFERRED (requires gallery≥2 per identity, post-G1) |

## Remaining Open P0 (unchanged from prior pass)
| ID | Title | Status |
|----|-------|--------|
| CVI-P0-001 | MiewID runtime input 160×160 vs ONNX 440×440 | OPEN |
| CVI-P0-002 | MiewID ONNX uses AvgPool instead of official GeM | OPEN |
| CVI-P0-003 | MiewID misclassified as "nose" — it's wildlife re-ID | OPEN |
| CVI-P0-004 | TinyViTBackbone is random 3-layer CNN named "ViT" | OPEN |
| CVI-P0-005 | DNPMask UNet is fully random — destroys signal | OPEN |
| CVI-P0-006 | LandmarkEvidencer uses random CNN+GNN | OPEN |
| CVI-P0-007 | SuperAnimal ONNX is 9 KB dummy (needs re-export) | OPEN |
| CVI-P0-009 | pickle serialization in calibrator | OPEN |
| CVI-P0-010 | Fake uncertainty (epistemic=0.05 hardcoded) | OPEN |
| CVI-P0-011 | Video API not supported despite README claims | OPEN |
| CVI-P0-012 | MiewID-msv3 has no license (production risk) | OPEN |
| CVI-P0-013 | 3 duplicate FAISS index implementations | OPEN |
| CVI-P0-015 | trainer.py has no encode() method for inference | OPEN |

## Blocked Items (unchanged)
- MiewID license UNVERIFIED (needs upstream repo inspection)
- ConvNeXt is CC-BY-NC (cannot use in production)
- SuperAnimal ONNX re-export needs real HRNet wrapper
- WSL2 GPU subprocess limitation (affects some tests but not runtime)

## Next Exact Action
Commit current changes, then proceed to G2 gate (model correctness review) or address P0 model-level findings.
