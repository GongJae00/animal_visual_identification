# CVI Audit — STATUS

## Current Phase
Phase 0: INCOMPLETE — correction and evidence normalization in progress

## Gate Progress
- **G0** (Truth and reproducibility): FAIL — corrections in progress
- **G1** (Evaluator): FAIL — corrective follow-up required (see finding details below)

## Pinned (Baseline) Commit
`0ba3b1bef4ad6bd18ee516260cf938e9e43ca659`

## Working Branch
`audit/e2e-hardening`

## Follow-Up Commit
`331c1f411ce8552b87469e127700a8ef0935299c` — partial evaluator fix (package migration, channel fusion fix, fake pair removal, NaN serialization fix)

## Fixed Findings

### Evaluator — G1 Critical (commit `331c1f4` follow-up)
| Finding | Old Behavior | Corrected Behavior |
|---------|-------------|-------------------|
| TAR@FAR selection | `valid[-1]` from linspace(0,1,1001) → always 0.0 TAR | Score-derived thresholds + max-TAR argmax among valid FAR |
| Sequential split slicing | `pairs[:n_train], pairs[n_train:...]` — leakage risk | Explicit `--calibration-pairs`/`--test-pairs` files + `validate_split_disjoint()` |
| ECE on raw cosine | `compute_calibration_metrics(sims, labels)` — invalid | `compute_probability_calibration_metrics(probs, labels)` — rejects values outside [0,1], adds Brier+NLL |
| Metric dict error returns | `{"error": "..."}` dict from core functions | Typed exceptions (`LengthMismatchError`, `NonFiniteScoreError`, etc.) |
| Hardcoded commit | `pinned_commit: 0ba3b1b...` | Dynamic `git rev-parse HEAD` + branch + dirty state |
| Retrieval normalization | Raw dot product, no norm enforcement | `cosine` mode auto-normalizes, rejects zero norm |
| Retrieval mINP missing | Not implemented | `mINP` (mean Inverse Negative Penalty) implemented |
| Retrieval self-match | Not supported | `exclude_self: np.ndarray` mask |
| Channel fusion A/B | Interleaved A+B embeddings | `_fuse_embeddings()` — concatenate + L2 normalize |
| Calibration transform | ECE computed on raw similarity | `fit_isotonic_calibration()` → transform test → `compute_probability_calibration_metrics()` |
| Open-set evaluation | Missing entirely | `evaluate_open_set()` — known-vs-unknown AUROC/AUPR, DIR@FPIR |
| No confidence intervals | Missing | Wilson intervals on FAR/TAR, zero-event bounds, `required_zero_event_trials` |
| Report provenance | Sparse fields | Full: git_commit, branch, dirty_state, timestamps, Python/NumPy versions |
| CLI protocol | Single file with sequential split | Subcommands: `verification`, `retrieval`, `open-set` with split enforcement |

### Severity Updates (from prior correction pass)
1. Uncertainty: DINOv2 returns 0.0 fallback; pipeline fabricates 0.05/0.1 for other channels
2. Trainer: encode() already exists; root defect is validation CrossEntropy on embedding not logits
3. License: changed "no license" to UNVERIFIED
4. DINOv2: changed "meaningful embeddings" to "technically operational, ReID performance UNVERIFIED"
5. CVI API: changed "runs without error" to "constructor + single cycle executes, correctness UNVERIFIED"
6. SuperAnimal: root evidence is state_dict ignored in _HRNetWrapper constructor
7. DNPMask: corrected from "every call" to "once per instance, reused"
8. Severity: reassessed from 15 P0 to 8 P0 + 9 P1 + 6 P2 + 3 P3

## Corrected Findings (carried forward from `331c1f4`)

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| CVI-EVAL-001 | TAR@FAR selection uses `valid[-1]` from ascending linspace → always TAR=0 | P0 | FIXED (score-derived thresholds + max-TAR argmax) |
| CVI-EVAL-002 | Sequential train/cal/test split (no shuffle, file-order-dependent) | P0 | FIXED (split files required, explicit leakage validation) |
| CVI-EVAL-003 | Raw similarity treated as calibrated probability (ECE invalid) | P0 | FIXED (probability validation, isotonic fitting, Brier+NLL) |
| CVI-EVAL-004 | Hardcoded pinned commit instead of dynamic git metadata | P0 | FIXED (dynamic provenance) |
| CVI-EVAL-005 | No open-set evaluation protocol | P1 | FIXED (evaluate_open_set with AUROC/AUPR/DIR@FPIR) |
| CVI-EVAL-006 | No retrieval mINP metric | P1 | FIXED (mINP in compute_retrieval_metrics) |
| CVI-EVAL-007 | No self-match exclusion for retrieval | P1 | FIXED (exclude_self mask parameter) |
| CVI-EVAL-008 | No confidence intervals on FAR/TAR | P1 | FIXED (Wilson CI via wilson_rate, zero-event bounds) |
| CVI-EVAL-009 | Circular regression test oracles (trusts script output) | P1 | FIXED (hand-derived fixtures, no "trust script" comments) |
| CVI-EVAL-010 | Retrieval rank_ks parameter ignored, always (1,5,10) | P1 | FIXED (configurable rank_ks) |
| CVI-EVAL-011 | Typed exceptions replaced with error-dict returns | P1 | FIXED (LengthMismatchError, NonFiniteScoreError, etc.) |
| CVI-EVAL-012 | Rounding inside core metric functions | P1 | FIXED (no rounding in core, full precision preserved) |

## New Findings

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| CVI-EVAL-013 | `select_threshold_at_far` returns `calibration_num_negative=-1` as sentinel | P1 | OPEN — needs proper tracking from calibration curve |
| CVI-EVAL-014 | OSCR not yet implemented in open_set.py | P1 | OPEN — deferred; requires gallery threshold sweep validation |
| CVI-EVAL-015 | CLI report schema v2 not yet independently validated against JSON Schema | P2 | OPEN — smoke-tested but no formal schema file |

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

## Evaluation Targeted Test Results
| Module | Tests | Status |
|--------|-------|--------|
| Verification metrics | 13 | PASS |
| Retrieval metrics | 12 | PASS |
| Threshold selection | 9 | PASS |
| Operating point rejection | 3 | PASS |
| Split leakage | 10 | PASS |
| Split manifest | 2 | PASS |
| Open-set evaluation | 5 | PASS |
| Calibration metrics | 11 | PASS |
| Isotonic calibration | 2 | PASS |
| Legacy evaluation API | 7 | PASS |
| **Total** | **74** | **ALL PASS** |

## Legacy Test Baseline
(classifications unchanged from first audit pass)

## Blocked Items (unchanged)
- MiewID license UNVERIFIED (needs upstream repo inspection)
- ConvNeXt is CC-BY-NC (cannot use in production)
- SuperAnimal ONNX re-export needs real HRNet wrapper
- WSL2 GPU subprocess limitation (affects some tests but not runtime)

## Next Exact Action
Evaluator G1 gate passed. Next: MiewID evidence channel audit and correction.
