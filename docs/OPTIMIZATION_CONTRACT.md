# Optimization Contract

## Purpose

Every algorithmic or systems change is evaluated against a frozen reference.
Optimization is accepted only when identity safety is non-inferior and at least
one resource or performance dimension improves with uncertainty accounted for.

## Evaluation vectors

Let the safety and utility vector be

\[
A =
(\mathrm{FNMR}_{FMR},
\mathrm{FNIR}_{FPIR},
\mathrm{FPIR},
\mathrm{rank}_k,
\mathrm{coverage},
\mathrm{latency}_{decision},
\mathrm{track\ purity})
\]

with fixed operating points and subgroup definitions. Let the resource vector
be

\[
R =
(T_{p50}, T_{p95}, T_{max},
M_{peak}^{GPU}, M_{peak}^{CPU},
E_{dog\text{-}hour},
B_{storage},
C_{decode}, C_{detect}, C_{track}, C_{embed}, C_{search}).
\]

No single weighted score replaces these vectors. Safety-critical metrics are
constraints; cost metrics define the Pareto comparison.

Backend, precision, compiler, or engine candidates first pass the separate
label-blind canonical-cache gate in `docs/NUMERICAL_ADMISSION.md`. That gate
detects numerical drift cheaply but never substitutes for the protected
biometric and resource comparison below.
Production batch changes also pass the separately precommitted
batch-composition gate in `docs/BATCH_INVARIANCE.md`; a batch pass is likewise
non-promoting.

## Promotion rule

For candidate \(b\) and frozen reference \(a\), define signed degradation
\(d_k(b,a)\) so that positive values are worse. Candidate \(b\) is promotable
only if:

\[
\operatorname{UCB}_{1-\alpha}(d_k(b,a)) \leq \epsilon_k
\quad \text{for every protected metric } k,
\]

and at least one predeclared resource or utility metric has a strict improvement
whose confidence interval excludes zero. The non-inferiority margins
\(\epsilon_k\), confidence method, seeds, split, threshold-calibration set, and
hardware state are fixed before the comparison.

If sample size cannot resolve non-inferiority, the result is `INCONCLUSIVE`, not
`EQUIVALENT`.

`REJECT` is reserved for a protected degradation whose lower confidence bound
also exceeds its margin, or for another predeclared hard failure. If the
confidence interval crosses the margin, non-inferiority is unresolved and the
comparison is `INCONCLUSIVE` even when resource savings are clear.

## Operational compute model

For stream duration \(T\), the first-order compute budget is

\[
C_{total} =
C_{decode}(T)
+ n_{det}C_{det}
+ n_{track}C_{track}
+ n_qC_q
+ n_{embed}C_{embed}
+ n_{search}C_{search}
+ C_{aggregation}.
\]

The event rates \(n_*\) are measured, not inferred from nominal FPS. A scheduling
optimization must report both per-call cost and the changed call rate. Kernel
speed alone is not an end-to-end result.

## Peak-memory model

\[
M_{peak} =
M_{weights}
+ M_{engine\ workspace}
+ M_{activations}(B,H,W,P)
+ M_{video\ buffers}
+ M_{track\ state}
+ M_{gallery}
+ M_{allocator\ reserve}.
\]

Measurements must include steady-state and transition peaks, warm-up, dynamic
shape changes, concurrent streams, and allocator-reserved memory. Batch size
increases are not memory improvements.

For \(N\) identities, \(P\) prototypes, embedding dimension \(D\), and storage
width \(w\) bytes, raw gallery storage is:

\[
M_{gallery}=NPDw.
\]

Index overhead is measured separately.

## Storage and I/O model

For bitrate \(b\) in Mbit/s and duration \(t\) in seconds:

\[
S_{bytes}=\frac{10^6bt}{8}.
\]

Raw retention, derived clips, crops, embeddings, and logs are separately
budgeted. Re-encoding must never replace protected acquisition evidence unless
the retention policy explicitly allows it.

## Stage-specific optimization policy

| Stage | Primary low-cost lever | Protected invariant |
|---|---|---|
| Decode | NVDEC, zero-copy, bounded buffers | identical decoded timestamps and declared pixel tolerance |
| Stream monitor | cheap deterministic statistics | degraded streams cannot be certified as healthy |
| Detection | input sizing, compilation, FP16/INT8 | dog-event recall and multiple-dog safety |
| Tracking | sparse detector calls, motion prediction | track purity, fragmentation, decision horizon |
| Region extraction | shared backbone, ROI reuse | region availability and spatial correspondence |
| Quality | cheap features, early rejection | usable evidence recall and subgroup coverage |
| Selection | online submodular/diversity selection | no future access beyond decision horizon |
| Embedding | AMP, TensorRT, distillation, INT8 | low-FMR/FNMR and cross-modal geometry |
| Aggregation | streaming sufficient statistics | prototype semantics and uncertainty |
| Search | exact search first, ANN when justified | candidate recall at frozen gallery scale |
| Open set | calibrated compact scoring | FPIR, FNIR, margin, and review coverage |
| Temporal policy | sequential early stopping | false assignment and maximum decision time |
| Template update | deferred, reversible prototypes | protected references and contamination bound |

## Precision policy

1. Establish a numerically inspected FP32 or mixed-precision training reference.
2. Establish an FP16 deployment reference.
3. Evaluate INT8 PTQ per component with representative RGB, IR, transition,
   coat, quality, and hard-negative calibration samples.
4. Use selective FP16 fallbacks or QAT only where PTQ violates a protected
   metric.
5. Keep normalization, similarity accumulation, calibration, and threshold
   logic in FP16 or FP32 unless separate evidence supports lower precision.
6. INT4/FP4 is not an initial target for the convolutional identity pipeline.

Quantization calibration and identity-score calibration use separate data roles.
Neither may access the final test split.

## Algorithmic efficiency policy

- Begin with the strongest feasible reference, not the smallest convenient
  reference.
- Compare shared-backbone and multi-backbone designs at matched information
  access, resolution, training data, and compute.
- Prefer event-driven execution, reuse, sufficient statistics, and reduced
  redundant inference before compressing the identity representation.
- Distillation is accepted only against the same teacher checkpoint, data,
  augmentation, split, and operating thresholds.
- Pruning is accepted only with measured hardware speedup; parameter sparsity
  alone is not a deployment improvement.
- Approximate search is introduced only when exact search violates a measured
  latency or memory budget.
- Dynamic policies must include a worst-case cap and a fixed-policy comparator.

## Required benchmark receipt

Every optimization result records:

- immutable reference and candidate config hashes;
- code revision and dependency lock;
- dataset/manifest/split hashes;
- calibration and test roles;
- camera count, codec, resolution, FPS, and clip duration;
- device, driver, precision, batch and stream concurrency;
- warm-up and repetition protocol;
- accuracy, safety, resource vectors, confidence intervals, and subgroup table;
- profiler attribution and bottleneck classification;
- keep, reject, or inconclusive decision.
