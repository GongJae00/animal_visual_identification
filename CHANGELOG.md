# Version History

Current behavior is defined by `docs/ARCHITECTURE.md`,
`docs/CONFIGURATION.md`, and `docs/KNOWN_LIMITATIONS.md`. This file records why
versioned changes occurred, where their evidence lives, and which persisted
contracts remain readable.

## Unreleased

### Part 1 - Parser Policy And Full128 Materialization

#### Parser policy v6 and Full128 route policy v3 - 2026-08-13

- Reason: the v5 parser requested dog and cat candidates and the generic route
  policy rejected useful single-dog observations when any additional candidate
  was present. Auxiliary datasets also lacked a deterministic single-dog
  selection rule.
- Change: parser policy v6 requests dogs only. AP-10K keeps authoritative global
  bbox association; DogFLW and Oxford select the largest valid dog by foreground
  pixels; identity-bearing datasets require exactly one post-suppression dog.
  Full128 route-plan v3 binds those rules and records candidate counts, selection,
  authority, and terminal reasons in lineage.
- Evidence: Oxford dog evaluation report digest
  `9a7cf4db1e32dd247797f06c29bb53f2908b2ba1c520f58d1fb402fb5faa97df`;
  AP-10K panel digest
  `360a07e976418b72bf2c9092a975616cbf514a635cf5aac7a6dfcb8ebf70cab6`;
  runtime manifest digest
  `21ca43fe71cbf7bcdd0f13bb39dea4045b029e4b8beebbd2c58de800a299363a`;
  Full128 route-plan digest
  `4bed7c8d2689307bbd2018d9400d79c352c018ffd298e24ca725eceb0fc522f8`;
  assembly digest
  `714c429f97888d7072e8951a7179384f778adcd2b3664c5809408a66b0b8f162`.
  These artifacts are external and research-only.
- Outcome: 36,195 of 49,253 observations materialized, 518 more than v5, with
  no v5-success-to-v6-terminal regressions. The comparison mixes parser policy,
  runtime source closure, and route-policy versions; it is not a biometric result.
- Compatibility: parser policies v4/v5, parser runtime manifest/bundle v1, and
  Full128 route-plan v2 remain accepted
  because completed external artifacts bind those schemas. They are compatibility
  readers, not current defaults.

#### Parser batch protocol - 2026-08-13

- Reason: batch sizes 8 and 16 offered modest throughput gains but had not been
  checked for output equivalence.
- Change: a fixed 512-image benchmark records one warm-up and three measured runs,
  parser/end-to-end timing, peak CUDA memory, exact and semantic fingerprints, and
  terminal-decision mismatches.
- Evidence: benchmark digest
  `3b75f00eff209587294e4348c84f9b4723634a536d39e8ec156ebca87f31851c`.
- Outcome: batch 4 remains canonical because batches 8 and 16 changed semantic
  predictions and terminal decisions despite higher throughput.

### Part 2 - Persisted Contracts

- Gallery v4 and v5 remain readable; gallery v3 requires explicit migration.
- Full128 route-plan v2 remains readable beside current v3.
- Parser policies v4/v5 and runtime manifest/bundle v1 remain readable beside
  current dog-only policy v6 and runtime v2.
- Successor private evaluation v1 remains readable; new reports use compact v2
  traces that omit repeated Q/K vectors and bind them through the embedding cache.

### Part 3 - Research Decisions

- Full128 governance-v2 selected B5-SPATIAL by DEV point estimates in three seeds,
  but its precommitted paired interval gate returned `NO_GO`. B3 remains the stable
  research candidate. Evidence digests are listed in
  `docs/RESEARCH_PROGRESS.md`.

## 0.4.0 - 2026-08-03

### Breaking

- Replaced the `cvi.CVI` import surface with
  `canine_identity.IdentityEngine`; `Match` remains the public result type.
- Reorganized the former `src/cvi` tree into functional top-level packages for
  data, identity governance, localization, identity methods, learning, fusion,
  retrieval, evaluation, runtime, and operations.
- Moved source-checkout commands from `tools/` to `workflows/` and removed
  unsupported compatibility exports, deployment facades, and duplicate runtime
  scaffolding.

### Architecture And Performance

- Added recursive logical source provenance v2 while retaining narrow readers
  for persisted Face and PDQ v1 evidence.
- Added dependency-boundary tests that keep evaluation and operations out of the
  public runtime and algorithm packages.
- Removed duplicate A4 frozen-backbone inference, decode-only label collection,
  eager gallery metadata copies, and an unnecessary retrieval score-matrix copy.
- Added a single-distribution build containing all functional packages and the
  versioned evaluation schemas.

### Validation

- Reproduced the full 15-epoch A4 training trajectory, selected epoch, and DEV
  metrics after the migration.
- Reproduced all eight DogFaceNet, MPDD, and SiBeTan external protocol results;
  the new report differs only by the newly generated checkpoint hash.
- Verified the complete test suite, wheel contents, isolated package import,
  documentation links, lockfile, and source diff.

### Research Infrastructure

- Added strict AP-10K and DogFLW adapters, content-bound localization caches,
  instance-aware ROI manifests, and research-only face ReID tooling.
- Added a content-addressed model catalog with logical role aliases; filesystem
  paths no longer imply research or deployment admission.
- Added a separate provisional generated-identity namespace with explicit
  merge, supersede, and rejection states.

### Security And Compliance

- Bound generic ONNX model and preprocessing manifests to gallery contracts.
- Made ROI crop paths, bytes, dimensions, identity propagation, and landmark
  geometry fail closed.
- Kept Ultralytics outside package extras; localization adapters require a
  separately managed research environment.
- Removed host-specific experiment scripts, generated dataset visualizations,
  and stale internal architecture audit material from the public surface.

## 0.3.0 - 2026-07-27

### Breaking

- Removed legacy index implementations from the supported public runtime.
- Made runtime configuration and `index_dir` explicit instead of relying on
  implicit storage or compatibility defaults.
- Advanced persisted galleries to gallery schema v4; incompatible galleries
  must be migrated explicitly rather than accepted silently.
- Added the versioned evaluation schema as a packaged `cvi.schemas` resource.

### Security And Compliance

- Disabled model exporters whose artifacts or licenses were not admitted to a
  supported runtime contract.
- Removed direct Torch Hub execution; public and research evaluators now require
  caller-supplied or receipt-bound local model artifacts.
- Removed Ultralytics from the published training extra. Detection remains
  outside the canonical product, and user-supplied detectors are separately
  licensed.
- Changed manual or license-gated dataset selectors to fail with acquisition
  guidance instead of reporting an empty directory as a successful download.
- Disabled the unpinned DogFaceNet Hugging Face materializer. All dataset
  selectors now require manual acquisition and fail before network or framework
  imports; the default `all` selector is an explicit successful no-op.

### Claims

- This release makes no biometric accuracy, identification performance, or
  deployment-readiness claim. Unit and synthetic tests validate software
  contracts, not biometric performance.
- Removed unverified dataset cardinality and operational-safety wording from
  source documentation.
