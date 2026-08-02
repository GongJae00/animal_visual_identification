# Changelog

## Unreleased

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
