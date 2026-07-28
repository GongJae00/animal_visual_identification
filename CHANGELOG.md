# Changelog

## Unreleased

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
