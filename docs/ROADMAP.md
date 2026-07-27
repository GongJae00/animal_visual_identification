# Roadmap

This roadmap is a sequence of admission gates, not a delivery schedule or a
performance promise. A later stage does not become current scope until its
implementation, artifacts, tests, and evaluation evidence are public and
reproducible.

## 1. Reproducible Appearance Baseline

- Replace the unpinned Torch Hub example path with a fully source-bound local
  artifact and preprocessing contract for routine evaluation.
- Publish an identity-disjoint protocol with duplicate, sequence, and crop
  leakage controls.
- Report retrieval and verification metrics with protocol definitions and
  uncertainty, without promoting them to an open-set decision.

## 2. Data And Model Governance

- Resolve code, weight, and dataset licenses for every candidate artifact.
- Document acquisition boundaries and immutable provenance without bundling
  restricted files.
- Define privacy, retention, deletion, and access requirements for real animal
  and owner data before operational collection.

## 3. Calibrated Decision Boundary

- Separate development, calibration, and final evaluation identities.
- Freeze a threshold-selection procedure and evaluate unknown-dog exposure,
  accepted wrong-identity risk, and review coverage.
- Connect open-set behavior to `CVI` only after the boundary is versioned and
  artifact-bound.

## 4. Evidence Admission

- Evaluate each proposed channel against the frozen appearance baseline.
- Require exact model/export/preprocessing parity and explicit missing-evidence
  behavior.
- Admit a channel only if a leakage-controlled ablation shows useful identity
  information, calibrated score compatibility, and justified compute cost.

## 5. Video Integration

- Define a pinned detector and crop contract before connecting detection.
- Add sequence-aware tracking, frame-quality selection, and temporal aggregation
  with tests for ordering, dropped frames, occlusion, and identity switches.
- Evaluate track-level behavior independently from crop-level behavior.

## 6. Deployment Hardening

- Connect strict CPU and guarded CUDA inference to the same canonical gallery
  and decision contracts.
- Define read-only serving, gallery update, migration, backup, rollback, and
  multi-process behavior.
- Add resource limits, observability, dependency review, privacy controls, and
  threat-model testing before presenting CVI as a service.

Current boundaries remain authoritative in
[Architecture](ARCHITECTURE.md) and [Known Limitations](KNOWN_LIMITATIONS.md).
