# Contributor Guide For Automated And Human Changes

This file describes portable repository rules. Treat the implementation and
tests in the current checkout as the source of truth; do not infer capability
from historical plans or filenames.

## Product Boundary

The canonical public runtime is `cvi.CVI`. It performs crop-level closed-set
enrollment and retrieval. Detection, tracking, temporal aggregation, open-set
decisions, and deployment facades may have implementation scaffolding but are
not connected product capabilities.

Do not add performance, deployment, dataset-size, or model-license claims
without reproducible evidence and an attributable source. Unit and synthetic
tests are not biometric validation.

## Source Layout

| Path | Responsibility |
|---|---|
| `src/cvi/api.py` | Strict public `CVI` and `Match` API |
| `src/cvi/evidence/` | Evidence interfaces, extractors, and artifact contracts |
| `src/cvi/fusion/` | Score weighting, calibration utilities, and research aggregators |
| `src/cvi/index/` | Gallery persistence and exact candidate scoring |
| `src/cvi/pipeline/` | Crop enrollment and search orchestration |
| `src/cvi/evaluation/` | Verification, retrieval, calibration, and open-set metrics |
| `src/cvi/deployment/` | Reserved, currently disabled deployment facades |
| `src/cvi/train/`, `src/cvi/trainer.py` | Training configuration and implementation |
| `tools/` | Commands run from a source checkout |
| `configs/` | Protocol/backend examples with independent schemas |
| `tests/` | Behavioral, contract, and regression tests |

Flat modules under `src/cvi/` are active parts of the package. Do not move or
rename them solely for cosmetic restructuring; first audit imports, tools, and
persisted schema compatibility.

## Invariants

1. Enrollment identities are canonical UUIDv5 values, separate from display
   names, source labels, sample tokens, and track identifiers.
2. Evaluation and training partitions must be identity-disjoint where the
   protocol requires it; random frame splitting is not an acceptable shortcut.
3. Required evidence fails closed. Optional evidence must be explicit in
   config v2 and remains auditable in gallery state.
4. Model, preprocessing, gallery, and receipt schemas must remain versioned and
   content-bound. Do not silently accept incompatible legacy artifacts.
5. External datasets, weights, caches, and experiment outputs stay outside Git.
6. CUDA behavior is optional and guarded. Portable CPU behavior must not import
   CUDA-only dependencies at package import time.
7. Never commit secrets, private animal or owner data, credentials, or licensed
   artifacts.

## Environment

Use Linux, Python 3.12, and `uv`. Select one ONNX Runtime lane.

```bash
uv sync --extra cpu --extra data --extra models --extra training --group dev
# Or, for CUDA work, replace --extra cpu with --extra cuda.
```

The `cpu` and `cuda` extras should not be combined. Source-checkout tools are
invoked as `uv run python tools/<tool>.py --help`; they are not installed
console scripts unless packaging explicitly adds such an entry point.

## Change Workflow

1. Read the affected implementation, tests, and persisted schema before
   editing.
2. Make the smallest change that preserves the invariants above.
3. Add or update tests for behavior changes and failure cases.
4. Run focused tests first, then the full suite when dependencies permit.
5. Check documentation links, command paths, and examples against the checkout.
6. Report tests that were not run and any remaining platform or artifact risk.

```bash
uv run pytest tests/test_public_runtime_contracts.py
uv run pytest
git diff --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for public contribution and security
reporting expectations.
