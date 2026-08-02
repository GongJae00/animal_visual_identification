# Contributor Guide For Automated And Human Changes

Treat current implementation, tests, schemas, and [Architecture](docs/ARCHITECTURE.md) as the source of truth. Do not infer capabilities from historical plans or filenames.

## Product Boundary

The canonical public runtime is `canine_identity.IdentityEngine`. It performs crop-level closed-set enrollment and retrieval. Detection, tracking, temporal aggregation, open-set decisions, report generation, and operations code are not connected product capabilities.

Do not add performance, deployment, dataset-size, or model-license claims without reproducible evidence and an attributable source. Unit and synthetic tests are not biometric validation.

## Invariants

1. Enrollment identities are canonical UUIDv5 values, separate from display names, source labels, sample tokens, and track identifiers.
2. Evaluation and training partitions must be identity-disjoint where the protocol requires it; random frame splitting is not an acceptable shortcut.
3. Required evidence fails closed. Optional evidence is explicit in config v2 and remains auditable in gallery state.
4. Model, preprocessing, gallery, source, and receipt schemas remain versioned and content-bound. Persisted `cvi.*` identifiers are compatibility contracts, not Python package names.
5. External datasets, weights, caches, galleries, and experiment outputs stay outside Git.
6. CUDA behavior is optional and guarded. Portable CPU behavior must not import CUDA-only dependencies at package import time.
7. Never commit secrets, private animal or owner data, credentials, or licensed artifacts.
8. Respect the dependency direction enforced by `tests/test_dependency_boundaries.py`; algorithms must not depend on evaluation or operations.

## Environment And Workflow

Use Linux, Python 3.12, and `uv`. Select either the `cpu` or `cuda` extra, never both.

```bash
uv sync --extra cpu --extra data --extra models --extra training --group dev
uv run python workflows/<command>.py --help
```

Read affected implementation, tests, and persisted schemas before editing. Make the smallest compatible change, add failure-path tests for behavior changes, run focused tests before the full suite, and verify documentation paths and wheel contents when packaging changes.

```bash
uv run pytest tests/test_public_runtime_contracts.py
uv run pytest
git diff --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for public contribution and security reporting expectations.
