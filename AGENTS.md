# Contributor Guide For Automated And Human Changes

Treat current implementation, tests, schemas, and [Architecture](docs/ARCHITECTURE.md) as the source of truth. Do not infer capabilities from historical plans or filenames.

## Product Boundary

The canonical public runtime is `runtime.IdentityEngine`. It performs crop-level closed-set enrollment and retrieval. Detection, tracking, temporal aggregation, open-set decisions, report generation, and operations code are not connected product capabilities.

Do not add performance, deployment, dataset-size, or model-license claims without reproducible evidence and an attributable source. Unit and synthetic tests are not biometric validation.

## Pipeline Names

User-facing stage names map onto existing packages. Do not rename packages to match the stage names.

| Stage | Package | Meaning |
|---|---|---|
| Parsing | `parsing/` | Frozen detection/segmentation. Not called by `IdentityEngine`. |
| Identification | `embedding/` | Channel encoders and evidence contracts. |
| GenID | `retrieval/` enroll + `identity/` | Gallery K/V rows for a canonical UUIDv5. |
| ReID | `retrieval/` search | Availability-aware weighted cosine, not attention. |
| Evaluation | `evaluation/` | Metrics and protocols. Algorithms must not import it. |

First-user setup lives in [setup/](setup/README.md). Functional commands live in [workflows/README.md](workflows/README.md). Completed ablations live in [legacy/version/](legacy/version/README.md); report numbers live in [docs/RESEARCH_PROGRESS.md](docs/RESEARCH_PROGRESS.md).

- `parsing/`: frozen crops and masks. Do not import embedding, retrieval, evaluation, systems, or workflows. `IdentityEngine` must not import it.
- `embedding/`: encoders and evidence. Appearance is the current end-to-end channel. Do not import evaluation, systems, or workflows. Do not load optional model stacks at import time.
- `retrieval/`: enroll writes K/V rows; search is exact cosine. Preserve gallery bytes. Do not import evaluation, systems, or workflows.
- `workflows/`: keep `workflows/<command>.py`. Logic lives in the owner package. Update `workflows/README.md` in the same change.

## Compactness

1. No new top-level files or directories unless the user asks for them.
2. No new `docs/*.md`. Edit the existing document, or split only when one file has two owners.
3. No new `workflows/*.py` unless the logic already lives in an owner package, the CLI is a thin argparse wrapper, and `workflows/README.md` gains one line. When consolidating CLIs, keep the tested filename and absorb siblings as subcommands; do not invent a new command path.
4. Completed comparisons go to `legacy/version/`. Do not add new ablation modules under top-level `experiments/` or new protocol essays under `docs/`. Update the result table in `docs/RESEARCH_PROGRESS.md` instead.
5. Functional command paths stay `workflows/<command>.py`. Ablation CLIs stay `legacy/version/<set>/workflows/<command>.py`. One README per set. Do not dump protocol essays into `legacy/`.
6. Do not rename `parsing`, `embedding`, `retrieval`, `evaluation`, `runtime`, `data`, `identity`, `contracts`, `foundation`, `systems`, or `visualization`.
7. Do not move or rename root `LiteratureReview.md`.
8. No speculative `except Exception` or silent fallback. Cleanup-and-re-raise is allowed.
9. Do not add Java-style getters/setters. Public surfaces are explicit exports and dataclasses.
10. Prefer deleting unused code over wrapping it. Delete only when the module has no import, no test, and no current doc/CHANGELOG command.
11. Physical path moves create new source provenance. Do not move files to make names prettier.

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
./setup/check_env.sh cpu
uv run python workflows/<command>.py --help
```

Read affected implementation, tests, and persisted schemas before editing. Make the smallest compatible change, add failure-path tests for behavior changes, run focused tests before the full suite, and verify documentation paths and wheel contents when packaging changes.

```bash
uv run pytest tests/test_public_runtime_contracts.py
uv run pytest
git diff --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for public contribution and security reporting expectations.
