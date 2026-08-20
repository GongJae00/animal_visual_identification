# Contributor Guide For Automated And Human Changes

Treat current implementation, tests, schemas, and [Architecture](docs/ARCHITECTURE.md) as the source of truth for behavior. This file is the law for the target tree. Do not infer capabilities from historical plans or filenames. Do not create empty destination packages, move modules, or change Python imports until the wave that owns that path.

## Product Boundary

The public runtime is crop-level closed-set enrollment and search. Detection, tracking, temporal aggregation, open-set decisions, report generation, and operations code are not connected product capabilities.

Public import:

```python
from prototype.runtime import IdentityEngine, Match
```

Behavior stays: caller-provided crops, fail-closed required evidence, explicit optional evidence, local gallery, availability-aware weighted cosine (not attention), identity-level max template with deterministic ordering.

`evaluation.open_set` and `evaluation.open_set_calibration` are research protocols, not `IdentityEngine` capabilities. Fitting, threshold, and policy selection stay off evaluation identities.

Do not add performance, deployment, dataset-size, or model-license claims without reproducible evidence and an attributable source. Unit and synthetic tests are not biometric validation. Public API, gallery, and failure-behavior changes require contract and wheel-installed tests.

## Frozen Contracts

Do not change these:

- Persisted `cvi.*` schema identifier strings (compatibility contracts, not Python package names)
- Public field `dog_id` and env `CANINE_IDENTITY_DATA_DIR`
- `IdentityEngine` + `Match` behavior described above
- Root `LiteratureReview.md`
- Python 3.12, `uv`, `cpu` xor `cuda`
- Vendor names (`PetReIDExtractor`, Pet-ReID, MiewID, ONNX, PDQ)
- External weights, galleries, PNG outputs stay out of Git

## Target Tree

Capability nouns at top. `parsing/` and `identification/` split into `training/` + `export/`. 등록 = `enrollment/`, 갤러리 = `gallery/`, 검색 = `search/`. Visualization observes from outside. Prototype composes `export/` only. `training/` never ships.

```text
parsing/{training,export,commands}
identification/{training,export,commands}
representation/{channels,evidence,quality,commands}
enrollment/{registry,binding,write,commands}
gallery/{store,schema,migration,commands}
search/{scoring,matching,commands}
evaluation/{splits,search_metrics,verification,controls,commands}
visualization/{parsing,identification,representation,enrollment,gallery,search,commands}
prototype/{runtime,export,commands}
data/{acquisition,public_sources,commands}
shared/{foundation,contracts}
operations/{workers,measurement,video}
archive/{full128,appearance_face_nose,nose_metric,nose,face,shared_helpers}
setup/  tests/{parsing,identification,representation,enrollment,gallery,search,evaluation,visualization,prototype,data,shared,operations,archive}  docs/  paper/  LiteratureReview.md
```

Reading order:

`parsing → identification → representation → enrollment → gallery → search → evaluation`

GenID and ReID are not stage names. Do not revert 등록/검색 to those labels. `search/` is the working directory name for 검색; a more precise folder name may replace it later.

`export/` children of parsing: `detection/ segmentation/ regions/ quality/ crops/`.
`training/` and `export/` children of identification: `appearance/ face/ nose/` (same names both sides).

Outputs (not in Git): `Visualization/vis/00_parsing/` … `05_search/`. Paper figure registry `00`–`17` is not this sequence.

## Pipeline Ownership

| Stage | Package | Meaning |
|---|---|---|
| Parsing | `parsing/` | Detection/segmentation. `training/` vs `export/`. Not called by `IdentityEngine`. |
| Identification | `identification/` | Channel encoders. Appearance is the current end-to-end channel. `training/` vs `export/`. |
| Representation | `representation/` | Channel packing, evidence, quality. No trainers. |
| 등록 | `enrollment/` | Gallery K/V rows for a canonical UUIDv5. |
| Gallery | `gallery/` | Store, schema, migration. Preserve gallery bytes. |
| 검색 | `search/` | Availability-aware weighted cosine, not attention. |
| Evaluation | `evaluation/` | Metrics and protocols. Algorithms must not import it. |

`prototype/` is the only ship surface. `visualization/` is an observer: it imports `export/` only, never `training/`, and is not a producer stage. First-user setup lives in [setup/](setup/README.md). Report numbers live in [docs/RESEARCH_PROGRESS.md](docs/RESEARCH_PROGRESS.md). Completed comparisons go to `archive/`.

Handoff types live on the producing stage. `shared/contracts` holds persisted schemas only.

## Commands

Commands live at `<stage>/commands/<verb>.py`. Thin argparse. Logic lives in the owner package. One verb per job; siblings become subcommands.

```bash
uv run python -m parsing.commands.parse --help
```

No `workflows/`, `utils/`, `misc/`, `helpers/`, or `common/` dumps.

## Laws

1. No new top-level name outside the target tree. Deeper subtrees are allowed only under an existing stage, named by a function noun, when one module has two owners. Do not add a folder to put leftovers.
2. Do not create empty destination packages ahead of the wave that moves code into them.
3. `training/` is not imported by `export/`, `prototype/`, `visualization/`, or later stages.
4. Stages do not import `evaluation`, `visualization`, `operations`, or `archive`.
5. Visualization imports `export/` only. Writes only to gitignored `Visualization/vis/{00_parsing..05_search}/`.
6. Prototype composes `export/` only. `IdentityEngine` must not import parsing, training, visualization, evaluation, or operations.
7. No compatibility shim left at the end of a wave. Same-commit import and test updates.
8. Delete only with evidence: no import, no test, no current command, no receipt entrypoint. Prefer delete over wrap.
9. Touching a file: remove GenID/ReID as *stage* names, silent `except Exception` fallbacks, fat `__init__.py` implementations, unused helpers. Vendor names and historical CHANGELOG entries may keep ReID.
10. Physical path moves create new source provenance. Do not fake old hashes.

## Compactness

1. No new `docs/*.md`. Edit the existing document, or split only when one file has two owners.
2. No new `<stage>/commands/*.py` unless the logic already lives in an owner package, the CLI is a thin argparse wrapper, and the stage README or root README gains one line. When consolidating CLIs, keep the tested verb and absorb siblings as subcommands.
3. Completed comparisons go to `archive/`. Do not add new ablation modules under top-level `experiments/` or new protocol essays under `docs/`. Update the result table in `docs/RESEARCH_PROGRESS.md` instead.
4. Do not move or rename root `LiteratureReview.md`.
5. No speculative `except Exception` or silent fallback. Cleanup-and-re-raise is allowed.
6. Do not add Java-style getters/setters. Public surfaces are explicit exports and dataclasses.
7. Prefer deleting unused code over wrapping it. Delete only when the module has no import, no test, and no current doc/CHANGELOG command.
8. Each stage `__init__.py`: 5–15 lines, public types only.
9. Root `AGENTS.md` is the only agent law. Do not add per-package `AGENTS.md`.

## Clean Replacement

Edits replace dirty code in place. Do not reintroduce agent residue, and do not wrap unused code to keep it for later. New work must read as if the previous full-repo cleanup already happened.

Dirty — delete on evidence, and refuse to add:

- Comments or doc lines that narrate the next statement, restate a name, or use `TODO` / `NOTE` / `FIXME` / `IMPORTANT` banners
- Live comments that treat GenID or ReID as pipeline stages. Vendor names (`Pet-ReID`, MiewID) and historical CHANGELOG entries may keep ReID
- Unused helpers, wrappers, example configs not loaded by a current command or test, and modules with no command, test, receipt, or `IdentityEngine` owner
- Silent `except Exception`, default-on-failure, and sentinel placeholders (`__GIT_FAILED__`, `UNVERIFIED` written as success)
- Fat `__init__.py` implementations, dump folders (`utils/`, `misc/`, `helpers/`, `common/`, `workflows/`), leftover compatibility shims after a move
- New knobs, retries, fallbacks, or optional paths for hypothetical callers
- New `docs/*.md`, per-package `AGENTS.md`, or a README that restates the package
- Performance, deployment, dataset-size, or license claims without a cited reproducible source

Keep, after classifying the site:

- Cleanup-and-re-raise, `raise X from exc`, fail-closed mapping to `QualityReason`, and destructor swallows
- Comments that record a non-obvious constraint: schema identifier, fail-closed rule, provenance, or filesystem semantics (including DrvFS `renameat2` fallback — that is not leftover native-disk optimization)
- Archive families as completed comparisons, not as live features

When touching a file: read implementation, tests, and persisted schemas first. Classify every broad `except` before changing it. Delete only with evidence (no import, no test, no current command, no receipt entrypoint). Same-commit import and test updates. No shim at the end of a wave. New code exists only when a current command, test, or `IdentityEngine` path needs it.

## Invariants

1. Enrollment identities are canonical UUIDv5 values, separate from display names, source labels, sample tokens, and track identifiers.
2. Evaluation and training partitions must be identity-disjoint where the protocol requires it; random frame splitting is not an acceptable shortcut.
3. Required evidence fails closed. Optional evidence is explicit in config v2 and remains auditable in gallery state.
4. Model, preprocessing, gallery, source, and receipt schemas remain versioned and content-bound. Persisted `cvi.*` identifiers are compatibility contracts, not Python package names.
5. External datasets, weights, caches, galleries, and experiment outputs stay outside Git. Acquisition is not admission: download and extraction fail closed, reject traversal and ambiguous archives, and never imply a dataset is admitted.
6. CUDA behavior is optional and guarded. Portable CPU behavior must not import CUDA-only dependencies at package import time.
7. Never commit secrets, private animal or owner data, credentials, or licensed artifacts.
8. Respect the dependency direction enforced by `tests/test_dependency_boundaries.py`; algorithms must not depend on evaluation or operations. Update that test in the same wave a package split lands. Tests live under `tests/<stage>/`; archive families live under `tests/archive/<family>/`.

## Environment And Workflow

Use Linux, Python 3.12, and `uv`. Select either the `cpu` or `cuda` extra, never both.

```bash
./setup/check_env.sh cpu
uv run python -m <stage>.commands.<verb> --help
```

Read affected implementation, tests, and persisted schemas before editing. Make the smallest compatible change, add failure-path tests for behavior changes, run focused tests before the full suite, and verify documentation paths and wheel contents when packaging changes.

```bash
uv run pytest tests/test_dependency_boundaries.py tests/prototype/test_public_runtime_contracts.py
uv run pytest
git diff --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for public contribution and security reporting expectations.
