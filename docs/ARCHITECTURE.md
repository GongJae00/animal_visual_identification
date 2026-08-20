# Architecture

This document is the authority for repository ownership and dependency direction. Persisted schema identifiers are capability-owned contract strings; they are not Python namespaces.

## Public Runtime

`prototype.runtime.IdentityEngine` is the only public runtime. It accepts caller-provided `PIL.Image` crops and performs strict closed-set enrollment and search. It does not decode video or invoke detection, tracking, temporal aggregation, open-set rejection, or a serving facade. The public import is `from prototype.runtime import IdentityEngine, Match`. Persisted schema identifiers are capability-owned contract strings, not Python namespaces.

```text
explicit config v2
    -> receipt-bound evidence extractors
    -> required/optional EvidenceObservation handling
    -> role-neutral channel embeddings
    -> RetrievalQuery (Q)
    -> GalleryKey (K) / GalleryValue (V) rows
    -> exact availability-aware QK weighted cosine
    -> maximum template value per registered UUIDv5 identity
    -> ordered Match values
```

The supported public Python export surface is `IdentityEngine` and `Match`, implemented by `prototype/runtime/__init__.py` and `engine.py`. Gallery bytes, scorer ordering, identity aggregation, and UUID rules remain versioned contracts.

## Research Pipeline

Target order: `parsing → identification → representation → enrollment → gallery → search → evaluation`. 등록 is `enrollment/`. 검색 is `search/`. GenID and ReID are not stage names. Vendor names (Pet-ReID, MiewID) stay. Visualization observes from outside; prototype composes `export/` only.

The packages below are the target tree. See [AGENTS.md](../AGENTS.md).

| Stage | Package | Runtime meaning |
|---|---|---|
| Parsing | `parsing/` | Frozen detection/segmentation that materializes crops and masks. Not invoked by `IdentityEngine`. |
| Identification | `identification/` | Encoders that emit per-channel embeddings. Appearance is the live channel. |
| Representation | `representation/` | Evidence contracts, quality, and channel packing. |
| 등록 | `enrollment/` | Persist GalleryKey / GalleryValue rows for a registry UUIDv5. |
| Gallery | `gallery/` | Store and v3→v4 migration. |
| 검색 | `search/` | Score a RetrievalQuery against stored keys with exact cosine, then aggregate by identity. |
| Evaluation | `evaluation/` | Identity-disjoint protocols and metrics. Algorithm packages must not import it. |

Commands: `uv run python -m parsing.commands.parse --help`, `identification.commands.train`, `identification.commands.export`, `representation.commands.embed`, `enrollment.commands.enroll`, `gallery.commands.migrate`, `evaluation.commands.evaluate`, `visualization.commands.render`, `prototype.commands.export`, `data.commands.download`, `data.commands.audit`, `operations.commands.measure`. Search has no extra CLI; the product is `IdentityEngine.search`. First-user environment setup lives under `setup/`.

## Retrieval Roles

Query, gallery-key, and gallery-value names describe retrieval roles, not an
attention mechanism. There are no learned Q/K/V projections, attention matrices,
softmax weights, or value mixing in the canonical runtime.

| Role | Runtime meaning |
|---|---|
| `RetrievalQuery` (Q) | Available, independently normalized channel vectors extracted from one query crop |
| `GalleryKey` (K) | Available channel vectors stored for one enrollment template |
| `GalleryValue` (V) | Canonical identity, template identity, content binding, breed, metadata, and enrollment provenance aligned to that key |

For each template, scoring uses only channels available in both Q and K and
renormalizes configured channel weights over that intersection. Multiple template
keys may map to the same registered identity. Identity aggregation selects the
maximum-scoring complete template value, with deterministic template and identity
tie-breaking.

`Encoder` names a model that produces a vector. `Embedding` names an individual
normalized channel vector. `Representation` names a structured collection of
embeddings and evidence state. Query and key roles are assigned only after the
same representation path has run for query or enrollment input.

## Package Ownership

| Path | Responsibility |
|---|---|
| `shared/foundation/` | Deterministic hashing, protected I/O, publication, and retained-file primitives |
| `shared/contracts/` | Model, source, runtime-library, parity, PDQ, and cross-domain evidence contracts; external model-asset intake lives under `shared/contracts/intake/` |
| `data/` | Dataset adapters, acquisition, crop export, source route planning, and shared data primitives; public-corpus intake lives under `data/public_sources/` |
| `enrollment/` | Canonical UUIDv5 registry, registered-only gallery policy, and crop/vector write |
| `gallery/` | On-disk key/value store and v3→v4 migration |
| `search/` | Query / gallery-key / gallery-value scoring and crop matching. Not attention. |
| `parsing/` | Detection, segmentation, regions, quality, and crops. `training/` vs `export/`; commands at `parsing/commands/parse.py`. Identification embedding trainers are not parsing-owned. |
| `identification/` | Channel encoders. `training/` vs `export/` for appearance, face, and nose. Commands at `identification/commands/`. |
| `representation/` | Evidence contracts, quality observations, and channel packing. No trainers. |
| `archive/` | Completed comparison families: `full128`, `appearance_face_nose`, `nose_metric`, `nose`, `face`, `shared_helpers`. Live identification path is Appearance. |
| `data/audit/` | PDQ, pHash, and geometry audit helpers. Not a pipeline stage. |
| `evaluation/` | Verification, search metrics, calibration, robustness, protected evaluation, localization, controls, integrity, and identity-disjoint splits |
| `prototype/` | Public crop-level runtime and ONNX export backends |
| `operations/` | Isolated workers, decode/ONNX/capacity measurement, and video helpers |
| `visualization/` | Pipeline observer (`Visualization/vis/00_parsing` … `05_search`) and paper rasters (`Visualization/paper/`). Stages do not import it. |
| `setup/` | Environment, bootstrap, and release guidance |
| `tests/` | Behavioral, contract, security, numerical, packaging, and dependency-boundary tests, split by stage under `tests/<stage>/` |

`Visualization/` is an ignored local-output directory. `paper/` is guidance only.

## Dependency Direction

`tests/test_dependency_boundaries.py` enforces these rules with an AST import scan. Tests are split by stage under `tests/<stage>/`; completed-comparison tests live under `tests/archive/<family>/`.

1. `shared.foundation` imports no other internal package.
2. `shared.contracts` depends only on itself and `shared.foundation`.
3. `data` does not import enrollment, gallery, search, parsing, identification, representation, evaluation, operations, prototype, or visualization.
4. Algorithm packages do not import `evaluation`, `operations`, `archive`, or `visualization`.
   Parsing additionally does not import identification, representation, enrollment, gallery, search, prototype, or operations.
   `identification.export` does not import `identification.training`.
   Visualization imports `export/` only, never `training/`.
5. `prototype.runtime` does not import training, evaluation, operations, visualization, or parsing.
6. `evaluation` may consume all algorithm packages.
7. `operations` may wrap algorithms, prototype export, and evaluation, but algorithms do not import it.
8. The complete top-level internal package graph must remain acyclic.

## Versioned Compatibility

Source provenance v3 binds a logical component, entry points, existing parent package initializers, and the recursive closure of repository-local Python imports. Physical path moves therefore produce explicit new provenance instead of pretending old file hashes still describe current code. Complete v1/v2 inventories remain readable where their persisted readers require them.

Compatibility is narrow:

- Face F4/F5 checkpoint validation accepts only the recorded pre-move architecture and dataset hashes. New Face contracts use `identification.face.architecture_input_contract.v3` and source closure v2.
- PDQ builder provenance requires current `data/audit/pdq` sources and `archive.shared_helpers.commands.build_native_pdq_worker`.
- Gallery v4 and v5 are accepted; gallery v3 requires explicit migration. Schema identifiers are capability-owned.
- Gallery v4 and v5 are accepted; gallery v3 requires explicit migration.
- Full128 route-plan v2, parser policies v4/v5, and parser runtime
  manifest/bundle v1 remain readable for completed external artifacts. Current
  generation uses route-plan v3, dog-only parser policy v6, and parser runtime v2.

## External Artifacts

Raw datasets, licensed source archives, weights, checkpoints, galleries, caches, and mutable run outputs stay outside Git. Tracked JSON files are contracts or major experiment definitions, not results. See [Data and Models](DATA_AND_MODELS.md) and [Storage Security](STORAGE_SECURITY.md).
