# Architecture

This document is the authority for repository ownership and dependency direction. Persisted `cvi.*` values are historical schema identifiers; they are not Python namespaces.

## Public Runtime

`canine_identity.IdentityEngine` is the only public runtime. It accepts caller-provided `PIL.Image` crops and performs strict closed-set enrollment and retrieval. It does not decode video or invoke detection, tracking, temporal aggregation, open-set rejection, or a serving facade.

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

The supported public Python export surface is `IdentityEngine` and `Match`, implemented by `canine_identity/__init__.py` and `engine.py`. Gallery bytes, scorer ordering, identity aggregation, and UUID rules remain versioned contracts.

## Retrieval Roles

QKV names describe retrieval roles, not an attention mechanism. There are no
learned Q/K/V projections, attention matrices, softmax weights, or value mixing in
the canonical runtime.

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
| `foundation/` | Deterministic hashing, protected I/O, publication, and retained-file primitives |
| `contracts/` | Model, source, runtime-library, intake, parity, and fail-closed region-evidence schemas |
| `data_pipeline/` | Dataset adapters, acquisition, manifests, crop export, and duplicate evidence intake |
| `identity_governance/` | UUID registries, duplicate closure, split roles, exposure, research admission, and retrospective identity K-folds |
| `localization/` | Detection, ROI geometry, prediction caches, identity-free localization folds, and region materialization/training |
| `identity_methods/` | Backbones plus Appearance, Face, Nose, and classical identity methods |
| `representation_learning/` | Trainable representations, heads, objectives, and training orchestration |
| `evidence_fusion/` | Evidence observations, quality, calibration state, and research-only aggregation utilities |
| `identity_retrieval/` | QKV retrieval contracts, K/V gallery persistence, exact QK scoring, identity aggregation, and crop enrollment/search pipelines |
| `evaluation/` | Verification, retrieval, calibration, robustness, controls, cache evaluation, and protected evaluation |
| `canine_identity/` | Public crop-level runtime only |
| `operations/` | Isolated workers, runtime discovery, ONNX execution, supervision, and telemetry |
| `workflows/` | Source-checkout commands that orchestrate owned packages |
| `experiments/` | Research-only branch comparisons and major experiment configs |
| `visualization/` | Contract-bound research figures and report rendering; generated reports remain outside Git |
| `setup/` | Environment, bootstrap, and release guidance |
| `tests/` | Behavioral, contract, security, numerical, packaging, and dependency-boundary tests |

`Visualization/` is an ignored local-output directory. `paper/` contains guidance only until a manuscript and evidence scope are explicitly approved.

The dataset-stratified identity and localization K-fold manifests are exposed retrospective research protocols, not independent final tests. The three-region A/F/N artifact manifest distinguishes verified semantic masks from model candidates, geometric proxies, and crop source-validity masks; incomplete evidence remains explicit and does not imply segmentation capability.

## Dependency Direction

`tests/test_dependency_boundaries.py` enforces these rules with an AST import scan:

1. `foundation` imports no other internal package.
2. `contracts` depends only on itself and `foundation`.
3. Algorithm packages do not import `evaluation` or `operations`.
4. `canine_identity` does not import learning, evaluation, operations, experiments, workflows, or apps.
5. `evaluation` may consume all algorithm packages.
6. `operations` may wrap algorithms and evaluation, but algorithms do not import it.

## Versioned Compatibility

Source provenance v2 binds a logical component and entry points to the recursive closure of repository-local Python imports. Physical path moves therefore produce explicit new provenance instead of pretending old file hashes still describe current code.

Compatibility is narrow:

- Face F4/F5 checkpoint validation accepts only the recorded pre-move architecture and dataset hashes. New Face contracts use `canine_identity.faceid_architecture_input_contract.v3` and source closure v2.
- Existing PDQ receipts continue to parse and verify `cvi.offline_tool_provenance.v1`; newly built provenance uses the logical v2 closure.
- Existing gallery, checkpoint, receipt, and evaluation schema identifiers remain unchanged unless the source-provenance metadata itself is explicitly versioned.
- Gallery v4 and v5 are accepted; gallery v3 requires explicit migration.
- Full128 route-plan v2, parser policies v4/v5, and parser runtime
  manifest/bundle v1 remain readable for completed external artifacts. Current
  generation uses route-plan v3, dog-only parser policy v6, and parser runtime v2.

## External Artifacts

Raw datasets, licensed source archives, weights, checkpoints, galleries, caches, and mutable run outputs stay outside Git. Tracked JSON files are contracts or major experiment definitions, not results. See [Data and Models](DATA_AND_MODELS.md) and [Storage Security](STORAGE_SECURITY.md).
