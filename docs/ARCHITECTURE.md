# Architecture

This document is the authority for repository ownership and dependency direction. Persisted `cvi.*` values are historical schema identifiers; they are not Python namespaces.

## Public Runtime

`canine_identity.IdentityEngine` is the only public runtime. It accepts caller-provided `PIL.Image` crops and performs strict closed-set enrollment and retrieval. It does not decode video or invoke detection, tracking, temporal aggregation, open-set rejection, or a serving facade.

```text
explicit config v2
    -> receipt-bound evidence extractors
    -> required/optional EvidenceObservation handling
    -> gallery manifest v4 and content-addressed sidecars
    -> exact available-intersection weighted cosine
    -> maximum template score per registered UUIDv5 identity
    -> ordered Match values
```

The public package contains only `engine.py` and exports `IdentityEngine` and `Match`. Gallery bytes, scorer ordering, identity aggregation, and UUID rules remain versioned contracts.

## Package Ownership

| Path | Responsibility |
|---|---|
| `foundation/` | Deterministic hashing, protected I/O, publication, and retained-file primitives |
| `artifact_contracts/` | Model, source, runtime-library, intake, parity, and schema resources |
| `data_pipeline/` | Dataset adapters, acquisition, manifests, crop export, and duplicate evidence intake |
| `identity_governance/` | UUID registries, duplicate closure, split roles, exposure, and training/research admission |
| `localization/` | Detection, ROI geometry, prediction caches, and Nose-region materialization/training |
| `identity_methods/` | Backbones plus Appearance, Face, Nose, and classical identity methods |
| `representation_learning/` | Trainable representations, heads, objectives, and training orchestration |
| `evidence_fusion/` | Evidence observations, quality, calibration state, score fusion, and temporal research utilities |
| `identity_retrieval/` | Gallery persistence, exact scoring, and crop enrollment/search pipelines |
| `evaluation/` | Verification, retrieval, calibration, robustness, controls, cache evaluation, and protected evaluation |
| `canine_identity/` | Public crop-level runtime only |
| `operations/` | Isolated workers, runtime discovery, ONNX execution, supervision, and telemetry |
| `workflows/` | Source-checkout commands that orchestrate owned packages |
| `experiments/` | Research-only branch comparisons and major experiment configs |
| `apps/report/` | Optional report generation application; generated reports remain outside Git |
| `setup/` | Environment, bootstrap, and release guidance |
| `tests/` | Behavioral, contract, security, numerical, packaging, and dependency-boundary tests |

`Visualization/` is reserved with `.gitkeep`; it does not contain or imply current result artifacts. `paper/` contains guidance only until a manuscript and evidence scope are explicitly approved.

## Dependency Direction

`tests/test_dependency_boundaries.py` enforces these rules with an AST import scan:

1. `foundation` imports no other internal package.
2. `artifact_contracts` depends only on itself and `foundation`.
3. Algorithm packages do not import `evaluation` or `operations`.
4. `canine_identity` does not import learning, evaluation, operations, experiments, workflows, or apps.
5. `evaluation` may consume all algorithm packages.
6. `operations` may wrap algorithms and evaluation, but algorithms do not import it.

## Provenance Compatibility

Source provenance v2 binds a logical component and entry points to the recursive closure of repository-local Python imports. Physical path moves therefore produce explicit new provenance instead of pretending old file hashes still describe current code.

Historical compatibility is narrow:

- Face F4/F5 checkpoint validation accepts only the recorded pre-move architecture and dataset hashes. New Face contracts use `canine_identity.faceid_architecture_input_contract.v3` and source closure v2.
- Existing PDQ receipts continue to parse and verify `cvi.offline_tool_provenance.v1`; newly built provenance uses the logical v2 closure.
- Existing gallery, checkpoint, receipt, and evaluation schema identifiers remain unchanged unless the source-provenance metadata itself is explicitly versioned.

## External Artifacts

Raw datasets, licensed source archives, weights, checkpoints, galleries, caches, and mutable run outputs stay outside Git. Tracked JSON files are contracts or major experiment definitions, not results. See [Data and Models](DATA_AND_MODELS.md) and [Storage Security](STORAGE_SECURITY.md).
