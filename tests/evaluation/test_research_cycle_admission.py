from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from enrollment.registry.identity_registry import (
    compute_identity_token,
    compute_public_subject_token,
    compute_sample_token,
    compute_sequence_token,
)
from evaluation.splits.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    PublicSplitEvidenceEdge,
    PublicSplitSample,
    PublicSplitSourceBundle,
)
from shared.foundation.provenance import content_sha256
from evaluation.splits.research.research_cycle_admission import (
    IdentityTargetMode,
    ResearchCycleManifest,
    ResearchLicenseLane,
    ResearchRole,
    ResearchSourceAdmission,
    ResearchSourceAdmissions,
    ResearchSourceRole,
    build_research_cycle_manifest,
)
from evaluation.splits.role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
)

_IDENTITY_DATASETS = ("dogfacenet224", "mpdd", "sibetan", "yt-bb-dog")
_ALL_DATASETS = (*_IDENTITY_DATASETS, "ap10k-dog", "dogflw")
_BINDING_NAMES = (
    "exact_duplicate_graph_sha256",
    "geometric_verifier_sha256",
    "image_content_receipts_sha256",
    "pdq_candidates_sha256",
    "phash_candidates_sha256",
    "review_adjudication_sha256",
    "semantic_receipts_sha256",
)

def _sha(value: str) -> str:
    return content_sha256({"fixture": value})

def _sample(
    dataset: str,
    identity_index: int,
    *,
    variant: str = "original",
    sequence: str | None = None,
    paired_source_sample_id: str | None = None,
) -> PublicSplitSample:
    dataset_identity_id = f"{dataset}:v1:fixture:{identity_index}"
    source_sample_id = f"{dataset_identity_id}:sample:{variant}"
    identity_token = compute_identity_token(dataset_identity_id)
    return PublicSplitSample(
        sample_token=compute_sample_token(source_sample_id),
        identity_token=identity_token,
        sequence_token=compute_sequence_token(
            sequence or f"{dataset_identity_id}:sequence", identity_token
        ),
        source_sample_id=source_sample_id,
        dataset_identity_id=dataset_identity_id,
        dataset_name=dataset,
        source_variant=variant,
        original_split="test" if dataset == "yt-bb-dog" else None,
        raw_frame_index=identity_index,
        paired_source_sample_id=paired_source_sample_id,
        in_no_mono_subset=None,
        region="FACE",
    )

def _edge(
    left: PublicSplitSample,
    right: PublicSplitSample,
    relation: EvidenceRelation,
) -> PublicSplitEvidenceEdge:
    left_token, right_token = sorted((left.sample_token, right.sample_token))
    return PublicSplitEvidenceEdge(
        left_sample_token=left_token,
        right_sample_token=right_token,
        relation=relation,
        evidence_token=_sha(f"{left_token}:{right_token}:{relation.value}"),
    )

def _fixture(*, unresolved: bool = False):
    bindings = tuple((name, _sha(name)) for name in _BINDING_NAMES)
    samples = [
        _sample(dataset, identity_index)
        for dataset in _IDENTITY_DATASETS
        for identity_index in range(20)
    ]
    yt_original = next(
        item
        for item in samples
        if item.dataset_name == "yt-bb-dog" and item.dataset_identity_id.endswith(":0")
    )
    samples.append(
        _sample(
            "yt-bb-dog",
            0,
            variant="random_background",
            sequence=f"{yt_original.dataset_identity_id}:sequence",
            paired_source_sample_id=yt_original.source_sample_id,
        )
    )
    dogface = next(
        item
        for item in samples
        if item.dataset_name == "dogfacenet224"
        and item.dataset_identity_id.endswith(":0")
    )
    mpdd = next(
        item
        for item in samples
        if item.dataset_name == "mpdd" and item.dataset_identity_id.endswith(":0")
    )
    edges = [
        _edge(yt_original, samples[-1], EvidenceRelation.DEPENDENCY),
        _edge(dogface, mpdd, EvidenceRelation.EXACT_CONFIRMED),
    ]
    if unresolved:
        sibetan = [item for item in samples if item.dataset_name == "sibetan"]
        edges.append(
            _edge(sibetan[0], sibetan[1], EvidenceRelation.REVIEW_UNRESOLVED)
        )
    source = PublicSplitSourceBundle(bindings, tuple(reversed(samples)))
    graph = FrozenPublicSplitEvidenceGraph(
        bindings,
        tuple(
            sorted(
                edges,
                key=lambda item: (
                    item.left_sample_token,
                    item.right_sample_token,
                    item.relation.value,
                    item.evidence_token,
                ),
            )
        ),
    )
    historical = next(item for item in samples if item.dataset_name == "dogfacenet224")
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=_sha("historical-assignment"),
        kind=ExposureDeclarationKind.PRIOR_EVALUATION,
        revoked=True,
        records=(
            RoleExposureDeclarationRecord(
                sample_token=historical.sample_token,
                identity_token=historical.identity_token,
                public_subject_token=compute_public_subject_token(
                    historical.dataset_identity_id
                ),
                stage=ExposureStage.FINAL_TEST_SCORED,
            ),
        ),
    )
    ledger = merge_role_exposure_declarations((declaration,))
    receipt = create_role_exposure_receipt(ledger)
    source_documents = {
        dataset: {"schema_version": "fixture.source.v1", "dataset": dataset}
        for dataset in _ALL_DATASETS
    }
    admissions = ResearchSourceAdmissions(
        tuple(
            ResearchSourceAdmission(
                dataset_name=dataset,
                source_manifest_sha256=content_sha256(source_documents[dataset]),
                license_id=f"FIXTURE-{dataset}",
                license_lane=ResearchLicenseLane.RESEARCH_ONLY,
                source_role=(
                    ResearchSourceRole.IDENTITY_RESEARCH
                    if dataset in _IDENTITY_DATASETS
                    else ResearchSourceRole.AUXILIARY_ONLY
                ),
                identity_target_mode=(
                    IdentityTargetMode.CANONICAL_REGISTERED_UUIDV5
                    if dataset in _IDENTITY_DATASETS
                    else IdentityTargetMode.NONE
                ),
            )
            for dataset in sorted(_ALL_DATASETS)
        )
    )
    return source, graph, admissions, ledger, receipt, historical, source_documents

def _build(*, unresolved: bool = False):
    source, graph, admissions, ledger, receipt, historical, documents = _fixture(
        unresolved=unresolved
    )
    manifest = build_research_cycle_manifest(
        cycle_name="fixture-cycle-2026",
        source=source,
        graph=graph,
        source_admissions=admissions,
        role_exposure_ledger=ledger,
        role_exposure_receipt=receipt,
    )
    return manifest, (source, graph, admissions, ledger, receipt, historical, documents)

def test_assignments_are_deterministic_indivisible_and_target_70_15_15() -> None:
    first, fixture = _build()
    source, graph, admissions, ledger, receipt, _, _ = fixture
    second = build_research_cycle_manifest(
        cycle_name="fixture-cycle-2026",
        source=PublicSplitSourceBundle(source.evidence_bindings, tuple(source.samples)),
        graph=graph,
        source_admissions=admissions,
        role_exposure_ledger=ledger,
        role_exposure_receipt=receipt,
    )

    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert ResearchCycleManifest.from_dict(first.to_dict()) == first
    role_by_identity = {
        item.identity_token: item.role for item in first.identity_assignments
    }
    roles_by_component: dict[str, set[ResearchRole | None]] = {}
    roles_by_sequence: dict[str, set[ResearchRole | None]] = {}
    for sample in first.sample_assignments:
        assert sample.role == role_by_identity[sample.identity_token]
        roles_by_component.setdefault(sample.component_token, set()).add(sample.role)
        roles_by_sequence.setdefault(sample.sequence_token, set()).add(sample.role)
    assert all(len(roles) == 1 for roles in roles_by_component.values())
    assert all(len(roles) == 1 for roles in roles_by_sequence.values())

    counts = {dataset: dict(values) for dataset, values in first.dataset_role_counts}
    for dataset in _IDENTITY_DATASETS:
        assert sum(counts[dataset].values()) == 20
        assert counts[dataset] == {
            ResearchRole.RESEARCH_FIT.value: 14,
            ResearchRole.RESEARCH_DEV.value: 3,
            ResearchRole.RESEARCH_CAL.value: 3,
            "QUARANTINED": 0,
        }

def test_cross_dataset_duplicate_component_cannot_cross_research_roles() -> None:
    manifest, fixture = _build()
    source = fixture[0]
    duplicate_identities = {
        item.identity_token
        for item in source.samples
        if item.dataset_identity_id.endswith(":0")
        and item.dataset_name in {"dogfacenet224", "mpdd"}
    }
    assignments = [
        item for item in manifest.identity_assignments if item.identity_token in duplicate_identities
    ]
    assert len(assignments) == 2
    assert assignments[0].role == assignments[1].role
    assert assignments[0].allocation_block_token == assignments[1].allocation_block_token
    assert set(assignments[0].component_tokens) == set(assignments[1].component_tokens)

def test_auxiliary_sources_reject_identity_targets_and_identity_roles() -> None:
    with pytest.raises(ValueError, match="never receive identity targets"):
        ResearchSourceAdmission(
            dataset_name="ap10k-dog",
            source_manifest_sha256=_sha("ap10k"),
            license_id="FIXTURE",
            license_lane=ResearchLicenseLane.RESEARCH_ONLY,
            source_role=ResearchSourceRole.AUXILIARY_ONLY,
            identity_target_mode=IdentityTargetMode.CANONICAL_REGISTERED_UUIDV5,
        )
    with pytest.raises(ValueError, match="auxiliary-only roles"):
        ResearchSourceAdmission(
            dataset_name="dogflw",
            source_manifest_sha256=_sha("dogflw"),
            license_id="FIXTURE",
            license_lane=ResearchLicenseLane.RESEARCH_ONLY,
            source_role=ResearchSourceRole.IDENTITY_RESEARCH,
            identity_target_mode=IdentityTargetMode.NONE,
        )

    manifest, _ = _build()
    auxiliary = {
        item.dataset_name: item for item in manifest.source_admissions
        if item.dataset_name in {"ap10k-dog", "dogflw"}
    }
    assert set(auxiliary) == {"ap10k-dog", "dogflw"}
    assert all(
        item.source_role is ResearchSourceRole.AUXILIARY_ONLY
        and item.identity_target_mode is IdentityTargetMode.NONE
        for item in auxiliary.values()
    )
    assert not {
        item.dataset_name for item in manifest.identity_assignments
    } & set(auxiliary)

def test_historical_maximum_is_preserved_without_final_interpretation() -> None:
    manifest, fixture = _build()
    historical = fixture[5]
    assignment = next(
        item
        for item in manifest.identity_assignments
        if item.identity_token == historical.identity_token
    )
    assert assignment.historical_maximum_exposure is ExposureStage.FINAL_TEST_SCORED
    assert assignment.role in set(ResearchRole)
    assert manifest.final_evaluation_permitted is False
    assert manifest.score_inputs_used is False
    assert all("FINAL" not in role.value for role in ResearchRole)
    assert "NO_FINAL_EVALUATION" in manifest.interpretation

    changed = copy.deepcopy(manifest.to_dict())
    changed["final_evaluation_permitted"] = True
    with pytest.raises(ValueError, match="cannot permit final evaluation"):
        ResearchCycleManifest.from_dict(changed)

def test_unresolved_review_quarantines_without_partial_component_assignment() -> None:
    manifest, fixture = _build(unresolved=True)
    source = fixture[0]
    unresolved_identities = {
        item.identity_token
        for item in source.samples
        if item.dataset_name == "sibetan"
        and item.dataset_identity_id.endswith((":0", ":1"))
    }
    assignments = [
        item for item in manifest.identity_assignments if item.identity_token in unresolved_identities
    ]
    assert len(assignments) == 2
    assert all(item.role is None for item in assignments)
    assert all("UNRESOLVED_REVIEW" in item.quarantine_reasons for item in assignments)

def test_uuid5_and_namespace_are_strict() -> None:
    manifest, _ = _build()
    changed = copy.deepcopy(manifest.to_dict())
    changed["registered_identity_namespace_uuid"] = changed["cycle_namespace_uuid"]
    with pytest.raises(ValueError, match="registered identity namespace"):
        ResearchCycleManifest.from_dict(changed)

    changed = copy.deepcopy(manifest.to_dict())
    changed["identity_assignments"][0]["registered_dog_id"] = (
        "00000000-0000-4000-8000-000000000000"
    )
    with pytest.raises(ValueError, match="canonical UUIDv5"):
        ResearchCycleManifest.from_dict(changed)

    changed = copy.deepcopy(manifest.to_dict())
    changed["identity_assignments"][0]["registered_dog_id"] = manifest.cycle_id
    with pytest.raises(ValueError, match="not deterministic in its namespace"):
        ResearchCycleManifest.from_dict(changed)

def test_source_checkout_cli_builds_content_bound_manifest() -> None:
    _, fixture = _build()
    source, graph, admissions, ledger, receipt, _, documents = fixture
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = {
            "source": root / "source.json",
            "graph": root / "graph.json",
            "admissions": root / "admissions.json",
            "ledger": root / "ledger.json",
            "receipt": root / "receipt.json",
            "output": root / "manifest.json",
        }
        payloads = {
            "source": source.to_dict(),
            "graph": graph.to_dict(),
            "admissions": admissions.to_dict(),
            "ledger": ledger.to_dict(),
            "receipt": receipt.to_dict(),
        }
        for name, payload in payloads.items():
            paths[name].write_text(json.dumps(payload), encoding="utf-8")
        command = [
            sys.executable,
            "evaluation/splits/research/build_research_cycle_manifest.py",
            "--cycle-name",
            "fixture-cycle-2026",
            "--source-bundle",
            str(paths["source"]),
            "--dependency-graph",
            str(paths["graph"]),
            "--source-admissions",
            str(paths["admissions"]),
            "--role-exposure-ledger",
            str(paths["ledger"]),
            "--role-exposure-receipt",
            str(paths["receipt"]),
            "--output",
            str(paths["output"]),
        ]
        for dataset in sorted(documents):
            path = root / f"{dataset}.json"
            path.write_text(json.dumps(documents[dataset]), encoding="utf-8")
            command.extend(("--source-manifest", dataset, str(path)))
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        output = ResearchCycleManifest.from_dict(
            json.loads(paths["output"].read_text(encoding="utf-8"))
        )
        summary = json.loads(completed.stdout)
        assert summary["manifest_sha256"] == output.manifest_sha256
        assert summary["final_evaluation_permitted"] is False
