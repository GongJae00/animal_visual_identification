from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from shared.foundation.provenance import content_sha256
from evaluation.splits.research.dataset_stratified_kfold import (
    DatasetStratifiedIdentityKFoldManifest,
    DatasetStratifiedKFoldPolicy,
    HeldOutSampleRole,
    build_dataset_stratified_identity_kfold,
    dataset_stratified_kfold_bundle,
    materialize_identity_fold,
    read_dataset_stratified_identity_kfold,
)
from enrollment.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
    compute_sequence_token,
)
from evaluation.splits.research.research_cycle_admission import (
    CVI_RESEARCH_CYCLE_NAMESPACE,
    IdentityTargetMode,
    ResearchCycleManifest,
    ResearchIdentityAssignment,
    ResearchLicenseLane,
    ResearchRole,
    ResearchSampleAssignment,
    ResearchSourceAdmission,
    ResearchSourceAdmissions,
    ResearchSourceRole,
    compute_research_cycle_id,
)
from evaluation.splits.role_exposure import ExposureStage

_IDENTITY_DATASETS = ("dogfacenet224", "mpdd", "sibetan", "yt-bb-dog")
_ALL_DATASETS = (*_IDENTITY_DATASETS, "ap10k-dog", "dogflw")

def _sha(value: str) -> str:
    return content_sha256({"fixture": value})

def _research_cycle() -> ResearchCycleManifest:
    admissions = ResearchSourceAdmissions(
        tuple(
            ResearchSourceAdmission(
                dataset_name=dataset,
                source_manifest_sha256=_sha(f"source:{dataset}"),
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
    identities: list[ResearchIdentityAssignment] = []
    samples: list[ResearchSampleAssignment] = []
    shared_component = _sha("cross-dataset-duplicate-component")
    cross_block = _sha("cross-dataset-allocation-block")
    cross_components = tuple(
        sorted(
            (
                shared_component,
                _sha("dogfacenet224:0:component:1"),
                _sha("dogfacenet224:0:component:2"),
                _sha("mpdd:0:component:1"),
                _sha("mpdd:0:component:2"),
            )
        )
    )
    role_counts: dict[str, Counter[str]] = {
        dataset: Counter() for dataset in _IDENTITY_DATASETS
    }
    for dataset in _IDENTITY_DATASETS:
        for identity_index in range(10):
            label = f"{dataset}:v1:fixture:{identity_index}"
            identity_token = compute_identity_token(label)
            quarantined = dataset == "sibetan" and identity_index == 9
            if quarantined:
                role = None
                reasons = ("UNRESOLVED_REVIEW",)
            elif identity_index < 7:
                role = ResearchRole.RESEARCH_FIT
                reasons = ()
            elif identity_index < 9:
                role = ResearchRole.RESEARCH_DEV
                reasons = ()
            else:
                role = ResearchRole.RESEARCH_CAL
                reasons = ()
            role_counts[dataset]["QUARANTINED" if role is None else role.value] += 1
            if identity_index == 0 and dataset in {"dogfacenet224", "mpdd"}:
                block = cross_block
                components = cross_components
                own_components = (
                    shared_component,
                    _sha(f"{dataset}:0:component:1"),
                    _sha(f"{dataset}:0:component:2"),
                )
            else:
                block = _sha(f"{dataset}:{identity_index}:block")
                component_count = 1 if dataset == "dogfacenet224" and identity_index == 9 else 3
                own_components = tuple(
                    _sha(f"{dataset}:{identity_index}:component:{index}")
                    for index in range(component_count)
                )
                components = tuple(sorted(own_components))
            identities.append(
                ResearchIdentityAssignment(
                    identity_token=identity_token,
                    dataset_identity_id=label,
                    registered_dog_id=compute_registered_dog_id(label),
                    dataset_name=dataset,
                    role=role,
                    allocation_block_token=block,
                    component_tokens=components,
                    historical_maximum_exposure=(
                        ExposureStage.FINAL_TEST_SCORED
                        if dataset == "dogfacenet224" and identity_index == 0
                        else None
                    ),
                    quarantine_reasons=reasons,
                )
            )
            for sample_index, component in enumerate(own_components):
                source_id = f"{label}:sample:{sample_index}"
                samples.append(
                    ResearchSampleAssignment(
                        sample_token=compute_sample_token(source_id),
                        identity_token=identity_token,
                        sequence_token=compute_sequence_token(
                            (
                                f"{label}:sequence:unsafe-shared"
                                if dataset == "dogfacenet224"
                                and identity_index == 0
                                and sample_index < 2
                                else f"{label}:sequence:shared"
                                if dataset == "mpdd" and identity_index == 1 and sample_index < 2
                                else f"{label}:sequence:{sample_index}"
                            ),
                            identity_token,
                        ),
                        component_token=component,
                        source_variant="original",
                        role=role,
                    )
                )
            if dataset == "yt-bb-dog" and identity_index == 0:
                source_id = f"{label}:sample:random-background"
                samples.append(
                    ResearchSampleAssignment(
                        sample_token=compute_sample_token(source_id),
                        identity_token=identity_token,
                        sequence_token=compute_sequence_token(
                            f"{label}:sequence:0", identity_token
                        ),
                        component_token=own_components[0],
                        source_variant="random_background",
                        role=role,
                    )
                )
    count_keys = (*tuple(role.value for role in ResearchRole), "QUARANTINED")
    dataset_role_counts = tuple(
        (
            dataset,
            tuple((key, role_counts[dataset][key]) for key in count_keys),
        )
        for dataset in sorted(_IDENTITY_DATASETS)
    )
    cycle_name = "kfold-fixture-cycle"
    return ResearchCycleManifest(
        cycle_name=cycle_name,
        cycle_id=compute_research_cycle_id(cycle_name),
        cycle_namespace_uuid=str(CVI_RESEARCH_CYCLE_NAMESPACE),
        registered_identity_namespace_uuid=str(REGISTERED_DOG_NAMESPACE),
        source_bundle_sha256=_sha("source-bundle"),
        dependency_graph_sha256=_sha("dependency-graph"),
        source_admissions_sha256=admissions.admissions_sha256,
        source_admissions=admissions.sources,
        role_exposure_ledger_sha256=_sha("exposure-ledger"),
        role_exposure_receipt_sha256=_sha("exposure-receipt"),
        target_percentages=(
            ("RESEARCH_FIT", 70),
            ("RESEARCH_DEV", 15),
            ("RESEARCH_CAL", 15),
        ),
        dataset_role_counts=dataset_role_counts,
        identity_assignments=tuple(
            sorted(identities, key=lambda item: item.identity_token)
        ),
        sample_assignments=tuple(sorted(samples, key=lambda item: item.sample_token)),
    )

def _build() -> DatasetStratifiedIdentityKFoldManifest:
    return build_dataset_stratified_identity_kfold(
        protocol_name="unified-afn-five-fold-v1",
        research_cycle=_research_cycle(),
        policy=DatasetStratifiedKFoldPolicy(),
    )

def test_fold_manifest_is_deterministic_round_trippable_and_dataset_stratified() -> None:
    first = _build()
    second = _build()
    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert DatasetStratifiedIdentityKFoldManifest.from_dict(first.to_dict()) == first
    counts = {
        dataset: values for dataset, values, _ in first.dataset_fold_counts
    }
    assert set(counts) == set(_IDENTITY_DATASETS)
    assert all(len(values) == 5 and min(values) >= 1 for values in counts.values())
    assert first.final_evaluation_permitted is False
    assert first.score_inputs_used is False
    assert "NO_FINAL_EVALUATION" in first.interpretation

def test_every_identity_rotates_test_once_dev_once_and_train_three_times() -> None:
    manifest = _build()
    observed: dict[str, Counter[str]] = {
        item.identity_token: Counter() for item in manifest.identity_assignments
    }
    for fold_index in range(manifest.policy.fold_count):
        view = materialize_identity_fold(manifest, fold_index)
        assert view["view_sha256"] == content_sha256(view["view"])
        for row in view["view"]["identity_assignments"]:
            observed[row["identity_token"]][row["stage"]] += 1
    for item in manifest.identity_assignments:
        if item.home_fold is None:
            assert observed[item.identity_token] == {"QUARANTINED": 5}
        else:
            assert observed[item.identity_token] == {"TRAIN": 3, "DEV": 1, "TEST": 1}

def test_allocation_blocks_and_components_never_cross_folds_or_gallery_query() -> None:
    manifest = _build()
    folds_by_block: dict[str, set[int | None]] = {}
    for item in manifest.identity_assignments:
        folds_by_block.setdefault(item.allocation_block_token, set()).add(item.home_fold)
    assert all(len(folds) == 1 for folds in folds_by_block.values())

    roles_by_component: dict[tuple[str, str], set[HeldOutSampleRole]] = {}
    identities_by_component: dict[str, set[str]] = {}
    for sample in manifest.sample_assignments:
        if sample.source_variant != "original":
            continue
        roles_by_component.setdefault(
            (sample.identity_token, sample.component_token), set()
        ).add(sample.held_out_role)
        identities_by_component.setdefault(sample.component_token, set()).add(
            sample.identity_token
        )
    assert all(len(roles) == 1 for roles in roles_by_component.values())
    cross_component = next(
        component for component, identities in identities_by_component.items() if len(identities) > 1
    )
    assert {
        role
        for (identity, component), roles in roles_by_component.items()
        if component == cross_component
        for role in roles
    } == {HeldOutSampleRole.EXCLUDED}
    unsafe_sample = next(
        item
        for item in manifest.sample_assignments
        if item.component_token == cross_component
        and item.identity_token
        == compute_identity_token("dogfacenet224:v1:fixture:0")
    )
    assert unsafe_sample.training_eligible is False
    unsafe_sequence = unsafe_sample.sequence_token
    unsafe_unit_samples = [
        item
        for item in manifest.sample_assignments
        if item.identity_token == unsafe_sample.identity_token
        and item.sequence_token == unsafe_sequence
    ]
    assert len(unsafe_unit_samples) == 2
    assert all(
        item.held_out_role is HeldOutSampleRole.EXCLUDED
        and item.training_eligible is False
        for item in unsafe_unit_samples
    )
    train_view = next(
        materialize_identity_fold(manifest, fold)["view"]
        for fold in range(manifest.policy.fold_count)
        if next(
            row
            for row in materialize_identity_fold(manifest, fold)["view"][
                "identity_assignments"
            ]
            if row["identity_token"] == unsafe_sample.identity_token
        )["stage"]
        == "TRAIN"
    )
    unsafe_row = next(
        row
        for row in train_view["sample_assignments"]
        if row["sample_token"] == unsafe_sample.sample_token
    )
    assert unsafe_row["sample_role"] == "EXCLUDED"

def test_gallery_query_uses_image_count_and_excludes_insufficient_identity() -> None:
    manifest = _build()
    dogface_nine = compute_identity_token("dogfacenet224:v1:fixture:9")
    excluded = [
        item for item in manifest.sample_assignments if item.identity_token == dogface_nine
    ]
    assert excluded and {item.held_out_role for item in excluded} == {
        HeldOutSampleRole.EXCLUDED
    }
    regular = compute_identity_token("mpdd:v1:fixture:1")
    roles = Counter(
        item.held_out_role
        for item in manifest.sample_assignments
        if item.identity_token == regular
    )
    assert roles == {HeldOutSampleRole.GALLERY: 1, HeldOutSampleRole.QUERY: 2}
    shared_sequence_roles = {
        item.held_out_role
        for item in manifest.sample_assignments
        if item.identity_token == regular
        and item.sequence_token
        == compute_sequence_token(
            "mpdd:v1:fixture:1:sequence:shared", regular
        )
    }
    assert len(shared_sequence_roles) == 1
    random_background = next(
        item for item in manifest.sample_assignments if item.source_variant == "random_background"
    )
    assert random_background.held_out_role is HeldOutSampleRole.CONTROL_ONLY

def test_historical_exposure_is_preserved_without_final_claim() -> None:
    manifest = _build()
    historical = next(
        item
        for item in manifest.identity_assignments
        if item.historical_maximum_exposure is ExposureStage.FINAL_TEST_SCORED
    )
    assert historical.home_fold is not None
    changed = copy.deepcopy(manifest.to_dict())
    changed["final_evaluation_permitted"] = True
    with pytest.raises(ValueError, match="cannot permit final evaluation"):
        DatasetStratifiedIdentityKFoldManifest.from_dict(changed)

def test_policy_and_fold_index_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least three"):
        DatasetStratifiedKFoldPolicy(fold_count=2)
    with pytest.raises(ValueError, match="distinct fold"):
        DatasetStratifiedKFoldPolicy(dev_offset=5)
    manifest = _build()
    with pytest.raises(ValueError, match="outside"):
        materialize_identity_fold(manifest, 5)

    changed = copy.deepcopy(manifest.to_dict())
    changed["dataset_fold_counts"]["dogfacenet224"]["folds"][0] = True
    with pytest.raises(TypeError, match="counts must be integers"):
        DatasetStratifiedIdentityKFoldManifest.from_dict(changed)

def test_persisted_manifest_bundle_detects_tamper(tmp_path: Path) -> None:
    manifest = _build()
    path = tmp_path / "kfold.json"
    path.write_text(json.dumps(dataset_stratified_kfold_bundle(manifest)), encoding="utf-8")
    assert read_dataset_stratified_identity_kfold(path) == manifest
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest"]["protocol_name"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle digest differs"):
        read_dataset_stratified_identity_kfold(path)

def test_cli_writes_manifest_and_requested_views_without_overwrite() -> None:
    cycle = _research_cycle()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        cycle_path = root / "cycle.json"
        output = root / "kfold.json"
        cycle_path.write_text(json.dumps(cycle.to_dict()), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "evaluation.commands.evaluate",
            "identity-kfold",
            "--protocol-name",
            "unified-afn-five-fold-v1",
            "--research-cycle",
            str(cycle_path),
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr
        bundle = json.loads(output.read_text(encoding="utf-8"))
        manifest = DatasetStratifiedIdentityKFoldManifest.from_dict(bundle["manifest"])
        assert json.loads(completed.stdout)["manifest_sha256"] == manifest.manifest_sha256
        assert bundle["manifest_sha256"] == content_sha256(bundle["manifest"])
        repeated = subprocess.run(command, capture_output=True, text=True, check=False)
        assert repeated.returncode != 0
        assert "refusing to overwrite" in repeated.stderr
