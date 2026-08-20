from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from data.types import UnifiedCanidSample
from evaluation.localization_kfold import (
    LocalizationKFoldManifest,
    LocalizationKFoldPolicy,
    build_localization_kfold_manifest,
    build_localization_source_manifest,
    localization_kfold_bundle,
    materialize_localization_fold,
    read_localization_kfold,
)
from shared.foundation.provenance import content_sha256
from enrollment.registry.identity_registry import compute_sample_token
from evaluation import localization_kfold_cli as localization_kfold_workflow

def _sha(value: str) -> str:
    return content_sha256({"fixture": value})

def _sample(
    dataset: str,
    group: int,
    instance: int,
    *,
    image_sha256: str | None = None,
) -> UnifiedCanidSample:
    return UnifiedCanidSample(
        sample_id=compute_sample_token(f"{dataset}:{group}:{instance}"),
        dataset_name=dataset,
        dataset_version="fixture-v1",
        source_group_id=f"group:{group}",
        image_path=f"{dataset}/{group}.png",
        image_sha256=image_sha256 or _sha(f"{dataset}:image:{group}"),
        width=64,
        height=48,
        split_role=("train", "val", "test")[group % 3],
    )

def _samples() -> tuple[UnifiedCanidSample, ...]:
    shared = _sha("cross-dataset-exact-image")
    values: list[UnifiedCanidSample] = []
    for group in range(10):
        for instance in range(2):
            values.append(
                _sample(
                    "ap10k-dog",
                    group,
                    instance,
                    image_sha256=shared if group == 0 else None,
                )
            )
        values.append(
            _sample(
                "dogflw",
                group,
                0,
                image_sha256=shared if group == 0 else None,
            )
        )
    return tuple(values)

def _build() -> LocalizationKFoldManifest:
    samples = _samples()
    return build_localization_kfold_manifest(
        samples,
        protocol_name="unified-localization-five-fold-v1",
        policy=LocalizationKFoldPolicy(),
        source_manifests={
            dataset: build_localization_source_manifest(samples, dataset=dataset)
            for dataset in ("ap10k-dog", "dogflw")
        },
    )

def test_manifest_is_deterministic_round_trippable_and_balanced_by_source_unit() -> None:
    first = _build()
    second = _build()
    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert LocalizationKFoldManifest.from_dict(first.to_dict()) == first
    counts = dict(first.dataset_unit_counts)
    assert set(counts) == {"ap10k-dog", "dogflw"}
    assert all(len(values) == 5 and min(values) >= 1 for values in counts.values())
    assert first.identity_target_mode == "NONE"
    assert first.final_evaluation_permitted is False

def test_ap10k_instances_source_groups_and_exact_cross_dataset_images_stay_together() -> None:
    manifest = _build()
    fold_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)
    fold_by_image: dict[str, set[int]] = defaultdict(set)
    sample_lookup = {sample.sample_id: sample for sample in _samples()}
    for item in manifest.assignments:
        sample = sample_lookup[item.sample_id]
        fold_by_group[(sample.dataset_name, sample.source_group_id)].add(item.home_fold)
        fold_by_image[item.image_sha256].add(item.home_fold)
    assert all(len(folds) == 1 for folds in fold_by_group.values())
    assert all(len(folds) == 1 for folds in fold_by_image.values())

def test_every_localization_sample_rotates_test_once_dev_once_train_three_times() -> None:
    manifest = _build()
    observed: dict[str, Counter[str]] = {
        item.sample_id: Counter() for item in manifest.assignments
    }
    for fold in range(5):
        bundle = materialize_localization_fold(manifest, fold)
        assert bundle["view_sha256"] == content_sha256(bundle["view"])
        for row in bundle["view"]["assignments"]:
            observed[row["sample_id"]][row["stage"]] += 1
            assert row["identity_target_mode"] == "NONE"
    assert all(counts == {"TRAIN": 3, "DEV": 1, "TEST": 1} for counts in observed.values())

def test_identity_targets_and_final_interpretation_fail_closed() -> None:
    samples = list(_samples())
    object.__setattr__(samples[0], "raw_identity_id", "dog-1")
    with pytest.raises(ValueError, match="must not carry identity targets"):
        build_localization_kfold_manifest(
            tuple(samples),
            protocol_name="fixture",
            policy=LocalizationKFoldPolicy(),
            source_manifests={
                dataset: build_localization_source_manifest(
                    tuple(samples), dataset=dataset
                )
                for dataset in ("ap10k-dog", "dogflw")
            },
        )
    manifest = _build()
    changed = copy.deepcopy(manifest.to_dict())
    changed["final_evaluation_permitted"] = True
    with pytest.raises(ValueError, match="cannot use scores or permit final"):
        LocalizationKFoldManifest.from_dict(changed)

def test_live_adapter_projection_must_match_admitted_source_manifest() -> None:
    samples = _samples()
    source_manifests = {
        dataset: build_localization_source_manifest(samples, dataset=dataset)
        for dataset in ("ap10k-dog", "dogflw")
    }
    changed = (replace(samples[0], image_sha256=_sha("changed-live-image")), *samples[1:])
    with pytest.raises(ValueError, match="live adapter projection differs"):
        build_localization_kfold_manifest(
            changed,
            protocol_name="fixture",
            policy=LocalizationKFoldPolicy(),
            source_manifests=source_manifests,
        )

def test_insufficient_dataset_units_and_invalid_fold_index_fail() -> None:
    sparse = tuple(
        sample
        for sample in _samples()
        if int(sample.source_group_id.split(":")[1]) < 4
    )
    with pytest.raises(ValueError, match="at least one unit per fold"):
        build_localization_kfold_manifest(
            sparse,
            protocol_name="fixture",
            policy=LocalizationKFoldPolicy(),
            source_manifests={
                dataset: build_localization_source_manifest(sparse, dataset=dataset)
                for dataset in ("ap10k-dog", "dogflw")
            },
        )
    with pytest.raises(ValueError, match="outside policy"):
        materialize_localization_fold(_build(), 5)

def test_persisted_bundle_count_types_and_duplicate_bindings_fail_closed(
    tmp_path,
) -> None:
    manifest = _build()
    path = tmp_path / "localization-kfold.json"
    path.write_text(json.dumps(localization_kfold_bundle(manifest)), encoding="utf-8")
    assert read_localization_kfold(path) == manifest

    tampered = localization_kfold_bundle(manifest)
    tampered["manifest"]["protocol_name"] = "tampered"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle digest differs"):
        read_localization_kfold(path)

    changed = copy.deepcopy(manifest.to_dict())
    changed["dataset_unit_counts"]["ap10k-dog"][0] = True
    with pytest.raises(TypeError, match="integer fold arrays"):
        LocalizationKFoldManifest.from_dict(changed)

    with pytest.raises(ValueError, match="bindings differ"):
        replace(
            manifest,
            source_manifest_sha256s=(
                ("ap10k-dog", _sha("one")),
                ("ap10k-dog", _sha("two")),
                ("dogflw", _sha("three")),
            ),
        )

def test_cli_dispatches_only_explicit_source_manifest_route(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        localization_kfold_workflow.sys,
        "argv",
        ["build_localization_kfold.py", "--help"],
    )
    with pytest.raises(SystemExit) as kfold_exit:
        localization_kfold_workflow.main()
    assert kfold_exit.value.code == 0
    kfold_help = capsys.readouterr().out
    assert "--protocol-name" in kfold_help
    assert "--fold-count" in kfold_help
    assert "--dataset" not in kfold_help

    monkeypatch.setattr(
        localization_kfold_workflow.sys,
        "argv",
        ["build_localization_kfold.py", "source-manifest", "--help"],
    )
    with pytest.raises(SystemExit) as source_exit:
        localization_kfold_workflow.main()
    assert source_exit.value.code == 0
    source_help = capsys.readouterr().out
    assert "--dataset {ap10k-dog,dogflw}" in source_help
    assert "--output OUTPUT" in source_help
    assert "--protocol-name" not in source_help

def test_source_manifest_route_matches_projection_and_refuses_overwrite(
    tmp_path, monkeypatch, capsys
) -> None:
    samples = tuple(
        sample for sample in _samples() if sample.dataset_name == "dogflw"
    )
    expected = build_localization_source_manifest(samples, dataset="dogflw")
    monkeypatch.setitem(
        localization_kfold_workflow.ADAPTERS,
        "dogflw",
        lambda _root: samples,
    )
    output = tmp_path / "dogflw-source-manifest.json"
    monkeypatch.setattr(
        localization_kfold_workflow.sys,
        "argv",
        [
            "build_localization_kfold.py",
            "source-manifest",
            "--dataset",
            "dogflw",
            "--output",
            str(output),
        ],
    )

    assert localization_kfold_workflow.main() == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == {
        "status": "CREATED_LOCALIZATION_SOURCE_MANIFEST",
        "dataset_name": "dogflw",
        "manifest_sha256": content_sha256(expected),
        "sample_count": len(expected["records"]),
        "output": str(output),
    }

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite localization source manifest",
    ):
        localization_kfold_workflow.main()
