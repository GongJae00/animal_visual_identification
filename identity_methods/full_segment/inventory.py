"""Content-bound Full128 experiment inventory preparation."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from data_pipeline.source_lock import SOURCE_REGISTRY, get_record
from data_pipeline.types import DatasetAdmission
from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from identity_governance.full_split_census import (
    FullStatus,
    IdentityEvidenceKind,
    RegionStatus,
    TerminalRole,
    UnifiedFullObservation,
    ViewScope,
    allocate_unified_full_split,
    build_unified_full_census,
    unified_full_split_bundle,
    validate_unified_full_split_bundle,
)
from identity_methods.full_segment.manifests import build_baseline_family_manifest
from localization.full_segment_cache import validate_full_segment_cache_bundle
from localization.full_segment_contracts import FullSegmentObservation
from localization.full_segment_crop import verify_full_crop_artifacts

INVENTORY_SCHEMA = "cvi.full128_experiment_inventory.v2"
BUNDLE_SCHEMA = "cvi.full128_experiment_inventory_bundle.v2"
ADMISSION_STATE_SCHEMA = "cvi.full128_source_registry_admission_state.v1"

FULL128_REQUEST_ROW_FIELDS = frozenset(
    {
        "dataset_name",
        "dataset_version",
        "official_split",
        "identity_evidence_kind",
        "identity_namespace_uuid",
        "identity_token",
        "sample_token",
        "source_group",
        "capture_group",
        "sequence_group",
        "duplicate_component",
        "terminal_role",
        "original_source_sha256",
        "effective_source_sha256",
        "lineage_receipt_path",
        "full_segment_cache_path",
        "full_rgb_path",
        "full_mask_path",
    }
)

_INVENTORY_RECORD_FIELDS = {
    "dataset_name",
    "dataset_version",
    "dataset_admission",
    "official_split",
    "identity_evidence_kind",
    "identity_namespace_uuid",
    "identity_token",
    "sample_token",
    "source_group",
    "capture_group",
    "sequence_group",
    "duplicate_component",
    "gradient_eligible",
    "validation_only",
    "terminal_role",
    "source_observation_sha256",
    "original_source_sha256",
    "effective_source_sha256",
    "lineage_receipt_path",
    "lineage_receipt_file_sha256",
    "lineage_receipt_sha256",
    "view_scope",
    "route",
    "full_status",
    "face_status",
    "nose_status",
    "full_segment_cache_path",
    "full_segment_cache_file_sha256",
    "full_segment_cache_sha256",
    "full_rgb_path",
    "full_rgb_sha256",
    "full_mask_path",
    "full_mask_sha256",
    "crop_record_sha256",
    "crop_artifacts_present",
}

_ADMITTED_DATASET_STATES = {
    DatasetAdmission.ADMIT_TRAIN,
    DatasetAdmission.ADMIT_VALIDATION_ONLY,
    DatasetAdmission.ADMIT_TEACHER_ONLY,
}
_MAX_CACHE_BYTES = 536_870_912
_MAX_CROP_BYTES = 67_108_864
_VALIDATION_BATCHES_PER_WORKER = 8


@dataclass(frozen=True, slots=True)
class _PrevalidatedArtifact:
    sample_token: str
    sample_directory: Path
    observation: FullSegmentObservation
    full_segment_cache_file_sha256: str
    full_segment_cache_sha256: str
    lineage_receipt_file_sha256: str | None
    lineage_receipt_sha256: str | None
    crop_record_sha256: str | None
    full_rgb_sha256: str | None
    full_mask_sha256: str | None


@dataclass(frozen=True, slots=True)
class _PrevalidatedMaterialization:
    artifact_root: Path
    artifacts: tuple[_PrevalidatedArtifact, ...]


@dataclass(frozen=True, slots=True)
class _PreparedInventoryRow:
    row: dict[str, Any]
    expected: UnifiedFullObservation
    dataset: Any
    gradient_eligible: bool
    validation_only: bool
    cache_path: Path
    cache_relative: str
    rgb_path: Path
    rgb_relative: str
    mask_path: Path
    mask_relative: str


@dataclass(frozen=True, slots=True)
class _ArtifactValidationInput:
    root: Path
    row: Mapping[str, Any]
    cache_path: Path
    rgb_path: Path
    mask_path: Path


def build_full128_experiment_inventory(
    *,
    unified_full_split: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    validation_workers: int = 1,
) -> dict[str, Any]:
    """Verify materialized Full evidence and bind it to an allocated split."""

    return _build_full128_experiment_inventory(
        unified_full_split=unified_full_split,
        request_rows=request_rows,
        artifact_root=artifact_root,
        prevalidated=None,
        validation_workers=validation_workers,
    )


def _build_full128_experiment_inventory_from_prevalidated(
    *,
    unified_full_split: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    prevalidated: _PrevalidatedMaterialization,
) -> dict[str, Any]:
    """Build from evidence produced by the immediately preceding assembly pass."""

    if type(prevalidated) is not _PrevalidatedMaterialization or any(
        type(artifact) is not _PrevalidatedArtifact
        for artifact in prevalidated.artifacts
    ):
        raise TypeError("Full128 prevalidation must be typed assembly evidence")
    return _build_full128_experiment_inventory(
        unified_full_split=unified_full_split,
        request_rows=request_rows,
        artifact_root=artifact_root,
        prevalidated=prevalidated,
        validation_workers=1,
    )


def _build_full128_experiment_inventory(
    *,
    unified_full_split: Mapping[str, Any],
    request_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    prevalidated: _PrevalidatedMaterialization | None,
    validation_workers: int,
) -> dict[str, Any]:

    workers = _validated_worker_count(validation_workers)
    source_manifest, _ = validate_unified_full_split_bundle(unified_full_split)
    root = _validated_artifact_root(artifact_root)
    prevalidated_by_sample: dict[str, _PrevalidatedArtifact] | None = None
    if prevalidated is not None:
        if prevalidated.artifact_root != root:
            raise ValueError("Full128 prevalidation artifact root differs")
        prevalidated_by_sample = {
            artifact.sample_token: artifact for artifact in prevalidated.artifacts
        }
        if len(prevalidated_by_sample) != len(prevalidated.artifacts):
            raise ValueError("Full128 prevalidation repeats a sample token")
    rows = tuple(request_rows)
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Full128 request rows must be a non-empty array of objects")

    source_by_sample = {
        observation.sample_token: observation
        for observation in source_manifest.observations
    }
    if len(rows) != len(source_by_sample):
        raise ValueError(
            "Full128 request rows must cover the unified Full split exactly"
        )

    prepared_rows: list[_PreparedInventoryRow] = []
    seen: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        _exact_keys(row, FULL128_REQUEST_ROW_FIELDS, "Full128 request row")
        sample_token = row["sample_token"]
        if not isinstance(sample_token, str) or sample_token in seen:
            raise ValueError("Full128 request rows repeat or omit a sample token")
        seen.add(sample_token)
        try:
            expected = source_by_sample[sample_token]
        except KeyError:
            raise ValueError(
                "Full128 request sample is absent from the unified Full split"
            ) from None

        dataset = _admitted_dataset(row, expected)
        _validate_caller_metadata(row, expected)
        gradient_eligible, validation_only = _admission_flags(
            dataset.admission, expected
        )

        cache_path, cache_relative = _artifact_path(
            root,
            row["full_segment_cache_path"],
            filename="full-segment-cache.json",
            label="Full segment cache",
        )
        rgb_path, rgb_relative = _artifact_path(
            root, row["full_rgb_path"], filename="full.png", label="Full RGB"
        )
        mask_path, mask_relative = _artifact_path(
            root,
            row["full_mask_path"],
            filename="full-mask.png",
            label="Full mask",
        )
        if len({cache_path.parent, rgb_path.parent, mask_path.parent}) != 1:
            raise ValueError(
                "Full128 materialization paths must share one sample directory"
            )
        prepared_rows.append(
            _PreparedInventoryRow(
                row=row,
                expected=expected,
                dataset=dataset,
                gradient_eligible=gradient_eligible,
                validation_only=validation_only,
                cache_path=cache_path,
                cache_relative=cache_relative,
                rgb_path=rgb_path,
                rgb_relative=rgb_relative,
                mask_path=mask_path,
                mask_relative=mask_relative,
            )
        )

    if seen != set(source_by_sample):
        raise ValueError(
            "Full128 request rows must cover the unified Full split exactly"
        )
    if prevalidated_by_sample is None:
        artifacts = _deep_validate_artifacts(
            root=root,
            prepared_rows=prepared_rows,
            validation_workers=workers,
        )
    else:
        resolved_artifacts: list[_PrevalidatedArtifact] = []
        for prepared in prepared_rows:
            row = prepared.row
            sample_token = row["sample_token"]
            try:
                artifact = prevalidated_by_sample[sample_token]
            except KeyError:
                raise ValueError(
                    "Full128 prevalidation must cover the unified Full split exactly"
                ) from None
            _validate_prevalidated_artifact(
                artifact,
                row=row,
                cache_path=prepared.cache_path,
                rgb_path=prepared.rgb_path,
                mask_path=prepared.mask_path,
            )
            resolved_artifacts.append(artifact)
        if seen != set(prevalidated_by_sample):
            raise ValueError(
                "Full128 prevalidation must cover the unified Full split exactly"
            )
        artifacts = tuple(resolved_artifacts)

    observations: list[UnifiedFullObservation] = []
    inventory_records: list[dict[str, Any]] = []
    for prepared, artifact in zip(prepared_rows, artifacts, strict=True):
        row = prepared.row
        expected = prepared.expected
        dataset = prepared.dataset
        sample_token = row["sample_token"]
        observation = artifact.observation
        lineage_relative = None
        if row["lineage_receipt_path"] is not None:
            _, lineage_relative = _artifact_path(
                root,
                row["lineage_receipt_path"],
                filename="lineage-receipt.json",
                label="Full128 derived lineage receipt",
            )
        artifact_present = artifact.crop_record_sha256 is not None

        rebuilt = UnifiedFullObservation(
            dataset_name=dataset.canonical_name,
            official_split=row["official_split"],
            identity_evidence_kind=IdentityEvidenceKind(row["identity_evidence_kind"]),
            identity_namespace_uuid=row["identity_namespace_uuid"],
            identity_token=row["identity_token"],
            sample_token=sample_token,
            source_group=row["source_group"],
            capture_group=row["capture_group"],
            sequence_group=row["sequence_group"],
            duplicate_component=row["duplicate_component"],
            gradient_eligible=prepared.gradient_eligible,
            validation_only=prepared.validation_only,
            full_status=FullStatus(observation.full_status.value),
            face_status=RegionStatus(observation.face_observability.value),
            nose_status=RegionStatus(observation.nose_observability.value),
            view_scope=ViewScope(observation.source_view_scope.value),
            source_observation_sha256=observation.observation_sha256,
            terminal_role=TerminalRole(row["terminal_role"]),
        )
        if rebuilt != expected:
            raise ValueError(
                "Full128 cache, admission, or request metadata differs from the "
                "validated unified Full split"
            )
        observations.append(rebuilt)
        inventory_records.append(
            {
                "dataset_name": dataset.canonical_name,
                "dataset_version": dataset.version,
                "dataset_admission": dataset.admission.value,
                "official_split": rebuilt.official_split,
                "identity_evidence_kind": rebuilt.identity_evidence_kind.value,
                "identity_namespace_uuid": rebuilt.identity_namespace_uuid,
                "identity_token": rebuilt.identity_token,
                "sample_token": rebuilt.sample_token,
                "source_group": rebuilt.source_group,
                "capture_group": rebuilt.capture_group,
                "sequence_group": rebuilt.sequence_group,
                "duplicate_component": rebuilt.duplicate_component,
                "gradient_eligible": rebuilt.gradient_eligible,
                "validation_only": rebuilt.validation_only,
                "terminal_role": rebuilt.terminal_role.value,
                "source_observation_sha256": observation.observation_sha256,
                "original_source_sha256": row["original_source_sha256"],
                "effective_source_sha256": observation.source_sha256,
                "lineage_receipt_path": lineage_relative,
                "lineage_receipt_file_sha256": artifact.lineage_receipt_file_sha256,
                "lineage_receipt_sha256": artifact.lineage_receipt_sha256,
                "view_scope": observation.source_view_scope.value,
                "route": observation.route.value,
                "full_status": observation.full_status.value,
                "face_status": observation.face_observability.value,
                "nose_status": observation.nose_observability.value,
                "full_segment_cache_path": prepared.cache_relative,
                "full_segment_cache_file_sha256": (
                    artifact.full_segment_cache_file_sha256
                ),
                "full_segment_cache_sha256": artifact.full_segment_cache_sha256,
                "full_rgb_path": prepared.rgb_relative,
                "full_rgb_sha256": artifact.full_rgb_sha256,
                "full_mask_path": prepared.mask_relative,
                "full_mask_sha256": artifact.full_mask_sha256,
                "crop_record_sha256": artifact.crop_record_sha256,
                "crop_artifacts_present": artifact_present,
            }
        )

    rebuilt_manifest = allocate_unified_full_split(
        allocation_name=source_manifest.allocation_name,
        observations=observations,
        policy=source_manifest.policy,
    )
    if rebuilt_manifest != source_manifest:
        raise ValueError("Full128 rebuilt allocation differs from the validated split")
    census = build_unified_full_census(rebuilt_manifest)
    split_bundle = unified_full_split_bundle(rebuilt_manifest, census)

    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "records": sorted(inventory_records, key=lambda item: item["sample_token"]),
    }
    baseline_family = build_baseline_family_manifest()
    admission_state = _source_registry_admission_state()
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "artifact_root": os.fspath(root),
        "content_kind": "METADATA_ONLY",
        "source_registry_admission_state": admission_state,
        "source_registry_admission_sha256": content_sha256(admission_state),
        "baseline_family_manifest": baseline_family,
        "baseline_family_sha256": content_sha256(baseline_family),
        "split_manifest_sha256": rebuilt_manifest.manifest_sha256,
        "split_census_sha256": census.census_sha256,
        "split_bundle": split_bundle,
        "inventory_sha256": content_sha256(inventory),
        "inventory": inventory,
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_full128_experiment_inventory_bundle(
    value: object,
    *,
    validation_workers: int = 1,
) -> dict[str, Any]:
    """Re-verify a Full128 bundle against current contracts and artifact bytes."""

    workers = _validated_worker_count(validation_workers)
    expected = {
        "schema_version",
        "artifact_root",
        "content_kind",
        "source_registry_admission_state",
        "source_registry_admission_sha256",
        "baseline_family_manifest",
        "baseline_family_sha256",
        "split_manifest_sha256",
        "split_census_sha256",
        "split_bundle",
        "inventory_sha256",
        "inventory",
        "bundle_sha256",
    }
    _exact_keys(value, expected, "Full128 experiment inventory bundle")
    bundle = dict(value)
    if bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("Full128 experiment inventory bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("Full128 experiment inventory bundle digest differs")
    inventory = bundle["inventory"]
    _exact_keys(inventory, {"schema_version", "records"}, "Full128 inventory")
    if inventory["schema_version"] != INVENTORY_SCHEMA or not isinstance(
        inventory["records"], list
    ):
        raise ValueError("Full128 inventory schema differs")
    for record in inventory["records"]:
        _exact_keys(record, _INVENTORY_RECORD_FIELDS, "Full128 inventory record")
    rows = [
        {
            "dataset_name": record["dataset_name"],
            "dataset_version": record["dataset_version"],
            "official_split": record["official_split"],
            "identity_evidence_kind": record["identity_evidence_kind"],
            "identity_namespace_uuid": record["identity_namespace_uuid"],
            "identity_token": record["identity_token"],
            "sample_token": record["sample_token"],
            "source_group": record["source_group"],
            "capture_group": record["capture_group"],
            "sequence_group": record["sequence_group"],
            "duplicate_component": record["duplicate_component"],
            "terminal_role": record["terminal_role"],
            "original_source_sha256": record["original_source_sha256"],
            "effective_source_sha256": record["effective_source_sha256"],
            "lineage_receipt_path": (
                None
                if record["lineage_receipt_path"] is None
                else os.fspath(
                    Path(bundle["artifact_root"]) / record["lineage_receipt_path"]
                )
            ),
            "full_segment_cache_path": os.fspath(
                Path(bundle["artifact_root"]) / record["full_segment_cache_path"]
            ),
            "full_rgb_path": os.fspath(
                Path(bundle["artifact_root"]) / record["full_rgb_path"]
            ),
            "full_mask_path": os.fspath(
                Path(bundle["artifact_root"]) / record["full_mask_path"]
            ),
        }
        for record in inventory["records"]
    ]
    rebuilt = build_full128_experiment_inventory(
        unified_full_split=bundle["split_bundle"],
        request_rows=rows,
        artifact_root=Path(bundle["artifact_root"]),
        validation_workers=workers,
    )
    if rebuilt != bundle:
        raise ValueError("Full128 experiment inventory bundle content differs")
    return bundle


def _deep_validate_artifacts(
    *,
    root: Path,
    prepared_rows: Sequence[_PreparedInventoryRow],
    validation_workers: int,
) -> tuple[_PrevalidatedArtifact, ...]:
    inputs = (
        _ArtifactValidationInput(
            root=root,
            row=prepared.row,
            cache_path=prepared.cache_path,
            rgb_path=prepared.rgb_path,
            mask_path=prepared.mask_path,
        )
        for prepared in prepared_rows
    )
    if validation_workers == 1:
        return tuple(_deep_validate_artifact_input(item) for item in inputs)

    artifacts: list[_PrevalidatedArtifact] = []
    batch_size = validation_workers * _VALIDATION_BATCHES_PER_WORKER
    with ThreadPoolExecutor(max_workers=validation_workers) as executor:
        while batch := tuple(islice(inputs, batch_size)):
            artifacts.extend(executor.map(_deep_validate_artifact_input, batch))
    return tuple(artifacts)


def _deep_validate_artifact_input(
    value: _ArtifactValidationInput,
) -> _PrevalidatedArtifact:
    return _deep_validate_artifact(
        root=value.root,
        row=value.row,
        cache_path=value.cache_path,
        rgb_path=value.rgb_path,
        mask_path=value.mask_path,
    )


def _deep_validate_artifact(
    *,
    root: Path,
    row: Mapping[str, Any],
    cache_path: Path,
    rgb_path: Path,
    mask_path: Path,
) -> _PrevalidatedArtifact:
    cache_document = read_strict_json_document(
        _bound_existing_file(root, cache_path, "Full segment cache"),
        maximum_bytes=_MAX_CACHE_BYTES,
        maximum_string_characters=_MAX_CACHE_BYTES,
    )
    cache = validate_full_segment_cache_bundle(cache_document.payload)
    records = cache["records"]
    if len(records) != 1:
        raise ValueError(
            "Full128 materialization cache must contain exactly one record"
        )
    cache_record = records[0]
    observation = FullSegmentObservation.from_dict(cache_record["observation"])
    sample_token = row["sample_token"]
    if (
        cache_record["source_id"] != sample_token
        or observation.source_id != sample_token
    ):
        raise ValueError("Full128 cache source ID and sample token differ")
    for field in ("original_source_sha256", "effective_source_sha256"):
        _require_sha256(row[field], f"Full128 {field}")
    if row["effective_source_sha256"] != observation.source_sha256:
        raise ValueError(
            "Full128 effective source digest differs from the materialized cache"
        )

    lineage_file_sha256 = None
    lineage_sha256 = None
    lineage_value = row["lineage_receipt_path"]
    if lineage_value is None:
        if row["original_source_sha256"] != row["effective_source_sha256"]:
            raise ValueError(
                "Full128 non-derived row must retain its original source digest"
            )
    else:
        lineage_path, _ = _artifact_path(
            root,
            lineage_value,
            filename="lineage-receipt.json",
            label="Full128 derived lineage receipt",
        )
        if lineage_path.parent != cache_path.parent:
            raise ValueError(
                "Full128 lineage and materialization paths must share one sample directory"
            )
        lineage_document = read_strict_json_document(
            _bound_existing_file(root, lineage_path, "Full128 derived lineage receipt"),
            maximum_bytes=1_048_576,
        )
        from identity_methods.full_segment.materialization import (
            validate_derived_lineage_receipt,
        )

        lineage = validate_derived_lineage_receipt(
            lineage_document.payload,
            expected_parent_sha256=row["original_source_sha256"],
            expected_child_sha256=row["effective_source_sha256"],
        )
        if lineage["sample_token"] != sample_token or (
            lineage["child_width"],
            lineage["child_height"],
        ) != (observation.source_width, observation.source_height):
            raise ValueError(
                "Full128 derived lineage sample or dimensions differ from cache"
            )
        lineage_file_sha256 = lineage_document.raw_sha256
        lineage_sha256 = lineage["lineage_sha256"]

    crop = cache_record["crop"]
    if observation.full_status.value in {"USABLE", "REVIEW"}:
        if crop is None:
            raise ValueError("usable or review Full128 row requires a crop record")
        rgb_bytes = _read_bound_file(root, rgb_path, "Full RGB", _MAX_CROP_BYTES)
        mask_bytes = _read_bound_file(root, mask_path, "Full mask", _MAX_CROP_BYTES)
        verify_full_crop_artifacts(crop, rgb_bytes, mask_bytes)
    else:
        if crop is not None:
            raise ValueError("unobservable Full128 row cannot retain a crop record")
        if any(path.exists() or path.is_symlink() for path in (rgb_path, mask_path)):
            raise ValueError("unobservable Full128 row cannot retain crop artifacts")
    return _PrevalidatedArtifact(
        sample_token=sample_token,
        sample_directory=cache_path.parent,
        observation=observation,
        full_segment_cache_file_sha256=cache_document.raw_sha256,
        full_segment_cache_sha256=cache_document.payload["cache_sha256"],
        lineage_receipt_file_sha256=lineage_file_sha256,
        lineage_receipt_sha256=lineage_sha256,
        crop_record_sha256=None if crop is None else crop["crop_record_sha256"],
        full_rgb_sha256=None if crop is None else crop["full_rgb_sha256"],
        full_mask_sha256=None if crop is None else crop["full_mask_sha256"],
    )


def _validate_prevalidated_artifact(
    artifact: _PrevalidatedArtifact,
    *,
    row: Mapping[str, Any],
    cache_path: Path,
    rgb_path: Path,
    mask_path: Path,
) -> None:
    sample_token = row["sample_token"]
    if (
        artifact.sample_token != sample_token
        or artifact.sample_directory != cache_path.parent
        or rgb_path.parent != artifact.sample_directory
        or mask_path.parent != artifact.sample_directory
        or artifact.observation.source_id != sample_token
    ):
        raise ValueError("Full128 prevalidation artifact binding differs")
    for field in ("original_source_sha256", "effective_source_sha256"):
        _require_sha256(row[field], f"Full128 {field}")
    if row["effective_source_sha256"] != artifact.observation.source_sha256:
        raise ValueError(
            "Full128 effective source digest differs from the materialized cache"
        )
    for value, label in (
        (artifact.full_segment_cache_file_sha256, "Full128 cache file"),
        (artifact.full_segment_cache_sha256, "Full128 cache"),
    ):
        _require_sha256(value, label)

    lineage_values = (
        artifact.lineage_receipt_file_sha256,
        artifact.lineage_receipt_sha256,
    )
    if row["lineage_receipt_path"] is None:
        if any(value is not None for value in lineage_values):
            raise ValueError("Full128 non-derived prevalidation retains lineage")
        if row["original_source_sha256"] != row["effective_source_sha256"]:
            raise ValueError(
                "Full128 non-derived row must retain its original source digest"
            )
    else:
        lineage_path = Path(row["lineage_receipt_path"])
        if lineage_path.parent != artifact.sample_directory:
            raise ValueError(
                "Full128 lineage and materialization paths must share one sample directory"
            )
        for value, label in zip(
            lineage_values,
            ("Full128 lineage receipt file", "Full128 lineage receipt"),
            strict=True,
        ):
            _require_sha256(value, label)

    crop_values = (
        artifact.crop_record_sha256,
        artifact.full_rgb_sha256,
        artifact.full_mask_sha256,
    )
    observable = artifact.observation.full_status.value in {"USABLE", "REVIEW"}
    if observable:
        for value, label in zip(
            crop_values,
            ("Full128 crop record", "Full128 RGB", "Full128 mask"),
            strict=True,
        ):
            _require_sha256(value, label)
    elif any(value is not None for value in crop_values):
        raise ValueError("unobservable Full128 prevalidation retains crop artifacts")


def _admitted_dataset(row: Mapping[str, Any], expected: UnifiedFullObservation) -> Any:
    dataset_name = row["dataset_name"]
    if not isinstance(dataset_name, str) or dataset_name != expected.dataset_name:
        raise ValueError("Full128 dataset name differs from the unified Full split")
    try:
        dataset = get_record(dataset_name)
    except KeyError:
        raise ValueError("Full128 dataset is absent from source_lock") from None
    if dataset.admission not in _ADMITTED_DATASET_STATES:
        raise ValueError(f"Full128 dataset is blocked by source_lock: {dataset_name}")
    if row["dataset_version"] != dataset.version:
        raise ValueError("Full128 dataset version differs from source_lock")
    return dataset


def _admission_flags(
    admission: DatasetAdmission,
    expected: UnifiedFullObservation,
) -> tuple[bool, bool]:
    if admission is DatasetAdmission.ADMIT_TRAIN:
        if expected.validation_only:
            raise ValueError(
                "train-admitted Full128 row cannot claim validation-only status"
            )
        return expected.gradient_eligible, False
    required = {
        DatasetAdmission.ADMIT_VALIDATION_ONLY: (False, True),
        DatasetAdmission.ADMIT_TEACHER_ONLY: (False, False),
    }[admission]
    if (expected.gradient_eligible, expected.validation_only) != required:
        raise ValueError("Full128 row eligibility differs from source admission")
    return required


def _validate_caller_metadata(
    row: Mapping[str, Any], expected: UnifiedFullObservation
) -> None:
    expected_values = {
        "official_split": expected.official_split,
        "identity_evidence_kind": expected.identity_evidence_kind.value,
        "identity_namespace_uuid": expected.identity_namespace_uuid,
        "identity_token": expected.identity_token,
        "source_group": expected.source_group,
        "capture_group": expected.capture_group,
        "sequence_group": expected.sequence_group,
        "duplicate_component": expected.duplicate_component,
        "terminal_role": expected.terminal_role.value,
    }
    for field, value in expected_values.items():
        if row[field] != value:
            label = (
                "identity UUID namespace"
                if field == "identity_namespace_uuid"
                else field.replace("_", " ")
            )
            raise ValueError(
                f"Full128 request {label} differs from the unified Full split"
            )


def _source_registry_admission_state() -> dict[str, Any]:
    return {
        "schema_version": ADMISSION_STATE_SCHEMA,
        "datasets": [
            {
                "canonical_name": record.canonical_name,
                "version": record.version,
                "admission": record.admission.value,
            }
            for record in sorted(SOURCE_REGISTRY, key=lambda item: item.canonical_name)
        ],
    }


def _validated_artifact_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError("Full128 artifact root must be an absolute path")
    if value.is_symlink():
        raise ValueError("Full128 artifact root must not be a symlink")
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Full128 artifact root must exist") from exc
    if resolved != value or not resolved.is_dir():
        raise ValueError(
            "Full128 artifact root must be a canonical non-symlink directory"
        )
    return resolved


def _artifact_path(
    root: Path, value: object, *, filename: str, label: str
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be non-empty text")
    path = Path(value)
    if not path.is_absolute() or os.fspath(path) != value or path.name != filename:
        raise ValueError(f"{label} path must be a canonical absolute {filename} path")
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError(f"{label} path must be under the artifact root") from None
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} path must be safe beneath the artifact root")
    return path, relative.as_posix()


def _bound_existing_file(root: Path, path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if (
        resolved != path
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not resolved.is_file()
    ):
        raise ValueError(
            f"{label} must be a regular non-symlink file under artifact root"
        )
    return resolved


def _read_bound_file(root: Path, path: Path, label: str, maximum: int) -> bytes:
    bound = _bound_existing_file(root, path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(bound, flags)
    chunks: list[bytes] = []
    observed = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError(f"{label} size or file type differs")
        while chunk := os.read(descriptor, min(1_048_576, maximum + 1 - observed)):
            observed += len(chunk)
            if observed > maximum:
                raise ValueError(f"{label} exceeds byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or observed != before.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return b"".join(chunks)


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _validated_worker_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Full128 validation workers must be an integer")
    if value <= 0:
        raise ValueError("Full128 validation workers must be positive")
    return value


def _exact_keys(value: object, expected: set[str] | frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} fields differ")


__all__ = [
    "ADMISSION_STATE_SCHEMA",
    "BUNDLE_SCHEMA",
    "FULL128_REQUEST_ROW_FIELDS",
    "INVENTORY_SCHEMA",
    "build_full128_experiment_inventory",
    "validate_full128_experiment_inventory_bundle",
]
