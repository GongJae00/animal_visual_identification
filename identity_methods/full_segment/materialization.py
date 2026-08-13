"""Resumable, content-bound Full128 route-plan materialization and assembly."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import io
import math
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image, UnidentifiedImageError

from artifact_contracts.animal_parsing_runtime import (
    SUPPORTED_BUNDLE_SCHEMAS as SUPPORTED_PARSING_BUNDLE_SCHEMAS,
)
from artifact_contracts.animal_parsing_runtime import (
    AnimalParsingRuntimeManifest,
)
from artifact_contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
)
from artifact_contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)
from data_pipeline.source_lock import get_record
from data_pipeline.types import DatasetAdmission
from foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
)
from foundation.protected_publication import (
    fsync_directory,
    rename_directory_noreplace,
)
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
)
from identity_governance.generated_identity_registry import (
    GENERATED_DOG_NAMESPACE,
    compute_generated_identity_id,
    compute_source_cluster_token,
)
from identity_governance.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_registered_dog_id,
)
from identity_methods.full_segment.route_plan import (
    RouteIntent,
    validate_full128_route_plan_bundle,
)
from localization.animal_instance_segmentation import (
    AnimalInstanceSegmentationRuntime,
)
from localization.animal_parsing import (
    AnimalParsingPolicy,
    AnimalParsingPrediction,
    AnimalParsingRuntime,
)
from localization.foreground_segmentation import ForegroundSegmentationRuntime
from localization.full_segment_cache import (
    CACHE_SCHEMA,
    FROZEN_PARSING_BINDING_SCHEMA,
    FROZEN_PARSING_SCHEMA,
    LEGACY_CACHE_SCHEMA,
    build_full_segment_cache,
    freeze_animal_parsing_prediction,
    thaw_animal_parsing_prediction,
    validate_full_segment_cache_bundle,
)
from localization.full_segment_contracts import (
    AnimalAssociation,
    AssociationKind,
    BodyMaskPolicy,
    BodyMaskPolicyKind,
    FullSegmentObservation,
    ObservationRoute,
    SourceViewScope,
)
from localization.full_segment_crop import verify_full_crop_artifacts

PARSER_CACHE_RECEIPT_SCHEMA = "cvi.full128_parser_cache_receipt.v1"
PARSER_RUNTIME_BINDING_SCHEMA = "cvi.full128_parser_runtime_binding.v2"
ASSOCIATION_AUTHORITY_SCHEMA = "cvi.full128_parser_association_authority.v1"
POLICY_SELECTION_AUTHORITY_SCHEMA = (
    "cvi.full128_parser_policy_selection_authority.v1"
)
SELECTION_LINEAGE_SCHEMA = "cvi.full128_parser_selection_lineage.v1"
DERIVED_LINEAGE_SCHEMA = "cvi.full128_derived_source_lineage.v1"
EXECUTION_RECEIPT_SCHEMA = "cvi.full128_sample_execution_receipt.v1"
SHARD_SELECTION_SCHEMA = "cvi.full128_shard_selection.v1"
ASSEMBLY_SCHEMA = "cvi.full128_materialization_assembly.v1"
INVENTORY_REQUEST_SCHEMA = "cvi.full128_experiment_inventory_request.v2"

_MAX_SOURCE_BYTES = 67_108_864
_MAX_SOURCE_PIXELS = 33_554_432
_MAX_JSON_BYTES = 536_870_912
_MAX_PLAN_BYTES = 2_147_483_648
_IOU_THRESHOLD = 0.5
_RUNTIME_FIELDS = {
    "schema_version",
    "parser_runtime_manifest_sha256",
    "parser_runtime_bundle_raw_sha256",
    "parser_policy_sha256",
    "foreground_model_manifest_sha256",
    "foreground_model_bundle_raw_sha256",
    "instance_model_manifest_sha256",
    "instance_model_bundle_raw_sha256",
    "device",
    "job_batch_size",
    "instance_batch_size",
    "foreground_batch_size",
    "publication_workers",
    "shape_policy",
    "oom_policy",
}
_PARSER_RECEIPT_FIELDS = {
    "schema_version",
    "parser_cache_key",
    "source_sha256",
    "source_width",
    "source_height",
    "runtime",
    "prediction_sha256",
    "frozen_json_sha256",
    "receipt_sha256",
}
_LINEAGE_FIELDS = {
    "schema_version",
    "sample_token",
    "route_intent",
    "parent_source_sha256",
    "evidence_artifact",
    "bbox_xyxy",
    "aligned_bbox_xyxy",
    "crop_policy",
    "child_encoding",
    "child_sha256",
    "child_width",
    "child_height",
    "lineage_sha256",
}
_EXECUTION_FIELDS = {
    "schema_version",
    "sample_token",
    "plan_record_sha256",
    "plan_sha256",
    "original_source_sha256",
    "effective_source_sha256",
    "route_intent",
    "actual_route",
    "parser_lineage",
    "derived_lineage",
    "terminal_reason",
    "shard_selection",
    "outputs",
    "receipt_sha256",
}
_OUTPUT_FIELDS = {
    "full_segment_cache_file_sha256",
    "full_segment_cache_sha256",
    "observation_sha256",
    "crop_record_sha256",
    "full_rgb_sha256",
    "full_mask_sha256",
    "derived_source_sha256",
    "lineage_receipt_file_sha256",
    "lineage_receipt_sha256",
}


@dataclass(frozen=True, slots=True)
class BoundParserRuntime:
    """One constructed parser plus the immutable inputs that define it."""

    runtime: Any
    parser_runtime_manifest_sha256: str
    parser_runtime_bundle_raw_sha256: str
    parser_policy_sha256: str
    foreground_model_manifest_sha256: str
    foreground_model_bundle_raw_sha256: str
    instance_model_manifest_sha256: str
    instance_model_bundle_raw_sha256: str
    device: str
    job_batch_size: int = 1
    instance_batch_size: int = 1
    foreground_batch_size: int = 1
    publication_workers: int = 1
    shape_policy: str = "EXACT_PREPROCESSED_SHAPE_BUCKETS"
    oom_policy: str = "FAIL_CLOSED_NO_RETRY"

    def __post_init__(self) -> None:
        if self.runtime is None:
            raise ValueError("bound parser runtime must contain a runtime")
        for field in (
            "parser_runtime_manifest_sha256",
            "parser_runtime_bundle_raw_sha256",
            "parser_policy_sha256",
            "foreground_model_manifest_sha256",
            "foreground_model_bundle_raw_sha256",
            "instance_model_manifest_sha256",
            "instance_model_bundle_raw_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("parser runtime device must be non-empty text")
        for field in (
            "job_batch_size",
            "instance_batch_size",
            "foreground_batch_size",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"parser runtime {field} must be positive")
        if self.publication_workers not in {1, 4}:
            raise ValueError("parser runtime publication_workers differs")
        if (
            self.shape_policy != "EXACT_PREPROCESSED_SHAPE_BUCKETS"
            or self.oom_policy != "FAIL_CLOSED_NO_RETRY"
        ):
            raise ValueError("parser runtime batch execution policy differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PARSER_RUNTIME_BINDING_SCHEMA,
            "parser_runtime_manifest_sha256": (self.parser_runtime_manifest_sha256),
            "parser_runtime_bundle_raw_sha256": (self.parser_runtime_bundle_raw_sha256),
            "parser_policy_sha256": self.parser_policy_sha256,
            "foreground_model_manifest_sha256": (self.foreground_model_manifest_sha256),
            "foreground_model_bundle_raw_sha256": (
                self.foreground_model_bundle_raw_sha256
            ),
            "instance_model_manifest_sha256": (self.instance_model_manifest_sha256),
            "instance_model_bundle_raw_sha256": (self.instance_model_bundle_raw_sha256),
            "device": self.device,
            "job_batch_size": self.job_batch_size,
            "instance_batch_size": self.instance_batch_size,
            "foreground_batch_size": self.foreground_batch_size,
            "publication_workers": self.publication_workers,
            "shape_policy": self.shape_policy,
            "oom_policy": self.oom_policy,
        }


@dataclass(frozen=True, slots=True)
class _ArtifactInput:
    path: Path
    payload: bytes
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class _JobInputs:
    root: Path
    source: _ArtifactInput
    source_image: Image.Image
    evidence: dict[str, _ArtifactInput]


@dataclass(frozen=True, slots=True)
class _Decision:
    actual_route: ObservationRoute
    source_view_scope: SourceViewScope
    association: AnimalAssociation | None
    parser_lineage: dict[str, Any] | None
    terminal_reason: str | None


@dataclass(frozen=True, slots=True)
class _ValidatedSample:
    observation: FullSegmentObservation
    execution_receipt: dict[str, Any]
    full_segment_cache_file_sha256: str
    full_segment_cache_sha256: str
    lineage_receipt_file_sha256: str | None
    lineage_receipt_sha256: str | None
    crop_record_sha256: str | None
    full_rgb_sha256: str | None
    full_mask_sha256: str | None


def read_route_plan_bundle(path: Path) -> dict[str, Any]:
    """Read a large route-plan bundle under explicit structural limits."""

    return read_strict_json_document(
        path,
        maximum_bytes=_MAX_PLAN_BYTES,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    ).payload


def build_bound_parser_runtime(
    *,
    route_plan_bundle: Mapping[str, Any],
    parsing_runtime_manifest: Path,
    foreground_model_dir: Path,
    foreground_model_manifest: Path,
    instance_model_dir: Path,
    instance_model_manifest: Path,
    device: str,
    repository_root: Path | None = None,
) -> BoundParserRuntime:
    """Construct exactly one parser from frozen, route-plan-bound artifacts."""

    bundle = validate_full128_route_plan_bundle(
        dict(route_plan_bundle), verify_files=False
    )
    if device not in {"cpu", "cuda"}:
        raise ValueError("parser device must be cpu or cuda")
    repository = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    document = read_strict_json_document(
        parsing_runtime_manifest, maximum_bytes=16_777_216
    )
    parsing_bundle = document.payload
    if (
        set(parsing_bundle) != {"schema_version", "manifest_sha256", "manifest"}
        or parsing_bundle["schema_version"] not in SUPPORTED_PARSING_BUNDLE_SCHEMAS
        or not isinstance(parsing_bundle["manifest"], dict)
        or content_sha256(parsing_bundle["manifest"])
        != parsing_bundle["manifest_sha256"]
    ):
        raise ValueError("animal parsing runtime bundle differs")
    manifest = AnimalParsingRuntimeManifest.from_dict(parsing_bundle["manifest"])
    if manifest.manifest_sha256 != parsing_bundle["manifest_sha256"]:
        raise ValueError("animal parsing runtime manifest digest differs")
    if manifest.manifest_sha256 != bundle["parser_runtime_manifest_sha256"]:
        raise ValueError("route-plan parser runtime manifest differs")
    if manifest.policy_sha256 != bundle["parser_policy_sha256"]:
        raise ValueError("route-plan parser policy differs")
    if "dog" not in manifest.supported_classes:
        raise ValueError("route-plan parser runtime does not support dog")
    installed = {
        name: importlib.metadata.version(distribution)
        for name, distribution in {
            "numpy": "numpy",
            "pillow": "Pillow",
            "torch": "torch",
            "torchvision": "torchvision",
            "transformers": "transformers",
        }.items()
    }
    if installed != manifest.runtime_libraries:
        raise ValueError("animal parsing runtime libraries differ")
    for binding in manifest.source_files:
        source = repository.joinpath(*binding.relative_path.split("/"))
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != binding.byte_size
            or _sha256_path(source) != binding.sha256
        ):
            raise ValueError(
                f"frozen animal parsing source differs: {binding.relative_path}"
            )

    foreground = ForegroundSegmentationArtifact.load(
        model_directory=foreground_model_dir,
        manifest_bundle_path=foreground_model_manifest,
    )
    instance = InstanceSegmentationArtifact.load(
        model_directory=instance_model_dir,
        manifest_bundle_path=instance_model_manifest,
    )
    if (
        foreground.manifest.manifest_sha256 != manifest.foreground_model_manifest_sha256
        or foreground.bundle_sha256 != manifest.foreground_model_bundle_raw_sha256
    ):
        raise ValueError("foreground model differs from frozen parser")
    if (
        instance.manifest.manifest_sha256 != manifest.instance_model_manifest_sha256
        or instance.bundle_sha256 != manifest.instance_model_bundle_raw_sha256
    ):
        raise ValueError("instance model differs from frozen parser")
    policy = AnimalParsingPolicy.from_dict(manifest.policy)
    if policy.policy_sha256 != manifest.policy_sha256:
        raise ValueError("animal parsing policy differs from frozen parser")
    runtime = AnimalParsingRuntime(
        instance_runtime=AnimalInstanceSegmentationRuntime(
            artifact=instance,
            device=device,
            mask_threshold=policy.foreground_threshold,
        ),
        foreground_runtime=ForegroundSegmentationRuntime(
            artifact=foreground,
            device=device,
            threshold=policy.foreground_threshold,
        ),
        policy=policy,
    )
    batching = manifest.inference_batching
    return BoundParserRuntime(
        runtime=runtime,
        parser_runtime_manifest_sha256=manifest.manifest_sha256,
        parser_runtime_bundle_raw_sha256=document.raw_sha256,
        parser_policy_sha256=policy.policy_sha256,
        foreground_model_manifest_sha256=foreground.manifest.manifest_sha256,
        foreground_model_bundle_raw_sha256=foreground.bundle_sha256,
        instance_model_manifest_sha256=instance.manifest.manifest_sha256,
        instance_model_bundle_raw_sha256=instance.bundle_sha256,
        device=device,
        job_batch_size=batching["job_batch_size"],
        instance_batch_size=batching["instance_batch_size"],
        foreground_batch_size=batching["foreground_batch_size"],
        publication_workers=batching["publication_workers"],
        shape_policy=batching["shape_policy"],
        oom_policy=batching["oom_policy"],
    )


def materialize_full128_route_plan(
    route_plan_bundle: Mapping[str, Any],
    *,
    output_root: Path,
    parser_runtime: BoundParserRuntime | None,
    verify_plan_files_upfront: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
    maximum_jobs: int | None = None,
    deep_validation: bool = False,
) -> dict[str, Any]:
    """Materialize one deterministic shard without a mutable progress ledger."""

    bundle = validate_full128_route_plan_bundle(
        dict(route_plan_bundle), verify_files=verify_plan_files_upfront
    )
    _validate_shard_arguments(shard_count, shard_index, maximum_jobs)
    if parser_runtime is not None:
        _validate_runtime_against_plan(parser_runtime, bundle)
    root = _prepare_output_root(output_root, create=True)
    samples_root = _private_child_directory(root, "samples", create=True)
    parser_root = _private_child_directory(root, "parser-cache", create=True)
    jobs = _selected_jobs(
        bundle["plan"]["records"],
        shard_count=shard_count,
        shard_index=shard_index,
        maximum_jobs=maximum_jobs,
    )
    if parser_runtime is not None and parser_runtime.job_batch_size > 1:
        if shard_count != 1:
            raise ValueError("batched Full128 parsing requires one canonical shard")
        if maximum_jobs is not None and maximum_jobs % parser_runtime.job_batch_size:
            raise ValueError("maximum_jobs must align to the parser job batch size")
        return _materialize_batched_jobs(
            jobs,
            bundle=bundle,
            samples_root=samples_root,
            parser_root=parser_root,
            parser_runtime=parser_runtime,
            shard_count=shard_count,
            shard_index=shard_index,
            maximum_jobs=maximum_jobs,
        )
    counts: Counter[str] = Counter()
    for assignment_key, rows in jobs:
        targets = tuple(samples_root / row["sample_token"] for row in rows)
        if all(target.exists() or target.is_symlink() for target in targets):
            if deep_validation:
                inputs = _load_job_inputs(rows)
                shard_selection = _shard_selection(
                    assignment_key, shard_count=shard_count, shard_index=shard_index
                )
                results = _materialize_job(
                    rows,
                    inputs=inputs,
                    bundle=bundle,
                    samples_root=samples_root,
                    parser_root=parser_root,
                    parser_runtime=parser_runtime,
                    shard_selection=shard_selection,
                )
                counts.update(results)
            else:
                parser_receipt = (
                    _validate_parser_cache_receipt(
                        parser_root / rows[0]["parser_cache_key"],
                        rows=rows,
                        bundle=bundle,
                    )
                    if rows[0]["route_intent"] == RouteIntent.BODY_PARSING.value
                    else None
                )
                for row, target in zip(rows, targets, strict=True):
                    receipt = _fast_validate_sample_output(
                        target,
                        row=row,
                        bundle=bundle,
                        parser_cache_receipt=parser_receipt,
                    )
                    counts["SKIPPED"] += 1
                    if receipt["actual_route"] == ObservationRoute.NONE.value:
                        counts["TERMINAL"] += 1
            counts["jobs"] += 1
            continue
        inputs = _load_job_inputs(rows)
        shard_selection = _shard_selection(
            assignment_key, shard_count=shard_count, shard_index=shard_index
        )
        results = _materialize_job(
            rows,
            inputs=inputs,
            bundle=bundle,
            samples_root=samples_root,
            parser_root=parser_root,
            parser_runtime=parser_runtime,
            shard_selection=shard_selection,
        )
        counts.update(results)
        counts["jobs"] += 1
    return {
        "plan_sha256": bundle["plan_sha256"],
        "shard_count": shard_count,
        "shard_index": shard_index,
        "maximum_jobs": maximum_jobs,
        "selected_job_count": len(jobs),
        "created_sample_count": counts["CREATED"],
        "skipped_sample_count": counts["SKIPPED"],
        "terminal_sample_count": counts["TERMINAL"],
    }


def _materialize_batched_jobs(
    jobs: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    bundle: Mapping[str, Any],
    samples_root: Path,
    parser_root: Path,
    parser_runtime: BoundParserRuntime,
    shard_count: int,
    shard_index: int,
    maximum_jobs: int | None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    size = parser_runtime.job_batch_size
    units = _batched_job_units(jobs)
    for start in range(0, len(units), size):
        chunk = units[start : start + size]
        targets = tuple(
            samples_root / row["sample_token"]
            for unit in chunk
            for _assignment_key, rows in unit
            for row in rows
        )
        if all(target.exists() or target.is_symlink() for target in targets):
            for unit in chunk:
                for _assignment_key, rows in unit:
                    parser_receipt = _validate_parser_cache_receipt(
                        parser_root / rows[0]["parser_cache_key"],
                        rows=rows,
                        bundle=bundle,
                    )
                    for row in rows:
                        receipt = _fast_validate_sample_output(
                            samples_root / row["sample_token"],
                            row=row,
                            bundle=bundle,
                            parser_cache_receipt=parser_receipt,
                        )
                        counts["SKIPPED"] += 1
                        if receipt["actual_route"] == ObservationRoute.NONE.value:
                            counts["TERMINAL"] += 1
                    counts["jobs"] += 1
            continue

        inputs_by_unit = tuple(_load_job_inputs(unit[0][1]) for unit in chunk)
        predictions = parser_runtime.runtime.predict_batch(
            tuple(inputs.source_image for inputs in inputs_by_unit),
            instance_batch_size=parser_runtime.instance_batch_size,
            foreground_batch_size=parser_runtime.foreground_batch_size,
        )
        if len(predictions) != len(chunk):
            raise RuntimeError("Full128 batched parser prediction count differs")
        work = tuple(zip(chunk, inputs_by_unit, predictions, strict=True))

        def publish(item: tuple[Any, ...]) -> Counter[str]:
            unit, primary_inputs, prediction = item
            results: Counter[str] = Counter()
            for index, (assignment_key, rows) in enumerate(unit):
                results.update(
                    _materialize_job(
                        rows,
                        inputs=(
                            primary_inputs if index == 0 else _load_job_inputs(rows)
                        ),
                        bundle=bundle,
                        samples_root=samples_root,
                        parser_root=parser_root,
                        parser_runtime=parser_runtime,
                        shard_selection=_shard_selection(
                            assignment_key,
                            shard_count=shard_count,
                            shard_index=shard_index,
                        ),
                        prediction=prediction if index == 0 else None,
                    )
                )
            return results

        if parser_runtime.publication_workers == 1:
            for results in map(publish, work):
                counts.update(results)
                counts["jobs"] += 1
        else:
            with ThreadPoolExecutor(
                max_workers=min(parser_runtime.publication_workers, len(work)),
                thread_name_prefix="full128-publish",
            ) as executor:
                for results in executor.map(publish, work):
                    counts.update(results)
                    counts["jobs"] += 1
    return {
        "plan_sha256": bundle["plan_sha256"],
        "shard_count": shard_count,
        "shard_index": shard_index,
        "maximum_jobs": maximum_jobs,
        "selected_job_count": len(jobs),
        "created_sample_count": counts["CREATED"],
        "skipped_sample_count": counts["SKIPPED"],
        "terminal_sample_count": counts["TERMINAL"],
    }


def _batched_job_units(
    jobs: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> tuple[tuple[tuple[str, Sequence[dict[str, Any]]], ...], ...]:
    grouped: dict[str, list[tuple[str, Sequence[dict[str, Any]]]]] = {}
    for assignment_key, rows in jobs:
        grouped.setdefault(assignment_key, []).append((assignment_key, rows))
    return tuple(tuple(unit) for unit in grouped.values())


def assemble_full128_materialization(
    route_plan_bundle: Mapping[str, Any],
    *,
    output_root: Path,
    allocation_name: str = "full128-route-plan-v1",
    verify_plan_files_upfront: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Require complete coverage, then build the split, census, and inventory."""

    if verify_plan_files_upfront:
        raise ValueError(
            "Full128 assembly validates plan artifacts during its one-pass deep read"
        )
    if progress is not None and not callable(progress):
        raise TypeError("Full128 assembly progress must be callable or None")
    bundle = validate_full128_route_plan_bundle(
        dict(route_plan_bundle), verify_files=False
    )
    root = _prepare_output_root(output_root, create=False)
    samples_root = _private_child_directory(root, "samples", create=False)
    parser_root = _private_child_directory(root, "parser-cache", create=False)
    validated: dict[str, _ValidatedSample] = {}
    jobs = _selected_jobs(
        bundle["plan"]["records"],
        shard_count=1,
        shard_index=0,
        maximum_jobs=None,
    )
    for completed_jobs, (_assignment_key, rows) in enumerate(jobs, start=1):
        inputs = _load_job_inputs(rows)
        decisions, frozen, cache_receipt = _job_decisions(
            rows,
            inputs=inputs,
            bundle=bundle,
            parser_root=parser_root,
            parser_runtime=None,
            allow_cache_creation=False,
        )
        for row in rows:
            target = samples_root / row["sample_token"]
            if not target.exists() or target.is_symlink():
                raise ValueError(
                    "Full128 assembly requires complete sample coverage; missing "
                    f"sample: {row['sample_token']}"
                )
            derived = _derived_artifacts(row, inputs)
            validated[row["sample_token"]] = _validate_sample_output(
                target,
                row=row,
                bundle=bundle,
                inputs=inputs,
                decision=decisions[row["sample_token"]],
                frozen=frozen,
                parser_cache_receipt=cache_receipt,
                derived=derived,
            )
        if progress is not None:
            progress(completed_jobs, len(jobs))
    records = bundle["plan"]["records"]
    if len(validated) != len(records):
        raise ValueError("Full128 assembly coverage differs from the route plan")

    observations = tuple(
        _unified_observation(row, validated[row["sample_token"]].observation)
        for row in records
    )
    try:
        manifest = allocate_unified_full_split(
            allocation_name=allocation_name,
            observations=observations,
        )
    except ValueError as exc:
        raise ValueError(
            "Full128 official role topology conflicts with identity/group/duplicate "
            f"closure: {exc}"
        ) from exc
    census = build_unified_full_census(manifest)
    split_bundle = unified_full_split_bundle(manifest, census)
    role_by_sample = {
        item.sample_token: item.terminal_role.value for item in manifest.observations
    }
    request_rows = [
        _inventory_request_row(
            row,
            output_root=root,
            validated=validated[row["sample_token"]],
            terminal_role=role_by_sample[row["sample_token"]],
        )
        for row in records
    ]
    inventory_request = {
        "schema_version": INVENTORY_REQUEST_SCHEMA,
        "rows": request_rows,
    }
    from identity_methods.full_segment.inventory import (
        _build_full128_experiment_inventory_from_prevalidated,
        _PrevalidatedArtifact,
        _PrevalidatedMaterialization,
    )

    prevalidated = _PrevalidatedMaterialization(
        artifact_root=root,
        artifacts=tuple(
            _PrevalidatedArtifact(
                sample_token=row["sample_token"],
                sample_directory=samples_root / row["sample_token"],
                observation=sample.observation,
                full_segment_cache_file_sha256=(sample.full_segment_cache_file_sha256),
                full_segment_cache_sha256=sample.full_segment_cache_sha256,
                lineage_receipt_file_sha256=(sample.lineage_receipt_file_sha256),
                lineage_receipt_sha256=sample.lineage_receipt_sha256,
                crop_record_sha256=sample.crop_record_sha256,
                full_rgb_sha256=sample.full_rgb_sha256,
                full_mask_sha256=sample.full_mask_sha256,
            )
            for row in records
            for sample in (validated[row["sample_token"]],)
        ),
    )
    inventory_bundle = _build_full128_experiment_inventory_from_prevalidated(
        unified_full_split=split_bundle,
        request_rows=request_rows,
        artifact_root=root,
        prevalidated=prevalidated,
    )
    topology_report = {
        "official_eval_observation_counts": dict(
            sorted(
                Counter(
                    f"{item.dataset_name}|{item.official_split}"
                    for item in manifest.observations
                    if item.terminal_role is TerminalRole.EVAL
                ).items()
            )
        ),
        "terminal_role_counts": dict(sorted(Counter(role_by_sample.values()).items())),
        "overlap_report": census.to_dict()["overlap_report"],
    }
    payload = {
        "schema_version": ASSEMBLY_SCHEMA,
        "plan_sha256": bundle["plan_sha256"],
        "sample_count": len(records),
        "allocation_name": allocation_name,
        "topology_report": topology_report,
        "unified_full_split": split_bundle,
        "inventory_request": inventory_request,
        "inventory_bundle": inventory_bundle,
    }
    return {**payload, "assembly_sha256": content_sha256(payload)}


def migrate_full128_compact_sample_caches(
    route_plan_bundle: Mapping[str, Any],
    *,
    output_root: Path,
    maximum_samples: int | None = None,
    verify_plan_files_upfront: bool = True,
) -> dict[str, Any]:
    """Atomically migrate validated legacy BODY_PARSING sample directories."""

    bundle = validate_full128_route_plan_bundle(
        dict(route_plan_bundle), verify_files=verify_plan_files_upfront
    )
    if maximum_samples is not None and (
        isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples <= 0
    ):
        raise ValueError("maximum_samples must be positive or None")
    root = _prepare_output_root(output_root, create=False)
    samples_root = _private_child_directory(root, "samples", create=False)
    parser_root = _private_child_directory(root, "parser-cache", create=False)
    counts: Counter[str] = Counter()
    bytes_before = 0
    bytes_after = 0
    stop = False
    for _, rows in _selected_jobs(
        bundle["plan"]["records"],
        shard_count=1,
        shard_index=0,
        maximum_jobs=None,
    ):
        inputs = _load_job_inputs(rows)
        decisions, frozen, cache_receipt = _job_decisions(
            rows,
            inputs=inputs,
            bundle=bundle,
            parser_root=parser_root,
            parser_runtime=None,
            allow_cache_creation=False,
        )
        for row in rows:
            target = samples_root / row["sample_token"]
            if not target.exists() or target.is_symlink():
                raise ValueError(
                    "Full128 compact migration requires complete sample coverage"
                )
            decision = decisions[row["sample_token"]]
            derived = _derived_artifacts(row, inputs)
            validated = _validate_sample_output(
                target,
                row=row,
                bundle=bundle,
                inputs=inputs,
                decision=decision,
                frozen=frozen,
                parser_cache_receipt=cache_receipt,
                derived=derived,
            )
            if decision.actual_route is not ObservationRoute.BODY_PARSING:
                counts["NOT_APPLICABLE"] += 1
                continue
            cache_document = read_strict_json_document(
                target / "full-segment-cache.json",
                maximum_bytes=_MAX_JSON_BYTES,
                maximum_string_characters=_MAX_JSON_BYTES,
            )
            if cache_document.payload["cache"]["schema_version"] != LEGACY_CACHE_SCHEMA:
                counts["ALREADY_COMPACT"] += 1
                continue
            if maximum_samples is not None and counts["MIGRATED"] >= maximum_samples:
                stop = True
                break
            if frozen is None or cache_receipt is None:
                raise AssertionError("parsed migration lost parser cache provenance")
            old_size = cache_document.byte_size
            migrated_size = _migrate_sample_cache_directory(
                target,
                row=row,
                bundle=bundle,
                inputs=inputs,
                decision=decision,
                frozen=frozen,
                parser_cache_receipt=cache_receipt,
                derived=derived,
                validated=validated,
            )
            if migrated_size is None:
                counts["ATOMIC_EXCHANGE_UNAVAILABLE"] += 1
                stop = True
                break
            counts["MIGRATED"] += 1
            bytes_before += old_size
            bytes_after += migrated_size
        if stop:
            break
    return {
        "plan_sha256": bundle["plan_sha256"],
        "migrated_sample_count": counts["MIGRATED"],
        "already_compact_sample_count": counts["ALREADY_COMPACT"],
        "not_applicable_sample_count": counts["NOT_APPLICABLE"],
        "atomic_exchange_available": not counts["ATOMIC_EXCHANGE_UNAVAILABLE"],
        "legacy_cache_bytes": bytes_before,
        "compact_cache_bytes": bytes_after,
        "saved_cache_bytes": bytes_before - bytes_after,
    }


def validate_derived_lineage_receipt(
    value: object,
    *,
    expected_parent_sha256: str | None = None,
    expected_child_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a persisted derived source lineage record."""

    if not isinstance(value, dict) or set(value) != _LINEAGE_FIELDS:
        raise ValueError("Full128 derived lineage fields differ")
    lineage = dict(value)
    if lineage["schema_version"] != DERIVED_LINEAGE_SCHEMA:
        raise ValueError("Full128 derived lineage schema differs")
    payload = {key: item for key, item in lineage.items() if key != "lineage_sha256"}
    _require_sha256(lineage["lineage_sha256"], "derived lineage")
    if content_sha256(payload) != lineage["lineage_sha256"]:
        raise ValueError("Full128 derived lineage digest differs")
    for field in ("sample_token", "parent_source_sha256", "child_sha256"):
        _require_sha256(lineage[field], field)
    if expected_parent_sha256 is not None and (
        lineage["parent_source_sha256"] != expected_parent_sha256
    ):
        raise ValueError("Full128 derived lineage parent source differs")
    if expected_child_sha256 is not None and (
        lineage["child_sha256"] != expected_child_sha256
    ):
        raise ValueError("Full128 derived lineage child source differs")
    if lineage["route_intent"] not in {
        RouteIntent.DERIVED_NATIVE_FACE.value,
        RouteIntent.DERIVED_NATIVE_HEAD.value,
    }:
        raise ValueError("Full128 derived lineage route differs")
    evidence = lineage["evidence_artifact"]
    if not isinstance(evidence, dict) or set(evidence) != {
        "relative_path",
        "sha256",
        "byte_size",
    }:
        raise ValueError("Full128 derived lineage evidence binding differs")
    _safe_relative(evidence["relative_path"], "derived evidence")
    _require_sha256(evidence["sha256"], "derived evidence")
    if (
        isinstance(evidence["byte_size"], bool)
        or not isinstance(evidence["byte_size"], int)
        or evidence["byte_size"] <= 0
    ):
        raise ValueError("Full128 derived evidence byte size differs")
    _finite_box(lineage["bbox_xyxy"], integer=False)
    aligned = _finite_box(lineage["aligned_bbox_xyxy"], integer=True)
    if (
        lineage["crop_policy"]
        != {
            "minimum_rounding": "FLOOR",
            "maximum_rounding": "CEIL",
            "color_mode": "RGB",
        }
        or lineage["child_encoding"] != "PNG_RGB_LOSSLESS"
    ):
        raise ValueError("Full128 derived crop policy differs")
    width = _positive_int(lineage["child_width"], "derived child width")
    height = _positive_int(lineage["child_height"], "derived child height")
    if (aligned[2] - aligned[0], aligned[3] - aligned[1]) != (width, height):
        raise ValueError("Full128 derived child dimensions differ from bbox")
    return lineage


def _materialize_job(
    rows: Sequence[dict[str, Any]],
    *,
    inputs: _JobInputs,
    bundle: Mapping[str, Any],
    samples_root: Path,
    parser_root: Path,
    parser_runtime: BoundParserRuntime | None,
    shard_selection: dict[str, Any],
    prediction: AnimalParsingPrediction | None = None,
) -> Counter[str]:
    decisions, frozen, cache_receipt = _job_decisions(
        rows,
        inputs=inputs,
        bundle=bundle,
        parser_root=parser_root,
        parser_runtime=parser_runtime,
        allow_cache_creation=True,
        prediction=prediction,
    )
    counts: Counter[str] = Counter()
    for row in rows:
        target = samples_root / row["sample_token"]
        decision = decisions[row["sample_token"]]
        derived = _derived_artifacts(row, inputs)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ValueError(
                    "Full128 sample output must be a non-symlink directory"
                )
            _validate_sample_output(
                target,
                row=row,
                bundle=bundle,
                inputs=inputs,
                decision=decision,
                frozen=frozen,
                parser_cache_receipt=cache_receipt,
                derived=derived,
            )
            counts["SKIPPED"] += 1
        else:
            _publish_sample(
                target,
                row=row,
                bundle=bundle,
                inputs=inputs,
                decision=decision,
                frozen=frozen,
                parser_cache_receipt=cache_receipt,
                derived=derived,
                shard_selection=shard_selection,
            )
            counts["CREATED"] += 1
        if decision.actual_route is ObservationRoute.NONE:
            counts["TERMINAL"] += 1
    return counts


def _job_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    inputs: _JobInputs,
    bundle: Mapping[str, Any],
    parser_root: Path,
    parser_runtime: BoundParserRuntime | None,
    allow_cache_creation: bool,
    prediction: AnimalParsingPrediction | None = None,
) -> tuple[dict[str, _Decision], dict[str, Any] | None, dict[str, Any] | None]:
    route = RouteIntent(rows[0]["route_intent"])
    if any(RouteIntent(row["route_intent"]) is not route for row in rows):
        raise ValueError("Full128 source job mixes route intents")
    if route is not RouteIntent.BODY_PARSING:
        if prediction is not None:
            raise ValueError("direct Full128 route cannot receive a parser prediction")
        return (
            {row["sample_token"]: _direct_decision(row, inputs) for row in rows},
            None,
            None,
        )
    cache_key = rows[0]["parser_cache_key"]
    if any(row["parser_cache_key"] != cache_key for row in rows):
        raise ValueError("Full128 parser job mixes parser cache keys")
    cache_dir = parser_root / cache_key
    if cache_dir.exists() or cache_dir.is_symlink():
        supplied_prediction = prediction
        frozen, receipt, prediction = _validate_parser_cache(
            cache_dir, rows=rows, bundle=bundle
        )
        if (
            supplied_prediction is not None
            and freeze_animal_parsing_prediction(supplied_prediction)[
                "prediction_sha256"
            ]
            != receipt["prediction_sha256"]
        ):
            raise ValueError(
                "Full128 fixed-batch prediction differs from existing parser cache"
            )
    else:
        if not allow_cache_creation:
            raise ValueError(
                "Full128 assembly requires complete parser-cache coverage; missing "
                f"cache: {cache_key}"
            )
        if any(
            (parser_root.parent / "samples" / row["sample_token"]).exists()
            for row in rows
        ):
            raise ValueError(
                "Full128 parser cache is missing for an existing sample output"
            )
        if parser_runtime is None:
            raise ValueError(
                "BODY_PARSING materialization requires a bound parser runtime"
            )
        if prediction is None:
            frozen, receipt, prediction = _publish_parser_cache(
                cache_dir,
                rows=rows,
                bundle=bundle,
                source=inputs.source,
                source_image=inputs.source_image,
                parser_runtime=parser_runtime,
            )
        else:
            frozen, receipt, prediction = _publish_parser_prediction(
                cache_dir,
                rows=rows,
                bundle=bundle,
                source=inputs.source,
                parser_runtime=parser_runtime,
                prediction=prediction,
            )
    decisions = _parser_decisions(rows, prediction=prediction, cache_receipt=receipt)
    return decisions, frozen, receipt


def _direct_decision(row: Mapping[str, Any], inputs: _JobInputs) -> _Decision:
    route = RouteIntent(row["route_intent"])
    if route is RouteIntent.BODY_MASK:
        trimap = inputs.evidence["trimap_artifact"].payload
        scope = (
            SourceViewScope.BODY_TRUNCATED
            if _trimap_foreground_touches_border(trimap)
            else SourceViewScope.BODY_AVAILABLE
        )
        return _Decision(ObservationRoute.BODY_MASK, scope, None, None, None)
    if route in {RouteIntent.NATIVE_FACE, RouteIntent.DERIVED_NATIVE_FACE}:
        return _Decision(
            ObservationRoute.NATIVE_FACE,
            SourceViewScope.FACE_NATIVE,
            None,
            None,
            None,
        )
    if route is RouteIntent.DERIVED_NATIVE_HEAD:
        return _Decision(
            ObservationRoute.NATIVE_HEAD,
            SourceViewScope.HEAD_NATIVE,
            None,
            None,
            None,
        )
    raise ValueError("Full128 direct decision received a parser route")


def _parser_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    prediction: AnimalParsingPrediction,
    cache_receipt: dict[str, Any],
) -> dict[str, _Decision]:
    if all(row["dataset_name"] == "ap10k-dog" for row in rows):
        return _ap10k_parser_decisions(
            rows, prediction=prediction, cache_receipt=cache_receipt
        )
    if any(row["dataset_name"] == "ap10k-dog" for row in rows):
        raise ValueError("AP-10K parser source cannot mix with another dataset")
    if all(row["schema_version"] == "cvi.full128_route_plan_record.v2" for row in rows):
        return _legacy_parser_decisions(
            rows, prediction=prediction, cache_receipt=cache_receipt
        )
    if any(row["schema_version"] != "cvi.full128_route_plan_record.v3" for row in rows):
        raise ValueError("Full128 parser rows mix route-plan record schemas")
    datasets = {row["dataset_name"] for row in rows}
    if len(datasets) != 1:
        raise ValueError("Full128 parser source cannot mix datasets")
    if next(iter(datasets)) in {"dogflw", "oxford-pets-dog"}:
        return _largest_valid_dog_parser_decisions(
            rows, prediction=prediction, cache_receipt=cache_receipt
        )
    return _single_distinct_dog_parser_decisions(
        rows, prediction=prediction, cache_receipt=cache_receipt
    )


def _legacy_parser_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    prediction: AnimalParsingPrediction,
    cache_receipt: dict[str, Any],
) -> dict[str, _Decision]:
    instances = prediction.instances
    dogs = tuple(instance for instance in instances if instance.class_name == "dog")
    if not dogs:
        return {
            row["sample_token"]: _terminal_parser_decision(
                row,
                cache_receipt=cache_receipt,
                reason="NO_PARSED_DOG_INSTANCE",
                scope=SourceViewScope.UNAVAILABLE,
            )
            for row in rows
        }
    if len(instances) != 1 or len(dogs) != 1:
        return {
            row["sample_token"]: _terminal_parser_decision(
                row,
                cache_receipt=cache_receipt,
                reason="PARSER_INSTANCE_CARDINALITY_AMBIGUOUS",
                scope=SourceViewScope.AMBIGUOUS,
            )
            for row in rows
        }
    association = AnimalAssociation(AssociationKind.EXACTLY_ONE, 0)
    lineage = _parser_lineage(cache_receipt, association=association, authority=None)
    return {
        row["sample_token"]: _selected_parser_decision(
            prediction.instances[0], association=association, parser_lineage=lineage
        )
        for row in rows
    }


def _largest_valid_dog_parser_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    prediction: AnimalParsingPrediction,
    cache_receipt: dict[str, Any],
) -> dict[str, _Decision]:
    dogs = tuple(
        instance for instance in prediction.instances if instance.class_name == "dog"
    )
    valid_dogs = tuple(
        instance
        for instance in dogs
        if instance.quality.state in {"USABLE", "REVIEW"}
        and instance.mask_box_xyxy is not None
    )
    if not valid_dogs:
        reason = (
            "NO_PARSED_DOG_INSTANCE"
            if not dogs
            else "NO_VALID_PARSED_DOG_INSTANCE"
        )
        selection = _selection_lineage(
            prediction,
            dogs=dogs,
            valid_dogs=valid_dogs,
            rule="LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS",
            selected=None,
            terminal_reason=reason,
        )
        return {
            row["sample_token"]: _terminal_parser_decision(
                row,
                cache_receipt=cache_receipt,
                reason=reason,
                scope=SourceViewScope.UNAVAILABLE,
                selection=selection,
            )
            for row in rows
        }
    selected = min(
        valid_dogs,
        key=lambda instance: (-instance.quality.foreground_pixels, instance.instance_index),
    )
    selection = _selection_lineage(
        prediction,
        dogs=dogs,
        valid_dogs=valid_dogs,
        rule="LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS",
        selected=selected,
        terminal_reason=None,
    )
    return {
        row["sample_token"]: _policy_selected_parser_decision(
            row,
            selected=selected,
            cache_receipt=cache_receipt,
            selection=selection,
        )
        for row in rows
    }


def _single_distinct_dog_parser_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    prediction: AnimalParsingPrediction,
    cache_receipt: dict[str, Any],
) -> dict[str, _Decision]:
    dogs = tuple(
        instance for instance in prediction.instances if instance.class_name == "dog"
    )
    valid_dogs = tuple(
        instance for instance in dogs if instance.quality.state in {"USABLE", "REVIEW"}
    )
    if len(dogs) != 1:
        reason = (
            "NO_PARSED_DOG_INSTANCE"
            if not dogs
            else "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS"
        )
        selection = _selection_lineage(
            prediction,
            dogs=dogs,
            valid_dogs=valid_dogs,
            rule="REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
            selected=None,
            terminal_reason=reason,
        )
        return {
            row["sample_token"]: _terminal_parser_decision(
                row,
                cache_receipt=cache_receipt,
                reason=reason,
                scope=(
                    SourceViewScope.UNAVAILABLE
                    if not dogs
                    else SourceViewScope.AMBIGUOUS
                ),
                selection=selection,
            )
            for row in rows
        }
    selected = dogs[0]
    selection = _selection_lineage(
        prediction,
        dogs=dogs,
        valid_dogs=valid_dogs,
        rule="REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
        selected=selected,
        terminal_reason=(
            "SELECTED_DOG_PARSING_UNUSABLE"
            if selected.quality.state == "UNUSABLE"
            else None
        ),
    )
    return {
        row["sample_token"]: _policy_selected_parser_decision(
            row,
            selected=selected,
            cache_receipt=cache_receipt,
            selection=selection,
        )
        for row in rows
    }


def _policy_selected_parser_decision(
    row: Mapping[str, Any],
    *,
    selected: Any,
    cache_receipt: dict[str, Any],
    selection: dict[str, Any],
) -> _Decision:
    authority_payload = {
        "schema_version": POLICY_SELECTION_AUTHORITY_SCHEMA,
        "plan_record_sha256": row["record_sha256"],
        "prediction_sha256": cache_receipt["prediction_sha256"],
        "parser_policy_sha256": cache_receipt["runtime"]["parser_policy_sha256"],
        "selection": selection,
    }
    authority = {
        **authority_payload,
        "authority_sha256": content_sha256(authority_payload),
    }
    association = AnimalAssociation(
        AssociationKind.AUTHORITATIVE,
        selected.instance_index,
        authority["authority_sha256"],
    )
    lineage = _parser_lineage(
        cache_receipt,
        association=association,
        authority=authority,
        selection=selection,
    )
    return _selected_parser_decision(
        selected, association=association, parser_lineage=lineage
    )


def _selection_lineage(
    prediction: AnimalParsingPrediction,
    *,
    dogs: Sequence[Any],
    valid_dogs: Sequence[Any],
    rule: str,
    selected: Any | None,
    terminal_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_LINEAGE_SCHEMA,
        "rule": rule,
        "prediction_instance_count": len(prediction.instances),
        "post_suppression_dog_count": len(dogs),
        "valid_post_suppression_dog_count": len(valid_dogs),
        "selected_instance_count": 0 if selected is None else 1,
        "selected_instance_index": (
            None if selected is None else selected.instance_index
        ),
        "selected_foreground_pixels": (
            None if selected is None else selected.quality.foreground_pixels
        ),
        "terminal_reason": terminal_reason,
    }


def _ap10k_parser_decisions(
    rows: Sequence[dict[str, Any]],
    *,
    prediction: AnimalParsingPrediction,
    cache_receipt: dict[str, Any],
) -> dict[str, _Decision]:
    dogs = tuple(
        instance for instance in prediction.instances if instance.class_name == "dog"
    )
    valid_dogs = tuple(
        instance for instance in dogs if instance.quality.state in {"USABLE", "REVIEW"}
    )
    if not dogs:
        selection = _selection_lineage(
            prediction,
            dogs=dogs,
            valid_dogs=valid_dogs,
            rule="AP10K_AUTHORITATIVE_GLOBAL_BBOX_MATCH",
            selected=None,
            terminal_reason="NO_PARSED_DOG_INSTANCE",
        )
        return {
            row["sample_token"]: _terminal_parser_decision(
                row,
                cache_receipt=cache_receipt,
                reason="NO_PARSED_DOG_INSTANCE",
                scope=SourceViewScope.UNAVAILABLE,
                selection=(
                    selection
                    if row["schema_version"] == "cvi.full128_route_plan_record.v3"
                    else None
                ),
            )
            for row in rows
        }
    matrix = [
        [
            _box_iou(row["route_evidence"]["bbox_xyxy"], dog.detector_box_xyxy)
            for dog in dogs
        ]
        for row in rows
    ]
    result = _unique_global_assignment(matrix, threshold=_IOU_THRESHOLD)
    if result is None:
        selection = _selection_lineage(
            prediction,
            dogs=dogs,
            valid_dogs=valid_dogs,
            rule="AP10K_AUTHORITATIVE_GLOBAL_BBOX_MATCH",
            selected=None,
            terminal_reason="AP10K_GLOBAL_BBOX_ASSOCIATION_AMBIGUOUS",
        )
        return {
            row["sample_token"]: _terminal_parser_decision(
                row,
                cache_receipt=cache_receipt,
                reason="AP10K_GLOBAL_BBOX_ASSOCIATION_AMBIGUOUS",
                scope=SourceViewScope.AMBIGUOUS,
                selection=(
                    selection
                    if row["schema_version"] == "cvi.full128_route_plan_record.v3"
                    else None
                ),
            )
            for row in rows
        }
    decisions: dict[str, _Decision] = {}
    for row_index, dog_index in enumerate(result):
        row = rows[row_index]
        selected = dogs[dog_index]
        selection = _selection_lineage(
            prediction,
            dogs=dogs,
            valid_dogs=valid_dogs,
            rule="AP10K_AUTHORITATIVE_GLOBAL_BBOX_MATCH",
            selected=selected,
            terminal_reason=(
                "SELECTED_DOG_PARSING_UNUSABLE"
                if selected.quality.state == "UNUSABLE"
                else None
            ),
        )
        plan_authority = row["route_evidence"]["association_intent"]["authority_sha256"]
        authority_payload = {
            "schema_version": ASSOCIATION_AUTHORITY_SCHEMA,
            "prediction_sha256": cache_receipt["prediction_sha256"],
            "plan_authority_sha256": plan_authority,
            "annotation_bbox_xyxy": row["route_evidence"]["bbox_xyxy"],
            "selected_instance_index": selected.instance_index,
            "selected_detector_box_xyxy": list(selected.detector_box_xyxy),
            "match_iou": matrix[row_index][dog_index],
            "minimum_match_iou": _IOU_THRESHOLD,
        }
        authority = {
            **authority_payload,
            "authority_sha256": content_sha256(authority_payload),
        }
        association = AnimalAssociation(
            AssociationKind.AUTHORITATIVE,
            selected.instance_index,
            authority["authority_sha256"],
        )
        lineage = _parser_lineage(
            cache_receipt,
            association=association,
            authority=authority,
            selection=(
                selection
                if row["schema_version"] == "cvi.full128_route_plan_record.v3"
                else None
            ),
        )
        decisions[row["sample_token"]] = _selected_parser_decision(
            selected, association=association, parser_lineage=lineage
        )
    return decisions


def _selected_parser_decision(
    instance: Any,
    *,
    association: AnimalAssociation,
    parser_lineage: dict[str, Any],
) -> _Decision:
    if instance.quality.state == "UNUSABLE":
        return _Decision(
            ObservationRoute.NONE,
            SourceViewScope.UNAVAILABLE,
            None,
            parser_lineage,
            "SELECTED_DOG_PARSING_UNUSABLE",
        )
    scope = (
        SourceViewScope.BODY_TRUNCATED
        if instance.quality.touches_source_border
        else SourceViewScope.BODY_AVAILABLE
    )
    return _Decision(
        ObservationRoute.BODY_PARSING,
        scope,
        association,
        parser_lineage,
        None,
    )


def _terminal_parser_decision(
    row: Mapping[str, Any],
    *,
    cache_receipt: dict[str, Any],
    reason: str,
    scope: SourceViewScope,
    selection: dict[str, Any] | None = None,
) -> _Decision:
    del row
    return _Decision(
        ObservationRoute.NONE,
        scope,
        None,
        _parser_lineage(
            cache_receipt,
            association=None,
            authority=None,
            selection=selection,
        ),
        reason,
    )


def _parser_lineage(
    cache_receipt: Mapping[str, Any],
    *,
    association: AnimalAssociation | None,
    authority: dict[str, Any] | None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lineage = {
        "parser_cache_key": cache_receipt["parser_cache_key"],
        "parser_cache_receipt_sha256": cache_receipt["receipt_sha256"],
        "prediction_sha256": cache_receipt["prediction_sha256"],
        "association": None if association is None else association.to_dict(),
        "association_authority": authority,
    }
    if selection is not None:
        lineage["selection"] = selection
    return lineage


def _publish_parser_cache(
    cache_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    bundle: Mapping[str, Any],
    source: _ArtifactInput,
    source_image: Image.Image,
    parser_runtime: BoundParserRuntime,
) -> tuple[dict[str, Any], dict[str, Any], AnimalParsingPrediction]:
    _validate_runtime_against_plan(parser_runtime, bundle)
    prediction = parser_runtime.runtime.predict(source_image)
    return _publish_parser_prediction(
        cache_dir,
        rows=rows,
        bundle=bundle,
        source=source,
        parser_runtime=parser_runtime,
        prediction=prediction,
    )


def _publish_parser_prediction(
    cache_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    bundle: Mapping[str, Any],
    source: _ArtifactInput,
    parser_runtime: BoundParserRuntime,
    prediction: AnimalParsingPrediction,
) -> tuple[dict[str, Any], dict[str, Any], AnimalParsingPrediction]:
    _validate_runtime_against_plan(parser_runtime, bundle)
    if not isinstance(prediction, AnimalParsingPrediction):
        raise TypeError("Full128 parser runtime returned an invalid prediction")
    if (
        prediction.source_width != rows[0]["source_width"]
        or prediction.source_height != rows[0]["source_height"]
        or prediction.policy_sha256 != bundle["parser_policy_sha256"]
    ):
        raise ValueError(
            "Full128 parser prediction differs from bound source or policy"
        )
    frozen = freeze_animal_parsing_prediction(prediction)
    frozen_bytes = json_document_bytes(frozen)
    runtime_binding = parser_runtime.to_dict()
    receipt_payload = {
        "schema_version": PARSER_CACHE_RECEIPT_SCHEMA,
        "parser_cache_key": rows[0]["parser_cache_key"],
        "source_sha256": source.sha256,
        "source_width": prediction.source_width,
        "source_height": prediction.source_height,
        "runtime": runtime_binding,
        "prediction_sha256": frozen["prediction_sha256"],
        "frozen_json_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": content_sha256(receipt_payload),
    }
    parent = cache_dir.parent
    published = True
    with TemporaryDirectory(prefix=f".{cache_dir.name}-", dir=parent) as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir(mode=0o700)
        _write_new(staging / "frozen.json", frozen_bytes)
        _write_new(staging / "receipt.json", json_document_bytes(receipt))
        fsync_directory(staging)
        try:
            rename_directory_noreplace(staging, cache_dir)
        except FileExistsError:
            published = False
    fsync_directory(parent)
    if published:
        return frozen, receipt, prediction
    existing_frozen, existing_receipt, existing_prediction = _validate_parser_cache(
        cache_dir, rows=rows, bundle=bundle
    )
    if existing_receipt["prediction_sha256"] != receipt["prediction_sha256"]:
        raise ValueError("concurrent Full128 parser cache prediction differs")
    return existing_frozen, existing_receipt, existing_prediction


def _validate_parser_cache(
    cache_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], AnimalParsingPrediction]:
    _require_exact_directory(cache_dir, {"frozen.json", "receipt.json"}, "parser cache")
    frozen_document = read_strict_json_document(
        cache_dir / "frozen.json",
        maximum_bytes=_MAX_JSON_BYTES,
        maximum_string_characters=_MAX_JSON_BYTES,
    )
    receipt = _validate_parser_cache_receipt(
        cache_dir,
        rows=rows,
        bundle=bundle,
        observed_frozen_json_sha256=frozen_document.raw_sha256,
    )
    prediction = thaw_animal_parsing_prediction(frozen_document.payload)
    frozen = frozen_document.payload
    expected = {
        "prediction_sha256": frozen["prediction_sha256"],
        "frozen_json_sha256": frozen_document.raw_sha256,
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise ValueError(f"Full128 parser cache receipt {field} differs")
    prediction_payload = frozen["prediction"]
    if prediction_payload["policy_sha256"] != bundle["parser_policy_sha256"]:
        raise ValueError("Full128 parser cache policy differs from route plan")
    return frozen, receipt, prediction


def _validate_parser_cache_receipt(
    cache_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    bundle: Mapping[str, Any],
    observed_frozen_json_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate parser lineage without loading or decoding frozen parser arrays."""

    _require_exact_directory(cache_dir, {"frozen.json", "receipt.json"}, "parser cache")
    receipt = read_strict_json_document(
        cache_dir / "receipt.json", maximum_bytes=1_048_576
    ).payload
    if not isinstance(receipt, dict) or set(receipt) != _PARSER_RECEIPT_FIELDS:
        raise ValueError("Full128 parser cache receipt fields differ")
    if receipt["schema_version"] != PARSER_CACHE_RECEIPT_SCHEMA:
        raise ValueError("Full128 parser cache receipt schema differs")
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    _require_sha256(receipt["receipt_sha256"], "parser cache receipt")
    if content_sha256(payload) != receipt["receipt_sha256"]:
        raise ValueError("Full128 parser cache receipt digest differs")
    _validate_runtime_binding(receipt["runtime"], bundle=bundle)
    first = rows[0]
    if any(
        row["source_sha256"] != first["source_sha256"]
        or row["source_width"] != first["source_width"]
        or row["source_height"] != first["source_height"]
        or row["parser_cache_key"] != first["parser_cache_key"]
        for row in rows
    ):
        raise ValueError("Full128 parser cache job source bindings differ")
    expected = {
        "parser_cache_key": first["parser_cache_key"],
        "source_sha256": first["source_sha256"],
        "source_width": first["source_width"],
        "source_height": first["source_height"],
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise ValueError(f"Full128 parser cache receipt {field} differs")
    _require_sha256(receipt["prediction_sha256"], "parser cache prediction")
    _require_sha256(receipt["frozen_json_sha256"], "parser cache frozen JSON")
    frozen_json_sha256 = (
        _sha256_path(cache_dir / "frozen.json")
        if observed_frozen_json_sha256 is None
        else observed_frozen_json_sha256
    )
    if frozen_json_sha256 != receipt["frozen_json_sha256"]:
        raise ValueError("Full128 parser cache frozen JSON digest differs")
    return receipt


def _publish_sample(
    target: Path,
    *,
    row: dict[str, Any],
    bundle: Mapping[str, Any],
    inputs: _JobInputs,
    decision: _Decision,
    frozen: dict[str, Any] | None,
    parser_cache_receipt: dict[str, Any] | None,
    derived: tuple[bytes, dict[str, Any]] | None,
    shard_selection: dict[str, Any],
) -> None:
    from identity_methods.full_segment.sample_materialization import (
        REQUEST_SCHEMA as MATERIALIZATION_REQUEST_SCHEMA,
    )
    from identity_methods.full_segment.sample_materialization import (
        run_prevalidated as materialize_one,
    )

    with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        temporary_root = Path(temporary)
        effective_path = inputs.source.path
        effective_sha256 = row["source_sha256"]
        if derived is not None:
            derived_bytes, _ = derived
            effective_path = temporary_root / "derived-source.png"
            _write_new(effective_path, derived_bytes)
            effective_sha256 = hashlib.sha256(derived_bytes).hexdigest()
        request: dict[str, Any] = {
            "schema_version": MATERIALIZATION_REQUEST_SCHEMA,
            "source_id": row["sample_token"],
            "source_image_path": os.fspath(effective_path),
            "source_sha256": effective_sha256,
            "source_view_scope": decision.source_view_scope.value,
            "route": decision.actual_route.value,
            "frozen_parsing_path": None,
            "association": None,
            "face_observability": (
                "NATIVE"
                if decision.actual_route
                in {ObservationRoute.NATIVE_FACE, ObservationRoute.NATIVE_HEAD}
                else "NOT_RUN"
            ),
            "nose_observability": "NOT_RUN",
            "target_size": row["target_size"],
            "context_fraction": row["context_fraction"],
            "background_rgb": row["background_rgb"],
        }
        if decision.actual_route is ObservationRoute.BODY_PARSING:
            request["frozen_parsing_path"] = os.fspath(
                target.parent.parent
                / "parser-cache"
                / row["parser_cache_key"]
                / "frozen.json"
            )
            request["association"] = decision.association.to_dict()
        elif decision.actual_route is ObservationRoute.BODY_MASK:
            evidence = row["route_evidence"]
            policy = BodyMaskPolicy(
                BodyMaskPolicyKind.OXFORD_IIIT_PET_TRIMAP,
                permitted_labels=tuple(evidence["observed_labels"]),
            )
            mask = inputs.evidence["trimap_artifact"]
            request.update(
                {
                    "authoritative_mask_path": os.fspath(mask.path),
                    "authoritative_mask_sha256": mask.sha256,
                    "body_mask_policy": policy.to_dict(),
                }
            )
        staging = temporary_root / "sample"
        materialize_one(
            request,
            output_dir=staging,
            source_bytes=(derived[0] if derived is not None else inputs.source.payload),
            frozen_parsing=(
                frozen
                if decision.actual_route is ObservationRoute.BODY_PARSING
                else None
            ),
            frozen_json_sha256=(
                parser_cache_receipt["frozen_json_sha256"]
                if decision.actual_route is ObservationRoute.BODY_PARSING
                and parser_cache_receipt is not None
                else None
            ),
        )
        if derived is not None:
            derived_bytes, lineage = derived
            _write_new(staging / "derived-source.png", derived_bytes)
            _write_new(staging / "lineage-receipt.json", json_document_bytes(lineage))
        outputs = _sample_output_metadata(staging)
        derived_lineage = (
            None
            if derived is None
            else {
                "lineage_receipt_sha256": derived[1]["lineage_sha256"],
                "derived_source_sha256": derived[1]["child_sha256"],
                "child_width": derived[1]["child_width"],
                "child_height": derived[1]["child_height"],
            }
        )
        receipt_payload = {
            "schema_version": EXECUTION_RECEIPT_SCHEMA,
            "sample_token": row["sample_token"],
            "plan_record_sha256": row["record_sha256"],
            "plan_sha256": bundle["plan_sha256"],
            "original_source_sha256": row["source_sha256"],
            "effective_source_sha256": effective_sha256,
            "route_intent": row["route_intent"],
            "actual_route": decision.actual_route.value,
            "parser_lineage": decision.parser_lineage,
            "derived_lineage": derived_lineage,
            "terminal_reason": decision.terminal_reason,
            "shard_selection": shard_selection,
            "outputs": outputs,
        }
        receipt = {
            **receipt_payload,
            "receipt_sha256": content_sha256(receipt_payload),
        }
        _write_new(staging / "execution-receipt.json", json_document_bytes(receipt))
        for path in staging.iterdir():
            if path.is_file():
                os.chmod(path, 0o600)
        fsync_directory(staging)
        try:
            rename_directory_noreplace(staging, target)
        except FileExistsError:
            pass
    fsync_directory(target.parent)
    _validate_sample_output(
        target,
        row=row,
        bundle=bundle,
        inputs=inputs,
        decision=decision,
        frozen=frozen,
        parser_cache_receipt=parser_cache_receipt,
        derived=derived,
    )


def _fast_validate_sample_output(
    target: Path,
    *,
    row: Mapping[str, Any],
    bundle: Mapping[str, Any],
    parser_cache_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate resumable output without reopening large semantic artifacts."""

    receipt_path = target / "execution-receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            "Full128 sample output is partial or lacks an execution receipt"
        )
    receipt = read_strict_json_document(receipt_path, maximum_bytes=4_194_304).payload
    if not isinstance(receipt, dict) or set(receipt) != _EXECUTION_FIELDS:
        raise ValueError("Full128 execution receipt fields differ")
    if receipt["schema_version"] != EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("Full128 execution receipt schema differs")
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    _require_sha256(receipt["receipt_sha256"], "execution receipt")
    if content_sha256(payload) != receipt["receipt_sha256"]:
        raise ValueError("Full128 execution receipt digest differs")
    expected = {
        "sample_token": row["sample_token"],
        "plan_record_sha256": row["record_sha256"],
        "plan_sha256": bundle["plan_sha256"],
        "original_source_sha256": row["source_sha256"],
        "route_intent": row["route_intent"],
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise ValueError(f"Full128 execution receipt {field} differs")
    _require_sha256(receipt["effective_source_sha256"], "effective source")
    _validate_recorded_shard(receipt["shard_selection"], row=row)

    intent = RouteIntent(row["route_intent"])
    try:
        actual_route = ObservationRoute(receipt["actual_route"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Full128 execution receipt actual route differs") from exc
    direct_routes = {
        RouteIntent.BODY_MASK: ObservationRoute.BODY_MASK,
        RouteIntent.NATIVE_FACE: ObservationRoute.NATIVE_FACE,
        RouteIntent.DERIVED_NATIVE_FACE: ObservationRoute.NATIVE_FACE,
        RouteIntent.DERIVED_NATIVE_HEAD: ObservationRoute.NATIVE_HEAD,
    }
    if intent is RouteIntent.BODY_PARSING:
        if actual_route not in {ObservationRoute.BODY_PARSING, ObservationRoute.NONE}:
            raise ValueError("Full128 parsed execution route differs")
        if parser_cache_receipt is None:
            raise ValueError("Full128 parsed execution lacks parser cache receipt")
        lineage = receipt["parser_lineage"]
        lineage_fields = {
            "parser_cache_key",
            "parser_cache_receipt_sha256",
            "prediction_sha256",
            "association",
            "association_authority",
        }
        if row["schema_version"] == "cvi.full128_route_plan_record.v3":
            lineage_fields.add("selection")
        if not isinstance(lineage, dict) or set(lineage) != lineage_fields:
            raise ValueError("Full128 parser lineage fields differ")
        lineage_expected = {
            "parser_cache_key": parser_cache_receipt["parser_cache_key"],
            "parser_cache_receipt_sha256": parser_cache_receipt["receipt_sha256"],
            "prediction_sha256": parser_cache_receipt["prediction_sha256"],
        }
        if any(lineage[field] != value for field, value in lineage_expected.items()):
            raise ValueError("Full128 parser lineage differs from parser cache receipt")
        if lineage["association"] is not None:
            AnimalAssociation.from_dict(lineage["association"])
    else:
        if (
            actual_route is not direct_routes[intent]
            or receipt["parser_lineage"] is not None
        ):
            raise ValueError("Full128 direct execution route or parser lineage differs")
    if (actual_route is ObservationRoute.NONE) != (
        receipt["terminal_reason"] is not None
    ):
        raise ValueError("Full128 execution terminal reason differs")

    derived = intent in {
        RouteIntent.DERIVED_NATIVE_FACE,
        RouteIntent.DERIVED_NATIVE_HEAD,
    }
    expected_files = {
        "full-segment-observation.json",
        "full-segment-cache.json",
        "execution-receipt.json",
    }
    if actual_route is not ObservationRoute.NONE:
        expected_files.update({"full.png", "full-mask.png"})
    if derived:
        expected_files.update({"derived-source.png", "lineage-receipt.json"})
    _require_exact_directory(target, expected_files, "sample output")

    outputs = receipt["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != _OUTPUT_FIELDS:
        raise ValueError("Full128 execution output fields differ")
    for field in _OUTPUT_FIELDS:
        value = outputs[field]
        if value is not None:
            _require_sha256(value, f"execution output {field}")
    cache_path = target / "full-segment-cache.json"
    cache_stat = cache_path.stat(follow_symlinks=False)
    if not stat.S_ISREG(cache_stat.st_mode) or cache_stat.st_size <= 0:
        raise ValueError("Full128 sample cache file type or size differs")
    if _sha256_path(cache_path) != outputs["full_segment_cache_file_sha256"]:
        raise ValueError("Full128 fast-resume sample cache digest differs")

    observation = FullSegmentObservation.from_dict(
        read_strict_json_document(
            target / "full-segment-observation.json", maximum_bytes=16_777_216
        ).payload
    )
    if (
        observation.source_id != row["sample_token"]
        or observation.source_sha256 != receipt["effective_source_sha256"]
        or observation.route is not actual_route
        or observation.observation_sha256 != outputs["observation_sha256"]
    ):
        raise ValueError(
            "Full128 fast-resume observation differs from execution receipt"
        )
    if intent is RouteIntent.BODY_PARSING:
        association = receipt["parser_lineage"]["association"]
        if actual_route is ObservationRoute.BODY_PARSING:
            if (
                observation.parsing_prediction_sha256
                != parser_cache_receipt["prediction_sha256"]
            ):
                raise ValueError(
                    "Full128 fast-resume observation parser digest differs"
                )
            if (
                None
                if observation.association is None
                else observation.association.to_dict()
            ) != association:
                raise ValueError("Full128 fast-resume parser association differs")
        elif (
            observation.parsing_prediction_sha256 is not None
            or observation.association is not None
        ):
            raise ValueError("Full128 terminal parser observation retains parsing")

    if actual_route is ObservationRoute.NONE:
        if any(
            outputs[field] is not None
            for field in (
                "crop_record_sha256",
                "full_rgb_sha256",
                "full_mask_sha256",
            )
        ):
            raise ValueError("Full128 terminal execution retains crop hashes")
    else:
        for filename, field in (
            ("full.png", "full_rgb_sha256"),
            ("full-mask.png", "full_mask_sha256"),
        ):
            if _sha256_path(target / filename) != outputs[field]:
                raise ValueError(f"Full128 fast-resume {filename} digest differs")

    if derived:
        lineage_document = read_strict_json_document(
            target / "lineage-receipt.json", maximum_bytes=1_048_576
        )
        lineage = validate_derived_lineage_receipt(
            lineage_document.payload,
            expected_parent_sha256=row["source_sha256"],
            expected_child_sha256=receipt["effective_source_sha256"],
        )
        expected_derived = {
            "lineage_receipt_sha256": lineage["lineage_sha256"],
            "derived_source_sha256": lineage["child_sha256"],
            "child_width": lineage["child_width"],
            "child_height": lineage["child_height"],
        }
        if (
            receipt["derived_lineage"] != expected_derived
            or outputs["lineage_receipt_file_sha256"] != lineage_document.raw_sha256
            or outputs["lineage_receipt_sha256"] != lineage["lineage_sha256"]
            or outputs["derived_source_sha256"] != lineage["child_sha256"]
            or _sha256_path(target / "derived-source.png") != lineage["child_sha256"]
        ):
            raise ValueError("Full128 fast-resume derived lineage differs")
    elif receipt["derived_lineage"] is not None or any(
        outputs[field] is not None
        for field in (
            "derived_source_sha256",
            "lineage_receipt_file_sha256",
            "lineage_receipt_sha256",
        )
    ):
        raise ValueError("Full128 non-derived execution retains derived lineage")
    return receipt


def _validate_sample_output(
    target: Path,
    *,
    row: Mapping[str, Any],
    bundle: Mapping[str, Any],
    inputs: _JobInputs,
    decision: _Decision,
    frozen: dict[str, Any] | None,
    parser_cache_receipt: dict[str, Any] | None,
    derived: tuple[bytes, dict[str, Any]] | None,
) -> _ValidatedSample:
    expected_files = {
        "full-segment-observation.json",
        "full-segment-cache.json",
        "execution-receipt.json",
    }
    if decision.actual_route is not ObservationRoute.NONE:
        expected_files.update({"full.png", "full-mask.png"})
    if derived is not None:
        expected_files.update({"derived-source.png", "lineage-receipt.json"})
    _require_exact_directory(target, expected_files, "sample output")
    cache_document = read_strict_json_document(
        target / "full-segment-cache.json",
        maximum_bytes=_MAX_JSON_BYTES,
        maximum_string_characters=_MAX_JSON_BYTES,
    )
    cache = validate_full_segment_cache_bundle(cache_document.payload)
    if len(cache["records"]) != 1:
        raise ValueError("Full128 sample cache must contain exactly one record")
    cache_record = cache["records"][0]
    observation_document = read_strict_json_document(
        target / "full-segment-observation.json", maximum_bytes=16_777_216
    )
    observation = FullSegmentObservation.from_dict(observation_document.payload)
    if (
        cache_record["source_id"] != row["sample_token"]
        or cache_record["observation"] != observation.to_dict()
        or observation.source_id != row["sample_token"]
        or observation.route is not decision.actual_route
        or observation.source_view_scope is not decision.source_view_scope
    ):
        raise ValueError("Full128 sample observation differs from its execution")
    effective_sha256 = row["source_sha256"]
    effective_dimensions = (row["source_width"], row["source_height"])
    lineage_receipt_file_sha256 = None
    lineage_receipt_sha256 = None
    if derived is not None:
        expected_bytes, expected_lineage = derived
        actual_bytes = _read_regular_absolute(
            target / "derived-source.png",
            maximum_bytes=_MAX_SOURCE_BYTES,
            label="Full128 derived source",
        )
        if actual_bytes != expected_bytes:
            raise ValueError("Full128 derived source differs from deterministic crop")
        lineage_document = read_strict_json_document(
            target / "lineage-receipt.json", maximum_bytes=1_048_576
        )
        lineage = validate_derived_lineage_receipt(
            lineage_document.payload,
            expected_parent_sha256=row["source_sha256"],
            expected_child_sha256=hashlib.sha256(actual_bytes).hexdigest(),
        )
        if lineage != expected_lineage:
            raise ValueError("Full128 derived lineage differs from route-plan evidence")
        lineage_receipt_file_sha256 = lineage_document.raw_sha256
        lineage_receipt_sha256 = lineage["lineage_sha256"]
        effective_sha256 = lineage["child_sha256"]
        effective_dimensions = (lineage["child_width"], lineage["child_height"])
    if (
        observation.source_sha256 != effective_sha256
        or (
            observation.source_width,
            observation.source_height,
        )
        != effective_dimensions
    ):
        raise ValueError("Full128 effective source differs from observation")
    if decision.actual_route is ObservationRoute.BODY_PARSING:
        if frozen is None or parser_cache_receipt is None:
            raise ValueError("Full128 parsed sample lacks shared parser provenance")
        _validate_sample_parser_binding(
            cache_record["frozen_parsing"],
            cache_schema=cache["schema_version"],
            observation=observation,
            frozen=frozen,
            parser_cache_receipt=parser_cache_receipt,
        )
        if observation.association != decision.association:
            raise ValueError("Full128 sample parser association differs")
    elif cache_record["frozen_parsing"] is not None:
        raise ValueError("Full128 non-parsed sample retains frozen parsing")
    crop = cache_record["crop"]
    if decision.actual_route is ObservationRoute.NONE:
        if crop is not None:
            raise ValueError("Full128 terminal sample cannot retain a crop")
    else:
        if crop is None:
            raise ValueError("Full128 observable sample lacks a crop")
        rgb = _read_regular_absolute(
            target / "full.png", maximum_bytes=_MAX_SOURCE_BYTES, label="Full RGB"
        )
        mask = _read_regular_absolute(
            target / "full-mask.png",
            maximum_bytes=_MAX_SOURCE_BYTES,
            label="Full mask",
        )
        verify_full_crop_artifacts(crop, rgb, mask)
    outputs = {
        "full_segment_cache_file_sha256": cache_document.raw_sha256,
        "full_segment_cache_sha256": cache_document.payload["cache_sha256"],
        "observation_sha256": observation.observation_sha256,
        "crop_record_sha256": None if crop is None else crop["crop_record_sha256"],
        "full_rgb_sha256": None if crop is None else crop["full_rgb_sha256"],
        "full_mask_sha256": None if crop is None else crop["full_mask_sha256"],
        "derived_source_sha256": None if derived is None else effective_sha256,
        "lineage_receipt_file_sha256": lineage_receipt_file_sha256,
        "lineage_receipt_sha256": lineage_receipt_sha256,
    }
    if set(outputs) != _OUTPUT_FIELDS:
        raise AssertionError("Full128 output metadata implementation differs")
    receipt = read_strict_json_document(
        target / "execution-receipt.json", maximum_bytes=4_194_304
    ).payload
    if not isinstance(receipt, dict) or set(receipt) != _EXECUTION_FIELDS:
        raise ValueError("Full128 execution receipt fields differ")
    if receipt["schema_version"] != EXECUTION_RECEIPT_SCHEMA:
        raise ValueError("Full128 execution receipt schema differs")
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    _require_sha256(receipt["receipt_sha256"], "execution receipt")
    if content_sha256(payload) != receipt["receipt_sha256"]:
        raise ValueError("Full128 execution receipt digest differs")
    expected_derived_lineage = (
        None
        if derived is None
        else {
            "lineage_receipt_sha256": derived[1]["lineage_sha256"],
            "derived_source_sha256": derived[1]["child_sha256"],
            "child_width": derived[1]["child_width"],
            "child_height": derived[1]["child_height"],
        }
    )
    expected = {
        "sample_token": row["sample_token"],
        "plan_record_sha256": row["record_sha256"],
        "plan_sha256": bundle["plan_sha256"],
        "original_source_sha256": row["source_sha256"],
        "effective_source_sha256": effective_sha256,
        "route_intent": row["route_intent"],
        "actual_route": decision.actual_route.value,
        "parser_lineage": decision.parser_lineage,
        "derived_lineage": expected_derived_lineage,
        "terminal_reason": decision.terminal_reason,
        "outputs": outputs,
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise ValueError(f"Full128 execution receipt {field} differs")
    _validate_recorded_shard(receipt["shard_selection"], row=row)
    return _ValidatedSample(
        observation=observation,
        execution_receipt=receipt,
        full_segment_cache_file_sha256=cache_document.raw_sha256,
        full_segment_cache_sha256=cache_document.payload["cache_sha256"],
        lineage_receipt_file_sha256=lineage_receipt_file_sha256,
        lineage_receipt_sha256=lineage_receipt_sha256,
        crop_record_sha256=None if crop is None else crop["crop_record_sha256"],
        full_rgb_sha256=None if crop is None else crop["full_rgb_sha256"],
        full_mask_sha256=None if crop is None else crop["full_mask_sha256"],
    )


def _validate_sample_parser_binding(
    value: object,
    *,
    cache_schema: str,
    observation: FullSegmentObservation,
    frozen: dict[str, Any],
    parser_cache_receipt: Mapping[str, Any],
) -> None:
    if cache_schema == LEGACY_CACHE_SCHEMA:
        if value != frozen:
            raise ValueError("Full128 sample frozen parsing differs from parser cache")
        return
    if cache_schema != CACHE_SCHEMA or not isinstance(value, dict):
        raise ValueError("Full128 sample frozen parsing binding schema differs")
    if observation.association is None:
        raise ValueError("Full128 parsed sample lacks an observation association")
    expected = _expected_frozen_parsing_binding(
        frozen,
        frozen_json_sha256=parser_cache_receipt["frozen_json_sha256"],
        source_id=observation.source_id,
        source_sha256=observation.source_sha256,
        association=observation.association,
    )
    if value != expected:
        raise ValueError(
            "Full128 sample frozen parsing binding differs from parser cache"
        )


def _expected_frozen_parsing_binding(
    frozen: Mapping[str, Any],
    *,
    frozen_json_sha256: str,
    source_id: str,
    source_sha256: str,
    association: AnimalAssociation,
) -> dict[str, Any]:
    prediction = frozen["prediction"]
    return {
        "schema_version": FROZEN_PARSING_BINDING_SCHEMA,
        "frozen_schema_version": FROZEN_PARSING_SCHEMA,
        "prediction_sha256": frozen["prediction_sha256"],
        "frozen_content_sha256": content_sha256(frozen),
        "frozen_json_sha256": frozen_json_sha256,
        "policy_sha256": prediction["policy_sha256"],
        "source_id": source_id,
        "source_sha256": source_sha256,
        "source_width": prediction["source_width"],
        "source_height": prediction["source_height"],
        "instance_count": len(prediction["instances"]),
        "association": association.to_dict(),
    }


def _migrate_sample_cache_directory(
    target: Path,
    *,
    row: Mapping[str, Any],
    bundle: Mapping[str, Any],
    inputs: _JobInputs,
    decision: _Decision,
    frozen: dict[str, Any],
    parser_cache_receipt: Mapping[str, Any],
    derived: tuple[bytes, dict[str, Any]] | None,
    validated: _ValidatedSample,
) -> int | None:
    observation = validated.observation
    if observation.association is None:
        raise ValueError("Full128 parsed migration lacks an observation association")
    old_cache = validate_full_segment_cache_bundle(
        read_strict_json_document(
            target / "full-segment-cache.json",
            maximum_bytes=_MAX_JSON_BYTES,
            maximum_string_characters=_MAX_JSON_BYTES,
        ).payload
    )
    record = old_cache["records"][0]
    binding = _expected_frozen_parsing_binding(
        frozen,
        frozen_json_sha256=parser_cache_receipt["frozen_json_sha256"],
        source_id=observation.source_id,
        source_sha256=observation.source_sha256,
        association=observation.association,
    )
    compact_bundle = build_full_segment_cache(
        (
            {
                "source_id": record["source_id"],
                "observation": record["observation"],
                "frozen_parsing": binding,
                "crop": record["crop"],
            },
        )
    )
    compact_bytes = json_document_bytes(compact_bundle)
    target_identity = _node_identity(target.stat(follow_symlinks=False))
    with TemporaryDirectory(
        prefix=f".{target.name}-migration-", dir=target.parent
    ) as temporary:
        staging = Path(temporary) / "sample"
        staging.mkdir(mode=0o700)
        for source in target.iterdir():
            if source.name in {"full-segment-cache.json", "execution-receipt.json"}:
                continue
            payload = _read_regular_absolute(
                source,
                maximum_bytes=_MAX_JSON_BYTES,
                label=f"Full128 migration source {source.name}",
            )
            _write_new(staging / source.name, payload)
        _write_new(staging / "full-segment-cache.json", compact_bytes)
        outputs = _sample_output_metadata(staging)
        receipt_payload = {
            key: value
            for key, value in validated.execution_receipt.items()
            if key != "receipt_sha256"
        }
        receipt_payload["outputs"] = outputs
        migrated_receipt = {
            **receipt_payload,
            "receipt_sha256": content_sha256(receipt_payload),
        }
        _write_new(
            staging / "execution-receipt.json", json_document_bytes(migrated_receipt)
        )
        for path in staging.iterdir():
            os.chmod(path, 0o600)
        fsync_directory(staging)
        _validate_sample_output(
            staging,
            row=row,
            bundle=bundle,
            inputs=inputs,
            decision=decision,
            frozen=frozen,
            parser_cache_receipt=dict(parser_cache_receipt),
            derived=derived,
        )
        if _node_identity(target.stat(follow_symlinks=False)) != target_identity:
            raise RuntimeError("Full128 sample changed during compact migration")
        staging_identity = _node_identity(staging.stat(follow_symlinks=False))
        if not _atomic_exchange_directories(staging, target):
            return None
        fsync_directory(target.parent)
        exchanged = True
        try:
            if (
                _node_identity(target.stat(follow_symlinks=False)) != staging_identity
                or _node_identity(staging.stat(follow_symlinks=False))
                != target_identity
            ):
                raise RuntimeError(
                    "Full128 compact migration exchange identity differs"
                )
            _validate_sample_output(
                target,
                row=row,
                bundle=bundle,
                inputs=inputs,
                decision=decision,
                frozen=frozen,
                parser_cache_receipt=dict(parser_cache_receipt),
                derived=derived,
            )
            exchanged = False
        finally:
            if exchanged:
                if not _atomic_exchange_directories(staging, target):
                    raise RuntimeError("Full128 compact migration rollback unavailable")
                fsync_directory(target.parent)
        return len(compact_bytes)


def _atomic_exchange_directories(left: Path, right: Path) -> bool:
    if os.name != "posix":
        return False
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2)
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        return False
    raise OSError(error_number, os.strerror(error_number), right)


def _sample_output_metadata(directory: Path) -> dict[str, Any]:
    cache_document = read_strict_json_document(
        directory / "full-segment-cache.json",
        maximum_bytes=_MAX_JSON_BYTES,
        maximum_string_characters=_MAX_JSON_BYTES,
    )
    cache = validate_full_segment_cache_bundle(cache_document.payload)
    if len(cache["records"]) != 1:
        raise ValueError("Full128 sample cache must contain exactly one record")
    record = cache["records"][0]
    observation = FullSegmentObservation.from_dict(record["observation"])
    crop = record["crop"]
    lineage_document = (
        read_strict_json_document(
            directory / "lineage-receipt.json", maximum_bytes=1_048_576
        )
        if (directory / "lineage-receipt.json").exists()
        else None
    )
    values = {
        "full_segment_cache_file_sha256": cache_document.raw_sha256,
        "full_segment_cache_sha256": cache_document.payload["cache_sha256"],
        "observation_sha256": observation.observation_sha256,
        "crop_record_sha256": None if crop is None else crop["crop_record_sha256"],
        "full_rgb_sha256": None if crop is None else crop["full_rgb_sha256"],
        "full_mask_sha256": None if crop is None else crop["full_mask_sha256"],
        "derived_source_sha256": (
            None
            if not (directory / "derived-source.png").exists()
            else _sha256_path(directory / "derived-source.png")
        ),
        "lineage_receipt_file_sha256": (
            None if lineage_document is None else lineage_document.raw_sha256
        ),
        "lineage_receipt_sha256": (
            None
            if lineage_document is None
            else validate_derived_lineage_receipt(lineage_document.payload)[
                "lineage_sha256"
            ]
        ),
    }
    if set(values) != _OUTPUT_FIELDS:
        raise AssertionError("Full128 output metadata implementation differs")
    return values


def _derived_artifacts(
    row: Mapping[str, Any], inputs: _JobInputs
) -> tuple[bytes, dict[str, Any]] | None:
    route = RouteIntent(row["route_intent"])
    if route not in {
        RouteIntent.DERIVED_NATIVE_FACE,
        RouteIntent.DERIVED_NATIVE_HEAD,
    }:
        return None
    evidence = row["route_evidence"]
    if route is RouteIntent.DERIVED_NATIVE_FACE:
        bbox = evidence["bbox_xyxy"]
        binding = evidence["label_artifact"]
    else:
        bbox = evidence["head_bbox_xyxy"]
        binding = evidence["head_artifact"]
    x1, y1, x2, y2 = _finite_box(bbox, integer=False)
    aligned = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    if not (
        0 <= aligned[0] < aligned[2] <= row["source_width"]
        and 0 <= aligned[1] < aligned[3] <= row["source_height"]
    ):
        raise ValueError("Full128 derived bbox exceeds the parent source")
    with inputs.source_image.convert("RGB") as image:
        crop = image.crop(aligned)
        output = io.BytesIO()
        crop.save(output, format="PNG", optimize=False, compress_level=9)
        child = output.getvalue()
    child_sha256 = hashlib.sha256(child).hexdigest()
    payload = {
        "schema_version": DERIVED_LINEAGE_SCHEMA,
        "sample_token": row["sample_token"],
        "route_intent": route.value,
        "parent_source_sha256": row["source_sha256"],
        "evidence_artifact": dict(binding),
        "bbox_xyxy": list(bbox),
        "aligned_bbox_xyxy": list(aligned),
        "crop_policy": {
            "minimum_rounding": "FLOOR",
            "maximum_rounding": "CEIL",
            "color_mode": "RGB",
        },
        "child_encoding": "PNG_RGB_LOSSLESS",
        "child_sha256": child_sha256,
        "child_width": aligned[2] - aligned[0],
        "child_height": aligned[3] - aligned[1],
    }
    lineage = {**payload, "lineage_sha256": content_sha256(payload)}
    validate_derived_lineage_receipt(
        lineage,
        expected_parent_sha256=row["source_sha256"],
        expected_child_sha256=child_sha256,
    )
    return child, lineage


def _load_job_inputs(rows: Sequence[dict[str, Any]]) -> _JobInputs:
    first = rows[0]
    if any(
        row["source_path"] != first["source_path"]
        or row["source_sha256"] != first["source_sha256"]
        or row["source_byte_size"] != first["source_byte_size"]
        or row["source_width"] != first["source_width"]
        or row["source_height"] != first["source_height"]
        for row in rows
    ):
        raise ValueError("Full128 source job contains inconsistent source bindings")
    dataset = get_record(first["dataset_name"])
    root = _canonical_root(Path(dataset.data_root), dataset.canonical_name)
    source = _read_bound_artifact(
        root,
        {
            "relative_path": first["source_path"],
            "sha256": first["source_sha256"],
            "byte_size": first["source_byte_size"],
        },
        label="Full128 source image",
        maximum_bytes=_MAX_SOURCE_BYTES,
    )
    source_image = _image_from_bytes(source.payload, "Full128 source image")
    if source_image.size != (
        first["source_width"],
        first["source_height"],
    ):
        raise ValueError("Full128 source dimensions differ from route plan")
    evidence: dict[str, _ArtifactInput] = {}
    loaded_artifacts: dict[tuple[str, str, int], _ArtifactInput] = {}
    for row in rows:
        for field in (
            "annotation_artifact",
            "label_artifact",
            "trimap_artifact",
            "head_artifact",
        ):
            binding = row["route_evidence"].get(field)
            if binding is None:
                continue
            key = (binding["relative_path"], binding["sha256"], binding["byte_size"])
            loaded = loaded_artifacts.get(key)
            if loaded is None:
                loaded = _read_bound_artifact(
                    root,
                    binding,
                    label=f"Full128 {field}",
                    maximum_bytes=max(_MAX_SOURCE_BYTES, binding["byte_size"]),
                )
                loaded_artifacts[key] = loaded
            prior = evidence.setdefault(field, loaded)
            if prior.sha256 != loaded.sha256 or prior.path != loaded.path:
                raise ValueError(f"Full128 job mixes {field} bindings")
    return _JobInputs(root, source, source_image, evidence)


def _unified_observation(
    row: Mapping[str, Any], observation: FullSegmentObservation
) -> UnifiedFullObservation:
    dataset = get_record(row["dataset_name"])
    name = dataset.canonical_name
    identity_metadata = row["identity_metadata"]
    if name in {"ap10k-dog", "dogflw", "oxford-pets-dog"}:
        identity_kind = IdentityEvidenceKind.NONE
        identity_namespace = None
        identity_token = None
        fixed_role = TerminalRole.AUXILIARY
    elif name == "yt-bb-dog":
        raw_identity = identity_metadata["raw_identity_id"]
        if not isinstance(raw_identity, str) or not raw_identity:
            raise ValueError("YT-BB Full128 row lacks a video-track label")
        cluster = compute_source_cluster_token(
            f"{name}\0{row['dataset_version']}\0video-track\0{raw_identity}"
        )
        identity_kind = IdentityEvidenceKind.GENERATED
        identity_namespace = str(GENERATED_DOG_NAMESPACE)
        identity_token = compute_generated_identity_id(
            "cvi.full128.yt-bb-dog.video-track:v1", cluster
        )
        fixed_role = TerminalRole.EVAL if row["split"] == "test" else None
    else:
        registered = identity_metadata["registered_identity_id"]
        if not isinstance(registered, str):
            raise ValueError(f"{name} Full128 row lacks registered identity authority")
        raw_identity = identity_metadata["raw_identity_id"]
        if not isinstance(raw_identity, str) or not raw_identity:
            raise ValueError(f"{name} Full128 row lacks a registered source label")
        dataset_identity = {
            "dogfacenet224": f"dogfacenet224:v1:web-folder:{raw_identity}",
            "mpdd": f"mpdd:v1:device-capture:{raw_identity}",
            "sibetan": f"sibetan:v1:gt-json:{raw_identity}",
        }.get(name)
        if dataset_identity is None or registered != compute_registered_dog_id(
            dataset_identity
        ):
            raise ValueError(f"{name} Full128 registered UUIDv5 authority differs")
        identity_kind = IdentityEvidenceKind.REGISTERED
        identity_namespace = str(REGISTERED_DOG_NAMESPACE)
        identity_token = registered
        fixed_role = (
            TerminalRole.EVAL
            if name == "sibetan"
            or (name == "mpdd" and row["split"] in {"query", "gallery"})
            else None
        )
    validation_only = dataset.admission is DatasetAdmission.ADMIT_VALIDATION_ONLY
    gradient_eligible = (
        dataset.admission is DatasetAdmission.ADMIT_TRAIN
        and identity_kind is not IdentityEvidenceKind.NONE
        and fixed_role is not TerminalRole.EVAL
    )
    capture = row["capture_metadata"]
    source_group = _group_token(name, "source", capture["source_group_id"])
    capture_group = (
        source_group
        if capture["capture_group_id"] is None
        else _group_token(name, "capture", capture["capture_group_id"])
    )
    adapter_metadata = row["source_metadata"]["adapter_metadata"]
    sequence = adapter_metadata.get("unverified_sequence_token")
    raw_identity = identity_metadata["raw_identity_id"]
    sequence_group = (
        _group_token(name, "sequence", f"{raw_identity}\0{sequence}")
        if isinstance(sequence, str) and sequence
        else capture_group
    )
    return UnifiedFullObservation(
        dataset_name=name,
        official_split=row["split"],
        identity_evidence_kind=identity_kind,
        identity_namespace_uuid=identity_namespace,
        identity_token=identity_token,
        sample_token=row["sample_token"],
        source_group=source_group,
        capture_group=capture_group,
        sequence_group=sequence_group,
        duplicate_component=row["duplicate_component"],
        gradient_eligible=gradient_eligible,
        validation_only=validation_only,
        full_status=FullStatus(observation.full_status.value),
        face_status=RegionStatus(observation.face_observability.value),
        nose_status=RegionStatus(observation.nose_observability.value),
        view_scope=ViewScope(observation.source_view_scope.value),
        source_observation_sha256=observation.observation_sha256,
        terminal_role=fixed_role,
    )


def _inventory_request_row(
    row: Mapping[str, Any],
    *,
    output_root: Path,
    validated: _ValidatedSample,
    terminal_role: str,
) -> dict[str, Any]:
    sample_dir = output_root / "samples" / row["sample_token"]
    unified = _unified_observation(row, validated.observation)
    execution = validated.execution_receipt
    return {
        "dataset_name": row["dataset_name"],
        "dataset_version": row["dataset_version"],
        "official_split": row["split"],
        "identity_evidence_kind": unified.identity_evidence_kind.value,
        "identity_namespace_uuid": unified.identity_namespace_uuid,
        "identity_token": unified.identity_token,
        "sample_token": row["sample_token"],
        "source_group": unified.source_group,
        "capture_group": unified.capture_group,
        "sequence_group": unified.sequence_group,
        "duplicate_component": row["duplicate_component"],
        "terminal_role": terminal_role,
        "original_source_sha256": execution["original_source_sha256"],
        "effective_source_sha256": execution["effective_source_sha256"],
        "lineage_receipt_path": (
            None
            if execution["derived_lineage"] is None
            else os.fspath(sample_dir / "lineage-receipt.json")
        ),
        "full_segment_cache_path": os.fspath(sample_dir / "full-segment-cache.json"),
        "full_rgb_path": os.fspath(sample_dir / "full.png"),
        "full_mask_path": os.fspath(sample_dir / "full-mask.png"),
    }


def _selected_jobs(
    records: Sequence[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
    maximum_jobs: int | None,
) -> tuple[tuple[str, tuple[dict[str, Any], ...]], ...]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = row["parser_cache_key"] or row["source_sha256"]
        grouped[(key, row["dataset_name"], row["source_path"])].append(row)
    jobs = []
    for (key, _dataset_name, _source_path), rows in grouped.items():
        if int(key, 16) % shard_count != shard_index:
            continue
        jobs.append((key, tuple(sorted(rows, key=lambda item: item["sample_token"]))))
    jobs.sort(
        key=lambda item: (
            item[1][0]["source_sha256"],
            item[1][0]["dataset_name"],
            item[1][0]["source_path"],
            item[0],
        )
    )
    if maximum_jobs is not None:
        jobs = jobs[:maximum_jobs]
    return tuple(jobs)


def _shard_selection(
    assignment_key: str, *, shard_count: int, shard_index: int
) -> dict[str, Any]:
    return {
        "schema_version": SHARD_SELECTION_SCHEMA,
        "assignment_kind": "PARSER_CACHE_KEY_OR_SOURCE_SHA256",
        "assignment_key": assignment_key,
        "shard_count": shard_count,
        "assigned_shard": int(assignment_key, 16) % shard_count,
        "executed_shard": shard_index,
    }


def _validate_recorded_shard(value: object, *, row: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "assignment_kind",
        "assignment_key",
        "shard_count",
        "assigned_shard",
        "executed_shard",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Full128 recorded shard selection fields differ")
    key = row["parser_cache_key"] or row["source_sha256"]
    count = value["shard_count"]
    if (
        value["schema_version"] != SHARD_SELECTION_SCHEMA
        or value["assignment_kind"] != "PARSER_CACHE_KEY_OR_SOURCE_SHA256"
        or value["assignment_key"] != key
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or value["assigned_shard"] != int(key, 16) % count
        or value["executed_shard"] != value["assigned_shard"]
    ):
        raise ValueError("Full128 recorded shard selection differs")


def _validate_shard_arguments(
    shard_count: int, shard_index: int, maximum_jobs: int | None
) -> None:
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("Full128 shard index/count differ")
    if maximum_jobs is not None and (
        isinstance(maximum_jobs, bool)
        or not isinstance(maximum_jobs, int)
        or maximum_jobs <= 0
    ):
        raise ValueError("maximum_jobs must be positive or None")


def _validate_runtime_against_plan(
    runtime: BoundParserRuntime, bundle: Mapping[str, Any]
) -> None:
    if (
        runtime.parser_runtime_manifest_sha256
        != bundle["parser_runtime_manifest_sha256"]
        or runtime.parser_policy_sha256 != bundle["parser_policy_sha256"]
    ):
        raise ValueError("bound parser runtime or policy differs from route plan")


def _validate_runtime_binding(value: object, *, bundle: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != _RUNTIME_FIELDS:
        raise ValueError("Full128 parser runtime binding fields differ")
    if value["schema_version"] != PARSER_RUNTIME_BINDING_SCHEMA:
        raise ValueError("Full128 parser runtime binding schema differs")
    digest_fields = {
        "parser_runtime_manifest_sha256",
        "parser_runtime_bundle_raw_sha256",
        "parser_policy_sha256",
        "foreground_model_manifest_sha256",
        "foreground_model_bundle_raw_sha256",
        "instance_model_manifest_sha256",
        "instance_model_bundle_raw_sha256",
    }
    for field in digest_fields:
        _require_sha256(value[field], field)
    if not isinstance(value["device"], str) or not value["device"]:
        raise ValueError("Full128 parser runtime device differs")
    for field in (
        "job_batch_size",
        "instance_batch_size",
        "foreground_batch_size",
    ):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"Full128 parser runtime {field} differs")
    if value["publication_workers"] not in {1, 4}:
        raise ValueError("Full128 parser runtime publication workers differ")
    if (
        value["shape_policy"] != "EXACT_PREPROCESSED_SHAPE_BUCKETS"
        or value["oom_policy"] != "FAIL_CLOSED_NO_RETRY"
    ):
        raise ValueError("Full128 parser runtime batch policy differs")
    if (
        value["parser_runtime_manifest_sha256"]
        != bundle["parser_runtime_manifest_sha256"]
        or value["parser_policy_sha256"] != bundle["parser_policy_sha256"]
    ):
        raise ValueError("Full128 parser cache runtime differs from route plan")


def _unique_global_assignment(
    weights: Sequence[Sequence[float]], *, threshold: float
) -> tuple[int, ...] | None:
    if not weights:
        return ()
    width = len(weights[0])
    if width < len(weights) or any(len(row) != width for row in weights):
        return None
    best = _maximum_weight_assignment(weights, threshold=threshold, banned=None)
    if best is None:
        return None
    assignment, score = best
    for row_index, column_index in enumerate(assignment):
        alternative = _maximum_weight_assignment(
            weights,
            threshold=threshold,
            banned=(row_index, column_index),
        )
        if alternative is not None and math.isclose(
            alternative[1], score, rel_tol=0.0, abs_tol=1e-12
        ):
            return None
    return assignment


def _maximum_weight_assignment(
    weights: Sequence[Sequence[float]],
    *,
    threshold: float,
    banned: tuple[int, int] | None,
) -> tuple[tuple[int, ...], float] | None:
    """Rectangular Hungarian assignment with forbidden low-IoU edges."""

    row_count = len(weights)
    column_count = len(weights[0])
    infinity = 1_000_000.0
    costs = [
        [
            (
                infinity
                if weight < threshold or banned == (row_index, column_index)
                else 1.0 - weight
            )
            for column_index, weight in enumerate(row)
        ]
        for row_index, row in enumerate(weights)
    ]
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        p[0] = row_index
        minimum = [infinity] * (column_count + 1)
        used = [False] * (column_count + 1)
        column0 = 0
        while True:
            used[column0] = True
            active_row = p[column0]
            delta = infinity
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = costs[active_row - 1][column - 1] - u[active_row] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            if delta >= infinity / 2:
                return None
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment) or any(
        weights[row][column] < threshold for row, column in enumerate(assignment)
    ):
        return None
    score = sum(weights[row][column] for row, column in enumerate(assignment))
    return tuple(assignment), score


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _trimap_foreground_touches_border(payload: bytes) -> bool:
    with _image_from_bytes(payload, "Oxford trimap") as image:
        if len(image.getbands()) != 1:
            raise ValueError("Oxford trimap must be single-channel")
        image.load()
        values = image.get_flattened_data()
        width, height = image.size
        return any(
            values[index] == 1
            for index in (
                *range(width),
                *range((height - 1) * width, height * width),
                *range(0, height * width, width),
                *range(width - 1, height * width, width),
            )
        )


def _prepare_output_root(value: Path, *, create: bool) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError("Full128 output root must be an absolute path")
    repository = Path(__file__).resolve().parents[2]
    requested = Path(os.path.abspath(os.fspath(value)))
    if requested == repository or requested.is_relative_to(repository):
        raise ValueError("Full128 output root must remain outside the repository")
    if requested.exists() or requested.is_symlink():
        if requested.is_symlink() or not requested.is_dir():
            raise ValueError("Full128 output root must be a non-symlink directory")
        resolved = requested.resolve(strict=True)
        if resolved != requested:
            raise ValueError("Full128 output root must be canonical")
        return resolved
    if not create:
        raise FileNotFoundError(f"Full128 output root does not exist: {requested}")
    parent = requested.parent.resolve(strict=True)
    if parent == repository or parent.is_relative_to(repository):
        raise ValueError("Full128 output root must remain outside the repository")
    try:
        requested.mkdir(mode=0o700)
    except FileExistsError:
        if requested.is_symlink() or not requested.is_dir():
            raise ValueError(
                "Full128 output root must be a non-symlink directory"
            ) from None
        resolved = requested.resolve(strict=True)
        if resolved != requested:
            raise ValueError("Full128 output root must be canonical") from None
        return resolved
    fsync_directory(parent)
    return requested


def _private_child_directory(root: Path, name: str, *, create: bool) -> Path:
    child = root / name
    if child.exists() or child.is_symlink():
        if (
            child.is_symlink()
            or not child.is_dir()
            or child.resolve(strict=True) != child
        ):
            raise ValueError(
                f"Full128 {name} must be a canonical non-symlink directory"
            )
        return child
    if not create:
        raise FileNotFoundError(f"Full128 {name} directory is missing")
    try:
        child.mkdir(mode=0o700)
    except FileExistsError:
        if (
            child.is_symlink()
            or not child.is_dir()
            or child.resolve(strict=True) != child
        ):
            raise ValueError(
                f"Full128 {name} must be a canonical non-symlink directory"
            ) from None
        return child
    fsync_directory(root)
    return child


def _canonical_root(root: Path, dataset_name: str) -> Path:
    if not root.is_absolute():
        raise ValueError(f"{dataset_name} dataset root must be absolute")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{dataset_name} dataset root must resolve to a directory")
    return resolved


def _read_bound_artifact(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
    maximum_bytes: int,
) -> _ArtifactInput:
    if set(binding) != {"relative_path", "sha256", "byte_size"}:
        raise ValueError(f"{label} binding fields differ")
    relative = _safe_relative(binding["relative_path"], label)
    _require_sha256(binding["sha256"], label)
    size = binding["byte_size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= maximum_bytes
    ):
        raise ValueError(f"{label} byte size differs")
    payload = _read_secure_relative(
        root, relative, maximum_bytes=maximum_bytes, label=label
    )
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise ValueError(f"{label} differs from route-plan binding")
    return _ArtifactInput(
        root.joinpath(*PurePosixPath(relative).parts), payload, binding["sha256"], size
    )


def _read_secure_relative(
    root: Path, relative: str, *, maximum_bytes: int, label: str
) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Full128 materialization requires O_NOFOLLOW")
    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptors: list[int] = []
    try:
        root_fd = os.open(root, directory_flags)
        descriptors.append(root_fd)
        root_before = os.fstat(root_fd)
        current = root_fd
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(child)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"{label} path component is not a directory")
            _require_named_identity(current, part, child_stat, label)
            current = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise ValueError(f"{label} must be a bounded non-empty regular file")
        _require_named_identity(current, parts[-1], before, label)
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(
            descriptor, min(1_048_576, maximum_bytes + 1 - observed)
        ):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{label} exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require_named_identity(current, parts[-1], after, label)
        root_named = os.stat(root, follow_symlinks=False)
        if (
            _file_identity(before) != _file_identity(after)
            or _node_identity(root_before) != _node_identity(root_named)
            or observed != before.st_size
        ):
            raise RuntimeError(f"{label} changed while being read")
    except OSError as exc:
        raise ValueError(
            f"{label} path must not traverse symlinks: {relative}"
        ) from exc
    finally:
        for descriptor_to_close in reversed(descriptors):
            os.close(descriptor_to_close)
    return b"".join(chunks)


def _read_regular_absolute(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{label} must be a canonical regular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    observed = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise ValueError(f"{label} size or file type differs")
        while chunk := os.read(
            descriptor, min(1_048_576, maximum_bytes + 1 - observed)
        ):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{label} exceeds byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(before) != _file_identity(after) or observed != before.st_size:
        raise RuntimeError(f"{label} changed while being read")
    return b"".join(chunks)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Full128 artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_exact_directory(path: Path, names: set[str], label: str) -> None:
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"Full128 {label} must be a canonical non-symlink directory")
    observed: set[str] = set()
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"Full128 {label} contains a non-regular artifact")
        observed.add(child.name)
    if observed != names:
        raise ValueError(f"Full128 {label} is partial or contains unexpected artifacts")


def _image_from_bytes(payload: bytes, label: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if (
                opened.width <= 0
                or opened.height <= 0
                or opened.width * opened.height > _MAX_SOURCE_PIXELS
            ):
                raise ValueError(f"{label} dimensions exceed policy")
            opened.load()
            return opened.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{label} is not a bounded supported image") from exc


def _image_size(payload: bytes, label: str) -> tuple[int, int]:
    with _image_from_bytes(payload, label) as image:
        return image.size


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_token(dataset: str, kind: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Full128 {dataset} {kind} group is missing")
    return f"{dataset}\0{kind}\0{value}"


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be non-empty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe {label} path: {value!r}")
    return value


def _finite_box(value: object, *, integer: bool) -> tuple[Any, Any, Any, Any]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Full128 bbox must contain four values")
    if integer:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("Full128 aligned bbox values must be integers")
        box = tuple(value)
    else:
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in value
        ):
            raise ValueError("Full128 bbox values must be finite numbers")
        box = tuple(float(item) for item in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Full128 bbox must be non-empty")
    return box


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _require_named_identity(
    parent_fd: int, name: str, expected: os.stat_result, label: str
) -> None:
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _node_identity(observed) != _node_identity(expected):
        raise RuntimeError(f"{label} named path changed while being read")


def _node_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


__all__ = [
    "ASSEMBLY_SCHEMA",
    "DERIVED_LINEAGE_SCHEMA",
    "EXECUTION_RECEIPT_SCHEMA",
    "INVENTORY_REQUEST_SCHEMA",
    "PARSER_CACHE_RECEIPT_SCHEMA",
    "BoundParserRuntime",
    "assemble_full128_materialization",
    "build_bound_parser_runtime",
    "materialize_full128_route_plan",
    "migrate_full128_compact_sample_caches",
    "read_route_plan_bundle",
    "validate_derived_lineage_receipt",
]
