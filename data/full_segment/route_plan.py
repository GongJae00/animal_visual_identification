"""Deterministic, content-bound route planning for Full128 source samples."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from data.adapters import ADAPTERS
from data.source_lock import SOURCE_REGISTRY, get_record
from data.types import DatasetAdmission, UnifiedCanidSample
from shared.foundation.provenance import content_sha256

ROUTE_PLAN_SCHEMA = "archive.full128.route_plan.v3"
ROUTE_PLAN_RECORD_SCHEMA = "archive.full128.route_plan_record.v3"
ROUTE_PLAN_BUNDLE_SCHEMA = "archive.full128.route_plan_bundle.v3"
ROUTE_POLICY_SCHEMA = "archive.full128.route_policy.v3"
_LEGACY_ROUTE_PLAN_SCHEMA = "archive.full128.route_plan.v2"
_LEGACY_ROUTE_PLAN_RECORD_SCHEMA = "archive.full128.route_plan_record.v2"
_LEGACY_ROUTE_PLAN_BUNDLE_SCHEMA = "archive.full128.route_plan_bundle.v2"
_LEGACY_ROUTE_POLICY_SCHEMA = "archive.full128.route_policy.v2"
ADMISSION_STATE_SCHEMA = "archive.full128.route_source_registry_admission_state.v1"
PARSER_CACHE_KEY_SCHEMA = "archive.full128.parser_cache_key.v1"

CANONICAL_DATASETS = (
    "ap10k-dog",
    "dogflw",
    "dogfacenet224",
    "mpdd",
    "oxford-pets-dog",
    "sibetan",
    "yt-bb-dog",
)

_ADMITTED_STATES = {
    DatasetAdmission.ADMIT_TRAIN,
    DatasetAdmission.ADMIT_VALIDATION_ONLY,
    DatasetAdmission.ADMIT_TEACHER_ONLY,
}
_TARGET_SIZE = 224
_CONTEXT_FRACTION = 0.05
_BACKGROUND_RGB = [127, 127, 127]
_MAX_SOURCE_BYTES = 67_108_864
_MAX_SOURCE_PIXELS = 33_554_432
_MAX_LABEL_BYTES = 268_435_456
_MAX_JSON_NODES = 5_000_000
_MAX_XML_BYTES = 1_048_576


class RouteIntent(str, Enum):
    BODY_PARSING = "BODY_PARSING"
    BODY_MASK = "BODY_MASK"
    NATIVE_FACE = "NATIVE_FACE"
    DERIVED_NATIVE_FACE = "DERIVED_NATIVE_FACE"
    DERIVED_NATIVE_HEAD = "DERIVED_NATIVE_HEAD"


@dataclass(frozen=True, slots=True)
class _ArtifactBinding:
    root: Path
    relative_path: str
    sha256: str
    byte_size: int
    file_identity: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
        }


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    artifact: _ArtifactBinding
    width: int
    height: int


@dataclass(slots=True)
class _BuildContext:
    bindings: dict[tuple[Path, str], _ArtifactBinding]
    sources: dict[tuple[Path, str], _SourceBinding]
    ap10k_documents: dict[
        tuple[Path, str],
        tuple[
            _ArtifactBinding,
            dict[int, dict[str, Any]],
            dict[int, dict[str, Any]],
        ],
    ]


def build_parser_cache_key(
    source_sha256: str,
    *,
    parser_runtime_manifest_sha256: str,
    parser_policy_sha256: str,
) -> str:
    """Bind a parser cache entry to source bytes and the complete parser runtime."""

    _require_sha256(source_sha256, "source_sha256")
    _require_sha256(parser_runtime_manifest_sha256, "parser_runtime_manifest_sha256")
    _require_sha256(parser_policy_sha256, "parser_policy_sha256")
    return content_sha256(
        {
            "schema_version": PARSER_CACHE_KEY_SCHEMA,
            "source_sha256": source_sha256,
            "parser_runtime_manifest_sha256": parser_runtime_manifest_sha256,
            "parser_policy_sha256": parser_policy_sha256,
        }
    )


def build_full128_route_plan(
    *,
    parser_runtime_manifest_sha256: str,
    parser_policy_sha256: str,
    maximum_samples_per_dataset: int | None = None,
    dataset_names: Sequence[str] = CANONICAL_DATASETS,
    samples_by_dataset: Mapping[str, Sequence[UnifiedCanidSample]] | None = None,
    dogface_classes_train_path: Path | None = None,
    dogface_classes_test_path: Path | None = None,
) -> dict[str, Any]:
    """Build a metadata-only route plan from admitted adapter samples.

    ``maximum_samples_per_dataset`` selects the sample-token-sorted prefix of each
    complete adapter result. It changes only recorded selection, never row policy.
    """

    _require_sha256(parser_runtime_manifest_sha256, "parser_runtime_manifest_sha256")
    _require_sha256(parser_policy_sha256, "parser_policy_sha256")
    maximum = _maximum_samples(maximum_samples_per_dataset)
    names = _validated_dataset_names(dataset_names)
    records = {name: get_record(name) for name in names}

    if (dogface_classes_train_path is None) != (dogface_classes_test_path is None):
        raise ValueError("DogFace train and test class paths must be provided together")
    if samples_by_dataset is None:
        loaded = {}
        for name in names:
            if name == "dogfacenet224":
                loaded[name] = ADAPTERS[name](
                    Path(records[name].data_root),
                    classes_train_path=dogface_classes_train_path,
                    classes_test_path=dogface_classes_test_path,
                )
            else:
                loaded[name] = ADAPTERS[name](Path(records[name].data_root))
    else:
        if dogface_classes_train_path is not None:
            raise ValueError(
                "DogFace class paths cannot accompany preloaded Full128 samples"
            )
        if set(samples_by_dataset) != set(names):
            raise ValueError("Full128 samples must cover the canonical dataset set")
        loaded = {name: tuple(samples_by_dataset[name]) for name in names}

    context = _BuildContext(bindings={}, sources={}, ap10k_documents={})
    selected_by_dataset: dict[str, tuple[UnifiedCanidSample, ...]] = {}
    selection_rows: list[dict[str, Any]] = []
    all_tokens: set[str] = set()
    for name in names:
        samples = tuple(loaded[name])
        if not samples:
            raise ValueError(f"Full128 adapter returned no samples: {name}")
        ordered = tuple(sorted(samples, key=lambda item: item.sample_id))
        tokens = [sample.sample_id for sample in ordered]
        if len(tokens) != len(set(tokens)) or any(
            token in all_tokens for token in tokens
        ):
            raise ValueError("Full128 sample tokens must be globally unique")
        all_tokens.update(tokens)
        selected = ordered if maximum is None else ordered[:maximum]
        selected_by_dataset[name] = selected
        selection_rows.append(
            {
                "dataset_name": name,
                "available_sample_count": len(ordered),
                "selected_sample_count": len(selected),
                "available_sample_tokens_sha256": content_sha256(tokens),
                "selected_sample_tokens_sha256": content_sha256(
                    [sample.sample_id for sample in selected]
                ),
            }
        )

    plan_records: list[dict[str, Any]] = []
    for name in names:
        dataset = records[name]
        root = _canonical_dataset_root(Path(dataset.data_root), name)
        for sample in selected_by_dataset[name]:
            plan_records.append(
                _build_record(
                    sample,
                    dataset=dataset,
                    root=root,
                    parser_runtime_manifest_sha256=parser_runtime_manifest_sha256,
                    parser_policy_sha256=parser_policy_sha256,
                    context=context,
                )
            )

    plan_records.sort(key=lambda item: item["sample_token"])
    for binding in context.bindings.values():
        _verify_retained_binding_identity(binding)
    selection = {
        "mode": (
            "COMPLETE_DATASETS"
            if maximum is None
            else "DETERMINISTIC_MAXIMUM_PER_DATASET"
        ),
        "ordering": "SAMPLE_TOKEN_ASC",
        "maximum_samples_per_dataset": maximum,
        "datasets": selection_rows,
    }
    plan = {
        "schema_version": ROUTE_PLAN_SCHEMA,
        "selection": selection,
        "records": plan_records,
    }
    route_policy = _route_policy()
    admission_state = _source_registry_admission_state()
    payload = {
        "schema_version": ROUTE_PLAN_BUNDLE_SCHEMA,
        "content_kind": "METADATA_ONLY",
        "executes_image_crop_or_animal_parsing": False,
        "parser_runtime_manifest_sha256": parser_runtime_manifest_sha256,
        "parser_policy_sha256": parser_policy_sha256,
        "source_registry_admission_state": admission_state,
        "source_registry_admission_sha256": content_sha256(admission_state),
        "route_policy": route_policy,
        "route_policy_sha256": content_sha256(route_policy),
        "plan": plan,
        "plan_sha256": content_sha256(plan),
    }
    return {**payload, "bundle_sha256": content_sha256(payload)}


def validate_full128_route_plan_bundle(
    value: object, *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate bundle hashes, current policy/admission, and bound file identities."""

    expected = {
        "schema_version",
        "content_kind",
        "executes_image_crop_or_animal_parsing",
        "parser_runtime_manifest_sha256",
        "parser_policy_sha256",
        "source_registry_admission_state",
        "source_registry_admission_sha256",
        "route_policy",
        "route_policy_sha256",
        "plan",
        "plan_sha256",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Full128 route-plan bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] not in {
        ROUTE_PLAN_BUNDLE_SCHEMA,
        _LEGACY_ROUTE_PLAN_BUNDLE_SCHEMA,
    }:
        raise ValueError("Full128 route-plan bundle schema differs")
    payload = {key: item for key, item in bundle.items() if key != "bundle_sha256"}
    if bundle["bundle_sha256"] != content_sha256(payload):
        raise ValueError("Full128 route-plan bundle digest differs")
    if (
        bundle["content_kind"] != "METADATA_ONLY"
        or bundle["executes_image_crop_or_animal_parsing"] is not False
    ):
        raise ValueError("Full128 route plan must remain metadata-only")
    for field in ("parser_runtime_manifest_sha256", "parser_policy_sha256"):
        _require_sha256(bundle[field], field)

    legacy = bundle["schema_version"] == _LEGACY_ROUTE_PLAN_BUNDLE_SCHEMA
    policy = _legacy_route_policy() if legacy else _route_policy()
    if bundle["route_policy"] != policy or bundle[
        "route_policy_sha256"
    ] != content_sha256(policy):
        raise ValueError("Full128 route policy differs from the current policy")
    admission_state = _source_registry_admission_state()
    if bundle["source_registry_admission_state"] != admission_state or bundle[
        "source_registry_admission_sha256"
    ] != content_sha256(admission_state):
        raise ValueError("Full128 source-registry admission state differs")

    plan = bundle["plan"]
    if (
        not isinstance(plan, dict)
        or set(plan) != {"schema_version", "selection", "records"}
        or plan["schema_version"]
        != (_LEGACY_ROUTE_PLAN_SCHEMA if legacy else ROUTE_PLAN_SCHEMA)
        or not isinstance(plan["records"], list)
        or bundle["plan_sha256"] != content_sha256(plan)
    ):
        raise ValueError("Full128 route plan schema or digest differs")
    records = plan["records"]
    tokens = [
        record.get("sample_token") for record in records if isinstance(record, dict)
    ]
    if (
        len(tokens) != len(records)
        or tokens != sorted(tokens)
        or len(tokens) != len(set(tokens))
    ):
        raise ValueError("Full128 route-plan records are not uniquely token-sorted")

    for row in records:
        _validate_record(row, bundle=bundle, verify_files=verify_files)
    _validate_selection(plan["selection"], records)
    return bundle


def _build_record(
    sample: UnifiedCanidSample,
    *,
    dataset: Any,
    root: Path,
    parser_runtime_manifest_sha256: str,
    parser_policy_sha256: str,
    context: _BuildContext,
) -> dict[str, Any]:
    _validate_sample(sample, dataset)
    source = _source_binding(sample, root, context)
    route, evidence = _route_evidence(sample, root=root, context=context)
    parser_cache_key = None
    if route is RouteIntent.BODY_PARSING:
        parser_cache_key = build_parser_cache_key(
            sample.image_sha256,
            parser_runtime_manifest_sha256=parser_runtime_manifest_sha256,
            parser_policy_sha256=parser_policy_sha256,
        )

    payload = {
        "schema_version": ROUTE_PLAN_RECORD_SCHEMA,
        "sample_token": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "dataset_version": sample.dataset_version,
        "source_path": source.artifact.relative_path,
        "source_sha256": source.artifact.sha256,
        "source_byte_size": source.artifact.byte_size,
        "source_width": source.width,
        "source_height": source.height,
        "identity_metadata": {
            "registered_identity_id": sample.registered_identity_id,
            "generated_identity_id": sample.generated_identity_id,
            "raw_identity_id": sample.raw_identity_id,
        },
        "capture_metadata": {
            "source_group_id": sample.source_group_id,
            "capture_group_id": sample.capture_group_id,
            "capture_group_kind": sample.capture_group_kind.value,
            "camera_id": sample.camera_id,
            "timestamp_ms": sample.timestamp_ms,
        },
        "source_metadata": {
            "species": sample.species,
            "breed": sample.breed,
            "label_availability": _json_copy(sample.label_availability),
            "adapter_metadata": _json_copy(sample.metadata),
        },
        "split": sample.split_role,
        "admission": dataset.admission.value,
        "route_intent": route.value,
        "target_size": _TARGET_SIZE,
        "context_fraction": _CONTEXT_FRACTION,
        "background_rgb": list(_BACKGROUND_RGB),
        "duplicate_component": sample.image_sha256,
        "parser_cache_key": parser_cache_key,
        "route_evidence": evidence,
    }
    return {**payload, "record_sha256": content_sha256(payload)}


def _route_evidence(
    sample: UnifiedCanidSample, *, root: Path, context: _BuildContext
) -> tuple[RouteIntent, dict[str, Any]]:
    if sample.dataset_name == "dogfacenet224":
        return RouteIntent.BODY_PARSING, {
            "kind": "PUBLISHER_FACE_CROP_SOURCE",
            "source_sha256": sample.image_sha256,
            "association_intent": "REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
        }
    if sample.dataset_name == "dogflw":
        return _dogflw_evidence(sample, root=root, context=context)
    if sample.dataset_name == "oxford-pets-dog":
        return _oxford_evidence(sample, root=root, context=context)
    if sample.dataset_name == "ap10k-dog":
        return RouteIntent.BODY_PARSING, _ap10k_evidence(
            sample, root=root, context=context
        )
    return RouteIntent.BODY_PARSING, {
        "kind": "PARSER_REQUIRED",
        "association_intent": "REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
    }


def _dogflw_evidence(
    sample: UnifiedCanidSample, *, root: Path, context: _BuildContext
) -> tuple[RouteIntent, dict[str, Any]]:
    source_path = PurePosixPath(sample.image_path)
    if (
        len(source_path.parts) != 4
        or source_path.parts[:3] != ("DogFLW", sample.split_role, "images")
        or source_path.suffix.lower() != ".png"
    ):
        raise ValueError("DogFLW source path differs from adapter schema")
    label_relative = PurePosixPath(
        "DogFLW", sample.split_role, "labels", f"{source_path.stem}.json"
    ).as_posix()
    label, document, normalized_constants = _read_json_artifact(
        root,
        label_relative,
        context=context,
        subject="DogFLW label artifact",
        normalize_nonstandard_constants=True,
    )
    raw_box = document.get("bounding_boxes")
    artifact_box = _finite_box(raw_box)
    if artifact_box != sample.face_box_xyxy:
        raise ValueError("DogFLW label bbox differs from UnifiedCanidSample")
    bbox_state = (
        "VALID_WITHIN_SOURCE"
        if artifact_box is not None
        and _box_within(artifact_box, sample.width, sample.height)
        else "INVALID_OR_OUT_OF_BOUNDS"
    )
    evidence = {
        "kind": "DOGFLW_FACE_BBOX",
        "label_artifact": label.to_dict(),
        "bbox_xyxy": None if artifact_box is None else list(artifact_box),
        "bbox_state": bbox_state,
        "normalized_nonstandard_constants": normalized_constants,
        "annotation_usage": "AUDIT_AND_FUTURE_PARSER_DEVELOPMENT_ONLY",
        "association_intent": "SELECT_LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS",
    }
    return RouteIntent.BODY_PARSING, evidence


def _oxford_evidence(
    sample: UnifiedCanidSample, *, root: Path, context: _BuildContext
) -> tuple[RouteIntent, dict[str, Any]]:
    source_path = PurePosixPath(sample.image_path)
    if (
        len(source_path.parts) != 2
        or source_path.parts[0] != "images"
        or source_path.suffix.lower() != ".jpg"
        or sample.foreground_mask_path != f"annotations/trimaps/{source_path.stem}.png"
    ):
        raise ValueError("Oxford source or trimap path differs from adapter schema")
    trimap, payload = _read_artifact(
        root,
        sample.foreground_mask_path,
        maximum_bytes=_MAX_SOURCE_BYTES,
        subject="Oxford trimap artifact",
        context=context,
    )
    trimap_state, labels, trimap_width, trimap_height = _trimap_state(
        payload, sample=sample
    )
    trimap_evidence = {
        "kind": "OXFORD_PUBLISHER_ANNOTATIONS",
        "trimap_artifact": trimap.to_dict(),
        "trimap_width": trimap_width,
        "trimap_height": trimap_height,
        "observed_labels": labels,
        "trimap_state": trimap_state,
        "label_policy": {"foreground": 1, "excluded": [2, 3]},
        "annotation_usage": "AUDIT_AND_FUTURE_PARSER_DEVELOPMENT_ONLY",
        "head_annotation_state": (
            "AVAILABLE" if sample.head_roi_xyxy is not None else "ABSENT"
        ),
        "association_intent": "SELECT_LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS",
    }
    if sample.head_roi_xyxy is not None:
        xml_relative = f"annotations/xmls/{source_path.stem}.xml"
        head_artifact, xml_payload = _read_artifact(
            root,
            xml_relative,
            maximum_bytes=_MAX_XML_BYTES,
            subject="Oxford head XML artifact",
            context=context,
        )
        xml_box = _oxford_head_box(
            xml_payload,
            image_name=source_path.name,
            width=sample.width,
            height=sample.height,
        )
        if xml_box != sample.head_roi_xyxy:
            raise ValueError("Oxford XML head bbox differs from UnifiedCanidSample")
        trimap_evidence.update(
            {
                "head_artifact": head_artifact.to_dict(),
                "head_bbox_xyxy": list(xml_box),
            }
        )
    return RouteIntent.BODY_PARSING, trimap_evidence


def _ap10k_evidence(
    sample: UnifiedCanidSample, *, root: Path, context: _BuildContext
) -> dict[str, Any]:
    if sample.split_role not in {"train", "val", "test"}:
        raise ValueError("AP-10K split differs from adapter schema")
    relative = f"ap-10k/annotations/ap10k-{sample.split_role}-split1.json"
    cache_key = (root, relative)
    if cache_key not in context.ap10k_documents:
        artifact, document, _ = _read_json_artifact(
            root,
            relative,
            context=context,
            subject="AP-10K annotation artifact",
        )
        images, annotations = _ap10k_indexes(document)
        context.ap10k_documents[cache_key] = (artifact, images, annotations)
    artifact, images, annotations = context.ap10k_documents[cache_key]

    annotation_id = sample.metadata.get("annotation_id")
    image_id = sample.metadata.get("image_id")
    if (
        isinstance(annotation_id, bool)
        or not isinstance(annotation_id, int)
        or isinstance(image_id, bool)
        or not isinstance(image_id, int)
    ):
        raise TypeError("AP-10K sample lacks integer annotation authority metadata")
    try:
        annotation = annotations[annotation_id]
        image = images[image_id]
    except KeyError as exc:
        raise ValueError(
            "AP-10K sample authority is absent from annotation artifact"
        ) from exc
    if (
        annotation.get("image_id") != image_id
        or annotation.get("category_id") != 8
        or not isinstance(image.get("file_name"), str)
        or f"ap-10k/data/{image['file_name']}" != sample.image_path
    ):
        raise ValueError("AP-10K annotation-to-source association differs")
    raw_bbox = annotation.get("bbox")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise ValueError("AP-10K authoritative bbox schema differs")
    try:
        x, y, width, height = (float(value) for value in raw_bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("AP-10K authoritative bbox values differ") from exc
    bbox = (x, y, x + width, y + height)
    if (
        not all(math.isfinite(value) for value in bbox)
        or bbox != sample.dog_boxes_xyxy
        or not _box_within(bbox, sample.width, sample.height)
    ):
        raise ValueError("AP-10K authoritative bbox differs or is out of bounds")
    authority = {
        "annotation_artifact_sha256": artifact.sha256,
        "annotation_id": annotation_id,
        "image_id": image_id,
        "bbox_xyxy": list(bbox),
    }
    return {
        "kind": "AP10K_AUTHORITATIVE_BBOX_ASSOCIATION",
        "annotation_artifact": artifact.to_dict(),
        "annotation_id": annotation_id,
        "image_id": image_id,
        "bbox_xyxy": list(bbox),
        "association_intent": {
            "kind": "AUTHORITATIVE_BBOX_MATCH",
            "authority_sha256": content_sha256(authority),
        },
    }


def _source_binding(
    sample: UnifiedCanidSample, root: Path, context: _BuildContext
) -> _SourceBinding:
    relative = _safe_relative_path(sample.image_path, "Full128 source image")
    key = (root, relative)
    cached = context.sources.get(key)
    if cached is not None:
        if cached.artifact.sha256 != sample.image_sha256 or (
            cached.width,
            cached.height,
        ) != (sample.width, sample.height):
            raise ValueError("repeated source path has inconsistent adapter metadata")
        return cached
    artifact, payload = _read_artifact(
        root,
        relative,
        maximum_bytes=_MAX_SOURCE_BYTES,
        expected_sha256=sample.image_sha256,
        subject="Full128 source image",
        context=context,
    )
    width, height = _verified_image_size(payload, subject="Full128 source image")
    if (width, height) != (sample.width, sample.height):
        raise ValueError("Full128 source dimensions differ from UnifiedCanidSample")
    binding = _SourceBinding(artifact=artifact, width=width, height=height)
    context.sources[key] = binding
    return binding


def _read_json_artifact(
    root: Path,
    relative: str,
    *,
    context: _BuildContext,
    subject: str,
    normalize_nonstandard_constants: bool = False,
) -> tuple[_ArtifactBinding, dict[str, Any], dict[str, int]]:
    artifact, payload = _read_artifact(
        root,
        relative,
        maximum_bytes=_MAX_LABEL_BYTES,
        subject=subject,
        context=context,
    )
    normalized: Counter[str] = Counter()

    def parse_constant(token: str) -> None:
        if not normalize_nonstandard_constants:
            raise ValueError(f"non-standard JSON numeric constant: {token}")
        normalized[token] += 1

    try:
        text = payload.decode("utf-8", errors="strict")
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=parse_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ValueError(f"{subject} must be bounded UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise TypeError(f"{subject} root must be an object")
    _bounded_json_nodes(document)
    return artifact, document, dict(sorted(normalized.items()))


def _read_artifact(
    root: Path,
    relative: str | None,
    *,
    maximum_bytes: int,
    subject: str,
    context: _BuildContext,
    expected_sha256: str | None = None,
) -> tuple[_ArtifactBinding, bytes]:
    if relative is None:
        raise ValueError(f"{subject} path is missing")
    relative = _safe_relative_path(relative, subject)
    payload, binding = _read_regular_file(
        root,
        relative,
        maximum_bytes=maximum_bytes,
        subject=subject,
    )
    if expected_sha256 is not None and binding.sha256 != expected_sha256:
        raise ValueError(f"{subject} SHA-256 differs from UnifiedCanidSample")
    key = (root, relative)
    previous = context.bindings.setdefault(key, binding)
    if previous != binding:
        raise RuntimeError(f"{subject} changed between retained reads")
    return binding, payload


def _read_regular_file(
    root: Path, relative: str, *, maximum_bytes: int, subject: str
) -> tuple[bytes, _ArtifactBinding]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Full128 route planning requires O_NOFOLLOW")
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
    bindings: list[tuple[int, str, int]] = []
    try:
        root_fd = os.open(root, directory_flags)
        descriptors.append(root_fd)
        root_initial = os.fstat(root_fd)
        current = root_fd
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(child)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"{subject} path component must be a directory")
            _require_named_identity(current, part, child_stat, subject)
            bindings.append((current, part, child))
            current = child
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_size <= 0
            or initial.st_size > maximum_bytes
        ):
            raise ValueError(f"{subject} must be a bounded non-empty regular file")
        _require_named_identity(current, parts[-1], initial, subject)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(
            descriptor, min(1_048_576, maximum_bytes + 1 - observed)
        ):
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError(f"{subject} exceeds byte limit")
            digest.update(chunk)
            chunks.append(chunk)
        final = os.fstat(descriptor)
        _require_named_identity(current, parts[-1], final, subject)
        for parent_fd, name, child_fd in bindings:
            _require_named_identity(parent_fd, name, os.fstat(child_fd), subject)
        root_named = os.stat(root, follow_symlinks=False)
        if _node_identity(root_initial) != _node_identity(root_named):
            raise RuntimeError(f"{subject} dataset root changed during retained read")
        if (
            _file_identity(initial) != _file_identity(final)
            or observed != initial.st_size
        ):
            raise RuntimeError(f"{subject} changed during retained read")
    except FileNotFoundError:
        raise FileNotFoundError(f"{subject} does not exist: {relative}") from None
    except OSError as exc:
        raise ValueError(
            f"{subject} path must not traverse symlinks: {relative}"
        ) from exc
    finally:
        for descriptor_to_close in reversed(descriptors):
            os.close(descriptor_to_close)
    binding = _ArtifactBinding(
        root=root,
        relative_path=relative,
        sha256=digest.hexdigest(),
        byte_size=observed,
        file_identity=_file_identity(initial),
    )
    return b"".join(chunks), binding


def _verify_retained_binding_identity(binding: _ArtifactBinding) -> None:
    directory_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptors: list[int] = []
    try:
        root_fd = os.open(binding.root, directory_flags)
        descriptors.append(root_fd)
        root_initial = os.fstat(root_fd)
        current = root_fd
        for part in PurePosixPath(binding.relative_path).parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(child)
            child_stat = os.fstat(child)
            _require_named_identity(
                current, part, child_stat, "Full128 retained artifact"
            )
            current = child
        name = PurePosixPath(binding.relative_path).parts[-1]
        descriptor = os.open(name, file_flags, dir_fd=current)
        descriptors.append(descriptor)
        observed = os.fstat(descriptor)
        _require_named_identity(current, name, observed, "Full128 retained artifact")
        root_named = os.stat(binding.root, follow_symlinks=False)
        if (
            _node_identity(root_initial) != _node_identity(root_named)
            or _file_identity(observed) != binding.file_identity
            or observed.st_size != binding.byte_size
        ):
            raise RuntimeError(
                "Full128 retained artifact changed before plan completion"
            )
    except OSError as exc:
        raise RuntimeError(
            "Full128 retained artifact changed before plan completion"
        ) from exc
    finally:
        for descriptor_to_close in reversed(descriptors):
            os.close(descriptor_to_close)


def _validate_record(
    value: object, *, bundle: Mapping[str, Any], verify_files: bool
) -> None:
    fields = {
        "schema_version",
        "sample_token",
        "dataset_name",
        "dataset_version",
        "source_path",
        "source_sha256",
        "source_byte_size",
        "source_width",
        "source_height",
        "identity_metadata",
        "capture_metadata",
        "source_metadata",
        "split",
        "admission",
        "route_intent",
        "target_size",
        "context_fraction",
        "background_rgb",
        "duplicate_component",
        "parser_cache_key",
        "route_evidence",
        "record_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Full128 route-plan record fields differ")
    row = value
    expected_record_schema = (
        _LEGACY_ROUTE_PLAN_RECORD_SCHEMA
        if bundle["schema_version"] == _LEGACY_ROUTE_PLAN_BUNDLE_SCHEMA
        else ROUTE_PLAN_RECORD_SCHEMA
    )
    if row["schema_version"] != expected_record_schema:
        raise ValueError("Full128 route-plan record schema differs")
    payload = {key: item for key, item in row.items() if key != "record_sha256"}
    _require_sha256(row["record_sha256"], "record_sha256")
    if row["record_sha256"] != content_sha256(payload):
        raise ValueError("Full128 route-plan record digest differs")
    _require_sha256(row["sample_token"], "sample_token")
    _require_sha256(row["source_sha256"], "source_sha256")
    if (
        isinstance(row["source_byte_size"], bool)
        or not isinstance(row["source_byte_size"], int)
        or not 0 < row["source_byte_size"] <= _MAX_SOURCE_BYTES
        or isinstance(row["source_width"], bool)
        or not isinstance(row["source_width"], int)
        or isinstance(row["source_height"], bool)
        or not isinstance(row["source_height"], int)
        or row["source_width"] <= 0
        or row["source_height"] <= 0
        or row["source_width"] * row["source_height"] > _MAX_SOURCE_PIXELS
    ):
        raise ValueError("Full128 route-plan source bounds differ")
    _safe_relative_path(row["source_path"], "Full128 source image")
    dataset = get_record(row["dataset_name"])
    if (
        dataset.canonical_name not in CANONICAL_DATASETS
        or dataset.version != row["dataset_version"]
        or dataset.admission.value != row["admission"]
    ):
        raise ValueError("Full128 route-plan dataset admission differs")
    route = RouteIntent(row["route_intent"])
    if (
        row["target_size"] != _TARGET_SIZE
        or row["context_fraction"] != _CONTEXT_FRACTION
        or row["background_rgb"] != _BACKGROUND_RGB
        or row["duplicate_component"] != row["source_sha256"]
    ):
        raise ValueError("Full128 route-plan source or target policy differs")
    expected_cache = None
    if route is RouteIntent.BODY_PARSING:
        expected_cache = build_parser_cache_key(
            row["source_sha256"],
            parser_runtime_manifest_sha256=bundle["parser_runtime_manifest_sha256"],
            parser_policy_sha256=bundle["parser_policy_sha256"],
        )
    if row["parser_cache_key"] != expected_cache:
        raise ValueError("Full128 parser cache key differs")
    _validate_route_evidence(row, route)
    if not verify_files:
        return
    root = _canonical_dataset_root(Path(dataset.data_root), dataset.canonical_name)
    source_payload, source = _read_regular_file(
        root,
        _safe_relative_path(row["source_path"], "Full128 source image"),
        maximum_bytes=row["source_byte_size"],
        subject="Full128 source image",
    )
    if (
        source.sha256 != row["source_sha256"]
        or source.byte_size != row["source_byte_size"]
    ):
        raise ValueError("Full128 bound source artifact differs")
    if _verified_image_size(source_payload, subject="Full128 source image") != (
        row["source_width"],
        row["source_height"],
    ):
        raise ValueError("Full128 bound source dimensions differ")
    evidence = row["route_evidence"]
    for field in (
        "annotation_artifact",
        "label_artifact",
        "trimap_artifact",
        "head_artifact",
    ):
        artifact = evidence.get(field)
        if artifact is not None:
            _verify_serialized_artifact(root, artifact, field)


def _validate_selection(value: object, records: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "ordering",
        "maximum_samples_per_dataset",
        "datasets",
    }:
        raise ValueError("Full128 route-plan selection fields differ")
    mode = value["mode"]
    maximum = value["maximum_samples_per_dataset"]
    if mode == "COMPLETE_DATASETS":
        if maximum is not None:
            raise ValueError("complete Full128 selection cannot have a maximum")
    elif mode == "DETERMINISTIC_MAXIMUM_PER_DATASET":
        _maximum_samples(maximum)
    else:
        raise ValueError("Full128 route-plan selection mode differs")
    if value["ordering"] != "SAMPLE_TOKEN_ASC":
        raise ValueError("Full128 route-plan selection ordering differs")
    datasets = value["datasets"]
    if not isinstance(datasets, list) or len(datasets) != len(CANONICAL_DATASETS):
        raise ValueError("Full128 route-plan dataset selection differs")
    tokens_by_dataset = {
        name: [row["sample_token"] for row in records if row["dataset_name"] == name]
        for name in CANONICAL_DATASETS
    }
    for expected_name, item in zip(CANONICAL_DATASETS, datasets, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "dataset_name",
            "available_sample_count",
            "selected_sample_count",
            "available_sample_tokens_sha256",
            "selected_sample_tokens_sha256",
        }:
            raise ValueError("Full128 route-plan dataset selection fields differ")
        selected_tokens = tokens_by_dataset[expected_name]
        available_count = item["available_sample_count"]
        selected_count = item["selected_sample_count"]
        if (
            item["dataset_name"] != expected_name
            or isinstance(available_count, bool)
            or not isinstance(available_count, int)
            or isinstance(selected_count, bool)
            or not isinstance(selected_count, int)
            or available_count < selected_count
            or selected_count != len(selected_tokens)
            or selected_count <= 0
        ):
            raise ValueError("Full128 route-plan dataset selection counts differ")
        for field in (
            "available_sample_tokens_sha256",
            "selected_sample_tokens_sha256",
        ):
            _require_sha256(item[field], field)
        if item["selected_sample_tokens_sha256"] != content_sha256(selected_tokens):
            raise ValueError("Full128 selected sample-token digest differs")
        if mode == "COMPLETE_DATASETS" and (
            available_count != selected_count
            or item["available_sample_tokens_sha256"]
            != item["selected_sample_tokens_sha256"]
        ):
            raise ValueError("complete Full128 dataset selection differs")
        if mode == "DETERMINISTIC_MAXIMUM_PER_DATASET" and selected_count > maximum:
            raise ValueError("bounded Full128 dataset selection exceeds its maximum")
        if mode == "DETERMINISTIC_MAXIMUM_PER_DATASET" and selected_count != min(
            maximum, available_count
        ):
            raise ValueError("bounded Full128 dataset selection is not a full prefix")


def _validate_route_evidence(row: Mapping[str, Any], route: RouteIntent) -> None:
    evidence = row["route_evidence"]
    if not isinstance(evidence, dict):
        raise TypeError("Full128 route evidence must be an object")
    for field in (
        "annotation_artifact",
        "label_artifact",
        "trimap_artifact",
        "head_artifact",
    ):
        artifact = evidence.get(field)
        if artifact is not None:
            _validate_serialized_artifact(artifact, field)
    dataset = row["dataset_name"]
    legacy = row["schema_version"] == _LEGACY_ROUTE_PLAN_RECORD_SCHEMA
    exact_one_intent = (
        "REQUIRE_EXACTLY_ONE_OR_SEPARATE_AUTHORITY"
        if legacy
        else "REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG"
    )
    auxiliary_intent = (
        "REQUIRE_EXACTLY_ONE_OR_SEPARATE_AUTHORITY"
        if legacy
        else "SELECT_LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS"
    )
    if dataset == "dogfacenet224":
        if route is not RouteIntent.BODY_PARSING or evidence != {
            "kind": "PUBLISHER_FACE_CROP_SOURCE",
            "source_sha256": row["source_sha256"],
            "association_intent": exact_one_intent,
        }:
            raise ValueError("DogFaceNet Full128 route evidence differs")
        return
    if dataset == "dogflw":
        expected_fields = {
            "kind",
            "label_artifact",
            "bbox_xyxy",
            "bbox_state",
            "normalized_nonstandard_constants",
            "annotation_usage",
            "association_intent",
        }
        if (
            set(evidence) != expected_fields
            or evidence.get("kind") != "DOGFLW_FACE_BBOX"
        ):
            raise ValueError("DogFLW Full128 route evidence fields differ")
        if route is not RouteIntent.BODY_PARSING or evidence.get("bbox_state") not in {
            "VALID_WITHIN_SOURCE",
            "INVALID_OR_OUT_OF_BOUNDS",
        }:
            raise ValueError("DogFLW Full128 route intent differs")
        valid = evidence.get("bbox_state") == "VALID_WITHIN_SOURCE"
        constants = evidence.get("normalized_nonstandard_constants")
        if not isinstance(constants, dict) or any(
            token not in {"NaN", "Infinity", "-Infinity"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for token, count in constants.items()
        ):
            raise ValueError("DogFLW non-standard JSON constant audit differs")
        box = _finite_box(evidence.get("bbox_xyxy"))
        within = box is not None and _box_within(
            box, row["source_width"], row["source_height"]
        )
        if (
            within != valid
            or evidence.get("annotation_usage")
            != "AUDIT_AND_FUTURE_PARSER_DEVELOPMENT_ONLY"
            or evidence.get("association_intent")
            != auxiliary_intent
        ):
            raise ValueError("DogFLW Full128 bbox evidence differs")
        return
    if dataset == "oxford-pets-dog":
        expected_fields = {
            "kind",
            "trimap_artifact",
            "trimap_width",
            "trimap_height",
            "observed_labels",
            "trimap_state",
            "label_policy",
            "annotation_usage",
            "head_annotation_state",
            "association_intent",
        }
        head_available = evidence.get("head_annotation_state") == "AVAILABLE"
        if head_available:
            expected_fields.update({"head_artifact", "head_bbox_xyxy"})
        if (
            set(evidence) != expected_fields
            or route is not RouteIntent.BODY_PARSING
            or evidence.get("kind") != "OXFORD_PUBLISHER_ANNOTATIONS"
            or evidence.get("label_policy") != {"foreground": 1, "excluded": [2, 3]}
            or evidence.get("annotation_usage")
            != "AUDIT_AND_FUTURE_PARSER_DEVELOPMENT_ONLY"
            or evidence.get("head_annotation_state") not in {"AVAILABLE", "ABSENT"}
            or evidence.get("association_intent")
            != auxiliary_intent
            or evidence.get("trimap_state")
            not in {
                "VALID_FOREGROUND",
                "INVALID_PIXEL_MODE",
                "DIMENSIONS_DIFFER",
                "INVALID_LABELS",
                "ALL_EXCLUDED",
            }
        ):
            raise ValueError("Oxford Full128 route evidence fields differ")
        valid = evidence.get("trimap_state") == "VALID_FOREGROUND"
        labels = evidence.get("observed_labels")
        if (
            not isinstance(labels, list)
            or any(
                isinstance(label, bool)
                or not isinstance(label, int)
                or not 0 <= label <= 255
                for label in labels
            )
            or labels != sorted(set(labels))
            or isinstance(evidence.get("trimap_width"), bool)
            or not isinstance(evidence.get("trimap_width"), int)
            or isinstance(evidence.get("trimap_height"), bool)
            or not isinstance(evidence.get("trimap_height"), int)
            or evidence["trimap_width"] <= 0
            or evidence["trimap_height"] <= 0
        ):
            raise ValueError("Oxford Full128 trimap metadata differs")
        if valid and (
            not set(labels).issubset({1, 2, 3})
            or 1 not in labels
            or (evidence["trimap_width"], evidence["trimap_height"])
            != (row["source_width"], row["source_height"])
        ):
            raise ValueError("Oxford Full128 foreground trimap evidence differs")
        if head_available:
            head_box = _finite_box(evidence.get("head_bbox_xyxy"))
            if head_box is None or not _box_within(
                head_box, row["source_width"], row["source_height"]
            ):
                raise ValueError("Oxford Full128 head bbox evidence differs")
        return
    if dataset == "ap10k-dog":
        if (
            route is not RouteIntent.BODY_PARSING
            or set(evidence)
            != {
                "kind",
                "annotation_artifact",
                "annotation_id",
                "image_id",
                "bbox_xyxy",
                "association_intent",
            }
            or evidence.get("kind") != "AP10K_AUTHORITATIVE_BBOX_ASSOCIATION"
        ):
            raise ValueError("AP-10K Full128 route evidence differs")
        annotation_id = evidence.get("annotation_id")
        image_id = evidence.get("image_id")
        bbox = _finite_box(evidence.get("bbox_xyxy"))
        association = evidence.get("association_intent")
        if (
            isinstance(annotation_id, bool)
            or not isinstance(annotation_id, int)
            or isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or bbox is None
            or not _box_within(bbox, row["source_width"], row["source_height"])
            or not isinstance(association, dict)
            or set(association) != {"kind", "authority_sha256"}
            or association.get("kind") != "AUTHORITATIVE_BBOX_MATCH"
        ):
            raise ValueError("AP-10K Full128 bbox association evidence differs")
        _require_sha256(association["authority_sha256"], "authority_sha256")
        authority = {
            "annotation_artifact_sha256": evidence["annotation_artifact"]["sha256"],
            "annotation_id": annotation_id,
            "image_id": image_id,
            "bbox_xyxy": list(bbox),
        }
        if association["authority_sha256"] != content_sha256(authority):
            raise ValueError("AP-10K Full128 bbox authority digest differs")
        return
    if route is not RouteIntent.BODY_PARSING or evidence != {
        "kind": "PARSER_REQUIRED",
        "association_intent": exact_one_intent,
    }:
        raise ValueError("Full128 generic body-parsing evidence differs")


def _validate_serialized_artifact(value: object, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "relative_path",
        "sha256",
        "byte_size",
    }:
        raise ValueError(f"Full128 {label} binding fields differ")
    _safe_relative_path(value["relative_path"], label)
    _require_sha256(value["sha256"], f"{label} sha256")
    if (
        isinstance(value["byte_size"], bool)
        or not isinstance(value["byte_size"], int)
        or value["byte_size"] <= 0
        or value["byte_size"] > _MAX_LABEL_BYTES
    ):
        raise ValueError(f"Full128 {label} byte size differs")


def _verify_serialized_artifact(root: Path, value: object, label: str) -> None:
    _validate_serialized_artifact(value, label)
    _, binding = _read_regular_file(
        root,
        _safe_relative_path(value["relative_path"], label),
        maximum_bytes=value["byte_size"],
        subject=f"Full128 {label}",
    )
    if binding.sha256 != value["sha256"] or binding.byte_size != value["byte_size"]:
        raise ValueError(f"Full128 bound {label} differs")


def _validated_dataset_names(dataset_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(dataset_names, (str, bytes)):
        raise TypeError("Full128 dataset names must be a sequence")
    names = tuple(dataset_names)
    if len(names) != len(set(names)) or any(
        not isinstance(name, str) for name in names
    ):
        raise ValueError("Full128 dataset names must be unique text")
    for name in names:
        try:
            record = get_record(name)
        except KeyError:
            raise ValueError(
                f"Full128 dataset is absent from source_lock: {name}"
            ) from None
        if record.admission not in _ADMITTED_STATES:
            raise ValueError(f"Full128 dataset is blocked by source_lock: {name}")
    if set(names) != set(CANONICAL_DATASETS):
        raise ValueError("Full128 route plan requires the canonical seven datasets")
    return CANONICAL_DATASETS


def _validate_sample(sample: object, dataset: Any) -> None:
    if not isinstance(sample, UnifiedCanidSample):
        raise TypeError("Full128 adapters must return UnifiedCanidSample values")
    if (
        sample.dataset_name != dataset.canonical_name
        or sample.dataset_version != dataset.version
    ):
        raise ValueError("Full128 sample dataset metadata differs from source_lock")
    _require_sha256(sample.image_sha256, "UnifiedCanidSample image_sha256")
    if (
        isinstance(sample.width, bool)
        or not isinstance(sample.width, int)
        or isinstance(sample.height, bool)
        or not isinstance(sample.height, int)
        or sample.width <= 0
        or sample.height <= 0
        or sample.width * sample.height > _MAX_SOURCE_PIXELS
    ):
        raise ValueError("Full128 source dimensions exceed policy")
    if not isinstance(sample.split_role, str) or not sample.split_role:
        raise ValueError("Full128 sample split must be non-empty text")


def _maximum_samples(value: int | None) -> int | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError("maximum_samples_per_dataset must be positive or None")
    return value


def _canonical_dataset_root(root: Path, dataset_name: str) -> Path:
    if not root.is_absolute() or Path(os.path.abspath(os.fspath(root))) != root:
        raise ValueError(f"{dataset_name} canonical dataset root must be absolute")
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(
            f"{dataset_name} canonical dataset root is missing"
        ) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(
            f"{dataset_name} canonical dataset root must resolve to a directory"
        )
    return resolved


def _safe_relative_path(value: object, subject: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{subject} path must be non-empty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe {subject} relative path: {value!r}")
    return value


def _verified_image_size(payload: bytes, *, subject: str) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > _MAX_SOURCE_PIXELS:
                raise ValueError(f"{subject} dimensions exceed policy")
            opened.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{subject} is not a bounded supported image") from exc
    return width, height


def _trimap_state(
    payload: bytes, *, sample: UnifiedCanidSample
) -> tuple[str, list[int], int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > _MAX_SOURCE_PIXELS:
                raise ValueError("Oxford trimap dimensions exceed policy")
            if opened.mode not in {"L", "P"}:
                opened.verify()
                return "INVALID_PIXEL_MODE", [], width, height
            opened.load()
            labels = sorted(set(opened.get_flattened_data()))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Oxford trimap is not a bounded supported image") from exc
    if (width, height) != (sample.width, sample.height):
        return "DIMENSIONS_DIFFER", labels, width, height
    if not set(labels).issubset({1, 2, 3}):
        return "INVALID_LABELS", labels, width, height
    if 1 not in labels:
        return "ALL_EXCLUDED", labels, width, height
    return "VALID_FOREGROUND", labels, width, height


def _oxford_head_box(
    payload: bytes, *, image_name: str, width: int, height: int
) -> tuple[float, float, float, float]:
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("unsafe Oxford head XML artifact")
    try:
        annotation = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("Oxford head XML artifact is malformed") from exc
    objects = annotation.findall("object")
    if (
        annotation.findtext("filename") != image_name
        or len(objects) != 1
        or objects[0].findtext("name") != "dog"
    ):
        raise ValueError("Oxford head XML artifact schema differs")
    box = objects[0].find("bndbox")
    if box is None:
        raise ValueError("Oxford head XML artifact lacks a bbox")
    try:
        values = tuple(
            float(box.findtext(name, "")) for name in ("xmin", "ymin", "xmax", "ymax")
        )
    except ValueError as exc:
        raise ValueError("Oxford head XML bbox values differ") from exc
    if not _box_within(values, width, height):
        raise ValueError("Oxford head XML bbox is out of bounds")
    return values


def _finite_box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(item) for item in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        return None
    return box


def _box_within(box: Sequence[float], width: int, height: int) -> bool:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        return False
    x_min, y_min, x_max, y_max = box
    return 0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height


def _ap10k_indexes(
    document: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    raw_images = document.get("images")
    raw_annotations = document.get("annotations")
    if not isinstance(raw_images, list) or not isinstance(raw_annotations, list):
        raise TypeError("AP-10K annotation artifact schema differs")
    images: dict[int, dict[str, Any]] = {}
    annotations: dict[int, dict[str, Any]] = {}
    for value, target, label in (
        (raw_images, images, "image"),
        (raw_annotations, annotations, "annotation"),
    ):
        for row in value:
            if not isinstance(row, dict):
                raise TypeError(f"AP-10K {label} row must be an object")
            identifier = row.get("id")
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or identifier in target
            ):
                raise ValueError(f"AP-10K {label} IDs differ")
            target[identifier] = row
    return images, annotations


def _route_policy() -> dict[str, Any]:
    return {
        "schema_version": ROUTE_POLICY_SCHEMA,
        "canonical_datasets": list(CANONICAL_DATASETS),
        "target": {
            "size": _TARGET_SIZE,
            "context_fraction": _CONTEXT_FRACTION,
            "background_rgb": list(_BACKGROUND_RGB),
        },
        "parser_cache_key_schema": PARSER_CACHE_KEY_SCHEMA,
        "rules": {
            "dogfacenet224": (
                "BODY_PARSING;PUBLISHER_FACE_CROP_IS_SOURCE_ONLY;"
                "REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG"
            ),
            "dogflw": (
                "NORMALIZE_NONSTANDARD_NUMERIC_CONSTANTS_TO_NULL;"
                "BODY_PARSING;FACE_BBOX_AUDIT_ONLY;AUXILIARY_ONLY;"
                "SELECT_LARGEST_USABLE_OR_REVIEW_DOG_BY_FOREGROUND_PIXELS;"
                "TIE_BREAK_LOWEST_INSTANCE_INDEX"
            ),
            "oxford-pets-dog": (
                "BODY_PARSING;TRIMAP_AND_XML_HEAD_AUDIT_ONLY;AUXILIARY_ONLY;"
                "SELECT_LARGEST_USABLE_OR_REVIEW_DOG_BY_FOREGROUND_PIXELS;"
                "TIE_BREAK_LOWEST_INSTANCE_INDEX"
            ),
            "ap10k-dog": "BODY_PARSING_WITH_AUTHORITATIVE_BBOX_ASSOCIATION",
            "mpdd": "BODY_PARSING;REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
            "sibetan": "BODY_PARSING;REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
            "yt-bb-dog": (
                "BODY_PARSING;REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG;"
                "PRESERVE_VIDEO_TRACK_LABEL"
            ),
        },
    }


def _legacy_route_policy() -> dict[str, Any]:
    return {
        "schema_version": _LEGACY_ROUTE_POLICY_SCHEMA,
        "canonical_datasets": list(CANONICAL_DATASETS),
        "target": {
            "size": _TARGET_SIZE,
            "context_fraction": _CONTEXT_FRACTION,
            "background_rgb": list(_BACKGROUND_RGB),
        },
        "parser_cache_key_schema": PARSER_CACHE_KEY_SCHEMA,
        "rules": {
            "dogfacenet224": "BODY_PARSING;PUBLISHER_FACE_CROP_IS_SOURCE_ONLY",
            "dogflw": (
                "NORMALIZE_NONSTANDARD_NUMERIC_CONSTANTS_TO_NULL;"
                "BODY_PARSING;FACE_BBOX_AUDIT_ONLY"
            ),
            "oxford-pets-dog": "BODY_PARSING;TRIMAP_AND_XML_HEAD_AUDIT_ONLY",
            "ap10k-dog": "BODY_PARSING_WITH_AUTHORITATIVE_BBOX_ASSOCIATION",
            "mpdd": "BODY_PARSING",
            "sibetan": "BODY_PARSING",
            "yt-bb-dog": "BODY_PARSING",
        },
    }


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


def _json_copy(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Full128 sample metadata must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Full128 sample metadata keys must be text")
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    if isinstance(value, Enum):
        return _json_copy(value.value)
    raise TypeError("Full128 sample metadata must be JSON-compatible")


def _bounded_json_nodes(value: object) -> None:
    stack = [value]
    nodes = 0
    while stack:
        current = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError("label JSON exceeds node limit")
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_named_identity(
    parent_fd: int, name: str, opened: os.stat_result, subject: str
) -> None:
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError(f"{subject} path changed during retained read")


def _node_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


__all__ = [
    "ADMISSION_STATE_SCHEMA",
    "CANONICAL_DATASETS",
    "PARSER_CACHE_KEY_SCHEMA",
    "ROUTE_PLAN_BUNDLE_SCHEMA",
    "ROUTE_PLAN_RECORD_SCHEMA",
    "ROUTE_PLAN_SCHEMA",
    "ROUTE_POLICY_SCHEMA",
    "RouteIntent",
    "build_full128_route_plan",
    "build_parser_cache_key",
    "validate_full128_route_plan_bundle",
]
