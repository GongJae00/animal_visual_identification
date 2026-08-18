"""Materialize strict role-bound nose-region TRAIN/DEV crops."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from data.public.public_canine_manifest import (
    DOGFACE_DATASET,
    MPDD_DATASET,
    ArchiveReceiptBinding,
    PublicCanineRecord,
)
from data.public.public_canine_semantic_intake import derive_public_canine_semantics
from data.public.public_dataset_receipt_io import read_public_archive_receipt_bundle
from foundation.protected_io import (
    StrictJsonDocument,
    json_document_bytes,
    read_strict_json_document,
)
from foundation.provenance import content_sha256
from identity.registry.identity_registry import (
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
)
from identity.splits.protected_public_split import PublicSplitSourceBundle
from identity.splits.split_registry_binding import (
    validate_assignment_and_evaluator_binding,
)
from parsing.nose_region.manifest import (
    LICENSING_LANES,
    REQUIRED_DATASET_SPLITS,
    admitted_split_for_role,
    build_nose_region_manifest,
    build_protocol_plan,
    build_summary,
    encode_png_crop,
    frontality_from_keypoints,
    normalized_box_to_pixel_box,
)

_SOURCE_SPEC_SCHEMA = "cvi.nose_region_external_source_spec.v1"
_SOURCE_SCHEMA = "cvi.nose_region_archive_source.v1"
_EXISTING_SOURCE_SPEC_SCHEMA = "cvi.external_appearance_source_spec.v1"
_EXISTING_SOURCE_SCHEMA = "cvi.external_appearance_source.v1"
_ARCHIVE_DATASETS = frozenset({DOGFACE_DATASET, MPDD_DATASET})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_IMAGE_BYTES = 67_108_864
_MAX_IMAGE_PIXELS = 33_554_432
_MAX_COMPRESSION_RATIO = 200.0
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


@dataclass(frozen=True, slots=True)
class ArchiveSource:
    dataset_name: str
    archive_path: Path
    archive_receipt_path: Path
    dogface_classes_train_path: Path | None
    dogface_classes_test_path: Path | None


@dataclass(frozen=True, slots=True)
class Candidate:
    sample_token: str
    identity_token: str
    registered_dog_id: str
    dataset_name: str
    source_role: str
    split_role: str
    source_sample_id: str
    raw_frame_index: int
    capture_session_token: str


@dataclass(frozen=True, slots=True)
class ArchiveLocation:
    archive_path: Path
    record: PublicCanineRecord


@dataclass(frozen=True, slots=True)
class RoiLocation:
    path: Path
    expected_sha256: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-binding", required=True, type=Path)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--expected-split-receipt-sha256", required=True)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--external-source-spec", required=True, type=Path)
    parser.add_argument("--yt-roi-manifest", required=True, type=Path)
    parser.add_argument("--localizer-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--minimum-detector-confidence", type=float, default=0.5)
    parser.add_argument("--minimum-frontality", type=float, default=0.7)
    parser.add_argument("--minimum-native-short-side", type=int, default=32)
    return parser.parse_args(argv)


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _require_regular_file(path: Path, name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute: {path}")
    if path.is_symlink():
        raise ValueError(f"{name} path must not be a symlink: {path}")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} does not exist: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular file: {path}")


def _sha256_file(path: Path) -> str:
    _require_regular_file(path, "hashed input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, name: str) -> bytes:
    _require_regular_file(path, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or (named.st_dev, named.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise RuntimeError(f"{name} changed while being read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size or not payload:
        raise ValueError(f"{name} must be a non-empty stable file")
    return payload


def _source_spec(payload: object) -> tuple[ArchiveSource, ...]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "sources"}:
        raise ValueError("nose-region external source spec schema differs")
    schema = payload["schema_version"]
    if schema not in {_SOURCE_SPEC_SCHEMA, _EXISTING_SOURCE_SPEC_SCHEMA}:
        raise ValueError("unsupported nose-region external source spec")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError("external source spec sources must be an array")
    fields = {
        "schema_version",
        "dataset_name",
        "archive_path",
        "archive_receipt_path",
        "dogface_classes_train_path",
        "dogface_classes_test_path",
    }
    sources: list[ArchiveSource] = []
    for raw in raw_sources:
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError("nose-region archive source schema differs")
        expected_source_schema = (
            _SOURCE_SCHEMA
            if schema == _SOURCE_SPEC_SCHEMA
            else _EXISTING_SOURCE_SCHEMA
        )
        if raw["schema_version"] != expected_source_schema:
            raise ValueError("unsupported nose-region archive source schema")
        dataset = raw["dataset_name"]
        if schema == _EXISTING_SOURCE_SPEC_SCHEMA and dataset not in _ARCHIVE_DATASETS:
            continue
        if dataset not in _ARCHIVE_DATASETS:
            raise ValueError("nose-region archive dataset is unsupported")
        path_values = (raw["archive_path"], raw["archive_receipt_path"])
        if any(not isinstance(value, str) or not value for value in path_values):
            raise ValueError("archive source paths must be non-empty strings")
        train = raw["dogface_classes_train_path"]
        test = raw["dogface_classes_test_path"]
        if dataset == DOGFACE_DATASET:
            if not isinstance(train, str) or not train or not isinstance(test, str) or not test:
                raise ValueError("DogFace source requires both publisher class files")
        elif train is not None or test is not None:
            raise ValueError("only DogFace may provide publisher class files")
        source = ArchiveSource(
            dataset_name=dataset,
            archive_path=Path(raw["archive_path"]),
            archive_receipt_path=Path(raw["archive_receipt_path"]),
            dogface_classes_train_path=Path(train) if train else None,
            dogface_classes_test_path=Path(test) if test else None,
        )
        for name, path in (
            ("archive", source.archive_path),
            ("archive receipt", source.archive_receipt_path),
            ("DogFace train classes", source.dogface_classes_train_path),
            ("DogFace test classes", source.dogface_classes_test_path),
        ):
            if path is not None:
                _require_regular_file(path, name)
        sources.append(source)
    names = [source.dataset_name for source in sources]
    if set(names) != _ARCHIVE_DATASETS or len(set(names)) != len(names):
        raise ValueError("external source spec requires DogFace and MPDD exactly once")
    return tuple(sorted(sources, key=lambda source: source.dataset_name))


def _validate_source_bundle(
    payload: dict[str, Any], receipt: Mapping[str, Any]
) -> PublicSplitSourceBundle:
    source = PublicSplitSourceBundle.from_dict(payload)
    if receipt.get("source_bundle_sha256") != source.bundle_sha256:
        raise ValueError("split receipt does not bind the protected source bundle")
    if receipt.get("evidence_bindings") != [list(item) for item in source.evidence_bindings]:
        raise ValueError("split receipt evidence bindings differ from source bundle")
    raw_inputs = receipt.get("input_file_sha256s")
    if not isinstance(raw_inputs, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], str)
        for item in raw_inputs
    ):
        raise ValueError("split receipt input file hashes differ")
    input_hashes = dict(raw_inputs)
    if len(input_hashes) != len(raw_inputs) or input_hashes.get(
        "source_bundle_payload_sha256"
    ) != source.bundle_sha256:
        raise ValueError("split receipt source bundle input hash differs")
    return source


def _validate_labels_against_source(
    assignment: Mapping[str, Any],
    labels: Mapping[str, Any],
    source: PublicSplitSourceBundle,
) -> dict[str, Any]:
    source_by_token = {sample.sample_token: sample for sample in source.samples}
    assignment_by_token = {
        record["sample_token"]: record for record in assignment["records"]
    }
    labels_by_token: dict[str, Any] = {}
    for label in labels["records"]:
        token = label["sample_token"]
        sample = source_by_token.get(token)
        assigned = assignment_by_token.get(token)
        if sample is None or assigned is None:
            raise ValueError("protected labels reference a sample outside source/assignment")
        expected = (
            sample.identity_token,
            sample.source_sample_id,
            sample.dataset_identity_id,
            sample.sequence_token,
            sample.raw_frame_index,
            sample.original_split,
            sample.region,
            sample.dataset_name,
            sample.source_variant,
        )
        observed = (
            label["identity_token"],
            label["source_sample_id"],
            label["dataset_identity_id"],
            label["sequence_token"],
            label["raw_frame_index"],
            label["original_split"],
            label["region"],
            assigned["dataset_name"],
            assigned["source_variant"],
        )
        if observed != expected:
            raise ValueError("assignment, labels, and source bundle differ")
        if token != compute_sample_token(sample.source_sample_id) or (
            sample.identity_token != compute_identity_token(sample.dataset_identity_id)
        ):
            raise ValueError("protected source opaque token derivation differs")
        labels_by_token[token] = label
    if set(labels_by_token) != set(assignment_by_token):
        raise ValueError("protected label coverage differs from assignment")
    return labels_by_token


def _validate_registry_binding(
    payload: object,
    assignment: Mapping[str, Any],
    labels_by_token: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    fields = {
        "schema_version",
        "generated_at",
        "is_valid",
        "total_identities",
        "total_samples",
        "unregistered_tokens",
        "registry_manifest_sha256",
        "identity_summaries",
        "bindings",
        "assignment_sha256",
        "split_receipt_sha256",
        "tool_provenance",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("split registry binding schema differs")
    if payload["schema_version"] != "cvi.split_registry_binding.v2":
        raise ValueError("unsupported split registry binding schema")
    if payload["is_valid"] is not True or payload["unregistered_tokens"] != []:
        raise ValueError("split registry binding is not valid")
    if payload["assignment_sha256"] != receipt["assignment_sha256"] or (
        payload["split_receipt_sha256"] != receipt["receipt_sha256"]
    ):
        raise ValueError("split registry binding protected hashes differ")
    if payload["assignment_sha256"] != content_sha256(assignment):
        raise ValueError("split registry binding assignment hash differs")
    for field in ("registry_manifest_sha256", "manifest_sha256"):
        _require_sha256(payload[field], field)
    expected_manifest = content_sha256({
        key: value for key, value in payload.items() if key != "manifest_sha256"
    })
    if payload["manifest_sha256"] != expected_manifest:
        raise ValueError("split registry binding manifest digest differs")
    if not isinstance(payload["generated_at"], str) or not payload["generated_at"]:
        raise ValueError("split registry binding generated_at differs")
    if not isinstance(payload["tool_provenance"], dict):
        raise ValueError("split registry binding tool provenance differs")
    raw_bindings = payload["bindings"]
    binding_fields = {
        "identity_token",
        "registered_dog_id",
        "dataset_name",
        "identity_role",
        "model_access",
        "sample_disposition",
        "sample_count",
        "sample_tokens",
    }
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("split registry binding records must be non-empty")
    assignment_by_sample = {
        record["sample_token"]: record for record in assignment["records"]
    }
    registered_by_sample: dict[str, str] = {}
    identity_ids: dict[str, str] = {}
    for binding in raw_bindings:
        if not isinstance(binding, dict) or set(binding) != binding_fields:
            raise ValueError("split registry binding record schema differs")
        identity_token = _require_sha256(binding["identity_token"], "identity_token")
        samples = binding["sample_tokens"]
        if (
            not isinstance(samples, list)
            or not samples
            or samples != sorted(samples)
            or len(samples) != len(set(samples))
            or binding["sample_count"] != len(samples)
        ):
            raise ValueError("split registry binding sample coverage differs")
        registered_id = binding["registered_dog_id"]
        matching_label = next(
            (
                labels_by_token[token]
                for token in samples
                if token in labels_by_token
            ),
            None,
        )
        if matching_label is None:
            raise ValueError("split registry binding references no assigned samples")
        if identity_token != matching_label["identity_token"] or (
            registered_id
            != compute_registered_dog_id(matching_label["dataset_identity_id"])
        ):
            raise ValueError("split registry UUIDv5 binding differs from labels")
        if binding["dataset_name"] != matching_label["dataset_identity_id"].split(":", 1)[0]:
            raise ValueError("split registry dataset binding differs")
        prior = identity_ids.setdefault(identity_token, registered_id)
        if prior != registered_id:
            raise ValueError("identity token maps to multiple registered UUIDs")
        for token in samples:
            assigned = assignment_by_sample.get(token)
            if assigned is None or (
                assigned["identity_token"] != identity_token
                or assigned["identity_role"] != binding["identity_role"]
                or assigned["model_access"] != binding["model_access"]
                or assigned["sample_disposition"] != binding["sample_disposition"]
            ):
                raise ValueError("split registry binding differs from assignment")
            if token in registered_by_sample:
                raise ValueError("split registry binding repeats a sample token")
            registered_by_sample[token] = registered_id
    assignment_tokens = {record["sample_token"] for record in assignment["records"]}
    if set(registered_by_sample) != assignment_tokens:
        raise ValueError("split registry binding assignment coverage differs")
    if payload["total_samples"] != len(assignment_tokens) or payload[
        "total_identities"
    ] != len(identity_ids):
        raise ValueError("split registry binding totals differ")
    summaries: dict[tuple[str, str], tuple[set[str], int]] = defaultdict(
        lambda: (set(), 0)
    )
    for binding in raw_bindings:
        key = (binding["identity_role"], binding["model_access"])
        identities, count = summaries[key]
        identities.add(binding["identity_token"])
        summaries[key] = identities, count + binding["sample_count"]
    expected_summaries = [
        {
            "role": role,
            "access": access,
            "unique_identities": len(identities),
            "sample_count": count,
        }
        for (role, access), (identities, count) in sorted(summaries.items())
    ]
    if payload["identity_summaries"] != expected_summaries:
        raise ValueError("split registry binding identity summaries differ")
    return registered_by_sample


def _candidate_plan(
    assignment: Mapping[str, Any],
    labels_by_token: Mapping[str, Any],
    registered_by_sample: Mapping[str, str],
) -> tuple[tuple[Candidate, ...], dict[str, dict[str, Any]]]:
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"candidates": 0, "rejected": 0, "rejection_reasons": Counter()}
    )
    candidates: list[Candidate] = []
    for record in assignment["records"]:
        dataset = record["dataset_name"]
        role = record["identity_role"]
        split = admitted_split_for_role(role, dataset)
        if split is None:
            counts[dataset]["rejected"] += 1
            counts[dataset]["rejection_reasons"]["PROTECTED_ROLE_REJECTED"] += 1
            continue
        counts[dataset]["candidates"] += 1
        if record["source_variant"] != "original" or record[
            "sample_disposition"
        ] != "PRIMARY_ORACLE_CROP":
            counts[dataset]["rejected"] += 1
            counts[dataset]["rejection_reasons"]["NON_PRIMARY_SOURCE_REJECTED"] += 1
            continue
        label = labels_by_token[record["sample_token"]]
        candidates.append(
            Candidate(
                sample_token=record["sample_token"],
                identity_token=record["identity_token"],
                registered_dog_id=registered_by_sample[record["sample_token"]],
                dataset_name=dataset,
                source_role=role,
                split_role=split,
                source_sample_id=label["source_sample_id"],
                raw_frame_index=label["raw_frame_index"],
                capture_session_token=label["sequence_token"],
            )
        )
    normalized = {
        dataset: {
            "candidates": value["candidates"],
            "rejected": value["rejected"],
            "rejection_reasons": dict(sorted(value["rejection_reasons"].items())),
        }
        for dataset, value in counts.items()
    }
    return tuple(sorted(candidates, key=lambda value: (value.dataset_name, value.sample_token))), normalized


def _derive_archive_records(
    sources: Sequence[ArchiveSource], input_hashes: dict[str, str]
) -> tuple[dict[str, PublicCanineRecord], dict[str, str]]:
    records: dict[str, PublicCanineRecord] = {}
    archive_hashes: dict[str, str] = {}
    for source in sources:
        receipt = read_public_archive_receipt_bundle(source.archive_receipt_path)
        manifests, _ = derive_public_canine_semantics(
            dataset_name=source.dataset_name,
            archive_path=source.archive_path,
            binding=ArchiveReceiptBinding(
                source.dataset_name, receipt.archive_sha256, receipt.receipt_sha256
            ),
            dogface_classes_train=source.dogface_classes_train_path,
            dogface_classes_test=source.dogface_classes_test_path,
        )
        archive_hashes[source.dataset_name] = receipt.archive_sha256
        input_hashes[f"{source.dataset_name}_archive_sha256"] = receipt.archive_sha256
        input_hashes[f"{source.dataset_name}_archive_receipt_sha256"] = receipt.receipt_sha256
        input_hashes[f"{source.dataset_name}_archive_receipt_file_sha256"] = _sha256_file(
            source.archive_receipt_path
        )
        if source.dogface_classes_train_path is not None:
            classes_test = source.dogface_classes_test_path
            if classes_test is None:  # guarded by the strict source spec
                raise ValueError("DogFace test classes are absent")
            input_hashes["dogfacenet224_classes_train_sha256"] = _sha256_file(
                source.dogface_classes_train_path
            )
            input_hashes["dogfacenet224_classes_test_sha256"] = _sha256_file(
                classes_test
            )
        for manifest in manifests:
            for record in manifest.records:
                if record.source_sample_id in records:
                    raise ValueError("audited archive manifests repeat a source sample ID")
                records[record.source_sample_id] = record
    return records, archive_hashes


def _archive_locations(
    candidates: Sequence[Candidate],
    records: Mapping[str, PublicCanineRecord],
    sources: Sequence[ArchiveSource],
) -> dict[str, ArchiveLocation]:
    paths = {source.dataset_name: source.archive_path for source in sources}
    result: dict[str, ArchiveLocation] = {}
    for candidate in candidates:
        if candidate.dataset_name not in _ARCHIVE_DATASETS:
            continue
        record = records.get(candidate.source_sample_id)
        if record is None:
            raise ValueError("admitted archive sample is absent from audited manifest")
        if (
            record.dataset_name != candidate.dataset_name
            or compute_identity_token(record.dataset_identity_id) != candidate.identity_token
            or compute_registered_dog_id(record.dataset_identity_id)
            != candidate.registered_dog_id
        ):
            raise ValueError("admitted archive sample identity binding differs")
        result[candidate.sample_token] = ArchiveLocation(paths[candidate.dataset_name], record)
    return result


def _yt_roi_locations(
    candidates: Sequence[Candidate], manifest_path: Path, bundle: object
) -> tuple[dict[str, RoiLocation], set[str], dict[str, Any]]:
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
        or bundle["schema_version"] != "cvi.canid_roi_manifest_bundle.v2"
    ):
        raise ValueError("YT ROI manifest bundle schema differs")
    _require_sha256(bundle["manifest_sha256"], "YT ROI manifest SHA-256")
    manifest = bundle["manifest"]
    if (
        not isinstance(manifest, dict)
        or content_sha256(manifest) != bundle["manifest_sha256"]
        or set(manifest)
        != {
            "schema_version",
            "dataset_name",
            "dataset_version",
            "source_sample_ids",
            "prediction_cache_sha256s",
            "records",
        }
        or manifest["schema_version"] != "cvi.canid_roi_manifest.v2"
    ):
        raise ValueError("YT ROI manifest content binding differs")
    if manifest["dataset_name"] != "yt-bb-dog":
        raise ValueError("YT ROI manifest dataset differs")
    if not isinstance(manifest["records"], list):
        raise ValueError("YT ROI manifest records must be an array")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise ValueError("YT ROI manifest record must be an object")
        registered = record["registered_identity_id"]
        path = record["face_crop_path"]
        if registered is None or path is None:
            continue
        _require_sha256(record["face_crop_sha256"], "YT face crop SHA-256")
        image_name = PurePosixPath(record["image_path"]).name
        match = re.fullmatch(r"\d+_(\d+)\.(?:jpg|jpeg|png)", image_name, re.IGNORECASE)
        if match is None:
            continue
        key = (registered, int(match.group(1)))
        if key in by_key:
            raise ValueError("YT ROI manifest repeats an identity/frame face crop")
        by_key[key] = record
    root = manifest_path.parent.resolve(strict=True)
    result: dict[str, RoiLocation] = {}
    missing: set[str] = set()
    for candidate in candidates:
        if candidate.dataset_name != "yt-bb-dog":
            continue
        record = by_key.get((candidate.registered_dog_id, candidate.raw_frame_index))
        if record is None:
            missing.add(candidate.sample_token)
            continue
        if record["registered_identity_id"] != candidate.registered_dog_id:
            raise ValueError("YT ROI registered identity differs")
        relative = PurePosixPath(record["face_crop_path"])
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("YT ROI face crop path is unsafe")
        result[candidate.sample_token] = RoiLocation(
            path=path, expected_sha256=record["face_crop_sha256"]
        )
    return result, missing, manifest


def _safe_member_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("ZIP member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ZIP member path is unsafe")


def _read_member_bytes(archive: zipfile.ZipFile, record: PublicCanineRecord) -> bytes:
    _safe_member_path(record.member_path)
    try:
        info = archive.getinfo(record.member_path)
    except KeyError as exc:
        raise ValueError("audited image member is absent") from exc
    mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 0x1:
        raise ValueError("audited image member type is unsafe")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise ValueError("audited image member compression is unsupported")
    if info.CRC != record.member_crc32 or info.file_size != record.member_uncompressed_bytes:
        raise ValueError("audited image member metadata differs")
    if info.file_size > _MAX_IMAGE_BYTES or info.compress_size > _MAX_IMAGE_BYTES:
        raise ValueError("audited image member exceeds byte limit")
    if info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO:
        raise ValueError("audited image member exceeds compression-ratio limit")
    with archive.open(info, "r") as stream:
        payload = stream.read(_MAX_IMAGE_BYTES + 1)
    if len(payload) != info.file_size or len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError("audited image member byte count differs")
    return payload


def _open_verified_archives(
    stack: ExitStack,
    sources: Sequence[ArchiveSource],
    expected_hashes: Mapping[str, str],
) -> dict[Path, zipfile.ZipFile]:
    result: dict[Path, zipfile.ZipFile] = {}
    for source in sources:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(source.archive_path, flags)
        stream = stack.enter_context(os.fdopen(descriptor, "rb"))
        digest = hashlib.sha256()
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
        if digest.hexdigest() != expected_hashes[source.dataset_name]:
            raise ValueError("audited archive changed before materialization")
        stream.seek(0)
        result[source.archive_path] = stack.enter_context(zipfile.ZipFile(stream))
    return result


def _decode_image(payload: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as opened:
        width, height = opened.size
        if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
            raise ValueError("caller crop dimensions exceed policy")
        image = opened.convert("RGB")
        image.load()
    return image


def _detect(model: Any, device: Any, image: Any) -> tuple[list[float], float, float] | None:
    import torch
    from PIL import Image

    from parsing.nose_region.localizer import (
        INPUT_SIZE,
        NOSE_POINT_INDICES,
        image_to_tensor,
    )

    resized = image.resize(
        (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
    )
    tensor = image_to_tensor(resized).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(tensor)[0].detach().cpu()
    if prediction.shape != (8, 3) or not torch.isfinite(prediction).all():
        raise RuntimeError("nose localizer output must be finite [8,3]")
    if torch.any((prediction < 0.0) | (prediction > 1.0)):
        raise RuntimeError("nose localizer output must be normalized to [0, 1]")
    points: list[list[float]] = []
    for x, y, confidence in prediction.tolist():
        points.append([
            min(1.0, max(0.0, x)),
            min(1.0, max(0.0, y)),
            confidence,
        ])
    nose = [points[index] for index in NOSE_POINT_INDICES]
    confidence = sum(point[2] for point in nose) / len(nose)
    if confidence <= 0.0:
        return None
    margin = 0.08
    box = [
        max(0.0, min(point[0] for point in nose) - margin),
        max(0.0, min(point[1] for point in nose) - margin),
        min(1.0, max(point[0] for point in nose) + margin),
        min(1.0, max(point[1] for point in nose) + margin),
    ]
    return box, confidence, frontality_from_keypoints(points)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def _documents(args: argparse.Namespace) -> dict[str, StrictJsonDocument]:
    paths = {
        "registry_binding": args.registry_binding,
        "assignment": args.assignment,
        "labels": args.labels,
        "split_receipt": args.split_receipt,
        "source_bundle": args.source_bundle,
        "external_source_spec": args.external_source_spec,
        "yt_roi_manifest": args.yt_roi_manifest,
    }
    for name, path in paths.items():
        _require_regular_file(path, name.replace("_", " "))
    return {
        name: read_strict_json_document(
            path,
            maximum_bytes=536_870_912,
            maximum_nodes=20_000_000,
            maximum_keys=10_000_000,
            maximum_array_length=2_000_000,
        )
        for name, path in paths.items()
    }


def _base_input_hashes(
    documents: Mapping[str, StrictJsonDocument], checkpoint_sha256: str
) -> dict[str, str]:
    result = {"localizer_checkpoint_file_sha256": checkpoint_sha256}
    for name, document in documents.items():
        result[f"{name}_file_sha256"] = document.raw_sha256
        result[f"{name}_payload_sha256"] = document.canonical_payload_sha256
    return result


def _increment_rejection(
    counts: dict[str, dict[str, Any]], dataset: str, reason: str
) -> None:
    counts[dataset]["rejected"] += 1
    reasons = counts[dataset]["rejection_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def run(args: argparse.Namespace) -> dict[str, Any]:
    from parsing.nose_region.native_yt import load_localizer_checkpoint

    expected_receipt_sha256 = _require_sha256(
        args.expected_split_receipt_sha256, "expected split receipt SHA-256"
    )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    output_parent = args.output_dir.parent.resolve(strict=True)
    if not output_parent.is_dir():
        raise NotADirectoryError(output_parent)
    checkpoint_bytes = _read_regular_bytes(
        args.localizer_checkpoint, "localizer checkpoint"
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    documents = _documents(args)
    assignment = documents["assignment"].payload
    labels = documents["labels"].payload
    receipt = documents["split_receipt"].payload
    validate_assignment_and_evaluator_binding(
        assignment, receipt, labels, expected_receipt_sha256
    )
    source = _validate_source_bundle(documents["source_bundle"].payload, receipt)
    labels_by_token = _validate_labels_against_source(assignment, labels, source)
    registered_by_sample = _validate_registry_binding(
        documents["registry_binding"].payload,
        assignment,
        labels_by_token,
        receipt,
    )
    sources = _source_spec(documents["external_source_spec"].payload)
    input_hashes = _base_input_hashes(documents, checkpoint_sha256)
    input_hashes.update({
        "protected_assignment_sha256": receipt["assignment_sha256"],
        "protected_labels_sha256": receipt["evaluator_binding_sha256"],
        "protected_source_bundle_sha256": receipt["source_bundle_sha256"],
        "protected_split_receipt_sha256": receipt["receipt_sha256"],
        "registry_binding_manifest_sha256": documents["registry_binding"].payload[
            "manifest_sha256"
        ],
        "identity_registry_manifest_sha256": documents["registry_binding"].payload[
            "registry_manifest_sha256"
        ],
    })
    archive_records, archive_hashes = _derive_archive_records(sources, input_hashes)
    candidates, plan_counts = _candidate_plan(
        assignment, labels_by_token, registered_by_sample
    )
    archive_locations = _archive_locations(candidates, archive_records, sources)
    roi_locations, missing_yt_roi, roi_manifest = _yt_roi_locations(
        candidates,
        args.yt_roi_manifest,
        documents["yt_roi_manifest"].payload,
    )
    input_hashes["yt_roi_manifest_sha256"] = content_sha256(roi_manifest)
    if (
        len(archive_locations) + len(roi_locations) + len(missing_yt_roi)
        != len(candidates)
    ):
        raise ValueError("candidate source location coverage differs")

    model, device, localizer_bindings = load_localizer_checkpoint(
        checkpoint_bytes, args.device
    )
    input_hashes["localizer_bindings_sha256"] = localizer_bindings[
        "content_sha256"
    ]
    for name, digest in localizer_bindings["sources"].items():
        input_hashes[f"localizer_{name}"] = digest
    policy = {
        "minimum_detector_confidence": args.minimum_detector_confidence,
        "minimum_frontality": args.minimum_frontality,
        "minimum_native_short_side": args.minimum_native_short_side,
        "frontality_metric": (
            "EYE_MIDLINE_NOSE_OFFSET_ROLL_WITH_KEYPOINT_CONFIDENCE_V1"
        ),
        "crop_encoding": "PNG_RGB_LOSSLESS",
        "path_layout": "FLAT_SAMPLE_TOKEN_HASH",
    }
    plan = build_protocol_plan(
        input_sha256s=input_hashes,
        policy=policy,
        dataset_counts=plan_counts,
    )
    print(json.dumps(plan, sort_keys=True), flush=True)
    args.output_dir.mkdir(mode=0o700)
    _write_exclusive(
        args.output_dir / "protocol-plan.json", json_document_bytes(plan)
    )
    crop_root = args.output_dir / "crops"
    crop_root.mkdir(mode=0o700)

    final_counts = {
        dataset: {
            "candidates": values["candidates"],
            "admitted": 0,
            "rejected": values["rejected"],
            "rejection_reasons": dict(values["rejection_reasons"]),
        }
        for dataset, values in plan_counts.items()
    }
    records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        archives = _open_verified_archives(stack, sources, archive_hashes)
        for candidate in candidates:
            if candidate.sample_token in missing_yt_roi:
                _increment_rejection(
                    final_counts, candidate.dataset_name, "NO_VERIFIED_FACE_ROI"
                )
                continue
            if candidate.sample_token in archive_locations:
                location = archive_locations[candidate.sample_token]
                source_bytes = _read_member_bytes(
                    archives[location.archive_path], location.record
                )
            else:
                location = roi_locations[candidate.sample_token]
                source_bytes = _read_regular_bytes(
                    location.path, "YT ROI face crop"
                )
                if hashlib.sha256(source_bytes).hexdigest() != location.expected_sha256:
                    raise ValueError("YT ROI face crop changed after manifest validation")
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            image = _decode_image(source_bytes)
            detected = _detect(model, device, image)
            if detected is None:
                _increment_rejection(final_counts, candidate.dataset_name, "NO_NOSE_DETECTION")
                continue
            normalized_box, confidence, frontality = detected
            if confidence < policy["minimum_detector_confidence"]:
                _increment_rejection(final_counts, candidate.dataset_name, "LOW_CONFIDENCE")
                continue
            if frontality < policy["minimum_frontality"]:
                _increment_rejection(final_counts, candidate.dataset_name, "LOW_FRONTALITY")
                continue
            try:
                pixel_box = normalized_box_to_pixel_box(
                    normalized_box, image.width, image.height
                )
            except ValueError:
                _increment_rejection(final_counts, candidate.dataset_name, "INVALID_NOSE_BOX")
                continue
            native_short_side = min(
                pixel_box[2] - pixel_box[0], pixel_box[3] - pixel_box[1]
            )
            if native_short_side < policy["minimum_native_short_side"]:
                _increment_rejection(final_counts, candidate.dataset_name, "LOW_NATIVE_RESOLUTION")
                continue
            png_bytes, crop_size = encode_png_crop(image, pixel_box)
            crop_sha256 = hashlib.sha256(png_bytes).hexdigest()
            relative_path = f"crops/{candidate.sample_token}.png"
            _write_exclusive(crop_root / f"{candidate.sample_token}.png", png_bytes)
            final_counts[candidate.dataset_name]["admitted"] += 1
            records.append({
                "dataset_name": candidate.dataset_name,
                "sample_token": candidate.sample_token,
                "identity_token": candidate.identity_token,
                "registered_dog_id": candidate.registered_dog_id,
                "capture_session_token": candidate.capture_session_token,
                "source_sha256": source_sha256,
                "source_width": image.width,
                "source_height": image.height,
                "crop_path": relative_path,
                "crop_sha256": crop_sha256,
                "crop_width": crop_size[0],
                "crop_height": crop_size[1],
                "detector_confidence": confidence,
                "frontality": frontality,
                "nose_box_xyxy": list(pixel_box),
                "source_role": candidate.source_role,
                "split_role": candidate.split_role,
                "licensing_lane": LICENSING_LANES[candidate.dataset_name],
            })

    for dataset_name in REQUIRED_DATASET_SPLITS:
        if final_counts.get(dataset_name, {}).get("admitted", 0) <= 0:
            raise ValueError(f"required materialized dataset is absent: {dataset_name}")
    records.sort(key=lambda record: (record["dataset_name"], record["sample_token"]))
    summary = build_summary(
        input_sha256s=input_hashes,
        dataset_counts=final_counts,
        protocol_plan_sha256=plan["plan_sha256"],
    )
    bundle = build_nose_region_manifest(
        records=records,
        input_sha256s=input_hashes,
        policy=policy,
        summary=summary,
    )
    _write_exclusive(
        args.output_dir / "nose-region-manifest.json", json_document_bytes(bundle)
    )
    _write_exclusive(args.output_dir / "summary.json", json_document_bytes(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return bundle


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
