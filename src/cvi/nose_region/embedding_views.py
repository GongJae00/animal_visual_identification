"""Strict student-mask support cache and shared masked-RGB reconstruction."""

from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image

from cvi.evidence.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    NoseMaskManifest,
    UsageLane,
    preprocess_image,
)
from cvi.nose_region.native_yt import validate_manifest_bundle
from cvi.protected_io import json_document_bytes, read_strict_json_document
from cvi.protected_publication import fsync_directory, rename_directory_noreplace
from cvi.provenance import content_sha256


MANIFEST_SCHEMA = "cvi.nose_embedding_views.v1"
MANIFEST_FILENAME = "nose-embedding-views.json"
INTERPRETATION = "RESEARCH_ONLY_DERIVED_SUPPORT_CACHE_NOT_BIOMETRIC_VALIDATION"
IMAGE_SIZE = 224
OUTSIDE_SUPPORT_ORIGINAL_WEIGHT = 0.25
_SHA256_PATHS = (
    "src/cvi/nose_region/embedding_views.py",
    "src/cvi/nose_region/native_yt.py",
    "src/cvi/nose_region/segmentation_training.py",
    "src/cvi/evidence/artifact_manifest.py",
    "src/cvi/protected_io.py",
    "src/cvi/protected_publication.py",
    "src/cvi/provenance.py",
    "tools/prepare_nose_embedding_views.py",
)
_RECORD_FIELDS = {
    "sample_token",
    "registered_dog_id",
    "identity_token",
    "track_token",
    "frame_index",
    "source_crop_path",
    "source_crop_sha256",
    "support_path",
    "support_sha256",
    "support_fraction",
    "mean_probability",
    "mean_binary_uncertainty",
    "threshold_margin_le_0_05_fraction",
}


def student_masked_rgb(
    crop_rgb: np.ndarray,
    support: np.ndarray,
    *,
    outside_support_original_weight: float = OUTSIDE_SUPPORT_ORIGINAL_WEIGHT,
) -> np.ndarray:
    """Reconstruct the canonical 224 RGB student-mask embedding view."""

    image = np.asarray(crop_rgb)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("crop_rgb must be a nonempty HxWx3 uint8 array")
    if (
        isinstance(outside_support_original_weight, bool)
        or not isinstance(outside_support_original_weight, (int, float))
        or not math.isfinite(float(outside_support_original_weight))
        or not 0.0 <= float(outside_support_original_weight) <= 1.0
    ):
        raise ValueError("outside support original weight must be finite and in [0,1]")
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    mask = np.asarray(support)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("support must be exactly 224x224")
    if mask.dtype == np.bool_:
        binary = mask
    elif mask.dtype == np.uint8 and np.all((mask == 0) | (mask == 255)):
        binary = mask == 255
    else:
        raise ValueError("support must be boolean or binary 0/255 uint8")
    median = np.median(resized, axis=(0, 1)).astype(np.float32)
    weight = float(outside_support_original_weight)
    outside = weight * resized.astype(np.float32) + (1.0 - weight) * median
    return np.where(binary[..., None], resized, np.rint(outside)).astype(np.uint8)


def reconstruct_student_masked_rgb(
    *,
    native_root: Path,
    view_root: Path,
    native_record: Mapping[str, Any],
    view_record: Mapping[str, Any],
) -> np.ndarray:
    """Validate one cached support and reconstruct its masked RGB dataset view."""

    source_root = _absolute_directory(native_root, "native root")
    support_root = _absolute_directory(view_root, "view root")
    _validate_view_record(view_record)
    expected = {
        "sample_token": native_record.get("sample_token"),
        "registered_dog_id": native_record.get("registered_dog_id"),
        "identity_token": native_record.get("identity_token"),
        "track_token": native_record.get("track_token"),
        "frame_index": native_record.get("frame_index"),
        "source_crop_path": native_record.get("crop_path"),
        "source_crop_sha256": native_record.get("crop_sha256"),
    }
    for field, value in expected.items():
        if view_record[field] != value:
            raise ValueError(f"embedding view {field} differs from native record")
    crop_path = _bound_relative_file(
        source_root, view_record["source_crop_path"], view_record["source_crop_sha256"]
    )
    support_path = _bound_relative_file(
        support_root, view_record["support_path"], view_record["support_sha256"]
    )
    crop = _read_rgb_crop(crop_path, native_record)
    support = _read_support(support_path, view_record["support_sha256"])
    return student_masked_rgb(crop, support)


def prepare_embedding_views(
    *,
    native_bundle_path: Path,
    native_bundle_sha256: str,
    native_root: Path,
    student_lineage_path: Path,
    student_lineage_sha256: str,
    student_root: Path,
    mask_manifest_path: Path,
    mask_manifest_sha256: str,
    mask_onnx_path: Path,
    output_dir: Path,
    use_cuda: bool = False,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run the exact batch-one mask student and atomically publish supports."""

    expected_native = _require_sha256(native_bundle_sha256, "native_bundle_sha256")
    expected_lineage = _require_sha256(
        student_lineage_sha256, "student_lineage_sha256"
    )
    expected_manifest = _require_sha256(
        mask_manifest_sha256, "mask_manifest_sha256"
    )
    for path, name in (
        (native_bundle_path, "native bundle"),
        (student_lineage_path, "student lineage"),
        (mask_manifest_path, "mask runtime manifest"),
        (mask_onnx_path, "mask ONNX"),
    ):
        if not path.is_absolute() or path.is_symlink():
            raise ValueError(f"{name} must be an absolute non-symlink file")
    repository = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    output = _external_new_directory(output_dir, repository)
    root = _absolute_directory(native_root, "native root")
    lineage_root = _absolute_directory(student_root, "student root")

    native_document = read_strict_json_document(
        native_bundle_path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if native_document.canonical_payload_sha256 != expected_native:
        raise ValueError("native manifest bundle content SHA-256 differs from external pin")
    native_manifest = validate_manifest_bundle(native_document.payload, root=root)

    lineage_document = read_strict_json_document(student_lineage_path)
    if lineage_document.canonical_payload_sha256 != expected_lineage:
        raise ValueError("segmentation student lineage content SHA-256 differs from external pin")
    from cvi.nose_region.segmentation_training import validate_lineage_manifest

    validate_lineage_manifest(lineage_document.payload, lineage_root)
    lineage_artifacts = lineage_document.payload["artifacts"]
    _require_same_file(
        student_lineage_path, lineage_root / "artifact_lineage.json", "student lineage"
    )
    _require_same_file(
        mask_manifest_path,
        lineage_root / lineage_artifacts["runtime_manifest"]["path"],
        "mask runtime manifest",
    )
    _require_same_file(
        mask_onnx_path,
        lineage_root / lineage_artifacts["onnx"]["path"],
        "mask ONNX",
    )

    mask_document = read_strict_json_document(mask_manifest_path)
    if mask_document.canonical_payload_sha256 != expected_manifest:
        raise ValueError("NoseMaskManifest content SHA-256 differs from external pin")
    mask_manifest = NoseMaskManifest.from_dict(mask_document.payload)
    if mask_manifest.license.usage_lane is not UsageLane.RESEARCH_ONLY:
        raise ArtifactContractError("mask student must be in the RESEARCH_ONLY lane")
    if (
        mask_manifest.input_shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        or mask_manifest.output_shape != (1, 1, IMAGE_SIZE, IMAGE_SIZE)
        or mask_manifest.preprocessing.resize != "bilinear"
        or mask_manifest.preprocessing.clahe is not None
    ):
        raise ArtifactContractError(
            "mask student must declare batch-one 224 RGB bilinear preprocessing without CLAHE"
        )
    onnx_binding = _file_binding(mask_onnx_path)
    if (
        onnx_binding["sha256"] != mask_manifest.artifact_sha256
        or onnx_binding["sha256"] != lineage_artifacts["onnx"]["sha256"]
        or onnx_binding["byte_size"] != lineage_artifacts["onnx"]["bytes"]
    ):
        raise ArtifactContractError("mask ONNX hash or byte count differs from manifest lineage")
    runtime = ExactOnnxRuntime(mask_onnx_path, mask_manifest, use_cuda=use_cuda)

    code_sha256s = _code_sha256s(repository)
    transform = {
        "crop_resize": "PIL_RGB_BILINEAR",
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "support": "probability_greater_than_or_equal_to_threshold",
        "threshold": mask_manifest.threshold,
        "threshold_margin": 0.05,
        "inside_support": "retain_resized_RGB",
        "outside_support_original_weight": OUTSIDE_SUPPORT_ORIGINAL_WEIGHT,
        "outside_support_median_weight": 1.0 - OUTSIDE_SUPPORT_ORIGINAL_WEIGHT,
        "outside_support_median": "per_image_per_channel_resized_RGB_median",
        "outside_support_quantization": "numpy_rint_then_uint8",
    }
    records: list[dict[str, Any]] = []
    supports: dict[str, bytes] = {}
    for native_record in native_manifest["records"]:
        if native_record["record_state"] == "NO_ROI":
            continue
        crop_path = _bound_relative_file(
            root, native_record["crop_path"], native_record["crop_sha256"]
        )
        crop = _read_rgb_crop(crop_path, native_record)
        probability = runtime.run(
            preprocess_image(Image.fromarray(crop, mode="RGB"), mask_manifest)
        )[0, 0]
        if probability.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ArtifactContractError("student mask output must be exactly 224x224")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ArtifactContractError("student mask probabilities must be in [0,1]")
        support = probability >= mask_manifest.threshold
        support_bytes = _support_png_bytes(support)
        support_path = f"supports/{native_record['sample_token']}.png"
        support_sha256 = hashlib.sha256(support_bytes).hexdigest()
        supports[support_path] = support_bytes
        uncertainty = 1.0 - np.abs(2.0 * probability.astype(np.float64) - 1.0)
        records.append(
            {
                "sample_token": native_record["sample_token"],
                "registered_dog_id": native_record["registered_dog_id"],
                "identity_token": native_record["identity_token"],
                "track_token": native_record["track_token"],
                "frame_index": native_record["frame_index"],
                "source_crop_path": native_record["crop_path"],
                "source_crop_sha256": native_record["crop_sha256"],
                "support_path": support_path,
                "support_sha256": support_sha256,
                "support_fraction": float(support.mean()),
                "mean_probability": float(probability.mean()),
                "mean_binary_uncertainty": float(uncertainty.mean()),
                "threshold_margin_le_0_05_fraction": float(
                    (np.abs(probability - mask_manifest.threshold) <= 0.05).mean()
                ),
            }
        )
    if not records:
        raise ValueError("embedding view cache requires at least one native crop")
    if records != sorted(records, key=lambda row: row["sample_token"]):
        raise RuntimeError("native crop order is not canonical")

    body = {
        "schema_version": MANIFEST_SCHEMA,
        "source_binding": {
            "native_bundle_path": os.fspath(native_bundle_path),
            "native_bundle_file_sha256": native_document.raw_sha256,
            "native_bundle_payload_sha256": native_document.canonical_payload_sha256,
            "native_manifest_sha256": native_document.payload["manifest_sha256"],
            "native_root": os.fspath(root),
        },
        "student_binding": {
            "lineage_path": os.fspath(student_lineage_path),
            "lineage_file_sha256": lineage_document.raw_sha256,
            "lineage_payload_sha256": lineage_document.canonical_payload_sha256,
            "lineage_sha256": lineage_document.payload["lineage_sha256"],
            "student_root": os.fspath(lineage_root),
            "mask_manifest_path": os.fspath(mask_manifest_path),
            "mask_manifest_file_sha256": mask_document.raw_sha256,
            "mask_manifest_payload_sha256": mask_document.canonical_payload_sha256,
            "mask_onnx": onnx_binding,
            "usage_lane": UsageLane.RESEARCH_ONLY.value,
        },
        "runtime_binding": {
            "provider_lane": "CUDA" if use_cuda else "CPU",
            "input_name": mask_manifest.input_name,
            "input_shape": list(mask_manifest.input_shape),
            "output_name": mask_manifest.output_name,
            "output_shape": list(mask_manifest.output_shape),
            "preprocessing": mask_manifest.preprocessing.to_dict(),
        },
        "transform": transform,
        "transform_sha256": content_sha256(transform),
        "code_sha256s": code_sha256s,
        "records": records,
        "record_count": len(records),
        "interpretation": INTERPRETATION,
    }
    manifest = {**body, "manifest_sha256": content_sha256(body)}
    _publish(output, manifest, supports, repository=repository)
    validate_embedding_views_manifest(manifest, root=output, repository_root=repository)
    return manifest


def validate_embedding_views_manifest(
    payload: object,
    *,
    root: Path | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the strict support-cache schema and optionally all PNG bytes."""

    expected = {
        "schema_version",
        "source_binding",
        "student_binding",
        "runtime_binding",
        "transform",
        "transform_sha256",
        "code_sha256s",
        "records",
        "record_count",
        "interpretation",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("nose embedding views manifest keys differ")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if (
        payload["schema_version"] != MANIFEST_SCHEMA
        or payload["interpretation"] != INTERPRETATION
        or content_sha256(body) != payload["manifest_sha256"]
    ):
        raise ValueError("nose embedding views manifest content differs")
    _require_sha256(payload["manifest_sha256"], "manifest_sha256")
    source = payload["source_binding"]
    if not isinstance(source, dict) or set(source) != {
        "native_bundle_path",
        "native_bundle_file_sha256",
        "native_bundle_payload_sha256",
        "native_manifest_sha256",
        "native_root",
    }:
        raise ValueError("embedding view source binding differs")
    student = payload["student_binding"]
    if not isinstance(student, dict) or set(student) != {
        "lineage_path",
        "lineage_file_sha256",
        "lineage_payload_sha256",
        "lineage_sha256",
        "student_root",
        "mask_manifest_path",
        "mask_manifest_file_sha256",
        "mask_manifest_payload_sha256",
        "mask_onnx",
        "usage_lane",
    } or student["usage_lane"] != UsageLane.RESEARCH_ONLY.value:
        raise ValueError("embedding view student binding differs")
    runtime = payload["runtime_binding"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "provider_lane",
        "input_name",
        "input_shape",
        "output_name",
        "output_shape",
        "preprocessing",
    } or runtime["provider_lane"] not in {"CPU", "CUDA"}:
        raise ValueError("embedding view runtime binding differs")
    if runtime["input_shape"] != [1, 3, IMAGE_SIZE, IMAGE_SIZE] or runtime[
        "output_shape"
    ] != [1, 1, IMAGE_SIZE, IMAGE_SIZE] or runtime["preprocessing"].get(
        "resize"
    ) != "bilinear":
        raise ValueError("embedding view runtime shape or resize differs")
    transform = payload["transform"]
    expected_transform_keys = {
        "crop_resize",
        "image_size",
        "support",
        "threshold",
        "threshold_margin",
        "inside_support",
        "outside_support_original_weight",
        "outside_support_median_weight",
        "outside_support_median",
        "outside_support_quantization",
    }
    if (
        not isinstance(transform, dict)
        or set(transform) != expected_transform_keys
        or transform["crop_resize"] != "PIL_RGB_BILINEAR"
        or transform["image_size"] != [IMAGE_SIZE, IMAGE_SIZE]
        or transform["threshold_margin"] != 0.05
        or transform["outside_support_original_weight"]
        != OUTSIDE_SUPPORT_ORIGINAL_WEIGHT
        or transform["outside_support_median_weight"]
        != 1.0 - OUTSIDE_SUPPORT_ORIGINAL_WEIGHT
        or content_sha256(transform) != payload["transform_sha256"]
    ):
        raise ValueError("embedding view transform binding differs")
    for name in (
        "native_bundle_file_sha256",
        "native_bundle_payload_sha256",
        "native_manifest_sha256",
    ):
        _require_sha256(source[name], f"source_binding.{name}")
    for name in (
        "lineage_file_sha256",
        "lineage_payload_sha256",
        "lineage_sha256",
        "mask_manifest_file_sha256",
        "mask_manifest_payload_sha256",
    ):
        _require_sha256(student[name], f"student_binding.{name}")
    onnx = student["mask_onnx"]
    if not isinstance(onnx, dict) or set(onnx) != {"path", "sha256", "byte_size"}:
        raise ValueError("embedding view ONNX binding differs")
    _require_sha256(onnx["sha256"], "mask_onnx.sha256")
    if isinstance(onnx["byte_size"], bool) or not isinstance(onnx["byte_size"], int) or onnx[
        "byte_size"
    ] <= 0:
        raise ValueError("embedding view ONNX byte count differs")
    code = payload["code_sha256s"]
    if not isinstance(code, dict) or set(code) != set(_SHA256_PATHS):
        raise ValueError("embedding view code hash set differs")
    for name, digest in code.items():
        _require_sha256(digest, f"code_sha256s.{name}")
    repository = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    if code != _code_sha256s(repository):
        raise ValueError("embedding view code hashes differ from this checkout")
    records = payload["records"]
    if (
        not isinstance(records, list)
        or not records
        or records != sorted(records, key=lambda row: row["sample_token"])
        or payload["record_count"] != len(records)
        or len({row["sample_token"] for row in records}) != len(records)
    ):
        raise ValueError("embedding view record population differs")
    resolved_root = None if root is None else _absolute_directory(root, "view root")
    for record in records:
        _validate_view_record(record)
        if resolved_root is not None:
            support = _bound_relative_file(
                resolved_root, record["support_path"], record["support_sha256"]
            )
            values = _read_support(support, record["support_sha256"])
            if float((values == 255).mean()) != record["support_fraction"]:
                raise ValueError("cached support fraction differs")
    if resolved_root is not None:
        support_root = resolved_root / "supports"
        if support_root.is_symlink() or not support_root.is_dir():
            raise ValueError("embedding view support directory differs")
        if {path.name for path in resolved_root.iterdir()} != {
            MANIFEST_FILENAME,
            "supports",
        }:
            raise ValueError("embedding view cache contains unexpected root entries")
        if {path.relative_to(resolved_root).as_posix() for path in support_root.iterdir()} != {
            record["support_path"] for record in records
        }:
            raise ValueError("embedding view cache support file set differs")
    return payload


def load_embedding_views_manifest(
    manifest_path: Path,
    *,
    expected_payload_sha256: str,
    root: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Load a content-pinned cache manifest and validate every support file."""

    expected = _require_sha256(expected_payload_sha256, "expected_payload_sha256")
    resolved_root = _absolute_directory(root, "view root")
    _require_same_file(
        manifest_path, resolved_root / MANIFEST_FILENAME, "embedding views manifest"
    )
    document = read_strict_json_document(manifest_path)
    if document.canonical_payload_sha256 != expected:
        raise ValueError("embedding views manifest content SHA-256 differs from external pin")
    return validate_embedding_views_manifest(
        document.payload,
        root=resolved_root,
        repository_root=repository_root,
    )


def _validate_view_record(record: object) -> None:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("embedding view record keys differ")
    for name in (
        "sample_token",
        "identity_token",
        "track_token",
        "source_crop_sha256",
        "support_sha256",
    ):
        _require_sha256(record[name], name)
    try:
        registered = uuid.UUID(record["registered_dog_id"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("embedding view identity must be a canonical UUIDv5") from exc
    if registered.version != 5 or str(registered) != record["registered_dog_id"]:
        raise ValueError("embedding view identity must be a canonical UUIDv5")
    if isinstance(record["frame_index"], bool) or not isinstance(
        record["frame_index"], int
    ) or record["frame_index"] < 0:
        raise ValueError("embedding view frame index differs")
    crop = PurePosixPath(record["source_crop_path"])
    support = PurePosixPath(record["support_path"])
    if (
        record["source_crop_path"] != crop.as_posix()
        or crop.parts != ("crops", f"{record['sample_token']}.png")
        or record["support_path"] != support.as_posix()
        or support.parts != ("supports", f"{record['sample_token']}.png")
    ):
        raise ValueError("embedding view artifact path differs")
    for name in (
        "support_fraction",
        "mean_probability",
        "mean_binary_uncertainty",
        "threshold_margin_le_0_05_fraction",
    ):
        value = record[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"embedding view {name} differs")


def _publish(
    output: Path,
    manifest: dict[str, Any],
    supports: Mapping[str, bytes],
    *,
    repository: Path,
) -> None:
    parent = output.parent
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        support_dir = staging / "supports"
        support_dir.mkdir(mode=0o700)
        for relative, payload in supports.items():
            _write_exclusive(staging / relative, payload)
        _write_exclusive(staging / MANIFEST_FILENAME, json_document_bytes(manifest))
        validate_embedding_views_manifest(
            manifest, root=staging, repository_root=repository
        )
        fsync_directory(support_dir)
        fsync_directory(staging)
        rename_directory_noreplace(staging, output)
        fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
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


def _external_new_directory(path: Path, repository: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    output = parent / absolute.name
    if output.is_relative_to(repository):
        raise ValueError("embedding view cache must be outside the Git repository")
    return output


def _absolute_directory(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be a directory")
    return resolved


def _require_same_file(candidate: Path, expected: Path, name: str) -> None:
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink file")
    if candidate.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError(f"{name} path differs from segmentation lineage")


def _bound_relative_file(root: Path, relative: object, digest: object) -> Path:
    _require_sha256(digest, "artifact sha256")
    if not isinstance(relative, str):
        raise ValueError("artifact path must be a string")
    pure = PurePosixPath(relative)
    if relative != pure.as_posix() or pure.is_absolute() or ".." in pure.parts:
        raise ValueError("artifact path is unsafe")
    target = root.joinpath(*pure.parts)
    if target.is_symlink():
        raise ValueError("artifact path is unsafe")
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("artifact path is unsafe")
    if _file_sha256(resolved) != digest:
        raise ValueError("artifact SHA-256 differs")
    return resolved


def _read_rgb_crop(path: Path, record: Mapping[str, Any]) -> np.ndarray:
    payload = _read_regular_bytes(path)
    if hashlib.sha256(payload).hexdigest() != record["crop_sha256"]:
        raise ValueError("native crop SHA-256 changed before decoding")
    with Image.open(io.BytesIO(payload)) as opened:
        if (
            opened.format != "PNG"
            or opened.mode != "RGB"
            or opened.size != (record["crop_width"], record["crop_height"])
        ):
            raise ValueError("native crop image contract differs")
        values = np.asarray(opened, dtype=np.uint8)
    if values.shape != (record["crop_height"], record["crop_width"], 3):
        raise ValueError("native crop array shape differs")
    return values


def _read_support(path: Path, expected_sha256: str) -> np.ndarray:
    payload = _read_regular_bytes(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("cached support SHA-256 changed before decoding")
    with Image.open(io.BytesIO(payload)) as opened:
        if opened.format != "PNG" or opened.mode != "L" or opened.size != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError("cached support image contract differs")
        values = np.asarray(opened, dtype=np.uint8)
    if not np.all((values == 0) | (values == 255)):
        raise ValueError("cached support must be binary 0/255")
    return values


def _support_png_bytes(support: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(support.astype(np.uint8) * 255, mode="L").save(
        stream, format="PNG"
    )
    return stream.getvalue()


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"artifact must be an absolute regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    return {
        "path": os.fspath(resolved),
        "sha256": _file_sha256(resolved),
        "byte_size": resolved.stat(follow_symlinks=False).st_size,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input must be a regular file: {path}")
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
    ):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _read_regular_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input must be a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
    ):
        raise RuntimeError(f"input changed while reading: {path}")
    return b"".join(chunks)


def _code_sha256s(repository: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SHA256_PATHS:
        path = repository.joinpath(*PurePosixPath(relative).parts)
        hashes[relative] = _file_sha256(path)
    return hashes


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


__all__ = [
    "IMAGE_SIZE",
    "INTERPRETATION",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "OUTSIDE_SUPPORT_ORIGINAL_WEIGHT",
    "load_embedding_views_manifest",
    "prepare_embedding_views",
    "reconstruct_student_masked_rgb",
    "student_masked_rgb",
    "validate_embedding_views_manifest",
]
