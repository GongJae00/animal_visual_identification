"""Native-resolution YT-BB-Dog nose extraction primitives.

These functions operate on the publisher dog-crop bytes. They do not consume
the resized dog, face, or weak-nose JPEG artifacts from an ROI export.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from parsing.nose_region.localizer import KEYPOINT_ORDER, NOSE_POINT_INDICES
from parsing.nose_region.manifest import frontality_from_keypoints, normalized_box_to_pixel_box
from foundation.provenance import content_sha256


BUNDLE_SCHEMA = "cvi.yt_native_nose_manifest_bundle.v1"
MANIFEST_SCHEMA = "cvi.yt_native_nose_manifest.v1"
TEACHER_SCHEMA = "cvi.yt_native_nose_teacher_masks.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_MAX_IMAGE_BYTES = 67_108_864
_MAX_CONTAINER_BYTES = 4_294_967_296
_MAX_IMAGE_PIXELS = 33_554_432
_MAX_COMPRESSION_RATIO = 200.0
_QUALITY_FIELDS = {
    "blur_laplacian_variance",
    "blur_score",
    "saturation_mean",
    "clipped_pixel_fraction",
    "specular_fraction",
    "contrast_rms",
    "contrast_score",
    "jpeg_blocking_score",
    "noise_score",
    "native_short_side",
    "mask_uncertainty",
    "mask_available",
    "detector_confidence",
    "frontality",
}
_RECORD_FIELDS = {
    "dataset_name",
    "sample_token",
    "identity_token",
    "registered_dog_id",
    "source_sample_id",
    "source_role",
    "sequence_token",
    "track_token",
    "frame_index",
    "source_variant",
    "source_region",
    "source_archive_container_member",
    "source_archive_member",
    "source_sha256",
    "source_width",
    "source_height",
    "roi_metadata_available",
    "source_bytes_role",
    "intermediaries_used",
    "record_state",
    "usage",
    "quality_flags",
    "keypoints",
    "nose_box_xyxy",
    "crop_path",
    "crop_sha256",
    "crop_width",
    "crop_height",
    "soft_mask_path",
    "soft_mask_sha256",
    "binary_mask_path",
    "binary_mask_sha256",
    "mask_method",
    "quality",
}


@dataclass(frozen=True, slots=True)
class NativeYtSample:
    sample_token: str
    identity_token: str
    registered_dog_id: str
    source_sample_id: str
    sequence_token: str
    track_token: str
    frame_index: int
    source_role: str
    member_path: str
    member_crc32: int
    member_uncompressed_bytes: int
    container_member_path: str
    container_member_crc32: int
    container_member_uncompressed_bytes: int
    expected_source_sha256: str | None = None
    roi_metadata_available: bool = False

    def __post_init__(self) -> None:
        for name in ("sample_token", "identity_token", "sequence_token", "track_token"):
            _require_sha256(getattr(self, name), name)
        if self.expected_source_sha256 is not None:
            _require_sha256(self.expected_source_sha256, "expected_source_sha256")
        if self.source_role != "YT_FIT":
            raise ValueError("native YT extraction admits YT_FIT only")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        _safe_member_path(self.member_path)
        _safe_member_path(self.container_member_path)
        for name in (
            "member_crc32",
            "member_uncompressed_bytes",
            "container_member_crc32",
            "container_member_uncompressed_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.roi_metadata_available, bool):
            raise TypeError("roi_metadata_available must be boolean")


class NestedYtArchive:
    """One-pass reader for the original nested YT publisher archive."""

    def __init__(
        self,
        archive_path: Path,
        first_sample: NativeYtSample,
        *,
        expected_archive_sha256: str | None = None,
    ) -> None:
        path = Path(archive_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("YT archive must be an absolute regular non-symlink file")
        self._path = path
        if expected_archive_sha256 is not None:
            _require_sha256(expected_archive_sha256, "expected_archive_sha256")
        self._expected_archive_sha256 = expected_archive_sha256
        self._container_path = first_sample.container_member_path
        self._container_crc32 = first_sample.container_member_crc32
        self._container_size = first_sample.container_member_uncompressed_bytes
        self._stream = None
        self._outer = None
        self._nested_stream = None
        self._nested_file = None
        self._nested = None

    def __enter__(self) -> "NestedYtArchive":
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self._path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("YT archive must be a regular file")
            self._stream = os.fdopen(descriptor, "rb")
            descriptor = -1
            if self._expected_archive_sha256 is not None:
                digest = hashlib.sha256()
                while chunk := self._stream.read(1_048_576):
                    digest.update(chunk)
                if digest.hexdigest() != self._expected_archive_sha256:
                    raise ValueError("YT archive bytes differ from receipt binding")
                self._stream.seek(0)
            self._outer = zipfile.ZipFile(self._stream)
            container = _verified_info(
                self._outer,
                self._container_path,
                self._container_crc32,
                self._container_size,
                maximum_bytes=_MAX_CONTAINER_BYTES,
            )
            self._nested_stream = self._outer.open(container, "r")
            self._nested_file = tempfile.TemporaryFile()
            copied = 0
            while chunk := self._nested_stream.read(1_048_576):
                self._nested_file.write(chunk)
                copied += len(chunk)
            if copied != self._container_size:
                raise ValueError("YT nested archive byte count differs")
            self._nested_file.seek(0)
            self._nested = zipfile.ZipFile(self._nested_file)
            return self
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            self.__exit__(None, None, None)
            raise

    def read(self, sample: NativeYtSample) -> bytes:
        if self._nested is None:
            raise RuntimeError("nested YT archive reader is not open")
        if (
            sample.container_member_path,
            sample.container_member_crc32,
            sample.container_member_uncompressed_bytes,
        ) != (self._container_path, self._container_crc32, self._container_size):
            raise ValueError("YT samples do not share the audited original container")
        member = _verified_info(
            self._nested,
            sample.member_path,
            sample.member_crc32,
            sample.member_uncompressed_bytes,
            maximum_bytes=_MAX_IMAGE_BYTES,
        )
        with self._nested.open(member, "r") as image_stream:
            payload = image_stream.read(_MAX_IMAGE_BYTES + 1)
        if len(payload) != sample.member_uncompressed_bytes or not payload:
            raise ValueError("audited YT source member byte count differs")
        digest = hashlib.sha256(payload).hexdigest()
        if sample.expected_source_sha256 is not None and digest != sample.expected_source_sha256:
            raise ValueError("original YT source member SHA-256 differs from ROI metadata")
        return payload

    def __exit__(self, exc_type, exc, traceback) -> None:
        for resource in (
            self._nested,
            self._nested_file,
            self._nested_stream,
            self._outer,
            self._stream,
        ):
            if resource is not None:
                resource.close()


def read_nested_member_bytes(archive_path: Path, sample: NativeYtSample) -> bytes:
    """Read one audited original YT member from its nested publisher ZIP."""

    with NestedYtArchive(archive_path, sample) as archive:
        return archive.read(sample)


def decode_source_image(payload: bytes) -> Image.Image:
    if not payload or len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError("source image bytes exceed policy")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise ValueError("source image dimensions exceed policy")
            image = opened.convert("RGB")
            image.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("source member is not a decodable image") from exc
    return image


def validate_keypoints(prediction: object) -> list[list[float]] | None:
    """Validate one normalized [8,3] localizer result."""

    if prediction is None:
        return None
    array = np.asarray(prediction, dtype=np.float64)
    if array.shape != (len(KEYPOINT_ORDER), 3) or not np.isfinite(array).all():
        raise ValueError("localizer prediction must be finite [8,3]")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("localizer prediction must be normalized to [0,1]")
    return array.tolist()


def nose_geometry(
    keypoints: Sequence[Sequence[object]], image_width: int, image_height: int,
    *, margin: float = 0.08,
) -> tuple[tuple[int, int, int, int], float, float]:
    parsed = validate_keypoints(keypoints)
    if parsed is None:
        raise ValueError("nose geometry requires keypoints")
    nose = [parsed[index] for index in NOSE_POINT_INDICES]
    confidence = float(np.mean([point[2] for point in nose]))
    if confidence <= 0.0:
        raise ValueError("nose keypoint confidence is unavailable")
    normalized_box = (
        max(0.0, min(point[0] for point in nose) - margin),
        max(0.0, min(point[1] for point in nose) - margin),
        min(1.0, max(point[0] for point in nose) + margin),
        min(1.0, max(point[1] for point in nose) + margin),
    )
    box = normalized_box_to_pixel_box(normalized_box, image_width, image_height)
    return box, confidence, frontality_from_keypoints(parsed)


def mask_from_source_keypoints(
    crop: Image.Image,
    keypoints: Sequence[Sequence[object]],
    source_box: Sequence[int],
    source_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, str]:
    parsed = validate_keypoints(keypoints)
    if parsed is None:
        raise ValueError("automatic masks require keypoints")
    if len(source_box) != 4 or min(source_size) <= 0:
        raise ValueError("source geometry differs")
    left, top, right, bottom = source_box
    if crop.size != (right - left, bottom - top):
        raise ValueError("crop and source_box geometry differ")
    points = np.asarray(
        [
            (parsed[index][0] * source_size[0] - left, parsed[index][1] * source_size[1] - top)
            for index in NOSE_POINT_INDICES
        ],
        dtype=np.float32,
    )
    return _masks_from_crop_points(crop, points)


def teacher_masks(
    teacher: Image.Image,
    *, source_box: Sequence[int],
    source_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, str]:
    """Use an exact caller-supplied source-resolution teacher mask, unchanged."""

    if teacher.mode != "L" or teacher.size != source_size:
        raise ValueError("teacher mask must be an L image aligned to the source image")
    if len(source_box) != 4:
        raise ValueError("source_box must have four coordinates")
    crop = teacher.crop(tuple(source_box))
    values = np.asarray(crop, dtype=np.uint8)
    binary = Image.fromarray(np.where(values >= 128, 255, 0).astype(np.uint8), mode="L")
    return crop, binary, "EXTERNAL_EXACT_TEACHER"


def compute_quality(
    crop: Image.Image,
    soft_mask: Image.Image | None,
    *,
    native_short_side: int,
    detector_confidence: float | None,
    frontality: float | None,
    mask_uncertainty_override: float | None = None,
) -> dict[str, Any]:
    """Compute deterministic, non-reference image and mask quality signals."""

    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = min(1.0, laplacian_variance / 500.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation_mean = float(hsv[..., 1].mean() / 255.0)
    clipped = np.any((rgb <= 2) | (rgb >= 253), axis=2)
    clipped_fraction = float(clipped.mean())
    contrast_rms = float(gray.astype(np.float32).std())
    contrast_score = min(1.0, contrast_rms / 64.0)
    blocking = _jpeg_blocking(gray)
    median = cv2.medianBlur(gray, 3) if min(gray.shape) >= 3 else gray
    noise = min(1.0, float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32)))) / 32.0)
    mask_available = soft_mask is not None
    if soft_mask is None:
        support = np.ones(gray.shape, dtype=bool)
        mask_uncertainty = 1.0
    else:
        probability = np.asarray(soft_mask, dtype=np.float32) / 255.0
        support = probability >= 0.5
        if not np.any(support):
            support = probability > 0.0
        boundary = (probability > 0.02) & (probability < 0.98)
        mask_uncertainty = (
            float(np.mean(1.0 - np.abs(2.0 * probability[boundary] - 1.0)))
            if np.any(boundary)
            else 0.0
        )
    if mask_uncertainty_override is not None:
        if (
            isinstance(mask_uncertainty_override, bool)
            or not isinstance(mask_uncertainty_override, (int, float))
            or not math.isfinite(mask_uncertainty_override)
            or not 0.0 <= float(mask_uncertainty_override) <= 1.0
        ):
            raise ValueError("mask uncertainty override must be finite and in [0,1]")
        mask_uncertainty = float(mask_uncertainty_override)
    bright = (hsv[..., 2] >= 245) & (hsv[..., 1] <= 38)
    specular_fraction = float(bright[support].mean()) if np.any(support) else 0.0
    return {
        "blur_laplacian_variance": laplacian_variance,
        "blur_score": blur_score,
        "saturation_mean": saturation_mean,
        "clipped_pixel_fraction": clipped_fraction,
        "specular_fraction": specular_fraction,
        "contrast_rms": contrast_rms,
        "contrast_score": contrast_score,
        "jpeg_blocking_score": blocking,
        "noise_score": noise,
        "native_short_side": native_short_side,
        "mask_uncertainty": mask_uncertainty,
        "mask_available": mask_available,
        "detector_confidence": detector_confidence,
        "frontality": frontality,
    }


def process_native_sample(
    sample: NativeYtSample,
    source_bytes: bytes,
    prediction: object,
    *,
    policy: Mapping[str, Any],
    teacher_mask: Image.Image | None = None,
    teacher_mask_uncertainty: float | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Process one source row while retaining NO_ROI and low-quality outcomes."""

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if (teacher_mask is None) != (teacher_mask_uncertainty is None):
        raise ValueError("teacher mask and uncertainty must be supplied together")
    if sample.expected_source_sha256 is not None and source_sha256 != sample.expected_source_sha256:
        raise ValueError("original YT source member SHA-256 differs from ROI metadata")
    image = decode_source_image(source_bytes)
    points = validate_keypoints(prediction)
    base = {
        "dataset_name": "yt-bb-dog",
        "sample_token": sample.sample_token,
        "identity_token": sample.identity_token,
        "registered_dog_id": sample.registered_dog_id,
        "source_sample_id": sample.source_sample_id,
        "source_role": sample.source_role,
        "sequence_token": sample.sequence_token,
        "track_token": sample.track_token,
        "frame_index": sample.frame_index,
        "source_variant": "original",
        "source_region": "DOG_CROP",
        "source_archive_container_member": sample.container_member_path,
        "source_archive_member": sample.member_path,
        "source_sha256": source_sha256,
        "source_width": image.width,
        "source_height": image.height,
        "roi_metadata_available": sample.roi_metadata_available,
        "source_bytes_role": "ORIGINAL_PUBLISHER_DOG_CROP",
        "intermediaries_used": [],
    }
    if points is None or float(np.mean(np.asarray(points)[list(NOSE_POINT_INDICES), 2])) <= 0.0:
        quality = compute_quality(
            image, None, native_short_side=0, detector_confidence=None, frontality=None
        )
        return ({
            **base,
            "record_state": "NO_ROI",
            "usage": "QUALITY_SSL_ONLY",
            "quality_flags": ["NO_ROI"],
            "keypoints": None,
            "nose_box_xyxy": None,
            "crop_path": None,
            "crop_sha256": None,
            "crop_width": None,
            "crop_height": None,
            "soft_mask_path": None,
            "soft_mask_sha256": None,
            "binary_mask_path": None,
            "binary_mask_sha256": None,
            "mask_method": "UNAVAILABLE",
            "quality": quality,
        }, {})

    box, confidence, frontality = nose_geometry(points, image.width, image.height)
    crop = image.crop(box)
    if teacher_mask is None:
        soft, binary, mask_method = mask_from_source_keypoints(
            crop, points, box, image.size
        )
    else:
        soft, binary, mask_method = teacher_masks(
            teacher_mask, source_box=box, source_size=image.size
        )
    native_short_side = min(crop.size)
    quality = compute_quality(
        crop,
        soft,
        native_short_side=native_short_side,
        detector_confidence=confidence,
        frontality=frontality,
        mask_uncertainty_override=teacher_mask_uncertainty,
    )
    flags: list[str] = []
    if confidence < float(policy["minimum_detector_confidence"]):
        flags.append("LOW_DETECTOR_CONFIDENCE")
    if frontality < float(policy["minimum_frontality"]):
        flags.append("LOW_FRONTALITY")
    if native_short_side < int(policy["minimum_native_short_side"]):
        flags.append("LOW_NATIVE_RESOLUTION")
    if quality["mask_uncertainty"] > float(policy["maximum_mask_uncertainty"]):
        flags.append("HIGH_MASK_UNCERTAINTY")
    artifacts = {
        f"crops/{sample.sample_token}.png": _png_bytes(crop.convert("RGB")),
        f"soft_masks/{sample.sample_token}.png": _png_bytes(soft),
        f"binary_masks/{sample.sample_token}.png": _png_bytes(binary),
    }
    keypoint_rows = [
        {
            "name": name,
            "normalized_x": point[0],
            "normalized_y": point[1],
            "source_x": point[0] * image.width,
            "source_y": point[1] * image.height,
            "confidence": point[2],
        }
        for name, point in zip(KEYPOINT_ORDER, points, strict=True)
    ]
    return ({
        **base,
        "record_state": "LOW_QUALITY" if flags else "AVAILABLE",
        "usage": "QUALITY_SSL_ONLY" if flags else "IDENTITY_TRAINING",
        "quality_flags": flags,
        "keypoints": keypoint_rows,
        "nose_box_xyxy": list(box),
        "crop_path": f"crops/{sample.sample_token}.png",
        "crop_sha256": hashlib.sha256(artifacts[f"crops/{sample.sample_token}.png"]).hexdigest(),
        "crop_width": crop.width,
        "crop_height": crop.height,
        "soft_mask_path": f"soft_masks/{sample.sample_token}.png",
        "soft_mask_sha256": hashlib.sha256(artifacts[f"soft_masks/{sample.sample_token}.png"]).hexdigest(),
        "binary_mask_path": f"binary_masks/{sample.sample_token}.png",
        "binary_mask_sha256": hashlib.sha256(artifacts[f"binary_masks/{sample.sample_token}.png"]).hexdigest(),
        "mask_method": mask_method,
        "quality": quality,
    }, artifacts)


def load_localizer_checkpoint(checkpoint_bytes: bytes, device_name: str):
    """Load the exact trained localizer checkpoint contract used by preparation."""

    import torch
    import timm

    from parsing.nose_region.localizer import (
        INPUT_SIZE,
        MOBILENETV4_MODEL_NAME,
        MobileNetV4NoseLocalizer,
        mobilenetv4_feature_dim,
    )

    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema_version", "bindings", "selected_epoch", "model_state_dict"
    }:
        raise ValueError("nose localizer checkpoint schema differs")
    if checkpoint["schema_version"] != "cvi.nose_localizer.checkpoint.v1":
        raise ValueError("unsupported nose localizer checkpoint")
    if isinstance(checkpoint["selected_epoch"], bool) or not isinstance(checkpoint["selected_epoch"], int) or checkpoint["selected_epoch"] <= 0:
        raise ValueError("nose localizer selected epoch differs")
    bindings = checkpoint["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "schema_version", "sources", "split_counts", "training_config", "license", "content_sha256"
    }:
        raise ValueError("nose localizer checkpoint bindings schema differs")
    canonical = hashlib.sha256(
        json.dumps(
            {key: value for key, value in bindings.items() if key != "content_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    if bindings["schema_version"] != "cvi.nose_localizer.bindings.v1" or bindings["content_sha256"] != canonical:
        raise ValueError("nose localizer checkpoint bindings digest differs")
    license_payload = bindings["license"]
    if not isinstance(license_payload, dict) or set(license_payload) != {"license_id", "usage_lane", "reason"} or license_payload["license_id"] != "CC-BY-NC-4.0-derived" or license_payload["usage_lane"] != "RESEARCH_ONLY" or not isinstance(license_payload["reason"], str) or not license_payload["reason"]:
        raise ValueError("nose localizer checkpoint license lane differs")
    training = bindings["training_config"]
    expected_training = {
        "model_name", "input_size", "keypoint_order", "epochs", "batch_size",
        "learning_rate", "weight_decay", "seed", "publisher_split_policy",
    }
    if not isinstance(training, dict) or set(training) != expected_training or training["model_name"] != MOBILENETV4_MODEL_NAME or training["input_size"] != INPUT_SIZE or training["keypoint_order"] != list(KEYPOINT_ORDER):
        raise ValueError("nose localizer checkpoint training contract differs")
    split_counts = bindings["split_counts"]
    expected_splits = {"ap10k": {"train", "val", "test"}, "dogflw": {"train", "test"}}
    if not isinstance(split_counts, dict) or set(split_counts) != set(expected_splits):
        raise ValueError("nose localizer checkpoint split counts differ")
    for dataset, splits in expected_splits.items():
        values = split_counts[dataset]
        if not isinstance(values, dict) or set(values) != splits or any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in values.values()):
            raise ValueError("nose localizer checkpoint split counts differ")
    sources = bindings["sources"]
    if not isinstance(sources, dict) or set(sources) != {
        "ap10k_zip_sha256", "dogflw_zip_sha256", "backbone_safetensors_sha256"
    }:
        raise ValueError("nose localizer checkpoint source bindings differ")
    for name, digest in sources.items():
        _require_sha256(digest, f"localizer source {name}")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("nose localizer checkpoint state is empty")
    backbone = timm.create_model(MOBILENETV4_MODEL_NAME, pretrained=False)
    model = MobileNetV4NoseLocalizer(backbone, mobilenetv4_feature_dim(backbone))
    model.load_state_dict(state, strict=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    return model.to(device).eval(), device, bindings


def predict_localizer(model: Any, device: Any, image: Image.Image) -> list[list[float]]:
    import torch

    from parsing.nose_region.localizer import INPUT_SIZE, image_to_tensor

    resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    tensor = image_to_tensor(resized).unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction = model(tensor)[0].detach().cpu().numpy()
    parsed = validate_keypoints(prediction)
    if parsed is None:
        raise RuntimeError("localizer unexpectedly returned no prediction")
    return parsed


def build_manifest_bundle(
    *,
    records: Sequence[Mapping[str, Any]],
    input_sha256s: Mapping[str, str],
    policy: Mapping[str, Any],
    tool_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("native YT manifest must retain at least one record")
    hashes = dict(sorted(input_sha256s.items()))
    for name, digest in hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("input hash name must be non-empty")
        _require_sha256(digest, f"input_sha256s.{name}")
    normalized = [dict(record) for record in records]
    expected = sorted(normalized, key=lambda row: row["sample_token"])
    if normalized != expected or len({row["sample_token"] for row in normalized}) != len(normalized):
        raise ValueError("native YT records must be uniquely sorted by sample token")
    counts = {
        state: sum(row["record_state"] == state for row in normalized)
        for state in ("AVAILABLE", "LOW_QUALITY", "NO_ROI")
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_name": "yt-bb-dog",
        "source_variant": "original",
        "source_bytes_role": "ORIGINAL_PUBLISHER_DOG_CROP",
        "intermediaries_prohibited": ["ROI_DOG_CROP_JPEG", "ROI_FACE_CROP_JPEG", "ROI_WEAK_NOSE_CROP_JPEG"],
        "input_sha256s": hashes,
        "policy": dict(policy),
        "records": normalized,
        "record_counts": counts,
        "tool_provenance": dict(tool_provenance),
        "interpretation": "LOSSLESS_DERIVED_NOSE_ARTIFACTS_NOT_RAW_DATA_OR_BIOMETRIC_VALIDATION",
    }
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(manifest),
        "manifest": manifest,
    }
    validate_manifest_bundle(bundle)
    return bundle


def validate_manifest_bundle(
    bundle: object, *, root: Path | None = None
) -> dict[str, Any]:
    """Validate manifest schemas, hashes, outcomes, and optional artifacts."""

    if not isinstance(bundle, dict) or set(bundle) != {"schema_version", "manifest_sha256", "manifest"} or bundle["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("native YT manifest bundle schema differs")
    _require_sha256(bundle["manifest_sha256"], "manifest_sha256")
    manifest = bundle["manifest"]
    manifest_fields = {
        "schema_version",
        "dataset_name",
        "source_variant",
        "source_bytes_role",
        "intermediaries_prohibited",
        "input_sha256s",
        "policy",
        "records",
        "record_counts",
        "tool_provenance",
        "interpretation",
    }
    if not isinstance(manifest, dict) or set(manifest) != manifest_fields or manifest["schema_version"] != MANIFEST_SCHEMA or content_sha256(manifest) != bundle["manifest_sha256"]:
        raise ValueError("native YT manifest schema or content digest differs")
    if (
        manifest["dataset_name"] != "yt-bb-dog"
        or manifest["source_variant"] != "original"
        or manifest["source_bytes_role"] != "ORIGINAL_PUBLISHER_DOG_CROP"
        or manifest["interpretation"] != "LOSSLESS_DERIVED_NOSE_ARTIFACTS_NOT_RAW_DATA_OR_BIOMETRIC_VALIDATION"
    ):
        raise ValueError("native YT manifest source contract differs")
    if manifest["intermediaries_prohibited"] != [
        "ROI_DOG_CROP_JPEG",
        "ROI_FACE_CROP_JPEG",
        "ROI_WEAK_NOSE_CROP_JPEG",
    ]:
        raise ValueError("native YT prohibited intermediary contract differs")
    if not isinstance(manifest["input_sha256s"], dict) or not manifest["input_sha256s"]:
        raise ValueError("native YT input hashes differ")
    for name, digest in manifest["input_sha256s"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("native YT input hash name differs")
        _require_sha256(digest, f"input_sha256s.{name}")
    policy = manifest["policy"]
    required_policy = {
        "minimum_detector_confidence",
        "minimum_frontality",
        "minimum_native_short_side",
        "maximum_mask_uncertainty",
    }
    if not isinstance(policy, dict) or not required_policy <= set(policy):
        raise ValueError("native YT quality policy differs")
    for name in (
        "minimum_detector_confidence",
        "minimum_frontality",
        "maximum_mask_uncertainty",
    ):
        value = policy[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"native YT policy {name} differs")
    native_minimum = policy["minimum_native_short_side"]
    if (
        isinstance(native_minimum, bool)
        or not isinstance(native_minimum, int)
        or native_minimum <= 0
    ):
        raise ValueError("native YT policy minimum_native_short_side differs")
    records = manifest["records"]
    if not isinstance(records, list) or not records or records != sorted(records, key=lambda row: row["sample_token"]):
        raise ValueError("native YT records must be non-empty and sorted")
    resolved_root = root.resolve(strict=True) if root is not None else None
    if resolved_root is not None and not resolved_root.is_dir():
        raise ValueError("native YT manifest root must be a directory")
    seen: set[str] = set()
    counts = {state: 0 for state in ("AVAILABLE", "LOW_QUALITY", "NO_ROI")}
    for record in records:
        _validate_record(record, resolved_root, policy)
        token = record["sample_token"]
        if token in seen:
            raise ValueError("native YT manifest repeats a sample token")
        seen.add(token)
        counts[record["record_state"]] += 1
    if manifest["record_counts"] != counts:
        raise ValueError("native YT record counts differ")
    if not isinstance(manifest["tool_provenance"], dict):
        raise ValueError("native YT tool provenance differs")
    return manifest


def _validate_record(
    record: object, root: Path | None, policy: Mapping[str, Any]
) -> None:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise ValueError("native YT record schema differs")
    for field in ("sample_token", "identity_token", "sequence_token", "track_token", "source_sha256"):
        _require_sha256(record[field], field)
    try:
        registered = uuid.UUID(record["registered_dog_id"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("registered_dog_id must be a canonical UUIDv5") from exc
    if registered.version != 5 or str(registered) != record["registered_dog_id"]:
        raise ValueError("registered_dog_id must be a canonical UUIDv5")
    if record["dataset_name"] != "yt-bb-dog" or record["source_role"] != "YT_FIT" or record["source_variant"] != "original" or record["source_region"] != "DOG_CROP":
        raise ValueError("native YT record identity/source role differs")
    if record["source_bytes_role"] != "ORIGINAL_PUBLISHER_DOG_CROP" or record["intermediaries_used"] != []:
        raise ValueError("native YT record source bytes contract differs")
    _safe_member_path(record["source_archive_container_member"])
    _safe_member_path(record["source_archive_member"])
    for field in ("frame_index", "source_width", "source_height"):
        value = record[field]
        minimum = 0 if field == "frame_index" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} differs")
    if not isinstance(record["roi_metadata_available"], bool):
        raise ValueError("roi_metadata_available differs")
    state = record["record_state"]
    if state not in {"AVAILABLE", "LOW_QUALITY", "NO_ROI"}:
        raise ValueError("native YT record state differs")
    if record["usage"] != ("IDENTITY_TRAINING" if state == "AVAILABLE" else "QUALITY_SSL_ONLY"):
        raise ValueError("native YT record usage differs")
    if not isinstance(record["quality_flags"], list) or any(not isinstance(flag, str) or not flag for flag in record["quality_flags"]):
        raise ValueError("native YT quality flags differ")
    quality = record["quality"]
    if not isinstance(quality, dict) or set(quality) != _QUALITY_FIELDS or not isinstance(quality["mask_available"], bool):
        raise ValueError("native YT quality schema differs")
    for name, value in quality.items():
        if name == "mask_available" or value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"quality.{name} differs")
        if name not in {
            "blur_laplacian_variance",
            "contrast_rms",
            "native_short_side",
        } and value > 1.0:
            raise ValueError(f"quality.{name} differs")
    artifact_fields = (
        "crop_path", "crop_sha256", "crop_width", "crop_height",
        "soft_mask_path", "soft_mask_sha256", "binary_mask_path", "binary_mask_sha256",
    )
    if state == "NO_ROI":
        if record["quality_flags"] != ["NO_ROI"] or record["keypoints"] is not None or record["nose_box_xyxy"] is not None or record["mask_method"] != "UNAVAILABLE" or any(record[field] is not None for field in artifact_fields) or quality["mask_available"] is not False:
            raise ValueError("NO_ROI native YT record artifacts differ")
        return
    if not record["quality_flags"] and state != "AVAILABLE" or record["quality_flags"] and state != "LOW_QUALITY":
        raise ValueError("native YT quality state differs")
    if quality["detector_confidence"] is None or quality["frontality"] is None:
        raise ValueError("native YT localized quality is incomplete")
    expected_flags: list[str] = []
    if quality["detector_confidence"] < policy["minimum_detector_confidence"]:
        expected_flags.append("LOW_DETECTOR_CONFIDENCE")
    if quality["frontality"] < policy["minimum_frontality"]:
        expected_flags.append("LOW_FRONTALITY")
    if quality["native_short_side"] < policy["minimum_native_short_side"]:
        expected_flags.append("LOW_NATIVE_RESOLUTION")
    if quality["mask_uncertainty"] > policy["maximum_mask_uncertainty"]:
        expected_flags.append("HIGH_MASK_UNCERTAINTY")
    if record["quality_flags"] != expected_flags:
        raise ValueError("native YT quality flags differ from policy")
    if not isinstance(record["keypoints"], list) or len(record["keypoints"]) != len(KEYPOINT_ORDER):
        raise ValueError("native YT keypoint schema differs")
    keypoint_fields = {"name", "normalized_x", "normalized_y", "source_x", "source_y", "confidence"}
    for expected_name, point in zip(KEYPOINT_ORDER, record["keypoints"], strict=True):
        if not isinstance(point, dict) or set(point) != keypoint_fields or point["name"] != expected_name:
            raise ValueError("native YT keypoint schema differs")
        for field in keypoint_fields - {"name"}:
            value = point[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("native YT keypoint value differs")
        if not 0.0 <= point["normalized_x"] <= 1.0 or not 0.0 <= point["normalized_y"] <= 1.0 or not 0.0 <= point["confidence"] <= 1.0:
            raise ValueError("native YT normalized keypoint differs")
    box = record["nose_box_xyxy"]
    if not isinstance(box, list) or len(box) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise ValueError("native YT nose box differs")
    if not (0 <= box[0] < box[2] <= record["source_width"] and 0 <= box[1] < box[3] <= record["source_height"]):
        raise ValueError("native YT nose box differs")
    if (record["crop_width"], record["crop_height"]) != (box[2] - box[0], box[3] - box[1]):
        raise ValueError("native YT crop dimensions differ")
    if record["mask_method"] not in {"KEYPOINT_GEOMETRY", "KEYPOINT_GEOMETRY_GRABCUT", "EXTERNAL_EXACT_TEACHER"} or quality["mask_available"] is not True:
        raise ValueError("native YT mask contract differs")
    for prefix, mode in (("crop", "RGB"), ("soft_mask", "L"), ("binary_mask", "L")):
        relative = _artifact_path(record[f"{prefix}_path"], record["sample_token"], prefix)
        _require_sha256(record[f"{prefix}_sha256"], f"{prefix}_sha256")
        if root is not None:
            target = root.joinpath(*relative.parts)
            if target.is_symlink():
                raise ValueError("native YT artifact path is unsafe")
            try:
                resolved = target.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError("native YT artifact path is unsafe") from exc
            if not resolved.is_relative_to(root) or resolved.relative_to(root).as_posix() != relative.as_posix() or not resolved.is_file():
                raise ValueError("native YT artifact path is unsafe")
            payload = resolved.read_bytes()
            if hashlib.sha256(payload).hexdigest() != record[f"{prefix}_sha256"]:
                raise ValueError("native YT artifact SHA-256 differs")
            with Image.open(io.BytesIO(payload)) as opened:
                if opened.format != "PNG" or opened.mode != mode or opened.size != (record["crop_width"], record["crop_height"]):
                    raise ValueError("native YT artifact image contract differs")
                opened.load()


def _artifact_path(value: object, token: str, prefix: str) -> PurePosixPath:
    directories = {"crop": "crops", "soft_mask": "soft_masks", "binary_mask": "binary_masks"}
    if not isinstance(value, str):
        raise ValueError("native YT artifact path differs")
    path = PurePosixPath(value)
    if value != path.as_posix() or path.parts != (directories[prefix], f"{token}.png"):
        raise ValueError("native YT artifact path differs")
    return path


def _masks_from_crop_points(
    crop: Image.Image, points: np.ndarray
) -> tuple[Image.Image, Image.Image, str]:
    height, width = crop.height, crop.width
    if points.shape != (len(NOSE_POINT_INDICES), 2) or not np.isfinite(points).all():
        raise ValueError("nose mask points differ")
    center = points.mean(axis=0)
    span_x = max(float(np.ptp(points[:, 0])), width * 0.18, 2.0)
    span_y = max(float(np.ptp(points[:, 1])), height * 0.18, 2.0)
    geometry = np.zeros((height, width), dtype=np.uint8)
    axes = (
        max(1, int(math.ceil(span_x * 0.62))),
        max(1, int(math.ceil(span_y * 0.72))),
    )
    cv2.ellipse(
        geometry,
        (int(round(center[0])), int(round(center[1]))),
        axes,
        0.0,
        0.0,
        360.0,
        255,
        thickness=-1,
    )
    clipped = np.rint(points).astype(np.int32)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(width - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(height - 1, 0))
    hull = cv2.convexHull(clipped)
    cv2.fillConvexPoly(geometry, hull, 255)
    sigma = max(0.8, min(width, height) / 32.0)
    soft_geometry = cv2.GaussianBlur(
        geometry, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
    ).astype(np.float32) / 255.0
    grab_foreground: np.ndarray | None = None
    if min(width, height) >= 8 and np.any(geometry == 0) and np.any(geometry == 255):
        labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
        labels[soft_geometry > 0.08] = cv2.GC_PR_FGD
        labels[soft_geometry > 0.82] = cv2.GC_FGD
        labels[soft_geometry < 0.01] = cv2.GC_BGD
        try:
            cv2.setRNGSeed(0)
            cv2.grabCut(
                cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR),
                labels,
                None,
                np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64),
                3,
                cv2.GC_INIT_WITH_MASK,
            )
            grab_foreground = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD))
        except cv2.error:
            grab_foreground = None
    if grab_foreground is None:
        soft = soft_geometry
        binary_values = soft_geometry >= 0.68
        method = "KEYPOINT_GEOMETRY"
    else:
        soft = soft_geometry * np.where(grab_foreground, 1.0, 0.28)
        binary_values = (soft_geometry >= 0.58) & grab_foreground
        method = "KEYPOINT_GEOMETRY_GRABCUT"
    if not np.any(binary_values):
        nearest = np.unravel_index(int(np.argmax(soft)), soft.shape)
        binary_values[nearest] = True
    soft_image = Image.fromarray(np.rint(np.clip(soft, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    binary_image = Image.fromarray(np.where(binary_values, 255, 0).astype(np.uint8), mode="L")
    return soft_image, binary_image, method


def _jpeg_blocking(gray: np.ndarray) -> float:
    values = gray.astype(np.float32)
    boundary_parts: list[np.ndarray] = []
    interior_parts: list[np.ndarray] = []
    if values.shape[1] > 8:
        differences = np.abs(np.diff(values, axis=1))
        indexes = np.arange(1, values.shape[1])
        boundary_parts.append(differences[:, indexes % 8 == 0])
        interior_parts.append(differences[:, indexes % 8 != 0])
    if values.shape[0] > 8:
        differences = np.abs(np.diff(values, axis=0))
        indexes = np.arange(1, values.shape[0])
        boundary_parts.append(differences[indexes % 8 == 0, :])
        interior_parts.append(differences[indexes % 8 != 0, :])
    boundary = [part.ravel() for part in boundary_parts if part.size]
    interior = [part.ravel() for part in interior_parts if part.size]
    if not boundary or not interior:
        return 0.0
    excess = float(np.mean(np.concatenate(boundary)) - np.mean(np.concatenate(interior)))
    return min(1.0, max(0.0, excess / 32.0))


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _safe_member_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("ZIP member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ZIP member path is unsafe")
    return path


def _verified_info(
    archive: zipfile.ZipFile,
    member_path: str,
    expected_crc32: int,
    expected_size: int,
    *,
    maximum_bytes: int,
) -> zipfile.ZipInfo:
    _safe_member_path(member_path)
    try:
        info = archive.getinfo(member_path)
    except KeyError as exc:
        raise ValueError("audited ZIP member is absent") from exc
    mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 0x1:
        raise ValueError("audited ZIP member type is unsafe")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise ValueError("audited ZIP compression is unsupported")
    if info.CRC != expected_crc32 or info.file_size != expected_size:
        raise ValueError("audited ZIP member metadata differs")
    if info.file_size > maximum_bytes or info.compress_size > maximum_bytes:
        raise ValueError("audited ZIP member exceeds byte limit")
    if info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO:
        raise ValueError("audited ZIP member exceeds compression-ratio limit")
    return info


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


__all__ = [
    "BUNDLE_SCHEMA",
    "MANIFEST_SCHEMA",
    "NestedYtArchive",
    "NativeYtSample",
    "TEACHER_SCHEMA",
    "build_manifest_bundle",
    "compute_quality",
    "decode_source_image",
    "load_localizer_checkpoint",
    "mask_from_source_keypoints",
    "nose_geometry",
    "process_native_sample",
    "predict_localizer",
    "read_nested_member_bytes",
    "teacher_masks",
    "validate_manifest_bundle",
    "validate_keypoints",
]
