"""Deterministic token-keyed oracle crop sanitization with FFmpeg."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cvi.acquisition import sha256_file
from cvi.contracts import Modality
from cvi.pairing import PairConstructionResult
from cvi.provenance import content_sha256
from cvi.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    PairArtifactVerification,
    verify_pair_artifact_files,
)


@dataclass(frozen=True, slots=True)
class CropBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("crop coordinates and dimensions must be integers")
        if self.x < 0 or self.y < 0:
            raise ValueError("crop origin must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("crop dimensions must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CropBox:
        _require_exact_keys(
            payload,
            {"x", "y", "width", "height"},
            "crop box",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class OracleCropSource:
    sample_id: str
    source_path: str
    source_sha256: str
    modality: Modality
    crop: CropBox

    def __post_init__(self) -> None:
        _require_nonempty(self.sample_id, "sample_id")
        _require_nonempty(self.source_path, "source_path")
        _validate_sha256(self.source_sha256, "source_sha256")
        if self.modality not in (Modality.RGB, Modality.IR):
            raise ValueError("oracle crop source modality must be RGB or IR")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "modality": self.modality.value,
            "crop": self.crop.to_dict(),
        }

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_sha256": self.source_sha256,
            "modality": self.modality.value,
            "crop": self.crop.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OracleCropSource:
        _require_exact_keys(
            payload,
            {
                "sample_id",
                "source_path",
                "source_sha256",
                "modality",
                "crop",
            },
            "oracle crop source",
        )
        return cls(
            sample_id=payload["sample_id"],
            source_path=payload["source_path"],
            source_sha256=payload["source_sha256"],
            modality=Modality(payload["modality"]),
            crop=CropBox.from_dict(payload["crop"]),
        )


@dataclass(frozen=True, slots=True)
class CropExportPolicy:
    timeout_seconds_per_artifact: float = 30.0
    maximum_artifacts: int = 10_000
    maximum_source_file_bytes: int = 536_870_912
    maximum_source_pixels: int = 33_554_432
    maximum_crop_pixels: int = 16_777_216
    maximum_artifact_bytes: int = 67_108_864
    maximum_total_output_bytes: int = 8_589_934_592
    rgb_pixel_format: str = "rgb24"
    ir_pixel_format: str = "gray"
    output_media_type: str = "image/png"

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds_per_artifact, bool)
            or not isinstance(
                self.timeout_seconds_per_artifact, (int, float)
            )
            or not isfinite(self.timeout_seconds_per_artifact)
            or self.timeout_seconds_per_artifact <= 0
        ):
            raise ValueError("crop export timeout must be finite and positive")
        for name in (
            "maximum_artifacts",
            "maximum_source_file_bytes",
            "maximum_source_pixels",
            "maximum_crop_pixels",
            "maximum_artifact_bytes",
            "maximum_total_output_bytes",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.rgb_pixel_format != "rgb24":
            raise ValueError("RGB export pixel format is fixed to rgb24")
        if self.ir_pixel_format != "gray":
            raise ValueError("IR export pixel format is fixed to gray")
        if self.output_media_type != "image/png":
            raise ValueError("oracle crop export format is fixed to image/png")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | float | int]:
        return {
            "schema_version": "cvi.crop_export_policy.v1",
            "timeout_seconds_per_artifact": (
                self.timeout_seconds_per_artifact
            ),
            "maximum_artifacts": self.maximum_artifacts,
            "maximum_source_file_bytes": self.maximum_source_file_bytes,
            "maximum_source_pixels": self.maximum_source_pixels,
            "maximum_crop_pixels": self.maximum_crop_pixels,
            "maximum_artifact_bytes": self.maximum_artifact_bytes,
            "maximum_total_output_bytes": self.maximum_total_output_bytes,
            "rgb_pixel_format": self.rgb_pixel_format,
            "ir_pixel_format": self.ir_pixel_format,
            "output_media_type": self.output_media_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CropExportPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "timeout_seconds_per_artifact",
                "maximum_artifacts",
                "maximum_source_file_bytes",
                "maximum_source_pixels",
                "maximum_crop_pixels",
                "maximum_artifact_bytes",
                "maximum_total_output_bytes",
                "rgb_pixel_format",
                "ir_pixel_format",
                "output_media_type",
            },
            "crop export policy",
        )
        if payload["schema_version"] != "cvi.crop_export_policy.v1":
            raise ValueError("unsupported crop export policy schema")
        return cls(
            timeout_seconds_per_artifact=payload[
                "timeout_seconds_per_artifact"
            ],
            maximum_artifacts=payload["maximum_artifacts"],
            maximum_source_file_bytes=payload[
                "maximum_source_file_bytes"
            ],
            maximum_source_pixels=payload["maximum_source_pixels"],
            maximum_crop_pixels=payload["maximum_crop_pixels"],
            maximum_artifact_bytes=payload["maximum_artifact_bytes"],
            maximum_total_output_bytes=payload[
                "maximum_total_output_bytes"
            ],
            rgb_pixel_format=payload["rgb_pixel_format"],
            ir_pixel_format=payload["ir_pixel_format"],
            output_media_type=payload["output_media_type"],
        )


@dataclass(frozen=True, slots=True)
class ImageProbe:
    width: int
    height: int
    pixel_format: str
    format_name: str
    stream_tags: tuple[tuple[str, str], ...]
    format_tags: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CropExportReceipt:
    pair_set_sha256: str
    source_manifest_sha256: str
    export_policy_sha256: str
    ffmpeg_version: str
    artifact_manifest: PairArtifactManifest
    verification: PairArtifactVerification

    def __post_init__(self) -> None:
        _validate_sha256(self.pair_set_sha256, "pair_set_sha256")
        _validate_sha256(
            self.source_manifest_sha256,
            "source_manifest_sha256",
        )
        _validate_sha256(
            self.export_policy_sha256,
            "export_policy_sha256",
        )
        _require_nonempty(self.ffmpeg_version, "ffmpeg_version")
        if self.artifact_manifest.pair_set_sha256 != self.pair_set_sha256:
            raise ValueError("crop receipt artifact pair-set mismatch")
        if (
            self.verification.artifact_manifest_sha256
            != self.artifact_manifest.manifest_sha256
        ):
            raise ValueError("crop receipt artifact verification mismatch")
        if self.verification.verified_files != len(
            self.artifact_manifest.entries
        ):
            raise ValueError("crop receipt verified file count mismatch")
        if self.verification.verified_bytes != sum(
            entry.byte_size for entry in self.artifact_manifest.entries
        ):
            raise ValueError("crop receipt verified byte count mismatch")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cvi.crop_export_receipt.v1",
            "pair_set_sha256": self.pair_set_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "export_policy_sha256": self.export_policy_sha256,
            "ffmpeg_version": self.ffmpeg_version,
            "artifact_manifest": self.artifact_manifest.to_dict(),
            "verification": self.verification.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CropExportReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "pair_set_sha256",
                "source_manifest_sha256",
                "export_policy_sha256",
                "ffmpeg_version",
                "artifact_manifest",
                "verification",
            },
            "crop export receipt",
        )
        if payload["schema_version"] != "cvi.crop_export_receipt.v1":
            raise ValueError("unsupported crop export receipt schema")
        return cls(
            pair_set_sha256=payload["pair_set_sha256"],
            source_manifest_sha256=payload["source_manifest_sha256"],
            export_policy_sha256=payload["export_policy_sha256"],
            ffmpeg_version=payload["ffmpeg_version"],
            artifact_manifest=PairArtifactManifest.from_dict(
                payload["artifact_manifest"]
            ),
            verification=PairArtifactVerification.from_dict(
                payload["verification"]
            ),
        )


def oracle_crop_sources_from_payload(
    payload: dict[str, Any],
) -> tuple[OracleCropSource, ...]:
    _require_exact_keys(
        payload,
        {"schema_version", "sources"},
        "oracle crop source manifest",
    )
    if payload["schema_version"] != "cvi.oracle_crop_sources.v1":
        raise ValueError("unsupported oracle crop source schema")
    sources = payload["sources"]
    if not isinstance(sources, list):
        raise TypeError("oracle crop sources must be a list")
    return tuple(OracleCropSource.from_dict(item) for item in sources)


def probe_still_image(path: Path) -> ImageProbe:
    completed = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,pix_fmt:stream_tags:"
                "format=format_name:format_tags"
            ),
            "-of",
            "json",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("image must contain exactly one video stream")
    stream = streams[0]
    format_payload = payload.get("format", {})
    width = stream.get("width")
    height = stream.get("height")
    pixel_format = stream.get("pix_fmt")
    format_name = format_payload.get("format_name")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or not isinstance(pixel_format, str)
        or not isinstance(format_name, str)
    ):
        raise ValueError("image probe is incomplete")
    return ImageProbe(
        width=width,
        height=height,
        pixel_format=pixel_format,
        format_name=format_name,
        stream_tags=_normalized_tags(stream.get("tags")),
        format_tags=_normalized_tags(format_payload.get("tags")),
    )


def build_crop_command(
    source: Path,
    destination: Path,
    *,
    crop: CropBox,
    pixel_format: str,
) -> tuple[str, ...]:
    crop_filter = (
        f"crop=w={crop.width}:h={crop.height}:"
        f"x={crop.x}:y={crop.y}:exact=1,setsar=1"
    )
    return (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-vf",
        crop_filter,
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        pixel_format,
        "-f",
        "image2",
        str(destination),
    )


def export_oracle_crops(
    construction: PairConstructionResult,
    *,
    sources: tuple[OracleCropSource, ...],
    policy: CropExportPolicy,
    output_directory: Path,
) -> CropExportReceipt:
    if output_directory.is_symlink():
        raise ValueError("crop output directory must not be a symlink")
    output_root = output_directory.resolve(strict=True)
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    if any(output_root.iterdir()):
        raise ValueError("crop output directory must be empty")
    source_by_sample = _source_map(sources)
    if len(source_by_sample) > policy.maximum_artifacts:
        raise ValueError("crop source count exceeds maximum_artifacts")
    _validate_bindings(construction)
    binding_samples = {
        binding.sample_id for binding in construction.artifact_bindings
    }
    source_samples = set(source_by_sample)
    if binding_samples != source_samples:
        raise ValueError(
            "crop source IDs mismatch; "
            f"missing={sorted(binding_samples - source_samples)}, "
            f"extra={sorted(source_samples - binding_samples)}"
        )
    ffmpeg_version = subprocess.run(
        ("ffmpeg", "-version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout.splitlines()[0]
    entries: list[PairArtifactEntry] = []
    total_output_bytes = 0
    with TemporaryDirectory(
        prefix=".cvi-crop-export-",
        dir=output_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        for binding in construction.artifact_bindings:
            source = source_by_sample[binding.sample_id]
            unresolved_source = Path(source.source_path)
            if unresolved_source.is_symlink():
                raise ValueError("oracle crop source must not be a symlink")
            source_path = unresolved_source.resolve(strict=True)
            if not source_path.is_file():
                raise ValueError("oracle crop source must be a regular file")
            initial = source_path.stat()
            if initial.st_size > policy.maximum_source_file_bytes:
                raise ValueError(
                    "oracle crop source exceeds maximum_source_file_bytes"
                )
            if sha256_file(source_path) != source.source_sha256:
                raise ValueError(
                    f"oracle crop source hash mismatch: {source.sample_id}"
                )
            source_probe = probe_still_image(source_path)
            _validate_source_image(source_probe, policy)
            _validate_crop_inside(source.crop, source_probe)
            if (
                source.crop.width * source.crop.height
                > policy.maximum_crop_pixels
            ):
                raise ValueError("crop exceeds maximum_crop_pixels")
            destination = temporary_root / f"{binding.artifact_token}.png"
            pixel_format = (
                policy.rgb_pixel_format
                if source.modality is Modality.RGB
                else policy.ir_pixel_format
            )
            subprocess.run(
                build_crop_command(
                    source_path,
                    destination,
                    crop=source.crop,
                    pixel_format=pixel_format,
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=policy.timeout_seconds_per_artifact,
            )
            os.chmod(destination, 0o600)
            output_probe = probe_still_image(destination)
            _validate_exported_image(
                output_probe,
                source.crop,
                pixel_format,
            )
            artifact_size = destination.stat().st_size
            if artifact_size > policy.maximum_artifact_bytes:
                raise ValueError("crop artifact exceeds maximum_artifact_bytes")
            total_output_bytes += artifact_size
            if total_output_bytes > policy.maximum_total_output_bytes:
                raise ValueError(
                    "crop export exceeds maximum_total_output_bytes"
                )
            final = source_path.stat()
            if (
                initial.st_size != final.st_size
                or initial.st_mtime_ns != final.st_mtime_ns
                or sha256_file(source_path) != source.source_sha256
            ):
                raise RuntimeError(
                    f"oracle crop source changed during export: {source.sample_id}"
                )
            entries.append(
                PairArtifactEntry(
                    artifact_token=binding.artifact_token,
                    relative_path=destination.name,
                    content_sha256=sha256_file(destination),
                    byte_size=artifact_size,
                    media_type=policy.output_media_type,
                )
            )
        artifact_manifest = PairArtifactManifest(
            pair_set_sha256=construction.result_sha256,
            artifact_bindings_sha256=content_sha256(
                construction.artifact_binding_payload()
            ),
            entries=tuple(entries),
        )
        verify_pair_artifact_files(temporary_root, artifact_manifest)
        created: list[Path] = []
        try:
            for entry in entries:
                source_path = temporary_root / entry.relative_path
                destination = output_root / entry.relative_path
                os.link(source_path, destination)
                created.append(destination)
            verification = verify_pair_artifact_files(
                output_root,
                artifact_manifest,
            )
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    source_manifest_sha256 = content_sha256(
        [
            source.provenance_dict()
            for source in sorted(sources, key=lambda item: item.sample_id)
        ]
    )
    return CropExportReceipt(
        pair_set_sha256=construction.result_sha256,
        source_manifest_sha256=source_manifest_sha256,
        export_policy_sha256=policy.policy_sha256,
        ffmpeg_version=ffmpeg_version,
        artifact_manifest=artifact_manifest,
        verification=verification,
    )


def _source_map(
    sources: tuple[OracleCropSource, ...],
) -> dict[str, OracleCropSource]:
    if not sources:
        raise ValueError("oracle crop sources must not be empty")
    result = {source.sample_id: source for source in sources}
    if len(result) != len(sources):
        raise ValueError("oracle crop source sample IDs must be unique")
    return result


def _validate_bindings(construction: PairConstructionResult) -> None:
    tokens = tuple(
        binding.artifact_token for binding in construction.artifact_bindings
    )
    samples = tuple(
        binding.sample_id for binding in construction.artifact_bindings
    )
    if not tokens:
        raise ValueError("artifact bindings must not be empty")
    if len(tokens) != len(set(tokens)):
        raise ValueError("artifact binding tokens must be unique")
    if len(samples) != len(set(samples)):
        raise ValueError("artifact binding sample IDs must be unique")
    for token in tokens:
        if not isinstance(token, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            token,
        ):
            raise ValueError("artifact token is not a safe filename stem")


def _validate_source_image(
    image: ImageProbe,
    policy: CropExportPolicy,
) -> None:
    if image.format_name not in {"png_pipe", "jpeg_pipe"}:
        raise ValueError("oracle crop source must be PNG or JPEG")
    if image.width * image.height > policy.maximum_source_pixels:
        raise ValueError("oracle crop source exceeds maximum_source_pixels")


def _validate_crop_inside(crop: CropBox, image: ImageProbe) -> None:
    if crop.x + crop.width > image.width or crop.y + crop.height > image.height:
        raise ValueError("crop box exceeds source image bounds")


def _validate_exported_image(
    image: ImageProbe,
    crop: CropBox,
    pixel_format: str,
) -> None:
    if image.format_name != "png_pipe":
        raise ValueError("exported artifact is not PNG")
    if image.width != crop.width or image.height != crop.height:
        raise ValueError("exported artifact dimensions differ from crop")
    if image.pixel_format != pixel_format:
        raise ValueError("exported artifact pixel format mismatch")
    if image.stream_tags or image.format_tags:
        raise ValueError("exported artifact retains metadata tags")


def _normalized_tags(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("image tags must be an object")
    return tuple(
        sorted((str(key), str(item)) for key, item in value.items())
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
