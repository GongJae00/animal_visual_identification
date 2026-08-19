"""Immutable, content-addressed public crop artifact contracts."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from foundation.provenance import content_sha256


_PIXEL_HASH_DOMAIN = b"CVI_PIXEL_CANONICAL_RGB_V1\0"
_FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


@dataclass(frozen=True, slots=True)
class PublicCropVerificationPolicy:
    """Hard resource ceilings for untrusted reusable crop artifacts."""

    maximum_artifacts: int = 100_000
    maximum_encoded_bytes_per_file: int = 67_108_864
    maximum_total_encoded_bytes: int = 8_589_934_592
    maximum_width: int = 8_192
    maximum_height: int = 8_192
    maximum_decoded_pixels_per_image: int = 16_777_216
    maximum_total_decoded_pixels: int = 4_000_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY = PublicCropVerificationPolicy()


@dataclass(frozen=True, slots=True)
class PublicCropArtifact:
    """One exact crop for one opaque public sample and subject."""

    sample_token: str
    public_subject_token: str
    component_token: str
    source_variant: str
    relative_path: str
    content_sha256: str
    byte_size: int
    pixel_sha256: str
    width: int
    height: int
    mode: str
    format: str
    schema_version: str = "cvi.public_crop_artifact.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_crop_artifact.v1":
            raise ValueError("unsupported public crop artifact schema")
        for name in ("sample_token", "public_subject_token", "component_token"):
            _require_sha256(getattr(self, name), name)
        if self.sample_token == self.public_subject_token:
            raise ValueError("sample and public subject tokens must be distinct")
        _require_text(self.source_variant, "source_variant")
        if self.format not in _FORMAT_SUFFIX:
            raise ValueError("crop format must be JPEG, PNG, or WEBP")
        _require_text(self.mode, "mode", maximum=32)
        _require_canonical_path(self.relative_path, self.sample_token, self.format)
        _require_sha256(self.content_sha256, "content_sha256")
        _require_sha256(self.pixel_sha256, "pixel_sha256")
        for name in ("byte_size", "width", "height"):
            _require_positive_int(getattr(self, name), name)

    @property
    def artifact_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicCropArtifact:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "public crop artifact")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PublicCropManifest:
    artifacts: tuple[PublicCropArtifact, ...]
    interpretation: str = "PUBLIC_EXPERIMENTAL_SUBJECTS_ONLY_NOT_REGISTERED_DOG_IDENTITIES"
    schema_version: str = "cvi.public_crop_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_crop_manifest.v1":
            raise ValueError("unsupported public crop manifest schema")
        if self.interpretation != (
            "PUBLIC_EXPERIMENTAL_SUBJECTS_ONLY_NOT_REGISTERED_DOG_IDENTITIES"
        ):
            raise ValueError("public crop identity interpretation differs")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ValueError("public crop manifest must not be empty")
        if any(not isinstance(item, PublicCropArtifact) for item in self.artifacts):
            raise TypeError("manifest artifacts must be PublicCropArtifact")
        if tuple(sorted(self.artifacts, key=lambda item: item.sample_token)) != self.artifacts:
            raise ValueError("public crop artifacts must be sorted by sample token")
        _require_unique(
            tuple(item.sample_token for item in self.artifacts), "sample tokens"
        )
        _require_unique(
            tuple(item.relative_path for item in self.artifacts), "artifact paths"
        )

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def content_digest(self) -> str:
        return self.manifest_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicCropManifest:
        _require_exact_keys(
            payload,
            {"schema_version", "artifacts", "interpretation"},
            "public crop manifest",
        )
        if not isinstance(payload["artifacts"], list):
            raise TypeError("public crop artifacts must be a list")
        return cls(
            artifacts=tuple(
                PublicCropArtifact.from_dict(item) for item in payload["artifacts"]
            ),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class PublicCropVerification:
    crop_manifest_sha256: str
    verified_files: int
    verified_bytes: int
    decoded_rgb_pixels_verified: bool
    state: str = "PASS"
    schema_version: str = "cvi.public_crop_verification.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_crop_verification.v1":
            raise ValueError("unsupported public crop verification schema")
        _require_sha256(self.crop_manifest_sha256, "crop_manifest_sha256")
        for name in ("verified_files", "verified_bytes"):
            _require_nonnegative_int(getattr(self, name), name)
        if not isinstance(self.decoded_rgb_pixels_verified, bool):
            raise TypeError("decoded_rgb_pixels_verified must be boolean")
        if self.state != "PASS":
            raise ValueError("public crop verification state must be PASS")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicCropVerification:
        _require_exact_keys(
            payload, set(cls.__dataclass_fields__), "public crop verification"
        )
        return cls(**payload)


def canonical_rgb_pixel_sha256(width: int, height: int, pixels: bytes) -> str:
    """Hash RGB bytes with the same domain and geometry binding as content audits."""

    _require_positive_int(width, "width")
    _require_positive_int(height, "height")
    if not isinstance(pixels, bytes) or len(pixels) != width * height * 3:
        raise ValueError("RGB pixel bytes differ from declared geometry")
    digest = hashlib.sha256()
    digest.update(_PIXEL_HASH_DOMAIN)
    digest.update(struct.pack(">QQ", width, height))
    digest.update(pixels)
    return digest.hexdigest()


def verify_public_crop_manifest(
    root: Path,
    manifest: PublicCropManifest,
    *,
    verify_decoded_rgb_pixels: bool = True,
    policy: PublicCropVerificationPolicy = DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY,
) -> PublicCropVerification:
    """Verify an exact flat crop root without recursive path discovery."""

    if not isinstance(manifest, PublicCropManifest):
        raise TypeError("manifest must be PublicCropManifest")
    if not isinstance(verify_decoded_rgb_pixels, bool):
        raise TypeError("verify_decoded_rgb_pixels must be boolean")
    _require_verification_policy(policy)
    _require_artifacts_within_policy(manifest.artifacts, policy)

    verified_bytes = 0
    directory_fd = _open_crop_root(root)
    try:
        actual_paths = _scan_crop_root(directory_fd, policy)
        expected_paths = {item.relative_path for item in manifest.artifacts}
        missing = expected_paths - actual_paths
        extra = actual_paths - expected_paths
        if missing or extra:
            raise ValueError(
                "crop root paths mismatch; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        for artifact in manifest.artifacts:
            payload = _read_exact_regular_file(directory_fd, artifact, policy)
            if verify_decoded_rgb_pixels:
                _verify_decoded_image(payload, artifact, policy)
            verified_bytes += artifact.byte_size
    finally:
        os.close(directory_fd)
    return PublicCropVerification(
        crop_manifest_sha256=manifest.manifest_sha256,
        verified_files=len(manifest.artifacts),
        verified_bytes=verified_bytes,
        decoded_rgb_pixels_verified=verify_decoded_rgb_pixels,
    )


def read_verified_crop_artifact(
    root: Path,
    artifact: PublicCropArtifact,
    *,
    verify_decoded_rgb_pixels: bool = True,
    policy: PublicCropVerificationPolicy = DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY,
) -> bytes:
    """Read one crop through no-follow byte and decoded-pixel verification."""

    if not isinstance(artifact, PublicCropArtifact):
        raise TypeError("artifact must be PublicCropArtifact")
    if not isinstance(verify_decoded_rgb_pixels, bool):
        raise TypeError("verify_decoded_rgb_pixels must be boolean")
    _require_verification_policy(policy)
    _require_artifacts_within_policy((artifact,), policy)

    directory_fd = _open_crop_root(root)
    try:
        payload = _read_exact_regular_file(directory_fd, artifact, policy)
        if verify_decoded_rgb_pixels:
            _verify_decoded_image(payload, artifact, policy)
        return payload
    finally:
        os.close(directory_fd)


def _open_crop_root(root: Path) -> int:
    if root.is_symlink():
        raise ValueError("crop root must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise NotADirectoryError(root)
    return descriptor


def _scan_crop_root(
    directory_fd: int, policy: PublicCropVerificationPolicy
) -> set[str]:
    names: set[str] = set()
    with os.scandir(directory_fd) as entries:
        for count, entry in enumerate(entries, start=1):
            if count > policy.maximum_artifacts:
                raise ValueError("crop root entry count exceeds maximum_artifacts")
            observed = os.stat(
                entry.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError("crop root must not contain symlinks")
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError("crop root must contain regular files only")
            names.add(entry.name)
    return names


def _read_exact_regular_file(
    directory_fd: int,
    artifact: PublicCropArtifact,
    policy: PublicCropVerificationPolicy,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(
            artifact.relative_path, flags, dir_fd=directory_fd
        )
    except OSError as error:
        raise ValueError(f"crop artifact cannot be opened safely: {artifact.sample_token}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("crop artifact must be a regular file")
        if initial.st_size > policy.maximum_encoded_bytes_per_file:
            raise ValueError(
                "crop artifact exceeds maximum_encoded_bytes_per_file"
            )
        if initial.st_size != artifact.byte_size:
            raise ValueError(f"crop byte size mismatch: {artifact.sample_token}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        remaining = artifact.byte_size
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
            total += len(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"crop byte count mismatch: {artifact.sample_token}")
        final = os.fstat(descriptor)
        named_final = os.stat(
            artifact.relative_path,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            initial.st_dev != final.st_dev
            or initial.st_ino != final.st_ino
            or initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
        ):
            raise RuntimeError(f"crop changed during verification: {artifact.sample_token}")
        if not stat.S_ISREG(named_final.st_mode) or (
            named_final.st_dev,
            named_final.st_ino,
        ) != (final.st_dev, final.st_ino):
            raise RuntimeError(
                f"crop path changed during verification: {artifact.sample_token}"
            )
    finally:
        os.close(descriptor)
    if total != artifact.byte_size:
        raise ValueError(f"crop byte count mismatch: {artifact.sample_token}")
    if digest.hexdigest() != artifact.content_sha256:
        raise ValueError(f"crop content hash mismatch: {artifact.sample_token}")
    return b"".join(chunks)


def _verify_decoded_image(
    payload: bytes,
    artifact: PublicCropArtifact,
    policy: PublicCropVerificationPolicy,
) -> None:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError("Pillow is required for decoded crop verification") from error
    try:
        with Image.open(io.BytesIO(payload)) as image:
            _require_dimensions_within_policy(
                image.width, image.height, policy, "decoded crop"
            )
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("crop image must contain exactly one frame")
            if image.format != artifact.format:
                raise ValueError(f"crop format mismatch: {artifact.sample_token}")
            if image.mode != artifact.mode:
                raise ValueError(f"crop mode mismatch: {artifact.sample_token}")
            if image.size != (artifact.width, artifact.height):
                raise ValueError(f"crop dimensions mismatch: {artifact.sample_token}")
            image.load()
            with image.convert("RGB") as rgb:
                pixel_digest = canonical_rgb_pixel_sha256(
                    artifact.width, artifact.height, rgb.tobytes("raw", "RGB")
                )
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as error:
        raise ValueError(f"crop image decode failed: {artifact.sample_token}") from error
    if pixel_digest != artifact.pixel_sha256:
        raise ValueError(f"crop pixel hash mismatch: {artifact.sample_token}")


def _require_verification_policy(policy: object) -> None:
    if not isinstance(policy, PublicCropVerificationPolicy):
        raise TypeError("policy must be PublicCropVerificationPolicy")


def _require_artifacts_within_policy(
    artifacts: tuple[PublicCropArtifact, ...],
    policy: PublicCropVerificationPolicy,
) -> None:
    if len(artifacts) > policy.maximum_artifacts:
        raise ValueError("crop artifact count exceeds maximum_artifacts")
    total_encoded_bytes = 0
    total_decoded_pixels = 0
    for artifact in artifacts:
        if artifact.byte_size > policy.maximum_encoded_bytes_per_file:
            raise ValueError(
                "crop artifact exceeds maximum_encoded_bytes_per_file"
            )
        _require_dimensions_within_policy(
            artifact.width, artifact.height, policy, "crop artifact"
        )
        pixels = artifact.width * artifact.height
        total_encoded_bytes += artifact.byte_size
        total_decoded_pixels += pixels
        if total_encoded_bytes > policy.maximum_total_encoded_bytes:
            raise ValueError(
                "crop artifacts exceed maximum_total_encoded_bytes"
            )
        if total_decoded_pixels > policy.maximum_total_decoded_pixels:
            raise ValueError(
                "crop artifacts exceed maximum_total_decoded_pixels"
            )


def _require_dimensions_within_policy(
    width: int,
    height: int,
    policy: PublicCropVerificationPolicy,
    subject: str,
) -> None:
    if width > policy.maximum_width:
        raise ValueError(f"{subject} width exceeds maximum_width")
    if height > policy.maximum_height:
        raise ValueError(f"{subject} height exceeds maximum_height")
    if width * height > policy.maximum_decoded_pixels_per_image:
        raise ValueError(
            f"{subject} exceeds maximum_decoded_pixels_per_image"
        )


def _require_canonical_path(value: object, sample_token: str, image_format: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("crop relative_path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError("crop relative_path must be one canonical root filename")
    expected = f"{sample_token}{_FORMAT_SUFFIX[image_format]}"
    if value != expected:
        raise ValueError(f"crop relative_path must equal {expected}")


def _require_exact_keys(payload: object, expected: set[str], context: str) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_text(value: object, name: str, *, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"public crop manifest has duplicate {name}")
