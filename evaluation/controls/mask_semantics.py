"""Bounded mask-pixel semantic verification for shortcut controls."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from data.crop_export import probe_still_image
from evaluation.controls.policy import (
    ControlMaskManifest,
    ControlMaskVerification,
    MaskReviewStatus,
    MaskRole,
    verify_control_mask_files,
)
from evaluation.controls.scoring import (
    PairArtifactManifest,
    PairArtifactVerification,
    verify_pair_artifact_files,
)
from shared.foundation.provenance import content_sha256


@dataclass(frozen=True, slots=True)
class MaskSemanticPolicy:
    maximum_mask_pixels: int = 16_777_216
    raw_scan_chunk_bytes: int = 1_048_576
    timeout_seconds_per_mask: float = 30.0
    maximum_accessory_outside_dog_fraction: float = 0.0
    schema_version: str = "evaluation.mask_semantic_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.mask_semantic_policy.v1":
            raise ValueError("unsupported mask semantic policy schema")
        for name in ("maximum_mask_pixels", "raw_scan_chunk_bytes"):
            _require_positive_int(getattr(self, name), name)
        if (
            isinstance(self.timeout_seconds_per_mask, bool)
            or not isinstance(self.timeout_seconds_per_mask, (int, float))
            or not isfinite(self.timeout_seconds_per_mask)
            or self.timeout_seconds_per_mask <= 0
        ):
            raise ValueError("mask timeout must be finite and positive")
        value = self.maximum_accessory_outside_dog_fraction
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                "maximum accessory outside-dog fraction must be in [0, 1]"
            )

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "schema_version": self.schema_version,
            "maximum_mask_pixels": self.maximum_mask_pixels,
            "raw_scan_chunk_bytes": self.raw_scan_chunk_bytes,
            "timeout_seconds_per_mask": self.timeout_seconds_per_mask,
            "maximum_accessory_outside_dog_fraction": (
                self.maximum_accessory_outside_dog_fraction
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskSemanticPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "maximum_mask_pixels",
                "raw_scan_chunk_bytes",
                "timeout_seconds_per_mask",
                "maximum_accessory_outside_dog_fraction",
            },
            "mask semantic policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class MaskPixelStats:
    role: MaskRole
    foreground_pixels: int
    total_pixels: int

    def __post_init__(self) -> None:
        for name in ("foreground_pixels", "total_pixels"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_pixels <= 0:
            raise ValueError("total_pixels must be positive")
        if self.foreground_pixels > self.total_pixels:
            raise ValueError("foreground_pixels exceeds total_pixels")

    @property
    def foreground_fraction(self) -> float:
        return self.foreground_pixels / self.total_pixels

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "role": self.role.value,
            "foreground_pixels": self.foreground_pixels,
            "total_pixels": self.total_pixels,
            "foreground_fraction": self.foreground_fraction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaskPixelStats:
        _require_exact_keys(
            payload,
            {
                "role",
                "foreground_pixels",
                "total_pixels",
                "foreground_fraction",
            },
            "mask pixel stats",
        )
        result = cls(
            role=MaskRole(payload["role"]),
            foreground_pixels=payload["foreground_pixels"],
            total_pixels=payload["total_pixels"],
        )
        fraction = payload["foreground_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not isfinite(fraction)
            or fraction != result.foreground_fraction
        ):
            raise ValueError("mask foreground fraction is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class MaskEntrySemanticReceipt:
    base_artifact_token: str
    width: int
    height: int
    masks: tuple[MaskPixelStats, ...]
    accessory_outside_dog_pixels: int | None
    accessory_outside_dog_fraction: float | None

    def __post_init__(self) -> None:
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        if not self.masks:
            raise ValueError("mask semantic entry must contain mask stats")
        roles = tuple(mask.role for mask in self.masks)
        if len(roles) != len(set(roles)):
            raise ValueError("mask semantic roles must be unique")
        pixels = self.width * self.height
        if any(mask.total_pixels != pixels for mask in self.masks):
            raise ValueError("mask semantic totals differ from dimensions")
        if any(mask.foreground_pixels <= 0 for mask in self.masks):
            raise ValueError("verified mask support must be nonempty")
        has_accessory = MaskRole.ACCESSORY in roles
        if has_accessory and MaskRole.DOG not in roles:
            raise ValueError(
                "verified accessory semantics require dog semantics"
            )
        if has_accessory != (
            self.accessory_outside_dog_pixels is not None
            and self.accessory_outside_dog_fraction is not None
        ):
            raise ValueError(
                "accessory containment fields must match accessory presence"
            )
        if not has_accessory:
            return
        outside = self.accessory_outside_dog_pixels
        fraction = self.accessory_outside_dog_fraction
        if (
            isinstance(outside, bool)
            or not isinstance(outside, int)
            or outside < 0
        ):
            raise ValueError(
                "accessory_outside_dog_pixels must be non-negative"
            )
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError(
                "accessory outside-dog fraction must be in [0, 1]"
            )
        accessory_pixels = next(
            mask.foreground_pixels
            for mask in self.masks
            if mask.role is MaskRole.ACCESSORY
        )
        if accessory_pixels <= 0:
            raise ValueError("accessory mask support must be nonempty")
        if outside > accessory_pixels or fraction != (
            outside / accessory_pixels
        ):
            raise ValueError("accessory containment statistics conflict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_artifact_token": self.base_artifact_token,
            "width": self.width,
            "height": self.height,
            "masks": [mask.to_dict() for mask in self.masks],
            "accessory_outside_dog_pixels": (
                self.accessory_outside_dog_pixels
            ),
            "accessory_outside_dog_fraction": (
                self.accessory_outside_dog_fraction
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> MaskEntrySemanticReceipt:
        _require_exact_keys(
            payload,
            {
                "base_artifact_token",
                "width",
                "height",
                "masks",
                "accessory_outside_dog_pixels",
                "accessory_outside_dog_fraction",
            },
            "mask semantic entry",
        )
        masks = payload["masks"]
        if not isinstance(masks, list):
            raise TypeError("mask semantic entry masks must be a list")
        return cls(
            base_artifact_token=payload["base_artifact_token"],
            width=payload["width"],
            height=payload["height"],
            masks=tuple(MaskPixelStats.from_dict(item) for item in masks),
            accessory_outside_dog_pixels=payload[
                "accessory_outside_dog_pixels"
            ],
            accessory_outside_dog_fraction=payload[
                "accessory_outside_dog_fraction"
            ],
        )


@dataclass(frozen=True, slots=True)
class MaskSemanticVerification:
    base_artifact_manifest_sha256: str
    base_artifact_verification_sha256: str
    mask_manifest_sha256: str
    mask_file_verification_sha256: str
    policy_sha256: str
    ffmpeg_version: str
    entries: tuple[MaskEntrySemanticReceipt, ...]
    schema_version: str = "evaluation.mask_semantic_verification.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.mask_semantic_verification.v1":
            raise ValueError(
                "unsupported mask semantic verification schema"
            )
        for name in (
            "base_artifact_manifest_sha256",
            "base_artifact_verification_sha256",
            "mask_manifest_sha256",
            "mask_file_verification_sha256",
            "policy_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not isinstance(self.ffmpeg_version, str) or not (
            self.ffmpeg_version.strip()
        ):
            raise ValueError("ffmpeg_version must be non-empty")
        tokens = tuple(entry.base_artifact_token for entry in self.entries)
        if len(tokens) != len(set(tokens)):
            raise ValueError("mask semantic artifact tokens must be unique")

    @property
    def verification_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_artifact_manifest_sha256": (
                self.base_artifact_manifest_sha256
            ),
            "base_artifact_verification_sha256": (
                self.base_artifact_verification_sha256
            ),
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "mask_file_verification_sha256": (
                self.mask_file_verification_sha256
            ),
            "policy_sha256": self.policy_sha256,
            "ffmpeg_version": self.ffmpeg_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> MaskSemanticVerification:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "base_artifact_manifest_sha256",
                "base_artifact_verification_sha256",
                "mask_manifest_sha256",
                "mask_file_verification_sha256",
                "policy_sha256",
                "ffmpeg_version",
                "entries",
            },
            "mask semantic verification",
        )
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise TypeError("mask semantic entries must be a list")
        return cls(
            schema_version=payload["schema_version"],
            base_artifact_manifest_sha256=payload[
                "base_artifact_manifest_sha256"
            ],
            base_artifact_verification_sha256=payload[
                "base_artifact_verification_sha256"
            ],
            mask_manifest_sha256=payload["mask_manifest_sha256"],
            mask_file_verification_sha256=payload[
                "mask_file_verification_sha256"
            ],
            policy_sha256=payload["policy_sha256"],
            ffmpeg_version=payload["ffmpeg_version"],
            entries=tuple(
                MaskEntrySemanticReceipt.from_dict(item) for item in entries
            ),
        )


def build_mask_raw_decode_command(
    source: Path,
    destination: Path,
) -> tuple[str, ...]:
    return (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        "format=gray",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        str(destination),
    )


def verify_mask_pixel_semantics(
    *,
    base_root: Path,
    base_manifest: PairArtifactManifest,
    base_verification: PairArtifactVerification,
    mask_root: Path,
    mask_manifest: ControlMaskManifest,
    mask_file_verification: ControlMaskVerification,
    policy: MaskSemanticPolicy,
) -> MaskSemanticVerification:
    current_base = verify_pair_artifact_files(base_root, base_manifest)
    if current_base != base_verification:
        raise ValueError("base artifacts changed before mask semantic check")
    current_masks = verify_control_mask_files(mask_root, mask_manifest)
    if current_masks != mask_file_verification:
        raise ValueError("mask artifacts changed before semantic check")
    if (
        mask_manifest.base_artifact_manifest_sha256
        != base_manifest.manifest_sha256
    ):
        raise ValueError("mask and base artifact manifests differ")
    base_by_token = {
        entry.artifact_token: entry for entry in base_manifest.entries
    }
    mask_tokens = {
        entry.base_artifact_token for entry in mask_manifest.entries
    }
    if set(base_by_token) != mask_tokens:
        raise ValueError("mask manifest does not cover base artifact tokens")
    ffmpeg_version = subprocess.run(
        ("ffmpeg", "-version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout.splitlines()[0]
    receipts: list[MaskEntrySemanticReceipt] = []
    with TemporaryDirectory(prefix=".mask-semantic-") as temporary:
        temporary_root = Path(temporary)
        for entry in mask_manifest.entries:
            verified = tuple(
                mask
                for mask in entry.masks
                if mask.review_status is MaskReviewStatus.VERIFIED
            )
            if not verified:
                continue
            by_role = {mask.role: mask for mask in verified}
            if (
                MaskRole.ACCESSORY in by_role
                and MaskRole.DOG not in by_role
            ):
                raise ValueError(
                    "verified accessory mask requires a verified dog mask"
                )
            base_entry = base_by_token[entry.base_artifact_token]
            base_probe = probe_still_image(
                base_root / base_entry.relative_path
            )
            total_pixels = base_probe.width * base_probe.height
            if total_pixels > policy.maximum_mask_pixels:
                raise ValueError("base artifact exceeds maximum_mask_pixels")
            raw_paths: dict[MaskRole, Path] = {}
            stats: list[MaskPixelStats] = []
            try:
                for mask in verified:
                    source = mask_root / mask.relative_path
                    probe = probe_still_image(source)
                    if probe.format_name != "png_pipe":
                        raise ValueError("verified mask must decode as PNG")
                    if probe.pixel_format != "gray":
                        raise ValueError(
                            "verified mask pixel format must be gray"
                        )
                    if probe.stream_tags or probe.format_tags:
                        raise ValueError(
                            "verified mask must not contain metadata tags"
                        )
                    if (
                        probe.width != mask.width
                        or probe.height != mask.height
                        or probe.width != base_probe.width
                        or probe.height != base_probe.height
                    ):
                        raise ValueError(
                            "mask dimensions do not match manifest/base"
                        )
                    raw_path = temporary_root / f"{mask.role.value}.raw"
                    subprocess.run(
                        build_mask_raw_decode_command(source, raw_path),
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=policy.timeout_seconds_per_mask,
                    )
                    foreground = _scan_binary_raw(
                        raw_path,
                        expected_bytes=total_pixels,
                        chunk_bytes=policy.raw_scan_chunk_bytes,
                    )
                    if foreground == 0:
                        raise ValueError("verified mask support must be nonempty")
                    raw_paths[mask.role] = raw_path
                    stats.append(
                        MaskPixelStats(
                            mask.role,
                            foreground,
                            total_pixels,
                        )
                    )
                outside_pixels: int | None = None
                outside_fraction: float | None = None
                if MaskRole.ACCESSORY in raw_paths:
                    outside_pixels = _count_accessory_outside_dog(
                        raw_paths[MaskRole.DOG],
                        raw_paths[MaskRole.ACCESSORY],
                        expected_bytes=total_pixels,
                        chunk_bytes=policy.raw_scan_chunk_bytes,
                    )
                    accessory_foreground = next(
                        item.foreground_pixels
                        for item in stats
                        if item.role is MaskRole.ACCESSORY
                    )
                    outside_fraction = (
                        outside_pixels / accessory_foreground
                    )
                    if (
                        outside_fraction
                        > policy.maximum_accessory_outside_dog_fraction
                    ):
                        raise ValueError(
                            "accessory mask exceeds outside-dog tolerance"
                        )
                receipts.append(
                    MaskEntrySemanticReceipt(
                        base_artifact_token=entry.base_artifact_token,
                        width=base_probe.width,
                        height=base_probe.height,
                        masks=tuple(
                            sorted(stats, key=lambda item: item.role.value)
                        ),
                        accessory_outside_dog_pixels=outside_pixels,
                        accessory_outside_dog_fraction=outside_fraction,
                    )
                )
            finally:
                for raw_path in raw_paths.values():
                    raw_path.unlink(missing_ok=True)
    return MaskSemanticVerification(
        base_artifact_manifest_sha256=base_manifest.manifest_sha256,
        base_artifact_verification_sha256=content_sha256(
            base_verification.to_dict()
        ),
        mask_manifest_sha256=mask_manifest.manifest_sha256,
        mask_file_verification_sha256=content_sha256(
            mask_file_verification.to_dict()
        ),
        policy_sha256=policy.policy_sha256,
        ffmpeg_version=ffmpeg_version,
        entries=tuple(receipts),
    )


def _scan_binary_raw(
    path: Path,
    *,
    expected_bytes: int,
    chunk_bytes: int,
) -> int:
    if path.stat().st_size != expected_bytes:
        raise ValueError("decoded mask byte count differs from dimensions")
    foreground = 0
    seen = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            ones = chunk.count(255)
            zero = chunk.count(0)
            if ones + zero != len(chunk):
                raise ValueError("verified mask contains non-binary pixels")
            foreground += ones
            seen += len(chunk)
    if seen != expected_bytes:
        raise RuntimeError("decoded mask changed during binary scan")
    return foreground


def _count_accessory_outside_dog(
    dog_path: Path,
    accessory_path: Path,
    *,
    expected_bytes: int,
    chunk_bytes: int,
) -> int:
    outside = 0
    seen = 0
    with dog_path.open("rb") as dog, accessory_path.open("rb") as accessory:
        while True:
            dog_chunk = dog.read(chunk_bytes)
            accessory_chunk = accessory.read(chunk_bytes)
            if len(dog_chunk) != len(accessory_chunk):
                raise RuntimeError("dog/accessory raw masks differ in length")
            if not dog_chunk:
                break
            outside_bits = (
                int.from_bytes(accessory_chunk, byteorder="big")
                & ~int.from_bytes(dog_chunk, byteorder="big")
            ).bit_count()
            if outside_bits % 8:
                raise RuntimeError("binary mask bit accounting is inconsistent")
            outside += outside_bits // 8
            seen += len(dog_chunk)
    if seen != expected_bytes:
        raise RuntimeError("raw mask pair changed during containment scan")
    return outside


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
