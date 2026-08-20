"""Content-bound pair artifact manifests shared by preparation and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from data.acquisition import sha256_file
from shared.foundation.provenance import content_sha256


class PairArtifactBindingView(Protocol):
    artifact_token: str
    sample_id: str


class PairConstructionView(Protocol):
    @property
    def result_sha256(self) -> str: ...

    @property
    def artifact_bindings(self) -> Sequence[PairArtifactBindingView]: ...

    def artifact_binding_payload(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PairArtifactEntry:
    artifact_token: str
    relative_path: str
    content_sha256: str
    byte_size: int
    media_type: str

    def __post_init__(self) -> None:
        _require_nonempty(self.artifact_token, "artifact_token")
        _validate_artifact_relative_path(
            self.relative_path,
            self.artifact_token,
            self.media_type,
        )
        _validate_sha256(self.content_sha256, "content_sha256")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size <= 0
        ):
            raise ValueError("byte_size must be a positive integer")
        _require_nonempty(self.media_type, "media_type")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "artifact_token": self.artifact_token,
            "relative_path": self.relative_path,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairArtifactEntry:
        _require_exact_keys(
            payload,
            {
                "artifact_token",
                "relative_path",
                "content_sha256",
                "byte_size",
                "media_type",
            },
            "pair artifact entry",
        )
        return cls(
            artifact_token=payload["artifact_token"],
            relative_path=payload["relative_path"],
            content_sha256=payload["content_sha256"],
            byte_size=payload["byte_size"],
            media_type=payload["media_type"],
        )


@dataclass(frozen=True, slots=True)
class PairArtifactManifest:
    pair_set_sha256: str
    artifact_bindings_sha256: str
    entries: tuple[PairArtifactEntry, ...]
    schema_version: str = "evaluation.pair_artifact_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.pair_artifact_manifest.v1":
            raise ValueError("unsupported pair artifact manifest schema")
        _validate_sha256(self.pair_set_sha256, "pair_set_sha256")
        _validate_sha256(
            self.artifact_bindings_sha256,
            "artifact_bindings_sha256",
        )
        if not self.entries:
            raise ValueError("artifact manifest entries must not be empty")
        tokens = tuple(entry.artifact_token for entry in self.entries)
        if len(tokens) != len(set(tokens)):
            raise ValueError("artifact manifest tokens must be unique")
        paths = tuple(entry.relative_path for entry in self.entries)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact manifest paths must be unique")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_set_sha256": self.pair_set_sha256,
            "artifact_bindings_sha256": self.artifact_bindings_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairArtifactManifest:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "pair_set_sha256",
                "artifact_bindings_sha256",
                "entries",
            },
            "pair artifact manifest",
        )
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise TypeError("artifact manifest entries must be a list")
        return cls(
            schema_version=payload["schema_version"],
            pair_set_sha256=payload["pair_set_sha256"],
            artifact_bindings_sha256=payload["artifact_bindings_sha256"],
            entries=tuple(PairArtifactEntry.from_dict(item) for item in entries),
        )


@dataclass(frozen=True, slots=True)
class PairArtifactVerification:
    artifact_manifest_sha256: str
    verified_files: int
    verified_bytes: int

    def __post_init__(self) -> None:
        _validate_sha256(
            self.artifact_manifest_sha256,
            "artifact_manifest_sha256",
        )
        for name in ("verified_files", "verified_bytes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": "evaluation.pair_artifact_verification.v1",
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "verified_files": self.verified_files,
            "verified_bytes": self.verified_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairArtifactVerification:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_manifest_sha256",
                "verified_files",
                "verified_bytes",
            },
            "pair artifact verification",
        )
        if payload["schema_version"] != "evaluation.pair_artifact_verification.v1":
            raise ValueError("unsupported pair artifact verification schema")
        return cls(
            artifact_manifest_sha256=payload["artifact_manifest_sha256"],
            verified_files=payload["verified_files"],
            verified_bytes=payload["verified_bytes"],
        )


def validate_pair_artifact_manifest(
    construction: PairConstructionView,
    manifest: PairArtifactManifest,
) -> None:
    if manifest.pair_set_sha256 != construction.result_sha256:
        raise ValueError("artifact manifest pair-set hash mismatch")
    expected_bindings_hash = content_sha256(
        construction.artifact_binding_payload()
    )
    if manifest.artifact_bindings_sha256 != expected_bindings_hash:
        raise ValueError("artifact manifest binding hash mismatch")
    expected_tokens = {
        binding.artifact_token for binding in construction.artifact_bindings
    }
    actual_tokens = {entry.artifact_token for entry in manifest.entries}
    missing = expected_tokens - actual_tokens
    extra = actual_tokens - expected_tokens
    if missing or extra:
        raise ValueError(
            "artifact manifest token mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def verify_pair_artifact_files(
    root: Path,
    manifest: PairArtifactManifest,
) -> PairArtifactVerification:
    if root.is_symlink():
        raise ValueError("artifact root must not be a symlink")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    directory_entries = tuple(resolved_root.iterdir())
    if any(entry.is_symlink() for entry in directory_entries):
        raise ValueError("artifact directory must not contain symlinks")
    if any(not entry.is_file() for entry in directory_entries):
        raise ValueError("artifact directory must contain files only")
    expected_names = {entry.relative_path for entry in manifest.entries}
    actual_names = {entry.name for entry in directory_entries}
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing or extra:
        raise ValueError(
            "artifact directory entries mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    verified_bytes = 0
    for entry in manifest.entries:
        path = resolved_root / entry.relative_path
        initial = path.stat()
        if initial.st_size != entry.byte_size:
            raise ValueError(
                f"artifact byte-size mismatch: {entry.artifact_token}"
            )
        digest = sha256_file(path)
        final = path.stat()
        if (
            initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
        ):
            raise RuntimeError(
                f"artifact changed during verification: {entry.artifact_token}"
            )
        if digest != entry.content_sha256:
            raise ValueError(
                f"artifact content hash mismatch: {entry.artifact_token}"
            )
        verified_bytes += entry.byte_size
    return PairArtifactVerification(
        artifact_manifest_sha256=manifest.manifest_sha256,
        verified_files=len(manifest.entries),
        verified_bytes=verified_bytes,
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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


def _validate_artifact_relative_path(
    value: str,
    artifact_token: str,
    media_type: str,
) -> None:
    _require_nonempty(value, "relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise ValueError("artifact relative_path must be one filename")
    expected_extensions = {
        "image/png": {".png"},
        "image/jpeg": {".jpg", ".jpeg"},
    }
    if media_type not in expected_extensions:
        raise ValueError("unsupported artifact media_type")
    if path.suffix.casefold() not in expected_extensions[media_type]:
        raise ValueError("artifact extension does not match media_type")
    if path.stem != artifact_token:
        raise ValueError("artifact filename stem must equal artifact token")


__all__ = [
    "PairArtifactBindingView",
    "PairArtifactEntry",
    "PairArtifactManifest",
    "PairArtifactVerification",
    "PairConstructionView",
    "validate_pair_artifact_manifest",
    "verify_pair_artifact_files",
]
