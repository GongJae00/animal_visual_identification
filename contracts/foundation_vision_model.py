"""Content-bound local vision-foundation model artifacts."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

BUNDLE_SCHEMA = "cvi.foundation_vision_model_bundle.v1"
MANIFEST_SCHEMA = "cvi.foundation_vision_model_manifest.v1"
INTERPRETATION = (
    "EXACT_LOCAL_MODEL_AND_EXECUTABLE_SOURCE_BINDING_NOT_PERFORMANCE_OR_SAFETY_ADMISSION"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FoundationModelFamily(StrEnum):
    DINOV3_VIT = "DINOV3_VIT"
    CRADIO_V4 = "CRADIO_V4"


class FoundationModelUsageLane(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"


@dataclass(frozen=True, slots=True)
class FoundationFileBinding:
    relative_path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or self.relative_path != path.as_posix()
        ):
            raise ValueError("foundation model relative path is unsafe")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size <= 0
        ):
            raise ValueError("foundation model file size must be positive")
        _require_sha256(self.sha256, "foundation model file SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> FoundationFileBinding:
        if not isinstance(value, Mapping) or set(value) != {
            "relative_path",
            "byte_size",
            "sha256",
        }:
            raise ValueError("foundation model file binding schema differs")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FoundationVisionModelManifest:
    model_id: str
    source_revision: str
    family: FoundationModelFamily
    license_id: str
    license_url: str
    usage_lane: FoundationModelUsageLane
    patch_size: int
    dense_feature_dimension: int
    summary_dimension: int
    preferred_resolution: int
    maximum_resolution: int
    requires_local_code: bool
    weight: FoundationFileBinding
    config: FoundationFileBinding
    preprocessor: FoundationFileBinding
    executable_sources: tuple[FoundationFileBinding, ...]
    interpretation: str = INTERPRETATION
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA or self.interpretation != INTERPRETATION:
            raise ValueError("foundation model manifest schema or interpretation differs")
        for name in ("model_id", "source_revision", "license_id", "license_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"foundation model {name} must be canonical text")
        if self.source_revision.casefold() in {"main", "master", "head", "latest"}:
            raise ValueError("foundation model revision must be immutable")
        if not self.license_url.startswith("https://"):
            raise ValueError("foundation model license URL must use HTTPS")
        if not isinstance(self.family, FoundationModelFamily) or not isinstance(
            self.usage_lane, FoundationModelUsageLane
        ):
            raise TypeError("foundation model family and usage lane must be enums")
        for name in (
            "patch_size",
            "dense_feature_dimension",
            "summary_dimension",
            "preferred_resolution",
            "maximum_resolution",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"foundation model {name} must be positive")
        if self.preferred_resolution > self.maximum_resolution:
            raise ValueError("preferred resolution exceeds foundation model maximum")
        if self.preferred_resolution % self.patch_size != 0:
            raise ValueError("preferred resolution must align to patch size")
        if not isinstance(self.requires_local_code, bool):
            raise TypeError("requires_local_code must be boolean")
        if not all(
            isinstance(value, FoundationFileBinding)
            for value in (self.weight, self.config, self.preprocessor)
        ):
            raise TypeError("foundation model core files must be file bindings")
        if not isinstance(self.executable_sources, tuple) or any(
            not isinstance(value, FoundationFileBinding)
            for value in self.executable_sources
        ):
            raise TypeError("foundation executable sources must be file bindings")
        if self.requires_local_code != bool(self.executable_sources):
            raise ValueError("foundation local-code flag and source bindings differ")
        paths = [
            self.weight.relative_path,
            self.config.relative_path,
            self.preprocessor.relative_path,
            *(value.relative_path for value in self.executable_sources),
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("foundation model file bindings repeat a path")
        if self.executable_sources != tuple(
            sorted(self.executable_sources, key=lambda value: value.relative_path)
        ):
            raise ValueError("foundation executable sources must be sorted")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "family": self.family.value,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "usage_lane": self.usage_lane.value,
            "patch_size": self.patch_size,
            "dense_feature_dimension": self.dense_feature_dimension,
            "summary_dimension": self.summary_dimension,
            "preferred_resolution": self.preferred_resolution,
            "maximum_resolution": self.maximum_resolution,
            "requires_local_code": self.requires_local_code,
            "weight": self.weight.to_dict(),
            "config": self.config.to_dict(),
            "preprocessor": self.preprocessor.to_dict(),
            "executable_sources": [value.to_dict() for value in self.executable_sources],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, value: object) -> FoundationVisionModelManifest:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("foundation model manifest schema differs")
        values = dict(value)
        values["family"] = FoundationModelFamily(values["family"])
        values["usage_lane"] = FoundationModelUsageLane(values["usage_lane"])
        for name in ("weight", "config", "preprocessor"):
            values[name] = FoundationFileBinding.from_dict(values[name])
        raw_sources = values["executable_sources"]
        if not isinstance(raw_sources, list):
            raise TypeError("foundation executable sources must be an array")
        values["executable_sources"] = tuple(
            FoundationFileBinding.from_dict(item) for item in raw_sources
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class FoundationVisionArtifact:
    model_directory: Path
    manifest: FoundationVisionModelManifest
    manifest_document_sha256: str

    @classmethod
    def load(
        cls, *, model_directory: Path, manifest_bundle_path: Path
    ) -> FoundationVisionArtifact:
        root = Path(os.path.abspath(os.fspath(model_directory)))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("foundation model directory must be a regular directory")
        document = read_strict_json_document(
            manifest_bundle_path, maximum_bytes=16_777_216
        )
        bundle = document.payload
        if (
            set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
            or bundle["schema_version"] != BUNDLE_SCHEMA
            or not isinstance(bundle["manifest"], Mapping)
            or content_sha256(bundle["manifest"]) != bundle["manifest_sha256"]
        ):
            raise ValueError("foundation model bundle differs")
        manifest = FoundationVisionModelManifest.from_dict(bundle["manifest"])
        if manifest.manifest_sha256 != bundle["manifest_sha256"]:
            raise ValueError("foundation model manifest digest differs")
        artifact = cls(root, manifest, document.canonical_payload_sha256)
        artifact.revalidate_local_files()
        return artifact

    def revalidate_local_files(self) -> None:
        bindings = (
            self.manifest.weight,
            self.manifest.config,
            self.manifest.preprocessor,
            *self.manifest.executable_sources,
        )
        resolved_root = self.model_directory.resolve(strict=True)
        for binding in bindings:
            target = self.model_directory.joinpath(*PurePosixPath(binding.relative_path).parts)
            if target.is_symlink():
                raise ValueError("foundation model file must not be a symlink")
            resolved = target.resolve(strict=True)
            if not resolved.is_relative_to(resolved_root):
                raise ValueError("foundation model file escapes model directory")
            read_retained_regular_file(
                resolved,
                expected_bytes=binding.byte_size,
                expected_sha256=binding.sha256,
                maximum_bytes=4_294_967_296,
                subject="foundation model file",
            )


def foundation_model_bundle(
    manifest: FoundationVisionModelManifest,
) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest": manifest.to_dict(),
    }


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


__all__ = [
    "BUNDLE_SCHEMA",
    "FoundationFileBinding",
    "FoundationModelFamily",
    "FoundationModelUsageLane",
    "FoundationVisionArtifact",
    "FoundationVisionModelManifest",
    "foundation_model_bundle",
]
