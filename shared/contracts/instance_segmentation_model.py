"""Content binding for a local supervised instance-segmentation model."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.contracts.model_file_binding import ModelFileBinding
from shared.foundation.protected_io import read_strict_json_document
from shared.foundation.provenance import content_sha256
from shared.foundation.retained_file import read_retained_regular_file

BUNDLE_SCHEMA = "parsing.instance_segmentation_model_bundle.v1"
MANIFEST_SCHEMA = "parsing.instance_segmentation_model_manifest.v1"
INTERPRETATION = (
    "EXACT_LOCAL_INSTANCE_MODEL_BINDING_NOT_DOMAIN_OR_PERFORMANCE_VALIDATION"
)


@dataclass(frozen=True, slots=True)
class InstanceSegmentationModelManifest:
    model_id: str
    source_revision: str
    model_family: str
    training_label_space: str
    license_id: str
    license_url: str
    files: tuple[ModelFileBinding, ...]
    interpretation: str = INTERPRETATION
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA or self.interpretation != INTERPRETATION:
            raise ValueError("instance model manifest schema or interpretation differs")
        for name in (
            "model_id",
            "source_revision",
            "model_family",
            "training_label_space",
            "license_id",
            "license_url",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"instance model {name} must be canonical text")
        if self.source_revision.casefold() in {"main", "master", "head", "latest"}:
            raise ValueError("instance model revision must be immutable")
        if not self.license_url.startswith("https://"):
            raise ValueError("instance model license URL must use HTTPS")
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, ModelFileBinding) for item in self.files
        ):
            raise TypeError("instance model files must be file bindings")
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("instance model files must be sorted and unique")
        required = {"config.json", "model.safetensors", "preprocessor_config.json"}
        if not required <= set(paths):
            raise ValueError("instance model core files are incomplete")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "model_family": self.model_family,
            "training_label_space": self.training_label_space,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "files": [item.to_dict() for item in self.files],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, value: object) -> InstanceSegmentationModelManifest:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("instance model manifest schema differs")
        fields = dict(value)
        raw_files = fields["files"]
        if not isinstance(raw_files, list):
            raise TypeError("instance model files must be an array")
        fields["files"] = tuple(
            ModelFileBinding.from_dict(item) for item in raw_files
        )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class InstanceSegmentationArtifact:
    model_directory: Path
    manifest: InstanceSegmentationModelManifest
    bundle_sha256: str

    @classmethod
    def load(
        cls, *, model_directory: Path, manifest_bundle_path: Path
    ) -> InstanceSegmentationArtifact:
        root = Path(os.path.abspath(os.fspath(model_directory)))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("instance model directory must be a regular directory")
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
            raise ValueError("instance model bundle differs")
        manifest = InstanceSegmentationModelManifest.from_dict(bundle["manifest"])
        if manifest.manifest_sha256 != bundle["manifest_sha256"]:
            raise ValueError("instance model manifest digest differs")
        artifact = cls(root, manifest, document.raw_sha256)
        artifact.revalidate_local_files()
        return artifact

    def revalidate_local_files(self) -> None:
        root = self.model_directory.resolve(strict=True)
        for binding in self.manifest.files:
            target = root.joinpath(*binding.relative_path.split("/"))
            if target.is_symlink():
                raise ValueError("instance model file must not be a symlink")
            resolved = target.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("instance model file escapes model directory")
            read_retained_regular_file(
                resolved,
                expected_bytes=binding.byte_size,
                expected_sha256=binding.sha256,
                maximum_bytes=4_294_967_296,
                subject="instance model file",
            )


def instance_segmentation_model_bundle(
    manifest: InstanceSegmentationModelManifest,
) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest": manifest.to_dict(),
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "INTERPRETATION",
    "MANIFEST_SCHEMA",
    "InstanceSegmentationArtifact",
    "InstanceSegmentationModelManifest",
    "instance_segmentation_model_bundle",
]
