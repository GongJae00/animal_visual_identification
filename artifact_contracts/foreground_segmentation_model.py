"""Content binding for a local foreground-segmentation model snapshot."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_contracts.model_file_binding import ModelFileBinding
from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

BUNDLE_SCHEMA = "cvi.foreground_segmentation_model_bundle.v1"
MANIFEST_SCHEMA = "cvi.foreground_segmentation_model_manifest.v1"
INTERPRETATION = (
    "EXACT_LOCAL_FOREGROUND_MODEL_BINDING_NOT_SEMANTIC_OR_PERFORMANCE_VALIDATION"
)


@dataclass(frozen=True, slots=True)
class ForegroundSegmentationModelManifest:
    model_id: str
    source_revision: str
    model_family: str
    task: str
    license_id: str
    license_url: str
    input_multiple: int
    minimum_inference_side: int
    maximum_inference_side: int
    files: tuple[ModelFileBinding, ...]
    interpretation: str = INTERPRETATION
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA or self.interpretation != INTERPRETATION:
            raise ValueError("foreground model manifest schema or interpretation differs")
        for name in (
            "model_id",
            "source_revision",
            "model_family",
            "task",
            "license_id",
            "license_url",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"foreground model {name} must be canonical text")
        if self.source_revision.casefold() in {"main", "master", "head", "latest"}:
            raise ValueError("foreground model revision must be immutable")
        if not self.license_url.startswith("https://"):
            raise ValueError("foreground model license URL must use HTTPS")
        for name in (
            "input_multiple",
            "minimum_inference_side",
            "maximum_inference_side",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"foreground model {name} must be positive")
        if (
            self.minimum_inference_side > self.maximum_inference_side
            or self.minimum_inference_side % self.input_multiple
            or self.maximum_inference_side % self.input_multiple
        ):
            raise ValueError("foreground model inference bounds differ")
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, ModelFileBinding) for item in self.files
        ):
            raise TypeError("foreground model files must be file bindings")
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("foreground model files must be sorted and unique")
        required = {
            "BiRefNet_config.py",
            "birefnet.py",
            "config.json",
            "model.safetensors",
        }
        if not required <= set(paths):
            raise ValueError("foreground model core files are incomplete")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "model_family": self.model_family,
            "task": self.task,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "input_multiple": self.input_multiple,
            "minimum_inference_side": self.minimum_inference_side,
            "maximum_inference_side": self.maximum_inference_side,
            "files": [item.to_dict() for item in self.files],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, value: object) -> ForegroundSegmentationModelManifest:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("foreground model manifest schema differs")
        fields = dict(value)
        raw_files = fields["files"]
        if not isinstance(raw_files, list):
            raise TypeError("foreground model files must be an array")
        fields["files"] = tuple(
            ModelFileBinding.from_dict(item) for item in raw_files
        )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class ForegroundSegmentationArtifact:
    model_directory: Path
    manifest: ForegroundSegmentationModelManifest
    bundle_sha256: str

    @classmethod
    def load(
        cls, *, model_directory: Path, manifest_bundle_path: Path
    ) -> ForegroundSegmentationArtifact:
        root = Path(os.path.abspath(os.fspath(model_directory)))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("foreground model directory must be a regular directory")
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
            raise ValueError("foreground model bundle differs")
        manifest = ForegroundSegmentationModelManifest.from_dict(bundle["manifest"])
        if manifest.manifest_sha256 != bundle["manifest_sha256"]:
            raise ValueError("foreground model manifest digest differs")
        artifact = cls(root, manifest, document.raw_sha256)
        artifact.revalidate_local_files()
        return artifact

    def revalidate_local_files(self) -> None:
        root = self.model_directory.resolve(strict=True)
        for binding in self.manifest.files:
            target = root.joinpath(*binding.relative_path.split("/"))
            if target.is_symlink():
                raise ValueError("foreground model file must not be a symlink")
            resolved = target.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("foreground model file escapes model directory")
            read_retained_regular_file(
                resolved,
                expected_bytes=binding.byte_size,
                expected_sha256=binding.sha256,
                maximum_bytes=4_294_967_296,
                subject="foreground model file",
            )


def foreground_segmentation_model_bundle(
    manifest: ForegroundSegmentationModelManifest,
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
    "ForegroundSegmentationArtifact",
    "ForegroundSegmentationModelManifest",
    "foreground_segmentation_model_bundle",
]
