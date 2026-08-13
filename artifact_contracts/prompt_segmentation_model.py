"""Content binding for local prompt-segmentation model artifacts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_contracts.foundation_vision_model import FoundationFileBinding
from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

BUNDLE_SCHEMA = "cvi.prompt_segmentation_model_bundle.v1"
MANIFEST_SCHEMA = "cvi.prompt_segmentation_model_manifest.v1"
INTERPRETATION = "EXACT_LOCAL_MODEL_BINDING_NOT_SEMANTIC_OR_PERFORMANCE_VALIDATION"


@dataclass(frozen=True, slots=True)
class PromptSegmentationModelManifest:
    model_id: str
    source_revision: str
    model_family: str
    license_id: str
    license_url: str
    runtime_conversion: str
    files: tuple[FoundationFileBinding, ...]
    interpretation: str = INTERPRETATION
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA or self.interpretation != INTERPRETATION:
            raise ValueError("prompt model manifest schema or interpretation differs")
        for name in (
            "model_id",
            "source_revision",
            "model_family",
            "license_id",
            "license_url",
            "runtime_conversion",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"prompt model {name} must be canonical text")
        if self.source_revision.casefold() in {"main", "master", "head", "latest"}:
            raise ValueError("prompt model revision must be immutable")
        if not self.license_url.startswith("https://"):
            raise ValueError("prompt model license URL must use HTTPS")
        if not isinstance(self.files, tuple) or any(
            not isinstance(item, FoundationFileBinding) for item in self.files
        ):
            raise TypeError("prompt model files must be file bindings")
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("prompt model files must be sorted and unique")
        required = {"model.safetensors", "config.json", "preprocessor_config.json"}
        if not required <= set(paths):
            raise ValueError("prompt model core files are incomplete")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "model_family": self.model_family,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "runtime_conversion": self.runtime_conversion,
            "files": [item.to_dict() for item in self.files],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, value: object) -> PromptSegmentationModelManifest:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("prompt model manifest schema differs")
        fields = dict(value)
        raw_files = fields["files"]
        if not isinstance(raw_files, list):
            raise TypeError("prompt model files must be an array")
        fields["files"] = tuple(FoundationFileBinding.from_dict(item) for item in raw_files)
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class PromptSegmentationArtifact:
    model_directory: Path
    manifest: PromptSegmentationModelManifest
    bundle_sha256: str

    @classmethod
    def load(
        cls, *, model_directory: Path, manifest_bundle_path: Path
    ) -> PromptSegmentationArtifact:
        root = Path(os.path.abspath(os.fspath(model_directory)))
        if root.is_symlink() or not root.is_dir():
            raise ValueError("prompt model directory must be a regular directory")
        document = read_strict_json_document(manifest_bundle_path, maximum_bytes=1_048_576)
        bundle = document.payload
        if (
            set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
            or bundle["schema_version"] != BUNDLE_SCHEMA
            or not isinstance(bundle["manifest"], Mapping)
            or content_sha256(bundle["manifest"]) != bundle["manifest_sha256"]
        ):
            raise ValueError("prompt model bundle differs")
        manifest = PromptSegmentationModelManifest.from_dict(bundle["manifest"])
        if manifest.manifest_sha256 != bundle["manifest_sha256"]:
            raise ValueError("prompt model manifest digest differs")
        artifact = cls(root, manifest, document.raw_sha256)
        artifact.revalidate_local_files()
        return artifact

    def revalidate_local_files(self) -> None:
        root = self.model_directory.resolve(strict=True)
        for binding in self.manifest.files:
            target = root.joinpath(*binding.relative_path.split("/"))
            if target.is_symlink():
                raise ValueError("prompt model file must not be a symlink")
            resolved = target.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError("prompt model file escapes model directory")
            read_retained_regular_file(
                resolved,
                expected_bytes=binding.byte_size,
                expected_sha256=binding.sha256,
                maximum_bytes=4_294_967_296,
                subject="prompt model file",
            )


def prompt_segmentation_model_bundle(
    manifest: PromptSegmentationModelManifest,
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
    "PromptSegmentationArtifact",
    "PromptSegmentationModelManifest",
    "prompt_segmentation_model_bundle",
]
