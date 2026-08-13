"""Content binding for a candidate visible-animal parsing runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from contracts.model_file_binding import ModelFileBinding
from foundation.provenance import content_sha256

LEGACY_BUNDLE_SCHEMA = "cvi.animal_parsing_runtime_bundle.v1"
LEGACY_MANIFEST_SCHEMA = "cvi.animal_parsing_runtime_manifest.v1"
BUNDLE_SCHEMA = "cvi.animal_parsing_runtime_bundle.v2"
MANIFEST_SCHEMA = "cvi.animal_parsing_runtime_manifest.v2"
SUPPORTED_BUNDLE_SCHEMAS = frozenset({LEGACY_BUNDLE_SCHEMA, BUNDLE_SCHEMA})
QUALIFICATION = "CANDIDATE_NOT_HUMAN_PIXEL_MASK_VERIFIED"
INTERPRETATION = (
    "EXACT_PARSER_SOURCE_MODEL_POLICY_AND_EVALUATION_BINDING_NOT_PRODUCTION_PROMOTION"
)
_POLICY_FIELDS = {
    "schema_version",
    "class_names",
    "duplicate_mask_iou",
    "maximum_instances",
    "refinement_context_fraction",
    "semantic_support_threshold",
    "semantic_core_threshold",
    "foreground_threshold",
    "support_dilation_fraction",
    "minimum_support_dilation_pixels",
    "maximum_support_dilation_pixels",
    "minimum_mask_pixels",
    "minimum_semantic_shape_iou",
    "review_semantic_shape_iou",
    "review_ownership_retention",
    "minimum_ownership_retention",
    "review_component_count",
    "maximum_component_count",
}


@dataclass(frozen=True, slots=True)
class ParsingEvaluationBinding:
    name: str
    schema_version: str
    interpretation: str
    byte_size: int
    raw_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        for name in ("name", "schema_version", "interpretation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"parsing evaluation {name} must be canonical text")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size <= 0
        ):
            raise ValueError("parsing evaluation byte size must be positive")
        for name in ("raw_sha256", "content_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"parsing evaluation {name} differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "interpretation": self.interpretation,
            "byte_size": self.byte_size,
            "raw_sha256": self.raw_sha256,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ParsingEvaluationBinding:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise ValueError("parsing evaluation binding schema differs")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class AnimalParsingRuntimeManifest:
    parser_family: str
    qualification: str
    ontology: str
    ontology_description: str
    supported_classes: tuple[str, ...]
    policy: dict[str, Any]
    policy_sha256: str
    foreground_model_manifest_sha256: str
    foreground_model_bundle_raw_sha256: str
    instance_model_manifest_sha256: str
    instance_model_bundle_raw_sha256: str
    source_files: tuple[ModelFileBinding, ...]
    evaluation_reports: tuple[ParsingEvaluationBinding, ...]
    inference_batching: dict[str, Any] | None = None
    frozen_cache: dict[str, Any] | None = None
    runtime_libraries: dict[str, str] | None = None
    interpretation: str = INTERPRETATION
    schema_version: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema_version not in {LEGACY_MANIFEST_SCHEMA, MANIFEST_SCHEMA}
            or self.interpretation != INTERPRETATION
            or self.qualification != QUALIFICATION
        ):
            raise ValueError("animal parsing runtime status differs")
        for name in ("parser_family", "ontology", "ontology_description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"animal parsing runtime {name} must be canonical text"
                )
        if (
            not isinstance(self.supported_classes, tuple)
            or not self.supported_classes
            or any(
                not isinstance(name, str) or not name for name in self.supported_classes
            )
            or tuple(sorted(set(self.supported_classes))) != self.supported_classes
        ):
            raise ValueError(
                "animal parsing supported classes must be sorted and unique"
            )
        if (
            not isinstance(self.policy, dict)
            or content_sha256(self.policy) != self.policy_sha256
        ):
            raise ValueError("animal parsing policy digest differs")
        if self.schema_version == LEGACY_MANIFEST_SCHEMA:
            if any(
                value is not None
                for value in (
                    self.inference_batching,
                    self.frozen_cache,
                    self.runtime_libraries,
                )
            ):
                raise ValueError("legacy animal parsing runtime fields differ")
        else:
            self._validate_v2_fields()
        for name in (
            "policy_sha256",
            "foreground_model_manifest_sha256",
            "foreground_model_bundle_raw_sha256",
            "instance_model_manifest_sha256",
            "instance_model_bundle_raw_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"animal parsing runtime {name} differs")
        paths = [item.relative_path for item in self.source_files]
        if (
            not self.source_files
            or paths != sorted(paths)
            or len(paths) != len(set(paths))
        ):
            raise ValueError("animal parsing runtime source files differ")
        report_names = [item.name for item in self.evaluation_reports]
        if (
            not self.evaluation_reports
            or report_names != sorted(report_names)
            or len(report_names) != len(set(report_names))
        ):
            raise ValueError("animal parsing evaluation reports differ")

    def _validate_v2_fields(self) -> None:
        policy_classes = self.policy.get("class_names")
        policy_schema = self.policy.get("schema_version")
        if (
            set(self.policy) != _POLICY_FIELDS
            or policy_schema
            not in {
                "cvi.animal_parsing_policy.v4",
                "cvi.animal_parsing_policy.v5",
                "cvi.animal_parsing_policy.v6",
            }
            or not isinstance(policy_classes, list)
            or not policy_classes
            or any(not isinstance(name, str) or not name for name in policy_classes)
            or tuple(sorted(policy_classes)) != self.supported_classes
            or (
                policy_schema == "cvi.animal_parsing_policy.v6"
                and policy_classes != ["dog"]
            )
        ):
            raise ValueError("animal parsing policy classes differ from runtime")
        batching = self.inference_batching
        if (
            not isinstance(batching, dict)
            or set(batching)
            != {
                "job_batch_size",
                "instance_batch_size",
                "foreground_batch_size",
                "job_ordering",
                "publication_workers",
                "shape_policy",
                "oom_policy",
            }
            or any(
                isinstance(batching[field], bool)
                or not isinstance(batching[field], int)
                or batching[field] not in {1, 2, 4, 8, 16}
                for field in (
                    "job_batch_size",
                    "instance_batch_size",
                    "foreground_batch_size",
                )
            )
            or len(
                {
                    batching["job_batch_size"],
                    batching["instance_batch_size"],
                    batching["foreground_batch_size"],
                }
            )
            != 1
            or batching["job_ordering"] != "SOURCE_SHA256_ASC"
            or batching["publication_workers"] not in {1, 4}
            or batching["shape_policy"] != "EXACT_PREPROCESSED_SHAPE_BUCKETS"
            or batching["oom_policy"] != "FAIL_CLOSED_NO_RETRY"
        ):
            raise ValueError("animal parsing inference batching policy differs")
        if self.frozen_cache != {
            "array_encoding": "BASE64_ZLIB_C_ORDER",
            "zlib_level": 1,
            "retained_arrays": [
                "instance_probability",
                "foreground_probability",
                "ownership_probability",
                "hard_mask",
            ],
        }:
            raise ValueError("animal parsing frozen cache policy differs")
        required_libraries = {"numpy", "pillow", "torch", "torchvision", "transformers"}
        if (
            not isinstance(self.runtime_libraries, dict)
            or set(self.runtime_libraries) != required_libraries
            or any(
                not isinstance(name, str) or not isinstance(version, str) or not version
                for name, version in self.runtime_libraries.items()
            )
        ):
            raise ValueError("animal parsing runtime library binding differs")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "parser_family": self.parser_family,
            "qualification": self.qualification,
            "ontology": self.ontology,
            "ontology_description": self.ontology_description,
            "supported_classes": list(self.supported_classes),
            "policy": self.policy,
            "policy_sha256": self.policy_sha256,
            "foreground_model_manifest_sha256": (self.foreground_model_manifest_sha256),
            "foreground_model_bundle_raw_sha256": (
                self.foreground_model_bundle_raw_sha256
            ),
            "instance_model_manifest_sha256": self.instance_model_manifest_sha256,
            "instance_model_bundle_raw_sha256": (self.instance_model_bundle_raw_sha256),
            "source_files": [item.to_dict() for item in self.source_files],
            "evaluation_reports": [item.to_dict() for item in self.evaluation_reports],
            "interpretation": self.interpretation,
        }
        if self.schema_version == MANIFEST_SCHEMA:
            payload.update(
                inference_batching=self.inference_batching,
                frozen_cache=self.frozen_cache,
                runtime_libraries=self.runtime_libraries,
            )
        return payload

    @classmethod
    def from_dict(cls, value: object) -> AnimalParsingRuntimeManifest:
        if not isinstance(value, Mapping):
            raise ValueError(  # noqa: TRY004 - persisted artifact validation contract
                "animal parsing runtime manifest schema differs"
            )
        schema = value.get("schema_version")
        common_fields = set(cls.__dataclass_fields__) - {
            "inference_batching",
            "frozen_cache",
            "runtime_libraries",
        }
        expected_fields = (
            common_fields
            if schema == LEGACY_MANIFEST_SCHEMA
            else set(cls.__dataclass_fields__)
        )
        if set(value) != expected_fields:
            raise ValueError("animal parsing runtime manifest schema differs")
        fields = dict(value)
        supported_classes = fields["supported_classes"]
        source_files = fields["source_files"]
        evaluation_reports = fields["evaluation_reports"]
        if (
            not isinstance(supported_classes, list)
            or not isinstance(source_files, list)
            or not isinstance(evaluation_reports, list)
        ):
            raise TypeError("animal parsing runtime manifest arrays differ")
        fields["supported_classes"] = tuple(supported_classes)
        fields["source_files"] = tuple(
            ModelFileBinding.from_dict(item) for item in source_files
        )
        fields["evaluation_reports"] = tuple(
            ParsingEvaluationBinding.from_dict(item) for item in evaluation_reports
        )
        return cls(**fields)


def animal_parsing_runtime_bundle(
    manifest: AnimalParsingRuntimeManifest,
) -> dict[str, Any]:
    return {
        "schema_version": (
            LEGACY_BUNDLE_SCHEMA
            if manifest.schema_version == LEGACY_MANIFEST_SCHEMA
            else BUNDLE_SCHEMA
        ),
        "manifest_sha256": manifest.manifest_sha256,
        "manifest": manifest.to_dict(),
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "INTERPRETATION",
    "LEGACY_BUNDLE_SCHEMA",
    "LEGACY_MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA",
    "QUALIFICATION",
    "SUPPORTED_BUNDLE_SCHEMAS",
    "AnimalParsingRuntimeManifest",
    "ParsingEvaluationBinding",
    "animal_parsing_runtime_bundle",
]
