"""Deterministic, bounded pixel transforms for visual shortcut controls.

The executor is an offline audit path.  It intentionally spends extra decode
work to verify every produced pixel against the declared transform equation.
It is not part of the online identity-inference hot path.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from data.acquisition import sha256_file
from data.crop_export import ImageProbe, probe_still_image
from evaluation.controls import (
    ControlMaskManifest,
    ControlMaskVerification,
    ControlTransformTask,
    MaskReviewStatus,
    MaskRole,
    VisualControlKind,
    VisualControlPolicy,
    control_artifact_token,
    verify_control_mask_files,
)
from evaluation.mask_semantics import MaskSemanticVerification
from evaluation.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    PairArtifactVerification,
    verify_pair_artifact_files,
)
from foundation.provenance import content_sha256

SUPPORTED_SEMANTICS_VERSION = "cvi.visual_control_transform.v1"
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True, slots=True)
class ControlTransformConfig:
    kind: VisualControlKind
    semantics_version: str = SUPPORTED_SEMANTICS_VERSION
    neutral_value: int = 0
    blur_sigma_fraction_of_min_edge: float | None = None
    minimum_blur_sigma_pixels: float | None = None
    maximum_blur_sigma_pixels: float | None = None
    blur_steps: int | None = None
    schema_version: str = "cvi.control_transform_config.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_transform_config.v1":
            raise ValueError("unsupported control transform config schema")
        if self.kind is VisualControlKind.ORIGINAL:
            raise ValueError("ORIGINAL does not require a transform config")
        if self.semantics_version != SUPPORTED_SEMANTICS_VERSION:
            raise ValueError("unsupported visual-control semantics version")
        if (
            isinstance(self.neutral_value, bool)
            or not isinstance(self.neutral_value, int)
            or self.neutral_value != 0
        ):
            raise ValueError("neutral pixel value is fixed to zero")
        if self.kind is VisualControlKind.BODY_BLURRED:
            for name in (
                "blur_sigma_fraction_of_min_edge",
                "minimum_blur_sigma_pixels",
                "maximum_blur_sigma_pixels",
            ):
                value = getattr(self, name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        f"BODY_BLURRED {name} must be finite and positive"
                    )
            if self.blur_sigma_fraction_of_min_edge > 1:
                raise ValueError(
                    "blur sigma fraction must not exceed one min edge"
                )
            if self.maximum_blur_sigma_pixels > 1024:
                raise ValueError("maximum blur sigma exceeds FFmpeg limit")
            if (
                self.minimum_blur_sigma_pixels
                > self.maximum_blur_sigma_pixels
            ):
                raise ValueError("blur sigma bounds are reversed")
            if (
                isinstance(self.blur_steps, bool)
                or not isinstance(self.blur_steps, int)
                or not 1 <= self.blur_steps <= 6
            ):
                raise ValueError(
                    "BODY_BLURRED blur_steps must be an integer in [1, 6]"
                )
        elif any(
            value is not None
            for value in (
                self.blur_sigma_fraction_of_min_edge,
                self.minimum_blur_sigma_pixels,
                self.maximum_blur_sigma_pixels,
                self.blur_steps,
            )
        ):
            raise ValueError(
                "blur parameters are allowed only for BODY_BLURRED"
            )

    @property
    def transform_config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "semantics_version": self.semantics_version,
            "neutral_value": self.neutral_value,
            "blur_sigma_fraction_of_min_edge": (
                self.blur_sigma_fraction_of_min_edge
            ),
            "minimum_blur_sigma_pixels": self.minimum_blur_sigma_pixels,
            "maximum_blur_sigma_pixels": self.maximum_blur_sigma_pixels,
            "blur_steps": self.blur_steps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlTransformConfig:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "kind",
                "semantics_version",
                "neutral_value",
                "blur_sigma_fraction_of_min_edge",
                "minimum_blur_sigma_pixels",
                "maximum_blur_sigma_pixels",
                "blur_steps",
            },
            "control transform config",
        )
        return cls(
            schema_version=payload["schema_version"],
            kind=VisualControlKind(payload["kind"]),
            semantics_version=payload["semantics_version"],
            neutral_value=payload["neutral_value"],
            blur_sigma_fraction_of_min_edge=payload[
                "blur_sigma_fraction_of_min_edge"
            ],
            minimum_blur_sigma_pixels=payload[
                "minimum_blur_sigma_pixels"
            ],
            maximum_blur_sigma_pixels=payload[
                "maximum_blur_sigma_pixels"
            ],
            blur_steps=payload["blur_steps"],
        )


@dataclass(frozen=True, slots=True)
class ControlTransformConfigManifest:
    configs: tuple[ControlTransformConfig, ...]
    schema_version: str = "cvi.control_transform_config_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_transform_config_manifest.v1":
            raise ValueError(
                "unsupported control transform config manifest schema"
            )
        if not self.configs:
            raise ValueError("transform config manifest must not be empty")
        kinds = tuple(config.kind for config in self.configs)
        if len(kinds) != len(set(kinds)):
            raise ValueError("transform configs must be unique by kind")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "configs": [config.to_dict() for config in self.configs],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> ControlTransformConfigManifest:
        _require_exact_keys(
            payload,
            {"schema_version", "configs"},
            "control transform config manifest",
        )
        configs = payload["configs"]
        if not isinstance(configs, list):
            raise TypeError("control transform configs must be a list")
        return cls(
            schema_version=payload["schema_version"],
            configs=tuple(
                ControlTransformConfig.from_dict(item) for item in configs
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlTransformExecutionPolicy:
    timeout_seconds_per_process: float = 30.0
    maximum_tasks: int = 10_000
    maximum_source_file_bytes: int = 67_108_864
    maximum_source_pixels: int = 16_777_216
    maximum_total_task_pixels: int = 1_000_000_000
    maximum_artifact_bytes: int = 67_108_864
    maximum_total_output_bytes: int = 8_589_934_592
    maximum_validation_raw_bytes_per_group: int = 536_870_912
    validation_chunk_pixels: int = 262_144
    rgb_pixel_format: str = "rgb24"
    ir_pixel_format: str = "gray"
    output_media_type: str = "image/png"
    png_prediction: str = "mixed"
    schema_version: str = "cvi.control_transform_execution_policy.v1"

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "cvi.control_transform_execution_policy.v1"
        ):
            raise ValueError(
                "unsupported control transform execution policy schema"
            )
        if (
            isinstance(self.timeout_seconds_per_process, bool)
            or not isinstance(
                self.timeout_seconds_per_process,
                (int, float),
            )
            or not isfinite(self.timeout_seconds_per_process)
            or self.timeout_seconds_per_process <= 0
        ):
            raise ValueError("process timeout must be finite and positive")
        for name in (
            "maximum_tasks",
            "maximum_source_file_bytes",
            "maximum_source_pixels",
            "maximum_total_task_pixels",
            "maximum_artifact_bytes",
            "maximum_total_output_bytes",
            "maximum_validation_raw_bytes_per_group",
            "validation_chunk_pixels",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.rgb_pixel_format != "rgb24":
            raise ValueError("RGB control format is fixed to rgb24")
        if self.ir_pixel_format != "gray":
            raise ValueError("IR control format is fixed to gray")
        if self.output_media_type != "image/png":
            raise ValueError("control output format is fixed to image/png")
        if self.png_prediction != "mixed":
            raise ValueError("PNG prediction is fixed to mixed")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "schema_version": self.schema_version,
            "timeout_seconds_per_process": (
                self.timeout_seconds_per_process
            ),
            "maximum_tasks": self.maximum_tasks,
            "maximum_source_file_bytes": self.maximum_source_file_bytes,
            "maximum_source_pixels": self.maximum_source_pixels,
            "maximum_total_task_pixels": self.maximum_total_task_pixels,
            "maximum_artifact_bytes": self.maximum_artifact_bytes,
            "maximum_total_output_bytes": self.maximum_total_output_bytes,
            "maximum_validation_raw_bytes_per_group": (
                self.maximum_validation_raw_bytes_per_group
            ),
            "validation_chunk_pixels": self.validation_chunk_pixels,
            "rgb_pixel_format": self.rgb_pixel_format,
            "ir_pixel_format": self.ir_pixel_format,
            "output_media_type": self.output_media_type,
            "png_prediction": self.png_prediction,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> ControlTransformExecutionPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "timeout_seconds_per_process",
                "maximum_tasks",
                "maximum_source_file_bytes",
                "maximum_source_pixels",
                "maximum_total_task_pixels",
                "maximum_artifact_bytes",
                "maximum_total_output_bytes",
                "maximum_validation_raw_bytes_per_group",
                "validation_chunk_pixels",
                "rgb_pixel_format",
                "ir_pixel_format",
                "output_media_type",
                "png_prediction",
            },
            "control transform execution policy",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlArtifactManifest:
    transform_tasks_sha256: str
    transform_config_manifest_sha256: str
    entries: tuple[PairArtifactEntry, ...]
    schema_version: str = "cvi.control_artifact_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_artifact_manifest.v1":
            raise ValueError("unsupported control artifact manifest schema")
        _validate_sha256(
            self.transform_tasks_sha256,
            "transform_tasks_sha256",
        )
        _validate_sha256(
            self.transform_config_manifest_sha256,
            "transform_config_manifest_sha256",
        )
        if not self.entries:
            raise ValueError("control artifact manifest must not be empty")
        tokens = tuple(entry.artifact_token for entry in self.entries)
        paths = tuple(entry.relative_path for entry in self.entries)
        if len(tokens) != len(set(tokens)):
            raise ValueError("control artifact tokens must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("control artifact paths must be unique")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transform_tasks_sha256": self.transform_tasks_sha256,
            "transform_config_manifest_sha256": (
                self.transform_config_manifest_sha256
            ),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlArtifactManifest:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "transform_tasks_sha256",
                "transform_config_manifest_sha256",
                "entries",
            },
            "control artifact manifest",
        )
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise TypeError("control artifact entries must be a list")
        return cls(
            schema_version=payload["schema_version"],
            transform_tasks_sha256=payload["transform_tasks_sha256"],
            transform_config_manifest_sha256=payload[
                "transform_config_manifest_sha256"
            ],
            entries=tuple(PairArtifactEntry.from_dict(item) for item in entries),
        )


@dataclass(frozen=True, slots=True)
class ControlArtifactVerification:
    artifact_manifest_sha256: str
    verified_files: int
    verified_bytes: int
    schema_version: str = "cvi.control_artifact_verification.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_artifact_verification.v1":
            raise ValueError(
                "unsupported control artifact verification schema"
            )
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
            "schema_version": self.schema_version,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "verified_files": self.verified_files,
            "verified_bytes": self.verified_bytes,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> ControlArtifactVerification:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "artifact_manifest_sha256",
                "verified_files",
                "verified_bytes",
            },
            "control artifact verification",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlTransformCost:
    transform_tasks: int
    unique_base_decodes: int
    unique_mask_decodes: int
    validation_blur_decodes: int
    subprocess_calls: int
    total_task_pixels: int
    output_bytes: int
    peak_validation_raw_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "transform_tasks",
            "unique_base_decodes",
            "unique_mask_decodes",
            "validation_blur_decodes",
            "subprocess_calls",
            "total_task_pixels",
            "output_bytes",
            "peak_validation_raw_bytes",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "transform_tasks": self.transform_tasks,
            "unique_base_decodes": self.unique_base_decodes,
            "unique_mask_decodes": self.unique_mask_decodes,
            "validation_blur_decodes": self.validation_blur_decodes,
            "subprocess_calls": self.subprocess_calls,
            "total_task_pixels": self.total_task_pixels,
            "output_bytes": self.output_bytes,
            "peak_validation_raw_bytes": self.peak_validation_raw_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlTransformCost:
        _require_exact_keys(
            payload,
            {
                "transform_tasks",
                "unique_base_decodes",
                "unique_mask_decodes",
                "validation_blur_decodes",
                "subprocess_calls",
                "total_task_pixels",
                "output_bytes",
                "peak_validation_raw_bytes",
            },
            "control transform cost",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ControlTransformReceipt:
    plan_sha256: str
    scoring_requests_sha256: str
    transform_tasks_sha256: str
    base_artifact_manifest_sha256: str
    base_artifact_verification_sha256: str
    mask_manifest_sha256: str
    mask_verification_sha256: str
    mask_semantic_verification_sha256: str
    transform_config_manifest_sha256: str
    execution_policy_sha256: str
    ffmpeg_version: str
    artifact_manifest: ControlArtifactManifest
    verification: ControlArtifactVerification
    cost: ControlTransformCost
    schema_version: str = "cvi.control_transform_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.control_transform_receipt.v1":
            raise ValueError("unsupported control transform receipt schema")
        for name in (
            "plan_sha256",
            "scoring_requests_sha256",
            "transform_tasks_sha256",
            "base_artifact_manifest_sha256",
            "base_artifact_verification_sha256",
            "mask_manifest_sha256",
            "mask_verification_sha256",
            "mask_semantic_verification_sha256",
            "transform_config_manifest_sha256",
            "execution_policy_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not isinstance(self.ffmpeg_version, str) or not (
            self.ffmpeg_version.strip()
        ):
            raise ValueError("ffmpeg_version must be non-empty")
        if (
            self.artifact_manifest.transform_tasks_sha256
            != self.transform_tasks_sha256
            or (
                self.artifact_manifest.transform_config_manifest_sha256
                != self.transform_config_manifest_sha256
            )
        ):
            raise ValueError("control receipt output manifest binding mismatch")
        if (
            self.verification.artifact_manifest_sha256
            != self.artifact_manifest.manifest_sha256
            or self.verification.verified_files
            != len(self.artifact_manifest.entries)
            or self.verification.verified_bytes
            != sum(
                entry.byte_size for entry in self.artifact_manifest.entries
            )
        ):
            raise ValueError("control receipt file verification mismatch")
        if (
            self.cost.transform_tasks
            != len(self.artifact_manifest.entries)
            or self.cost.output_bytes
            != self.verification.verified_bytes
        ):
            raise ValueError("control receipt cost does not match artifacts")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "scoring_requests_sha256": self.scoring_requests_sha256,
            "transform_tasks_sha256": self.transform_tasks_sha256,
            "base_artifact_manifest_sha256": (
                self.base_artifact_manifest_sha256
            ),
            "base_artifact_verification_sha256": (
                self.base_artifact_verification_sha256
            ),
            "mask_manifest_sha256": self.mask_manifest_sha256,
            "mask_verification_sha256": self.mask_verification_sha256,
            "mask_semantic_verification_sha256": (
                self.mask_semantic_verification_sha256
            ),
            "transform_config_manifest_sha256": (
                self.transform_config_manifest_sha256
            ),
            "execution_policy_sha256": self.execution_policy_sha256,
            "ffmpeg_version": self.ffmpeg_version,
            "artifact_manifest": self.artifact_manifest.to_dict(),
            "verification": self.verification.to_dict(),
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlTransformReceipt:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "plan_sha256",
                "scoring_requests_sha256",
                "transform_tasks_sha256",
                "base_artifact_manifest_sha256",
                "base_artifact_verification_sha256",
                "mask_manifest_sha256",
                "mask_verification_sha256",
                "mask_semantic_verification_sha256",
                "transform_config_manifest_sha256",
                "execution_policy_sha256",
                "ffmpeg_version",
                "artifact_manifest",
                "verification",
                "cost",
            },
            "control transform receipt",
        )
        return cls(
            schema_version=payload["schema_version"],
            plan_sha256=payload["plan_sha256"],
            scoring_requests_sha256=payload[
                "scoring_requests_sha256"
            ],
            transform_tasks_sha256=payload["transform_tasks_sha256"],
            base_artifact_manifest_sha256=payload[
                "base_artifact_manifest_sha256"
            ],
            base_artifact_verification_sha256=payload[
                "base_artifact_verification_sha256"
            ],
            mask_manifest_sha256=payload["mask_manifest_sha256"],
            mask_verification_sha256=payload[
                "mask_verification_sha256"
            ],
            mask_semantic_verification_sha256=payload[
                "mask_semantic_verification_sha256"
            ],
            transform_config_manifest_sha256=payload[
                "transform_config_manifest_sha256"
            ],
            execution_policy_sha256=payload[
                "execution_policy_sha256"
            ],
            ffmpeg_version=payload["ffmpeg_version"],
            artifact_manifest=ControlArtifactManifest.from_dict(
                payload["artifact_manifest"]
            ),
            verification=ControlArtifactVerification.from_dict(
                payload["verification"]
            ),
            cost=ControlTransformCost.from_dict(payload["cost"]),
        )


def control_transform_tasks_from_payload(
    payload: dict[str, Any],
) -> tuple[str, str, tuple[ControlTransformTask, ...]]:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "plan_sha256",
            "scoring_requests_sha256",
            "tasks",
        },
        "visual control transform tasks",
    )
    if payload["schema_version"] != "cvi.visual_control_transform_tasks.v1":
        raise ValueError("unsupported visual control transform task schema")
    _validate_sha256(payload["plan_sha256"], "plan_sha256")
    _validate_sha256(
        payload["scoring_requests_sha256"],
        "scoring_requests_sha256",
    )
    tasks = payload["tasks"]
    if not isinstance(tasks, list):
        raise TypeError("visual control transform tasks must be a list")
    return (
        payload["plan_sha256"],
        payload["scoring_requests_sha256"],
        tuple(ControlTransformTask.from_dict(item) for item in tasks),
    )


def validate_control_policy_configs(
    control_policy: VisualControlPolicy,
    config_manifest: ControlTransformConfigManifest,
) -> None:
    """Require every executable recipe to name its exact pixel config."""

    executable_recipes = {
        recipe.kind: recipe
        for recipe in control_policy.recipes
        if recipe.kind is not VisualControlKind.ORIGINAL
    }
    configs = {config.kind: config for config in config_manifest.configs}
    if set(executable_recipes) != set(configs):
        raise ValueError(
            "control policy and transform config kinds must match exactly"
        )
    for kind, recipe in executable_recipes.items():
        config = configs[kind]
        if (
            recipe.transform_config_sha256
            != config.transform_config_sha256
            or recipe.semantics_version != config.semantics_version
        ):
            raise ValueError(
                f"control recipe does not bind executable config: {kind.value}"
            )


def build_control_transform_command(
    *,
    base: Path,
    mask: Path,
    destination: Path,
    config: ControlTransformConfig,
    pixel_format: str,
    width: int,
    height: int,
) -> tuple[str, ...]:
    """Build the fixed single-frame FFmpeg transform command."""

    base_filter = f"format={pixel_format},setsar=1"
    mask_filter = f"format={pixel_format},setsar=1"
    zero_filter = (
        "lutrgb=r=0:g=0:b=0"
        if pixel_format == "rgb24"
        else "lutyuv=y=0"
    )
    if config.kind is VisualControlKind.MASK_ONLY:
        graph = f"[1:v]{mask_filter}[out]"
    elif config.kind is VisualControlKind.BODY_BLURRED:
        sigma = _effective_blur_sigma(config, width, height)
        graph = (
            f"[0:v]{base_filter},split=2[base][blur_input];"
            f"[blur_input]gblur=sigma={_format_float(sigma)}:"
            f"steps={config.blur_steps}[blur];"
            f"[1:v]{mask_filter}[mask];"
            "[base][blur][mask]maskedmerge[out]"
        )
    else:
        graph = (
            f"[0:v]{base_filter},split=2[base][zero_input];"
            f"[zero_input]{zero_filter}[zero];"
            f"[1:v]{mask_filter}[mask];"
        )
        if config.kind in (
            VisualControlKind.DOG_ONLY,
            VisualControlKind.ACCESSORY_ONLY,
        ):
            graph += "[zero][base][mask]maskedmerge[out]"
        elif config.kind in (
            VisualControlKind.BACKGROUND_ONLY,
            VisualControlKind.ACCESSORY_MASKED,
        ):
            graph += "[base][zero][mask]maskedmerge[out]"
        else:
            raise ValueError("unsupported control transform kind")
    return (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
        "-i",
        str(base),
        "-i",
        str(mask),
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        pixel_format,
        "-pred",
        "mixed",
        "-f",
        "image2",
        str(destination),
    )


def execute_control_transforms(
    *,
    plan_sha256: str,
    scoring_requests_sha256: str,
    tasks: tuple[ControlTransformTask, ...],
    base_root: Path,
    base_manifest: PairArtifactManifest,
    base_verification: PairArtifactVerification,
    mask_root: Path,
    mask_manifest: ControlMaskManifest,
    mask_verification: ControlMaskVerification,
    mask_semantic_verification: MaskSemanticVerification,
    config_manifest: ControlTransformConfigManifest,
    policy: ControlTransformExecutionPolicy,
    output_directory: Path,
) -> ControlTransformReceipt:
    """Execute and independently verify a closed set of transform tasks."""

    _validate_sha256(plan_sha256, "plan_sha256")
    _validate_sha256(
        scoring_requests_sha256,
        "scoring_requests_sha256",
    )
    output_root = _empty_real_directory(
        output_directory,
        "control output",
    )
    base_root_resolved = _real_directory(base_root, "base artifact")
    mask_root_resolved = _real_directory(mask_root, "mask artifact")
    current_base = verify_pair_artifact_files(
        base_root_resolved,
        base_manifest,
    )
    current_masks = verify_control_mask_files(
        mask_root_resolved,
        mask_manifest,
    )
    if current_base != base_verification:
        raise ValueError("base artifacts changed before control transform")
    if current_masks != mask_verification:
        raise ValueError("mask artifacts changed before control transform")
    _validate_semantic_receipt(
        base_manifest,
        base_verification,
        mask_manifest,
        mask_verification,
        mask_semantic_verification,
    )
    if not tasks:
        raise ValueError("control transform tasks must not be empty")
    if len(tasks) > policy.maximum_tasks:
        raise ValueError("control task count exceeds maximum_tasks")
    task_tokens = tuple(task.control_artifact_token for task in tasks)
    if len(task_tokens) != len(set(task_tokens)):
        raise ValueError("control transform task tokens must be unique")
    if any(not _SAFE_TOKEN.fullmatch(token) for token in task_tokens):
        raise ValueError("control transform token is not a safe filename stem")

    base_by_token = {
        entry.artifact_token: entry for entry in base_manifest.entries
    }
    mask_by_base = {
        entry.base_artifact_token: entry for entry in mask_manifest.entries
    }
    config_by_kind = {config.kind: config for config in config_manifest.configs}
    task_kinds = {task.control_kind for task in tasks}
    if set(config_by_kind) != task_kinds:
        raise ValueError(
            "transform config kinds must exactly match transform task kinds"
        )

    tasks_by_base: dict[str, list[ControlTransformTask]] = {}
    probes: dict[str, ImageProbe] = {}
    total_task_pixels = 0
    peak_validation_raw_bytes = 0
    for task in tasks:
        probe = _validate_task_binding(
            task,
            base_root_resolved,
            base_by_token,
            mask_root_resolved,
            mask_by_base,
            mask_semantic_verification,
            config_by_kind,
            policy,
        )
        probes[task.base_artifact_token] = probe
        pixels = probe.width * probe.height
        total_task_pixels += pixels
        if total_task_pixels > policy.maximum_total_task_pixels:
            raise ValueError(
                "control transforms exceed maximum_total_task_pixels"
            )
        tasks_by_base.setdefault(task.base_artifact_token, []).append(task)

    ffmpeg_version = _ffmpeg_version()
    if ffmpeg_version != mask_semantic_verification.ffmpeg_version:
        raise ValueError(
            "FFmpeg version differs from mask semantic verification"
        )
    transform_tasks_sha256 = content_sha256(
        [task.to_dict() for task in tasks]
    )
    entries: list[PairArtifactEntry] = []
    output_bytes = 0
    base_decodes = 0
    mask_decodes = 0
    blur_decodes = 0
    subprocess_calls = 0

    with TemporaryDirectory(
        prefix=".cvi-control-transform-",
        dir=output_root.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        for base_token in sorted(tasks_by_base):
            grouped_tasks = sorted(
                tasks_by_base[base_token],
                key=lambda item: item.control_artifact_token,
            )
            base_entry = base_by_token[base_token]
            base_path = base_root_resolved / base_entry.relative_path
            probe = probes[base_token]
            channels = 3 if probe.pixel_format == "rgb24" else 1
            pixel_count = probe.width * probe.height
            roles = tuple(
                sorted(
                    {
                        role
                        for task in grouped_tasks
                        for role, _, _ in task.mask_artifacts
                    },
                    key=lambda role: role.value,
                )
            )
            has_blur = any(
                task.control_kind is VisualControlKind.BODY_BLURRED
                for task in grouped_tasks
            )
            raw_peak = pixel_count * (
                channels
                + len(roles)
                + channels
                + (channels if has_blur else 0)
            )
            if (
                raw_peak
                > policy.maximum_validation_raw_bytes_per_group
            ):
                raise ValueError(
                    "control validation exceeds raw-byte group cap"
                )
            peak_validation_raw_bytes = max(
                peak_validation_raw_bytes,
                raw_peak,
            )
            base_raw = temporary_root / f"{base_token}.base.raw"
            _run(
                _raw_decode_command(
                    base_path,
                    base_raw,
                    probe.pixel_format,
                ),
                policy,
            )
            base_decodes += 1
            subprocess_calls += 1
            raw_masks: dict[MaskRole, Path] = {}
            mask_paths: dict[MaskRole, Path] = {}
            try:
                manifest_entry = mask_by_base[base_token]
                for role in roles:
                    evidence = manifest_entry.mask_for(role)
                    if evidence is None:
                        raise RuntimeError("validated task mask disappeared")
                    source = mask_root_resolved / evidence.relative_path
                    destination = (
                        temporary_root
                        / f"{base_token}.{role.value.casefold()}.raw"
                    )
                    _run(
                        _raw_decode_command(source, destination, "gray"),
                        policy,
                    )
                    raw_masks[role] = destination
                    mask_paths[role] = source
                    mask_decodes += 1
                    subprocess_calls += 1

                for task in grouped_tasks:
                    config = config_by_kind[task.control_kind]
                    operative_role = _operative_mask_role(task.control_kind)
                    destination = (
                        temporary_root
                        / f"{task.control_artifact_token}.png"
                    )
                    _run(
                        build_control_transform_command(
                            base=base_path,
                            mask=mask_paths[operative_role],
                            destination=destination,
                            config=config,
                            pixel_format=probe.pixel_format,
                            width=probe.width,
                            height=probe.height,
                        ),
                        policy,
                    )
                    subprocess_calls += 1
                    os.chmod(destination, 0o600)
                    output_probe = probe_still_image(destination)
                    _validate_output_probe(output_probe, probe)
                    artifact_size = destination.stat().st_size
                    if artifact_size > policy.maximum_artifact_bytes:
                        raise ValueError(
                            "control artifact exceeds maximum_artifact_bytes"
                        )
                    output_bytes += artifact_size
                    if output_bytes > policy.maximum_total_output_bytes:
                        raise ValueError(
                            "control output exceeds total byte cap"
                        )
                    output_raw = (
                        temporary_root
                        / f"{task.control_artifact_token}.output.raw"
                    )
                    _run(
                        _raw_decode_command(
                            destination,
                            output_raw,
                            probe.pixel_format,
                        ),
                        policy,
                    )
                    subprocess_calls += 1
                    blur_raw: Path | None = None
                    try:
                        if (
                            task.control_kind
                            is VisualControlKind.BODY_BLURRED
                        ):
                            blur_raw = (
                                temporary_root
                                / f"{task.control_artifact_token}.blur.raw"
                            )
                            _run(
                                _blur_raw_command(
                                    base_path,
                                    blur_raw,
                                    probe.pixel_format,
                                    config,
                                    probe.width,
                                    probe.height,
                                ),
                                policy,
                            )
                            blur_decodes += 1
                            subprocess_calls += 1
                        _verify_transform_equation(
                            kind=task.control_kind,
                            base_raw=base_raw,
                            mask_raw=raw_masks[operative_role],
                            output_raw=output_raw,
                            blur_raw=blur_raw,
                            pixel_count=pixel_count,
                            channels=channels,
                            chunk_pixels=policy.validation_chunk_pixels,
                        )
                    finally:
                        output_raw.unlink(missing_ok=True)
                        if blur_raw is not None:
                            blur_raw.unlink(missing_ok=True)
                    entries.append(
                        PairArtifactEntry(
                            artifact_token=task.control_artifact_token,
                            relative_path=destination.name,
                            content_sha256=sha256_file(destination),
                            byte_size=artifact_size,
                            media_type=policy.output_media_type,
                        )
                    )
            finally:
                base_raw.unlink(missing_ok=True)
                for raw_mask in raw_masks.values():
                    raw_mask.unlink(missing_ok=True)

        artifact_manifest = ControlArtifactManifest(
            transform_tasks_sha256=transform_tasks_sha256,
            transform_config_manifest_sha256=(
                config_manifest.manifest_sha256
            ),
            entries=tuple(
                sorted(entries, key=lambda item: item.artifact_token)
            ),
        )
        verify_control_artifact_files(temporary_root, artifact_manifest)
        if (
            verify_pair_artifact_files(base_root_resolved, base_manifest)
            != base_verification
        ):
            raise RuntimeError(
                "base artifacts changed during control transform"
            )
        if (
            verify_control_mask_files(mask_root_resolved, mask_manifest)
            != mask_verification
        ):
            raise RuntimeError(
                "mask artifacts changed during control transform"
            )
        created: list[Path] = []
        try:
            for entry in artifact_manifest.entries:
                source = temporary_root / entry.relative_path
                destination = output_root / entry.relative_path
                os.link(source, destination)
                created.append(destination)
            verification = verify_control_artifact_files(
                output_root,
                artifact_manifest,
            )
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise

    cost = ControlTransformCost(
        transform_tasks=len(tasks),
        unique_base_decodes=base_decodes,
        unique_mask_decodes=mask_decodes,
        validation_blur_decodes=blur_decodes,
        subprocess_calls=subprocess_calls,
        total_task_pixels=total_task_pixels,
        output_bytes=output_bytes,
        peak_validation_raw_bytes=peak_validation_raw_bytes,
    )
    return ControlTransformReceipt(
        plan_sha256=plan_sha256,
        scoring_requests_sha256=scoring_requests_sha256,
        transform_tasks_sha256=transform_tasks_sha256,
        base_artifact_manifest_sha256=base_manifest.manifest_sha256,
        base_artifact_verification_sha256=content_sha256(
            base_verification.to_dict()
        ),
        mask_manifest_sha256=mask_manifest.manifest_sha256,
        mask_verification_sha256=content_sha256(
            mask_verification.to_dict()
        ),
        mask_semantic_verification_sha256=(
            mask_semantic_verification.verification_sha256
        ),
        transform_config_manifest_sha256=config_manifest.manifest_sha256,
        execution_policy_sha256=policy.policy_sha256,
        ffmpeg_version=ffmpeg_version,
        artifact_manifest=artifact_manifest,
        verification=verification,
        cost=cost,
    )


def _validate_task_binding(
    task: ControlTransformTask,
    base_root: Path,
    base_by_token: dict[str, PairArtifactEntry],
    mask_root: Path,
    mask_by_base: dict[str, Any],
    semantic: MaskSemanticVerification,
    config_by_kind: dict[VisualControlKind, ControlTransformConfig],
    policy: ControlTransformExecutionPolicy,
) -> ImageProbe:
    if task.base_artifact_token not in base_by_token:
        raise ValueError("transform task references an unknown base artifact")
    if task.base_artifact_token not in mask_by_base:
        raise ValueError("transform task has no mask-manifest entry")
    config = config_by_kind.get(task.control_kind)
    if config is None:
        raise ValueError("transform task has no matching config")
    if (
        task.transform_config_sha256
        != config.transform_config_sha256
        or task.semantics_version != config.semantics_version
    ):
        raise ValueError("transform task config or semantics hash mismatch")
    base_entry = base_by_token[task.base_artifact_token]
    expected_token = control_artifact_token(
        base_content_sha256=base_entry.content_sha256,
        kind=task.control_kind,
        transform_config_sha256=task.transform_config_sha256,
        semantics_version=task.semantics_version,
        mask_artifacts=task.mask_artifacts,
    )
    if task.control_artifact_token != expected_token:
        raise ValueError("control artifact token is not content-addressed")
    if base_entry.byte_size > policy.maximum_source_file_bytes:
        raise ValueError("base artifact exceeds source-file byte cap")
    probe = probe_still_image(base_root / base_entry.relative_path)
    _validate_base_probe(probe, policy)
    pixels = probe.width * probe.height
    if pixels > policy.maximum_source_pixels:
        raise ValueError("base artifact exceeds source-pixel cap")

    mask_entry = mask_by_base[task.base_artifact_token]
    semantic_by_base = {
        entry.base_artifact_token: entry for entry in semantic.entries
    }
    semantic_entry = semantic_by_base.get(task.base_artifact_token)
    if semantic_entry is None:
        raise ValueError("task base lacks mask semantic verification")
    if (
        semantic_entry.width != probe.width
        or semantic_entry.height != probe.height
    ):
        raise ValueError("semantic receipt dimensions differ from base")
    semantic_roles = {item.role: item for item in semantic_entry.masks}
    for role, token, digest in task.mask_artifacts:
        evidence = mask_entry.mask_for(role)
        if (
            evidence is None
            or evidence.review_status is not MaskReviewStatus.VERIFIED
            or evidence.artifact_token != token
            or evidence.content_sha256 != digest
        ):
            raise ValueError("task mask binding differs from verified mask")
        if evidence.byte_size > policy.maximum_source_file_bytes:
            raise ValueError("mask artifact exceeds source-file byte cap")
        if role not in semantic_roles:
            raise ValueError("task mask lacks pixel semantic verification")
        stats = semantic_roles[role]
        if stats.total_pixels != pixels or stats.foreground_pixels <= 0:
            raise ValueError("task mask semantic statistics are invalid")
        mask_probe = probe_still_image(mask_root / evidence.relative_path)
        if (
            mask_probe.format_name != "png_pipe"
            or mask_probe.pixel_format != "gray"
            or mask_probe.width != probe.width
            or mask_probe.height != probe.height
            or mask_probe.stream_tags
            or mask_probe.format_tags
        ):
            raise ValueError("task mask probe violates verified contract")
    return probe


def _validate_semantic_receipt(
    base_manifest: PairArtifactManifest,
    base_verification: PairArtifactVerification,
    mask_manifest: ControlMaskManifest,
    mask_verification: ControlMaskVerification,
    semantic: MaskSemanticVerification,
) -> None:
    expected = (
        base_manifest.manifest_sha256,
        content_sha256(base_verification.to_dict()),
        mask_manifest.manifest_sha256,
        content_sha256(mask_verification.to_dict()),
    )
    actual = (
        semantic.base_artifact_manifest_sha256,
        semantic.base_artifact_verification_sha256,
        semantic.mask_manifest_sha256,
        semantic.mask_file_verification_sha256,
    )
    if actual != expected:
        raise ValueError(
            "mask semantic verification is stale or bound to other inputs"
        )


def _validate_base_probe(
    image: ImageProbe,
    policy: ControlTransformExecutionPolicy,
) -> None:
    if image.format_name != "png_pipe":
        raise ValueError("control base artifact must be PNG")
    if image.pixel_format not in {
        policy.rgb_pixel_format,
        policy.ir_pixel_format,
    }:
        raise ValueError("control base pixel format must be rgb24 or gray")
    if image.stream_tags or image.format_tags:
        raise ValueError("control base artifact must not contain metadata")


def _validate_output_probe(output: ImageProbe, base: ImageProbe) -> None:
    if output.format_name != "png_pipe":
        raise ValueError("control output is not PNG")
    if (
        output.width != base.width
        or output.height != base.height
        or output.pixel_format != base.pixel_format
    ):
        raise ValueError("control output geometry or pixel format changed")
    if output.stream_tags or output.format_tags:
        raise ValueError("control output contains metadata")


def _operative_mask_role(kind: VisualControlKind) -> MaskRole:
    if kind in (
        VisualControlKind.ACCESSORY_ONLY,
        VisualControlKind.ACCESSORY_MASKED,
    ):
        return MaskRole.ACCESSORY
    return MaskRole.DOG


def _raw_decode_command(
    source: Path,
    destination: Path,
    pixel_format: str,
) -> tuple[str, ...]:
    return (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        f"format={pixel_format}",
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        str(destination),
    )


def _blur_raw_command(
    source: Path,
    destination: Path,
    pixel_format: str,
    config: ControlTransformConfig,
    width: int,
    height: int,
) -> tuple[str, ...]:
    sigma = _effective_blur_sigma(config, width, height)
    return (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        (
            f"format={pixel_format},gblur="
            f"sigma={_format_float(sigma)}:"
            f"steps={config.blur_steps}"
        ),
        "-frames:v",
        "1",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        str(destination),
    )


def _verify_transform_equation(
    *,
    kind: VisualControlKind,
    base_raw: Path,
    mask_raw: Path,
    output_raw: Path,
    blur_raw: Path | None,
    pixel_count: int,
    channels: int,
    chunk_pixels: int,
) -> None:
    _require_file_size(base_raw, pixel_count * channels)
    _require_file_size(mask_raw, pixel_count)
    _require_file_size(output_raw, pixel_count * channels)
    if kind is VisualControlKind.BODY_BLURRED:
        if blur_raw is None:
            raise RuntimeError("BODY_BLURRED validation lacks blur reference")
        _require_file_size(blur_raw, pixel_count * channels)
    elif blur_raw is not None:
        raise RuntimeError("unexpected blur reference for non-blur control")

    remaining = pixel_count
    with (
        base_raw.open("rb") as base_handle,
        mask_raw.open("rb") as mask_handle,
        output_raw.open("rb") as output_handle,
        (
            blur_raw.open("rb")
            if blur_raw is not None
            else _NullBinaryReader()
        ) as blur_handle,
    ):
        pixel_offset = 0
        while remaining:
            count = min(remaining, chunk_pixels)
            base = _read_exact(base_handle, count * channels)
            mask = _read_exact(mask_handle, count)
            output = _read_exact(output_handle, count * channels)
            blur = (
                _read_exact(blur_handle, count * channels)
                if blur_raw is not None
                else b""
            )
            _verify_chunk(
                kind=kind,
                base=base,
                mask=mask,
                output=output,
                blur=blur,
                channels=channels,
                pixel_offset=pixel_offset,
            )
            remaining -= count
            pixel_offset += count


def _verify_chunk(
    *,
    kind: VisualControlKind,
    base: bytes,
    mask: bytes,
    output: bytes,
    blur: bytes,
    channels: int,
    pixel_offset: int,
) -> None:
    if any(value not in (0, 255) for value in mask):
        raise ValueError("control mask changed or is not binary")
    start = 0
    while start < len(mask):
        value = mask[start]
        opposite = b"\xff" if value == 0 else b"\x00"
        end = mask.find(opposite, start + 1)
        if end < 0:
            end = len(mask)
        byte_start = start * channels
        byte_end = end * channels
        actual = output[byte_start:byte_end]
        if kind is VisualControlKind.MASK_ONLY:
            expected = bytes((value,)) * (byte_end - byte_start)
        elif kind in (
            VisualControlKind.DOG_ONLY,
            VisualControlKind.ACCESSORY_ONLY,
        ):
            expected = (
                base[byte_start:byte_end]
                if value == 255
                else bytes(byte_end - byte_start)
            )
        elif kind in (
            VisualControlKind.BACKGROUND_ONLY,
            VisualControlKind.ACCESSORY_MASKED,
        ):
            expected = (
                base[byte_start:byte_end]
                if value == 0
                else bytes(byte_end - byte_start)
            )
        elif kind is VisualControlKind.BODY_BLURRED:
            expected = (
                blur[byte_start:byte_end]
                if value == 255
                else base[byte_start:byte_end]
            )
        else:
            raise ValueError("unsupported transform-equation kind")
        if actual != expected:
            raise ValueError(
                "control output violates transform equation near pixel "
                f"{pixel_offset + start}"
            )
        start = end


class _NullBinaryReader:
    def __enter__(self) -> _NullBinaryReader:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        return b""


def _read_exact(handle: Any, expected: int) -> bytes:
    data = handle.read(expected)
    if len(data) != expected:
        raise RuntimeError("raw validation input changed during scan")
    return data


def _require_file_size(path: Path, expected: int) -> None:
    if path.stat().st_size != expected:
        raise ValueError("decoded raw byte count differs from dimensions")


def verify_control_artifact_files(
    root: Path,
    manifest: ControlArtifactManifest,
) -> ControlArtifactVerification:
    resolved = _real_directory(root, "control artifact")
    directory_entries = tuple(resolved.iterdir())
    if any(entry.is_symlink() for entry in directory_entries):
        raise ValueError("control artifact directory contains a symlink")
    if any(not entry.is_file() for entry in directory_entries):
        raise ValueError("control artifact directory must contain files only")
    expected_names = {entry.relative_path for entry in manifest.entries}
    actual_names = {entry.name for entry in directory_entries}
    if actual_names != expected_names:
        raise ValueError("control artifact directory is not a closed set")
    verified_bytes = 0
    for entry in manifest.entries:
        path = resolved / entry.relative_path
        initial = path.stat()
        if initial.st_size != entry.byte_size:
            raise ValueError("control artifact byte-size mismatch")
        digest = sha256_file(path)
        final = path.stat()
        if (
            initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
        ):
            raise RuntimeError("control artifact changed during verification")
        if digest != entry.content_sha256:
            raise ValueError("control artifact content hash mismatch")
        verified_bytes += entry.byte_size
    return ControlArtifactVerification(
        artifact_manifest_sha256=manifest.manifest_sha256,
        verified_files=len(manifest.entries),
        verified_bytes=verified_bytes,
    )


def _real_directory(path: Path, context: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{context} directory must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _empty_real_directory(path: Path, context: str) -> Path:
    resolved = _real_directory(path, context)
    if any(resolved.iterdir()):
        raise ValueError(f"{context} directory must be empty")
    return resolved


def _run(
    command: tuple[str, ...],
    policy: ControlTransformExecutionPolicy,
) -> None:
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=policy.timeout_seconds_per_process,
    )


def _ffmpeg_version() -> str:
    return subprocess.run(
        ("ffmpeg", "-version"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout.splitlines()[0]


def _format_float(value: float | None) -> str:
    if value is None:
        raise ValueError("blur sigma is required")
    return format(value, ".17g")


def _effective_blur_sigma(
    config: ControlTransformConfig,
    width: int,
    height: int,
) -> float:
    _require_positive_int(width, "width")
    _require_positive_int(height, "height")
    if config.kind is not VisualControlKind.BODY_BLURRED:
        raise ValueError("effective blur sigma requires BODY_BLURRED config")
    raw = config.blur_sigma_fraction_of_min_edge * min(width, height)
    return min(
        max(raw, config.minimum_blur_sigma_pixels),
        config.maximum_blur_sigma_pixels,
    )


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
