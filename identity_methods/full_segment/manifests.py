"""Content-bound manifests for full-segment baseline artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundation.provenance import content_sha256

_METHODS = {"CLASSICAL128", "SEGMENT_FULL_MASKED_GAP128_RESNET18"}
BASELINE_FAMILY_SCHEMA = "cvi.full128_baseline_family.v1"
BASELINE_VARIANTS = (
    ("B0", "CLASSICAL128", "FIT_SCALER_PCA"),
    ("B1", "SEGMENT_FULL_MASKED_GAP128_RESNET18", "RANDOM_SCRATCH"),
    ("B2", "SEGMENT_FULL_MASKED_GAP128_RESNET18", "SUPERVISED_IMAGENET"),
)


def build_baseline_family_manifest() -> dict[str, Any]:
    """Return the fixed three-member Full128 comparison family."""

    return {
        "schema_version": BASELINE_FAMILY_SCHEMA,
        "family_id": "FULL128_B0_B1_B2",
        "embedding_dimension": 128,
        "output_dtype": "float32",
        "output_normalization": "L2",
        "variants": [
            {
                "variant_id": variant_id,
                "method": method,
                "initialization": initialization,
            }
            for variant_id, method, initialization in BASELINE_VARIANTS
        ],
    }


def build_preprocessing_manifest(*, method: str) -> dict[str, Any]:
    if method not in _METHODS:
        raise ValueError("unsupported full-segment preprocessing method")
    if method == "CLASSICAL128":
        preprocessing = {
            "rgb_layout": "HWC_RGB",
            "accepted_rgb_ranges": ["uint8_0_255", "float_0_1"],
            "mask_layout": "HW_BINARY",
            "resize": [64, 64],
            "rgb_interpolation": "AREA",
            "mask_interpolation": "NEAREST",
            "foreground_only": True,
        }
    else:
        preprocessing = {
            "rgb_layout": "BCHW_RGB",
            "rgb_range": [0.0, 1.0],
            "mask_layout": "B1HW_BINARY",
            "minimum_spatial_size": [32, 32],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "background_fill": "NORMALIZATION_MEAN_NEUTRAL",
            "pooling_mask_resize": "AREA",
        }
    return {
        "schema_version": "cvi.full_segment_preprocessing_manifest.v1",
        "method": method,
        "preprocessing": preprocessing,
    }


def build_embedding_manifest(
    *, method: str, component_metadata: Mapping[str, object] | None = None
) -> dict[str, Any]:
    if method not in _METHODS:
        raise ValueError("unsupported full-segment embedding method")
    manifest: dict[str, Any] = {
        "schema_version": "cvi.full_segment_embedding_manifest.v1",
        "method": method,
        "output_dimension": 128,
        "output_dtype": "float32",
        "output_normalization": "L2",
        "output_dimension_semantics": "UNINTERPRETED_COORDINATE_INDEX",
    }
    if method == "CLASSICAL128":
        if not isinstance(component_metadata, Mapping):
            raise ValueError("Classical128 embedding manifest requires component metadata")
        manifest["components"] = dict(component_metadata)
    elif component_metadata is not None:
        raise ValueError("MaskedGAP128 has no semantic component groups")
    return manifest


def build_model_manifest(*, method: str) -> dict[str, Any]:
    """Describe the fixed executable architecture without training-run claims."""

    if method not in _METHODS:
        raise ValueError("unsupported full-segment model method")
    if method == "CLASSICAL128":
        architecture = {
            "descriptor": "EXACT_HOG_HSV_UNIFORM_LBP",
            "scaler": "SKLEARN_STANDARD_SCALER",
            "projection": "SKLEARN_PCA_FULL_SVD_128",
        }
    else:
        architecture = {
            "backbone": "TORCHVISION_RESNET18",
            "feature_channels": 512,
            "pooling": "AREA_MASKED_GLOBAL_AVERAGE",
            "projection": "LINEAR_512_TO_128",
        }
    return {
        "schema_version": "cvi.full128_model_manifest.v1",
        "method": method,
        "architecture": architecture,
    }


def build_checkpoint_manifest(
    *,
    method: str,
    checkpoint_sha256: str,
    preprocessing_manifest: Mapping[str, Any],
    embedding_manifest: Mapping[str, Any],
    initialization: str,
    initialization_sha256: str | None,
    initialization_source_contract_sha256: str | None = None,
    initialization_intake_receipt_sha256: str | None = None,
    initialization_usage_lane: str | None = None,
    fit_partition: str | None = None,
) -> dict[str, Any]:
    _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    if preprocessing_manifest.get("method") != method:
        raise ValueError("checkpoint preprocessing method differs")
    if embedding_manifest.get("method") != method:
        raise ValueError("checkpoint embedding method differs")
    if method == "CLASSICAL128":
        if initialization != "FIT_SCALER_PCA" or initialization_sha256 is not None:
            raise ValueError("Classical128 checkpoint initialization differs")
        if any(
            value is not None
            for value in (
                initialization_source_contract_sha256,
                initialization_intake_receipt_sha256,
                initialization_usage_lane,
            )
        ):
            raise ValueError("Classical128 checkpoint has no pretrained source")
        if fit_partition != "FIT":
            raise ValueError("Classical128 checkpoint must bind estimator fitting to FIT")
    elif method == "SEGMENT_FULL_MASKED_GAP128_RESNET18":
        if initialization not in {"RANDOM_SCRATCH", "SUPERVISED_IMAGENET"}:
            raise ValueError("MaskedGAP128 checkpoint initialization differs")
        if fit_partition is not None:
            raise ValueError("MaskedGAP128 checkpoint has no PCA fit partition")
        if initialization == "SUPERVISED_IMAGENET":
            _require_sha256(initialization_sha256, "initialization_sha256")
            _require_sha256(
                initialization_source_contract_sha256,
                "initialization_source_contract_sha256",
            )
            _require_sha256(
                initialization_intake_receipt_sha256,
                "initialization_intake_receipt_sha256",
            )
            if initialization_usage_lane not in {
                "RESEARCH_ONLY",
                "DEPLOYMENT_CANDIDATE",
            }:
                raise ValueError("supervised initialization usage lane differs")
        elif any(
            value is not None
            for value in (
                initialization_sha256,
                initialization_source_contract_sha256,
                initialization_intake_receipt_sha256,
                initialization_usage_lane,
            )
        ):
            raise ValueError("random scratch initialization has no pretrained source")
    else:
        raise ValueError("unsupported full-segment checkpoint method")
    return {
        "schema_version": "cvi.full_segment_checkpoint_manifest.v2",
        "method": method,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_manifest_sha256": content_sha256(preprocessing_manifest),
        "embedding_manifest_sha256": content_sha256(embedding_manifest),
        "initialization": initialization,
        "initialization_sha256": initialization_sha256,
        "initialization_source_contract_sha256": (
            initialization_source_contract_sha256
        ),
        "initialization_intake_receipt_sha256": (
            initialization_intake_receipt_sha256
        ),
        "initialization_usage_lane": initialization_usage_lane,
        "fit_partition": fit_partition,
    }


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return content_sha256(manifest)


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "BASELINE_FAMILY_SCHEMA",
    "BASELINE_VARIANTS",
    "build_baseline_family_manifest",
    "build_checkpoint_manifest",
    "build_embedding_manifest",
    "build_model_manifest",
    "build_preprocessing_manifest",
    "manifest_sha256",
]
