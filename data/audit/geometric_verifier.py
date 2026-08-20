"""Label-blind geometric/photometric near-duplicate verification.

Candidate generation and identity adjudication are deliberately outside this
module.  The verifier consumes only opaque sample tokens, receipt-bound
canonical RGB pixels, and candidate-channel evidence.  Its thresholds are
synthetic initialization values: a result from this module MUST NOT enter a
protected split until a separately frozen calibration/adjudication receipt
admits the policy.

OpenCV and NumPy are optional runtime dependencies.  Their absence is a typed
``UNRESOLVED`` outcome, never a negative duplicate decision.

The reference path intentionally recomputes SIFT and warped photometry for
each candidate/hypothesis.  It is a correctness oracle, not a throughput
implementation; a receipt-bound feature cache remains required before a full
public audit.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping

from shared.foundation.protected_io import write_private_json_bundle
from shared.foundation.provenance import content_sha256


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PIXEL_HASH_DOMAIN = b"CVI_PIXEL_CANONICAL_RGB_V1\0"
THRESHOLD_STATUS = "INITIALIZATION_ONLY_NOT_CALIBRATED"
INPUT_INTERPRETATION = (
    "LABEL_BLIND_OPAQUE_CANDIDATES_AND_RECEIPT_BOUND_CANONICAL_RGB_ONLY"
)
OUTPUT_INTERPRETATION = (
    "SYNTHETIC_INITIALIZATION_EVIDENCE_NOT_A_FROZEN_DUPLICATE_GRAPH_OR_SPLIT_EDGE"
)
D4_TRANSFORMS = (
    "ORIGINAL",
    "ROT90CCW",
    "ROT180",
    "ROT270CCW",
    "FLIP_X",
    "FLIP_Y",
    "FLIP_PLUS_DIAGONAL",
    "FLIP_MINUS_DIAGONAL",
)
_CANDIDATE_CHANNELS = frozenset({"PHASH", "PDQ", "FROZEN_EMBEDDING"})
_CHANNEL_BINDING = {
    "PHASH": "phash_candidates_sha256",
    "PDQ": "pdq_candidates_sha256",
    "FROZEN_EMBEDDING": "frozen_embedding_candidates_sha256",
}
_IMAGE_BINDING = "image_content_receipts_sha256"


class GeometricDecision(StrEnum):
    GEOMETRIC_CONFIRMED = "GEOMETRIC_CONFIRMED"
    GEOMETRIC_REJECTED = "GEOMETRIC_REJECTED"
    UNRESOLVED = "UNRESOLVED"


class GeometricReason(StrEnum):
    CONFIRMED_GEOMETRIC_AND_PHOTOMETRIC = (
        "CONFIRMED_GEOMETRIC_AND_PHOTOMETRIC"
    )
    REJECTED_PHOTOMETRIC_CONTRADICTION = "REJECTED_PHOTOMETRIC_CONTRADICTION"
    UNRESOLVED_BACKEND_UNAVAILABLE = "UNRESOLVED_BACKEND_UNAVAILABLE"
    UNRESOLVED_BACKEND_VERSION_MISMATCH = "UNRESOLVED_BACKEND_VERSION_MISMATCH"
    UNRESOLVED_LOW_TEXTURE = "UNRESOLVED_LOW_TEXTURE"
    UNRESOLVED_INSUFFICIENT_MUTUAL_MATCHES = (
        "UNRESOLVED_INSUFFICIENT_MUTUAL_MATCHES"
    )
    UNRESOLVED_NO_STABLE_MODEL = "UNRESOLVED_NO_STABLE_MODEL"
    UNRESOLVED_INSUFFICIENT_SPATIAL_SUPPORT = (
        "UNRESOLVED_INSUFFICIENT_SPATIAL_SUPPORT"
    )
    UNRESOLVED_INSUFFICIENT_FULL_OVERLAP = (
        "UNRESOLVED_INSUFFICIENT_FULL_OVERLAP"
    )
    UNRESOLVED_PHOTOMETRIC_AMBIGUOUS = "UNRESOLVED_PHOTOMETRIC_AMBIGUOUS"
    UNRESOLVED_BACKEND_ERROR = "UNRESOLVED_BACKEND_ERROR"


class GeometricAuditCapacityExceeded(RuntimeError):
    """Raised before partial publication when a frozen work cap is exceeded."""


@dataclass(frozen=True, slots=True)
class GeometricImageBinding:
    opaque_sample_id: str
    canonical_width: int
    canonical_height: int
    pixel_sha256: str
    schema_version: str = "cvi.geometric_image_binding.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.geometric_image_binding.v1":
            raise ValueError("unsupported geometric image binding schema")
        _digest(self.opaque_sample_id, "opaque sample ID")
        _digest(self.pixel_sha256, "pixel SHA-256")
        _positive_int(self.canonical_width, "canonical_width")
        _positive_int(self.canonical_height, "canonical_height")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometricImageBinding":
        _exact(payload, set(cls.__dataclass_fields__), "geometric image binding")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class GeometricCandidatePair:
    left_opaque_sample_id: str
    right_opaque_sample_id: str
    candidate_channels: tuple[str, ...]
    candidate_evidence_tokens: tuple[str, ...]
    right_d4_hypotheses: tuple[str, ...] = ("ORIGINAL",)
    schema_version: str = "cvi.geometric_candidate_pair.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.geometric_candidate_pair.v1":
            raise ValueError("unsupported geometric candidate schema")
        _digest(self.left_opaque_sample_id, "left opaque sample ID")
        _digest(self.right_opaque_sample_id, "right opaque sample ID")
        if self.left_opaque_sample_id >= self.right_opaque_sample_id:
            raise ValueError("candidate endpoints must be distinct and sorted")
        if (
            not self.candidate_channels
            or self.candidate_channels != tuple(sorted(set(self.candidate_channels)))
            or not set(self.candidate_channels) <= _CANDIDATE_CHANNELS
        ):
            raise ValueError("candidate channels must be sorted, unique, and allowed")
        if (
            not self.candidate_evidence_tokens
            or self.candidate_evidence_tokens
            != tuple(sorted(set(self.candidate_evidence_tokens)))
        ):
            raise ValueError("candidate evidence tokens must be sorted and unique")
        for value in self.candidate_evidence_tokens:
            _digest(value, "candidate evidence token")
        order = {name: index for index, name in enumerate(D4_TRANSFORMS)}
        if (
            not self.right_d4_hypotheses
            or len(set(self.right_d4_hypotheses)) != len(self.right_d4_hypotheses)
            or any(value not in order for value in self.right_d4_hypotheses)
            or tuple(sorted(self.right_d4_hypotheses, key=order.__getitem__))
            != self.right_d4_hypotheses
        ):
            raise ValueError("D4 hypotheses must be unique and in frozen order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left_opaque_sample_id": self.left_opaque_sample_id,
            "right_opaque_sample_id": self.right_opaque_sample_id,
            "candidate_channels": list(self.candidate_channels),
            "candidate_evidence_tokens": list(self.candidate_evidence_tokens),
            "right_d4_hypotheses": list(self.right_d4_hypotheses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometricCandidatePair":
        _exact(payload, set(cls.__dataclass_fields__), "geometric candidate")
        values = dict(payload)
        for name in (
            "candidate_channels",
            "candidate_evidence_tokens",
            "right_d4_hypotheses",
        ):
            if not isinstance(values[name], list):
                raise TypeError(f"{name} must be a JSON array")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GeometricVerifierRequest:
    candidates: tuple[GeometricCandidatePair, ...]
    images: tuple[GeometricImageBinding, ...]
    evidence_bindings: tuple[tuple[str, str], ...]
    interpretation: str = INPUT_INTERPRETATION
    schema_version: str = "cvi.geometric_verifier_request.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.geometric_verifier_request.v1":
            raise ValueError("unsupported geometric request schema")
        if self.interpretation != INPUT_INTERPRETATION:
            raise ValueError("geometric request interpretation differs")
        if not self.candidates:
            raise ValueError("geometric request must contain candidates")
        expected = tuple(
            sorted(
                self.candidates,
                key=lambda item: (
                    item.left_opaque_sample_id,
                    item.right_opaque_sample_id,
                ),
            )
        )
        if self.candidates != expected:
            raise ValueError("geometric candidates must be pair-sorted")
        pair_keys = tuple(
            (item.left_opaque_sample_id, item.right_opaque_sample_id)
            for item in self.candidates
        )
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("duplicate geometric candidate pair")
        if not self.images or self.images != tuple(
            sorted(self.images, key=lambda item: item.opaque_sample_id)
        ):
            raise ValueError("geometric image bindings must be nonempty and sorted")
        image_ids = tuple(item.opaque_sample_id for item in self.images)
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("duplicate geometric image binding")
        referenced = {token for pair in pair_keys for token in pair}
        if referenced != set(image_ids):
            raise ValueError("image bindings must exactly cover candidate endpoints")
        _bindings(self.evidence_bindings)
        binding_names = {name for name, _ in self.evidence_bindings}
        required_names = {_IMAGE_BINDING} | {
            _CHANNEL_BINDING[channel]
            for candidate in self.candidates
            for channel in candidate.candidate_channels
        }
        if binding_names != required_names:
            raise ValueError(
                "evidence bindings must exactly match image receipts and candidate channels"
            )

    @property
    def request_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidates": [item.to_dict() for item in self.candidates],
            "images": [item.to_dict() for item in self.images],
            "evidence_bindings": [list(item) for item in self.evidence_bindings],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometricVerifierRequest":
        _exact(payload, set(cls.__dataclass_fields__), "geometric request")
        if not isinstance(payload["candidates"], list) or not isinstance(
            payload["images"], list
        ):
            raise TypeError("request candidates and images must be JSON arrays")
        return cls(
            candidates=tuple(
                GeometricCandidatePair.from_dict(item)
                for item in payload["candidates"]
            ),
            images=tuple(
                GeometricImageBinding.from_dict(item) for item in payload["images"]
            ),
            evidence_bindings=_binding_tuple(payload["evidence_bindings"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class GeometricVerifierPolicy:
    threshold_status: str = THRESHOLD_STATUS
    backend_admission_status: str = (
        "REFERENCE_API_SMOKE_ONLY_THIRD_PARTY_LICENSE_RECEIPT_PENDING"
    )
    opencv_reference_version: str = "4.13.0"
    numpy_reference_version: str = "2.5.1"
    deterministic_seed: int = 734_221
    opencv_threads: int = 1
    opencl_enabled: bool = False
    feature_maximum_side: int = 1600
    maximum_candidates: int = 1_000_000
    maximum_output_records: int = 1_000_000
    maximum_image_pixels: int = 33_554_432
    maximum_total_candidate_pixel_visits: int = 250_000_000_000
    maximum_d4_hypotheses_per_pair: int = 8
    maximum_keypoints: int = 4096
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6
    rootsift_epsilon: float = 1e-12
    lowe_ratio: float = 0.78
    minimum_keypoints: int = 12
    minimum_mutual_matches: int = 10
    ransac_reprojection_fraction: float = 0.004
    ransac_maximum_iterations: int = 10_000
    ransac_confidence: float = 0.999
    minimum_inliers: int = 10
    minimum_inlier_ratio: float = 0.45
    maximum_median_symmetric_error_fraction: float = 0.006
    maximum_p95_symmetric_error_fraction: float = 0.015
    minimum_hull_area_fraction: float = 0.025
    spatial_grid_size: int = 4
    minimum_occupied_grid_cells: int = 4
    interior_margin_fraction: float = 0.05
    minimum_interior_inlier_fraction: float = 0.35
    minimum_point_eigenvalue_ratio: float = 0.01
    minimum_affine_singular_ratio: float = 0.05
    minimum_affine_scale: float = 0.05
    maximum_affine_scale: float = 20.0
    maximum_homography_condition: float = 100_000_000.0
    minimum_projective_denominator: float = 0.05
    minimum_projected_area_ratio: float = 0.01
    maximum_projected_area_ratio: float = 100.0
    overlap_mask_erosion_pixels: int = 2
    ssim_window_size: int = 11
    ssim_gaussian_sigma: float = 1.5
    minimum_overlap_fraction_each_direction: float = 0.15
    minimum_overlap_pixels_each_direction: int = 4096
    confirmation_minimum_ssim: float = 0.72
    confirmation_minimum_gradient_correlation: float = 0.65
    rejection_maximum_ssim: float = 0.36
    rejection_maximum_gradient_correlation: float = 0.28
    schema_version: str = "cvi.geometric_verifier_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.geometric_verifier_policy.v1":
            raise ValueError("unsupported geometric verifier policy")
        if self.threshold_status != THRESHOLD_STATUS:
            raise ValueError("geometric thresholds are not marked initialization-only")
        if self.backend_admission_status != (
            "REFERENCE_API_SMOKE_ONLY_THIRD_PARTY_LICENSE_RECEIPT_PENDING"
        ):
            raise ValueError("geometric backend admission status differs")
        if (
            self.opencv_reference_version,
            self.numpy_reference_version,
        ) != ("4.13.0", "2.5.1"):
            raise ValueError("geometric reference backend versions differ")
        if self.deterministic_seed < 0 or self.deterministic_seed > 2**31 - 1:
            raise ValueError("deterministic seed is outside OpenCV int range")
        if self.opencv_threads != 1 or self.opencl_enabled is not False:
            raise ValueError("reference verifier requires single-threaded OpenCV/OpenCL off")
        integer_bounds = (
            ("feature_maximum_side", 64, 4096),
            ("maximum_candidates", 1, 1_000_000),
            ("maximum_output_records", 1, 1_000_000),
            ("maximum_image_pixels", 4096, 67_108_864),
            ("maximum_total_candidate_pixel_visits", 8192, 500_000_000_000),
            ("maximum_d4_hypotheses_per_pair", 1, 8),
            ("maximum_keypoints", 32, 8192),
            ("minimum_keypoints", 4, 8192),
            ("minimum_mutual_matches", 4, 8192),
            ("ransac_maximum_iterations", 100, 100_000),
            ("minimum_inliers", 4, 8192),
            ("spatial_grid_size", 2, 16),
            ("minimum_occupied_grid_cells", 1, 256),
            ("overlap_mask_erosion_pixels", 0, 32),
            ("ssim_window_size", 3, 31),
            ("minimum_overlap_pixels_each_direction", 64, 10_000_000),
        )
        for name, low, high in integer_bounds:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} is outside the supported range")
        if self.maximum_output_records < self.maximum_candidates:
            raise ValueError("output cap must cover the candidate cap")
        if self.minimum_keypoints > self.maximum_keypoints:
            raise ValueError("minimum keypoints exceeds keypoint cap")
        if self.minimum_mutual_matches > self.maximum_keypoints:
            raise ValueError("minimum matches exceeds keypoint cap")
        if self.minimum_inliers > self.minimum_mutual_matches:
            raise ValueError("minimum inliers exceeds minimum matches")
        if self.minimum_occupied_grid_cells > self.spatial_grid_size**2:
            raise ValueError("occupied-cell threshold exceeds grid capacity")
        if self.ssim_window_size % 2 != 1:
            raise ValueError("SSIM window size must be odd")
        unit_fields = (
            "lowe_ratio",
            "ransac_reprojection_fraction",
            "ransac_confidence",
            "minimum_inlier_ratio",
            "maximum_median_symmetric_error_fraction",
            "maximum_p95_symmetric_error_fraction",
            "minimum_hull_area_fraction",
            "interior_margin_fraction",
            "minimum_interior_inlier_fraction",
            "minimum_point_eigenvalue_ratio",
            "minimum_affine_singular_ratio",
            "minimum_overlap_fraction_each_direction",
            "confirmation_minimum_ssim",
            "confirmation_minimum_gradient_correlation",
            "rejection_maximum_ssim",
            "rejection_maximum_gradient_correlation",
        )
        for name in unit_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < 1:
                raise ValueError(f"{name} must be strictly inside (0,1)")
        if self.rejection_maximum_ssim >= self.confirmation_minimum_ssim:
            raise ValueError("SSIM rejection and confirmation bands overlap")
        if (
            self.rejection_maximum_gradient_correlation
            >= self.confirmation_minimum_gradient_correlation
        ):
            raise ValueError("gradient rejection and confirmation bands overlap")
        positive_fields = (
            "sift_contrast_threshold",
            "sift_edge_threshold",
            "sift_sigma",
            "ssim_gaussian_sigma",
            "rootsift_epsilon",
            "minimum_affine_scale",
            "maximum_affine_scale",
            "maximum_homography_condition",
            "minimum_projective_denominator",
            "minimum_projected_area_ratio",
            "maximum_projected_area_ratio",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_affine_scale >= self.maximum_affine_scale:
            raise ValueError("affine scale range is empty")
        if self.minimum_projected_area_ratio >= self.maximum_projected_area_ratio:
            raise ValueError("projected-area range is empty")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometricVerifierPolicy":
        _exact(payload, set(cls.__dataclass_fields__), "geometric verifier policy")
        candidate = cls(**payload)
        if candidate != cls():
            raise ValueError("geometric policy differs from frozen initialization")
        return candidate


@dataclass(frozen=True, slots=True)
class GeometricPairResult:
    left_opaque_sample_id: str
    right_opaque_sample_id: str
    decision: GeometricDecision
    reason: GeometricReason
    selected_right_d4: str | None
    selected_model: str | None
    metrics: tuple[tuple[str, int | float | str], ...]
    candidate_evidence_tokens: tuple[str, ...]
    evidence_token: str
    schema_version: str = "cvi.geometric_pair_result.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.geometric_pair_result.v1":
            raise ValueError("unsupported geometric pair result schema")
        _digest(self.left_opaque_sample_id, "left result sample ID")
        _digest(self.right_opaque_sample_id, "right result sample ID")
        _digest(self.evidence_token, "pair evidence token")
        if self.left_opaque_sample_id >= self.right_opaque_sample_id:
            raise ValueError("result endpoints must be distinct and sorted")
        if not isinstance(self.decision, GeometricDecision) or not isinstance(
            self.reason, GeometricReason
        ):
            raise TypeError("result decision/reason must use typed enums")
        if self.decision is GeometricDecision.GEOMETRIC_CONFIRMED and self.reason is not GeometricReason.CONFIRMED_GEOMETRIC_AND_PHOTOMETRIC:
            raise ValueError("confirmed decision/reason differs")
        if self.decision is GeometricDecision.GEOMETRIC_REJECTED and self.reason is not GeometricReason.REJECTED_PHOTOMETRIC_CONTRADICTION:
            raise ValueError("rejected decision/reason differs")
        if self.decision is GeometricDecision.UNRESOLVED and not self.reason.value.startswith("UNRESOLVED_"):
            raise ValueError("unresolved decision/reason differs")
        if self.selected_right_d4 is not None and self.selected_right_d4 not in D4_TRANSFORMS:
            raise ValueError("selected D4 hypothesis is invalid")
        if self.selected_model is not None and self.selected_model not in {
            "PARTIAL_AFFINE",
            "AFFINE",
            "HOMOGRAPHY_USAC_MAGSAC",
        }:
            raise ValueError("selected model is invalid")
        if self.decision is not GeometricDecision.UNRESOLVED and (
            self.selected_right_d4 is None or self.selected_model is None
        ):
            raise ValueError("decisive result requires selected D4 and model")
        if self.metrics != tuple(sorted(self.metrics)) or len(dict(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be key-sorted and unique")
        for key, value in self.metrics:
            if not isinstance(key, str) or not key:
                raise ValueError("metric names must be nonempty strings")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("metrics must not contain NaN or infinity")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise TypeError("metric values must be JSON scalar measurements")
        if (
            not self.candidate_evidence_tokens
            or self.candidate_evidence_tokens
            != tuple(sorted(set(self.candidate_evidence_tokens)))
        ):
            raise ValueError("result candidate evidence tokens must be canonical")
        for value in self.candidate_evidence_tokens:
            _digest(value, "result candidate evidence token")
        expected_token = content_sha256(_pair_evidence_payload(
            left=self.left_opaque_sample_id,
            right=self.right_opaque_sample_id,
            decision=self.decision,
            reason=self.reason,
            selected_right_d4=self.selected_right_d4,
            selected_model=self.selected_model,
            metrics=self.metrics,
            candidate_evidence_tokens=self.candidate_evidence_tokens,
        ))
        if self.evidence_token != expected_token:
            raise ValueError("pair evidence token differs from result content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left_opaque_sample_id": self.left_opaque_sample_id,
            "right_opaque_sample_id": self.right_opaque_sample_id,
            "decision": self.decision.value,
            "reason": self.reason.value,
            "selected_right_d4": self.selected_right_d4,
            "selected_model": self.selected_model,
            "metrics": {key: value for key, value in self.metrics},
            "candidate_evidence_tokens": list(self.candidate_evidence_tokens),
            "evidence_token": self.evidence_token,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometricPairResult":
        _exact(payload, set(cls.__dataclass_fields__), "geometric pair result")
        metrics = payload["metrics"]
        tokens = payload["candidate_evidence_tokens"]
        if not isinstance(metrics, Mapping) or not isinstance(tokens, list):
            raise TypeError("geometric result metrics/tokens fields differ")
        return cls(
            left_opaque_sample_id=payload["left_opaque_sample_id"],
            right_opaque_sample_id=payload["right_opaque_sample_id"],
            decision=GeometricDecision(payload["decision"]),
            reason=GeometricReason(payload["reason"]),
            selected_right_d4=payload["selected_right_d4"],
            selected_model=payload["selected_model"],
            metrics=tuple(sorted(metrics.items())),
            candidate_evidence_tokens=tuple(tokens),
            evidence_token=payload["evidence_token"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class GeometricVerifierEvidence:
    request_sha256: str
    policy: GeometricVerifierPolicy
    backend: tuple[tuple[str, str | int | bool], ...]
    results: tuple[GeometricPairResult, ...]
    counts: tuple[tuple[str, int], ...]
    evidence_bindings: tuple[tuple[str, str], ...]
    interpretation: str = OUTPUT_INTERPRETATION
    schema_version: str = "cvi.geometric_verifier_evidence.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.geometric_verifier_evidence.v1":
            raise ValueError("unsupported geometric evidence schema")
        _digest(self.request_sha256, "request SHA-256")
        if self.interpretation != OUTPUT_INTERPRETATION:
            raise ValueError("geometric evidence interpretation differs")
        if self.policy.threshold_status != THRESHOLD_STATUS:
            raise ValueError("geometric evidence policy is not initialization-only")
        if self.results != tuple(
            sorted(
                self.results,
                key=lambda item: (
                    item.left_opaque_sample_id,
                    item.right_opaque_sample_id,
                ),
            )
        ):
            raise ValueError("geometric results must be pair-sorted")
        expected_counts = {decision.value: 0 for decision in GeometricDecision}
        for item in self.results:
            expected_counts[item.decision.value] += 1
        if self.counts != tuple(sorted(expected_counts.items())):
            raise ValueError("geometric result counts differ")
        if self.backend != tuple(sorted(self.backend)) or len(dict(self.backend)) != len(self.backend):
            raise ValueError("backend evidence must be key-sorted and unique")
        _bindings(self.evidence_bindings)

    @property
    def evidence_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy.policy_sha256,
            "threshold_status": self.policy.threshold_status,
            "backend": {key: value for key, value in self.backend},
            "results": [item.to_dict() for item in self.results],
            "counts": {key: value for key, value in self.counts},
            "evidence_bindings": [list(item) for item in self.evidence_bindings],
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometricVerifierEvidence":
        expected = set(cls.__dataclass_fields__) | {"policy_sha256", "threshold_status"}
        _exact(payload, expected, "geometric verifier evidence")
        policy_raw = payload["policy"]
        backend = payload["backend"]
        results = payload["results"]
        counts = payload["counts"]
        if (
            not isinstance(policy_raw, Mapping)
            or not isinstance(backend, Mapping)
            or not isinstance(results, list)
            or not isinstance(counts, Mapping)
        ):
            raise TypeError("geometric evidence collection fields differ")
        policy = GeometricVerifierPolicy.from_dict(policy_raw)
        if (
            payload["policy_sha256"] != policy.policy_sha256
            or payload["threshold_status"] != policy.threshold_status
        ):
            raise ValueError("geometric evidence policy binding differs")
        return cls(
            request_sha256=payload["request_sha256"],
            policy=policy,
            backend=tuple(sorted(backend.items())),
            results=tuple(GeometricPairResult.from_dict(item) for item in results),
            counts=tuple(sorted(counts.items())),
            evidence_bindings=_binding_tuple(payload["evidence_bindings"]),
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


def canonical_rgb_sha256(rgb: Any) -> str:
    """Hash a canonical uint8 HxWx3 raster like the image-content audit."""

    np = importlib.import_module("numpy")
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("canonical RGB must be a uint8 HxWx3 raster")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("canonical RGB geometry must be positive")
    digest = hashlib.sha256()
    digest.update(_PIXEL_HASH_DOMAIN)
    digest.update(struct.pack(">QQ", width, height))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def verify_geometric_request(
    request: GeometricVerifierRequest,
    *,
    image_loader: Callable[[str], Any],
    policy: GeometricVerifierPolicy = GeometricVerifierPolicy(),
) -> GeometricVerifierEvidence:
    """Verify a bounded candidate set without labels or role metadata."""

    if not isinstance(request, GeometricVerifierRequest):
        raise TypeError("request must be GeometricVerifierRequest")
    if not isinstance(policy, GeometricVerifierPolicy):
        raise TypeError("policy must be GeometricVerifierPolicy")
    if policy != GeometricVerifierPolicy():
        raise ValueError("only the frozen initialization policy is accepted")
    if not callable(image_loader):
        raise TypeError("image_loader must be callable")
    _preflight(request, policy)
    backend = _load_backend()
    if backend is None:
        backend_rows = (
            ("availability", "UNAVAILABLE"),
            ("deterministic_seed", policy.deterministic_seed),
            ("opencv_threads", policy.opencv_threads),
            ("opencl_enabled", policy.opencl_enabled),
        )
        results = tuple(
            _result(
                candidate,
                GeometricDecision.UNRESOLVED,
                GeometricReason.UNRESOLVED_BACKEND_UNAVAILABLE,
            )
            for candidate in request.candidates
        )
        return _evidence(request, policy, backend_rows, results)

    np, cv2 = backend
    if (
        str(cv2.__version__) != policy.opencv_reference_version
        or str(np.__version__) != policy.numpy_reference_version
    ):
        backend_rows = tuple(sorted({
            "availability": "UNSUPPORTED_VERSION",
            "numpy_version": str(np.__version__),
            "opencv_version": str(cv2.__version__),
            "expected_numpy_version": policy.numpy_reference_version,
            "expected_opencv_version": policy.opencv_reference_version,
            "deterministic_seed": policy.deterministic_seed,
            "opencv_threads": policy.opencv_threads,
            "opencl_enabled": policy.opencl_enabled,
        }.items()))
        results = tuple(
            _result(
                candidate,
                GeometricDecision.UNRESOLVED,
                GeometricReason.UNRESOLVED_BACKEND_VERSION_MISMATCH,
            )
            for candidate in request.candidates
        )
        return _evidence(request, policy, backend_rows, results)
    cv2.setNumThreads(policy.opencv_threads)
    cv2.setRNGSeed(policy.deterministic_seed)
    try:
        cv2.ocl.setUseOpenCL(policy.opencl_enabled)
    except AttributeError:
        pass
    backend_rows = tuple(sorted({
        "availability": "AVAILABLE",
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "feature": "SIFT_ROOTSIFT_MUTUAL_RATIO",
        "homography": "USAC_MAGSAC",
        "deterministic_seed": policy.deterministic_seed,
        "opencv_threads": policy.opencv_threads,
        "opencl_enabled": policy.opencl_enabled,
    }.items()))
    bindings = {item.opaque_sample_id: item for item in request.images}
    results: list[GeometricPairResult] = []
    for candidate in request.candidates:
        left = _load_bound_image(
            candidate.left_opaque_sample_id,
            bindings[candidate.left_opaque_sample_id],
            image_loader,
            policy,
            np,
        )
        right = _load_bound_image(
            candidate.right_opaque_sample_id,
            bindings[candidate.right_opaque_sample_id],
            image_loader,
            policy,
            np,
        )
        try:
            results.append(_verify_pair(candidate, left, right, policy, np, cv2))
        except cv2.error:
            results.append(_result(
                candidate,
                GeometricDecision.UNRESOLVED,
                GeometricReason.UNRESOLVED_BACKEND_ERROR,
            ))
    return _evidence(request, policy, backend_rows, tuple(results))


def publish_geometric_evidence(
    path: Path,
    evidence: GeometricVerifierEvidence,
    *,
    tool_provenance: Mapping[str, Any],
) -> str:
    """Publish one private, content-bound JSON artifact without overwrite."""

    if not isinstance(evidence, GeometricVerifierEvidence):
        raise TypeError("evidence must be GeometricVerifierEvidence")
    if not isinstance(tool_provenance, Mapping) or not tool_provenance:
        raise ValueError("tool provenance must be a nonempty object")
    provenance = dict(tool_provenance)
    bundle = {
        "schema_version": "cvi.geometric_verifier_bundle.v1",
        "evidence": evidence.to_dict(),
        "evidence_sha256": evidence.evidence_sha256,
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
    }
    bundle["bundle_sha256"] = content_sha256(bundle)
    write_private_json_bundle(((path, bundle),))
    return bundle["bundle_sha256"]


def read_geometric_evidence_bundle(path: Path) -> GeometricVerifierEvidence:
    """Read and authenticate one no-overwrite geometric result chunk."""

    from shared.foundation.protected_io import read_strict_json_object

    bundle = read_strict_json_object(path)
    expected = {
        "schema_version",
        "evidence",
        "evidence_sha256",
        "tool_provenance",
        "tool_provenance_sha256",
        "bundle_sha256",
    }
    if set(bundle) != expected or bundle["schema_version"] != (
        "cvi.geometric_verifier_bundle.v1"
    ):
        raise ValueError("geometric verifier bundle fields differ")
    unsigned = dict(bundle)
    observed_bundle_sha256 = unsigned.pop("bundle_sha256")
    if content_sha256(unsigned) != observed_bundle_sha256:
        raise ValueError("geometric verifier bundle digest differs")
    if content_sha256(bundle["tool_provenance"]) != bundle[
        "tool_provenance_sha256"
    ]:
        raise ValueError("geometric verifier provenance digest differs")
    evidence = GeometricVerifierEvidence.from_dict(bundle["evidence"])
    if evidence.evidence_sha256 != bundle["evidence_sha256"]:
        raise ValueError("geometric verifier evidence digest differs")
    return evidence


def _load_backend() -> tuple[Any, Any] | None:
    try:
        np = importlib.import_module("numpy")
        cv2 = importlib.import_module("cv2")
    except (ImportError, ModuleNotFoundError):
        return None
    if not hasattr(cv2, "SIFT_create") or not hasattr(cv2, "USAC_MAGSAC"):
        return None
    return np, cv2


def _preflight(
    request: GeometricVerifierRequest, policy: GeometricVerifierPolicy
) -> None:
    if len(request.candidates) > policy.maximum_candidates:
        raise GeometricAuditCapacityExceeded("candidate cap exceeded")
    if len(request.candidates) > policy.maximum_output_records:
        raise GeometricAuditCapacityExceeded("output-record cap exceeded")
    images = {item.opaque_sample_id: item for item in request.images}
    for item in request.images:
        pixels = item.canonical_width * item.canonical_height
        if pixels > policy.maximum_image_pixels:
            raise GeometricAuditCapacityExceeded("image-pixel cap exceeded")
    visits = 0
    for item in request.candidates:
        if len(item.right_d4_hypotheses) > policy.maximum_d4_hypotheses_per_pair:
            raise GeometricAuditCapacityExceeded("D4-hypothesis cap exceeded")
        visits += (
            images[item.left_opaque_sample_id].canonical_width
            * images[item.left_opaque_sample_id].canonical_height
        )
        visits += (
            images[item.right_opaque_sample_id].canonical_width
            * images[item.right_opaque_sample_id].canonical_height
            * len(item.right_d4_hypotheses)
        )
        if visits > policy.maximum_total_candidate_pixel_visits:
            raise GeometricAuditCapacityExceeded("candidate pixel-visit cap exceeded")


def _load_bound_image(
    token: str,
    binding: GeometricImageBinding,
    loader: Callable[[str], Any],
    policy: GeometricVerifierPolicy,
    np: Any,
) -> Any:
    array = np.asarray(loader(token))
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image loader must return canonical uint8 HxWx3 RGB")
    height, width = array.shape[:2]
    if (width, height) != (binding.canonical_width, binding.canonical_height):
        raise ValueError("loaded image geometry differs from protected binding")
    if width * height > policy.maximum_image_pixels:
        raise GeometricAuditCapacityExceeded("loaded image exceeds pixel cap")
    if canonical_rgb_sha256(array) != binding.pixel_sha256:
        raise ValueError("loaded canonical RGB digest differs from protected binding")
    return np.ascontiguousarray(array)


def _verify_pair(
    candidate: GeometricCandidatePair,
    left_rgb: Any,
    right_rgb: Any,
    policy: GeometricVerifierPolicy,
    np: Any,
    cv2: Any,
) -> GeometricPairResult:
    left_gray = _feature_gray(left_rgb, policy, np, cv2)
    left_features = _rootsift(left_gray, policy, np, cv2)
    if left_features is None:
        return _result(candidate, GeometricDecision.UNRESOLVED, GeometricReason.UNRESOLVED_LOW_TEXTURE)
    saw_matches = False
    saw_model = False
    saw_spatial = False
    saw_overlap = False
    hypotheses: list[dict[str, Any]] = []
    for d4_index, transform_name in enumerate(candidate.right_d4_hypotheses):
        right_transformed = _apply_d4(right_rgb, transform_name, np)
        right_gray = _feature_gray(right_transformed, policy, np, cv2)
        right_features = _rootsift(right_gray, policy, np, cv2)
        if right_features is None:
            continue
        matches = _mutual_ratio_matches(
            left_features[1], right_features[1], policy, cv2
        )
        if len(matches) < policy.minimum_mutual_matches:
            continue
        saw_matches = True
        source = np.asarray(
            [left_features[0][left_index].pt for left_index, _, _ in matches],
            dtype=np.float64,
        )
        target = np.asarray(
            [right_features[0][right_index].pt for _, right_index, _ in matches],
            dtype=np.float64,
        )
        pair_seed = int(
            hashlib.sha256(
                (candidate.left_opaque_sample_id + candidate.right_opaque_sample_id + transform_name).encode("ascii")
            ).hexdigest()[:8],
            16,
        ) & 0x7FFFFFFF
        cv2.setRNGSeed(policy.deterministic_seed ^ pair_seed)
        models = _estimate_models(source, target, left_gray.shape, right_gray.shape, policy, np, cv2)
        if models:
            saw_model = True
        for model in models:
            if not model["spatial_pass"]:
                continue
            saw_spatial = True
            photo = _photometric(
                left_gray,
                right_gray,
                model["homography"],
                policy,
                np,
                cv2,
            )
            if photo is None:
                continue
            saw_overlap = True
            merged = dict(model)
            merged.update(photo)
            merged["right_d4"] = transform_name
            merged["d4_index"] = d4_index
            hypotheses.append(merged)
    if not hypotheses:
        if not saw_matches:
            reason = GeometricReason.UNRESOLVED_INSUFFICIENT_MUTUAL_MATCHES
        elif not saw_model:
            reason = GeometricReason.UNRESOLVED_NO_STABLE_MODEL
        elif not saw_spatial:
            reason = GeometricReason.UNRESOLVED_INSUFFICIENT_SPATIAL_SUPPORT
        elif not saw_overlap:
            reason = GeometricReason.UNRESOLVED_INSUFFICIENT_FULL_OVERLAP
        else:
            reason = GeometricReason.UNRESOLVED_BACKEND_ERROR
        return _result(candidate, GeometricDecision.UNRESOLVED, reason)
    selected = max(
        hypotheses,
        key=lambda item: (
            item["inlier_count"],
            item["inlier_ratio"],
            -item["median_symmetric_error_fraction"],
            -item["model_index"],
            -item["d4_index"],
        ),
    )
    ssim = min(selected["forward_ssim"], selected["backward_ssim"])
    gradient = min(
        selected["forward_gradient_correlation"],
        selected["backward_gradient_correlation"],
    )
    metrics = {
        key: value
        for key, value in selected.items()
        if key not in {"homography", "spatial_pass", "model_index", "d4_index", "right_d4"}
    }
    if (
        ssim >= policy.confirmation_minimum_ssim
        and gradient >= policy.confirmation_minimum_gradient_correlation
    ):
        return _result(
            candidate,
            GeometricDecision.GEOMETRIC_CONFIRMED,
            GeometricReason.CONFIRMED_GEOMETRIC_AND_PHOTOMETRIC,
            selected_right_d4=selected["right_d4"],
            selected_model=selected["model_name"],
            metrics=metrics,
        )
    if (
        ssim <= policy.rejection_maximum_ssim
        and gradient <= policy.rejection_maximum_gradient_correlation
    ):
        return _result(
            candidate,
            GeometricDecision.GEOMETRIC_REJECTED,
            GeometricReason.REJECTED_PHOTOMETRIC_CONTRADICTION,
            selected_right_d4=selected["right_d4"],
            selected_model=selected["model_name"],
            metrics=metrics,
        )
    return _result(
        candidate,
        GeometricDecision.UNRESOLVED,
        GeometricReason.UNRESOLVED_PHOTOMETRIC_AMBIGUOUS,
        selected_right_d4=selected["right_d4"],
        selected_model=selected["model_name"],
        metrics=metrics,
    )


def _feature_gray(rgb: Any, policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> Any:
    height, width = rgb.shape[:2]
    scale = min(1.0, policy.feature_maximum_side / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def _rootsift(gray: Any, policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> tuple[tuple[Any, ...], Any] | None:
    sift = cv2.SIFT_create(
        nfeatures=policy.maximum_keypoints,
        contrastThreshold=policy.sift_contrast_threshold,
        edgeThreshold=policy.sift_edge_threshold,
        sigma=policy.sift_sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < policy.minimum_keypoints:
        return None
    order = sorted(
        range(len(keypoints)),
        key=lambda index: (
            -float(keypoints[index].response),
            float(keypoints[index].pt[1]),
            float(keypoints[index].pt[0]),
            float(keypoints[index].size),
            float(keypoints[index].angle),
            int(keypoints[index].octave),
        ),
    )[: policy.maximum_keypoints]
    keypoints = tuple(keypoints[index] for index in order)
    descriptors = np.asarray(descriptors[order], dtype=np.float32)
    l1 = np.sum(np.abs(descriptors), axis=1, keepdims=True)
    descriptors = np.sqrt(descriptors / np.maximum(l1, policy.rootsift_epsilon))
    return keypoints, np.ascontiguousarray(descriptors, dtype=np.float32)


def _mutual_ratio_matches(left: Any, right: Any, policy: GeometricVerifierPolicy, cv2: Any) -> list[tuple[int, int, float]]:
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def directional(a: Any, b: Any) -> dict[int, tuple[int, float]]:
        result: dict[int, tuple[int, float]] = {}
        for row in matcher.knnMatch(a, b, k=2):
            if len(row) != 2:
                continue
            first, second = row
            if first.distance < policy.lowe_ratio * second.distance:
                result[int(first.queryIdx)] = (int(first.trainIdx), float(first.distance))
        return result

    forward = directional(left, right)
    backward = directional(right, left)
    mutual = [
        (left_index, right_index, distance)
        for left_index, (right_index, distance) in forward.items()
        if backward.get(right_index, (-1, 0.0))[0] == left_index
    ]
    return sorted(mutual, key=lambda item: (item[2], item[0], item[1]))


def _estimate_models(source: Any, target: Any, left_shape: tuple[int, ...], right_shape: tuple[int, ...], policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> list[dict[str, Any]]:
    threshold = policy.ransac_reprojection_fraction * max(
        math.hypot(left_shape[1], left_shape[0]),
        math.hypot(right_shape[1], right_shape[0]),
    )
    estimators: list[tuple[str, Any]] = []
    partial, partial_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=policy.ransac_maximum_iterations,
        confidence=policy.ransac_confidence,
        refineIters=10,
    )
    estimators.append(("PARTIAL_AFFINE", (partial, partial_mask)))
    affine, affine_mask = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=policy.ransac_maximum_iterations,
        confidence=policy.ransac_confidence,
        refineIters=10,
    )
    estimators.append(("AFFINE", (affine, affine_mask)))
    homography, homography_mask = cv2.findHomography(
        source,
        target,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=threshold,
        maxIters=policy.ransac_maximum_iterations,
        confidence=policy.ransac_confidence,
    )
    estimators.append(("HOMOGRAPHY_USAC_MAGSAC", (homography, homography_mask)))
    results: list[dict[str, Any]] = []
    for model_index, (name, (matrix, mask)) in enumerate(estimators):
        if matrix is None or mask is None:
            continue
        if name != "HOMOGRAPHY_USAC_MAGSAC":
            matrix = np.vstack([matrix, np.asarray([0.0, 0.0, 1.0])])
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            continue
        result = _geometry_metrics(
            source, target, mask, matrix, left_shape, right_shape, policy, np, cv2
        )
        if result is None:
            continue
        result["model_name"] = name
        result["model_index"] = model_index
        results.append(result)
    return results


def _geometry_metrics(source: Any, target: Any, mask: Any, homography: Any, left_shape: tuple[int, ...], right_shape: tuple[int, ...], policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> dict[str, Any] | None:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    forward = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    backward = cv2.perspectiveTransform(target.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        return None
    forward_error = np.linalg.norm(forward - target, axis=1) / math.hypot(right_shape[1], right_shape[0])
    backward_error = np.linalg.norm(backward - source, axis=1) / math.hypot(left_shape[1], left_shape[0])
    symmetric = np.maximum(forward_error, backward_error)
    estimator_inliers = np.asarray(mask).reshape(-1).astype(bool)
    if len(estimator_inliers) != len(source):
        return None
    count = int(np.count_nonzero(estimator_inliers))
    if count < policy.minimum_inliers:
        return None
    inlier_ratio = count / len(source)
    src = source[estimator_inliers]
    dst = target[estimator_inliers]
    median = float(np.median(symmetric[estimator_inliers]))
    p95 = float(np.percentile(symmetric[estimator_inliers], 95))
    hull_left = _hull_fraction(src, left_shape, cv2)
    hull_right = _hull_fraction(dst, right_shape, cv2)
    cells_left = _occupied_cells(src, left_shape, policy.spatial_grid_size, np)
    cells_right = _occupied_cells(dst, right_shape, policy.spatial_grid_size, np)
    interior_left = _interior_fraction(src, left_shape, policy.interior_margin_fraction, np)
    interior_right = _interior_fraction(dst, right_shape, policy.interior_margin_fraction, np)
    point_ratio = min(_point_eigen_ratio(src, np), _point_eigen_ratio(dst, np))
    transform_valid, condition, projected_area = _transform_nondegenerate(
        homography, left_shape, right_shape, policy, np, cv2
    )
    spatial_pass = (
        inlier_ratio >= policy.minimum_inlier_ratio
        and median <= policy.maximum_median_symmetric_error_fraction
        and p95 <= policy.maximum_p95_symmetric_error_fraction
        and hull_left >= policy.minimum_hull_area_fraction
        and hull_right >= policy.minimum_hull_area_fraction
        and cells_left >= policy.minimum_occupied_grid_cells
        and cells_right >= policy.minimum_occupied_grid_cells
        and interior_left >= policy.minimum_interior_inlier_fraction
        and interior_right >= policy.minimum_interior_inlier_fraction
        and point_ratio >= policy.minimum_point_eigenvalue_ratio
        and transform_valid
    )
    return {
        "homography": homography,
        "spatial_pass": bool(spatial_pass),
        "mutual_match_count": int(len(source)),
        "inlier_count": count,
        "inlier_ratio": float(inlier_ratio),
        "median_symmetric_error_fraction": median,
        "p95_symmetric_error_fraction": p95,
        "left_hull_area_fraction": hull_left,
        "right_hull_area_fraction": hull_right,
        "left_occupied_grid_cells": cells_left,
        "right_occupied_grid_cells": cells_right,
        "left_interior_inlier_fraction": interior_left,
        "right_interior_inlier_fraction": interior_right,
        "minimum_point_eigenvalue_ratio": point_ratio,
        "homography_condition": condition,
        "projected_area_ratio": projected_area,
    }


def _transform_nondegenerate(homography: Any, left_shape: tuple[int, ...], right_shape: tuple[int, ...], policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> tuple[bool, float, float]:
    normalized = homography / max(abs(float(homography[2, 2])), 1e-12)
    condition = float(np.linalg.cond(normalized))
    singular = np.linalg.svd(normalized[:2, :2], compute_uv=False)
    scale_low, scale_high = float(min(singular)), float(max(singular))
    singular_ratio = scale_low / max(scale_high, 1e-12)
    height, width = left_shape[:2]
    corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return False, condition, 0.0
    projected_float = projected.astype(np.float32)
    signed_area = float(cv2.contourArea(projected_float, oriented=True))
    projected_ratio = signed_area / max(1.0, float(right_shape[0] * right_shape[1]))
    probes = np.vstack([corners, np.mean(corners, axis=0, keepdims=True)])
    denominators = (
        homography[2, 0] * probes[:, 0]
        + homography[2, 1] * probes[:, 1]
        + homography[2, 2]
    )
    denominator_valid = (
        np.isfinite(denominators).all()
        and np.min(np.abs(denominators)) >= policy.minimum_projective_denominator
        and (np.all(denominators > 0) or np.all(denominators < 0))
    )
    jacobian_valid = all(
        _local_projective_jacobian_valid(
            homography, float(point[0]), float(point[1]), policy, np
        )
        for point in probes
    )
    valid = (
        math.isfinite(condition)
        and condition <= policy.maximum_homography_condition
        and singular_ratio >= policy.minimum_affine_singular_ratio
        and scale_low >= policy.minimum_affine_scale
        and scale_high <= policy.maximum_affine_scale
        and policy.minimum_projected_area_ratio <= projected_ratio <= policy.maximum_projected_area_ratio
        and bool(cv2.isContourConvex(projected_float))
        and denominator_valid
        and jacobian_valid
    )
    return valid, condition, projected_ratio


def _local_projective_jacobian_valid(
    homography: Any,
    x: float,
    y: float,
    policy: GeometricVerifierPolicy,
    np: Any,
) -> bool:
    denominator = float(
        homography[2, 0] * x + homography[2, 1] * y + homography[2, 2]
    )
    if not math.isfinite(denominator) or abs(denominator) < policy.minimum_projective_denominator:
        return False
    numerator_x = float(
        homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]
    )
    numerator_y = float(
        homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]
    )
    denominator_squared = denominator**2
    jacobian = np.asarray([
        [
            (homography[0, 0] * denominator - numerator_x * homography[2, 0]) / denominator_squared,
            (homography[0, 1] * denominator - numerator_x * homography[2, 1]) / denominator_squared,
        ],
        [
            (homography[1, 0] * denominator - numerator_y * homography[2, 0]) / denominator_squared,
            (homography[1, 1] * denominator - numerator_y * homography[2, 1]) / denominator_squared,
        ],
    ], dtype=np.float64)
    if not np.isfinite(jacobian).all() or float(np.linalg.det(jacobian)) <= 0:
        return False
    singular = np.linalg.svd(jacobian, compute_uv=False)
    low, high = float(min(singular)), float(max(singular))
    return (
        low >= policy.minimum_affine_scale
        and high <= policy.maximum_affine_scale
        and low / max(high, 1e-12) >= policy.minimum_affine_singular_ratio
    )


def _photometric(left: Any, right: Any, homography: Any, policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> dict[str, Any] | None:
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    forward = _warp_compare(left, right, homography, policy, np, cv2)
    backward = _warp_compare(right, left, inverse, policy, np, cv2)
    if forward is None or backward is None:
        return None
    return {
        "forward_overlap_fraction": forward[0],
        "backward_overlap_fraction": backward[0],
        "forward_overlap_pixels": forward[1],
        "backward_overlap_pixels": backward[1],
        "forward_ssim": forward[2],
        "backward_ssim": backward[2],
        "forward_gradient_correlation": forward[3],
        "backward_gradient_correlation": backward[3],
    }


def _warp_compare(source: Any, target: Any, homography: Any, policy: GeometricVerifierPolicy, np: Any, cv2: Any) -> tuple[float, int, float, float] | None:
    height, width = target.shape[:2]
    warped = cv2.warpPerspective(
        source,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpPerspective(
        np.full(source.shape, 255, dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if policy.overlap_mask_erosion_pixels:
        radius = policy.overlap_mask_erosion_pixels
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel)
    valid = mask > 0
    count = int(np.count_nonzero(valid))
    fraction = count / (height * width)
    if count < policy.minimum_overlap_pixels_each_direction or fraction < policy.minimum_overlap_fraction_each_direction:
        return None
    ssim = _masked_local_ssim(warped, target, mask, policy, np, cv2)
    if ssim is None:
        return None
    gx = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
    tx = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    ty = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)
    first = np.hypot(gx, gy)[valid].astype(np.float64)
    second = np.hypot(tx, ty)[valid].astype(np.float64)
    gradient = _correlation(first, second, np)
    if gradient is None:
        return None
    return float(fraction), count, float(ssim), float(gradient)


def _masked_local_ssim(
    first: Any,
    second: Any,
    overlap_mask: Any,
    policy: GeometricVerifierPolicy,
    np: Any,
    cv2: Any,
) -> float | None:
    """Mean standard local SSIM over every window wholly inside overlap."""

    x = first.astype(np.float64) / 255.0
    y = second.astype(np.float64) / 255.0
    window = (policy.ssim_window_size, policy.ssim_window_size)
    sigma = policy.ssim_gaussian_sigma
    mean_first = cv2.GaussianBlur(x, window, sigma)
    mean_second = cv2.GaussianBlur(y, window, sigma)
    variance_first = cv2.GaussianBlur(x * x, window, sigma) - mean_first**2
    variance_second = cv2.GaussianBlur(y * y, window, sigma) - mean_second**2
    covariance = cv2.GaussianBlur(x * y, window, sigma) - mean_first * mean_second
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
        (mean_first**2 + mean_second**2 + c1)
        * (variance_first + variance_second + c2)
    )
    radius = policy.ssim_window_size // 2
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    valid = cv2.erode(overlap_mask, kernel) > 0
    if int(np.count_nonzero(valid)) < policy.minimum_overlap_pixels_each_direction:
        return None
    return max(-1.0, min(1.0, float(np.mean(value[valid]))))


def _correlation(first: Any, second: Any, np: Any) -> float | None:
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    return max(-1.0, min(1.0, float(np.dot(first, second) / denominator)))


def _hull_fraction(points: Any, shape: tuple[int, ...], cv2: Any) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype("float32"))
    return abs(float(cv2.contourArea(hull))) / max(1.0, float(shape[0] * shape[1]))


def _occupied_cells(points: Any, shape: tuple[int, ...], grid: int, np: Any) -> int:
    x = np.clip((points[:, 0] / max(1, shape[1]) * grid).astype(int), 0, grid - 1)
    y = np.clip((points[:, 1] / max(1, shape[0]) * grid).astype(int), 0, grid - 1)
    return len(set(zip(x.tolist(), y.tolist(), strict=True)))


def _interior_fraction(points: Any, shape: tuple[int, ...], margin: float, np: Any) -> float:
    x_margin, y_margin = shape[1] * margin, shape[0] * margin
    inside = (
        (points[:, 0] >= x_margin)
        & (points[:, 0] <= shape[1] - x_margin)
        & (points[:, 1] >= y_margin)
        & (points[:, 1] <= shape[0] - y_margin)
    )
    return float(np.mean(inside))


def _point_eigen_ratio(points: Any, np: Any) -> float:
    if len(points) < 3:
        return 0.0
    covariance = np.cov(points.T, bias=True)
    values = np.linalg.eigvalsh(covariance)
    return float(max(0.0, values[0]) / max(float(values[-1]), 1e-12))


def _apply_d4(rgb: Any, name: str, np: Any) -> Any:
    if name == "ORIGINAL":
        result = rgb
    elif name == "ROT90CCW":
        result = np.rot90(rgb, 1)
    elif name == "ROT180":
        result = np.rot90(rgb, 2)
    elif name == "ROT270CCW":
        result = np.rot90(rgb, 3)
    elif name == "FLIP_X":
        result = np.flip(rgb, axis=0)
    elif name == "FLIP_Y":
        result = np.flip(rgb, axis=1)
    elif name == "FLIP_PLUS_DIAGONAL":
        result = np.transpose(rgb, (1, 0, 2))
    elif name == "FLIP_MINUS_DIAGONAL":
        result = np.flip(np.transpose(rgb, (1, 0, 2)), axis=(0, 1))
    else:
        raise ValueError("unsupported D4 transform")
    return np.ascontiguousarray(result)


def _result(
    candidate: GeometricCandidatePair,
    decision: GeometricDecision,
    reason: GeometricReason,
    *,
    selected_right_d4: str | None = None,
    selected_model: str | None = None,
    metrics: Mapping[str, int | float | str] | None = None,
) -> GeometricPairResult:
    metric_rows = tuple(sorted((metrics or {}).items()))
    payload = _pair_evidence_payload(
        left=candidate.left_opaque_sample_id,
        right=candidate.right_opaque_sample_id,
        decision=decision,
        reason=reason,
        selected_right_d4=selected_right_d4,
        selected_model=selected_model,
        metrics=metric_rows,
        candidate_evidence_tokens=candidate.candidate_evidence_tokens,
    )
    return GeometricPairResult(
        left_opaque_sample_id=candidate.left_opaque_sample_id,
        right_opaque_sample_id=candidate.right_opaque_sample_id,
        decision=decision,
        reason=reason,
        selected_right_d4=selected_right_d4,
        selected_model=selected_model,
        metrics=metric_rows,
        candidate_evidence_tokens=candidate.candidate_evidence_tokens,
        evidence_token=content_sha256(payload),
    )


def _pair_evidence_payload(
    *,
    left: str,
    right: str,
    decision: GeometricDecision,
    reason: GeometricReason,
    selected_right_d4: str | None,
    selected_model: str | None,
    metrics: tuple[tuple[str, int | float | str], ...],
    candidate_evidence_tokens: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "left_opaque_sample_id": left,
        "right_opaque_sample_id": right,
        "decision": decision.value,
        "reason": reason.value,
        "selected_right_d4": selected_right_d4,
        "selected_model": selected_model,
        "metrics": {key: value for key, value in metrics},
        "candidate_evidence_tokens": list(candidate_evidence_tokens),
        "threshold_status": THRESHOLD_STATUS,
    }


def _evidence(
    request: GeometricVerifierRequest,
    policy: GeometricVerifierPolicy,
    backend: tuple[tuple[str, str | int | bool], ...],
    results: tuple[GeometricPairResult, ...],
) -> GeometricVerifierEvidence:
    counts = {decision.value: 0 for decision in GeometricDecision}
    for item in results:
        counts[item.decision.value] += 1
    return GeometricVerifierEvidence(
        request_sha256=request.request_sha256,
        policy=policy,
        backend=tuple(sorted(backend)),
        results=results,
        counts=tuple(sorted(counts.items())),
        evidence_bindings=request.evidence_bindings,
    )


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _bindings(value: tuple[tuple[str, str], ...]) -> None:
    if not isinstance(value, tuple) or not value or value != tuple(sorted(value)):
        raise ValueError("evidence bindings must be a nonempty sorted tuple")
    if len({name for name, _ in value}) != len(value):
        raise ValueError("duplicate evidence binding name")
    for name, digest in value:
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("evidence binding name is invalid")
        _digest(digest, "evidence binding digest")


def _binding_tuple(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise TypeError("evidence_bindings must be a JSON array")
    rows: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError("each evidence binding must be a two-item array")
        rows.append((item[0], item[1]))
    return tuple(rows)


def _exact(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} fields differ from strict schema")
