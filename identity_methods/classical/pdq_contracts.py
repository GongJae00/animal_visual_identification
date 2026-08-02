"""Strict label-blind contracts for externally computed Meta PDQ evidence.

This module does not compute PDQ hashes.  Native hashing is a separately
audited boundary which must provide the eight dihedral hashes in the exact
order declared by :data:`PDQ_D4_ORIENTATIONS`.  Hashes are retained in that
order; they are never sorted or collapsed into a canonical fingerprint.

Only opaque SHA-256 sample tokens may cross this boundary.  Identity, role,
split, dataset, path, camera, cage, time, and accessory fields are deliberately
absent from every contract below.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


PDQ_BITS = 256
PDQ_HEX_DIGITS = 64
PDQ_ORIENTATION_COUNT = 8
PDQ_D4_ORIENTATIONS: tuple[str, ...] = (
    "ORIGINAL",
    "ROT90CCW",
    "ROT180",
    "ROT270CCW",
    "FLIP_X",
    "FLIP_Y",
    "FLIP_PLUS_DIAGONAL",
    "FLIP_MINUS_DIAGONAL",
)

PDQ_INITIAL_QUALITY_THRESHOLD = 50
PDQ_MAXIMUM_EXACT_DISTANCE = 31
PDQ_QUALITY_THRESHOLD_STATUS = (
    "INITIALIZATION_ONLY_NOT_CALIBRATION_ADMISSION"
)
PDQ_ELIGIBLE_SEARCHED = "PDQ_ELIGIBLE_SEARCHED"
PDQ_INELIGIBLE_LOW_QUALITY = "PDQ_INELIGIBLE_LOW_QUALITY"
PDQ_NOT_IN_AUDIT = "PDQ_NOT_IN_AUDIT"

_OPAQUE_ID = re.compile(r"[0-9a-f]{64}\Z")
_PDQ_HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PDQFingerprint:
    """One sample's ordered eight D4 hashes and upstream PDQ quality."""

    opaque_sample_id: str
    d4_hashes: tuple[str, ...]
    quality: int
    schema_version: str = "cvi.pdq_fingerprint.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_fingerprint.v1":
            raise ValueError("unsupported PDQ fingerprint schema")
        _require_opaque_id(self.opaque_sample_id, "opaque sample ID")
        if not isinstance(self.d4_hashes, tuple):
            raise TypeError("d4_hashes must be an immutable ordered tuple")
        if len(self.d4_hashes) != PDQ_ORIENTATION_COUNT:
            raise ValueError("PDQ requires exactly eight ordered D4 hashes")
        for hash_value in self.d4_hashes:
            if not isinstance(hash_value, str) or not _PDQ_HEX.fullmatch(hash_value):
                raise ValueError(
                    "each PDQ hash must be exactly 256 bits as 64 lowercase hex digits"
                )
        _require_quality(self.quality, "quality")

    @property
    def hash_integers(self) -> tuple[int, ...]:
        """Return integer views without changing the retained D4 order."""

        return tuple(int(value, 16) for value in self.d4_hashes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "opaque_sample_id": self.opaque_sample_id,
            "d4_hashes": list(self.d4_hashes),
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PDQFingerprint":
        _require_exact_fields(
            payload,
            {"schema_version", "opaque_sample_id", "d4_hashes", "quality"},
            "PDQ fingerprint",
        )
        hashes = payload["d4_hashes"]
        if not isinstance(hashes, list):
            raise TypeError("PDQ fingerprint d4_hashes must be a JSON array")
        return cls(
            opaque_sample_id=payload["opaque_sample_id"],
            d4_hashes=tuple(hashes),
            quality=payload["quality"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class PDQNearDuplicateCandidate:
    """Minimum ordered-orientation distance for one opaque sample pair."""

    left_opaque_sample_id: str
    right_opaque_sample_id: str
    minimum_hamming_distance: int
    left_orientation: str
    right_orientation: str
    left_quality: int
    right_quality: int
    minimum_quality: int
    distance_threshold: int
    quality_threshold: int
    schema_version: str = "cvi.pdq_near_duplicate_candidate.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_near_duplicate_candidate.v1":
            raise ValueError("unsupported PDQ candidate schema")
        _require_opaque_id(self.left_opaque_sample_id, "left candidate sample ID")
        _require_opaque_id(self.right_opaque_sample_id, "right candidate sample ID")
        if self.left_opaque_sample_id >= self.right_opaque_sample_id:
            raise ValueError("candidate IDs must be strictly increasing")
        _require_distance_threshold(self.distance_threshold)
        if (
            isinstance(self.minimum_hamming_distance, bool)
            or not isinstance(self.minimum_hamming_distance, int)
            or not 0 <= self.minimum_hamming_distance <= self.distance_threshold
        ):
            raise ValueError("candidate distance must be inside the bound threshold")
        for name, value in (
            ("left_orientation", self.left_orientation),
            ("right_orientation", self.right_orientation),
        ):
            if value not in PDQ_D4_ORIENTATIONS:
                raise ValueError(f"{name} is not a fixed PDQ D4 orientation")
        _require_quality(self.left_quality, "left_quality")
        _require_quality(self.right_quality, "right_quality")
        _require_quality(self.minimum_quality, "minimum_quality")
        _require_quality(self.quality_threshold, "quality_threshold")
        if self.minimum_quality != min(self.left_quality, self.right_quality):
            raise ValueError("minimum_quality does not match pair qualities")
        if self.minimum_quality < self.quality_threshold:
            raise ValueError("candidate contains a low-quality ineligible sample")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left_opaque_sample_id": self.left_opaque_sample_id,
            "right_opaque_sample_id": self.right_opaque_sample_id,
            "minimum_hamming_distance": self.minimum_hamming_distance,
            "left_orientation": self.left_orientation,
            "right_orientation": self.right_orientation,
            "left_quality": self.left_quality,
            "right_quality": self.right_quality,
            "minimum_quality": self.minimum_quality,
            "distance_threshold": self.distance_threshold,
            "quality_threshold": self.quality_threshold,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PDQNearDuplicateCandidate":
        expected = {
            "schema_version",
            "left_opaque_sample_id",
            "right_opaque_sample_id",
            "minimum_hamming_distance",
            "left_orientation",
            "right_orientation",
            "left_quality",
            "right_quality",
            "minimum_quality",
            "distance_threshold",
            "quality_threshold",
        }
        _require_exact_fields(payload, expected, "PDQ candidate")
        return cls(
            left_opaque_sample_id=payload["left_opaque_sample_id"],
            right_opaque_sample_id=payload["right_opaque_sample_id"],
            minimum_hamming_distance=payload["minimum_hamming_distance"],
            left_orientation=payload["left_orientation"],
            right_orientation=payload["right_orientation"],
            left_quality=payload["left_quality"],
            right_quality=payload["right_quality"],
            minimum_quality=payload["minimum_quality"],
            distance_threshold=payload["distance_threshold"],
            quality_threshold=payload["quality_threshold"],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class PDQSearchPolicy:
    """Bounded exact-search policy; 50/31 are initialization, not admission."""

    distance_threshold: int = PDQ_MAXIMUM_EXACT_DISTANCE
    quality_threshold: int = PDQ_INITIAL_QUALITY_THRESHOLD
    quality_threshold_status: str = PDQ_QUALITY_THRESHOLD_STATUS
    orientation_order: tuple[str, ...] = PDQ_D4_ORIENTATIONS
    mih_slot_count: int = 16
    mih_slot_bits: int = 16
    mih_query_slot_radius: int = 1
    maximum_samples: int = 50_000
    maximum_orientations: int = 400_000
    maximum_raw_posting_visits: int = 1_500_000_000
    maximum_unique_orientation_inspections: int = 800_000_000
    maximum_accepted_sample_candidates: int = 1_000_000
    schema_version: str = "cvi.public_canine_pdq_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_canine_pdq_policy.v1":
            raise ValueError("unsupported public canine PDQ policy")
        _require_distance_threshold(self.distance_threshold)
        _require_quality(self.quality_threshold, "quality_threshold")
        if self.quality_threshold_status != PDQ_QUALITY_THRESHOLD_STATUS:
            raise ValueError("PDQ quality threshold status differs")
        if self.orientation_order != PDQ_D4_ORIENTATIONS:
            raise ValueError("PDQ D4 orientation order differs")
        if (
            self.mih_slot_count,
            self.mih_slot_bits,
            self.mih_query_slot_radius,
        ) != (16, 16, 1):
            raise ValueError("PDQ MIH partition semantics are fixed at 16x16/radius1")
        limits = (
            ("maximum_samples", self.maximum_samples, 50_000),
            ("maximum_orientations", self.maximum_orientations, 400_000),
            (
                "maximum_raw_posting_visits",
                self.maximum_raw_posting_visits,
                1_500_000_000,
            ),
            (
                "maximum_unique_orientation_inspections",
                self.maximum_unique_orientation_inspections,
                800_000_000,
            ),
            (
                "maximum_accepted_sample_candidates",
                self.maximum_accepted_sample_candidates,
                1_000_000,
            ),
        )
        for name, value, ceiling in limits:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > ceiling:
                raise ValueError(f"{name} exceeds the frozen safety ceiling")

    @property
    def policy_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "distance_threshold": self.distance_threshold,
            "quality_threshold": self.quality_threshold,
            "quality_threshold_status": self.quality_threshold_status,
            "orientation_order": list(self.orientation_order),
            "mih_slot_count": self.mih_slot_count,
            "mih_slot_bits": self.mih_slot_bits,
            "mih_query_slot_radius": self.mih_query_slot_radius,
            "maximum_samples": self.maximum_samples,
            "maximum_orientations": self.maximum_orientations,
            "maximum_raw_posting_visits": self.maximum_raw_posting_visits,
            "maximum_unique_orientation_inspections": (
                self.maximum_unique_orientation_inspections
            ),
            "maximum_accepted_sample_candidates": (
                self.maximum_accepted_sample_candidates
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PDQSearchPolicy":
        expected = {
            "schema_version",
            "distance_threshold",
            "quality_threshold",
            "quality_threshold_status",
            "orientation_order",
            "mih_slot_count",
            "mih_slot_bits",
            "mih_query_slot_radius",
            "maximum_samples",
            "maximum_orientations",
            "maximum_raw_posting_visits",
            "maximum_unique_orientation_inspections",
            "maximum_accepted_sample_candidates",
        }
        _require_exact_fields(payload, expected, "PDQ search policy")
        orientation_order = payload["orientation_order"]
        if not isinstance(orientation_order, list):
            raise TypeError("PDQ orientation_order must be a JSON array")
        return cls(
            distance_threshold=payload["distance_threshold"],
            quality_threshold=payload["quality_threshold"],
            quality_threshold_status=payload["quality_threshold_status"],
            orientation_order=tuple(orientation_order),
            mih_slot_count=payload["mih_slot_count"],
            mih_slot_bits=payload["mih_slot_bits"],
            mih_query_slot_radius=payload["mih_query_slot_radius"],
            maximum_samples=payload["maximum_samples"],
            maximum_orientations=payload["maximum_orientations"],
            maximum_raw_posting_visits=payload["maximum_raw_posting_visits"],
            maximum_unique_orientation_inspections=payload[
                "maximum_unique_orientation_inspections"
            ],
            maximum_accepted_sample_candidates=payload[
                "maximum_accepted_sample_candidates"
            ],
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class PDQSearchResult:
    """Bounded candidate evidence with explicit low-quality non-admission."""

    candidates: tuple[PDQNearDuplicateCandidate, ...]
    eligible_sample_ids: tuple[str, ...]
    ineligible_low_quality_sample_ids: tuple[str, ...]
    preflight_raw_posting_visits: int
    unique_orientation_inspections: int
    indexed_orientation_count: int
    distance_threshold: int
    quality_threshold: int
    quality_threshold_status: str = PDQ_QUALITY_THRESHOLD_STATUS
    schema_version: str = "cvi.pdq_search_result.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_search_result.v1":
            raise ValueError("unsupported PDQ search-result schema")
        if self.quality_threshold_status != PDQ_QUALITY_THRESHOLD_STATUS:
            raise ValueError("PDQ search result misstates threshold admission")
        _require_distance_threshold(self.distance_threshold)
        _require_quality(self.quality_threshold, "quality_threshold")
        for name, values in (
            ("eligible_sample_ids", self.eligible_sample_ids),
            (
                "ineligible_low_quality_sample_ids",
                self.ineligible_low_quality_sample_ids,
            ),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise ValueError(f"{name} must be sorted and unique")
            for value in values:
                _require_opaque_id(value, name)
        if set(self.eligible_sample_ids) & set(self.ineligible_low_quality_sample_ids):
            raise ValueError("eligible and low-quality sample sets overlap")
        for name in (
            "preflight_raw_posting_visits",
            "unique_orientation_inspections",
            "indexed_orientation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.indexed_orientation_count != (
            len(self.eligible_sample_ids) * PDQ_ORIENTATION_COUNT
        ):
            raise ValueError("indexed orientation count differs from eligible samples")
        expected_pairs = tuple(
            sorted(
                self.candidates,
                key=lambda item: (
                    item.left_opaque_sample_id,
                    item.right_opaque_sample_id,
                ),
            )
        )
        if self.candidates != expected_pairs:
            raise ValueError("PDQ candidates must be deterministically pair-sorted")
        pair_keys = tuple(
            (item.left_opaque_sample_id, item.right_opaque_sample_id)
            for item in self.candidates
        )
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("PDQ result contains duplicate sample pairs")
        eligible = set(self.eligible_sample_ids)
        for item in self.candidates:
            if (
                item.left_opaque_sample_id not in eligible
                or item.right_opaque_sample_id not in eligible
            ):
                raise ValueError("PDQ candidate references an ineligible sample")
            if (
                item.distance_threshold != self.distance_threshold
                or item.quality_threshold != self.quality_threshold
            ):
                raise ValueError("PDQ candidate threshold differs from result")

    def sample_status(self, opaque_id: str) -> str:
        _require_opaque_id(opaque_id, "opaque sample ID")
        if _contains_sorted(self.ineligible_low_quality_sample_ids, opaque_id):
            return PDQ_INELIGIBLE_LOW_QUALITY
        if _contains_sorted(self.eligible_sample_ids, opaque_id):
            return PDQ_ELIGIBLE_SEARCHED
        return PDQ_NOT_IN_AUDIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidates": [item.to_dict() for item in self.candidates],
            "eligible_sample_ids": list(self.eligible_sample_ids),
            "ineligible_low_quality_sample_ids": list(
                self.ineligible_low_quality_sample_ids
            ),
            "preflight_raw_posting_visits": self.preflight_raw_posting_visits,
            "unique_orientation_inspections": self.unique_orientation_inspections,
            "indexed_orientation_count": self.indexed_orientation_count,
            "distance_threshold": self.distance_threshold,
            "quality_threshold": self.quality_threshold,
            "quality_threshold_status": self.quality_threshold_status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PDQSearchResult":
        expected = {
            "schema_version",
            "candidates",
            "eligible_sample_ids",
            "ineligible_low_quality_sample_ids",
            "preflight_raw_posting_visits",
            "unique_orientation_inspections",
            "indexed_orientation_count",
            "distance_threshold",
            "quality_threshold",
            "quality_threshold_status",
        }
        _require_exact_fields(payload, expected, "PDQ search result")
        for name in (
            "candidates",
            "eligible_sample_ids",
            "ineligible_low_quality_sample_ids",
        ):
            if not isinstance(payload[name], list):
                raise TypeError(f"PDQ search result {name} must be a JSON array")
        return cls(
            candidates=tuple(
                PDQNearDuplicateCandidate.from_dict(item)
                for item in payload["candidates"]
            ),
            eligible_sample_ids=tuple(payload["eligible_sample_ids"]),
            ineligible_low_quality_sample_ids=tuple(
                payload["ineligible_low_quality_sample_ids"]
            ),
            preflight_raw_posting_visits=payload["preflight_raw_posting_visits"],
            unique_orientation_inspections=payload[
                "unique_orientation_inspections"
            ],
            indexed_orientation_count=payload["indexed_orientation_count"],
            distance_threshold=payload["distance_threshold"],
            quality_threshold=payload["quality_threshold"],
            quality_threshold_status=payload["quality_threshold_status"],
            schema_version=payload["schema_version"],
        )


def _require_distance_threshold(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= PDQ_MAXIMUM_EXACT_DISTANCE
    ):
        raise ValueError("PDQ distance threshold must be inside exact range 0..31")


def _require_quality(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be an integer inside 0..100")


def _require_opaque_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 token")


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], name: str
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} fields differ from the strict schema")


def _contains_sorted(values: tuple[str, ...], target: str) -> bool:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low < len(values) and values[low] == target


assert len(PDQ_D4_ORIENTATIONS) == PDQ_ORIENTATION_COUNT
assert len(set(PDQ_D4_ORIENTATIONS)) == PDQ_ORIENTATION_COUNT
