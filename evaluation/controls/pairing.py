"""Bounded, deterministic oracle verification pair construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Any, Iterable

from shared.contracts.contracts import Modality
from evaluation.splits.tracklet_split import (
    EvaluationStage,
    SplitManifest,
    SplitRole,
    TrackletRecord,
)
from evaluation.protected_verification import VerificationDirection
from shared.foundation.provenance import content_sha256


class PairStratum(StrEnum):
    POSITIVE = "POSITIVE"
    SAME_BREED = "SAME_BREED"
    SAME_COAT = "SAME_COAT"
    SAME_SIZE = "SAME_SIZE"
    SHARED_CAGE_HISTORY = "SHARED_CAGE_HISTORY"
    RANDOM = "RANDOM"


@dataclass(frozen=True, slots=True)
class DogAttributes:
    registered_dog_id: str
    breed_primary: str | None
    breed_confidence: float | None
    mixed_breed: bool
    coat_colors: tuple[str, ...]
    coat_patterns: tuple[str, ...]
    size_class: str | None

    def __post_init__(self) -> None:
        _require_nonempty(self.registered_dog_id, "registered_dog_id")
        if not isinstance(self.mixed_breed, bool):
            raise TypeError("mixed_breed must be boolean")
        if self.breed_primary is None and self.breed_confidence is not None:
            raise ValueError("breed confidence requires a breed label")
        if self.breed_primary is not None:
            _require_nonempty(self.breed_primary, "breed_primary")
            if self.breed_confidence is None:
                raise ValueError("breed label requires explicit confidence")
        if self.breed_confidence is not None and (
            isinstance(self.breed_confidence, bool)
            or not isinstance(self.breed_confidence, (int, float))
            or not isfinite(self.breed_confidence)
            or not 0 <= self.breed_confidence <= 1
        ):
            raise ValueError("breed_confidence must be finite and in [0, 1]")
        _require_unique_labels(self.coat_colors, "coat_colors")
        _require_unique_labels(self.coat_patterns, "coat_patterns")
        if self.size_class is not None:
            _require_nonempty(self.size_class, "size_class")

    @property
    def breed_key(self) -> str | None:
        if self.mixed_breed or self.breed_primary is None:
            return None
        return _normalize_label(self.breed_primary)

    @property
    def coat_key(self) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        colors = tuple(sorted(_normalize_label(item) for item in self.coat_colors))
        patterns = tuple(
            sorted(_normalize_label(item) for item in self.coat_patterns)
        )
        if not colors and not patterns:
            return None
        return (colors, patterns)

    @property
    def size_key(self) -> str | None:
        return (
            None
            if self.size_class is None
            else _normalize_label(self.size_class)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "registered_dog_id": self.registered_dog_id,
            "breed_primary": self.breed_primary,
            "breed_confidence": self.breed_confidence,
            "mixed_breed": self.mixed_breed,
            "coat_colors": list(self.coat_colors),
            "coat_patterns": list(self.coat_patterns),
            "size_class": self.size_class,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DogAttributes:
        _require_exact_keys(
            payload,
            {
                "registered_dog_id",
                "breed_primary",
                "breed_confidence",
                "mixed_breed",
                "coat_colors",
                "coat_patterns",
                "size_class",
            },
            "dog attributes",
        )
        coat_colors = payload["coat_colors"]
        coat_patterns = payload["coat_patterns"]
        if not isinstance(coat_colors, list) or not isinstance(
            coat_patterns, list
        ):
            raise TypeError("coat colors and patterns must be lists")
        return cls(
            registered_dog_id=payload["registered_dog_id"],
            breed_primary=payload["breed_primary"],
            breed_confidence=payload["breed_confidence"],
            mixed_breed=payload["mixed_breed"],
            coat_colors=tuple(coat_colors),
            coat_patterns=tuple(coat_patterns),
            size_class=payload["size_class"],
        )


@dataclass(frozen=True, slots=True)
class NegativeQuota:
    stratum: PairStratum
    pairs_per_query: int

    def __post_init__(self) -> None:
        if self.stratum is PairStratum.POSITIVE:
            raise ValueError("positive quota is configured separately")
        _require_positive_int(self.pairs_per_query, "pairs_per_query")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "stratum": self.stratum.value,
            "pairs_per_query": self.pairs_per_query,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NegativeQuota:
        _require_exact_keys(
            payload,
            {"stratum", "pairs_per_query"},
            "negative quota",
        )
        return cls(
            stratum=PairStratum(payload["stratum"]),
            pairs_per_query=payload["pairs_per_query"],
        )


@dataclass(frozen=True, slots=True)
class PairingPolicy:
    name: str
    stage: EvaluationStage
    direction: VerificationDirection
    positive_pairs_per_query: int
    negative_quotas: tuple[NegativeQuota, ...]
    maximum_queries_per_dog: int
    maximum_pairs_per_query: int
    maximum_candidate_scans_per_stratum: int
    minimum_breed_confidence: float
    seed: int

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "name")
        if self.stage is EvaluationStage.TRAINING:
            raise ValueError("oracle pairing is calibration or test only")
        _require_positive_int(
            self.positive_pairs_per_query,
            "positive_pairs_per_query",
        )
        _require_positive_int(
            self.maximum_queries_per_dog,
            "maximum_queries_per_dog",
        )
        _require_positive_int(
            self.maximum_pairs_per_query,
            "maximum_pairs_per_query",
        )
        _require_positive_int(
            self.maximum_candidate_scans_per_stratum,
            "maximum_candidate_scans_per_stratum",
        )
        if not self.negative_quotas:
            raise ValueError("at least one negative quota is required")
        strata = tuple(quota.stratum for quota in self.negative_quotas)
        if len(strata) != len(set(strata)):
            raise ValueError("negative strata must be unique")
        if PairStratum.RANDOM in strata and strata[-1] is not PairStratum.RANDOM:
            raise ValueError("RANDOM must be the final negative stratum")
        requested = self.positive_pairs_per_query + sum(
            quota.pairs_per_query for quota in self.negative_quotas
        )
        if requested > self.maximum_pairs_per_query:
            raise ValueError("declared pair quotas exceed maximum_pairs_per_query")
        if (
            isinstance(self.minimum_breed_confidence, bool)
            or not isinstance(self.minimum_breed_confidence, (int, float))
            or not isfinite(self.minimum_breed_confidence)
            or not 0 <= self.minimum_breed_confidence <= 1
        ):
            raise ValueError(
                "minimum_breed_confidence must be finite and in [0, 1]"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def reference_modality(self) -> Modality:
        return _direction_modalities(self.direction)[0]

    @property
    def query_modality(self) -> Modality:
        return _direction_modalities(self.direction)[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.pairing_policy.v1",
            "name": self.name,
            "stage": self.stage.value,
            "direction": self.direction.value,
            "positive_pairs_per_query": self.positive_pairs_per_query,
            "negative_quotas": [
                quota.to_dict() for quota in self.negative_quotas
            ],
            "maximum_queries_per_dog": self.maximum_queries_per_dog,
            "maximum_pairs_per_query": self.maximum_pairs_per_query,
            "maximum_candidate_scans_per_stratum": (
                self.maximum_candidate_scans_per_stratum
            ),
            "minimum_breed_confidence": self.minimum_breed_confidence,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairingPolicy:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "name",
                "stage",
                "direction",
                "positive_pairs_per_query",
                "negative_quotas",
                "maximum_queries_per_dog",
                "maximum_pairs_per_query",
                "maximum_candidate_scans_per_stratum",
                "minimum_breed_confidence",
                "seed",
            },
            "pairing policy",
        )
        if payload["schema_version"] != "evaluation.pairing_policy.v1":
            raise ValueError("unsupported pairing policy schema")
        quotas = payload["negative_quotas"]
        if not isinstance(quotas, list):
            raise TypeError("negative_quotas must be a list")
        return cls(
            name=payload["name"],
            stage=EvaluationStage(payload["stage"]),
            direction=VerificationDirection(payload["direction"]),
            positive_pairs_per_query=payload["positive_pairs_per_query"],
            negative_quotas=tuple(
                NegativeQuota.from_dict(item) for item in quotas
            ),
            maximum_queries_per_dog=payload["maximum_queries_per_dog"],
            maximum_pairs_per_query=payload["maximum_pairs_per_query"],
            maximum_candidate_scans_per_stratum=(
                payload["maximum_candidate_scans_per_stratum"]
            ),
            minimum_breed_confidence=payload["minimum_breed_confidence"],
            seed=payload["seed"],
        )


@dataclass(frozen=True, slots=True)
class PairScoringRequest:
    pair_id: str
    query_artifact_token: str
    reference_artifact_token: str

    def __post_init__(self) -> None:
        _require_nonempty(self.pair_id, "pair_id")
        _require_nonempty(
            self.query_artifact_token,
            "query_artifact_token",
        )
        _require_nonempty(
            self.reference_artifact_token,
            "reference_artifact_token",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "query_artifact_token": self.query_artifact_token,
            "reference_artifact_token": self.reference_artifact_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairScoringRequest:
        _require_exact_keys(
            payload,
            {
                "pair_id",
                "query_artifact_token",
                "reference_artifact_token",
            },
            "pair scoring request",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PairArtifactBinding:
    artifact_token: str
    sample_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.artifact_token, "artifact_token")
        _require_nonempty(self.sample_id, "sample_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_token": self.artifact_token,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairArtifactBinding:
        _require_exact_keys(
            payload,
            {"artifact_token", "sample_id"},
            "pair artifact binding",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PairGroundTruth:
    pair_id: str
    query_dog_id: str
    reference_dog_id: str
    query_session_id: str
    reference_session_id: str
    stratum: PairStratum

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "query_dog_id",
            "reference_dog_id",
            "query_session_id",
            "reference_session_id",
        ):
            _require_nonempty(getattr(self, name), name)
        if self.query_session_id == self.reference_session_id:
            raise ValueError("verification pair must be session-disjoint")
        if self.stratum is PairStratum.POSITIVE and not self.same_identity:
            raise ValueError("positive pair must contain one identity")
        if self.stratum is not PairStratum.POSITIVE and self.same_identity:
            raise ValueError("negative pair must contain different identities")

    @property
    def same_identity(self) -> bool:
        return self.query_dog_id == self.reference_dog_id

    def to_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "query_dog_id": self.query_dog_id,
            "reference_dog_id": self.reference_dog_id,
            "query_session_id": self.query_session_id,
            "reference_session_id": self.reference_session_id,
            "stratum": self.stratum.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairGroundTruth:
        _require_exact_keys(
            payload,
            {
                "pair_id",
                "query_dog_id",
                "reference_dog_id",
                "query_session_id",
                "reference_session_id",
                "stratum",
            },
            "pair ground truth",
        )
        return cls(
            pair_id=payload["pair_id"],
            query_dog_id=payload["query_dog_id"],
            reference_dog_id=payload["reference_dog_id"],
            query_session_id=payload["query_session_id"],
            reference_session_id=payload["reference_session_id"],
            stratum=PairStratum(payload["stratum"]),
        )


@dataclass(frozen=True, slots=True)
class PairQuotaResult:
    query_sample_id: str
    stratum: PairStratum
    requested: int
    produced: int
    candidates_scanned: int
    scan_limit_reached: bool

    def __post_init__(self) -> None:
        _require_nonempty(self.query_sample_id, "query_sample_id")
        for name in ("requested", "produced", "candidates_scanned"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.produced > self.requested:
            raise ValueError("produced pairs cannot exceed requested pairs")
        if not isinstance(self.scan_limit_reached, bool):
            raise TypeError("scan_limit_reached must be boolean")

    @property
    def shortfall(self) -> int:
        return self.requested - self.produced

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "query_sample_id": self.query_sample_id,
            "stratum": self.stratum.value,
            "requested": self.requested,
            "produced": self.produced,
            "shortfall": self.shortfall,
            "candidates_scanned": self.candidates_scanned,
            "scan_limit_reached": self.scan_limit_reached,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PairQuotaResult:
        _require_exact_keys(
            payload,
            {
                "query_sample_id",
                "stratum",
                "requested",
                "produced",
                "shortfall",
                "candidates_scanned",
                "scan_limit_reached",
            },
            "pair quota result",
        )
        result = cls(
            query_sample_id=payload["query_sample_id"],
            stratum=PairStratum(payload["stratum"]),
            requested=payload["requested"],
            produced=payload["produced"],
            candidates_scanned=payload["candidates_scanned"],
            scan_limit_reached=payload["scan_limit_reached"],
        )
        if payload["shortfall"] != result.shortfall:
            raise ValueError("pair quota shortfall is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class PairConstructionResult:
    split_manifest_sha256: str
    pairing_policy_sha256: str
    attributes_sha256: str
    eligible_query_count: int
    selected_query_count: int
    dropped_query_count: int
    scoring_requests: tuple[PairScoringRequest, ...]
    artifact_bindings: tuple[PairArtifactBinding, ...]
    ground_truth: tuple[PairGroundTruth, ...]
    quotas: tuple[PairQuotaResult, ...]

    @property
    def result_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.pair_construction.v1",
            "split_manifest_sha256": self.split_manifest_sha256,
            "pairing_policy_sha256": self.pairing_policy_sha256,
            "attributes_sha256": self.attributes_sha256,
            "eligible_query_count": self.eligible_query_count,
            "selected_query_count": self.selected_query_count,
            "dropped_query_count": self.dropped_query_count,
            "scoring_requests": [
                request.to_dict() for request in self.scoring_requests
            ],
            "artifact_bindings": [
                binding.to_dict() for binding in self.artifact_bindings
            ],
            "ground_truth": [
                truth.to_dict() for truth in self.ground_truth
            ],
            "quotas": [quota.to_dict() for quota in self.quotas],
        }

    def scoring_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.pair_scoring_requests.v1",
            "pair_set_sha256": self.result_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "pairing_policy_sha256": self.pairing_policy_sha256,
            "requests": [
                request.to_dict() for request in self.scoring_requests
            ],
        }

    def ground_truth_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.pair_ground_truth.v1",
            "pair_set_sha256": self.result_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "pairing_policy_sha256": self.pairing_policy_sha256,
            "attributes_sha256": self.attributes_sha256,
            "ground_truth": [
                truth.to_dict() for truth in self.ground_truth
            ],
        }

    def artifact_binding_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.pair_artifact_bindings.v1",
            "pair_set_sha256": self.result_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "bindings": [
                binding.to_dict() for binding in self.artifact_bindings
            ],
        }

    def summary_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "evaluation.pair_construction_summary.v1",
            "pair_set_sha256": self.result_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "pairing_policy_sha256": self.pairing_policy_sha256,
            "attributes_sha256": self.attributes_sha256,
            "eligible_query_count": self.eligible_query_count,
            "selected_query_count": self.selected_query_count,
            "dropped_query_count": self.dropped_query_count,
            "pair_count": len(self.scoring_requests),
            "quota_results": [
                quota.to_dict() for quota in self.quotas
            ],
        }


def construct_verification_pairs(
    manifest: SplitManifest,
    *,
    attributes: tuple[DogAttributes, ...],
    policy: PairingPolicy,
) -> PairConstructionResult:
    """Construct bounded pairs without enumerating the Cartesian product."""

    blockers = manifest.gate_blockers()
    if blockers:
        raise ValueError(
            "split manifest is blocked: " + "; ".join(blockers[:10])
        )
    attributes_by_dog = _attributes_by_dog(attributes)
    stage_records = tuple(
        record
        for record in manifest.records
        if record.role.stage is policy.stage
    )
    gallery_role, query_role = _pair_roles(policy.stage)
    gallery = tuple(
        record
        for record in stage_records
        if record.role is gallery_role
        and record.modality is policy.reference_modality
    )
    all_queries = tuple(
        record
        for record in stage_records
        if record.role is query_role
        and record.modality is policy.query_modality
    )
    if not gallery or not all_queries:
        raise ValueError("pairing direction has no eligible gallery or query")
    relevant_dogs = {
        record.registered_dog_id for record in gallery + all_queries
    }
    attribute_dogs = set(attributes_by_dog)
    missing_attributes = relevant_dogs - attribute_dogs
    extra_attributes = attribute_dogs - relevant_dogs
    if missing_attributes or extra_attributes:
        raise ValueError(
            "dog attribute identity mismatch; "
            f"missing={sorted(missing_attributes)}, "
            f"extra={sorted(extra_attributes)}"
        )
    references_by_dog = _reference_session_representatives(
        gallery,
        seed=policy.seed,
    )
    selected_queries = _bounded_queries(
        all_queries,
        per_dog=policy.maximum_queries_per_dog,
        seed=policy.seed,
    )
    gallery_dogs = tuple(sorted(references_by_dog))
    cage_history = _cage_history(stage_records)
    indices = _negative_indices(
        gallery_dogs,
        attributes_by_dog,
        cage_history,
        minimum_breed_confidence=policy.minimum_breed_confidence,
    )
    scoring_requests: list[PairScoringRequest] = []
    ground_truth: list[PairGroundTruth] = []
    quota_results: list[PairQuotaResult] = []
    for query in selected_queries:
        positive_references = _distinct_session_references(
            references_by_dog.get(query.registered_dog_id, ()),
            excluded_session_id=query.session_id,
            ordering_key=(
                policy.seed,
                query.sample_id,
                PairStratum.POSITIVE.value,
            ),
        )
        positive_selected = positive_references[
            : policy.positive_pairs_per_query
        ]
        quota_results.append(
            PairQuotaResult(
                query.sample_id,
                PairStratum.POSITIVE,
                policy.positive_pairs_per_query,
                len(positive_selected),
                len(positive_references),
                False,
            )
        )
        for reference in positive_selected:
            request, truth = _make_pair(
                query,
                reference,
                PairStratum.POSITIVE,
                policy,
            )
            scoring_requests.append(request)
            ground_truth.append(truth)

        used_negative_dogs: set[str] = set()
        earlier_strata: list[PairStratum] = []
        for quota in policy.negative_quotas:
            pool = _negative_pool(
                quota.stratum,
                query=query,
                attributes=attributes_by_dog,
                cage_history=cage_history,
                indices=indices,
                all_gallery_dogs=gallery_dogs,
                minimum_breed_confidence=policy.minimum_breed_confidence,
            )
            ordered_pool = tuple(
                _rotated_values(
                    pool,
                    policy.seed,
                    query.sample_id,
                    quota.stratum.value,
                )
            )
            selected: list[TrackletRecord] = []
            candidates_scanned = 0
            for dog_id in ordered_pool:
                if (
                    candidates_scanned
                    == policy.maximum_candidate_scans_per_stratum
                ):
                    break
                candidates_scanned += 1
                if (
                    dog_id == query.registered_dog_id
                    or dog_id in used_negative_dogs
                    or any(
                        _matches_stratum(
                            earlier,
                            query=query,
                            candidate_dog_id=dog_id,
                            attributes=attributes_by_dog,
                            cage_history=cage_history,
                            minimum_breed_confidence=(
                                policy.minimum_breed_confidence
                            ),
                        )
                        for earlier in earlier_strata
                    )
                ):
                    continue
                reference = _first_reference(
                    references_by_dog[dog_id],
                    excluded_session_id=query.session_id,
                    ordering_key=(
                        policy.seed,
                        query.sample_id,
                        quota.stratum.value,
                        dog_id,
                    ),
                )
                if reference is None:
                    continue
                used_negative_dogs.add(dog_id)
                selected.append(reference)
                if len(selected) == quota.pairs_per_query:
                    break
            scan_limit_reached = (
                len(selected) < quota.pairs_per_query
                and candidates_scanned
                == policy.maximum_candidate_scans_per_stratum
                and len(ordered_pool) > candidates_scanned
            )
            quota_results.append(
                PairQuotaResult(
                    query.sample_id,
                    quota.stratum,
                    quota.pairs_per_query,
                    len(selected),
                    candidates_scanned,
                    scan_limit_reached,
                )
            )
            for reference in selected:
                request, truth = _make_pair(
                    query,
                    reference,
                    quota.stratum,
                    policy,
                )
                scoring_requests.append(request)
                ground_truth.append(truth)
            earlier_strata.append(quota.stratum)
    _validate_constructed_pairs(scoring_requests, ground_truth, policy)
    used_tokens = sorted(
        {
            token
            for request in scoring_requests
            for token in (
                request.query_artifact_token,
                request.reference_artifact_token,
            )
        }
    )
    token_to_sample: dict[str, str] = {}
    for record in gallery + all_queries:
        token = _artifact_token(policy, record.sample_id)
        prior = token_to_sample.setdefault(token, record.sample_id)
        if prior != record.sample_id:
            raise RuntimeError("artifact token collision")
    artifact_bindings = tuple(
        PairArtifactBinding(token, token_to_sample[token])
        for token in used_tokens
    )
    attributes_payload = [
        item.to_dict()
        for item in sorted(
            attributes,
            key=lambda item: item.registered_dog_id,
        )
    ]
    return PairConstructionResult(
        split_manifest_sha256=manifest.manifest_sha256,
        pairing_policy_sha256=policy.policy_sha256,
        attributes_sha256=content_sha256(attributes_payload),
        eligible_query_count=len(all_queries),
        selected_query_count=len(selected_queries),
        dropped_query_count=len(all_queries) - len(selected_queries),
        scoring_requests=tuple(scoring_requests),
        artifact_bindings=artifact_bindings,
        ground_truth=tuple(ground_truth),
        quotas=tuple(quota_results),
    )


def dog_attributes_from_payload(
    payload: dict[str, Any],
) -> tuple[DogAttributes, ...]:
    _require_exact_keys(
        payload,
        {"schema_version", "dogs"},
        "dog attribute manifest",
    )
    if payload["schema_version"] != "data.dog_attributes.v1":
        raise ValueError("unsupported dog attribute schema")
    dogs = payload["dogs"]
    if not isinstance(dogs, list):
        raise TypeError("dogs must be a list")
    return tuple(DogAttributes.from_dict(item) for item in dogs)


def pair_construction_from_bundle_payloads(
    scoring_payload: dict[str, Any],
    binding_payload: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> PairConstructionResult:
    """Reconstruct and authenticate a separated protected pair bundle."""

    _require_exact_keys(
        scoring_payload,
        {
            "schema_version",
            "pair_set_sha256",
            "split_manifest_sha256",
            "pairing_policy_sha256",
            "requests",
        },
        "pair scoring payload",
    )
    _require_exact_keys(
        binding_payload,
        {
            "schema_version",
            "pair_set_sha256",
            "split_manifest_sha256",
            "bindings",
        },
        "pair artifact binding payload",
    )
    _require_exact_keys(
        ground_truth_payload,
        {
            "schema_version",
            "pair_set_sha256",
            "split_manifest_sha256",
            "pairing_policy_sha256",
            "attributes_sha256",
            "ground_truth",
        },
        "pair ground-truth payload",
    )
    _require_exact_keys(
        summary_payload,
        {
            "schema_version",
            "pair_set_sha256",
            "split_manifest_sha256",
            "pairing_policy_sha256",
            "attributes_sha256",
            "eligible_query_count",
            "selected_query_count",
            "dropped_query_count",
            "pair_count",
            "quota_results",
        },
        "pair construction summary",
    )
    expected_versions = (
        (scoring_payload, "evaluation.pair_scoring_requests.v1"),
        (binding_payload, "evaluation.pair_artifact_bindings.v1"),
        (ground_truth_payload, "evaluation.pair_ground_truth.v1"),
        (summary_payload, "evaluation.pair_construction_summary.v1"),
    )
    for payload, expected in expected_versions:
        if payload["schema_version"] != expected:
            raise ValueError(f"unsupported pair bundle schema: {expected}")
    pair_set_sha256 = _common_bundle_field(
        "pair_set_sha256",
        scoring_payload,
        binding_payload,
        ground_truth_payload,
        summary_payload,
    )
    split_manifest_sha256 = _common_bundle_field(
        "split_manifest_sha256",
        scoring_payload,
        binding_payload,
        ground_truth_payload,
        summary_payload,
    )
    pairing_policy_sha256 = _common_bundle_field(
        "pairing_policy_sha256",
        scoring_payload,
        ground_truth_payload,
        summary_payload,
    )
    attributes_sha256 = _common_bundle_field(
        "attributes_sha256",
        ground_truth_payload,
        summary_payload,
    )
    for name, digest in (
        ("pair_set_sha256", pair_set_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
        ("pairing_policy_sha256", pairing_policy_sha256),
        ("attributes_sha256", attributes_sha256),
    ):
        _validate_sha256(digest, name)
    requests = _parse_object_list(
        scoring_payload["requests"],
        PairScoringRequest.from_dict,
        "requests",
    )
    bindings = _parse_object_list(
        binding_payload["bindings"],
        PairArtifactBinding.from_dict,
        "bindings",
    )
    ground_truth = _parse_object_list(
        ground_truth_payload["ground_truth"],
        PairGroundTruth.from_dict,
        "ground_truth",
    )
    quotas = _parse_object_list(
        summary_payload["quota_results"],
        PairQuotaResult.from_dict,
        "quota_results",
    )
    for name in (
        "eligible_query_count",
        "selected_query_count",
        "dropped_query_count",
        "pair_count",
    ):
        _require_nonnegative_int(summary_payload[name], name)
    if summary_payload["pair_count"] != len(requests):
        raise ValueError("pair bundle pair_count is inconsistent")
    if (
        summary_payload["selected_query_count"]
        + summary_payload["dropped_query_count"]
        != summary_payload["eligible_query_count"]
    ):
        raise ValueError("pair bundle query counts are inconsistent")
    result = PairConstructionResult(
        split_manifest_sha256=split_manifest_sha256,
        pairing_policy_sha256=pairing_policy_sha256,
        attributes_sha256=attributes_sha256,
        eligible_query_count=summary_payload["eligible_query_count"],
        selected_query_count=summary_payload["selected_query_count"],
        dropped_query_count=summary_payload["dropped_query_count"],
        scoring_requests=requests,
        artifact_bindings=bindings,
        ground_truth=ground_truth,
        quotas=quotas,
    )
    _validate_reconstructed_result(result)
    if result.result_sha256 != pair_set_sha256:
        raise ValueError("pair bundle content hash mismatch")
    return result


def _attributes_by_dog(
    attributes: tuple[DogAttributes, ...],
) -> dict[str, DogAttributes]:
    if not attributes:
        raise ValueError("dog attributes must not be empty")
    result = {item.registered_dog_id: item for item in attributes}
    if len(result) != len(attributes):
        raise ValueError("dog attribute IDs must be unique")
    return result


def _records_by_dog(
    records: Iterable[TrackletRecord],
) -> dict[str, tuple[TrackletRecord, ...]]:
    grouped: dict[str, list[TrackletRecord]] = {}
    for record in records:
        grouped.setdefault(record.registered_dog_id, []).append(record)
    return {
        dog_id: tuple(values) for dog_id, values in grouped.items()
    }


def _reference_session_representatives(
    records: tuple[TrackletRecord, ...],
    *,
    seed: int,
) -> dict[str, tuple[TrackletRecord, ...]]:
    grouped: dict[tuple[str, str], list[TrackletRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.registered_dog_id, record.session_id), []
        ).append(record)
    by_dog: dict[str, list[TrackletRecord]] = {}
    for (dog_id, session_id), values in grouped.items():
        representative = min(
            values,
            key=lambda record: _stable_rank(
                seed,
                "reference-session-template",
                dog_id,
                session_id,
                record.sample_id,
            ),
        )
        by_dog.setdefault(dog_id, []).append(representative)
    return {
        dog_id: tuple(sorted(values, key=lambda record: record.session_id))
        for dog_id, values in by_dog.items()
    }


def _bounded_queries(
    records: tuple[TrackletRecord, ...],
    *,
    per_dog: int,
    seed: int,
) -> tuple[TrackletRecord, ...]:
    grouped = _records_by_dog(records)
    selected: list[TrackletRecord] = []
    for dog_id in sorted(grouped):
        by_session: dict[str, list[TrackletRecord]] = {}
        for record in grouped[dog_id]:
            by_session.setdefault(record.session_id, []).append(record)
        session_representatives = [
            min(
                values,
                key=lambda record: _stable_rank(
                    seed,
                    "query-template",
                    dog_id,
                    record.sample_id,
                ),
            )
            for values in by_session.values()
        ]
        session_representatives.sort(
            key=lambda record: _stable_rank(
                seed,
                "query-session",
                dog_id,
                record.session_id,
                record.sample_id,
            )
        )
        selected.extend(session_representatives[:per_dog])
    selected.sort(key=lambda record: record.sample_id)
    return tuple(selected)


def _negative_indices(
    gallery_dogs: tuple[str, ...],
    attributes: dict[str, DogAttributes],
    cage_history: dict[str, frozenset[str]],
    *,
    minimum_breed_confidence: float,
) -> dict[PairStratum, dict[Any, tuple[str, ...]]]:
    mutable: dict[PairStratum, dict[Any, list[str]]] = {
        PairStratum.SAME_BREED: {},
        PairStratum.SAME_COAT: {},
        PairStratum.SAME_SIZE: {},
        PairStratum.SHARED_CAGE_HISTORY: {},
    }
    for dog_id in gallery_dogs:
        item = attributes[dog_id]
        if (
            item.breed_key is not None
            and item.breed_confidence is not None
            and item.breed_confidence >= minimum_breed_confidence
        ):
            mutable[PairStratum.SAME_BREED].setdefault(
                item.breed_key, []
            ).append(dog_id)
        if item.coat_key is not None:
            mutable[PairStratum.SAME_COAT].setdefault(
                item.coat_key, []
            ).append(dog_id)
        if item.size_key is not None:
            mutable[PairStratum.SAME_SIZE].setdefault(
                item.size_key, []
            ).append(dog_id)
        for cage_id in cage_history.get(dog_id, frozenset()):
            mutable[PairStratum.SHARED_CAGE_HISTORY].setdefault(
                cage_id, []
            ).append(dog_id)
    return {
        stratum: {
            key: tuple(sorted(set(dog_ids)))
            for key, dog_ids in values.items()
        }
        for stratum, values in mutable.items()
    }


def _negative_pool(
    stratum: PairStratum,
    *,
    query: TrackletRecord,
    attributes: dict[str, DogAttributes],
    cage_history: dict[str, frozenset[str]],
    indices: dict[PairStratum, dict[Any, tuple[str, ...]]],
    all_gallery_dogs: tuple[str, ...],
    minimum_breed_confidence: float,
) -> tuple[str, ...]:
    item = attributes[query.registered_dog_id]
    if stratum is PairStratum.RANDOM:
        return all_gallery_dogs
    if stratum is PairStratum.SAME_BREED:
        if (
            item.breed_key is None
            or item.breed_confidence is None
            or item.breed_confidence < minimum_breed_confidence
        ):
            return ()
        return indices[stratum].get(item.breed_key, ())
    if stratum is PairStratum.SAME_COAT:
        return (
            ()
            if item.coat_key is None
            else indices[stratum].get(item.coat_key, ())
        )
    if stratum is PairStratum.SAME_SIZE:
        return (
            ()
            if item.size_key is None
            else indices[stratum].get(item.size_key, ())
        )
    if stratum is PairStratum.SHARED_CAGE_HISTORY:
        dogs: set[str] = set()
        for cage_id in cage_history.get(
            query.registered_dog_id, frozenset()
        ):
            dogs.update(indices[stratum].get(cage_id, ()))
        return tuple(sorted(dogs))
    raise ValueError(f"unsupported negative stratum: {stratum}")


def _matches_stratum(
    stratum: PairStratum,
    *,
    query: TrackletRecord,
    candidate_dog_id: str,
    attributes: dict[str, DogAttributes],
    cage_history: dict[str, frozenset[str]],
    minimum_breed_confidence: float,
) -> bool:
    if stratum is PairStratum.RANDOM:
        return True
    query_item = attributes[query.registered_dog_id]
    candidate_item = attributes[candidate_dog_id]
    if stratum is PairStratum.SAME_BREED:
        return (
            query_item.breed_key is not None
            and candidate_item.breed_key == query_item.breed_key
            and query_item.breed_confidence is not None
            and candidate_item.breed_confidence is not None
            and query_item.breed_confidence >= minimum_breed_confidence
            and candidate_item.breed_confidence >= minimum_breed_confidence
        )
    if stratum is PairStratum.SAME_COAT:
        return (
            query_item.coat_key is not None
            and candidate_item.coat_key == query_item.coat_key
        )
    if stratum is PairStratum.SAME_SIZE:
        return (
            query_item.size_key is not None
            and candidate_item.size_key == query_item.size_key
        )
    if stratum is PairStratum.SHARED_CAGE_HISTORY:
        return bool(
            cage_history.get(query.registered_dog_id, frozenset())
            & cage_history.get(candidate_dog_id, frozenset())
        )
    raise ValueError(f"unsupported negative stratum: {stratum}")


def _distinct_session_references(
    records: Iterable[TrackletRecord],
    *,
    excluded_session_id: str,
    ordering_key: tuple[Any, ...],
) -> tuple[TrackletRecord, ...]:
    by_session: dict[str, list[TrackletRecord]] = {}
    for record in records:
        if record.session_id != excluded_session_id:
            by_session.setdefault(record.session_id, []).append(record)
    representatives = [
        min(
            values,
            key=lambda record: _stable_rank(
                *ordering_key,
                record.session_id,
                record.sample_id,
            ),
        )
        for values in by_session.values()
    ]
    return tuple(
        sorted(
            representatives,
            key=lambda record: _stable_rank(
                *ordering_key,
                record.session_id,
                record.sample_id,
            ),
        )
    )


def _first_reference(
    records: tuple[TrackletRecord, ...],
    *,
    excluded_session_id: str,
    ordering_key: tuple[Any, ...],
) -> TrackletRecord | None:
    eligible = tuple(
        record
        for record in records
        if record.session_id != excluded_session_id
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda record: _stable_rank(
            *ordering_key,
            record.session_id,
            record.sample_id,
        ),
    )


def _rotated_values(
    values: tuple[str, ...],
    *ordering_key: Any,
) -> Iterable[str]:
    if not values:
        return ()
    offset = int(_stable_rank(*ordering_key), 16) % len(values)
    return values[offset:] + values[:offset]


def _make_pair(
    query: TrackletRecord,
    reference: TrackletRecord,
    stratum: PairStratum,
    policy: PairingPolicy,
) -> tuple[PairScoringRequest, PairGroundTruth]:
    pair_digest = content_sha256(
        {
            "policy_sha256": policy.policy_sha256,
            "query_sample_id": query.sample_id,
            "reference_sample_id": reference.sample_id,
        }
    )
    pair_id = f"pair-{pair_digest[:24]}"
    return (
        PairScoringRequest(
            pair_id=pair_id,
            query_artifact_token=_artifact_token(policy, query.sample_id),
            reference_artifact_token=_artifact_token(
                policy, reference.sample_id
            ),
        ),
        PairGroundTruth(
            pair_id=pair_id,
            query_dog_id=query.registered_dog_id,
            reference_dog_id=reference.registered_dog_id,
            query_session_id=query.session_id,
            reference_session_id=reference.session_id,
            stratum=stratum,
        ),
    )


def _validate_constructed_pairs(
    requests: list[PairScoringRequest],
    ground_truth: list[PairGroundTruth],
    policy: PairingPolicy,
) -> None:
    request_ids = tuple(pair.pair_id for pair in requests)
    truth_ids = tuple(pair.pair_id for pair in ground_truth)
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("constructed pair IDs are not unique")
    if request_ids != truth_ids:
        raise RuntimeError("scoring and ground-truth pair order differs")
    truth_by_id = {truth.pair_id: truth for truth in ground_truth}
    by_query: dict[str, list[PairGroundTruth]] = {}
    for request in requests:
        truth = truth_by_id[request.pair_id]
        by_query.setdefault(request.query_artifact_token, []).append(truth)
        if truth.query_session_id == truth.reference_session_id:
            raise RuntimeError("constructed pair is not session-disjoint")
        if truth.stratum is PairStratum.POSITIVE and not truth.same_identity:
            raise RuntimeError("positive pair has different identities")
        if truth.stratum is not PairStratum.POSITIVE and truth.same_identity:
            raise RuntimeError("negative pair has the same identity")
    for query_pairs in by_query.values():
        if len(query_pairs) > policy.maximum_pairs_per_query:
            raise RuntimeError("constructed pairs exceed per-query cap")
        negative_dogs = [
            pair.reference_dog_id
            for pair in query_pairs
            if pair.stratum is not PairStratum.POSITIVE
        ]
        if len(negative_dogs) != len(set(negative_dogs)):
            raise RuntimeError("negative identity reused across strata")


def _cage_history(
    records: tuple[TrackletRecord, ...],
) -> dict[str, frozenset[str]]:
    mutable: dict[str, set[str]] = {}
    for record in records:
        mutable.setdefault(record.registered_dog_id, set()).add(
            record.cage_id
        )
    return {
        dog_id: frozenset(cages) for dog_id, cages in mutable.items()
    }


def _pair_roles(stage: EvaluationStage) -> tuple[SplitRole, SplitRole]:
    if stage is EvaluationStage.CALIBRATION:
        return (
            SplitRole.CALIBRATION_GALLERY,
            SplitRole.CALIBRATION_KNOWN_QUERY,
        )
    if stage is EvaluationStage.TEST:
        return (SplitRole.TEST_GALLERY, SplitRole.TEST_KNOWN_QUERY)
    raise ValueError("training stage has no oracle pair roles")


def _direction_modalities(
    direction: VerificationDirection,
) -> tuple[Modality, Modality]:
    mapping = {
        VerificationDirection.RGB_TO_RGB: (Modality.RGB, Modality.RGB),
        VerificationDirection.IR_TO_IR: (Modality.IR, Modality.IR),
        VerificationDirection.RGB_TO_IR: (Modality.IR, Modality.RGB),
        VerificationDirection.IR_TO_RGB: (Modality.RGB, Modality.IR),
    }
    return mapping[direction]


def _stable_rank(*parts: Any) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return sha256(encoded).hexdigest()


def _artifact_token(policy: PairingPolicy, sample_id: str) -> str:
    return "artifact-" + content_sha256(
        {
            "policy_sha256": policy.policy_sha256,
            "sample_id": sample_id,
        }
    )[:24]


def _common_bundle_field(
    name: str,
    *payloads: dict[str, Any],
) -> Any:
    values = tuple(payload[name] for payload in payloads)
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"pair bundle {name} mismatch")
    return values[0]


def _parse_object_list(
    value: Any,
    parser: Any,
    name: str,
) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return tuple(parser(item) for item in value)


def _validate_reconstructed_result(
    result: PairConstructionResult,
) -> None:
    request_ids = tuple(
        request.pair_id for request in result.scoring_requests
    )
    truth_ids = tuple(truth.pair_id for truth in result.ground_truth)
    if not request_ids:
        raise ValueError("pair bundle must contain at least one request")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("pair bundle request IDs must be unique")
    if request_ids != truth_ids:
        raise ValueError("pair bundle request and truth order differs")
    binding_tokens = tuple(
        binding.artifact_token for binding in result.artifact_bindings
    )
    if len(binding_tokens) != len(set(binding_tokens)):
        raise ValueError("pair bundle binding tokens must be unique")
    binding_samples = tuple(
        binding.sample_id for binding in result.artifact_bindings
    )
    if len(binding_samples) != len(set(binding_samples)):
        raise ValueError("pair bundle binding sample IDs must be unique")
    requested_tokens = {
        token
        for request in result.scoring_requests
        for token in (
            request.query_artifact_token,
            request.reference_artifact_token,
        )
    }
    if requested_tokens != set(binding_tokens):
        raise ValueError("pair bundle request and binding tokens differ")


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_unique_labels(values: tuple[str, ...], name: str) -> None:
    for value in values:
        _require_nonempty(value, name)
    normalized = tuple(_normalize_label(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique after normalization")


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
