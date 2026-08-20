"""Deterministic identity topology diagnostics in normalized embedding space."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from shared.foundation.provenance import content_sha256


IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION = (
    "cvi.embedding_identity_topology_manifest.v1"
)
FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION = (
    "cvi.fixed_multievidence_identity_topology_manifest.v1"
)
IDENTITY_TOPOLOGY_REPORT_SCHEMA_VERSION = (
    "cvi.embedding_identity_topology_audit.v1"
)

_REQUIRED_RECORD_FIELDS = {
    "sample_token",
    "identity_token",
    "session_token",
    "branch",
    "quality",
    "available",
    "embedding",
}


class IdentityTopologyError(ValueError):
    """Raised when an identity topology input violates its audit contract."""


@dataclass(frozen=True, slots=True)
class IdentityTopologyConfig:
    """Numerical policy for an embedding-space identity topology audit."""

    connectivity_cosine_distance_threshold: float = 0.2
    normalization_tolerance: float = 1e-6
    minimum_prototype_norm: float = 1e-12
    hubness_k: int = 1

    def __post_init__(self) -> None:
        for name in (
            "connectivity_cosine_distance_threshold",
            "normalization_tolerance",
            "minimum_prototype_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise IdentityTopologyError(f"{name} must be a finite number")
        if not 0.0 <= self.connectivity_cosine_distance_threshold <= 2.0:
            raise IdentityTopologyError(
                "connectivity_cosine_distance_threshold must be in [0, 2]"
            )
        if self.normalization_tolerance <= 0.0:
            raise IdentityTopologyError("normalization_tolerance must be positive")
        if self.minimum_prototype_norm <= 0.0:
            raise IdentityTopologyError("minimum_prototype_norm must be positive")
        if (
            isinstance(self.hubness_k, bool)
            or not isinstance(self.hubness_k, int)
            or self.hubness_k < 1
        ):
            raise IdentityTopologyError("hubness_k must be a positive integer")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "connectivity_cosine_distance_threshold": (
                self.connectivity_cosine_distance_threshold
            ),
            "hubness_k": self.hubness_k,
            "minimum_prototype_norm": self.minimum_prototype_norm,
            "normalization_tolerance": self.normalization_tolerance,
        }


@dataclass(frozen=True, slots=True)
class _Record:
    sample_token: str
    identity_token: str
    session_token: str
    branch: str
    quality: float | None
    available: bool
    embedding: np.ndarray | None
    rank_label: int | None


def _token(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise IdentityTopologyError(f"{context} must be a non-empty trimmed string")
    return value


def _quality(value: object, context: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise IdentityTopologyError(f"{context} must be null or finite and in [0, 1]")
    return float(value)


def _rank_label(value: object, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IdentityTopologyError(f"{context} must be null or a positive integer")
    return value


def _embedding(
    value: object,
    *,
    context: str,
    config: IdentityTopologyConfig,
) -> np.ndarray:
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise IdentityTopologyError(f"{context} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise IdentityTopologyError(f"{context} must be a non-empty 1-d vector")
    if not (
        np.issubdtype(vector.dtype, np.integer)
        or np.issubdtype(vector.dtype, np.floating)
    ):
        raise IdentityTopologyError(f"{context} must contain real numeric values")
    vector = vector.astype(np.float64, copy=False)
    if not np.isfinite(vector).all():
        raise IdentityTopologyError(f"{context} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm):
        raise IdentityTopologyError(f"{context} norm overflowed float64")
    if abs(norm - 1.0) > config.normalization_tolerance:
        raise IdentityTopologyError(
            f"{context} is not L2-normalized within tolerance "
            f"{config.normalization_tolerance}"
        )
    return vector / norm


def _parse_manifest(
    manifest: Mapping[str, Any], config: IdentityTopologyConfig
) -> tuple[_Record, ...]:
    if not isinstance(manifest, Mapping):
        raise TypeError("identity topology manifest must be an object")
    schema = manifest.get("schema_version")
    if schema == IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION:
        expected_fields = {"schema_version", "records"}
    elif schema == FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION:
        expected_fields = {
            "schema_version",
            "input_bindings",
            "input_bindings_sha256",
            "records",
        }
        bindings = manifest.get("input_bindings")
        if (
            not isinstance(bindings, dict)
            or not bindings
            or content_sha256(bindings) != manifest.get("input_bindings_sha256")
        ):
            raise IdentityTopologyError("fixed-panel topology bindings differ")
        expected_bindings = {
            "panel_bundle_content_sha256",
            "panel_sha256",
            "frozen_dinov2_sha256",
            "f5_checkpoint_sha256",
            "n3_lineage_content_sha256",
            "n3_runtime_manifest_content_sha256",
            "n3_runtime_manifest_raw_sha256",
            "n3_onnx_sha256",
            "execution",
        }
        if set(bindings) != expected_bindings:
            raise IdentityTopologyError("fixed-panel topology binding fields differ")
        for name, digest in bindings.items():
            if name == "execution":
                continue
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise IdentityTopologyError(
                    f"fixed-panel topology {name} must be lowercase SHA-256"
                )
        execution = bindings["execution"]
        if (
            not isinstance(execution, dict)
            or set(execution) != {"device", "n3_device", "batch_size"}
            or execution["device"] not in {"cpu", "cuda"}
            or execution["n3_device"] not in {"cpu", "cuda"}
            or isinstance(execution["batch_size"], bool)
            or not isinstance(execution["batch_size"], int)
            or execution["batch_size"] < 1
        ):
            raise IdentityTopologyError("fixed-panel topology execution differs")
    else:
        raise IdentityTopologyError("unsupported identity topology manifest schema")
    if set(manifest) != expected_fields:
        raise IdentityTopologyError("identity topology manifest fields differ")
    rows = manifest["records"]
    if not isinstance(rows, list) or not rows:
        raise IdentityTopologyError("identity topology records must be a non-empty list")

    parsed: list[_Record] = []
    keys: set[tuple[str, str]] = set()
    sample_metadata: dict[str, tuple[str, str]] = {}
    dimension: int | None = None
    available_count = 0
    for index, row in enumerate(rows):
        context = f"records[{index}]"
        if not isinstance(row, Mapping):
            raise IdentityTopologyError(f"{context} must be an object")
        fields = set(row)
        if fields not in (
            _REQUIRED_RECORD_FIELDS,
            _REQUIRED_RECORD_FIELDS | {"rank_label"},
        ):
            raise IdentityTopologyError(f"{context} fields differ")
        sample = _token(row["sample_token"], f"{context}.sample_token")
        identity = _token(row["identity_token"], f"{context}.identity_token")
        session = _token(row["session_token"], f"{context}.session_token")
        branch = _token(row["branch"], f"{context}.branch")
        key = (branch, sample)
        if key in keys:
            raise IdentityTopologyError(
                f"duplicate sample/branch binding: {sample!r}, {branch!r}"
            )
        keys.add(key)
        metadata = (identity, session)
        if sample in sample_metadata and sample_metadata[sample] != metadata:
            raise IdentityTopologyError(
                f"sample {sample!r} has incompatible identity/session bindings"
            )
        sample_metadata[sample] = metadata

        available = row["available"]
        if not isinstance(available, bool):
            raise IdentityTopologyError(f"{context}.available must be boolean")
        quality = _quality(row["quality"], f"{context}.quality")
        rank = _rank_label(row.get("rank_label"), f"{context}.rank_label")
        if available:
            if quality is None:
                raise IdentityTopologyError(
                    f"{context} available record is missing its quality binding"
                )
            if row["embedding"] is None:
                raise IdentityTopologyError(
                    f"{context} available record is missing its embedding binding"
                )
            vector = _embedding(
                row["embedding"], context=f"{context}.embedding", config=config
            )
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise IdentityTopologyError(
                    f"{context}.embedding has incompatible dimension {len(vector)}; "
                    f"expected {dimension}"
                )
            available_count += 1
        else:
            if row["embedding"] is not None:
                raise IdentityTopologyError(
                    f"{context} unavailable record must not bind an embedding"
                )
            if rank is not None:
                raise IdentityTopologyError(
                    f"{context} unavailable record must not bind a rank label"
                )
            vector = None
        parsed.append(
            _Record(sample, identity, session, branch, quality, available, vector, rank)
        )
    if available_count == 0:
        raise IdentityTopologyError("manifest contains no available embedding")
    return tuple(
        sorted(
            parsed,
            key=lambda record: (
                record.branch,
                record.identity_token,
                record.session_token,
                record.sample_token,
            ),
        )
    )


def validate_identity_topology_manifest(
    manifest: Mapping[str, Any],
    *,
    config: IdentityTopologyConfig | None = None,
) -> None:
    """Validate a generic or fixed-panel manifest and its vector bindings."""

    _parse_manifest(manifest, config or IdentityTopologyConfig())


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(left, right), -1.0, 1.0))


def _prototype(vectors: np.ndarray, config: IdentityTopologyConfig, context: str) -> tuple[np.ndarray, float]:
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= config.minimum_prototype_norm:
        raise IdentityTopologyError(f"{context} has a degenerate mean prototype")
    return mean / norm, float(np.clip(norm, 0.0, 1.0))


def _summary(values: list[float | int]) -> dict[str, int | float]:
    if not values:
        raise IdentityTopologyError("cannot summarize an empty metric")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise IdentityTopologyError("metric summary values must be finite")
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _unavailable(reason: str, **coverage: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **coverage}


def _components(node_count: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(node_count))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[right_root] = left_root
    return len({root(node) for node in range(node_count)})


def _cross_session(
    records: list[_Record],
    config: IdentityTopologyConfig,
    context: str,
) -> dict[str, Any]:
    by_session: dict[str, list[np.ndarray]] = {}
    for record in records:
        assert record.embedding is not None
        by_session.setdefault(record.session_token, []).append(record.embedding)
    sessions = sorted(by_session)
    if len(sessions) < 2:
        return _unavailable(
            "AT_LEAST_TWO_SESSIONS_REQUIRED",
            session_count=len(sessions),
            same_track_only=True,
            threshold=config.connectivity_cosine_distance_threshold,
        )
    prototypes = [
        _prototype(
            np.stack(by_session[session]),
            config,
            f"{context} session {session!r}",
        )[0]
        for session in sessions
    ]
    distances: list[float] = []
    edges: list[tuple[int, int]] = []
    for left in range(len(sessions)):
        for right in range(left + 1, len(sessions)):
            distance = _distance(prototypes[left], prototypes[right])
            distances.append(distance)
            if distance <= config.connectivity_cosine_distance_threshold:
                edges.append((left, right))
    component_count = _components(len(sessions), edges)
    possible_edges = len(distances)
    return {
        "available": True,
        "session_count": len(sessions),
        "same_track_only": False,
        "threshold": config.connectivity_cosine_distance_threshold,
        "edge_count": len(edges),
        "possible_edge_count": possible_edges,
        "connectivity_fraction": len(edges) / possible_edges,
        "component_count": component_count,
        "fragmented": component_count > 1,
        "maximum_cross_session_cosine_distance": max(distances),
    }


def _identity_geometry(
    records: list[_Record],
    config: IdentityTopologyConfig,
    context: str,
) -> tuple[dict[str, Any], np.ndarray]:
    vectors = np.stack([record.embedding for record in records])
    prototype, stability = _prototype(vectors, config, context)
    distances = [
        _distance(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    diameter = max(distances, default=0.0)
    sample_to_prototype = [_distance(vector, prototype) for vector in vectors]
    if len(vectors) < 2:
        leave_one_out = _unavailable("AT_LEAST_TWO_SAMPLES_REQUIRED", sample_count=1)
    else:
        drifts = []
        for index in range(len(vectors)):
            loo, _ = _prototype(
                np.delete(vectors, index, axis=0),
                config,
                f"{context} leave-one-out index {index}",
            )
            drifts.append(_distance(prototype, loo))
        leave_one_out = {
            "available": True,
            "sample_count": len(vectors),
            "prototype_cosine_distance": _summary(drifts),
        }
    return (
        {
            "available_sample_count": len(records),
            "session_count": len({record.session_token for record in records}),
            "quality": _summary([record.quality for record in records if record.quality is not None]),
            "normalized_prototype_stability": stability,
            "sample_to_normalized_prototype_cosine_distance": _summary(
                sample_to_prototype
            ),
            "intra_identity_cosine_diameter": diameter,
            "cross_session_connectivity": _cross_session(records, config, context),
            "leave_one_out_prototype_drift": leave_one_out,
        },
        prototype,
    )


def _branch_report(
    branch: str,
    records: list[_Record],
    config: IdentityTopologyConfig,
) -> dict[str, Any]:
    available = [record for record in records if record.available]
    unavailable_count = len(records) - len(available)
    records_by_identity: dict[str, list[_Record]] = {}
    for record in records:
        records_by_identity.setdefault(record.identity_token, []).append(record)
    identities: dict[str, dict[str, Any]] = {}
    prototypes: dict[str, np.ndarray] = {}
    for identity in sorted(records_by_identity):
        identity_records = records_by_identity[identity]
        identity_available = [record for record in identity_records if record.available]
        if not identity_available:
            unavailable = _unavailable("NO_AVAILABLE_EMBEDDINGS")
            qualities = [
                record.quality
                for record in identity_records
                if record.quality is not None
            ]
            identities[identity] = {
                "record_count": len(identity_records),
                "available_sample_count": 0,
                "unavailable_sample_count": len(identity_records),
                "session_count": 0,
                "quality": (
                    _summary(qualities)
                    if qualities
                    else _unavailable("NO_QUALITY_VALUES")
                ),
                "normalized_prototype_stability": unavailable,
                "sample_to_normalized_prototype_cosine_distance": unavailable,
                "intra_identity_cosine_diameter": unavailable,
                "cross_session_connectivity": _unavailable(
                    "NO_AVAILABLE_EMBEDDINGS",
                    session_count=0,
                    same_track_only=True,
                    threshold=config.connectivity_cosine_distance_threshold,
                ),
                "leave_one_out_prototype_drift": unavailable,
            }
            continue
        geometry, prototype = _identity_geometry(
            identity_available,
            config,
            f"branch {branch!r} identity {identity!r}",
        )
        geometry["record_count"] = len(identity_records)
        geometry["unavailable_sample_count"] = len(identity_records) - len(
            identity_available
        )
        identities[identity] = geometry
        prototypes[identity] = prototype

    identity_tokens = sorted(prototypes)
    if len(identity_tokens) < 2:
        for identity in identity_tokens:
            identities[identity]["nearest_impostor"] = _unavailable(
                "AT_LEAST_TWO_IDENTITIES_REQUIRED", identity_count=len(identity_tokens)
            )
            identities[identity]["hubness"] = _unavailable(
                "AT_LEAST_TWO_IDENTITIES_REQUIRED", identity_count=len(identity_tokens)
            )
        nearest_margins: list[float] = []
        occurrences: list[int] = []
        effective_k = 0
    else:
        matrix = np.stack([prototypes[identity] for identity in identity_tokens])
        distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
        np.fill_diagonal(distances, np.inf)
        order = np.argsort(distances, axis=1, kind="stable")
        effective_k = min(config.hubness_k, len(identity_tokens) - 1)
        occurrences_array = np.bincount(
            order[:, :effective_k].ravel(), minlength=len(identity_tokens)
        )
        nearest_margins = []
        occurrences = []
        for index, identity in enumerate(identity_tokens):
            nearest_index = int(order[index, 0])
            nearest_distance = float(distances[index, nearest_index])
            margin = nearest_distance - float(
                identities[identity]["intra_identity_cosine_diameter"]
            )
            nearest_margins.append(margin)
            occurrence = int(occurrences_array[index])
            occurrences.append(occurrence)
            identities[identity]["nearest_impostor"] = {
                "available": True,
                "identity_token": identity_tokens[nearest_index],
                "prototype_cosine_distance": nearest_distance,
                "diameter_adjusted_margin": margin,
            }
            identities[identity]["hubness"] = {
                "available": True,
                "effective_k": effective_k,
                "neighbor_occurrence_count": occurrence,
            }
    for identity in sorted(set(identities) - set(prototypes)):
        identities[identity]["nearest_impostor"] = _unavailable(
            "IDENTITY_HAS_NO_AVAILABLE_EMBEDDINGS"
        )
        identities[identity]["hubness"] = _unavailable(
            "IDENTITY_HAS_NO_AVAILABLE_EMBEDDINGS"
        )

    cross_session = [
        report["cross_session_connectivity"]
        for report in identities.values()
        if report["cross_session_connectivity"]["available"]
    ]
    loo = [
        report["leave_one_out_prototype_drift"][
            "prototype_cosine_distance"
        ]["mean"]
        for report in identities.values()
        if report["leave_one_out_prototype_drift"]["available"]
    ]
    aggregate: dict[str, Any] = {
        "record_count": len(records),
        "available_sample_count": len(available),
        "unavailable_sample_count": unavailable_count,
        "identity_count": len(identities),
        "topology_identity_count": len(prototypes),
        "session_count": len({record.session_token for record in available}),
        "same_track_only": not cross_session,
        "cross_session_identity_count": len(cross_session),
        "available_samples_per_identity": _summary(
            [report["available_sample_count"] for report in identities.values()]
        ),
        "sessions_per_identity": _summary(
            [report["session_count"] for report in identities.values()]
        ),
        "normalized_prototype_stability": (
            _summary(
                [
                    identities[identity]["normalized_prototype_stability"]
                    for identity in identity_tokens
                ]
            )
            if identity_tokens
            else _unavailable("NO_AVAILABLE_EMBEDDINGS")
        ),
        "intra_identity_cosine_diameter": (
            _summary(
                [
                    identities[identity]["intra_identity_cosine_diameter"]
                    for identity in identity_tokens
                ]
            )
            if identity_tokens
            else _unavailable("NO_AVAILABLE_EMBEDDINGS")
        ),
        "leave_one_out_prototype_drift_mean": (
            _summary(loo)
            if loo
            else _unavailable("NO_IDENTITY_HAS_TWO_AVAILABLE_SAMPLES")
        ),
        "cross_session_connectivity": (
            {
                "available": True,
                "eligible_identity_count": len(cross_session),
                "threshold": config.connectivity_cosine_distance_threshold,
                "fragmented_identity_count": sum(
                    int(item["fragmented"]) for item in cross_session
                ),
                "connectivity_fraction": _summary(
                    [item["connectivity_fraction"] for item in cross_session]
                ),
                "component_count": _summary(
                    [item["component_count"] for item in cross_session]
                ),
            }
            if cross_session
            else _unavailable(
                "NO_IDENTITY_HAS_TWO_SESSIONS",
                eligible_identity_count=0,
                threshold=config.connectivity_cosine_distance_threshold,
                same_track_only=True,
            )
        ),
        "nearest_impostor_margin": (
            _summary(nearest_margins)
            if nearest_margins
            else _unavailable("AT_LEAST_TWO_IDENTITIES_REQUIRED")
        ),
        "hubness": (
            {
                "available": True,
                "effective_k": effective_k,
                "neighbor_occurrence_count": _summary(occurrences),
                "zero_occurrence_fraction": float(
                    np.mean(np.asarray(occurrences) == 0)
                ),
            }
            if occurrences
            else _unavailable("AT_LEAST_TWO_IDENTITIES_REQUIRED")
        ),
    }
    return {"aggregate": aggregate, "identities": identities}


def _rank_outcomes(records: tuple[_Record, ...]) -> dict[str, Any]:
    ranked = [record for record in records if record.available and record.rank_label is not None]
    if not ranked:
        return _unavailable("RANK_LABELS_NOT_SUPPLIED")
    by_branch: dict[str, dict[str, _Record]] = {}
    for record in ranked:
        by_branch.setdefault(record.branch, {})[record.sample_token] = record
    comparisons: list[dict[str, Any]] = []
    branches = sorted(by_branch)
    for left_index, left in enumerate(branches):
        for right in branches[left_index + 1 :]:
            common = sorted(set(by_branch[left]) & set(by_branch[right]))
            if not common:
                continue
            left_ranks = np.asarray(
                [by_branch[left][sample].rank_label for sample in common], dtype=np.int64
            )
            right_ranks = np.asarray(
                [by_branch[right][sample].rank_label for sample in common], dtype=np.int64
            )

            def direction(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
                rescue = int(np.sum((source > 1) & (target == 1)))
                broken = int(np.sum((source == 1) & (target > 1)))
                return {
                    "rescue_count": rescue,
                    "break_count": broken,
                    "improved_rank_count": int(np.sum(target < source)),
                    "unchanged_rank_count": int(np.sum(target == source)),
                    "worsened_rank_count": int(np.sum(target > source)),
                    "rescue_fraction": rescue / len(common),
                    "break_fraction": broken / len(common),
                }

            comparisons.append(
                {
                    "left_branch": left,
                    "right_branch": right,
                    "paired_sample_count": len(common),
                    "left_to_right": direction(left_ranks, right_ranks),
                    "right_to_left": direction(right_ranks, left_ranks),
                }
            )
    if not comparisons:
        return _unavailable(
            "NO_CROSS_BRANCH_RANK_PAIRS",
            ranked_record_count=len(ranked),
            ranked_branch_count=len(branches),
        )
    return {
        "available": True,
        "rank_one_defines_success": True,
        "ranked_record_count": len(ranked),
        "comparisons": comparisons,
    }


def audit_identity_topology(
    manifest: Mapping[str, Any],
    *,
    config: IdentityTopologyConfig | None = None,
    code_sha256s: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit identity geometry without making physical or biometric claims."""

    if config is None:
        config = IdentityTopologyConfig()
    elif not isinstance(config, IdentityTopologyConfig):
        raise IdentityTopologyError("config must be an IdentityTopologyConfig")
    records = _parse_manifest(manifest, config)
    branches: dict[str, list[_Record]] = {}
    for record in records:
        branches.setdefault(record.branch, []).append(record)
    code_hashes = dict(sorted((code_sha256s or {}).items()))
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in code_hashes.items()
    ):
        raise IdentityTopologyError("code_sha256s must contain lowercase SHA-256 values")
    config_payload = config.to_dict()
    return {
        "schema_version": IDENTITY_TOPOLOGY_REPORT_SCHEMA_VERSION,
        "status": "PASS_EMBEDDING_SPACE_IDENTITY_TOPOLOGY_AUDIT",
        "interpretation": (
            "DESCRIPTIVE_NORMALIZED_EMBEDDING_SPACE_GEOMETRY_NOT_PHYSICAL_NOSE_"
            "TOPOLOGY_OR_BIOMETRIC_VALIDATION"
        ),
        "metric_definitions": {
            "distance": "ONE_MINUS_COSINE_ON_L2_NORMALIZED_VECTORS",
            "normalized_prototype_stability": (
                "L2_NORM_OF_ARITHMETIC_MEAN_OF_NORMALIZED_SAMPLE_VECTORS"
            ),
            "intra_identity_cosine_diameter": (
                "MAXIMUM_PAIRWISE_COSINE_DISTANCE_WITHIN_IDENTITY"
            ),
            "cross_session_connectivity": (
                "SESSION_PROTOTYPE_GRAPH_EDGE_WHEN_COSINE_DISTANCE_LE_THRESHOLD"
            ),
            "nearest_impostor_margin": (
                "NEAREST_OTHER_IDENTITY_PROTOTYPE_DISTANCE_MINUS_OWN_DIAMETER"
            ),
            "hubness": "IDENTITY_PROTOTYPE_K_NEAREST_NEIGHBOR_OCCURRENCE_COUNT",
            "leave_one_out_prototype_drift": (
                "COSINE_DISTANCE_BETWEEN_FULL_AND_ONE_SAMPLE_OMITTED_PROTOTYPES"
            ),
            "branch_rescue": "SOURCE_RANK_GT_1_AND_TARGET_RANK_EQ_1",
            "branch_break": "SOURCE_RANK_EQ_1_AND_TARGET_RANK_GT_1",
        },
        "provenance": {
            "input_sha256": content_sha256(manifest),
            "config_sha256": content_sha256(config_payload),
            "code_sha256s": code_hashes,
        },
        "config": config_payload,
        "population": {
            "record_count": len(records),
            "available_sample_branch_count": sum(record.available for record in records),
            "unavailable_sample_branch_count": sum(not record.available for record in records),
            "distinct_sample_count": len({record.sample_token for record in records}),
            "identity_count": len({record.identity_token for record in records}),
            "session_count": len({record.session_token for record in records}),
            "branch_count": len(branches),
            "embedding_dimension": len(
                next(record.embedding for record in records if record.embedding is not None)
            ),
        },
        "branches": {
            branch: _branch_report(branch, branches[branch], config)
            for branch in sorted(branches)
        },
        "branch_rank_outcomes": _rank_outcomes(records),
    }


__all__ = [
    "FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION",
    "IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION",
    "IDENTITY_TOPOLOGY_REPORT_SCHEMA_VERSION",
    "IdentityTopologyConfig",
    "IdentityTopologyError",
    "audit_identity_topology",
    "validate_identity_topology_manifest",
]
