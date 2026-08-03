"""Bounded N4 aggregation candidates for dispersed N3 frame embeddings.

This experiment operates only on normalized N3 embedding-space geometry. It
does not model or make claims about physical nose-ridge topology.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.retrieval import identity_clustered_bootstrap_ci
from experiments.fixed_multievidence import (
    FRAMES_PER_WINDOW,
    METHODS,
    file_sha256,
    read_fixed_panel,
    validate_fixed_topology_bindings,
    validate_panel_bundle,
)
from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.provenance import content_sha256

REPORT_SCHEMA_VERSION = "cvi.n4_robust_nose_evaluation.v1"
REPORT_BUNDLE_SCHEMA_VERSION = "cvi.n4_robust_nose_evaluation_bundle.v1"
N3_BRANCH = METHODS[2]
METRICS = ("Rank-1", "MRR", "Rank-5")
NORMALIZATION_TOLERANCE = 1e-6
MINIMUM_PROTOTYPE_NORM = 1e-12
LIMITATIONS = (
    "SAME_VIDEO_TRACK_GALLERY_AND_QUERY",
    "PUBLISHER_TEST_EXPOSED_DIAGNOSTIC",
    "WEAK_NOSE_ROI_INPUT",
    "CLOSED_SET_ONLY_NO_UNKNOWN_REJECTION",
    "NO_PHYSICAL_NOSE_TOPOLOGY_CLAIM",
    "NO_BIOMETRIC_VALIDATION_CLAIM",
)
_TOPOLOGY_FIELDS = {
    "sample_token",
    "identity_token",
    "session_token",
    "branch",
    "quality",
    "available",
    "embedding",
}
_PANEL_RECORD_FIELDS = {
    "sample_id",
    "instance_id",
    "registered_identity_id",
    "partition",
    "window_role",
    "publisher_frame_index",
    "split_role",
    "capture_group_id",
    "capture_group_kind",
    "source",
    "face",
    "weak_nose",
}
_CODE_PATHS = (
    "experiments/n4_robust_nose.py",
    "workflows/evaluate_n4_robust_nose.py",
)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One predeclared aggregation candidate and its deterministic tie cost."""

    name: str
    method: str
    parameter: int | None
    complexity: int
    prototype_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "parameter": self.parameter,
            "complexity": self.complexity,
            "prototype_count": self.prototype_count,
        }


CANDIDATES = (
    CandidateSpec("normalized_mean", "normalized_mean", None, 0, 1),
    CandidateSpec("medoid", "medoid", None, 1, 1),
    CandidateSpec("consensus_trimmed_mean_k2", "consensus_trimmed_mean", 2, 2, 1),
    CandidateSpec("consensus_trimmed_mean_k3", "consensus_trimmed_mean", 3, 2, 1),
    CandidateSpec("consensus_trimmed_mean_k4", "consensus_trimmed_mean", 4, 2, 1),
    CandidateSpec("quality_weighted_mean_p1", "quality_weighted_mean", 1, 2, 1),
    CandidateSpec("quality_weighted_mean_p2", "quality_weighted_mean", 2, 2, 1),
    CandidateSpec(
        "two_prototype_farthest_first",
        "two_prototype_farthest_first",
        None,
        3,
        2,
    ),
)
_CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}


@dataclass(frozen=True, slots=True)
class _WindowSample:
    sample_token: str
    quality: float
    embedding: np.ndarray


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _normalize(vector: object, context: str) -> np.ndarray:
    try:
        value = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a real numeric vector") from exc
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{context} must be a finite non-empty 1-d vector")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= MINIMUM_PROTOTYPE_NORM:
        raise ValueError(f"{context} has a degenerate norm")
    return value / norm


def _normalized_matrix(vectors: Sequence[object], context: str) -> np.ndarray:
    if not vectors:
        raise ValueError(f"{context} requires at least one vector")
    normalized = [
        _normalize(vector, f"{context}[{index}]")
        for index, vector in enumerate(vectors)
    ]
    dimension = len(normalized[0])
    if any(len(vector) != dimension for vector in normalized):
        raise ValueError(f"{context} vector dimensions differ")
    return np.stack(normalized)


def _sample_tokens(sample_tokens: Sequence[str], count: int) -> tuple[str, ...]:
    tokens = tuple(sample_tokens)
    if (
        len(tokens) != count
        or len(set(tokens)) != count
        or any(
            not isinstance(token, str) or not token or token.strip() != token
            for token in tokens
        )
    ):
        raise ValueError(
            "sample tokens must be aligned, unique, non-empty trimmed strings"
        )
    return tokens


def normalized_mean(vectors: Sequence[object]) -> np.ndarray:
    """Return the normalized arithmetic mean of normalized frame vectors."""

    matrix = _normalized_matrix(vectors, "normalized mean")
    return _normalize(np.mean(matrix, axis=0), "normalized mean prototype")


def _consensus_order(matrix: np.ndarray, tokens: tuple[str, ...]) -> list[int]:
    count = len(matrix)
    if count < 2:
        raise ValueError("consensus scoring requires at least two vectors")
    similarities = matrix @ matrix.T
    average = (similarities.sum(axis=1) - 1.0) / (count - 1)
    return sorted(
        range(count), key=lambda index: (-float(average[index]), tokens[index])
    )


def medoid(vectors: Sequence[object], sample_tokens: Sequence[str]) -> np.ndarray:
    """Return the highest-consensus input vector, breaking ties by sample token."""

    matrix = _normalized_matrix(vectors, "medoid")
    tokens = _sample_tokens(sample_tokens, len(matrix))
    return matrix[_consensus_order(matrix, tokens)[0]].copy()


def consensus_trimmed_mean(
    vectors: Sequence[object], sample_tokens: Sequence[str], retain: int
) -> np.ndarray:
    """Mean the ``retain`` vectors with highest average within-window cosine."""

    matrix = _normalized_matrix(vectors, "consensus-trimmed mean")
    tokens = _sample_tokens(sample_tokens, len(matrix))
    if (
        isinstance(retain, bool)
        or not isinstance(retain, int)
        or not 2 <= retain < len(matrix)
    ):
        raise ValueError("retain must be an integer in [2, vector_count)")
    selected = _consensus_order(matrix, tokens)[:retain]
    return _normalize(np.mean(matrix[selected], axis=0), "consensus-trimmed prototype")


def quality_weighted_mean(
    vectors: Sequence[object], qualities: Sequence[float], exponent: int
) -> np.ndarray:
    """Return a normalized mean using fixed nonnegative quality powers."""

    matrix = _normalized_matrix(vectors, "quality-weighted mean")
    quality = np.asarray(qualities, dtype=np.float64)
    if (
        quality.shape != (len(matrix),)
        or not np.isfinite(quality).all()
        or np.any((quality < 0.0) | (quality > 1.0))
    ):
        raise ValueError("qualities must be finite, aligned, and in [0, 1]")
    if exponent not in (1, 2) or isinstance(exponent, bool):
        raise ValueError("quality exponent must be exactly 1 or 2")
    weights = quality**exponent
    if float(weights.sum()) <= MINIMUM_PROTOTYPE_NORM:
        raise ValueError("quality-weighted mean has zero total bound quality")
    return _normalize(
        np.average(matrix, axis=0, weights=weights), "quality-weighted prototype"
    )


def _farthest_first(
    vectors: Sequence[object], sample_tokens: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = _normalized_matrix(vectors, "two-prototype farthest-first")
    tokens = _sample_tokens(sample_tokens, len(matrix))
    if len(matrix) < 2:
        raise ValueError("two-prototype farthest-first requires at least two vectors")
    first = min(range(len(tokens)), key=tokens.__getitem__)
    second = min(
        (index for index in range(len(tokens)) if index != first),
        key=lambda index: (float(matrix[first] @ matrix[index]), tokens[index]),
    )
    centers = (first, second)
    clusters: list[list[int]] = [[], []]
    for index, vector in enumerate(matrix):
        left = float(vector @ matrix[first])
        right = float(vector @ matrix[second])
        if left > right:
            cluster = 0
        elif right > left:
            cluster = 1
        else:
            cluster = 0 if tokens[first] < tokens[second] else 1
        clusters[cluster].append(index)
    if any(not cluster for cluster in clusters):
        raise ValueError("two-prototype farthest-first produced an empty cluster")
    prototypes = np.stack(
        [
            _normalize(
                np.mean(matrix[cluster], axis=0),
                f"farthest-first cluster {cluster_index}",
            )
            for cluster_index, cluster in enumerate(clusters)
        ]
    )
    details = {
        "seed_sample_tokens": [tokens[index] for index in centers],
        "clusters": [
            {
                "center_sample_token": tokens[center],
                "sample_tokens": sorted(tokens[index] for index in cluster),
            }
            for center, cluster in zip(centers, clusters, strict=True)
        ],
    }
    return prototypes, details


def two_prototype_farthest_first(
    vectors: Sequence[object], sample_tokens: Sequence[str]
) -> np.ndarray:
    """Return deterministic farthest-first cluster means as two prototypes."""

    return _farthest_first(vectors, sample_tokens)[0]


def _aggregate_candidate(
    samples: Sequence[_WindowSample], candidate: CandidateSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    vectors = [sample.embedding for sample in samples]
    tokens = [sample.sample_token for sample in samples]
    qualities = [sample.quality for sample in samples]
    details: dict[str, Any]
    if candidate.method == "normalized_mean":
        prototypes = normalized_mean(vectors)[None, :]
        details = {"retained_sample_tokens": sorted(tokens)}
    elif candidate.method == "medoid":
        matrix = _normalized_matrix(vectors, "medoid details")
        order = _consensus_order(matrix, _sample_tokens(tokens, len(matrix)))
        prototypes = matrix[order[:1]]
        details = {"retained_sample_tokens": [tokens[order[0]]]}
    elif candidate.method == "consensus_trimmed_mean":
        assert candidate.parameter is not None
        matrix = _normalized_matrix(vectors, "consensus details")
        order = _consensus_order(matrix, _sample_tokens(tokens, len(matrix)))
        retained = order[: candidate.parameter]
        prototypes = _normalize(
            np.mean(matrix[retained], axis=0), "consensus-trimmed prototype"
        )[None, :]
        details = {"retained_sample_tokens": [tokens[index] for index in retained]}
    elif candidate.method == "quality_weighted_mean":
        assert candidate.parameter is not None
        prototypes = quality_weighted_mean(vectors, qualities, candidate.parameter)[
            None, :
        ]
        details = {
            "quality_exponent": candidate.parameter,
            "retained_sample_tokens": sorted(tokens),
        }
    elif candidate.method == "two_prototype_farthest_first":
        prototypes, details = _farthest_first(vectors, tokens)
    else:
        raise AssertionError(f"unhandled candidate {candidate.name}")
    return prototypes, details


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic summary requires finite values")
    return {
        "count": len(array),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _window_diagnostic(
    samples: Sequence[_WindowSample], prototypes: np.ndarray, details: Mapping[str, Any]
) -> dict[str, Any]:
    matrix = np.stack([sample.embedding for sample in samples])
    mean_stability = float(np.linalg.norm(np.mean(matrix, axis=0)))
    pairwise = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
    nearest = np.min(np.clip(1.0 - matrix @ prototypes.T, 0.0, 2.0), axis=1)
    return {
        "input_normalized_mean_stability": mean_stability,
        "input_intra_window_cosine_diameter": float(np.max(pairwise)),
        "sample_to_nearest_candidate_prototype_cosine_distance": {
            "mean": float(np.mean(nearest)),
            "maximum": float(np.max(nearest)),
        },
        "prototype_count": len(prototypes),
        "prototype_pair_cosine_distance": (
            None
            if len(prototypes) == 1
            else float(np.clip(1.0 - prototypes[0] @ prototypes[1], 0.0, 2.0))
        ),
        "aggregation_details": dict(details),
    }


def _aggregate_diagnostics(
    per_identity: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    windows = [
        roles[role]
        for identity in sorted(per_identity)
        for roles in (per_identity[identity],)
        for role in ("gallery", "query")
    ]
    pair = [
        window["prototype_pair_cosine_distance"]
        for window in windows
        if window["prototype_pair_cosine_distance"] is not None
    ]
    return {
        "window_count": len(windows),
        "input_normalized_mean_stability": _summary(
            [window["input_normalized_mean_stability"] for window in windows]
        ),
        "input_intra_window_cosine_diameter": _summary(
            [window["input_intra_window_cosine_diameter"] for window in windows]
        ),
        "sample_to_nearest_candidate_prototype_cosine_distance_mean": _summary(
            [
                window["sample_to_nearest_candidate_prototype_cosine_distance"]["mean"]
                for window in windows
            ]
        ),
        "prototype_pair_cosine_distance": (
            _summary(pair)
            if pair
            else {"available": False, "reason": "CANDIDATE_USES_ONE_PROTOTYPE"}
        ),
    }


def _strict_unit_vector(value: object, context: str) -> np.ndarray:
    vector = _normalize(value, context)
    observed = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(observed))
    if abs(norm - 1.0) > NORMALIZATION_TOLERANCE:
        raise ValueError(f"{context} is not L2-normalized within tolerance")
    return vector


def _prepare_windows(
    panel_bundle: Mapping[str, Any], topology_manifest: Mapping[str, Any]
) -> tuple[dict[str, dict[str, dict[str, tuple[_WindowSample, ...]]]], dict[str, Any]]:
    panel = validate_panel_bundle(panel_bundle)
    validate_fixed_topology_bindings(panel_bundle, topology_manifest)
    topology_rows = topology_manifest.get("records")
    if not isinstance(topology_rows, list):
        raise TypeError("topology records differ")

    panel_by_sample: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(panel["records"]):
        if set(record) != _PANEL_RECORD_FIELDS:
            raise ValueError(
                f"panel record {index} fields differ from the fixed producer"
            )
        if set(record["source"]) != {"path", "sha256", "quality"}:
            raise ValueError("panel source binding fields differ")
        if set(record["face"]) != {"path", "sha256", "quality"}:
            raise ValueError("panel face binding fields differ")
        if set(record["weak_nose"]) != {
            "path",
            "sha256",
            "quality",
            "quality_semantics",
        }:
            raise ValueError("panel weak-nose binding fields differ")
        if (
            record["weak_nose"]["quality_semantics"]
            != "DOG_ROI_QUALITY_PROXY_NO_NOSE_SPECIFIC_SCORE"
        ):
            raise ValueError("panel weak-nose quality semantics differ")
        sample = record["sample_id"]
        panel_by_sample[sample] = record

    by_branch: dict[str, dict[str, Mapping[str, Any]]] = {
        method: {} for method in METHODS
    }
    for index, row in enumerate(topology_rows):
        if not isinstance(row, Mapping) or set(row) != _TOPOLOGY_FIELDS:
            raise ValueError(
                f"topology record {index} fields differ from the exact producer"
            )
        branch = row["branch"]
        if branch not in by_branch:
            raise ValueError("topology manifest contains a non-fixed-panel branch")
        sample = row["sample_token"]
        by_branch[branch][sample] = row
    expected_samples = set(panel_by_sample)
    if any(set(rows) != expected_samples for rows in by_branch.values()):
        raise ValueError("topology branch sample coverage differs from the fixed panel")

    for branch, rows in by_branch.items():
        for sample, topology in rows.items():
            record = panel_by_sample[sample]
            if (
                topology["identity_token"] != record["registered_identity_id"]
                or topology["session_token"] != record["capture_group_id"]
                or topology["available"] is not True
            ):
                raise ValueError(
                    "topology identity/session/availability binding differs from panel"
                )
            expected_quality = (
                record["face"]["quality"]["overall"]
                if branch == METHODS[1]
                else record["source"]["quality"]["overall"]
            )
            if topology["quality"] != expected_quality:
                raise ValueError("topology quality binding differs from panel")

    windows: dict[str, dict[str, dict[str, tuple[_WindowSample, ...]]]] = {
        "DEV": {},
        "EVAL": {},
    }
    n3_rows = by_branch[N3_BRANCH]
    n3_vectors: list[np.ndarray] = []
    for partition, identity_key in (
        ("DEV", "dev_identity_ids"),
        ("EVAL", "eval_identity_ids"),
    ):
        expected_identities = panel["population"][identity_key]
        for identity in expected_identities:
            identity_rows = [
                record
                for record in panel["records"]
                if record["registered_identity_id"] == identity
            ]
            if {record["partition"] for record in identity_rows} != {partition}:
                raise ValueError("panel identity partition binding differs")
            sessions = {record["capture_group_id"] for record in identity_rows}
            if len(sessions) != 1:
                raise ValueError("N4 is limited to one same-track session per identity")
            role_windows: dict[str, tuple[_WindowSample, ...]] = {}
            for role in ("gallery", "query"):
                rows = sorted(
                    (
                        record
                        for record in identity_rows
                        if record["window_role"] == role
                    ),
                    key=lambda record: (
                        record["publisher_frame_index"],
                        record["sample_id"],
                    ),
                )
                if len(rows) != FRAMES_PER_WINDOW:
                    raise ValueError(
                        "N4 requires exactly five gallery and five query samples"
                    )
                samples = []
                for record in rows:
                    topology = n3_rows[record["sample_id"]]
                    if record["weak_nose"]["quality"] != record["source"]["quality"]:
                        raise ValueError(
                            "weak-nose quality proxy differs from source quality binding"
                        )
                    vector = _strict_unit_vector(
                        topology["embedding"], f"N3 sample {record['sample_id']!r}"
                    )
                    n3_vectors.append(vector)
                    samples.append(
                        _WindowSample(
                            sample_token=record["sample_id"],
                            quality=float(topology["quality"]),
                            embedding=vector,
                        )
                    )
                role_windows[role] = tuple(samples)
            if {
                role_windows["gallery"][index].sample_token
                for index in range(FRAMES_PER_WINDOW)
            } & {
                role_windows["query"][index].sample_token
                for index in range(FRAMES_PER_WINDOW)
            }:
                raise ValueError("N4 gallery and query windows overlap")
            windows[partition][identity] = role_windows
        if list(windows[partition]) != expected_identities:
            raise ValueError("N4 identity list differs from the fixed panel")

    joined_hash = hashlib.sha256(
        np.ascontiguousarray(np.stack(n3_vectors), dtype="<f8").tobytes()
    ).hexdigest()
    bindings = {
        "panel_sha256": panel_bundle["panel_sha256"],
        "topology_manifest_sha256": content_sha256(topology_manifest),
        "dev_identity_ids_sha256": panel["population"]["dev_identity_ids_sha256"],
        "eval_identity_ids_sha256": panel["population"]["eval_identity_ids_sha256"],
        "n3_joined_embedding_float64_sha256": joined_hash,
        "n3_joined_embedding_hash_order": "DEV_THEN_EVAL_IDENTITY_LIST;GALLERY_THEN_QUERY;FRAME_INDEX_THEN_SAMPLE_TOKEN",
    }
    return windows, bindings


def rank_score_rows(
    scores: np.ndarray, identities: Sequence[str]
) -> list[dict[str, Any]]:
    """Rank a square identity score matrix with lexical gallery tie-breaking."""

    identity_list = list(identities)
    matrix = np.asarray(scores, dtype=np.float64)
    if identity_list != sorted(set(identity_list)):
        raise ValueError("identities must be sorted and unique")
    if (
        matrix.shape != (len(identity_list), len(identity_list))
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("score matrix must be finite and square over identities")
    rows = []
    for query_index, identity in enumerate(identity_list):
        order = sorted(
            range(len(identity_list)),
            key=lambda gallery_index: (
                -float(matrix[query_index, gallery_index]),
                identity_list[gallery_index],
            ),
        )
        rank = order.index(query_index) + 1
        rows.append(
            {
                "registered_identity_id": identity,
                "rank": rank,
                "Rank-1": float(rank == 1),
                "MRR": 1.0 / rank,
                "Rank-5": float(rank <= 5),
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("retrieval metrics require identity rows")
    return {
        "query_identity_count": len(rows),
        "gallery_identity_count": len(rows),
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
    }


def _score_candidate(
    partition_windows: Mapping[str, Mapping[str, Sequence[_WindowSample]]],
    identities: Sequence[str],
    candidate: CandidateSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    identity_list = list(identities)
    if identity_list != sorted(set(identity_list)) or set(partition_windows) != set(
        identity_list
    ):
        raise ValueError("candidate scoring identities differ from partition windows")
    prototypes: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for identity in identity_list:
        prototypes[identity] = {}
        diagnostics[identity] = {}
        for role in ("gallery", "query"):
            samples = partition_windows[identity][role]
            if len(samples) != FRAMES_PER_WINDOW:
                raise ValueError("candidate scoring requires exact K5 windows")
            values, details = _aggregate_candidate(samples, candidate)
            prototypes[identity][role] = values
            diagnostics[identity][role] = _window_diagnostic(samples, values, details)
    scores = np.empty((len(identity_list), len(identity_list)), dtype=np.float64)
    for query_index, query_identity in enumerate(identity_list):
        query = prototypes[query_identity]["query"]
        for gallery_index, gallery_identity in enumerate(identity_list):
            gallery = prototypes[gallery_identity]["gallery"]
            scores[query_index, gallery_index] = float(np.max(query @ gallery.T))
    return (
        rank_score_rows(scores, identity_list),
        _aggregate_diagnostics(diagnostics),
        diagnostics,
    )


def _validate_outcomes(outcomes: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    if set(outcomes) != set(_CANDIDATE_BY_NAME):
        raise ValueError("DEV outcomes must contain every predeclared candidate")
    expected_identities: list[str] | None = None
    for candidate in CANDIDATES:
        rows = list(outcomes[candidate.name])
        identities = [row.get("registered_identity_id") for row in rows]
        if identities != sorted(set(identities)) or len(rows) < 2:
            raise ValueError("DEV candidate rows must contain sorted unique identities")
        if expected_identities is None:
            expected_identities = identities
        elif identities != expected_identities:
            raise ValueError("DEV candidate rows are not exactly paired")
        for row in rows:
            rank = row.get("rank")
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 1 <= rank <= len(rows)
            ):
                raise ValueError("DEV candidate rank differs")
            expected = {
                "Rank-1": float(rank == 1),
                "MRR": 1.0 / rank,
                "Rank-5": float(rank <= 5),
            }
            if any(row.get(metric) != value for metric, value in expected.items()):
                raise ValueError("DEV candidate row metrics differ from rank")


def select_dev_candidate(
    outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Select exactly one predeclared candidate using DEV outcomes only."""

    _validate_outcomes(outcomes)
    metrics = {
        candidate.name: _metrics(outcomes[candidate.name]) for candidate in CANDIDATES
    }
    selected = min(
        CANDIDATES,
        key=lambda candidate: (
            -metrics[candidate.name]["Rank-1"],
            -metrics[candidate.name]["MRR"],
            -metrics[candidate.name]["Rank-5"],
            candidate.complexity,
            candidate.prototype_count,
            candidate.name,
        ),
    )
    return {
        "labels_used": "DEV_ONLY",
        "evaluation_labels_used_for_selection": False,
        "objective_lexicographic": ["Rank-1", "MRR", "Rank-5"],
        "tie_break": [
            "LOWER_COMPLEXITY",
            "FEWER_PROTOTYPES",
            "LEXICAL_CANDIDATE_NAME",
        ],
        "selected_candidate": selected.name,
        "selected_candidate_declaration": selected.to_dict(),
        "selected_dev_metrics": metrics[selected.name],
    }


def _bootstrap(
    outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: str,
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    intervals: dict[str, Any] = {}
    ordered = ["normalized_mean"] + (
        [] if selected == "normalized_mean" else [selected]
    )
    for candidate_index, candidate in enumerate(ordered):
        bootstrap_rows = [
            {
                "bootstrap_cluster_id": row["registered_identity_id"],
                **{metric: row[metric] for metric in METRICS},
            }
            for row in outcomes[candidate]
        ]
        intervals[candidate] = {
            metric: identity_clustered_bootstrap_ci(
                bootstrap_rows,
                metric=metric,
                resamples=resamples,
                seed=seed + candidate_index * len(METRICS) + metric_index,
                confidence_level=confidence_level,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    baseline = outcomes["normalized_mean"]
    candidate_rows = outcomes[selected]
    delta_rows = [
        {
            "bootstrap_cluster_id": left["registered_identity_id"],
            **{metric: right[metric] - left[metric] for metric in METRICS},
        }
        for left, right in zip(baseline, candidate_rows, strict=True)
    ]
    paired = {
        metric: identity_clustered_bootstrap_ci(
            delta_rows,
            metric=metric,
            resamples=resamples,
            seed=seed + 10_000 + metric_index,
            confidence_level=confidence_level,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    return intervals, paired


def _rescue_break(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if [row["registered_identity_id"] for row in baseline] != [
        row["registered_identity_id"] for row in candidate
    ]:
        raise ValueError("rescue/break rows are not paired")
    count = len(baseline)
    rescue = sum(
        left["rank"] > 1 and right["rank"] == 1
        for left, right in zip(baseline, candidate, strict=True)
    )
    broken = sum(
        left["rank"] == 1 and right["rank"] > 1
        for left, right in zip(baseline, candidate, strict=True)
    )
    return {
        "paired_identity_count": count,
        "rescue_count": rescue,
        "break_count": broken,
        "rescue_fraction": rescue / count,
        "break_fraction": broken / count,
    }


def build_n4_report(
    panel_bundle: Mapping[str, Any],
    topology_manifest: Mapping[str, Any],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
    input_file_bindings: Mapping[str, Any] | None = None,
    code_sha256s: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the deterministic DEV-selected and once-applied EVAL N4 report."""

    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not 0.0 < float(bootstrap_confidence_level) < 1.0:
        raise ValueError("bootstrap_confidence_level must be in (0, 1)")
    hashes = dict(sorted((code_sha256s or {}).items()))
    if any(
        _require_sha256(value, f"code hash for {path}") != value
        for path, value in hashes.items()
    ):
        raise ValueError("code SHA-256 binding differs")

    windows, semantic_bindings = _prepare_windows(panel_bundle, topology_manifest)
    panel = panel_bundle["panel"]
    dev_ids = panel["population"]["dev_identity_ids"]
    eval_ids = panel["population"]["eval_identity_ids"]
    dev_outcomes: dict[str, list[dict[str, Any]]] = {}
    dev_diagnostics: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        rows, aggregate, _ = _score_candidate(windows["DEV"], dev_ids, candidate)
        dev_outcomes[candidate.name] = rows
        dev_diagnostics[candidate.name] = aggregate
    selection = select_dev_candidate(dev_outcomes)
    selected = selection["selected_candidate"]

    eval_names = ["normalized_mean"] + (
        [] if selected == "normalized_mean" else [selected]
    )
    eval_outcomes: dict[str, list[dict[str, Any]]] = {}
    eval_diagnostics: dict[str, Any] = {}
    for name in eval_names:
        rows, aggregate, per_identity = _score_candidate(
            windows["EVAL"], eval_ids, _CANDIDATE_BY_NAME[name]
        )
        eval_outcomes[name] = rows
        eval_diagnostics[name] = {"aggregate": aggregate, "per_identity": per_identity}
    intervals, paired_intervals = _bootstrap(
        eval_outcomes,
        selected,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=float(bootstrap_confidence_level),
    )
    baseline_rows = eval_outcomes["normalized_mean"]
    selected_rows = eval_outcomes[selected]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS_BOUNDED_N4_EMBEDDING_SPACE_DIAGNOSTIC",
        "interpretation": (
            "DEV_SELECTED_N3_EMBEDDING_SPACE_AGGREGATION_ON_EXPOSED_SAME_TRACK_"
            "PUBLISHER_TEST_NOT_PHYSICAL_NOSE_TOPOLOGY_OR_BIOMETRIC_VALIDATION"
        ),
        "candidate_declarations": [candidate.to_dict() for candidate in CANDIDATES],
        "candidate_declarations_sha256": content_sha256(
            {"candidates": [candidate.to_dict() for candidate in CANDIDATES]}
        ),
        "protocol": {
            "input_branch": N3_BRANCH,
            "join_key": "sample_token_TO_panel.sample_id",
            "gallery_query_windows": "EXACT_FIXED_PANEL_K5",
            "retrieval": "CLOSED_SET_MAX_COSINE_ACROSS_CANDIDATE_PROTOTYPES",
            "ranking_tie_break": "LEXICAL_GALLERY_IDENTITY",
            "candidate_selection": "DEV_ONLY_LEXICOGRAPHIC_RANK1_MRR_RANK5",
            "evaluation_application": "ONE_FROZEN_CANDIDATE_APPLIED_ONCE_TO_IDENTITY_DISJOINT_EVAL",
            "embedding_space_topology_only": True,
            "physical_nose_topology_claim": False,
            "learned_network_added": False,
            "same_track_only": True,
            "publisher_test_exposed": True,
            "weak_nose_roi": True,
            "closed_set": True,
            "open_set": False,
            "bootstrap": {
                "cluster_unit": "registered_identity_id",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "confidence_level": float(bootstrap_confidence_level),
            },
            "limitations": list(LIMITATIONS),
        },
        "metric_definitions": {
            "input_normalized_mean_stability": "L2_NORM_OF_MEAN_OF_FIVE_NORMALIZED_FRAME_VECTORS",
            "input_intra_window_cosine_diameter": "MAXIMUM_PAIRWISE_ONE_MINUS_COSINE_WITHIN_K5_WINDOW",
            "sample_to_nearest_candidate_prototype_cosine_distance": "MINIMUM_ONE_MINUS_COSINE_TO_CANDIDATE_WINDOW_PROTOTYPES",
            "two_prototype_retrieval": "MAXIMUM_COSINE_OVER_QUERY_AND_GALLERY_PROTOTYPE_PAIRS",
        },
        "population": panel["population"],
        "input_bindings": {
            **semantic_bindings,
            "external_files": dict(input_file_bindings or {}),
        },
        "code_sha256s": hashes,
        "development": {
            "all_candidate_metrics": {
                candidate.name: _metrics(dev_outcomes[candidate.name])
                for candidate in CANDIDATES
            },
            "all_candidate_prototype_diagnostics": dev_diagnostics,
            "selection": selection,
        },
        "evaluation": {
            "selected_candidate": selected,
            "selected_vs_normalized_mean_metrics": {
                "normalized_mean": _metrics(baseline_rows),
                "selected_candidate": {
                    "candidate": selected,
                    **_metrics(selected_rows),
                },
            },
            "identity_bootstrap_cis": intervals,
            "paired_selected_minus_mean_identity_bootstrap_cis": paired_intervals,
            "selected_vs_mean_rescue_break": _rescue_break(
                baseline_rows, selected_rows
            ),
            "per_identity_ranks": [
                {
                    "registered_identity_id": identity,
                    "normalized_mean_rank": baseline_rows[index]["rank"],
                    "selected_candidate": selected,
                    "selected_candidate_rank": selected_rows[index]["rank"],
                }
                for index, identity in enumerate(eval_ids)
            ],
            "prototype_stability_and_dispersion_diagnostics": eval_diagnostics,
        },
    }
    return json.loads(json.dumps(report, allow_nan=False))


def validate_report_bundle(bundle: object) -> dict[str, Any]:
    """Validate the content binding and critical N4 report declarations."""

    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "report_sha256",
        "report",
    }:
        raise ValueError("N4 report bundle fields differ")
    if bundle["schema_version"] != REPORT_BUNDLE_SCHEMA_VERSION:
        raise ValueError("N4 report bundle schema differs")
    _require_sha256(bundle["report_sha256"], "report_sha256")
    report = bundle["report"]
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
    ):
        raise ValueError("N4 report schema differs")
    if content_sha256(report) != bundle["report_sha256"]:
        raise ValueError("N4 report digest differs")
    if report.get("candidate_declarations") != [
        candidate.to_dict() for candidate in CANDIDATES
    ]:
        raise ValueError("N4 candidate declarations differ")
    if report.get("protocol", {}).get("physical_nose_topology_claim") is not False:
        raise ValueError("N4 physical-topology limitation differs")
    return report


def evaluate_n4_robust_nose(
    *,
    panel_path: Path,
    panel_sha256: str,
    topology_manifest_path: Path,
    topology_sha256: str,
    n3_runtime_manifest_sha256: str,
    n3_onnx_sha256: str,
    output_path: Path,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Read externally pinned inputs and publish one no-overwrite N4 report."""

    repository = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    raw_output = Path(os.path.abspath(os.fspath(output_path)))
    output = raw_output.parent.resolve(strict=True) / raw_output.name
    if output.is_relative_to(repository):
        raise ValueError("N4 report must be written outside the Git repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite N4 report: {output}")

    panel_pin = _require_sha256(panel_sha256, "fixed panel canonical SHA-256")
    topology_pin = _require_sha256(topology_sha256, "topology canonical SHA-256")
    panel_document, _ = read_fixed_panel(panel_path, panel_pin)
    topology_document = read_strict_json_document(
        topology_manifest_path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if topology_document.canonical_payload_sha256 != topology_pin:
        raise ValueError("topology content SHA-256 differs from the external pin")
    runtime_pin = _require_sha256(
        n3_runtime_manifest_sha256, "N3 runtime manifest canonical SHA-256"
    )
    onnx_pin = _require_sha256(n3_onnx_sha256, "N3 ONNX SHA-256")
    validate_fixed_topology_bindings(
        panel_document.payload,
        topology_document.payload,
        n3_runtime_manifest_content_sha256=runtime_pin,
        n3_onnx_sha256=onnx_pin,
    )
    file_bindings = {
        "fixed_panel_bundle": {
            "path": os.fspath(panel_path),
            "raw_sha256": panel_document.raw_sha256,
            "content_sha256": panel_document.canonical_payload_sha256,
            "byte_size": panel_document.byte_size,
            "external_pin": panel_pin,
        },
        "topology_manifest": {
            "path": os.fspath(topology_manifest_path),
            "raw_sha256": topology_document.raw_sha256,
            "content_sha256": topology_document.canonical_payload_sha256,
            "byte_size": topology_document.byte_size,
            "external_pin": topology_pin,
        },
        "n3_runtime_manifest_content_sha256": runtime_pin,
        "n3_onnx_sha256": onnx_pin,
    }
    from artifact_contracts.source_provenance import build_source_provenance

    source_provenance = build_source_provenance(
        repository / relative for relative in _CODE_PATHS
    )
    code_hashes = {
        row["relative_path"]: row["content_sha256"]
        for row in source_provenance["code_source_files"]
    }
    report = build_n4_report(
        panel_document.payload,
        topology_document.payload,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence_level=bootstrap_confidence_level,
        input_file_bindings=file_bindings,
        code_sha256s=code_hashes,
    )
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA_VERSION,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    validate_report_bundle(bundle)
    write_private_json_bundle(((output, bundle),))
    return bundle


__all__ = [
    "CANDIDATES",
    "LIMITATIONS",
    "N3_BRANCH",
    "REPORT_BUNDLE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "build_n4_report",
    "consensus_trimmed_mean",
    "evaluate_n4_robust_nose",
    "medoid",
    "normalized_mean",
    "quality_weighted_mean",
    "rank_score_rows",
    "select_dev_candidate",
    "two_prototype_farthest_first",
    "validate_report_bundle",
]
