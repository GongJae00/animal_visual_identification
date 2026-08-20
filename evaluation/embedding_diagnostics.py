"""Aggregate, deterministic diagnostics for embedding matrices."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np


EMBEDDING_DIAGNOSTICS_SCHEMA_VERSION = "evaluation.embedding_diagnostics.v1"
_DOMAIN_CONFOUNDING_WARNING = (
    "Descriptive domain centroid shift is confounded by identity and session "
    "composition; it is not evidence of causal bias."
)


class EmbeddingDiagnosticsError(ValueError):
    """Raised when diagnostics inputs or configuration violate the contract."""


@dataclass(frozen=True, slots=True)
class EmbeddingDiagnosticsConfig:
    """Resource bounds and numerical policy for embedding diagnostics."""

    minimum_row_norm: float = 1e-12
    spectrum_max_samples: int = 2_048
    pairwise_max_samples: int = 1_024
    hubness_max_samples: int = 1_024
    hubness_k: int = 10
    numerical_rank_relative_tolerance: float = 1e-12
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_row_norm, bool)
            or not isinstance(self.minimum_row_norm, (int, float))
            or not np.isfinite(self.minimum_row_norm)
            or self.minimum_row_norm <= 0.0
        ):
            raise EmbeddingDiagnosticsError(
                "minimum_row_norm must be a finite positive number"
            )
        for name in (
            "spectrum_max_samples",
            "pairwise_max_samples",
            "hubness_max_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise EmbeddingDiagnosticsError(f"{name} must be an integer >= 2")
        if (
            isinstance(self.hubness_k, bool)
            or not isinstance(self.hubness_k, int)
            or self.hubness_k < 1
        ):
            raise EmbeddingDiagnosticsError("hubness_k must be a positive integer")
        if (
            isinstance(self.numerical_rank_relative_tolerance, bool)
            or not isinstance(
                self.numerical_rank_relative_tolerance, (int, float)
            )
            or not np.isfinite(self.numerical_rank_relative_tolerance)
            or not 0.0 < self.numerical_rank_relative_tolerance < 1.0
        ):
            raise EmbeddingDiagnosticsError(
                "numerical_rank_relative_tolerance must be finite and in (0, 1)"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EmbeddingDiagnosticsError("seed must be a non-negative integer")

    def to_dict(self) -> dict[str, int | float]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _unavailable(reason: str, **coverage: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **coverage}


def _as_embeddings(embeddings: Any) -> np.ndarray:
    try:
        matrix = np.asarray(embeddings)
    except (TypeError, ValueError) as exc:
        raise EmbeddingDiagnosticsError(
            "embeddings must be a rectangular 2-d numeric matrix"
        ) from exc
    if matrix.ndim != 2:
        raise EmbeddingDiagnosticsError(
            f"embeddings must be 2-d, got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise EmbeddingDiagnosticsError("embeddings must not have an empty axis")
    if not (
        np.issubdtype(matrix.dtype, np.integer)
        or np.issubdtype(matrix.dtype, np.floating)
    ):
        raise EmbeddingDiagnosticsError("embeddings must contain real numeric values")
    matrix = matrix.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise EmbeddingDiagnosticsError("embeddings contain non-finite values")
    return matrix


def _row_norms(matrix: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(matrix), axis=1)
    scaled = np.divide(
        matrix,
        scale[:, None],
        out=np.zeros_like(matrix),
        where=scale[:, None] != 0.0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        norms = scale * np.sqrt(np.sum(scaled * scaled, axis=1))
    if not np.isfinite(norms).all():
        raise EmbeddingDiagnosticsError("embedding row norms overflow float64")
    return norms


def _as_ids(values: Any, name: str, expected_length: int) -> tuple[Hashable, ...]:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise EmbeddingDiagnosticsError(f"{name} must be a 1-d ID array") from exc
    if array.ndim != 1:
        raise EmbeddingDiagnosticsError(f"{name} must be 1-d, got shape {array.shape}")
    if len(array) != expected_length:
        raise EmbeddingDiagnosticsError(
            f"{name} length {len(array)} != embedding count {expected_length}"
        )

    result: list[Hashable] = []
    for index, value in enumerate(array.tolist()):
        if value is None or isinstance(value, bool):
            raise EmbeddingDiagnosticsError(
                f"{name}[{index}] must be a non-null scalar ID"
            )
        if isinstance(value, (str, bytes)) and not value.strip():
            raise EmbeddingDiagnosticsError(f"{name}[{index}] must not be empty")
        if isinstance(value, complex):
            raise EmbeddingDiagnosticsError(f"{name}[{index}] must not be complex")
        if isinstance(value, float) and not np.isfinite(value):
            raise EmbeddingDiagnosticsError(f"{name}[{index}] must be finite")
        if not isinstance(value, Hashable):
            raise EmbeddingDiagnosticsError(f"{name}[{index}] must be hashable")
        result.append(value)
    return tuple(result)


def _as_quality(values: Any, expected_length: int) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise EmbeddingDiagnosticsError(
            "quality_scores must be a 1-d numeric array"
        ) from exc
    if array.ndim != 1:
        raise EmbeddingDiagnosticsError(
            f"quality_scores must be 1-d, got shape {array.shape}"
        )
    if len(array) != expected_length:
        raise EmbeddingDiagnosticsError(
            f"quality_scores length {len(array)} != embedding count {expected_length}"
        )
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise EmbeddingDiagnosticsError("quality_scores must contain real numbers")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise EmbeddingDiagnosticsError("quality_scores contain non-finite values")
    return result


def _finite_summary(values: np.ndarray) -> dict[str, int | float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise EmbeddingDiagnosticsError("cannot summarize empty or non-finite values")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        mean = 0.0
        standard_deviation = 0.0
    else:
        scaled = values / scale
        mean = float(scale * np.mean(scaled))
        standard_deviation = float(scale * np.std(scaled))
    quantiles = np.quantile(values, (0.05, 0.5, 0.95))
    result: dict[str, int | float] = {
        "count": len(values),
        "minimum": float(np.min(values)),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "maximum": float(np.max(values)),
        "mean": mean,
        "standard_deviation": standard_deviation,
    }
    if not all(
        isinstance(value, int) or np.isfinite(value) for value in result.values()
    ):
        raise EmbeddingDiagnosticsError("summary statistics overflow float64")
    return result


def _sample_indices(count: int, cap: int, seed: int) -> np.ndarray:
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    selected = np.random.default_rng(seed).choice(count, size=cap, replace=False)
    return np.sort(selected)


def _groups(ids: tuple[Hashable, ...]) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = {}
    for index, value in enumerate(ids):
        groups.setdefault(value, []).append(index)
    return groups


def _pairwise_cosine_values(
    directions: np.ndarray,
    *,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, int, bool]:
    selected = _sample_indices(len(directions), cap, seed)
    sampled = directions[selected]
    if len(sampled) < 2:
        return np.empty(0, dtype=np.float64), len(sampled), len(selected) < len(directions)
    similarities = sampled @ sampled.T
    upper = np.triu_indices(len(sampled), k=1)
    values = np.clip(similarities[upper], -1.0, 1.0)
    return values, len(sampled), len(selected) < len(directions)


def _pairwise_report(
    directions: np.ndarray,
    *,
    cap: int,
    seed: int,
) -> dict[str, Any]:
    values, sample_count, sampled = _pairwise_cosine_values(
        directions, cap=cap, seed=seed
    )
    coverage = {
        "source_count": len(directions),
        "sample_count": sample_count,
        "sampled": sampled,
    }
    if len(values) == 0:
        return _unavailable("at least two directions are required", **coverage)
    return {
        "available": True,
        **coverage,
        "pair_count": len(values),
        "summary": _finite_summary(values),
    }


def _centroid_directions(
    normalized: np.ndarray,
    grouped_indices: list[list[int]],
    minimum_norm: float,
) -> tuple[np.ndarray, int]:
    centroids: list[np.ndarray] = []
    degenerate = 0
    for indices in grouped_indices:
        centroid = np.mean(normalized[indices], axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= minimum_norm:
            degenerate += 1
            continue
        centroids.append(centroid / norm)
    if not centroids:
        return np.empty((0, normalized.shape[1]), dtype=np.float64), degenerate
    return np.asarray(centroids, dtype=np.float64), degenerate


def _covariance_spectrum(
    matrix: np.ndarray,
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    selected = _sample_indices(len(matrix), config.spectrum_max_samples, config.seed)
    sampled = matrix[selected]
    coverage = {
        "source_sample_count": len(matrix),
        "sample_count": len(sampled),
        "sampled": len(sampled) < len(matrix),
    }
    if len(sampled) < 2:
        return _unavailable("at least two samples are required", **coverage)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = sampled - np.mean(sampled, axis=0)
    if not np.isfinite(centered).all():
        raise EmbeddingDiagnosticsError("centering embeddings overflowed float64")
    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise EmbeddingDiagnosticsError(
            "centered covariance spectrum did not converge"
        ) from exc
    with np.errstate(over="ignore", invalid="ignore"):
        computed = singular_values * singular_values / (len(sampled) - 1)
    if not np.isfinite(computed).all():
        raise EmbeddingDiagnosticsError("covariance eigenvalues overflowed float64")
    eigenvalues = np.zeros(matrix.shape[1], dtype=np.float64)
    eigenvalues[: len(computed)] = computed
    spectrum_summary = _finite_summary(eigenvalues)
    eigenvalue_scale = float(eigenvalues[0])
    if eigenvalue_scale <= 0.0:
        return _unavailable(
            "centered embeddings have zero total variance",
            **coverage,
            eigenvalue_count=len(eigenvalues),
            eigenvalue_summary=spectrum_summary,
            numerical_rank=0,
        )
    total = float(eigenvalue_scale * np.sum(eigenvalues / eigenvalue_scale))
    if not np.isfinite(total):
        raise EmbeddingDiagnosticsError("total covariance variance overflowed float64")

    probabilities = eigenvalues[eigenvalues > 0.0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    threshold = float(
        eigenvalues[0] * config.numerical_rank_relative_tolerance
    )
    return {
        "available": True,
        **coverage,
        "eigenvalue_count": len(eigenvalues),
        "eigenvalue_summary": spectrum_summary,
        "total_variance": total,
        "effective_rank_entropy": float(np.exp(entropy)),
        "participation_ratio": float(1.0 / np.sum(probabilities * probabilities)),
        "numerical_rank": int(np.sum(eigenvalues > threshold)),
        "numerical_rank_relative_tolerance": (
            config.numerical_rank_relative_tolerance
        ),
        "top_eigenvalue_fraction": float(eigenvalues[0] / total),
    }


def _directional_geometry(
    normalized: np.ndarray,
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    centroid_norm = float(
        np.clip(np.linalg.norm(np.mean(normalized, axis=0)), 0.0, 1.0)
    )
    return {
        "available": True,
        "centroid_norm": centroid_norm,
        "off_diagonal_cosine": _pairwise_report(
            normalized,
            cap=config.pairwise_max_samples,
            seed=config.seed,
        ),
    }


def _identity_diagnostics(
    normalized: np.ndarray,
    identity_ids: tuple[Hashable, ...] | None,
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    if identity_ids is None:
        return _unavailable("identity_ids were not provided")
    groups = _groups(identity_ids)
    repeated = [indices for indices in groups.values() if len(indices) >= 2]
    coverage = {
        "identity_count": len(groups),
        "repeated_identity_count": len(repeated),
        "covered_sample_count": sum(map(len, repeated)),
        "coverage_fraction": sum(map(len, repeated)) / len(normalized),
    }
    if not repeated:
        return _unavailable("no identity has multiple samples", **coverage)

    dispersions = np.asarray(
        [
            1.0
            - float(
                np.clip(
                    np.linalg.norm(np.mean(normalized[indices], axis=0)),
                    0.0,
                    1.0,
                )
            )
            for indices in repeated
        ],
        dtype=np.float64,
    )
    centroids, degenerate = _centroid_directions(
        normalized, list(groups.values()), config.minimum_row_norm
    )
    between = _pairwise_report(
        centroids,
        cap=config.pairwise_max_samples,
        seed=config.seed,
    )
    between["eligible_identity_centroid_count"] = len(centroids)
    between["degenerate_identity_centroid_count"] = degenerate
    return {
        "available": True,
        **coverage,
        "within_identity_directional_dispersion": _finite_summary(dispersions),
        "between_identity_centroid_cosine": between,
    }


def _session_diagnostics(
    normalized: np.ndarray,
    identity_ids: tuple[Hashable, ...] | None,
    session_ids: tuple[Hashable, ...] | None,
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    if identity_ids is None or session_ids is None:
        missing = []
        if identity_ids is None:
            missing.append("identity_ids")
        if session_ids is None:
            missing.append("session_ids")
        return _unavailable(f"{', '.join(missing)} were not provided")

    identity_groups = _groups(identity_ids)
    spanning: list[tuple[list[int], dict[Hashable, list[int]]]] = []
    for indices in identity_groups.values():
        sessions: dict[Hashable, list[int]] = {}
        for index in indices:
            sessions.setdefault(session_ids[index], []).append(index)
        if len(sessions) >= 2:
            spanning.append((indices, sessions))
    coverage = {
        "identity_count": len(identity_groups),
        "cross_session_identity_count": len(spanning),
        "covered_sample_count": sum(len(indices) for indices, _ in spanning),
        "coverage_fraction": (
            sum(len(indices) for indices, _ in spanning) / len(normalized)
        ),
    }
    if not spanning:
        return _unavailable("no identity spans multiple sessions", **coverage)

    shifts: list[np.ndarray] = []
    eligible_centroids = 0
    degenerate_centroids = 0
    sampled_identity_count = 0
    for _, sessions in spanning:
        centroids, degenerate = _centroid_directions(
            normalized, list(sessions.values()), config.minimum_row_norm
        )
        values, _, sampled = _pairwise_cosine_values(
            centroids,
            cap=config.pairwise_max_samples,
            seed=config.seed,
        )
        if len(values):
            shifts.append(1.0 - values)
        eligible_centroids += len(centroids)
        degenerate_centroids += degenerate
        sampled_identity_count += int(sampled)
    if not shifts:
        return _unavailable(
            "cross-session centroids were degenerate",
            **coverage,
            eligible_session_centroid_count=eligible_centroids,
            degenerate_session_centroid_count=degenerate_centroids,
        )
    values = np.concatenate(shifts)
    return {
        "available": True,
        **coverage,
        "eligible_session_centroid_count": eligible_centroids,
        "degenerate_session_centroid_count": degenerate_centroids,
        "identities_with_capped_session_pairs": sampled_identity_count,
        "same_identity_cross_session_centroid_cosine_distance": (
            _finite_summary(values)
        ),
    }


def _domain_identity_coverage(
    identity_ids: tuple[Hashable, ...] | None,
    domain_ids: tuple[Hashable, ...],
) -> dict[str, Any]:
    if identity_ids is None:
        return _unavailable("identity_ids were not provided")
    identity_groups = _groups(identity_ids)
    spanning = [
        indices
        for indices in identity_groups.values()
        if len({domain_ids[index] for index in indices}) >= 2
    ]
    covered = sum(map(len, spanning))
    return {
        "available": True,
        "identity_count": len(identity_groups),
        "cross_domain_identity_count": len(spanning),
        "covered_sample_count": covered,
        "coverage_fraction": covered / len(domain_ids),
    }


def _same_identity_cross_domain_shift(
    normalized: np.ndarray,
    identity_ids: tuple[Hashable, ...] | None,
    domain_ids: tuple[Hashable, ...],
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    if identity_ids is None:
        return _unavailable("identity_ids were not provided")
    identity_groups = _groups(identity_ids)
    spanning: list[dict[Hashable, list[int]]] = []
    for indices in identity_groups.values():
        domains: dict[Hashable, list[int]] = {}
        for index in indices:
            domains.setdefault(domain_ids[index], []).append(index)
        if len(domains) >= 2:
            spanning.append(domains)
    coverage = {
        "identity_count": len(identity_groups),
        "cross_domain_identity_count": len(spanning),
    }
    if not spanning:
        return _unavailable("no identity spans multiple domains", **coverage)

    shifts: list[np.ndarray] = []
    eligible_centroids = 0
    degenerate_centroids = 0
    capped_identity_count = 0
    for domains in spanning:
        centroids, degenerate = _centroid_directions(
            normalized, list(domains.values()), config.minimum_row_norm
        )
        similarities, _, sampled = _pairwise_cosine_values(
            centroids,
            cap=config.pairwise_max_samples,
            seed=config.seed,
        )
        if len(similarities):
            shifts.append(1.0 - similarities)
        eligible_centroids += len(centroids)
        degenerate_centroids += degenerate
        capped_identity_count += int(sampled)
    if not shifts:
        return _unavailable(
            "cross-domain centroids were degenerate",
            **coverage,
            eligible_domain_centroid_count=eligible_centroids,
            degenerate_domain_centroid_count=degenerate_centroids,
        )
    return {
        "available": True,
        **coverage,
        "eligible_domain_centroid_count": eligible_centroids,
        "degenerate_domain_centroid_count": degenerate_centroids,
        "identities_with_capped_domain_pairs": capped_identity_count,
        "same_identity_cross_domain_centroid_cosine_distance": _finite_summary(
            np.concatenate(shifts)
        ),
        "confounding_warning": (
            "Matched identity coverage reduces identity-composition confounding, "
            "but session and capture conditions may still confound this shift."
        ),
    }


def _domain_diagnostics(
    normalized: np.ndarray,
    identity_ids: tuple[Hashable, ...] | None,
    domain_ids: tuple[Hashable, ...] | None,
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    if domain_ids is None:
        return _unavailable(
            "domain_ids were not provided",
            confounding_warning=_DOMAIN_CONFOUNDING_WARNING,
        )
    groups = _groups(domain_ids)
    coverage = {
        "sample_count": len(normalized),
        "coverage_fraction": 1.0,
        "domain_count": len(groups),
        "identity_overlap_coverage": _domain_identity_coverage(
            identity_ids, domain_ids
        ),
        "same_identity_cross_domain_shift": _same_identity_cross_domain_shift(
            normalized, identity_ids, domain_ids, config
        ),
        "confounding_warning": _DOMAIN_CONFOUNDING_WARNING,
    }
    if len(groups) < 2:
        return _unavailable("at least two domains are required", **coverage)
    centroids, degenerate = _centroid_directions(
        normalized, list(groups.values()), config.minimum_row_norm
    )
    similarities, sample_count, sampled = _pairwise_cosine_values(
        centroids,
        cap=config.pairwise_max_samples,
        seed=config.seed,
    )
    if len(similarities) == 0:
        return _unavailable(
            "fewer than two non-degenerate domain centroids",
            **coverage,
            eligible_domain_centroid_count=len(centroids),
            degenerate_domain_centroid_count=degenerate,
        )
    return {
        "available": True,
        **coverage,
        "eligible_domain_centroid_count": len(centroids),
        "degenerate_domain_centroid_count": degenerate,
        "centroid_sample_count": sample_count,
        "centroids_sampled": sampled,
        "between_domain_centroid_cosine_distance": _finite_summary(
            1.0 - similarities
        ),
    }


def _repeat_diagnostics(
    normalized: np.ndarray,
    norms: np.ndarray,
    repeat_ids: tuple[Hashable, ...] | None,
) -> dict[str, Any]:
    if repeat_ids is None:
        return _unavailable("repeat_ids were not provided")
    groups = _groups(repeat_ids)
    repeated = [indices for indices in groups.values() if len(indices) >= 2]
    coverage = {
        "repeat_id_count": len(groups),
        "repeated_group_count": len(repeated),
        "covered_embedding_count": sum(map(len, repeated)),
        "coverage_fraction": sum(map(len, repeated)) / len(normalized),
    }
    if not repeated:
        return _unavailable("no repeat ID has multiple embeddings", **coverage)
    dispersions = np.asarray(
        [
            1.0
            - float(
                np.clip(
                    np.linalg.norm(np.mean(normalized[indices], axis=0)),
                    0.0,
                    1.0,
                )
            )
            for indices in repeated
        ],
        dtype=np.float64,
    )
    log_norm_standard_deviations = np.asarray(
        [float(np.std(np.log(norms[indices]))) for indices in repeated],
        dtype=np.float64,
    )
    return {
        "available": True,
        **coverage,
        "within_repeat_directional_dispersion": _finite_summary(dispersions),
        "within_repeat_log_norm_standard_deviation": _finite_summary(
            log_norm_standard_deviations
        ),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _quality_diagnostics(
    norms: np.ndarray,
    quality_scores: np.ndarray | None,
) -> dict[str, Any]:
    if quality_scores is None:
        return _unavailable("quality_scores were not provided")
    coverage = {"sample_count": len(norms)}
    if len(norms) < 2:
        return _unavailable("at least two samples are required", **coverage)
    norm_ranks = _average_ranks(norms)
    quality_ranks = _average_ranks(quality_scores)
    norm_centered = norm_ranks - np.mean(norm_ranks)
    quality_centered = quality_ranks - np.mean(quality_ranks)
    denominator = float(
        np.linalg.norm(norm_centered) * np.linalg.norm(quality_centered)
    )
    if denominator == 0.0:
        constant = "embedding magnitudes" if not np.any(norm_centered) else "quality scores"
        return _unavailable(f"{constant} are constant", **coverage)
    coefficient = float(
        np.clip(
            np.dot(norm_centered, quality_centered) / denominator,
            -1.0,
            1.0,
        )
    )
    return {
        "available": True,
        **coverage,
        "spearman_rank_correlation": coefficient,
    }


def _hubness_diagnostics(
    normalized: np.ndarray,
    config: EmbeddingDiagnosticsConfig,
) -> dict[str, Any]:
    selected = _sample_indices(len(normalized), config.hubness_max_samples, config.seed)
    sampled = normalized[selected]
    coverage = {
        "source_sample_count": len(normalized),
        "sample_count": len(sampled),
        "sampled": len(sampled) < len(normalized),
        "requested_k": config.hubness_k,
    }
    if len(sampled) < 2:
        return _unavailable("at least two samples are required", **coverage)
    k = min(config.hubness_k, len(sampled) - 1)
    similarities = sampled @ sampled.T
    np.fill_diagonal(similarities, -np.inf)
    neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    occurrences = np.bincount(neighbors.ravel(), minlength=len(sampled)).astype(
        np.float64
    )
    mean = float(np.mean(occurrences))
    standard_deviation = float(np.std(occurrences))
    if standard_deviation == 0.0:
        skewness = 0.0
    else:
        skewness = float(
            np.mean(((occurrences - mean) / standard_deviation) ** 3)
        )
    slots = float(len(sampled) * k)
    top_count = max(1, ceil(0.1 * len(sampled)))
    top_fraction = float(np.sort(occurrences)[-top_count:].sum() / slots)
    concentration = float(np.sum((occurrences / slots) ** 2))
    return {
        "available": True,
        **coverage,
        "effective_k": k,
        "neighbor_slot_count": int(slots),
        "occurrence_summary": _finite_summary(occurrences),
        "occurrence_skewness": skewness,
        "zero_occurrence_fraction": float(np.mean(occurrences == 0.0)),
        "top_10_percent_occurrence_fraction": top_fraction,
        "occurrence_concentration_hhi": concentration,
    }


def compute_embedding_diagnostics(
    embeddings: Any,
    *,
    identity_ids: Any | None = None,
    session_ids: Any | None = None,
    domain_ids: Any | None = None,
    repeat_ids: Any | None = None,
    quality_scores: Any | None = None,
    config: EmbeddingDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Return a versioned aggregate report without sample IDs or vectors.

    Directional metrics use row-normalized embeddings. Covariance and magnitude
    metrics use the raw matrix. Pairwise and covariance-heavy calculations use
    deterministic caps from ``config`` and record their evaluated coverage.
    """

    if config is None:
        config = EmbeddingDiagnosticsConfig()
    elif not isinstance(config, EmbeddingDiagnosticsConfig):
        raise EmbeddingDiagnosticsError(
            "config must be an EmbeddingDiagnosticsConfig instance"
        )
    matrix = _as_embeddings(embeddings)
    norms = _row_norms(matrix)
    near_zero = norms <= config.minimum_row_norm
    if np.any(near_zero):
        raise EmbeddingDiagnosticsError(
            f"{int(np.sum(near_zero))} embedding row(s) have norm <= "
            f"minimum_row_norm {config.minimum_row_norm}"
        )
    normalized = matrix / norms[:, None]
    count = len(matrix)
    identities = (
        None if identity_ids is None else _as_ids(identity_ids, "identity_ids", count)
    )
    sessions = (
        None if session_ids is None else _as_ids(session_ids, "session_ids", count)
    )
    domains = None if domain_ids is None else _as_ids(domain_ids, "domain_ids", count)
    repeats = None if repeat_ids is None else _as_ids(repeat_ids, "repeat_ids", count)
    quality = (
        None if quality_scores is None else _as_quality(quality_scores, count)
    )

    return {
        "schema_version": EMBEDDING_DIAGNOSTICS_SCHEMA_VERSION,
        "sample_count": count,
        "nominal_dimension": matrix.shape[1],
        "config": config.to_dict(),
        "raw_norm_summary": _finite_summary(norms),
        "centered_covariance_spectrum": _covariance_spectrum(matrix, config),
        "normalized_centered_covariance_spectrum": _covariance_spectrum(
            normalized, config
        ),
        "normalized_directional_geometry": _directional_geometry(normalized, config),
        "identity_conditioned": _identity_diagnostics(
            normalized, identities, config
        ),
        "session_conditioned": _session_diagnostics(
            normalized, identities, sessions, config
        ),
        "domain_conditioned": _domain_diagnostics(
            normalized, identities, domains, config
        ),
        "repeat_noise": _repeat_diagnostics(normalized, norms, repeats),
        "magnitude_quality_association": _quality_diagnostics(norms, quality),
        "hubness": _hubness_diagnostics(normalized, config),
    }


__all__ = [
    "EMBEDDING_DIAGNOSTICS_SCHEMA_VERSION",
    "EmbeddingDiagnosticsConfig",
    "EmbeddingDiagnosticsError",
    "compute_embedding_diagnostics",
]
