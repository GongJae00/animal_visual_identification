from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from math import ceil
from typing import Any, Literal

import numpy as np


class RetrievalError(ValueError):
    pass


class EmbeddingNormError(RetrievalError):
    pass


class NonFiniteEmbeddingError(RetrievalError):
    pass


class MissingGalleryIdentityError(RetrievalError):
    pass


class ClosedSetViolation(RetrievalError):
    pass


class MetricInvariantError(RetrievalError):
    pass


class SampleIdValidationError(RetrievalError):
    pass


SelfMatchPolicy = Literal["include", "exclude"]
TemplateAggregation = Literal[
    "max",
    "mean",
    "median",
    "top_k_mean",
    "log_mean_exp",
    "quality_weighted_mean",
]


def _as_hashable_ids(
    values: np.ndarray,
    *,
    expected_length: int,
    name: str,
) -> tuple[Hashable, ...]:
    if values.ndim != 1:
        raise RetrievalError(f"{name} must be 1-d, got shape {values.shape}")
    if len(values) != expected_length:
        raise RetrievalError(
            f"{name} length {len(values)} != expected length {expected_length}"
        )
    result: list[Hashable] = []
    for index, value in enumerate(values.tolist()):
        if value is None or not isinstance(value, Hashable):
            raise RetrievalError(f"{name}[{index}] must be a non-null hashable ID")
        if isinstance(value, float) and not np.isfinite(value):
            raise RetrievalError(f"{name}[{index}] must be a finite ID")
        result.append(value)
    return tuple(result)


def _validate_unique_ids(values: tuple[Hashable, ...], name: str) -> None:
    seen: set[Hashable] = set()
    for value in values:
        if value in seen:
            raise SampleIdValidationError(f"duplicate template ID in {name}: {value!r}")
        seen.add(value)


def _validate_rank_ks(rank_ks: tuple[int, ...]) -> None:
    if not rank_ks:
        raise RetrievalError("rank_ks must not be empty")
    if len(set(rank_ks)) != len(rank_ks):
        raise RetrievalError("rank_ks must contain distinct values")
    if any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in rank_ks):
        raise RetrievalError("rank_ks must contain positive integers")


def _validate_embeddings(
    embs: np.ndarray,
    ids: np.ndarray,
    name: str,
) -> None:
    if embs.ndim != 2:
        raise RetrievalError(
            f"{name} embeddings must be 2-d, got shape {embs.shape}"
        )
    if ids.ndim != 1:
        raise RetrievalError(
            f"{name} ids must be 1-d, got shape {ids.shape}"
        )
    if len(embs) != len(ids):
        raise RetrievalError(
            f"{name} embeddings count {len(embs)} != ids count {len(ids)}"
        )
    if not np.all(np.isfinite(embs)):
        raise NonFiniteEmbeddingError(
            f"{name} embeddings contain non-finite values"
        )


def _validate_sample_ids(
    query_sample_ids: np.ndarray | None,
    gallery_sample_ids: np.ndarray | None,
    n_query: int,
    n_gallery: int,
) -> None:
    if query_sample_ids is None and gallery_sample_ids is None:
        return
    if query_sample_ids is None or gallery_sample_ids is None:
        raise SampleIdValidationError(
            "both query_sample_ids and gallery_sample_ids must be provided together"
        )
    if query_sample_ids.ndim != 1:
        raise SampleIdValidationError(
            f"query_sample_ids must be 1-d, got shape {query_sample_ids.shape}"
        )
    if gallery_sample_ids.ndim != 1:
        raise SampleIdValidationError(
            f"gallery_sample_ids must be 1-d, got shape {gallery_sample_ids.shape}"
        )
    if len(query_sample_ids) != n_query:
        raise SampleIdValidationError(
            f"query_sample_ids length {len(query_sample_ids)} != n_query {n_query}"
        )
    if len(gallery_sample_ids) != n_gallery:
        raise SampleIdValidationError(
            f"gallery_sample_ids length {len(gallery_sample_ids)} != n_gallery {n_gallery}"
        )
    unique_queries, q_counts = np.unique(query_sample_ids, return_counts=True)
    dup = unique_queries[q_counts > 1]
    if len(dup) > 0:
        raise SampleIdValidationError(
            f"duplicate sample IDs in query_sample_ids: {list(dup)}"
        )
    unique_gallery, g_counts = np.unique(gallery_sample_ids, return_counts=True)
    dup = unique_gallery[g_counts > 1]
    if len(dup) > 0:
        raise SampleIdValidationError(
            f"duplicate sample IDs in gallery_sample_ids: {list(dup)}"
        )


def _normalize_rows(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    zero_norm = norms.ravel() < 1e-8
    if zero_norm.any():
        n_zero = int(zero_norm.sum())
        raise EmbeddingNormError(
            f"{n_zero} row{'s' if n_zero > 1 else ''} ha{'ve' if n_zero > 1 else 's'} zero norm"
        )
    return embs / norms


def compute_cosine_score_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
) -> np.ndarray:
    """Return pairwise cosine scores after strict matrix validation."""

    query = np.asarray(query_embeddings)
    gallery = np.asarray(gallery_embeddings)
    if query.ndim != 2 or gallery.ndim != 2:
        raise RetrievalError("query and gallery embeddings must be 2-d")
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        raise RetrievalError("query and gallery embeddings must not be empty")
    if query.shape[1] != gallery.shape[1]:
        raise RetrievalError("query and gallery embedding dimensions must match")
    if not np.issubdtype(query.dtype, np.number) or not np.issubdtype(
        gallery.dtype, np.number
    ):
        raise RetrievalError("query and gallery embeddings must be numeric")
    query = query.astype(np.float64, copy=False)
    gallery = gallery.astype(np.float64, copy=False)
    if not np.isfinite(query).all() or not np.isfinite(gallery).all():
        raise NonFiniteEmbeddingError("embeddings contain non-finite values")
    return _normalize_rows(query) @ _normalize_rows(gallery).T


def _compute_ap_inp(
    ranked_pos: np.ndarray,
    n_relevant: int,
) -> tuple[float, float]:
    tp = np.cumsum(ranked_pos)
    rank = np.arange(1, len(ranked_pos) + 1, dtype=np.float64)
    precision = tp / rank
    ap = float(np.sum(precision * ranked_pos.astype(np.float64)) / n_relevant)
    positive_ranks = np.flatnonzero(ranked_pos) + 1
    last_positive_rank = int(positive_ranks[-1])
    inp = n_relevant / last_positive_rank
    if not 0.0 < inp <= 1.0:
        raise MetricInvariantError(
            f"INP={inp} outside (0,1]; "
            f"n_relevant={n_relevant}, last_positive_rank={last_positive_rank}"
        )
    return ap, inp


def evaluate_multi_template_closed_set(
    query_template_scores: np.ndarray,
    query_identity_ids: np.ndarray,
    gallery_template_identity_ids: np.ndarray,
    *,
    self_match_policy: SelfMatchPolicy,
    query_template_ids: np.ndarray | None = None,
    gallery_template_ids: np.ndarray | None = None,
    rank_ks: tuple[int, ...] = (1, 5, 10),
    aggregation: TemplateAggregation = "max",
    top_k: int | None = None,
    temperature: float | None = None,
    gallery_template_quality: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate closed-set retrieval after gallery-template aggregation.

    Gallery identities are frozen in first-template occurrence order. Scores
    tied after aggregation retain that order. With ``exclude``, exact
    query/gallery template matches are removed before identity aggregation.
    The default remains the persisted/runtime ``max`` contract; other methods
    are research/evaluation-only alternatives.
    """

    scores = np.asarray(query_template_scores)
    query_ids_array = np.asarray(query_identity_ids)
    gallery_ids_array = np.asarray(gallery_template_identity_ids)
    if scores.ndim != 2:
        raise RetrievalError(
            f"query_template_scores must be 2-d, got shape {scores.shape}"
        )
    n_query, n_gallery_templates = scores.shape
    if n_query == 0:
        raise RetrievalError("empty query set")
    if n_gallery_templates == 0:
        raise RetrievalError("empty gallery set")
    if not (
        np.issubdtype(scores.dtype, np.integer)
        or np.issubdtype(scores.dtype, np.floating)
    ):
        raise RetrievalError("query_template_scores must contain real numbers")
    scores = scores.astype(np.float64, copy=True)
    if not np.all(np.isfinite(scores)):
        raise RetrievalError("query_template_scores contain non-finite values")
    _validate_rank_ks(rank_ks)

    supported_aggregations = (
        "max",
        "mean",
        "median",
        "top_k_mean",
        "log_mean_exp",
        "quality_weighted_mean",
    )
    if not isinstance(aggregation, str) or aggregation not in supported_aggregations:
        raise RetrievalError(f"unsupported template aggregation: {aggregation!r}")
    if aggregation == "top_k_mean":
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalError("top_k must be a positive integer for top_k_mean")
    elif top_k is not None:
        raise RetrievalError("top_k is only valid for top_k_mean aggregation")
    effective_temperature: float | None = None
    if aggregation == "log_mean_exp":
        if temperature is None:
            effective_temperature = 1.0
        elif (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float, np.integer, np.floating))
            or not np.isfinite(temperature)
            or temperature <= 0.0
        ):
            raise RetrievalError(
                "temperature must be a positive finite number for log_mean_exp"
            )
        else:
            effective_temperature = float(temperature)
    elif temperature is not None:
        raise RetrievalError(
            "temperature is only valid for log_mean_exp aggregation"
        )

    quality: np.ndarray | None = None
    if aggregation == "quality_weighted_mean":
        if gallery_template_quality is None:
            raise RetrievalError(
                "gallery_template_quality is required for quality_weighted_mean"
            )
        quality = np.asarray(gallery_template_quality)
        if quality.ndim != 1 or quality.shape != (n_gallery_templates,):
            raise RetrievalError(
                "gallery_template_quality must be a 1-d vector aligned with gallery "
                f"templates, got shape {quality.shape}"
            )
        if not (
            np.issubdtype(quality.dtype, np.integer)
            or np.issubdtype(quality.dtype, np.floating)
        ):
            raise RetrievalError("gallery_template_quality must contain real numbers")
        quality = quality.astype(np.float64, copy=False)
        if not np.all(np.isfinite(quality)):
            raise RetrievalError("gallery_template_quality contains non-finite values")
        if np.any((quality < 0.0) | (quality > 1.0)):
            raise RetrievalError("gallery_template_quality values must be in [0, 1]")
    elif gallery_template_quality is not None:
        raise RetrievalError(
            "gallery_template_quality is only valid for quality_weighted_mean "
            "aggregation"
        )

    query_ids = _as_hashable_ids(
        query_ids_array,
        expected_length=n_query,
        name="query_identity_ids",
    )
    gallery_ids = _as_hashable_ids(
        gallery_ids_array,
        expected_length=n_gallery_templates,
        name="gallery_template_identity_ids",
    )
    gallery_groups: dict[Hashable, list[int]] = {}
    for template_index, identity_id in enumerate(gallery_ids):
        gallery_groups.setdefault(identity_id, []).append(template_index)
    identity_order = tuple(gallery_groups)
    identity_indices = {
        identity_id: index for index, identity_id in enumerate(identity_order)
    }

    if self_match_policy not in ("include", "exclude"):
        raise RetrievalError(
            "self_match_policy must be explicitly set to 'include' or 'exclude'"
        )
    if (query_template_ids is None) != (gallery_template_ids is None):
        raise SampleIdValidationError(
            "query_template_ids and gallery_template_ids must be provided together"
        )
    query_templates: tuple[Hashable, ...] | None = None
    gallery_templates: tuple[Hashable, ...] | None = None
    if query_template_ids is not None and gallery_template_ids is not None:
        query_templates = _as_hashable_ids(
            np.asarray(query_template_ids),
            expected_length=n_query,
            name="query_template_ids",
        )
        gallery_templates = _as_hashable_ids(
            np.asarray(gallery_template_ids),
            expected_length=n_gallery_templates,
            name="gallery_template_ids",
        )
        _validate_unique_ids(query_templates, "query_template_ids")
        _validate_unique_ids(gallery_templates, "gallery_template_ids")
        gallery_template_index = {
            template_id: index
            for index, template_id in enumerate(gallery_templates)
        }
        for query_index, template_id in enumerate(query_templates):
            gallery_index = gallery_template_index.get(template_id)
            if gallery_index is not None and (
                query_ids[query_index] != gallery_ids[gallery_index]
            ):
                raise SampleIdValidationError(
                    f"template ID {template_id!r} has inconsistent query/gallery identity"
                )
    if self_match_policy == "exclude" and query_templates is None:
        raise SampleIdValidationError(
            "self_match_policy='exclude' requires query_template_ids and "
            "gallery_template_ids"
        )

    if self_match_policy == "exclude":
        assert query_templates is not None and gallery_templates is not None
        gallery_template_index = {
            template_id: index
            for index, template_id in enumerate(gallery_templates)
        }
        for query_index, template_id in enumerate(query_templates):
            gallery_index = gallery_template_index.get(template_id)
            if gallery_index is None:
                continue
            scores[query_index, gallery_index] = -np.inf

    for query_index, query_identity_id in enumerate(query_ids):
        if query_identity_id not in identity_indices:
            raise ClosedSetViolation(
                f"query {query_index} (id={query_identity_id!r}) has no gallery identity"
            )
        relevant_scores = scores[query_index, gallery_groups[query_identity_id]]
        if not np.any(np.isfinite(relevant_scores)):
            raise ClosedSetViolation(
                f"query {query_index} (id={query_identity_id!r}) has no eligible "
                "gallery template after self-match policy"
            )

    identity_scores = np.empty((n_query, len(identity_order)), dtype=np.float64)
    for identity_index, identity_id in enumerate(identity_order):
        group_indices = gallery_groups[identity_id]
        group_scores = scores[:, group_indices]
        if aggregation == "max":
            identity_scores[:, identity_index] = np.max(group_scores, axis=1)
            continue
        for query_index in range(n_query):
            eligible = np.isfinite(group_scores[query_index])
            eligible_scores = group_scores[query_index, eligible]
            if len(eligible_scores) == 0:
                identity_scores[query_index, identity_index] = -np.inf
            elif aggregation == "mean":
                identity_scores[query_index, identity_index] = np.mean(eligible_scores)
            elif aggregation == "median":
                identity_scores[query_index, identity_index] = np.median(
                    eligible_scores
                )
            elif aggregation == "top_k_mean":
                assert top_k is not None
                selected = np.sort(eligible_scores)[-top_k:]
                identity_scores[query_index, identity_index] = np.mean(selected)
            elif aggregation == "log_mean_exp":
                assert effective_temperature is not None
                maximum = np.max(eligible_scores)
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    normalized_sum = np.sum(
                        np.exp((eligible_scores - maximum) / effective_temperature)
                    )
                identity_scores[query_index, identity_index] = maximum + (
                    effective_temperature
                    * (np.log(normalized_sum) - np.log(len(eligible_scores)))
                )
            else:
                assert quality is not None
                eligible_quality = quality[np.asarray(group_indices)[eligible]]
                total_quality = float(np.sum(eligible_quality))
                if total_quality <= 0.0:
                    raise RetrievalError(
                        "quality_weighted_mean has zero eligible quality weight for "
                        f"query {query_index}, gallery identity {identity_id!r}"
                    )
                identity_scores[query_index, identity_index] = np.average(
                    eligible_scores, weights=eligible_quality
                )

    query_rows: list[dict[str, Any]] = []
    rank_totals = {k: 0 for k in rank_ks}
    for query_index, query_identity_id in enumerate(query_ids):
        frozen_identity_index = identity_indices.get(query_identity_id)
        if frozen_identity_index is None:
            raise ClosedSetViolation(
                f"query {query_index} (id={query_identity_id!r}) has no gallery identity"
            )
        if not np.isfinite(identity_scores[query_index, frozen_identity_index]):
            raise ClosedSetViolation(
                f"query {query_index} (id={query_identity_id!r}) has no eligible "
                "gallery template after self-match policy"
            )
        order = np.argsort(-identity_scores[query_index], kind="stable")
        relevant_rank = int(np.flatnonzero(order == frozen_identity_index)[0]) + 1
        reciprocal_rank = 1.0 / relevant_rank
        rank_hits = {k: int(relevant_rank <= k) for k in rank_ks}
        for k, hit in rank_hits.items():
            rank_totals[k] += hit
        row: dict[str, Any] = {
            "query_index": query_index,
            "query_identity_id": query_identity_id,
            "bootstrap_cluster_id": query_identity_id,
            "relevant_rank": relevant_rank,
            "AP": reciprocal_rank,
            "INP": reciprocal_rank,
            "reciprocal_rank": reciprocal_rank,
        }
        row.update({f"Rank-{k}": float(hit) for k, hit in rank_hits.items()})
        query_rows.append(row)

    result: dict[str, Any] = {
        "num_queries": n_query,
        "num_gallery_templates": n_gallery_templates,
        "num_gallery_identities": len(identity_order),
        "closed_set": True,
        "ranking_unit": "gallery_identity",
        "aggregation": aggregation,
        "tie_policy": "stable_first_gallery_identity_occurrence",
        "self_match_policy": self_match_policy,
        "gallery_identity_order": list(identity_order),
        "query_rows": query_rows,
        "mAP": float(np.mean([row["AP"] for row in query_rows])),
        "mINP": float(np.mean([row["INP"] for row in query_rows])),
        "MRR": float(np.mean([row["reciprocal_rank"] for row in query_rows])),
    }
    if aggregation == "top_k_mean":
        result["aggregation_parameters"] = {"top_k": top_k}
    elif aggregation == "log_mean_exp":
        result["aggregation_parameters"] = {"temperature": effective_temperature}
    for k in rank_ks:
        result[f"Rank-{k}"] = rank_totals[k] / n_query
    return result


def identity_clustered_bootstrap_ci(
    query_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a deterministic percentile CI by resampling whole identities."""

    if not query_rows:
        raise RetrievalError("identity bootstrap requires query rows")
    if not isinstance(metric, str) or not metric:
        raise RetrievalError("metric must be a non-empty row field name")
    if not 0.0 < confidence_level < 1.0:
        raise RetrievalError("confidence_level must be in (0, 1)")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise RetrievalError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RetrievalError("seed must be a non-negative integer")

    groups: dict[Hashable, list[float]] = {}
    all_values: list[float] = []
    for row_index, row in enumerate(query_rows):
        if "bootstrap_cluster_id" not in row:
            raise RetrievalError(
                f"query row {row_index} is missing bootstrap_cluster_id"
            )
        cluster_id = row["bootstrap_cluster_id"]
        if cluster_id is None or not isinstance(cluster_id, Hashable):
            raise RetrievalError(
                f"query row {row_index} has an invalid bootstrap_cluster_id"
            )
        if metric not in row:
            raise RetrievalError(f"query row {row_index} is missing metric {metric!r}")
        value = row[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise RetrievalError(
                f"query row {row_index} metric {metric!r} must be numeric"
            )
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise RetrievalError(
                f"query row {row_index} metric {metric!r} must be finite"
            )
        groups.setdefault(cluster_id, []).append(numeric_value)
        all_values.append(numeric_value)
    if len(groups) < 2:
        raise RetrievalError("identity bootstrap requires at least two identities")

    cluster_sums = np.asarray(
        [sum(values) for values in groups.values()], dtype=np.float64
    )
    cluster_counts = np.asarray(
        [len(values) for values in groups.values()], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    cluster_count = len(groups)
    sampled_estimates = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        sampled_clusters = rng.integers(0, cluster_count, size=cluster_count)
        sampled_estimates[sample_index] = float(
            cluster_sums[sampled_clusters].sum()
            / cluster_counts[sampled_clusters].sum()
        )
    sampled_estimates.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_index = max(0, ceil(alpha * resamples) - 1)
    upper_index = min(resamples - 1, ceil((1.0 - alpha) * resamples) - 1)
    return {
        "metric": metric,
        "estimate": float(np.mean(all_values)),
        "lower_bound": float(sampled_estimates[lower_index]),
        "upper_bound": float(sampled_estimates[upper_index]),
        "confidence_level": confidence_level,
        "cluster_unit": "query_identity",
        "cluster_count": cluster_count,
        "query_row_count": len(query_rows),
        "resamples": resamples,
        "seed": seed,
        "interval_method": "whole_identity_percentile_bootstrap",
    }


def compute_retrieval_metrics(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    metric: Literal["cosine", "dot"] = "cosine",
    rank_ks: tuple[int, ...] = (1, 5, 10),
    query_sample_ids: np.ndarray | None = None,
    gallery_sample_ids: np.ndarray | None = None,
    closed_set: bool = True,
) -> dict:
    if metric != "cosine":
        raise RetrievalError(f"unsupported metric: {metric!r} (only 'cosine' supported)")
    n_query = len(query_embs)
    n_gallery = len(gallery_embs)
    if n_query == 0:
        raise RetrievalError("empty query set")
    if n_gallery == 0:
        raise RetrievalError("empty gallery set")
    _validate_embeddings(query_embs, query_ids, "query")
    _validate_embeddings(gallery_embs, gallery_ids, "gallery")
    _validate_sample_ids(query_sample_ids, gallery_sample_ids, n_query, n_gallery)
    q = _normalize_rows(query_embs)
    g = _normalize_rows(gallery_embs)
    exclude_self: np.ndarray | None = None
    if query_sample_ids is not None and gallery_sample_ids is not None:
        qs = np.asarray(query_sample_ids)
        gs = np.asarray(gallery_sample_ids)
        exclude_self = qs[:, None] == gs[None, :]
    sims = q @ g.T
    rank_hits = {f"Rank-{k}": 0 for k in rank_ks}
    aps = []
    inps = []
    num_valid = 0
    num_no_relevant = 0
    for i in range(n_query):
        row_sims = sims[i].copy()
        is_positive: np.ndarray = gallery_ids == query_ids[i]
        if exclude_self is not None:
            row_sims[exclude_self[i]] = -np.inf
            is_positive = is_positive & ~exclude_self[i]
        n_relevant = int(is_positive.sum())
        if n_relevant == 0:
            if closed_set:
                raise ClosedSetViolation(
                    f"query {i} (id={query_ids[i]}) has no matching gallery identity "
                    f"but closed_set=True"
                )
            aps.append(0.0)
            inps.append(0.0)
            num_no_relevant += 1
            continue
        num_valid += 1
        order = np.argsort(-row_sims, kind="stable")
        ranked_pos = is_positive[order]
        first_pos = int(np.where(ranked_pos)[0][0]) if ranked_pos.any() else n_gallery
        for k in rank_ks:
            if first_pos < k:
                rank_hits[f"Rank-{k}"] += 1
        ap, inp = _compute_ap_inp(ranked_pos, n_relevant)
        aps.append(ap)
        inps.append(inp)
    mAP = float(np.mean(aps)) if aps else 0.0
    mINP = float(np.mean(inps)) if inps else 0.0
    result = {
        "num_queries": n_query,
        "num_gallery": n_gallery,
        "num_valid_queries": num_valid,
        "num_queries_without_relevant_gallery": num_no_relevant,
        "metric": metric,
        "closed_set": closed_set,
        "mAP": mAP,
        "mINP": mINP,
    }
    for k in rank_ks:
        result[f"Rank-{k}"] = rank_hits[f"Rank-{k}"] / max(num_valid, 1)
    if exclude_self is not None:
        result["self_match_excluded"] = True
    return result
