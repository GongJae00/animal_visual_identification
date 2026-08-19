"""Fixed-panel availability and retrieval for exposed SiBeTan diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.retrieval import compute_cosine_score_matrix, identity_clustered_bootstrap_ci
from foundation.provenance import content_sha256


BRANCHES = ("A0_frozen_dinov2", "F0_frozen_dinov2", "N3_consistency_raw")
METHOD_BRANCHES = {
    BRANCHES[0]: (BRANCHES[0],),
    BRANCHES[1]: (BRANCHES[1],),
    BRANCHES[2]: (BRANCHES[2],),
    "A0_plus_F0": BRANCHES[:2],
    "A0_plus_N3": (BRANCHES[0], BRANCHES[2]),
    "F0_plus_N3": BRANCHES[1:],
    "A0_plus_F0_plus_N3": BRANCHES,
}
YT_BRANCH_NAMES = {
    BRANCHES[0]: "A0_frozen_dinov2_K5",
    BRANCHES[1]: "F0_frozen_dinov2_K5",
    BRANCHES[2]: "N3_consistency_raw_K5",
}
YT_FUSION_NAMES = {
    method: method.replace("N3", "N3") for method in tuple(METHOD_BRANCHES)[3:]
}


def frozen_transfer_weights(yt_report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Extract externally calibrated weights without consulting SiBeTan labels."""

    if yt_report.get("schema_version") in {
        "cvi.yt_masked_multievidence_policy_bundle.v1",
        "cvi.yt_masked_multievidence_policy_bundle.v2",
    }:
        if set(yt_report) != {"schema_version", "report_sha256", "report"}:
            raise ValueError("YT masked policy bundle fields differ")
        report = yt_report["report"]
        version = yt_report["schema_version"].rsplit(".", 1)[-1]
        expected_semantics = {
            "source_windows": "FIXED_EARLIEST5_LATEST5_BEFORE_BRANCH_AVAILABILITY",
            "aggregation": "L2_NORMALIZED_MEAN_AVAILABLE_FIXED_OBSERVATIONS",
            "normalization": "QUERY_ROW_ZSCORE_AVAILABLE_CANDIDATES_ONLY",
            "fusion": "CANDIDATE_WISE_POSITIVE_WEIGHT_RENORMALIZATION",
            "missing_score_sentinel": None,
        }
        if version == "v2":
            expected_semantics["reliability"] = "CONTINUOUS_FACE_AND_NOSE_QUALITY_WEIGHTED_PROTOTYPE_AND_FUSION"
        if (
            not isinstance(report, dict)
            or content_sha256(report) != yt_report["report_sha256"]
            or report.get("schema_version") != f"cvi.yt_masked_multievidence_policy.{version}"
            or report.get("status") != "PASS_YT_DEV_MASKED_FUSION_POLICY"
            or report.get("calibration", {}).get("labels_used") != "DEVELOPMENT_ONLY"
            or report.get("policy_semantics") != expected_semantics
        ):
            raise ValueError("YT masked policy content differs")
        fusions = report["calibration"]["fusions"]
        if set(fusions) != set(tuple(METHOD_BRANCHES)[3:]):
            raise ValueError("YT masked policy methods differ")
        result = {
            method: {
                branch: float(fusions[method]["selected_weights"][branch])
                for branch in METHOD_BRANCHES[method]
            }
            for method in fusions
        }
        if any(
            set(result[method]) != set(METHOD_BRANCHES[method])
            or not math.isclose(sum(result[method].values()), 1.0, abs_tol=1e-12)
            for method in result
        ):
            raise ValueError("YT masked policy weights differ")
        return result

    from legacy.version.afn.experiments.unified_multievidence import validate_report_bundle

    report = validate_report_bundle(dict(yt_report))
    result: dict[str, dict[str, float]] = {}
    for method in tuple(METHOD_BRANCHES)[3:]:
        fitted = report["calibration"]["fusions"][YT_FUSION_NAMES[method]]
        result[method] = {
            branch: float(fitted["selected_weights"][YT_BRANCH_NAMES[branch]])
            for branch in METHOD_BRANCHES[method]
        }
    return result


def evaluate_fixed_panel(
    *,
    gallery: Sequence[Mapping[str, str]],
    queries: Sequence[Mapping[str, str]],
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    transfer_weights: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Evaluate without dropping, replacing, or backfilling any panel member."""

    if not gallery or not queries:
        raise ValueError("fixed SiBeTan panel must contain gallery and query rows")
    gallery_rows = [dict(row) for row in gallery]
    query_rows = [dict(row) for row in queries]
    for role, rows in (("gallery", gallery_rows), ("query", query_rows)):
        if any(set(row) != {"sample_token", "identity_token"} for row in rows):
            raise ValueError(f"fixed SiBeTan {role} row schema differs")
        tokens = [row["sample_token"] for row in rows]
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"fixed SiBeTan {role} repeats a sample")
    if {row["sample_token"] for row in gallery_rows} & {
        row["sample_token"] for row in query_rows
    }:
        raise ValueError("fixed SiBeTan gallery and query overlap")
    gallery_identities = sorted({row["identity_token"] for row in gallery_rows})
    if {row["identity_token"] for row in query_rows} != set(gallery_identities):
        raise ValueError("fixed SiBeTan panel is not closed-set")
    shot_counts = {
        identity: sum(row["identity_token"] == identity for row in gallery_rows)
        for identity in gallery_identities
    }
    if len(set(shot_counts.values())) != 1:
        raise ValueError("fixed SiBeTan gallery shot differs by identity")
    shot = next(iter(shot_counts.values()))
    if shot not in {1, 3, 5}:
        raise ValueError("fixed SiBeTan gallery shot must be K1, K3, or K5")
    if set(embeddings) != set(BRANCHES):
        raise ValueError("SiBeTan embeddings must contain exactly A0, F0, and N3")
    if set(transfer_weights) != set(tuple(METHOD_BRANCHES)[3:]):
        raise ValueError("SiBeTan transfer policy fusion methods differ")

    methods: dict[str, Any] = {}
    for method, branches in METHOD_BRANCHES.items():
        weights = (
            {branches[0]: 1.0}
            if len(branches) == 1
            else dict(transfer_weights[method])
        )
        if set(weights) != set(branches) or not math.isclose(
            sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"SiBeTan {method} transfer weights differ")
        gallery_complete = {
            identity: all(
                row["sample_token"] in embeddings[branch]
                for row in gallery_rows
                if row["identity_token"] == identity
                for branch in branches
            )
            for identity in gallery_identities
        }
        available_queries = [
            row
            for row in query_rows
            if all(row["sample_token"] in embeddings[branch] for branch in branches)
        ]
        full_gallery = all(gallery_complete.values())
        evaluated = available_queries if full_gallery else []
        outcome: dict[str, Any] = {
            "branches": list(branches),
            "weights": weights,
            "fixed_query_count": len(query_rows),
            "fixed_gallery_template_count": len(gallery_rows),
            "fixed_gallery_identity_count": len(gallery_identities),
            "shot": shot,
            "available_query_count": len(available_queries),
            "complete_gallery_identity_count": sum(gallery_complete.values()),
            "full_gallery_available": full_gallery,
            "evaluated_query_count": len(evaluated),
            "query_coverage": len(evaluated) / len(query_rows),
            "abstained_query_count": len(query_rows) - len(evaluated),
            "status": "EVALUATED" if evaluated else "ABSTAINED_INCOMPLETE_FIXED_PANEL",
            "metrics": None,
            "query_rows": [],
        }
        if evaluated:
            branch_scores = {
                branch: _identity_scores(
                    evaluated, gallery_rows, gallery_identities, embeddings[branch]
                )
                for branch in branches
            }
            scores = (
                branch_scores[branches[0]]
                if len(branches) == 1
                else sum(
                    weights[branch] * _row_zscores(branch_scores[branch])
                    for branch in branches
                )
            )
            rows = _rank_rows(evaluated, gallery_identities, scores)
            outcome["query_rows"] = rows
            outcome["metrics"] = _metrics(rows)
        methods[method] = outcome
    return {
        "fixed_population": {
            "query_count": len(query_rows),
            "gallery_template_count": len(gallery_rows),
            "gallery_identity_count": len(gallery_identities),
            "shot": shot,
        },
        "methods": methods,
    }


def _identity_scores(
    queries: Sequence[Mapping[str, str]],
    gallery: Sequence[Mapping[str, str]],
    identities: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    query_vectors = np.stack([embeddings[row["sample_token"]] for row in queries])
    gallery_vectors = np.stack([embeddings[row["sample_token"]] for row in gallery])
    template_scores = compute_cosine_score_matrix(query_vectors, gallery_vectors)
    gallery_labels = np.asarray([row["identity_token"] for row in gallery])
    return np.stack(
        [template_scores[:, gallery_labels == identity].max(axis=1) for identity in identities],
        axis=1,
    )


def _row_zscores(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True)
    deviations = values.std(axis=1, ddof=0, keepdims=True)
    if np.any(deviations <= 1e-8):
        raise ValueError("SiBeTan score row has near-zero standard deviation")
    return (values - means) / deviations


def _rank_rows(
    queries: Sequence[Mapping[str, str]], identities: Sequence[str], scores: np.ndarray
) -> list[dict[str, Any]]:
    index_by_identity = {identity: index for index, identity in enumerate(identities)}
    rows = []
    for query, values in zip(queries, scores, strict=True):
        truth = index_by_identity[query["identity_token"]]
        order = np.argsort(-values, kind="stable")
        rank = int(np.flatnonzero(order == truth)[0]) + 1
        rows.append(
            {
                "sample_token": query["sample_token"],
                "identity_token": query["identity_token"],
                "bootstrap_cluster_id": query["identity_token"],
                "rank": rank,
                "Rank-1": float(rank == 1),
                "Rank-5": float(rank <= 5),
                "Rank-10": float(rank <= 10),
                "reciprocal_rank": 1.0 / rank,
            }
        )
    return rows


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "query_count": len(rows),
        **{
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
        },
    }


def evaluate_n4_substitution(
    *,
    gallery: Sequence[Mapping[str, str]],
    queries: Sequence[Mapping[str, str]],
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    adapted_nose_embeddings: Mapping[str, np.ndarray],
    transfer_weights: Mapping[str, Mapping[str, float]],
    quality: Mapping[str, Mapping[str, float]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compare N3 and N4 with identical panel, availability, quality, and weights."""

    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("N4 substitution bootstrap configuration differs")
    if set(adapted_nose_embeddings) != set(embeddings[BRANCHES[2]]):
        raise ValueError("N4 substitution must preserve exact N3 availability")
    candidate_embeddings = {
        branch: dict(values) for branch, values in embeddings.items()
    }
    candidate_embeddings[BRANCHES[2]] = {
        token: _normalize_vector(vector)
        for token, vector in adapted_nose_embeddings.items()
    }
    baseline = evaluate_effective_k_panel(
        gallery=gallery,
        queries=queries,
        embeddings=embeddings,
        transfer_weights=transfer_weights,
        quality=quality,
    )
    candidate = evaluate_effective_k_panel(
        gallery=gallery,
        queries=queries,
        embeddings=candidate_embeddings,
        transfer_weights=transfer_weights,
        quality=quality,
    )
    if (
        baseline["fixed_population"] != candidate["fixed_population"]
        or baseline["branch_availability"] != candidate["branch_availability"]
    ):
        raise RuntimeError("N4 substitution changed fixed population or availability")

    comparisons = {}
    for method_index, (method, branches) in enumerate(METHOD_BRANCHES.items()):
        if BRANCHES[2] not in branches:
            continue
        before = baseline["methods"][method]
        after = candidate["methods"][method]
        before_rows = before["query_rows"]
        after_rows = after["query_rows"]
        if [row["sample_token"] for row in before_rows] != [
            row["sample_token"] for row in after_rows
        ]:
            raise RuntimeError("N4 substitution changed paired query population")
        paired_cis = None
        rescue_break = None
        if before_rows:
            paired_cis = {
                metric: identity_clustered_bootstrap_ci(
                    [
                        {
                            "bootstrap_cluster_id": after_row["bootstrap_cluster_id"],
                            "delta": after_row[metric] - before_row[metric],
                        }
                        for before_row, after_row in zip(
                            before_rows, after_rows, strict=True
                        )
                    ],
                    metric="delta",
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + method_index,
                )
                for metric in ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
            }
            rescue_count = sum(
                before_row["Rank-1"] == 0.0 and after_row["Rank-1"] == 1.0
                for before_row, after_row in zip(before_rows, after_rows, strict=True)
            )
            break_count = sum(
                before_row["Rank-1"] == 1.0 and after_row["Rank-1"] == 0.0
                for before_row, after_row in zip(before_rows, after_rows, strict=True)
            )
            rescue_break = {
                "paired_query_count": len(before_rows),
                "rescue_count": rescue_count,
                "break_count": break_count,
                "rescue_fraction": rescue_count / len(before_rows),
                "break_fraction": break_count / len(before_rows),
            }
        comparisons[method] = {
            "branches": list(branches),
            "weights": before["weights"],
            "evaluated_query_count": before["evaluated_query_count"],
            "query_coverage": before["query_coverage"],
            "baseline_N3_metrics": before["metrics"],
            "candidate_N4_metrics": after["metrics"],
            "paired_N4_minus_N3_bootstrap_cis": paired_cis,
            "rescue_break": rescue_break,
            "candidate_N4_query_rows": after_rows,
        }
    return {
        "substitution": "N3_EMBEDDING_VECTOR_ONLY",
        "availability_quality_and_frozen_weights_unchanged": True,
        "fixed_population": baseline["fixed_population"],
        "branch_availability": baseline["branch_availability"][BRANCHES[2]],
        "methods": comparisons,
    }


__all__ = [
    "BRANCHES",
    "METHOD_BRANCHES",
    "evaluate_fixed_panel",
    "evaluate_n4_substitution",
    "frozen_transfer_weights",
    "evaluate_effective_k_panel",
    "fit_effective_k_weights",
    "face_reliability",
    "nose_reliability",
]


def face_reliability(*, upstream_overall: float, native_short_side: int) -> float:
    """Continuous head reliability frozen independently of identity scores."""
    if not 0.0 <= float(upstream_overall) <= 1.0 or native_short_side <= 0:
        raise ValueError("Face reliability inputs differ")
    size = math.sqrt(min(1.0, native_short_side / 64.0))
    return max(0.02, min(1.0, float(upstream_overall) * (0.5 + 0.5 * size)))


def nose_reliability(
    *, detector_confidence: float, frontality: float, native_short_side: int,
    blur_score: float, contrast_score: float,
) -> float:
    """Continuous muzzle reliability; profile and tiny evidence are downweighted, not rejected."""
    values = (detector_confidence, frontality, blur_score, contrast_score)
    if any(not 0.0 <= float(value) <= 1.0 for value in values) or native_short_side <= 0:
        raise ValueError("Nose reliability inputs differ")
    size = math.sqrt(min(1.0, native_short_side / 32.0))
    score = (
        float(detector_confidence)
        * (0.25 + 0.75 * float(blur_score))
        * (0.5 + 0.5 * float(contrast_score))
        * (0.35 + 0.65 * size)
        * (0.6 + 0.4 * float(frontality))
    )
    return max(0.02, min(1.0, score))


def fit_effective_k_weights(
    *, gallery: Sequence[Mapping[str, str]], queries: Sequence[Mapping[str, str]],
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    quality: Mapping[str, Mapping[str, float]] | None = None,
    resolution: int = 20,
) -> dict[str, Any]:
    """Fit masked fusion weights using labels from one development panel only."""
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
        raise ValueError("masked fusion resolution must be positive")
    fusion_methods = tuple(METHOD_BRANCHES)[3:]
    policy = {
        method: {branch: 1.0 / len(METHOD_BRANCHES[method]) for branch in METHOD_BRANCHES[method]}
        for method in fusion_methods
    }
    fitted = {}
    for method in fusion_methods:
        branches = METHOD_BRANCHES[method]
        best_key = None
        best_weights = None
        best_outcome = None
        feasible = 0
        for weights in _simplex_weights(len(branches), resolution):
            candidate = dict(policy)
            candidate[method] = dict(zip(branches, weights, strict=True))
            outcome = evaluate_effective_k_panel(
                gallery=gallery, queries=queries, embeddings=embeddings,
                transfer_weights=candidate, quality=quality,
            )["methods"][method]
            metrics = outcome["metrics"]
            if metrics is not None:
                feasible += 1
            key = (
                outcome["evaluated_query_count"],
                metrics["Rank-1"] if metrics else -1.0,
                metrics["reciprocal_rank"] if metrics else -1.0,
                metrics["Rank-5"] if metrics else -1.0,
                *weights,
            )
            if best_key is None or key > best_key:
                best_key, best_weights, best_outcome = key, weights, outcome
        if best_weights is None or best_outcome is None:
            raise RuntimeError("masked fusion search produced no candidate")
        policy[method] = dict(zip(branches, best_weights, strict=True))
        fitted[method] = {
            "branches": list(branches), "resolution": resolution,
            "candidate_count": resolution + 1 if len(branches) == 2 else (resolution + 1) * (resolution + 2) // 2,
            "feasible_candidate_count": feasible,
            "objective_lexicographic": ["evaluated_query_count", "Rank-1", "reciprocal_rank", "Rank-5"],
            "selected_weights": policy[method],
            "selected_metrics": best_outcome["metrics"],
            "selected_query_count": best_outcome["evaluated_query_count"],
        }
    return {"labels_used": "DEVELOPMENT_ONLY", "fusions": fitted}


def _simplex_weights(channels: int, resolution: int):
    if channels == 2:
        for first in range(resolution + 1):
            yield (first / resolution, (resolution - first) / resolution)
        return
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            yield (first / resolution, second / resolution, (resolution - first - second) / resolution)


def evaluate_effective_k_panel(
    *, gallery: Sequence[Mapping[str, str]], queries: Sequence[Mapping[str, str]],
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    transfer_weights: Mapping[str, Mapping[str, float]],
    quality: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed samples with sparse branch prototypes and masked fusion."""
    gallery_rows = [dict(row) for row in gallery]
    query_rows = [dict(row) for row in queries]
    _validate_panel_rows(gallery_rows, query_rows, embeddings, transfer_weights)
    quality_maps = (
        {branch: {token: 1.0 for token in embeddings[branch]} for branch in BRANCHES}
        if quality is None else {branch: dict(values) for branch, values in quality.items()}
    )
    _validate_quality_maps(embeddings, quality_maps)
    identities = sorted({row["identity_token"] for row in gallery_rows})
    nominal_k = len(gallery_rows) // len(identities)
    prototypes: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in BRANCHES}
    effective_k: dict[str, dict[str, int]] = {branch: {} for branch in BRANCHES}
    prototype_quality: dict[str, dict[str, float]] = {branch: {} for branch in BRANCHES}
    for branch in BRANCHES:
        for identity in identities:
            vectors = [
                embeddings[branch][row["sample_token"]]
                for row in gallery_rows
                if row["identity_token"] == identity
                and row["sample_token"] in embeddings[branch]
            ]
            vector_weights = [
                quality_maps[branch][row["sample_token"]]
                for row in gallery_rows
                if row["identity_token"] == identity
                and row["sample_token"] in embeddings[branch]
            ]
            effective_k[branch][identity] = len(vectors)
            if vectors:
                prototypes[branch][identity] = _prototype(vectors, vector_weights)
                prototype_quality[branch][identity] = float(np.mean(vector_weights))
    if any(value != nominal_k for value in effective_k[BRANCHES[0]].values()):
        raise ValueError("Appearance must cover every fixed gallery source")

    branch_scores: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        available_gallery = [identity for identity in identities if identity in prototypes[branch]]
        scores_by_query: dict[str, dict[str, float]] = {}
        for query in query_rows:
            token = query["sample_token"]
            if token not in embeddings[branch]:
                continue
            query_vector = _normalize_vector(embeddings[branch][token])
            scores_by_query[token] = {
                identity: float(np.dot(query_vector, prototypes[branch][identity]))
                for identity in available_gallery
            }
        branch_scores[branch] = {
            "gallery_effective_k": effective_k[branch],
            "gallery_effective_k_histogram": dict(sorted(Counter(effective_k[branch].values()).items())),
            "available_gallery_identity_count": len(available_gallery),
            "available_query_count": len(scores_by_query),
            "available_pair_count": sum(len(row) for row in scores_by_query.values()),
            "total_pair_count": len(query_rows) * len(identities),
            "pair_coverage": sum(len(row) for row in scores_by_query.values()) / (len(query_rows) * len(identities)),
            "gallery_mean_reliability": float(np.mean(list(prototype_quality[branch].values()))) if prototype_quality[branch] else 0.0,
            "query_mean_reliability": float(np.mean([quality_maps[branch][token] for token in scores_by_query])) if scores_by_query else 0.0,
            "scores": scores_by_query,
            "query_quality": {token: quality_maps[branch][token] for token in scores_by_query},
            "gallery_quality": prototype_quality[branch],
        }

    methods = {}
    for method, configured_branches in METHOD_BRANCHES.items():
        weights = ({configured_branches[0]: 1.0} if len(configured_branches) == 1 else dict(transfer_weights[method]))
        positive = [branch for branch in configured_branches if weights[branch] > 0.0]
        optional_declared = [branch for branch in configured_branches if branch != BRANCHES[0]]
        optional_positive = [branch for branch in positive if branch != BRANCHES[0]]
        rows, abstentions, patterns = [], Counter(), Counter()
        for query in query_rows:
            token = query["sample_token"]
            if len(configured_branches) > 1 and optional_declared and not any(
                token in branch_scores[branch]["scores"] for branch in optional_declared
            ):
                abstentions["QUERY_OPTIONAL_EVIDENCE_UNAVAILABLE"] += 1
                continue
            normalized: dict[str, dict[str, float]] = {}
            for branch in positive:
                raw = branch_scores[branch]["scores"].get(token, {})
                if len(raw) < 2:
                    continue
                values = np.asarray(list(raw.values()), dtype=np.float64)
                deviation = float(values.std(ddof=0))
                if deviation <= 1e-8:
                    continue
                mean = float(values.mean())
                normalized[branch] = {identity: (score - mean) / deviation for identity, score in raw.items()}
            fused: dict[str, float] = {}
            candidate_patterns = []
            for identity in identities:
                active = [branch for branch in positive if identity in normalized.get(branch, {})]
                effective_weights = {
                    branch: weights[branch] * math.sqrt(
                        branch_scores[branch]["query_quality"][token]
                        * branch_scores[branch]["gallery_quality"][identity]
                    )
                    for branch in active
                }
                active = [branch for branch in active if effective_weights[branch] > 0.0]
                denominator = sum(effective_weights[branch] for branch in active)
                if denominator <= 0.0:
                    break
                fused[identity] = sum(effective_weights[branch] * normalized[branch][identity] for branch in active) / denominator
                candidate_patterns.append("+".join(active))
            if len(fused) != len(identities):
                abstentions["FIXED_GALLERY_CANDIDATE_UNSCORABLE"] += 1
                continue
            if optional_positive and not any(
                branch in pattern for pattern in candidate_patterns for branch in optional_positive
            ):
                abstentions["NO_USABLE_OPTIONAL_PAIR"] += 1
                continue
            patterns.update(candidate_patterns)
            rows.append(_rank_sparse_row(query, identities, fused))
        equivalent = (
            BRANCHES[0] if len(configured_branches) > 1 and positive == [BRANCHES[0]] else None
        )
        methods[method] = {
            "branches": list(configured_branches), "weights": weights,
            "positive_weight_branches": positive, "equivalent_method": equivalent,
            "fixed_query_count": len(query_rows), "evaluated_query_count": len(rows),
            "abstained_query_count": len(query_rows) - len(rows),
            "query_coverage": len(rows) / len(query_rows),
            "abstention_reasons": dict(sorted(abstentions.items())),
            "candidate_active_branch_patterns": dict(sorted(patterns.items())),
            "metrics": _metrics(rows) if rows else None, "query_rows": rows,
        }
    for branch in BRANCHES:
        for internal in ("scores", "query_quality", "gallery_quality"):
            del branch_scores[branch][internal]
    return {
        "fixed_population": {
            "query_count": len(query_rows), "gallery_template_count": len(gallery_rows),
            "gallery_identity_count": len(identities), "nominal_k": nominal_k,
        },
        "branch_availability": branch_scores, "methods": methods,
    }


def _validate_panel_rows(gallery, queries, embeddings, transfer_weights) -> None:
    if not gallery or not queries or set(embeddings) != set(BRANCHES):
        raise ValueError("effective-K panel inputs differ")
    for rows in (gallery, queries):
        if any(set(row) != {"sample_token", "identity_token"} for row in rows):
            raise ValueError("effective-K panel row differs")
        if len({row["sample_token"] for row in rows}) != len(rows):
            raise ValueError("effective-K panel repeats a sample")
    fixed = {row["sample_token"] for row in (*gallery, *queries)}
    if {row["sample_token"] for row in gallery} & {row["sample_token"] for row in queries}:
        raise ValueError("effective-K panel roles overlap")
    if {row["identity_token"] for row in gallery} != {row["identity_token"] for row in queries}:
        raise ValueError("effective-K panel is not closed-set")
    counts = Counter(row["identity_token"] for row in gallery)
    if len(set(counts.values())) != 1 or next(iter(counts.values())) not in {1, 3, 5}:
        raise ValueError("effective-K panel nominal K differs")
    if set(embeddings[BRANCHES[0]]) != fixed:
        raise ValueError("Appearance must exactly cover the fixed panel")
    if any(not set(values).issubset(fixed) for values in embeddings.values()):
        raise ValueError("optional embeddings escape the fixed panel")
    if set(transfer_weights) != set(tuple(METHOD_BRANCHES)[3:]):
        raise ValueError("effective-K transfer weights differ")


def _validate_quality_maps(embeddings, quality) -> None:
    if set(quality) != set(BRANCHES):
        raise ValueError("effective-K quality branches differ")
    for branch in BRANCHES:
        if set(quality[branch]) != set(embeddings[branch]):
            raise ValueError("effective-K quality coverage differs")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in quality[branch].values()
        ):
            raise ValueError("effective-K quality must be finite in (0,1]")


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 1e-8:
        raise ValueError("embedding must be a finite nonzero vector")
    return vector / norm


def _prototype(values: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    normalized = np.stack([_normalize_vector(value) for value in values])
    return _normalize_vector(np.average(normalized, axis=0, weights=weights))


def _rank_sparse_row(query: Mapping[str, str], identities: Sequence[str], scores: Mapping[str, float]) -> dict[str, Any]:
    order = sorted(identities, key=lambda identity: (-scores[identity], identity))
    rank = order.index(query["identity_token"]) + 1
    return {
        "sample_token": query["sample_token"], "identity_token": query["identity_token"],
        "bootstrap_cluster_id": query["identity_token"], "rank": rank,
        "Rank-1": float(rank == 1), "Rank-5": float(rank <= 5),
        "Rank-10": float(rank <= 10), "reciprocal_rank": 1.0 / rank,
    }
