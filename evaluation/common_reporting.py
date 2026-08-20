"""Common cached-embedding evaluation and report-only result aggregation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.embedding_diagnostics import compute_embedding_diagnostics
from evaluation.search_metrics.metrics import identity_clustered_bootstrap_ci
from shared.foundation.provenance import canonical_json_bytes, content_sha256

CACHED_PROTOCOL_INPUT_SCHEMA_VERSION = "cvi.cached_protocol_input.v1"
COMMON_REPORT_SCHEMA_VERSION = "cvi.common_evaluation_report.v1"
MASTER_RESULTS_SCHEMA_VERSION = "cvi.master_results_table.v1"
EMBEDDING_DIMENSION = 128
RANK_KS = (1, 3, 5)
REGIONS = ("Full", "Face", "Nose")
GALLERY_SCOPES = ("isolated", "unified_balanced", "instance")
PROTOCOL_KINDS = ("IDENTITY_FREE", "OFFICIAL_IDENTITY")
SENSITIVITY_KINDS = ("mask", "box", "background")
ATTRIBUTION_KINDS = (
    "input_channel_intervention",
    "classical_feature_group",
    "feature_channel_sensitivity",
    "output_dimension_ablation",
)
FEATURE_CHANNEL_CONTEXTS = ("within", "between", "background")
ROTATIONAL_CAVEAT = (
    "Projected embedding dimensions are rotationally non-semantic; dimension "
    "ablation describes coordinates of this fixed projection only."
)


class CommonEvaluationError(ValueError):
    """Raised when common evaluation inputs or reports violate the contract."""


@dataclass(frozen=True, slots=True)
class ImmutableEvaluationReport:
    """A content-addressed report retained internally as canonical JSON bytes."""

    _canonical_report: bytes
    report_sha256: str

    @property
    def report(self) -> dict[str, Any]:
        value = json.loads(self._canonical_report)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMMON_REPORT_SCHEMA_VERSION,
            "report": self.report,
            "report_sha256": self.report_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImmutableEvaluationReport:
        _require_exact_keys(
            payload,
            {"schema_version", "report", "report_sha256"},
            "sealed evaluation report",
        )
        if payload["schema_version"] != COMMON_REPORT_SCHEMA_VERSION:
            raise CommonEvaluationError("unsupported common evaluation report schema")
        report = payload["report"]
        if not isinstance(report, dict):
            raise CommonEvaluationError("sealed report payload must be an object")
        _validate_sha256(payload["report_sha256"], "report_sha256")
        observed = content_sha256(report)
        if observed != payload["report_sha256"]:
            raise CommonEvaluationError("report hash differs; report was tampered with")
        _validate_report_content(report)
        return cls(canonical_json_bytes(report), observed)


def evaluate_cached_protocol(
    payload: Mapping[str, Any],
    *,
    bootstrap_resamples: int = 1_000,
    bootstrap_seed: int = 0,
) -> ImmutableEvaluationReport:
    """Evaluate one prepared protocol from one content-bound 128D cache family.

    The function consumes vectors and cohort membership, not aggregate metrics.
    Protocol-specific dataset code remains responsible for constructing the
    prepared input and for asserting that its cohorts implement the official
    dataset protocol.
    """

    _validate_bootstrap(bootstrap_resamples, bootstrap_seed)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "dataset",
            "protocol_id",
            "protocol_kind",
            "cache_family_sha256",
            "cache_manifest_sha256",
            "embedding_dimension",
            "requires_identity_disjoint_fitting",
            "fitting_identity_tokens",
            "declared_strata",
            "samples",
            "cohorts",
            "sensitivity_observations",
            "attribution_observations",
        },
        "cached protocol input",
    )
    if payload["schema_version"] != CACHED_PROTOCOL_INPUT_SCHEMA_VERSION:
        raise CommonEvaluationError("unsupported cached protocol input schema")
    for name in ("run_id", "dataset", "protocol_id"):
        _require_nonempty_string(payload[name], name)
    protocol_kind = payload["protocol_kind"]
    if protocol_kind not in PROTOCOL_KINDS:
        raise CommonEvaluationError("unsupported protocol_kind")
    for name in ("cache_family_sha256", "cache_manifest_sha256"):
        _validate_sha256(payload[name], name)
    if payload["embedding_dimension"] != EMBEDDING_DIMENSION:
        raise CommonEvaluationError("common evaluation requires 128D embeddings")
    requires_disjoint = payload["requires_identity_disjoint_fitting"]
    if not isinstance(requires_disjoint, bool):
        raise CommonEvaluationError(
            "requires_identity_disjoint_fitting must be boolean"
        )

    regions, views, qualities = _validate_declared_strata(payload["declared_strata"])
    samples, matrix, relevance_by_sample = _parse_samples(
        payload["samples"], protocol_kind
    )
    fitting_identities = _string_set(
        payload["fitting_identity_tokens"], "fitting_identity_tokens"
    )
    evaluation_identities = (
        set(relevance_by_sample.values())
        if protocol_kind == "OFFICIAL_IDENTITY"
        else set()
    )
    if protocol_kind == "IDENTITY_FREE" and fitting_identities:
        raise CommonEvaluationError(
            "identity-free protocols must not carry fitting identity labels"
        )
    overlap = fitting_identities & evaluation_identities
    if requires_disjoint and overlap:
        raise CommonEvaluationError("fitting/evaluation identity leakage detected")

    cohorts = _validate_cohorts(
        payload["cohorts"],
        sample_tokens=set(samples),
        regions=regions,
        views=views,
        qualities=qualities,
    )
    retrieval_modes, bootstrap_hooks, evaluated_tokens = _evaluate_cohorts(
        cohorts=cohorts,
        samples=samples,
        embeddings=matrix,
        relevance_by_sample=relevance_by_sample,
        protocol_kind=protocol_kind,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )

    diagnostic_identity_ids = (
        np.asarray([relevance_by_sample[token] for token in samples])
        if protocol_kind == "OFFICIAL_IDENTITY"
        else None
    )
    diagnostic_repeat_ids = (
        np.asarray([relevance_by_sample[token] for token in samples])
        if protocol_kind == "IDENTITY_FREE"
        else None
    )
    health = compute_embedding_diagnostics(
        matrix,
        identity_ids=diagnostic_identity_ids,
        repeat_ids=diagnostic_repeat_ids,
    )
    coverage = _build_coverage(cohorts, regions, views, qualities)
    sensitivity = _summarize_sensitivity(payload["sensitivity_observations"])
    attribution = _summarize_attribution(payload["attribution_observations"])
    report: dict[str, Any] = {
        "schema_version": COMMON_REPORT_SCHEMA_VERSION,
        "run_id": payload["run_id"],
        "dataset": payload["dataset"],
        "protocol_id": payload["protocol_id"],
        "protocol_kind": protocol_kind,
        "cache_binding": {
            "cache_family_sha256": payload["cache_family_sha256"],
            "cache_manifest_sha256": payload["cache_manifest_sha256"],
            "embedding_dimension": EMBEDDING_DIMENSION,
        },
        "common_axes": {
            "regions": list(REGIONS),
            "views": list(views),
            "qualities": list(qualities),
            "rank_ks": list(RANK_KS),
            "gallery_scopes": list(GALLERY_SCOPES),
        },
        "availability_coverage": coverage,
        "embedding_health": health,
        "retrieval_modes": retrieval_modes,
        "spatial_sensitivity": sensitivity,
        "attribution_summaries": attribution,
        "bootstrap_hooks": bootstrap_hooks,
        "partition_audit": {
            "requires_identity_disjoint_fitting": requires_disjoint,
            "fitting_identity_count": len(fitting_identities),
            "evaluation_identity_count": len(evaluation_identities),
            "fitting_evaluation_identity_overlap_count": len(overlap),
            "cohort_query_gallery_sample_overlap_count": 0,
            "evaluated_sample_count": len(evaluated_tokens),
            "fitting_identity_set_sha256": content_sha256(sorted(fitting_identities)),
            "evaluation_identity_set_sha256": content_sha256(
                sorted(evaluation_identities)
            ),
        },
        "limitations": [
            "Identity-free instance-invariance retrieval is not biometric identity retrieval.",
            ROTATIONAL_CAVEAT,
        ],
    }
    _validate_report_content(report)
    canonical = canonical_json_bytes(report)
    return ImmutableEvaluationReport(canonical, content_sha256(report))


def build_master_results_table(
    reports: Sequence[Mapping[str, Any] | ImmutableEvaluationReport],
) -> dict[str, Any]:
    """Build a deterministic long-form table exclusively from sealed reports."""

    if not reports:
        raise CommonEvaluationError("at least one sealed report is required")
    sealed: list[ImmutableEvaluationReport] = []
    for report in reports:
        if isinstance(report, ImmutableEvaluationReport):
            sealed.append(ImmutableEvaluationReport.from_dict(report.to_dict()))
        elif isinstance(report, Mapping):
            sealed.append(ImmutableEvaluationReport.from_dict(report))
        else:
            raise CommonEvaluationError("master-table inputs must be sealed reports")
    hashes = [item.report_sha256 for item in sealed]
    if len(hashes) != len(set(hashes)):
        raise CommonEvaluationError("duplicate reports are not allowed")
    family_hashes = {
        item.report["cache_binding"]["cache_family_sha256"] for item in sealed
    }
    if len(family_hashes) != 1:
        raise CommonEvaluationError("master table requires one cached embedding family")

    rows: list[dict[str, Any]] = []
    for item in sealed:
        report = item.report
        base = {
            "run_id": report["run_id"],
            "dataset": report["dataset"],
            "protocol_id": report["protocol_id"],
            "protocol_kind": report["protocol_kind"],
            "cache_family_sha256": report["cache_binding"]["cache_family_sha256"],
            "cache_manifest_sha256": report["cache_binding"]["cache_manifest_sha256"],
            "report_sha256": item.report_sha256,
        }
        rows.extend(_report_rows(report, base))
    rows.sort(key=_master_row_sort_key)
    columns = [
        "run_id",
        "dataset",
        "protocol_id",
        "protocol_kind",
        "cache_family_sha256",
        "cache_manifest_sha256",
        "report_sha256",
        "section",
        "metric_family",
        "metric_name",
        "status",
        "value",
        "numerator",
        "denominator",
        "region",
        "view",
        "quality",
        "gallery_scope",
        "rank_k",
        "ranking_unit",
        "lower_bound",
        "upper_bound",
    ]
    table_without_hash: dict[str, Any] = {
        "schema_version": MASTER_RESULTS_SCHEMA_VERSION,
        "source_report_sha256s": sorted(hashes),
        "columns": columns,
        "rows": [{column: row.get(column) for column in columns} for row in rows],
    }
    return {
        **table_without_hash,
        "table_sha256": content_sha256(table_without_hash),
    }


def _parse_samples(
    payload: Any,
    protocol_kind: str,
) -> tuple[dict[str, int], np.ndarray, dict[str, str]]:
    if not isinstance(payload, list) or not payload:
        raise CommonEvaluationError("samples must be a non-empty list")
    indices: dict[str, int] = {}
    vectors: list[list[float]] = []
    relevance: dict[str, str] = {}
    for index, sample in enumerate(payload):
        _require_exact_keys(
            sample,
            {"sample_token", "embedding", "identity_token", "instance_token"},
            f"sample {index}",
        )
        token = sample["sample_token"]
        _require_nonempty_string(token, f"sample {index} sample_token")
        if token in indices:
            raise CommonEvaluationError("sample tokens must be unique")
        vector = sample["embedding"]
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
            raise CommonEvaluationError("every cached embedding must be 128D")
        parsed_vector = [_finite_number(value, "embedding value") for value in vector]
        identity = sample["identity_token"]
        instance = sample["instance_token"]
        if protocol_kind == "OFFICIAL_IDENTITY":
            _require_nonempty_string(identity, "official identity_token")
            if instance is not None:
                _require_nonempty_string(instance, "instance_token")
            relevance[token] = identity
        else:
            if identity is not None:
                raise CommonEvaluationError(
                    "identity-free samples must not carry identity labels"
                )
            _require_nonempty_string(instance, "instance_token")
            relevance[token] = instance
        indices[token] = index
        vectors.append(parsed_vector)
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise CommonEvaluationError("cached embeddings must have finite nonzero norms")
    return indices, matrix, relevance


def _validate_declared_strata(payload: Any) -> tuple[tuple[str, ...], ...]:
    _require_exact_keys(payload, {"regions", "views", "qualities"}, "declared_strata")
    regions = _unique_strings(payload["regions"], "declared regions")
    views = _unique_strings(payload["views"], "declared views")
    qualities = _unique_strings(payload["qualities"], "declared qualities")
    if regions != REGIONS:
        raise CommonEvaluationError("declared regions must be Full, Face, Nose")
    return regions, views, qualities


def _validate_cohorts(
    payload: Any,
    *,
    sample_tokens: set[str],
    regions: tuple[str, ...],
    views: tuple[str, ...],
    qualities: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list) or not payload:
        raise CommonEvaluationError("cohorts must be a non-empty list")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, cohort in enumerate(payload):
        _require_exact_keys(
            cohort,
            {
                "cohort_id",
                "status",
                "reason",
                "query_sample_tokens",
                "gallery_sample_tokens",
                "region",
                "view",
                "quality",
                "gallery_scope",
            },
            f"cohort {index}",
        )
        cohort_id = cohort["cohort_id"]
        _require_nonempty_string(cohort_id, "cohort_id")
        if cohort_id in ids:
            raise CommonEvaluationError("cohort IDs must be unique")
        ids.add(cohort_id)
        if cohort["region"] not in regions:
            raise CommonEvaluationError("cohort region was not declared")
        if cohort["view"] not in views or cohort["quality"] not in qualities:
            raise CommonEvaluationError("cohort view/quality was not declared")
        if cohort["gallery_scope"] not in GALLERY_SCOPES:
            raise CommonEvaluationError("unsupported gallery_scope")
        status = cohort["status"]
        if status not in ("AVAILABLE", "NOT_APPLICABLE"):
            raise CommonEvaluationError("cohort status is unsupported")
        queries = _unique_strings(
            cohort["query_sample_tokens"], "query_sample_tokens", allow_empty=True
        )
        gallery = _unique_strings(
            cohort["gallery_sample_tokens"], "gallery_sample_tokens", allow_empty=True
        )
        unknown = (set(queries) | set(gallery)) - sample_tokens
        if unknown:
            raise CommonEvaluationError("cohort references unknown samples")
        if set(queries) & set(gallery):
            raise CommonEvaluationError("query/gallery sample leakage detected")
        reason = cohort["reason"]
        if status == "AVAILABLE":
            if not queries or not gallery:
                raise CommonEvaluationError(
                    "available cohorts must have query and gallery"
                )
            if reason is not None:
                raise CommonEvaluationError("available cohort reason must be null")
        else:
            _require_nonempty_string(reason, "NOT_APPLICABLE cohort reason")
            if queries or gallery:
                raise CommonEvaluationError(
                    "NOT_APPLICABLE cohorts must not carry query/gallery samples"
                )
        result.append(
            {**cohort, "query_sample_tokens": queries, "gallery_sample_tokens": gallery}
        )
    return tuple(result)


def _evaluate_cohorts(
    *,
    cohorts: tuple[dict[str, Any], ...],
    samples: dict[str, int],
    embeddings: np.ndarray,
    relevance_by_sample: dict[str, str],
    protocol_kind: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    active_name = (
        "official_identity"
        if protocol_kind == "OFFICIAL_IDENTITY"
        else "instance_invariance"
    )
    inactive_name = (
        "instance_invariance"
        if active_name == "official_identity"
        else "official_identity"
    )
    ranking_units = {
        "official_identity": "gallery_identity",
        "instance_invariance": "gallery_instance",
    }
    active_metrics: list[dict[str, Any]] = []
    hooks: list[dict[str, Any]] = []
    evaluated_tokens: set[str] = set()
    for cohort in cohorts:
        axes = _cohort_axes(cohort)
        if cohort["status"] == "NOT_APPLICABLE":
            for metric_name, rank_k in _metric_specs():
                active_metrics.append(
                    _not_applicable_metric(
                        metric_name,
                        rank_k,
                        axes,
                        ranking_units[active_name],
                        cohort["reason"],
                    )
                )
            hooks.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "status": "NOT_APPLICABLE",
                    "cluster_unit": (
                        "query_identity"
                        if protocol_kind == "OFFICIAL_IDENTITY"
                        else None
                    ),
                    "cluster_count": 0,
                    "reason": cohort["reason"],
                    "metrics": [],
                }
            )
            continue
        query_tokens = cohort["query_sample_tokens"]
        gallery_tokens = cohort["gallery_sample_tokens"]
        evaluated_tokens.update(query_tokens)
        evaluated_tokens.update(gallery_tokens)
        query_indices = np.asarray([samples[token] for token in query_tokens])
        gallery_indices = np.asarray([samples[token] for token in gallery_tokens])
        query = embeddings[query_indices]
        gallery = embeddings[gallery_indices]
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        scores = query @ gallery.T
        gallery_groups: dict[str, list[int]] = {}
        for gallery_index, token in enumerate(gallery_tokens):
            gallery_groups.setdefault(relevance_by_sample[token], []).append(
                gallery_index
            )
        group_order = tuple(gallery_groups)
        grouped_scores = np.asarray(
            [np.max(scores[:, gallery_groups[group]], axis=1) for group in group_order]
        ).T
        query_rows: list[dict[str, Any]] = []
        for query_index, token in enumerate(query_tokens):
            relevance = relevance_by_sample[token]
            if relevance not in gallery_groups:
                raise CommonEvaluationError(
                    "closed-set cohort query has no relevant gallery unit"
                )
            order = np.argsort(-grouped_scores[query_index], kind="stable")
            relevant_index = group_order.index(relevance)
            rank = int(np.flatnonzero(order == relevant_index)[0]) + 1
            row: dict[str, Any] = {
                "bootstrap_cluster_id": relevance,
                "MRR": 1.0 / rank,
            }
            row.update({f"Rank-{k}": float(rank <= k) for k in RANK_KS})
            query_rows.append(row)
        denominator = len(query_rows)
        for metric_name, rank_k in _metric_specs():
            values = [row[metric_name] for row in query_rows]
            active_metrics.append(
                {
                    **axes,
                    "metric_name": metric_name,
                    "rank_k": rank_k,
                    "ranking_unit": ranking_units[active_name],
                    "status": "AVAILABLE",
                    "value": float(np.mean(values)),
                    "numerator": float(math.fsum(values)),
                    "denominator": denominator,
                    "reason": None,
                }
            )
        cluster_count = len({row["bootstrap_cluster_id"] for row in query_rows})
        if protocol_kind == "IDENTITY_FREE":
            hooks.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "status": "NOT_APPLICABLE",
                    "cluster_unit": None,
                    "cluster_count": 0,
                    "reason": "official identity labels are unavailable",
                    "metrics": [],
                }
            )
        elif cluster_count < 2:
            hooks.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "status": "NOT_APPLICABLE",
                    "cluster_unit": "query_identity",
                    "cluster_count": cluster_count,
                    "reason": "at least two query identities are required",
                    "metrics": [],
                }
            )
        else:
            hooks.append(
                {
                    "cohort_id": cohort["cohort_id"],
                    "status": "AVAILABLE",
                    "cluster_unit": "query_identity",
                    "cluster_count": cluster_count,
                    "reason": None,
                    "metrics": [
                        identity_clustered_bootstrap_ci(
                            query_rows,
                            metric=metric_name,
                            resamples=bootstrap_resamples,
                            seed=bootstrap_seed,
                        )
                        for metric_name, _ in _metric_specs()
                    ],
                }
            )

    inactive_reason = (
        "official identity labels are unavailable"
        if inactive_name == "official_identity"
        else "official identity protocol does not use instance-invariance ranking"
    )
    return (
        {
            active_name: {
                "status": "AVAILABLE",
                "ranking_unit": ranking_units[active_name],
                "reason": None,
                "metrics": active_metrics,
            },
            inactive_name: {
                "status": "NOT_APPLICABLE",
                "ranking_unit": ranking_units[inactive_name],
                "reason": inactive_reason,
                "metrics": [],
            },
        },
        hooks,
        evaluated_tokens,
    )


def _build_coverage(
    cohorts: tuple[dict[str, Any], ...],
    regions: tuple[str, ...],
    views: tuple[str, ...],
    qualities: tuple[str, ...],
) -> list[dict[str, Any]]:
    available = [item for item in cohorts if item["status"] == "AVAILABLE"]
    denominator = sum(len(item["query_sample_tokens"]) for item in available)
    rows: list[dict[str, Any]] = []
    dimensions = (
        ("region", regions),
        ("view", views),
        ("quality", qualities),
        ("gallery_scope", GALLERY_SCOPES),
    )
    for axis, values in dimensions:
        for value in values:
            count = sum(
                len(item["query_sample_tokens"])
                for item in available
                if item[axis] == value
            )
            rows.append(
                {
                    "axis": axis,
                    "stratum": value,
                    "status": "AVAILABLE" if count else "NOT_APPLICABLE",
                    "numerator": count,
                    "denominator": denominator,
                    "coverage_fraction": count / denominator if denominator else None,
                    "reason": None if count else "no available query cohort",
                }
            )
    return rows


def _summarize_sensitivity(payload: Any) -> list[dict[str, Any]]:
    observations = _observation_groups(
        payload,
        allowed_categories=SENSITIVITY_KINDS,
        name="sensitivity_observations",
        attribution=False,
    )
    summaries = [
        _summary_record(
            category=category,
            name=name,
            context=context,
            dimension=dimension,
            metric_name=metric_name,
            values=values,
            unavailable_reason="matched intervention observations were not provided",
        )
        for (category, name, context, dimension, metric_name), values in sorted(
            observations.items()
        )
    ]
    present = {row["category"] for row in summaries}
    for category in SENSITIVITY_KINDS:
        if category not in present:
            summaries.append(
                _summary_record(
                    category=category,
                    name="ALL",
                    context=None,
                    dimension=None,
                    metric_name=None,
                    values=None,
                    unavailable_reason=(
                        "matched intervention observations were not provided"
                    ),
                )
            )
    return sorted(summaries, key=lambda row: (row["category"], row["name"]))


def _summarize_attribution(payload: Any) -> dict[str, Any]:
    observations = _observation_groups(
        payload,
        allowed_categories=ATTRIBUTION_KINDS,
        name="attribution_observations",
        attribution=True,
    )
    summaries: list[dict[str, Any]] = []
    for key in sorted(
        observations, key=lambda item: tuple(str(value) for value in item)
    ):
        category, name, context, dimension, metric_name = key
        summaries.append(
            _summary_record(
                category=category,
                name=name,
                context=context,
                dimension=dimension,
                metric_name=metric_name,
                values=observations[key],
                unavailable_reason="attribution observations were not provided",
            )
        )
    present_categories = {row["category"] for row in summaries}
    for category in ATTRIBUTION_KINDS:
        if category not in present_categories:
            summaries.append(
                _summary_record(
                    category=category,
                    name="ALL",
                    context=None,
                    dimension=None,
                    metric_name=None,
                    values=None,
                    unavailable_reason="attribution observations were not provided",
                )
            )
    present_feature_contexts = {
        row["context"]
        for row in summaries
        if row["category"] == "feature_channel_sensitivity"
        and row["status"] == "AVAILABLE"
    }
    for context in FEATURE_CHANNEL_CONTEXTS:
        if context not in present_feature_contexts:
            summaries.append(
                _summary_record(
                    category="feature_channel_sensitivity",
                    name="ALL",
                    context=context,
                    dimension=None,
                    metric_name=None,
                    values=None,
                    unavailable_reason="feature-channel context was not observed",
                )
            )
    summaries.sort(
        key=lambda row: (
            row["category"],
            row["name"],
            str(row["context"]),
            -1 if row["dimension"] is None else row["dimension"],
            str(row["metric_name"]),
        )
    )
    return {"rotational_caveat": ROTATIONAL_CAVEAT, "summaries": summaries}


def _observation_groups(
    payload: Any,
    *,
    allowed_categories: tuple[str, ...],
    name: str,
    attribution: bool,
) -> dict[tuple[str, str, str | None, int | None, str | None], list[float]]:
    if not isinstance(payload, list):
        raise CommonEvaluationError(f"{name} must be a list")
    groups: dict[tuple[str, str, str | None, int | None, str | None], list[float]] = {}
    for index, observation in enumerate(payload):
        _require_exact_keys(
            observation,
            {"category", "name", "context", "dimension", "metric_name", "values"},
            f"{name} {index}",
        )
        category = observation["category"]
        if category not in allowed_categories:
            raise CommonEvaluationError(f"unsupported {name} category")
        observation_name = observation["name"]
        _require_nonempty_string(observation_name, f"{name} name")
        context = observation["context"]
        dimension = observation["dimension"]
        metric_name = observation["metric_name"]
        if attribution and category == "feature_channel_sensitivity":
            if context not in FEATURE_CHANNEL_CONTEXTS:
                raise CommonEvaluationError("unsupported feature-channel context")
        elif context is not None:
            _require_nonempty_string(context, f"{name} context")
        if attribution and category == "output_dimension_ablation":
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or not 1 <= dimension <= EMBEDDING_DIMENSION
            ):
                raise CommonEvaluationError("ablation dimension must be in [1, 128]")
            _require_nonempty_string(metric_name, "ablation metric_name")
        elif dimension is not None or metric_name is not None:
            raise CommonEvaluationError(
                "dimension and metric_name are only valid for output ablation"
            )
        values = observation["values"]
        if not isinstance(values, list) or not values:
            raise CommonEvaluationError(f"{name} values must be non-empty")
        parsed = [_finite_number(value, f"{name} value") for value in values]
        key = (category, observation_name, context, dimension, metric_name)
        if key in groups:
            raise CommonEvaluationError(f"duplicate {name} group")
        groups[key] = parsed
    return groups


def _summary_record(
    *,
    category: str,
    name: str,
    context: str | None,
    dimension: int | None,
    metric_name: str | None,
    values: list[float] | None,
    unavailable_reason: str,
) -> dict[str, Any]:
    if values is None:
        return {
            "category": category,
            "name": name,
            "context": context,
            "dimension": dimension,
            "metric_name": metric_name,
            "status": "NOT_APPLICABLE",
            "value": None,
            "p05": None,
            "median": None,
            "p95": None,
            "numerator": None,
            "denominator": 0,
            "reason": unavailable_reason,
        }
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.05, 0.5, 0.95))
    return {
        "category": category,
        "name": name,
        "context": context,
        "dimension": dimension,
        "metric_name": metric_name,
        "status": "AVAILABLE",
        "value": float(np.mean(array)),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "numerator": float(math.fsum(values)),
        "denominator": len(values),
        "reason": None,
    }


def _validate_report_content(report: Mapping[str, Any]) -> None:
    _require_exact_keys(
        report,
        {
            "schema_version",
            "run_id",
            "dataset",
            "protocol_id",
            "protocol_kind",
            "cache_binding",
            "common_axes",
            "availability_coverage",
            "embedding_health",
            "retrieval_modes",
            "spatial_sensitivity",
            "attribution_summaries",
            "bootstrap_hooks",
            "partition_audit",
            "limitations",
        },
        "common evaluation report payload",
    )
    if report["schema_version"] != COMMON_REPORT_SCHEMA_VERSION:
        raise CommonEvaluationError("report payload schema differs")
    for name in ("run_id", "dataset", "protocol_id"):
        _require_nonempty_string(report[name], name)
    protocol_kind = report["protocol_kind"]
    if protocol_kind not in PROTOCOL_KINDS:
        raise CommonEvaluationError("report protocol kind differs")
    binding = report["cache_binding"]
    _require_exact_keys(
        binding,
        {"cache_family_sha256", "cache_manifest_sha256", "embedding_dimension"},
        "report cache binding",
    )
    _validate_sha256(binding["cache_family_sha256"], "cache_family_sha256")
    _validate_sha256(binding["cache_manifest_sha256"], "cache_manifest_sha256")
    if binding["embedding_dimension"] != EMBEDDING_DIMENSION:
        raise CommonEvaluationError("report embedding dimension differs")
    axes = report["common_axes"]
    _require_exact_keys(
        axes,
        {"regions", "views", "qualities", "rank_ks", "gallery_scopes"},
        "common axes",
    )
    if axes["regions"] != list(REGIONS):
        raise CommonEvaluationError("report region axes differ")
    if axes["rank_ks"] != list(RANK_KS):
        raise CommonEvaluationError("report rank axes differ")
    if axes["gallery_scopes"] != list(GALLERY_SCOPES):
        raise CommonEvaluationError("report gallery-scope axes differ")
    views = _unique_strings(axes["views"], "report views")
    qualities = _unique_strings(axes["qualities"], "report qualities")
    modes = report["retrieval_modes"]
    _require_exact_keys(
        modes, {"official_identity", "instance_invariance"}, "retrieval modes"
    )
    expected_active = (
        "official_identity"
        if protocol_kind == "OFFICIAL_IDENTITY"
        else "instance_invariance"
    )
    expected_units = {
        "official_identity": "gallery_identity",
        "instance_invariance": "gallery_instance",
    }
    for mode_name, mode in modes.items():
        _require_exact_keys(
            mode, {"status", "ranking_unit", "reason", "metrics"}, "retrieval mode"
        )
        if mode["ranking_unit"] != expected_units[mode_name]:
            raise CommonEvaluationError("mixed protocol/ranking units")
        expected_status = (
            "AVAILABLE" if mode_name == expected_active else "NOT_APPLICABLE"
        )
        if mode["status"] != expected_status:
            raise CommonEvaluationError("retrieval mode status contradicts protocol")
        if expected_status == "NOT_APPLICABLE":
            if mode["metrics"]:
                raise CommonEvaluationError("NOT_APPLICABLE retrieval mode has metrics")
            _require_nonempty_string(mode["reason"], "NOT_APPLICABLE reason")
        else:
            if not isinstance(mode["metrics"], list) or not mode["metrics"]:
                raise CommonEvaluationError("available retrieval mode has no metrics")
            for metric in mode["metrics"]:
                _validate_metric(
                    metric,
                    mode["ranking_unit"],
                    views=views,
                    qualities=qualities,
                )
    audit = report["partition_audit"]
    _require_exact_keys(
        audit,
        {
            "requires_identity_disjoint_fitting",
            "fitting_identity_count",
            "evaluation_identity_count",
            "fitting_evaluation_identity_overlap_count",
            "cohort_query_gallery_sample_overlap_count",
            "evaluated_sample_count",
            "fitting_identity_set_sha256",
            "evaluation_identity_set_sha256",
        },
        "partition audit",
    )
    if not isinstance(audit["requires_identity_disjoint_fitting"], bool):
        raise CommonEvaluationError("partition audit disjointness flag differs")
    for field in (
        "fitting_identity_count",
        "evaluation_identity_count",
        "fitting_evaluation_identity_overlap_count",
        "cohort_query_gallery_sample_overlap_count",
        "evaluated_sample_count",
    ):
        _nonnegative_integer(audit[field], field)
    for field in ("fitting_identity_set_sha256", "evaluation_identity_set_sha256"):
        _validate_sha256(audit[field], field)
    for field in (
        "fitting_evaluation_identity_overlap_count",
        "cohort_query_gallery_sample_overlap_count",
    ):
        if audit.get(field) != 0:
            raise CommonEvaluationError("report records evaluation leakage")
    _validate_denominated_records(report["availability_coverage"], "coverage")
    _validate_denominated_records(report["spatial_sensitivity"], "sensitivity")
    attribution = report["attribution_summaries"]
    if attribution.get("rotational_caveat") != ROTATIONAL_CAVEAT:
        raise CommonEvaluationError("output-dimension rotational caveat is missing")
    _validate_denominated_records(attribution.get("summaries"), "attribution")
    if not isinstance(report["embedding_health"], dict):
        raise CommonEvaluationError("embedding_health must be an object")
    if report["embedding_health"].get("nominal_dimension") != EMBEDDING_DIMENSION:
        raise CommonEvaluationError("embedding health dimension differs")
    if ROTATIONAL_CAVEAT not in report["limitations"]:
        raise CommonEvaluationError("report limitations omit rotational caveat")
    _require_finite_json(report)


def _validate_metric(
    metric: Any,
    ranking_unit: str,
    *,
    views: tuple[str, ...],
    qualities: tuple[str, ...],
) -> None:
    _require_exact_keys(
        metric,
        {
            "cohort_id",
            "region",
            "view",
            "quality",
            "gallery_scope",
            "metric_name",
            "rank_k",
            "ranking_unit",
            "status",
            "value",
            "numerator",
            "denominator",
            "reason",
        },
        "retrieval metric",
    )
    if metric["ranking_unit"] != ranking_unit:
        raise CommonEvaluationError("mixed protocol/ranking units")
    if metric["region"] not in REGIONS or metric["gallery_scope"] not in GALLERY_SCOPES:
        raise CommonEvaluationError("retrieval metric common axes differ")
    if metric["view"] not in views or metric["quality"] not in qualities:
        raise CommonEvaluationError("retrieval metric stratum axes differ")
    if metric["metric_name"] == "MRR":
        if metric["rank_k"] is not None:
            raise CommonEvaluationError("MRR must not carry rank_k")
    elif metric["metric_name"] in {f"Rank-{k}" for k in RANK_KS}:
        if metric["rank_k"] not in RANK_KS:
            raise CommonEvaluationError("retrieval rank_k differs")
    else:
        raise CommonEvaluationError("unsupported retrieval metric")
    _validate_denominated_record(metric, "retrieval metric")


def _validate_denominated_records(records: Any, name: str) -> None:
    if not isinstance(records, list) or not records:
        raise CommonEvaluationError(f"{name} records must be a non-empty list")
    for record in records:
        _validate_denominated_record(record, name)


def _validate_denominated_record(record: Any, name: str) -> None:
    if not isinstance(record, dict) or "denominator" not in record:
        raise CommonEvaluationError(f"{name} record is missing denominator")
    denominator = record["denominator"]
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
    ):
        raise CommonEvaluationError(f"{name} denominator is invalid")
    status = record.get("status")
    if status == "AVAILABLE":
        if denominator <= 0:
            raise CommonEvaluationError(
                f"available {name} denominator must be positive"
            )
        value = _finite_number(
            record.get("value", record.get("coverage_fraction")), f"{name} value"
        )
        numerator = record.get("numerator")
        if numerator is not None:
            parsed_numerator = _finite_number(numerator, f"{name} numerator")
            if not math.isclose(
                value,
                parsed_numerator / denominator,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise CommonEvaluationError(
                    f"{name} value differs from numerator/denominator"
                )
        if record.get("reason") is not None:
            raise CommonEvaluationError(f"available {name} reason must be null")
    elif status == "NOT_APPLICABLE":
        _require_nonempty_string(record.get("reason"), f"{name} NOT_APPLICABLE reason")
        if "value" in record and record["value"] is not None:
            raise CommonEvaluationError(f"NOT_APPLICABLE {name} has a value")
    else:
        raise CommonEvaluationError(f"{name} status is unsupported")


def _report_rows(report: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode_name, mode in report["retrieval_modes"].items():
        if mode["status"] == "NOT_APPLICABLE":
            rows.append(
                {
                    **base,
                    "section": "retrieval",
                    "metric_family": mode_name,
                    "metric_name": "ALL",
                    "status": "NOT_APPLICABLE",
                    "value": None,
                    "numerator": None,
                    "denominator": 0,
                    "ranking_unit": mode["ranking_unit"],
                }
            )
            continue
        hooks = {item["cohort_id"]: item for item in report["bootstrap_hooks"]}
        for metric in mode["metrics"]:
            interval = next(
                (
                    item
                    for item in hooks[metric["cohort_id"]]["metrics"]
                    if item["metric"] == metric["metric_name"]
                ),
                None,
            )
            rows.append(
                {
                    **base,
                    "section": "retrieval",
                    "metric_family": mode_name,
                    **{
                        key: metric[key]
                        for key in (
                            "metric_name",
                            "status",
                            "value",
                            "numerator",
                            "denominator",
                            "region",
                            "view",
                            "quality",
                            "gallery_scope",
                            "rank_k",
                            "ranking_unit",
                        )
                    },
                    "lower_bound": None
                    if interval is None
                    else interval["lower_bound"],
                    "upper_bound": None
                    if interval is None
                    else interval["upper_bound"],
                }
            )
    for coverage in report["availability_coverage"]:
        rows.append(
            {
                **base,
                "section": "coverage",
                "metric_family": coverage["axis"],
                "metric_name": coverage["stratum"],
                "status": coverage["status"],
                "value": coverage["coverage_fraction"],
                "numerator": coverage["numerator"],
                "denominator": coverage["denominator"],
            }
        )
    for section, records in (
        ("spatial_sensitivity", report["spatial_sensitivity"]),
        ("attribution", report["attribution_summaries"]["summaries"]),
    ):
        for record in records:
            rows.append(
                {
                    **base,
                    "section": section,
                    "metric_family": record["category"],
                    "metric_name": record["metric_name"] or record["name"],
                    "status": record["status"],
                    "value": record["value"],
                    "numerator": record["numerator"],
                    "denominator": record["denominator"],
                }
            )
    health_denominator = report["embedding_health"]["sample_count"]
    for path, value, status in _health_records(report["embedding_health"]):
        rows.append(
            {
                **base,
                "section": "embedding_health",
                "metric_family": "embedding_health",
                "metric_name": path,
                "status": status,
                "value": value,
                "numerator": None,
                "denominator": health_denominator if status == "AVAILABLE" else 0,
            }
        )
    return rows


def _health_records(
    payload: Any,
    prefix: str = "",
) -> list[tuple[str, int | float | None, str]]:
    rows: list[tuple[str, int | float | None, str]] = []
    if isinstance(payload, dict):
        if payload.get("available") is False:
            return [(prefix, None, "NOT_APPLICABLE")]
        for key in sorted(payload):
            if key in {
                "config",
                "schema_version",
                "available",
                "sample_count",
                "nominal_dimension",
            }:
                continue
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(_health_records(payload[key], path))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        rows.append((prefix, payload, "AVAILABLE"))
    return rows


def _master_row_sort_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        "" if row.get(key) is None else str(row[key])
        for key in (
            "dataset",
            "protocol_id",
            "run_id",
            "section",
            "metric_family",
            "metric_name",
            "region",
            "view",
            "quality",
            "gallery_scope",
            "rank_k",
        )
    )


def _cohort_axes(cohort: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cohort_id": cohort["cohort_id"],
        "region": cohort["region"],
        "view": cohort["view"],
        "quality": cohort["quality"],
        "gallery_scope": cohort["gallery_scope"],
    }


def _metric_specs() -> tuple[tuple[str, int | None], ...]:
    return (("Rank-1", 1), ("Rank-3", 3), ("Rank-5", 5), ("MRR", None))


def _not_applicable_metric(
    metric_name: str,
    rank_k: int | None,
    axes: dict[str, Any],
    ranking_unit: str,
    reason: str,
) -> dict[str, Any]:
    return {
        **axes,
        "metric_name": metric_name,
        "rank_k": rank_k,
        "ranking_unit": ranking_unit,
        "status": "NOT_APPLICABLE",
        "value": None,
        "numerator": None,
        "denominator": 0,
        "reason": reason,
    }


def _validate_bootstrap(resamples: int, seed: int) -> None:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise CommonEvaluationError("bootstrap_resamples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CommonEvaluationError("bootstrap_seed must be non-negative")


def _nonnegative_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CommonEvaluationError(f"{name} must be a non-negative integer")


def _unique_strings(
    payload: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(payload, list) or (not payload and not allow_empty):
        raise CommonEvaluationError(f"{name} must be a non-empty list")
    result: list[str] = []
    for value in payload:
        _require_nonempty_string(value, name)
        result.append(value)
    if len(result) != len(set(result)):
        raise CommonEvaluationError(f"{name} must contain unique values")
    return tuple(result)


def _string_set(payload: Any, name: str) -> set[str]:
    return set(_unique_strings(payload, name, allow_empty=True))


def _require_exact_keys(payload: Any, expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise CommonEvaluationError(f"{name} fields differ")


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CommonEvaluationError(f"{name} must be a non-empty string")


def _validate_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CommonEvaluationError(f"{name} must be a lowercase SHA-256 digest")


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CommonEvaluationError(f"{name} must be finite")
    return float(value)


def _require_finite_json(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_json(child)
    elif isinstance(value, list | tuple):
        for child in value:
            _require_finite_json(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise CommonEvaluationError("report contains nonfinite values")


__all__ = [
    "ATTRIBUTION_KINDS",
    "CACHED_PROTOCOL_INPUT_SCHEMA_VERSION",
    "COMMON_REPORT_SCHEMA_VERSION",
    "EMBEDDING_DIMENSION",
    "GALLERY_SCOPES",
    "MASTER_RESULTS_SCHEMA_VERSION",
    "RANK_KS",
    "REGIONS",
    "ROTATIONAL_CAVEAT",
    "CommonEvaluationError",
    "ImmutableEvaluationReport",
    "build_master_results_table",
    "evaluate_cached_protocol",
]
