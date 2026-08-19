"""Strict descriptive diagnostics for final SiBeTan evaluation reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.retrieval import identity_clustered_bootstrap_ci
from legacy.version.afn.experiments.sibetan_multievidence import BRANCHES, METHOD_BRANCHES
from foundation.provenance import content_sha256


SOURCE_BUNDLE_SCHEMA = "cvi.sibetan_multievidence_evaluation_bundle.v2"
SOURCE_REPORT_SCHEMA = "cvi.sibetan_multievidence_evaluation.v2"
DIAGNOSTIC_SCHEMA = "cvi.sibetan_diagnostics.v1"
DIAGNOSTIC_BUNDLE_SCHEMA = "cvi.sibetan_diagnostics_bundle.v1"

_METRICS = ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
_REPORT_FIELDS = {
    "schema_version",
    "status",
    "interpretation",
    "protocol",
    "transfer_weights",
    "evidence_state_counts",
    "panels",
    "input_bindings",
}
_PANEL_FIELDS = {
    "protocol",
    "episode",
    "shot",
    "population_sha256",
    "gallery_order_sha256",
    "query_order_sha256",
    "external_appearance_control",
    "fixed_population",
    "branch_availability",
    "methods",
}
_METHOD_FIELDS = {
    "branches",
    "weights",
    "positive_weight_branches",
    "equivalent_method",
    "fixed_query_count",
    "evaluated_query_count",
    "abstained_query_count",
    "query_coverage",
    "abstention_reasons",
    "candidate_active_branch_patterns",
    "metrics",
    "query_rows",
    "identity_clustered_bootstrap_cis",
    "paired_appearance_baseline_metrics",
    "paired_delta_bootstrap_cis",
}
_BRANCH_AVAILABILITY_FIELDS = {
    "gallery_effective_k",
    "gallery_effective_k_histogram",
    "available_gallery_identity_count",
    "available_query_count",
    "available_pair_count",
    "total_pair_count",
    "pair_coverage",
    "gallery_mean_reliability",
    "query_mean_reliability",
}
_INPUT_BINDING_FIELDS = {
    "split_receipt_sha256",
    "source_bundle_sha256",
    "assignment_sha256",
    "evidence_file_sha256",
    "evidence_manifest_sha256",
    "yt_policy_file_sha256",
    "yt_policy_report_sha256",
    "frozen_dinov2_sha256",
    "nose_runtime_manifest_sha256",
    "nose_onnx_sha256",
    "publisher_archives",
    "code_sha256s",
}
_CI_FIELDS = {
    "metric",
    "estimate",
    "lower_bound",
    "upper_bound",
    "confidence_level",
    "cluster_unit",
    "cluster_count",
    "query_row_count",
    "resamples",
    "seed",
    "interval_method",
}


def build_sibetan_diagnostic(
    source_bundle: Mapping[str, Any],
    *,
    source_file_sha256: str,
    source_canonical_sha256: str,
    code_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Validate one final report bundle and derive descriptive summaries."""

    _require_sha256(source_file_sha256, "source_file_sha256")
    _require_sha256(source_canonical_sha256, "source_canonical_sha256")
    code_hashes = _object(code_sha256s, "code_sha256s")
    if not code_hashes or any(not isinstance(path, str) or not path for path in code_hashes):
        raise ValueError("diagnostic code provenance paths differ")
    for path, digest in code_hashes.items():
        _require_sha256(digest, f"code_sha256s.{path}")

    bundle = _exact_object(
        source_bundle,
        {"schema_version", "report_sha256", "report"},
        "SiBeTan source bundle",
    )
    if bundle["schema_version"] != SOURCE_BUNDLE_SCHEMA:
        raise ValueError("SiBeTan source bundle schema differs")
    source_report_sha256 = _require_sha256(
        bundle["report_sha256"], "source report_sha256"
    )
    report = _exact_object(bundle["report"], _REPORT_FIELDS, "SiBeTan source report")
    if content_sha256(report) != source_report_sha256:
        raise ValueError("SiBeTan source report digest differs")
    _validate_report_header(report)

    transfer_weights = _validate_transfer_weights(report["transfer_weights"])
    panels = _list(report["panels"], "SiBeTan panels")
    if len(panels) != 3:
        raise ValueError("SiBeTan source report must contain K1, K3, and K5 panels")

    unavailable: list[dict[str, str]] = []
    panel_summaries = [
        _summarize_panel(
            panel,
            transfer_weights=transfer_weights,
            transfer_preprocessing=report["protocol"]["transfer_preprocessing"],
            unavailable=unavailable,
        )
        for panel in panels
    ]
    if [panel["shot"] for panel in panel_summaries] != [1, 3, 5]:
        raise ValueError("SiBeTan source panels must be ordered K1, K3, K5")

    bindings = _exact_object(
        report["input_bindings"], _INPUT_BINDING_FIELDS, "SiBeTan input_bindings"
    )
    for field in _INPUT_BINDING_FIELDS - {"publisher_archives", "code_sha256s"}:
        _require_sha256(bindings[field], field)
    source_code_hashes = _object(bindings["code_sha256s"], "SiBeTan source code_sha256s")
    if not source_code_hashes:
        raise ValueError("SiBeTan source code provenance is empty")
    for path, digest in source_code_hashes.items():
        if not isinstance(path, str) or not path:
            raise ValueError("SiBeTan source code provenance path differs")
        _require_sha256(digest, f"input_bindings.code_sha256s.{path}")
    policy_file_sha256 = _require_sha256(
        bindings.get("yt_policy_file_sha256"), "yt_policy_file_sha256"
    )
    policy_report_sha256 = _require_sha256(
        bindings.get("yt_policy_report_sha256"), "yt_policy_report_sha256"
    )
    result = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "PASS_DESCRIPTIVE_SIBETAN_DIAGNOSTIC",
        "interpretation": "DESCRIPTIVE_CONFOUND_SUMMARY_NO_CAUSAL_OR_BIOMETRIC_VALIDATION_CLAIM",
        "source_report": {
            "bundle_schema_version": SOURCE_BUNDLE_SCHEMA,
            "report_schema_version": SOURCE_REPORT_SCHEMA,
            "file_sha256": source_file_sha256,
            "canonical_bundle_sha256": source_canonical_sha256,
            "report_sha256": source_report_sha256,
            "input_bindings_sha256": content_sha256(bindings),
        },
        "transferred_fusion_policy": {
            "source": report["protocol"]["fusion_weight_source"],
            "sibetan_labels_used_for_policy_selection": report["protocol"][
                "sibetan_labels_used_for_policy_selection"
            ],
            "yt_policy_file_sha256": policy_file_sha256,
            "yt_policy_report_sha256": policy_report_sha256,
            "weights": transfer_weights,
        },
        "panels": panel_summaries,
        "unavailable_fields": sorted(
            unavailable, key=lambda item: (item["field"], item["reason"])
        ),
        "code_sha256s": dict(sorted(code_hashes.items())),
    }
    return result


def bundle_sibetan_diagnostic(report: Mapping[str, Any]) -> dict[str, Any]:
    """Content-bind a validated diagnostic report."""

    payload = _object(report, "SiBeTan diagnostic report")
    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA:
        raise ValueError("SiBeTan diagnostic report schema differs")
    return {
        "schema_version": DIAGNOSTIC_BUNDLE_SCHEMA,
        "report_sha256": content_sha256(payload),
        "report": dict(payload),
    }


def _validate_report_header(report: Mapping[str, Any]) -> None:
    if (
        report["schema_version"] != SOURCE_REPORT_SCHEMA
        or report["status"] != "PASS_EXPOSED_SIBETAN_FROZEN_TRANSFER_DIAGNOSTIC"
        or report["interpretation"]
        != "EXPOSED_SIBETAN_CROSS_SEQUENCE_FROZEN_TRANSFER_DIAGNOSTIC_NOT_FINAL_OR_BIOMETRIC_VALIDATION"
    ):
        raise ValueError("SiBeTan source report identity differs")
    protocol = _exact_object(
        report["protocol"],
        {
            "panel_membership",
            "missing_evidence",
            "fusion_weight_source",
            "sibetan_labels_used_for_policy_selection",
            "retrieval",
            "fusion",
            "branch_effective_k",
            "transfer_preprocessing",
            "external_control_used_in_fusion",
            "reliability",
            "bootstrap",
        },
        "SiBeTan protocol",
    )
    expected = {
        "panel_membership": "IMMUTABLE_PROTECTED_K1_K3_K5",
        "missing_evidence": "MASKED_WITHOUT_SENTINEL_BACKFILL_OR_IDENTITY_FILTERING",
        "fusion_weight_source": "YT_DEV_ONLY_EXTERNAL_SHA256_PIN",
        "sibetan_labels_used_for_policy_selection": False,
        "retrieval": "COSINE_OVER_L2_NORMALIZED_AVAILABLE_BRANCH_PROTOTYPES",
        "fusion": "MASKED_ROW_ZSCORE_THEN_CANDIDATE_RENORMALIZED_FROZEN_WEIGHTED_SUM",
        "branch_effective_k": "AVAILABLE_FIXED_SOURCE_OBSERVATIONS_ONLY",
        "transfer_preprocessing": "RECEIPT_BOUND_SHORTEST_EDGE_CENTER_CROP",
        "external_control_used_in_fusion": False,
        "reliability": "YT_DEV_FROZEN_CONTINUOUS_FACE_AND_NOSE_QUALITY",
    }
    if any(protocol[name] != value for name, value in expected.items()):
        raise ValueError("SiBeTan source protocol semantics differ")
    bootstrap = _exact_object(
        protocol["bootstrap"],
        {"cluster_unit", "resamples", "base_seed", "confidence_level"},
        "SiBeTan bootstrap protocol",
    )
    if (
        bootstrap["cluster_unit"] != "protected_identity_token"
        or _positive_int(bootstrap["resamples"], "bootstrap resamples") < 1
        or _nonnegative_int(bootstrap["base_seed"], "bootstrap base_seed") < 0
        or not math.isclose(
            _number(bootstrap["confidence_level"], "bootstrap confidence_level"),
            0.95,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("SiBeTan bootstrap protocol differs")
    state_counts = _exact_object(
        report["evidence_state_counts"], {"face", "nose"}, "SiBeTan evidence state counts"
    )
    for branch in ("face", "nose"):
        _validate_count_map(state_counts[branch], f"SiBeTan {branch} evidence state counts")


def _validate_transfer_weights(value: Any) -> dict[str, dict[str, float]]:
    weights = _object(value, "SiBeTan transfer_weights")
    fusion_methods = tuple(METHOD_BRANCHES)[3:]
    if set(weights) != set(fusion_methods):
        raise ValueError("SiBeTan transferred fusion methods differ")
    result: dict[str, dict[str, float]] = {}
    for method in fusion_methods:
        method_weights = _object(weights[method], f"{method} transfer weights")
        if set(method_weights) != set(METHOD_BRANCHES[method]):
            raise ValueError(f"{method} transferred branches differ")
        result[method] = {
            branch: _fraction(method_weights[branch], f"{method}.{branch} weight")
            for branch in METHOD_BRANCHES[method]
        }
        if not math.isclose(sum(result[method].values()), 1.0, abs_tol=1e-12):
            raise ValueError(f"{method} transferred weights do not sum to one")
    return result


def _summarize_panel(
    value: Any,
    *,
    transfer_weights: Mapping[str, Mapping[str, float]],
    transfer_preprocessing: str,
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    panel = _exact_object(value, _PANEL_FIELDS, "SiBeTan panel")
    shot = _positive_int(panel["shot"], "panel shot")
    if shot not in {1, 3, 5}:
        raise ValueError("SiBeTan panel shot differs")
    if panel["protocol"] != "SIBETAN_CROSS_SEQUENCE":
        raise ValueError("SiBeTan panel protocol differs")
    if not isinstance(panel["episode"], str) or not panel["episode"]:
        raise ValueError("SiBeTan panel episode differs")
    for field in ("population_sha256", "gallery_order_sha256", "query_order_sha256"):
        _require_sha256(panel[field], f"K{shot} {field}")

    population = _exact_object(
        panel["fixed_population"],
        {"query_count", "gallery_template_count", "gallery_identity_count", "nominal_k"},
        f"K{shot} fixed_population",
    )
    query_count = _positive_int(population["query_count"], f"K{shot} query_count")
    identity_count = _positive_int(
        population["gallery_identity_count"], f"K{shot} gallery_identity_count"
    )
    nominal_k = _positive_int(population["nominal_k"], f"K{shot} nominal_k")
    if (
        _positive_int(population["gallery_template_count"], f"K{shot} gallery_template_count")
        != shot * identity_count
        or nominal_k != shot
    ):
        raise ValueError(f"K{shot} fixed population differs")

    branch_summaries = _summarize_branch_availability(
        panel["branch_availability"],
        shot=shot,
        query_count=query_count,
        identity_count=identity_count,
    )
    methods = _object(panel["methods"], f"K{shot} methods")
    if set(methods) != set(METHOD_BRANCHES):
        raise ValueError(f"K{shot} methods differ")
    appearance_rows = _validate_method(
        methods[BRANCHES[0]],
        method=BRANCHES[0],
        shot=shot,
        fixed_query_count=query_count,
        transfer_weights=transfer_weights,
    )
    appearance_by_token = {row["sample_token"]: row for row in appearance_rows}
    method_summaries: dict[str, Any] = {}
    for method in METHOD_BRANCHES:
        rows = (
            appearance_rows
            if method == BRANCHES[0]
            else _validate_method(
                methods[method],
                method=method,
                shot=shot,
                fixed_query_count=query_count,
                transfer_weights=transfer_weights,
            )
        )
        method_summaries[method] = {
            "branches": list(METHOD_BRANCHES[method]),
            "evaluated_query_count": len(rows),
            "query_coverage": methods[method]["query_coverage"],
            "rank1_aggregation": _rank1_aggregation(
                rows, field=f"panels.K{shot}.methods.{method}.rank1_aggregation",
                unavailable=unavailable,
            ),
            "paired_appearance_delta": _paired_delta(
                method,
                rows,
                appearance_by_token=appearance_by_token,
                persisted_baseline=methods[method]["paired_appearance_baseline_metrics"],
                persisted_cis=methods[method]["paired_delta_bootstrap_cis"],
                field=f"panels.K{shot}.methods.{method}.paired_appearance_delta",
                unavailable=unavailable,
            ),
        }

    control = _summarize_preprocessing_control(
        panel["external_appearance_control"],
        appearance=methods[BRANCHES[0]],
        shot=shot,
        query_count=query_count,
        transfer_preprocessing=transfer_preprocessing,
    )
    return {
        "protocol": panel["protocol"],
        "episode": panel["episode"],
        "shot": shot,
        "population_sha256": panel["population_sha256"],
        "gallery_order_sha256": panel["gallery_order_sha256"],
        "query_order_sha256": panel["query_order_sha256"],
        "fixed_population": dict(population),
        "branch_availability": branch_summaries,
        "same_panel_preprocessing_control": control,
        "methods": method_summaries,
    }


def _summarize_branch_availability(
    value: Any, *, shot: int, query_count: int, identity_count: int
) -> dict[str, Any]:
    branches = _object(value, f"K{shot} branch_availability")
    if set(branches) != set(BRANCHES):
        raise ValueError(f"K{shot} branch availability differs")
    result = {}
    for branch in BRANCHES:
        row = _exact_object(
            branches[branch], _BRANCH_AVAILABILITY_FIELDS, f"K{shot} {branch} availability"
        )
        effective_k = _object(row["gallery_effective_k"], f"K{shot} {branch} effective K")
        if len(effective_k) != identity_count or any(
            not isinstance(identity, str)
            or not identity
            or _nonnegative_int(count, f"K{shot} {branch} effective K") > shot
            for identity, count in effective_k.items()
        ):
            raise ValueError(f"K{shot} {branch} effective K differs")
        histogram = Counter(effective_k.values())
        persisted_histogram = _object(
            row["gallery_effective_k_histogram"], f"K{shot} {branch} effective K histogram"
        )
        expected_histogram = {str(key): count for key, count in sorted(histogram.items())}
        if persisted_histogram != expected_histogram:
            raise ValueError(f"K{shot} {branch} effective K histogram differs")
        available_identities = sum(count > 0 for count in effective_k.values())
        available_queries = _nonnegative_int(
            row["available_query_count"], f"K{shot} {branch} available queries"
        )
        persisted_available_identities = _nonnegative_int(
            row["available_gallery_identity_count"],
            f"K{shot} {branch} available gallery identities",
        )
        persisted_available_pairs = _nonnegative_int(
            row["available_pair_count"], f"K{shot} {branch} available pairs"
        )
        persisted_total_pairs = _positive_int(
            row["total_pair_count"], f"K{shot} {branch} total pairs"
        )
        available_pairs = available_identities * available_queries
        total_pairs = query_count * identity_count
        expected_coverage = available_pairs / total_pairs
        if (
            persisted_available_identities != available_identities
            or persisted_available_pairs != available_pairs
            or persisted_total_pairs != total_pairs
            or available_queries > query_count
            or not math.isclose(
                _fraction(row["pair_coverage"], f"K{shot} {branch} pair coverage"),
                expected_coverage,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"K{shot} {branch} pair coverage differs")
        result[branch] = {
            "gallery_effective_k_histogram": expected_histogram,
            "available_gallery_identity_count": available_identities,
            "gallery_identity_count": identity_count,
            "available_query_count": available_queries,
            "fixed_query_count": query_count,
            "available_pair_count": available_pairs,
            "total_pair_count": total_pairs,
            "pair_coverage": expected_coverage,
            "gallery_mean_reliability": _fraction(
                row["gallery_mean_reliability"], f"K{shot} {branch} gallery reliability"
            ),
            "query_mean_reliability": _fraction(
                row["query_mean_reliability"], f"K{shot} {branch} query reliability"
            ),
        }
    return result


def _validate_method(
    value: Any,
    *,
    method: str,
    shot: int,
    fixed_query_count: int,
    transfer_weights: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    outcome = _exact_object(value, _METHOD_FIELDS, f"K{shot} {method} outcome")
    expected_branches = METHOD_BRANCHES[method]
    if outcome["branches"] != list(expected_branches):
        raise ValueError(f"K{shot} {method} branches differ")
    expected_weights = (
        {expected_branches[0]: 1.0}
        if len(expected_branches) == 1
        else dict(transfer_weights[method])
    )
    if outcome["weights"] != expected_weights:
        raise ValueError(f"K{shot} {method} weights differ from transferred policy")
    positive = [branch for branch in expected_branches if expected_weights[branch] > 0.0]
    if outcome["positive_weight_branches"] != positive:
        raise ValueError(f"K{shot} {method} positive-weight branches differ")
    expected_equivalent = (
        BRANCHES[0]
        if len(expected_branches) > 1 and positive == [BRANCHES[0]]
        else None
    )
    if outcome["equivalent_method"] != expected_equivalent:
        raise ValueError(f"K{shot} {method} equivalence marker differs")

    rows_value = _list(outcome["query_rows"], f"K{shot} {method} query_rows")
    rows = [_validate_query_row(row, method=method, shot=shot) for row in rows_value]
    if len({row["sample_token"] for row in rows}) != len(rows):
        raise ValueError(f"K{shot} {method} repeats a query row")
    evaluated = len(rows)
    persisted_fixed = _positive_int(
        outcome["fixed_query_count"], f"K{shot} {method} fixed query count"
    )
    persisted_evaluated = _nonnegative_int(
        outcome["evaluated_query_count"], f"K{shot} {method} evaluated query count"
    )
    persisted_abstained = _nonnegative_int(
        outcome["abstained_query_count"], f"K{shot} {method} abstained query count"
    )
    if (
        persisted_fixed != fixed_query_count
        or persisted_evaluated != evaluated
        or persisted_abstained != fixed_query_count - evaluated
        or not math.isclose(
            _fraction(outcome["query_coverage"], f"K{shot} {method} query coverage"),
            evaluated / fixed_query_count,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"K{shot} {method} query counts differ")
    _validate_count_map(outcome["abstention_reasons"], f"K{shot} {method} abstentions")
    _validate_count_map(
        outcome["candidate_active_branch_patterns"], f"K{shot} {method} branch patterns"
    )
    expected_metrics = _aggregate_metrics(rows) if rows else None
    _validate_metrics(outcome["metrics"], expected_metrics, f"K{shot} {method} metrics")
    cis = outcome["identity_clustered_bootstrap_cis"]
    if rows:
        ci_map = _object(cis, f"K{shot} {method} bootstrap CIs")
        if set(ci_map) != set(_METRICS):
            raise ValueError(f"K{shot} {method} bootstrap CI metrics differ")
        for metric in _METRICS:
            ci = _validate_ci_estimate(
                ci_map[metric],
                expected_metrics[metric],
                metric,
                f"K{shot} {method} CI",
                expected_query_count=len(rows),
                expected_cluster_count=len({row["identity_token"] for row in rows}),
            )
            expected_ci = identity_clustered_bootstrap_ci(
                rows,
                metric=metric,
                resamples=ci["resamples"],
                seed=ci["seed"],
                confidence_level=ci["confidence_level"],
            )
            if ci != expected_ci:
                raise ValueError(f"K{shot} {method} bootstrap CI bounds differ")
    elif cis is not None:
        raise ValueError(f"K{shot} {method} bootstrap CIs must be unavailable")
    return rows


def _validate_query_row(value: Any, *, method: str, shot: int) -> dict[str, Any]:
    row = _exact_object(
        value,
        {
            "sample_token",
            "identity_token",
            "bootstrap_cluster_id",
            "rank",
            "Rank-1",
            "Rank-5",
            "Rank-10",
            "reciprocal_rank",
        },
        f"K{shot} {method} query row",
    )
    if any(not isinstance(row[field], str) or not row[field] for field in ("sample_token", "identity_token")):
        raise ValueError(f"K{shot} {method} query identity differs")
    if row["bootstrap_cluster_id"] != row["identity_token"]:
        raise ValueError(f"K{shot} {method} bootstrap identity differs")
    rank = _positive_int(row["rank"], f"K{shot} {method} rank")
    expected = {
        "Rank-1": float(rank == 1),
        "Rank-5": float(rank <= 5),
        "Rank-10": float(rank <= 10),
        "reciprocal_rank": 1.0 / rank,
    }
    if any(
        not math.isclose(_number(row[metric], f"K{shot} {method} {metric}"), value, abs_tol=1e-12)
        for metric, value in expected.items()
    ):
        raise ValueError(f"K{shot} {method} query rank metrics differ")
    return dict(row)


def _rank1_aggregation(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    if not rows:
        unavailable.append({"field": field, "reason": "NO_EVALUATED_QUERY_OUTCOMES"})
        return {
            "status": "UNAVAILABLE",
            "reason": "NO_EVALUATED_QUERY_OUTCOMES",
            "query_count": 0,
            "identity_count": 0,
            "query_weighted_rank1": None,
            "equal_identity_rank1": None,
            "equal_identity_minus_query_weighted": None,
            "queries_per_identity_min": None,
            "queries_per_identity_max": None,
        }
    by_identity: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_identity[row["identity_token"]].append(float(row["Rank-1"]))
    query_weighted = sum(sum(values) for values in by_identity.values()) / len(rows)
    equal_identity = sum(sum(values) / len(values) for values in by_identity.values()) / len(by_identity)
    counts = [len(values) for values in by_identity.values()]
    return {
        "status": "AVAILABLE",
        "reason": None,
        "query_count": len(rows),
        "identity_count": len(by_identity),
        "query_weighted_rank1": query_weighted,
        "equal_identity_rank1": equal_identity,
        "equal_identity_minus_query_weighted": equal_identity - query_weighted,
        "queries_per_identity_min": min(counts),
        "queries_per_identity_max": max(counts),
    }


def _paired_delta(
    method: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    appearance_by_token: Mapping[str, Mapping[str, Any]],
    persisted_baseline: Any,
    persisted_cis: Any,
    field: str,
    unavailable: list[dict[str, str]],
) -> dict[str, Any]:
    if method == BRANCHES[0]:
        if persisted_baseline is not None or persisted_cis is not None:
            raise ValueError("appearance reference must not contain a paired delta")
        reason = "REFERENCE_METHOD"
    elif not rows:
        if persisted_baseline is not None or persisted_cis is not None:
            raise ValueError(f"{method} unavailable paired fields differ")
        reason = "NO_EVALUATED_QUERY_OUTCOMES"
    else:
        try:
            baselines = [appearance_by_token[row["sample_token"]] for row in rows]
        except KeyError as exc:
            raise ValueError(f"{method} query is absent from appearance reference") from exc
        expected_baseline = _aggregate_metrics(baselines)
        _validate_metrics(
            persisted_baseline, expected_baseline, f"{method} paired appearance baseline"
        )
        cis = _object(persisted_cis, f"{method} paired delta CIs")
        if set(cis) != set(_METRICS):
            raise ValueError(f"{method} paired delta CI metrics differ")
        deltas = {
            metric: _aggregate_metrics(rows)[metric] - expected_baseline[metric]
            for metric in _METRICS
        }
        for metric, estimate in deltas.items():
            ci = _validate_ci_estimate(
                cis[metric],
                estimate,
                "delta",
                f"{method} paired delta CI",
                expected_query_count=len(rows),
                expected_cluster_count=len({row["identity_token"] for row in rows}),
            )
            expected_ci = identity_clustered_bootstrap_ci(
                [
                    {
                        "bootstrap_cluster_id": row["identity_token"],
                        "delta": float(row[metric]) - float(baseline[metric]),
                    }
                    for row, baseline in zip(rows, baselines, strict=True)
                ],
                metric="delta",
                resamples=ci["resamples"],
                seed=ci["seed"],
                confidence_level=ci["confidence_level"],
            )
            if ci != expected_ci:
                raise ValueError(f"{method} paired delta CI bounds differ")
        return {
            "status": "AVAILABLE",
            "reason": None,
            "paired_query_count": len(rows),
            "method_minus_appearance": deltas,
            "identity_clustered_bootstrap_cis": dict(cis),
        }
    unavailable.append({"field": field, "reason": reason})
    return {
        "status": "UNAVAILABLE",
        "reason": reason,
        "paired_query_count": 0,
        "method_minus_appearance": None,
        "identity_clustered_bootstrap_cis": None,
    }


def _summarize_preprocessing_control(
    value: Any,
    *,
    appearance: Mapping[str, Any],
    shot: int,
    query_count: int,
    transfer_preprocessing: str,
) -> dict[str, Any]:
    control = _exact_object(
        value, {"preprocessing", "purpose", "metrics"}, f"K{shot} external control"
    )
    if (
        control["preprocessing"] != "BILINEAR_STRETCH_224X224"
        or control["purpose"]
        != "REPRODUCE_ESTABLISHED_PROTECTED_APPEARANCE_BASELINE_ONLY"
    ):
        raise ValueError(f"K{shot} external preprocessing control differs")
    external_metrics = _validate_metrics(
        control["metrics"], None, f"K{shot} external control metrics", require=True
    )
    transfer_metrics = _validate_metrics(
        appearance["metrics"], None, f"K{shot} transfer appearance metrics", require=True
    )
    if external_metrics["query_count"] != query_count or transfer_metrics["query_count"] != query_count:
        raise ValueError(f"K{shot} preprocessing control is not on the fixed query panel")
    return {
        "status": "AVAILABLE",
        "scope": "SAME_FIXED_PANEL_DESCRIPTIVE_COMPARISON",
        "receipt_bound_transfer_preprocessing": transfer_preprocessing,
        "external_control_preprocessing": control["preprocessing"],
        "external_control_purpose": control["purpose"],
        "query_count": query_count,
        "receipt_bound_transfer_rank1": transfer_metrics["Rank-1"],
        "external_control_rank1": external_metrics["Rank-1"],
        "receipt_bound_minus_external_rank1": (
            transfer_metrics["Rank-1"] - external_metrics["Rank-1"]
        ),
        "causal_interpretation": "NOT_SUPPORTED",
    }


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    return {
        "query_count": len(rows),
        **{
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in _METRICS
        },
    }


def _validate_metrics(
    value: Any,
    expected: Mapping[str, float | int] | None,
    name: str,
    *,
    require: bool = False,
) -> dict[str, float | int] | None:
    if value is None:
        if require or expected is not None:
            raise ValueError(f"{name} are unavailable")
        return None
    metrics = _exact_object(value, {"query_count", *_METRICS}, name)
    result: dict[str, float | int] = {
        "query_count": _nonnegative_int(metrics["query_count"], f"{name}.query_count")
    }
    result.update(
        {metric: _fraction(metrics[metric], f"{name}.{metric}") for metric in _METRICS}
    )
    if expected is None:
        return result
    if result["query_count"] != expected["query_count"] or any(
        not math.isclose(float(result[metric]), float(expected[metric]), abs_tol=1e-12)
        for metric in _METRICS
    ):
        raise ValueError(f"{name} differ from query outcomes")
    return result


def _validate_ci_estimate(
    value: Any,
    expected: float,
    metric: str,
    name: str,
    *,
    expected_query_count: int,
    expected_cluster_count: int,
) -> dict[str, Any]:
    ci = _exact_object(value, _CI_FIELDS, name)
    estimate = _number(ci["estimate"], f"{name}.estimate")
    lower = _number(ci["lower_bound"], f"{name}.lower_bound")
    upper = _number(ci["upper_bound"], f"{name}.upper_bound")
    if (
        ci["metric"] != metric
        or not math.isclose(estimate, expected, abs_tol=1e-12)
        or lower > upper
        or not 0.0 < _number(ci["confidence_level"], f"{name}.confidence_level") < 1.0
        or ci["cluster_unit"] != "query_identity"
        or _positive_int(ci["cluster_count"], f"{name}.cluster_count")
        != expected_cluster_count
        or _positive_int(ci["query_row_count"], f"{name}.query_row_count")
        != expected_query_count
        or _positive_int(ci["resamples"], f"{name}.resamples") < 1
        or _nonnegative_int(ci["seed"], f"{name}.seed") < 0
        or ci["interval_method"] != "whole_identity_percentile_bootstrap"
    ):
        raise ValueError(f"{name} estimate differs")
    return ci


def _validate_count_map(value: Any, name: str) -> None:
    counts = _object(value, name)
    if any(
        not isinstance(key, str) or not key or _nonnegative_int(count, name) < 0
        for key, count in counts.items()
    ):
        raise ValueError(f"{name} differs")


def _exact_object(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    result = _object(value, name)
    if set(result) != fields:
        raise ValueError(f"{name} fields differ")
    return result


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "DIAGNOSTIC_BUNDLE_SCHEMA",
    "DIAGNOSTIC_SCHEMA",
    "SOURCE_BUNDLE_SCHEMA",
    "SOURCE_REPORT_SCHEMA",
    "build_sibetan_diagnostic",
    "bundle_sibetan_diagnostic",
]
