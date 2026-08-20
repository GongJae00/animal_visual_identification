"""Lightweight public-report contracts for Full128 successor evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from shared.foundation.provenance import content_sha256

PUBLIC_REPORT_SCHEMA = "cvi.full128_successor_public_evaluation.v1"
DEV_SELECTION_SCHEMA = "cvi.full128_successor_dev_selection_receipt.v1"
PAIRED_BOOTSTRAP_SCHEMA = "cvi.full128_successor_paired_bootstrap.v1"
EVALUATION_SCOPES = ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
ENROLLMENT_KS = (1, 3, 5)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METRICS = ("Rank-1", "Rank-5", "Rank-10", "MRR")
_PUBLIC_LIMITATIONS = [
    "DEV is used for successor selection; CAL and exposed diagnostics are not selection inputs.",
    "EXPOSED_DIAGNOSTIC is retrospective and is not an independent final evaluation.",
    "The report evaluates exact closed-set cosine retrieval only.",
]
_SCOPE_INTERPRETATION = {
    "DEV": "MODEL_SELECTION_ONLY",
    "CAL": "CALIBRATION_REPORTING;NOT_SELECTION",
    "EXPOSED_DIAGNOSTIC": "RETROSPECTIVE_EXPOSED;NOT_FINAL_EVALUATION",
}


class Full128SuccessorEvaluationError(ValueError):
    """Raised when successor evaluation evidence violates a strict contract."""


def validate_public_successor_evaluation_report(value: object) -> dict[str, Any]:
    """Validate one aggregate-only report before terminal evidence extraction."""

    expected = {
        "schema_version",
        "visibility",
        "source_private_report_sha256",
        "evaluation_panel_sha256",
        "candidates",
        "dev_selection_receipt",
        "paired_identity_cluster_bootstrap",
        "scope_interpretation",
        "contains_embeddings",
        "contains_sample_or_identity_tokens",
        "contains_ranked_qkv_traces",
        "limitations",
        "public_report_sha256",
    }
    _keys(value, expected, "public successor report")
    report = dict(value)
    payload = {
        key: item for key, item in report.items() if key != "public_report_sha256"
    }
    if (
        report["schema_version"] != PUBLIC_REPORT_SCHEMA
        or report["visibility"] != "PUBLIC_AGGREGATE"
        or report["public_report_sha256"] != content_sha256(payload)
        or report["scope_interpretation"] != _SCOPE_INTERPRETATION
        or report["limitations"] != _PUBLIC_LIMITATIONS
        or any(
            report[field] is not False
            for field in (
                "contains_embeddings",
                "contains_sample_or_identity_tokens",
                "contains_ranked_qkv_traces",
            )
        )
    ):
        raise Full128SuccessorEvaluationError(
            "public successor report contract differs"
        )
    for field in (
        "source_private_report_sha256",
        "evaluation_panel_sha256",
        "public_report_sha256",
    ):
        _sha(report[field], field)
    _reject_private_public_keys(report)

    candidates = report["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise Full128SuccessorEvaluationError(
            "public report candidates must be non-empty"
        )
    if any(not isinstance(candidate, Mapping) for candidate in candidates):
        raise Full128SuccessorEvaluationError(
            "public report candidate must be an object"
        )
    candidate_ids = [candidate.get("successor_id") for candidate in candidates]
    if any(not isinstance(item, str) or not item.strip() for item in candidate_ids):
        raise Full128SuccessorEvaluationError(
            "public successor_id must be non-empty text"
        )
    if candidate_ids != sorted(set(candidate_ids)):
        raise Full128SuccessorEvaluationError(
            "public report candidates must be uniquely sorted"
        )
    dev_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        _keys(
            candidate,
            {
                "successor_id",
                "cache_descriptor_sha256",
                "scope_aggregates",
                "gallery_bindings",
            },
            "public successor candidate",
        )
        successor_id = candidate["successor_id"]
        _nonempty(successor_id, "public successor_id")
        _sha(candidate["cache_descriptor_sha256"], "public cache descriptor")
        aggregates = candidate["scope_aggregates"]
        if not isinstance(aggregates, list) or [
            item.get("scope") for item in aggregates
        ] != list(EVALUATION_SCOPES):
            raise Full128SuccessorEvaluationError(
                "public candidate scope aggregates differ"
            )
        for aggregate in aggregates:
            _validate_public_scope_aggregate(aggregate, successor_id=successor_id)
        dev_by_candidate[successor_id] = aggregates[0]
        bindings = candidate["gallery_bindings"]
        if not isinstance(bindings, list):
            raise Full128SuccessorEvaluationError(
                "public candidate gallery bindings must be an array"
            )
        for binding in bindings:
            _validate_public_gallery_binding(binding)

    receipt = _sanitize_selection_receipt(report["dev_selection_receipt"])
    if (
        receipt["selection_scope"] != "DEV_ONLY"
        or receipt["objective_metric"] != "Rank-1"
        or receipt["tie_policy"] != "SUCCESSOR_ID_ASC"
        or receipt["calibration_scope_used"] is not False
        or receipt["exposed_scope_used"] is not False
    ):
        raise Full128SuccessorEvaluationError("public DEV selection contract differs")
    selection_ids = [item["successor_id"] for item in receipt["candidates"]]
    if selection_ids != candidate_ids:
        raise Full128SuccessorEvaluationError(
            "public DEV selection candidate population differs"
        )
    for row in receipt["candidates"]:
        aggregate = dev_by_candidate[row["successor_id"]]
        objective = _finite(row["objective_value"], "DEV objective value")
        if (
            aggregate["status"] != "AVAILABLE"
            or row["result_sha256"] != aggregate["result_sha256"]
            or objective != aggregate["metrics"]["Rank-1"]
            or row["denominator"] != aggregate["query_count"]
        ):
            raise Full128SuccessorEvaluationError(
                "public DEV selection evidence differs from aggregate"
            )
    selected = min(
        receipt["candidates"],
        key=lambda item: (-item["objective_value"], item["successor_id"]),
    )
    if receipt["selected_successor_id"] != selected["successor_id"]:
        raise Full128SuccessorEvaluationError("public DEV selection result differs")

    paired = report["paired_identity_cluster_bootstrap"]
    if not isinstance(paired, list):
        raise Full128SuccessorEvaluationError("public paired results must be an array")
    paired_keys: set[tuple[str, str, str]] = set()
    for item in paired:
        parsed = _sanitize_paired_result(item)
        key = (
            parsed["left_successor_id"],
            parsed["right_successor_id"],
            parsed["scope"],
        )
        if (
            key in paired_keys
            or parsed["scope"] not in EVALUATION_SCOPES
            or parsed["left_successor_id"] not in dev_by_candidate
            or parsed["right_successor_id"] not in dev_by_candidate
            or parsed["left_successor_id"] == parsed["right_successor_id"]
            or [interval["metric"] for interval in parsed["intervals"]]
            != list(_METRICS)
        ):
            raise Full128SuccessorEvaluationError(
                "public paired result contract differs"
            )
        paired_keys.add(key)
        for interval in parsed["intervals"]:
            _validate_public_paired_interval(interval)
    return report


def _reject_private_public_keys(value: object) -> None:
    forbidden = {
        "sample_token",
        "registered_identity_id",
        "identity_token",
        "embedding",
        "query_rows",
        "ranked_private_qkv_traces",
        "Q",
        "K",
        "V",
        "pack_path",
    }
    if isinstance(value, Mapping):
        if forbidden & set(value):
            raise Full128SuccessorEvaluationError(
                "public report contains private evaluation fields"
            )
        for child in value.values():
            _reject_private_public_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private_public_keys(child)


def _sanitize_scope_aggregate(value: object) -> dict[str, Any]:
    expected = {
        "successor_id",
        "scope",
        "status",
        "reason",
        "query_count",
        "identity_count",
        "metrics",
        "result_sha256",
    }
    _keys(value, expected, "successor scope aggregate")
    payload = {key: item for key, item in value.items() if key != "result_sha256"}
    if value["result_sha256"] != content_sha256(payload):
        raise Full128SuccessorEvaluationError(
            "successor scope aggregate was tampered with"
        )
    metrics = value["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRICS):
        raise Full128SuccessorEvaluationError("successor scope metrics differ")
    return {
        "successor_id": value["successor_id"],
        "scope": value["scope"],
        "status": value["status"],
        "reason": value["reason"],
        "query_count": value["query_count"],
        "identity_count": value["identity_count"],
        "metrics": {metric: metrics[metric] for metric in _METRICS},
        "result_sha256": value["result_sha256"],
    }


def _sanitize_selection_receipt(value: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "selection_scope",
        "objective_metric",
        "tie_policy",
        "candidates",
        "selected_successor_id",
        "calibration_scope_used",
        "exposed_scope_used",
        "receipt_sha256",
    }
    _keys(value, expected, "DEV selection receipt")
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value["schema_version"] != DEV_SELECTION_SCHEMA or value[
        "receipt_sha256"
    ] != content_sha256(payload):
        raise Full128SuccessorEvaluationError("DEV selection receipt was tampered with")
    candidates = []
    for candidate in value["candidates"]:
        _keys(
            candidate,
            {"successor_id", "result_sha256", "objective_value", "denominator"},
            "DEV selection candidate",
        )
        candidates.append(dict(candidate))
    return {
        **payload,
        "candidates": candidates,
        "receipt_sha256": value["receipt_sha256"],
    }


def _sanitize_paired_result(value: object) -> dict[str, Any]:
    _keys(
        value,
        {"scope", "left_successor_id", "right_successor_id", "intervals"},
        "paired successor result",
    )
    intervals = []
    expected_interval = {
        "schema_version",
        "metric",
        "estimate",
        "lower_bound",
        "upper_bound",
        "confidence_level",
        "cluster_unit",
        "cluster_count",
        "paired_query_count",
        "resamples",
        "seed",
        "bootstrap_sha256",
    }
    for interval in value["intervals"]:
        _keys(interval, expected_interval, "paired bootstrap interval")
        payload = {
            key: item for key, item in interval.items() if key != "bootstrap_sha256"
        }
        if interval["schema_version"] != PAIRED_BOOTSTRAP_SCHEMA or interval[
            "bootstrap_sha256"
        ] != content_sha256(payload):
            raise Full128SuccessorEvaluationError(
                "paired bootstrap interval was tampered with"
            )
        intervals.append(dict(interval))
    return {
        "scope": value["scope"],
        "left_successor_id": value["left_successor_id"],
        "right_successor_id": value["right_successor_id"],
        "intervals": intervals,
    }


def _validate_public_scope_aggregate(value: object, *, successor_id: str) -> None:
    parsed = _sanitize_scope_aggregate(value)
    if (
        parsed["successor_id"] != successor_id
        or parsed["scope"] not in EVALUATION_SCOPES
    ):
        raise Full128SuccessorEvaluationError(
            "public scope aggregate candidate or scope differs"
        )
    metrics = parsed["metrics"]
    if parsed["status"] == "AVAILABLE":
        if (
            parsed["reason"] is not None
            or not _positive_count(parsed["query_count"])
            or not _positive_count(parsed["identity_count"])
            or any(
                not 0.0 <= _finite(metrics[metric], metric) <= 1.0
                for metric in _METRICS
            )
        ):
            raise Full128SuccessorEvaluationError(
                "available public scope aggregate contract differs"
            )
    elif parsed["status"] == "NOT_AVAILABLE":
        if (
            not isinstance(parsed["reason"], str)
            or not parsed["reason"].strip()
            or parsed["query_count"] != 0
            or parsed["identity_count"] != 0
            or any(metrics[metric] is not None for metric in _METRICS)
        ):
            raise Full128SuccessorEvaluationError(
                "unavailable public scope aggregate contract differs"
            )
    else:
        raise Full128SuccessorEvaluationError("public scope aggregate status differs")


def _validate_public_gallery_binding(value: object) -> None:
    _keys(
        value,
        {
            "scope",
            "dataset_name",
            "enrollment_k",
            "gallery_sha256",
            "scorer_hash",
            "template_count",
            "identity_count",
        },
        "public gallery binding",
    )
    if (
        value["scope"] not in EVALUATION_SCOPES
        or value["enrollment_k"] not in ENROLLMENT_KS
    ):
        raise Full128SuccessorEvaluationError("public gallery binding cohort differs")
    _nonempty(value["dataset_name"], "public gallery dataset")
    _sha(value["gallery_sha256"], "public gallery digest")
    _sha(value["scorer_hash"], "public gallery scorer")
    if not _positive_count(value["template_count"]) or not _positive_count(
        value["identity_count"]
    ):
        raise Full128SuccessorEvaluationError(
            "public gallery binding counts must be positive"
        )


def _validate_public_paired_interval(value: Mapping[str, Any]) -> None:
    if (
        value["schema_version"] != PAIRED_BOOTSTRAP_SCHEMA
        or value["confidence_level"] != 0.95
        or value["cluster_unit"] != "registered_identity_id"
        or not _positive_count(value["cluster_count"])
        or not _positive_count(value["paired_query_count"])
        or not _positive_count(value["resamples"])
        or isinstance(value["seed"], bool)
        or not isinstance(value["seed"], int)
        or value["seed"] < 0
    ):
        raise Full128SuccessorEvaluationError("public paired interval contract differs")
    lower = _finite(value["lower_bound"], "paired lower bound")
    estimate = _finite(value["estimate"], "paired estimate")
    upper = _finite(value["upper_bound"], "paired upper bound")
    if not -1.0 <= lower <= estimate <= upper <= 1.0:
        raise Full128SuccessorEvaluationError("public paired interval bounds differ")


def _positive_count(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Full128SuccessorEvaluationError(f"{label} fields differ")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Full128SuccessorEvaluationError(f"{label} must be lowercase SHA-256")


def _nonempty(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise Full128SuccessorEvaluationError(f"{label} must be non-empty text")


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise Full128SuccessorEvaluationError(f"{label} must be finite")
    return float(value)


__all__ = [
    "DEV_SELECTION_SCHEMA",
    "ENROLLMENT_KS",
    "EVALUATION_SCOPES",
    "PAIRED_BOOTSTRAP_SCHEMA",
    "PUBLIC_REPORT_SCHEMA",
    "Full128SuccessorEvaluationError",
    "validate_public_successor_evaluation_report",
]
