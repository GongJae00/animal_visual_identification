"""Schema-specific normalization adapters for existing aggregate reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.foundation.provenance import content_sha256
from visualization.contracts import FigureContractError, FigureData, SourceBinding
from visualization.privacy import PublicationScope
from visualization.registry import FIGURE_BY_ID


def adapt_master_results_table(
    value: Mapping[str, Any],
    *,
    figure_id: str = "13_primary_results_paired_deltas",
    scope: PublicationScope = PublicationScope.PAPER,
) -> FigureData:
    payload = dict(value)
    _require_figure_kind(figure_id, "result_forest")
    expected = {
        "schema_version",
        "source_report_sha256s",
        "columns",
        "rows",
        "table_sha256",
    }
    if (
        set(payload) != expected
        or payload["schema_version"] != "cvi.master_results_table.v1"
    ):
        raise FigureContractError("master results table schema differs")
    without_hash = {key: payload[key] for key in expected - {"table_sha256"}}
    if content_sha256(without_hash) != payload["table_sha256"]:
        raise FigureContractError(
            "master results table hash differs; input was tampered with"
        )
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise FigureContractError("master results rows must be an array")
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or row.get("section") != "retrieval":
            continue
        estimate = row.get("value")
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
            continue
        if not 0 <= float(estimate) <= 1:
            continue
        lower = row.get("lower_bound")
        upper = row.get("upper_bound")
        if lower is None:
            lower = estimate
        if upper is None:
            upper = estimate
        if isinstance(lower, bool) or not isinstance(lower, (int, float)):
            raise FigureContractError("master results lower_bound must be numeric")
        if isinstance(upper, bool) or not isinstance(upper, (int, float)):
            raise FigureContractError("master results upper_bound must be numeric")
        label_parts = [
            row.get("metric_name"),
            row.get("region"),
            row.get("gallery_scope"),
        ]
        label = " | ".join(str(item) for item in label_parts if item is not None)
        normalized.append(
            {
                "label": label,
                "estimate": float(estimate),
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    if not normalized:
        raise FigureContractError(
            "master results table has no available retrieval rows"
        )
    return FigureData.create(
        figure_id=figure_id,
        kind="result_forest",
        scope=scope,
        title="Retrieval results",
        caption="Aggregate retrieval estimates normalized from a content-bound master results table.",
        limitations=(
            "Intervals are shown only when present in the source table; point-only rows are not uncertainty estimates.",
        ),
        source_bindings=(
            SourceBinding(
                "master-results-table",
                payload["schema_version"],
                content_sha256(payload),
            ),
        ),
        payload={
            "rows": normalized,
            "x_label": "Metric value",
            "x_min": 0.0,
            "x_max": 1.0,
            "reference": None,
        },
    )


def adapt_common_evaluation_report(
    value: Mapping[str, Any],
    *,
    figure_id: str = "13_primary_results_paired_deltas",
    scope: PublicationScope = PublicationScope.PAPER,
) -> FigureData:
    _require_figure_kind(figure_id, "result_forest")
    from evaluation.common_reporting import ImmutableEvaluationReport

    sealed = ImmutableEvaluationReport.from_dict(value)
    report = sealed.report
    rows = []
    for mode_name in ("official_identity", "instance_invariance"):
        mode = report["retrieval_modes"][mode_name]
        if mode["status"] != "AVAILABLE":
            continue
        for metric in mode["metrics"]:
            value_number = metric["value"]
            if value_number is None or not 0 <= value_number <= 1:
                continue
            rows.append(
                {
                    "label": " | ".join(
                        str(item)
                        for item in (
                            metric["metric_name"],
                            metric["region"],
                            metric["gallery_scope"],
                        )
                        if item is not None
                    ),
                    "estimate": value_number,
                    "lower": value_number,
                    "upper": value_number,
                }
            )
    if not rows:
        raise FigureContractError(
            "common evaluation report has no available retrieval rows"
        )
    return FigureData.create(
        figure_id=figure_id,
        kind="result_forest",
        scope=scope,
        title="Common-protocol retrieval results",
        caption="Available aggregate retrieval values from one sealed common evaluation report.",
        limitations=tuple(report["limitations"]),
        source_bindings=(
            SourceBinding(
                "sealed-common-evaluation-report",
                value["schema_version"],
                content_sha256(dict(value)),
            ),
        ),
        payload={
            "rows": rows,
            "x_label": "Metric value",
            "x_min": 0.0,
            "x_max": 1.0,
            "reference": None,
        },
    )


def adapt_protected_evaluation_v3(
    value: Mapping[str, Any],
    *,
    figure_id: str = "13_primary_results_paired_deltas",
    scope: PublicationScope = PublicationScope.PRIVATE,
) -> FigureData:
    """Normalize v3 integrity metrics without promoting them to final results."""

    payload = dict(value)
    _require_figure_kind(figure_id, "result_forest")
    if payload.get("schema_version") != "cvi.evaluation.report.v3":
        raise FigureContractError("protected evaluation report schema differs")
    if scope is not PublicationScope.PRIVATE:
        raise PermissionError(
            "v3 protected reports are not valid for public or paper reporting"
        )
    if payload.get("receipt_chain_verified") is not True:
        raise FigureContractError("protected report receipt chain is not verified")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise FigureContractError("protected report metrics are missing")
    rows = [
        {
            "label": name,
            "estimate": metrics[name],
            "lower": metrics[name],
            "upper": metrics[name],
        }
        for name in ("mAP", "mINP", "MRR")
    ]
    rows.extend(
        {
            "label": f"Rank-{item['k']}",
            "estimate": item["value"],
            "lower": item["value"],
            "upper": item["value"],
        }
        for item in metrics["rank_at_k"]
    )
    return FigureData.create(
        figure_id=figure_id,
        kind="result_forest",
        scope=scope,
        title="Protected evaluation integrity metrics",
        caption="Receipt-chain-verified metrics retained for private integrity inspection only.",
        limitations=(
            "This report explicitly is not valid for model selection or final scientific reporting.",
        ),
        source_bindings=(
            SourceBinding(
                "protected-evaluation-report",
                payload["schema_version"],
                content_sha256(payload),
            ),
        ),
        payload={
            "rows": rows,
            "x_label": "Metric value",
            "x_min": 0.0,
            "x_max": 1.0,
            "reference": None,
        },
    )


def _require_figure_kind(figure_id: str, expected: str) -> None:
    spec = FIGURE_BY_ID.get(figure_id)
    if spec is None or spec.kind != expected:
        raise FigureContractError(f"{figure_id} is not a registered {expected} figure")
