from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.search_metrics.metrics import identity_clustered_bootstrap_ci
from archive.appearance_face_nose.experiments.sibetan_diagnostics import (
    DIAGNOSTIC_BUNDLE_SCHEMA,
    build_sibetan_diagnostic,
    bundle_sibetan_diagnostic,
)
from archive.appearance_face_nose.experiments.sibetan_multievidence import BRANCHES, METHOD_BRANCHES
from shared.foundation.provenance import content_sha256
from archive.appearance_face_nose.commands.audit_sibetan_diagnostics import main

_SHA = "a" * 64
_METRICS = ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
_WEIGHTS = {
    "A0_plus_F0": {BRANCHES[0]: 0.75, BRANCHES[1]: 0.25},
    "A0_plus_N3": {BRANCHES[0]: 0.75, BRANCHES[2]: 0.25},
    "F0_plus_N3": {BRANCHES[1]: 0.75, BRANCHES[2]: 0.25},
    "A0_plus_F0_plus_N3": {
        BRANCHES[0]: 0.5,
        BRANCHES[1]: 0.25,
        BRANCHES[2]: 0.25,
    },
}

def _query_row(token: str, identity: str, rank: int) -> dict[str, object]:
    return {
        "sample_token": token,
        "identity_token": identity,
        "bootstrap_cluster_id": identity,
        "rank": rank,
        "Rank-1": float(rank == 1),
        "Rank-5": float(rank <= 5),
        "Rank-10": float(rank <= 10),
        "reciprocal_rank": 1.0 / rank,
    }

def _metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    return {
        "query_count": len(rows),
        **{
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in _METRICS
        },
    }

def _method(
    method: str,
    rows: list[dict[str, object]],
    appearance_rows: list[dict[str, object]],
) -> dict[str, object]:
    branches = METHOD_BRANCHES[method]
    weights = {branches[0]: 1.0} if len(branches) == 1 else _WEIGHTS[method]
    metrics = _metrics(rows) if rows else None
    baseline_rows = {
        row["sample_token"]: row for row in appearance_rows
    }
    paired_baseline = None
    paired_cis = None
    if method != BRANCHES[0] and rows:
        selected_baselines = [baseline_rows[row["sample_token"]] for row in rows]
        paired_baseline = _metrics(selected_baselines)
        paired_cis = {
            metric: identity_clustered_bootstrap_ci(
                [
                    {
                        "bootstrap_cluster_id": row["identity_token"],
                        "delta": float(row[metric]) - float(baseline[metric]),
                    }
                    for row, baseline in zip(rows, selected_baselines, strict=True)
                ],
                metric="delta",
                resamples=100,
                seed=0,
            )
            for metric in _METRICS
        }
    return {
        "branches": list(branches),
        "weights": weights,
        "positive_weight_branches": [branch for branch in branches if weights[branch] > 0],
        "equivalent_method": None,
        "fixed_query_count": 3,
        "evaluated_query_count": len(rows),
        "abstained_query_count": 3 - len(rows),
        "query_coverage": len(rows) / 3,
        "abstention_reasons": {} if rows else {"NO_USABLE_OPTIONAL_PAIR": 3},
        "candidate_active_branch_patterns": {},
        "metrics": metrics,
        "query_rows": rows,
        "identity_clustered_bootstrap_cis": (
            {
                metric: identity_clustered_bootstrap_ci(
                    rows, metric=metric, resamples=100, seed=0
                )
                for metric in _METRICS
            }
            if rows
            else None
        ),
        "paired_appearance_baseline_metrics": paired_baseline,
        "paired_delta_bootstrap_cis": paired_cis,
    }

def _panel(shot: int, *, unavailable_method: str | None = None) -> dict[str, object]:
    appearance = [
        _query_row("q-a-1", "identity-a", 1),
        _query_row("q-a-2", "identity-a", 2),
        _query_row("q-b-1", "identity-b", 1),
    ]
    changed = [
        _query_row("q-a-1", "identity-a", 2),
        _query_row("q-a-2", "identity-a", 2),
        _query_row("q-b-1", "identity-b", 1),
    ]
    methods = {}
    for method in METHOD_BRANCHES:
        rows = appearance
        if method == "A0_plus_F0":
            rows = changed
        if method == unavailable_method:
            rows = []
        methods[method] = _method(method, rows, appearance)
    availability = {
        branch: {
            "gallery_effective_k": {"identity-a": shot, "identity-b": shot},
            "gallery_effective_k_histogram": {str(shot): 2},
            "available_gallery_identity_count": 2,
            "available_query_count": 3,
            "available_pair_count": 6,
            "total_pair_count": 6,
            "pair_coverage": 1.0,
            "gallery_mean_reliability": 0.8,
            "query_mean_reliability": 0.7,
        }
        for branch in BRANCHES
    }
    return {
        "protocol": "SIBETAN_CROSS_SEQUENCE",
        "episode": "fixed",
        "shot": shot,
        "population_sha256": _SHA,
        "gallery_order_sha256": _SHA,
        "query_order_sha256": _SHA,
        "external_appearance_control": {
            "preprocessing": "BILINEAR_STRETCH_224X224",
            "purpose": "REPRODUCE_ESTABLISHED_PROTECTED_APPEARANCE_BASELINE_ONLY",
            "metrics": _metrics(appearance),
        },
        "fixed_population": {
            "query_count": 3,
            "gallery_template_count": shot * 2,
            "gallery_identity_count": 2,
            "nominal_k": shot,
        },
        "branch_availability": availability,
        "methods": methods,
    }

def _source_bundle(*, unavailable_method: str | None = None) -> dict[str, object]:
    report = {
        "schema_version": "archive.sibetan.sibetan_multievidence_evaluation.v2",
        "status": "PASS_EXPOSED_SIBETAN_FROZEN_TRANSFER_DIAGNOSTIC",
        "interpretation": "EXPOSED_SIBETAN_CROSS_SEQUENCE_FROZEN_TRANSFER_DIAGNOSTIC_NOT_FINAL_OR_BIOMETRIC_VALIDATION",
        "protocol": {
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
            "bootstrap": {
                "cluster_unit": "protected_identity_token",
                "resamples": 100,
                "base_seed": 0,
                "confidence_level": 0.95,
            },
        },
        "transfer_weights": copy.deepcopy(_WEIGHTS),
        "evidence_state_counts": {
            "face": {"AVAILABLE": 6},
            "nose": {"AVAILABLE": 6},
        },
        "panels": [
            _panel(shot, unavailable_method=unavailable_method) for shot in (1, 3, 5)
        ],
        "input_bindings": {
            "split_receipt_sha256": _SHA,
            "source_bundle_sha256": _SHA,
            "assignment_sha256": _SHA,
            "evidence_file_sha256": _SHA,
            "evidence_manifest_sha256": _SHA,
            "yt_policy_file_sha256": _SHA,
            "yt_policy_report_sha256": _SHA,
            "frozen_dinov2_sha256": _SHA,
            "nose_runtime_manifest_sha256": _SHA,
            "nose_onnx_sha256": _SHA,
            "publisher_archives": [],
            "code_sha256s": {"producer.py": _SHA},
        },
    }
    return {
        "schema_version": "archive.sibetan.sibetan_multievidence_evaluation_bundle.v2",
        "report_sha256": content_sha256(report),
        "report": report,
    }

def _diagnostic(source: dict[str, object]) -> dict[str, object]:
    return build_sibetan_diagnostic(
        source,
        source_file_sha256="b" * 64,
        source_canonical_sha256="c" * 64,
        code_sha256s={"diagnostic.py": "d" * 64},
    )

def test_extracts_grounded_confound_summaries_deterministically() -> None:
    source = _source_bundle()

    first = _diagnostic(source)
    second = _diagnostic(copy.deepcopy(source))

    assert first == second
    assert bundle_sibetan_diagnostic(first) == bundle_sibetan_diagnostic(second)
    panel = first["panels"][0]
    rank1 = panel["methods"][BRANCHES[0]]["rank1_aggregation"]
    assert rank1["query_weighted_rank1"] == pytest.approx(2 / 3)
    assert rank1["equal_identity_rank1"] == pytest.approx(0.75)
    assert rank1["queries_per_identity_min"] == 1
    assert rank1["queries_per_identity_max"] == 2
    paired = panel["methods"]["A0_plus_F0"]["paired_appearance_delta"]
    assert paired["method_minus_appearance"]["Rank-1"] == pytest.approx(-1 / 3)
    assert panel["branch_availability"][BRANCHES[1]]["pair_coverage"] == 1.0
    assert panel["same_panel_preprocessing_control"]["causal_interpretation"] == "NOT_SUPPORTED"
    assert first["transferred_fusion_policy"]["weights"] == _WEIGHTS

def test_records_legitimate_unavailable_query_summaries() -> None:
    report = _diagnostic(_source_bundle(unavailable_method=BRANCHES[2]))

    method = report["panels"][0]["methods"][BRANCHES[2]]
    assert method["rank1_aggregation"]["status"] == "UNAVAILABLE"
    assert method["paired_appearance_delta"]["reason"] == "NO_EVALUATED_QUERY_OUTCOMES"
    fields = {item["field"] for item in report["unavailable_fields"]}
    assert f"panels.K1.methods.{BRANCHES[2]}.rank1_aggregation" in fields

@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: bundle.update(schema_version="archive.sibetan.sibetan_multievidence_evaluation_bundle.v1"),
        lambda bundle: bundle["report"]["panels"][0]["methods"][BRANCHES[0]]["query_rows"][0].update(rank=2),
        lambda bundle: bundle["report"]["panels"][0]["branch_availability"][BRANCHES[1]].update(pair_coverage=0.5),
        lambda bundle: bundle["report"]["panels"][0]["methods"]["A0_plus_F0"][
            "paired_delta_bootstrap_cis"
        ]["Rank-1"].update(lower_bound=-0.99),
        lambda bundle: bundle["report"]["panels"][0]["methods"][BRANCHES[0]][
            "identity_clustered_bootstrap_cis"
        ]["Rank-1"].update(lower_bound=-0.99),
    ],
)
def test_rejects_incompatible_or_internally_inconsistent_reports(mutation) -> None:
    source = _source_bundle()
    mutation(source)
    source["report_sha256"] = content_sha256(source["report"])

    with pytest.raises(ValueError):
        _diagnostic(source)

def test_rejects_tampered_embedded_report_digest() -> None:
    source = _source_bundle()
    source["report"]["status"] = "ALTERED"

    with pytest.raises(ValueError, match="digest differs"):
        _diagnostic(source)

def test_cli_writes_content_bound_sorted_json_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "diagnostic.json"
    source_path.write_text(json.dumps(_source_bundle()), encoding="utf-8")

    main(["--input", str(source_path), "--output", str(output_path)])

    output_text = output_path.read_text(encoding="utf-8")
    output = json.loads(output_text)
    assert output["schema_version"] == DIAGNOSTIC_BUNDLE_SCHEMA
    assert output["report_sha256"] == content_sha256(output["report"])
    assert output_text == json.dumps(
        output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    assert json.loads(capsys.readouterr().out)["status"] == "PASS_DESCRIPTIVE_SIBETAN_DIAGNOSTIC"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--input", str(source_path), "--output", str(output_path)])
