from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from archive.full128.evaluation.full128_successors import (
    Full128SuccessorEvaluationError,
    build_dev_selection_receipt,
    build_multiseed_terminal_successor_decision,
    validate_public_successor_evaluation_report,
)
from shared.foundation.provenance import content_sha256
from archive.full128.commands.decide_full128_successor_multiseed import main as decision_workflow

def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()

def _public_report(
    *, recorded_seed: int, b3_rank1: float, b5_rank1: float, lower_bound: float
) -> dict[str, Any]:
    def aggregate(successor_id: str, scope: str, rank1: float) -> dict[str, Any]:
        payload = {
            "successor_id": successor_id,
            "scope": scope,
            "status": "AVAILABLE",
            "reason": None,
            "query_count": 77,
            "identity_count": 77,
            "metrics": {
                "Rank-1": rank1,
                "Rank-5": rank1,
                "Rank-10": rank1,
                "MRR": rank1,
            },
        }
        return {**payload, "result_sha256": content_sha256(payload)}

    candidates = []
    dev_results = []
    for successor_id, rank1 in (("B3", b3_rank1), ("B5-SPATIAL", b5_rank1)):
        aggregates = [
            aggregate(successor_id, scope, rank1)
            for scope in ("DEV", "CAL", "EXPOSED_DIAGNOSTIC")
        ]
        dev_results.append(aggregates[0])
        candidates.append(
            {
                "successor_id": successor_id,
                "cache_descriptor_sha256": _sha(
                    f"cache:{recorded_seed}:{successor_id}"
                ),
                "scope_aggregates": aggregates,
                "gallery_bindings": [],
            }
        )
    intervals = []
    for metric in ("Rank-1", "Rank-5", "Rank-10", "MRR"):
        estimate = b5_rank1 - b3_rank1 if metric == "Rank-1" else 0.0
        payload = {
            "schema_version": "archive.full128.successor_paired_bootstrap.v1",
            "metric": metric,
            "estimate": estimate,
            "lower_bound": lower_bound if metric == "Rank-1" else 0.0,
            "upper_bound": estimate + 0.05 if metric == "Rank-1" else 0.0,
            "confidence_level": 0.95,
            "cluster_unit": "registered_identity_id",
            "cluster_count": 77,
            "paired_query_count": 77,
            "resamples": 10_000,
            "seed": recorded_seed,
        }
        intervals.append({**payload, "bootstrap_sha256": content_sha256(payload)})
    payload = {
        "schema_version": "archive.full128.successor_public_evaluation.v1",
        "visibility": "PUBLIC_AGGREGATE",
        "source_private_report_sha256": _sha(f"private:{recorded_seed}"),
        "evaluation_panel_sha256": _sha("terminal-panel"),
        "candidates": candidates,
        "dev_selection_receipt": build_dev_selection_receipt(dev_results),
        "paired_identity_cluster_bootstrap": [
            {
                "scope": "DEV",
                "left_successor_id": "B5-SPATIAL",
                "right_successor_id": "B3",
                "intervals": intervals,
            }
        ],
        "scope_interpretation": {
            "DEV": "MODEL_SELECTION_ONLY",
            "CAL": "CALIBRATION_REPORTING;NOT_SELECTION",
            "EXPOSED_DIAGNOSTIC": "RETROSPECTIVE_EXPOSED;NOT_FINAL_EVALUATION",
        },
        "contains_embeddings": False,
        "contains_sample_or_identity_tokens": False,
        "contains_ranked_qkv_traces": False,
        "limitations": [
            "DEV is used for successor selection; CAL and exposed diagnostics are not selection inputs.",
            "EXPOSED_DIAGNOSTIC is retrospective and is not an independent final evaluation.",
            "The report evaluates exact closed-set cosine retrieval only.",
        ],
    }
    return {**payload, "public_report_sha256": content_sha256(payload)}

def _sources(lower_bounds: tuple[float, float, float]) -> list[dict[str, Any]]:
    sources = []
    for seed_index, (recorded_seed, lower_bound) in enumerate(
        zip((20260811, 1, 2), lower_bounds, strict=True)
    ):
        report = _public_report(
            recorded_seed=recorded_seed,
            b3_rank1=0.76 + seed_index * 0.01,
            b5_rank1=0.78 + seed_index * 0.01,
            lower_bound=lower_bound,
        )
        sources.append(
            {
                "seed_index": seed_index,
                "report": report,
                "raw_sha256": _sha(f"raw-report:{seed_index}"),
                "canonical_payload_sha256": content_sha256(report),
                "byte_size": 1000 + seed_index,
            }
        )
    return sources

def _refresh_source_report(source: dict[str, Any]) -> None:
    report = source["report"]
    report["public_report_sha256"] = content_sha256(
        {key: value for key, value in report.items() if key != "public_report_sha256"}
    )
    source["canonical_payload_sha256"] = content_sha256(report)

def test_terminal_decision_is_content_bound_and_fails_nonpositive_gate() -> None:
    passing = build_multiseed_terminal_successor_decision(
        _sources((0.001, 0.002, 0.003))
    )
    assert passing["promotion_gate"]["decision"] == "GO"
    assert passing["seed_reports"][0]["report_recorded_seed"] == 20260811
    assert passing["seed_reports"][0]["dev_selection"]["role"].endswith(
        "NOT_SCIENTIFIC_PROMOTION"
    )
    assert passing["across_seed_summaries"]["selected_dev_rank1"]["seed_count"] == 3
    assert (
        passing["evidence_boundaries"]["calibration"]["selection_input_used"] is False
    )
    assert (
        passing["evidence_boundaries"]["independent_final"]["availability"]
        == "UNAVAILABLE"
    )
    assert passing["evidence_boundaries"]["run_seed_binding"] == {
        "available_evidence": "REPORT_RECORDED_BOOTSTRAP_SEED",
        "limitation": (
            "NO_SEPARATE_TRAINING_RUN_SEED_FIELD_IS_AVAILABLE_IN_THE_PUBLIC_REPORT"
        ),
    }

    failed = build_multiseed_terminal_successor_decision(_sources((-0.001, 0.0, 0.003)))
    assert failed["promotion_gate"]["decision"] == "NO_GO"
    assert failed["promotion_gate"]["failed_seed_indexes"] == [0, 1]
    assert failed["promotion_gate"]["passing_seed_count"] == 1
    serialized = json.dumps(failed, sort_keys=True)
    assert "sample_token" not in serialized
    assert '"embedding":' not in serialized

def test_terminal_decision_rejects_tamper_private_and_cal_selection() -> None:
    sources = _sources((0.001, 0.002, 0.003))
    tampered = deepcopy(sources)
    tampered[0]["report"]["candidates"][0]["scope_aggregates"][0]["metrics"][
        "Rank-1"
    ] = 0.1
    with pytest.raises(Full128SuccessorEvaluationError, match="contract differs"):
        build_multiseed_terminal_successor_decision(tampered)

    private = deepcopy(sources[0]["report"])
    private["sample_token"] = _sha("private-sample")
    with pytest.raises(Full128SuccessorEvaluationError, match="fields differ"):
        validate_public_successor_evaluation_report(private)

    cal_selected = deepcopy(sources)
    receipt = cal_selected[0]["report"]["dev_selection_receipt"]
    receipt["calibration_scope_used"] = True
    receipt["receipt_sha256"] = content_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    report = cal_selected[0]["report"]
    report["public_report_sha256"] = content_sha256(
        {key: value for key, value in report.items() if key != "public_report_sha256"}
    )
    cal_selected[0]["canonical_payload_sha256"] = content_sha256(report)
    with pytest.raises(Full128SuccessorEvaluationError, match="selection contract"):
        build_multiseed_terminal_successor_decision(cal_selected)

    duplicate = _sources((0.001, 0.002, 0.003))
    duplicate[2]["seed_index"] = 1
    with pytest.raises(Full128SuccessorEvaluationError, match="exactly 0, 1, and 2"):
        build_multiseed_terminal_successor_decision(duplicate)

def test_terminal_decision_rejects_repeated_semantic_seed_evidence() -> None:
    repeated_seed = _sources((0.001, 0.002, 0.003))
    repeated_seed[1]["report"]["paired_identity_cluster_bootstrap"][0]["intervals"][0][
        "seed"
    ] = repeated_seed[0]["report"]["paired_identity_cluster_bootstrap"][0]["intervals"][
        0
    ]["seed"]
    interval = repeated_seed[1]["report"]["paired_identity_cluster_bootstrap"][0][
        "intervals"
    ][0]
    interval["bootstrap_sha256"] = content_sha256(
        {key: value for key, value in interval.items() if key != "bootstrap_sha256"}
    )
    _refresh_source_report(repeated_seed[1])
    with pytest.raises(Full128SuccessorEvaluationError, match="bootstrap seeds"):
        build_multiseed_terminal_successor_decision(repeated_seed)

    repeated_cache = _sources((0.001, 0.002, 0.003))
    repeated_cache[1]["report"]["candidates"][1]["cache_descriptor_sha256"] = (
        repeated_cache[0]["report"]["candidates"][1]["cache_descriptor_sha256"]
    )
    _refresh_source_report(repeated_cache[1])
    with pytest.raises(Full128SuccessorEvaluationError, match="cache descriptors"):
        build_multiseed_terminal_successor_decision(repeated_cache)

def test_terminal_decision_rejects_byte_distinct_serializations_of_one_report(
    tmp_path: Path,
) -> None:
    report = _sources((0.001, 0.002, 0.003))[0]["report"]
    serializations = (
        json.dumps(report, separators=(",", ":")),
        json.dumps(report, indent=2),
        json.dumps(report, indent=4, sort_keys=True),
    )
    assert len({value.encode("utf-8") for value in serializations}) == 3

    arguments = []
    for seed_index, serialized in enumerate(serializations):
        path = tmp_path / f"serialization-{seed_index}.json"
        path.write_text(serialized, encoding="utf-8")
        arguments.extend(("--seed-report", f"{seed_index}={path}"))
    arguments.extend(("--output-directory", str(tmp_path / "decision")))

    with pytest.raises(
        Full128SuccessorEvaluationError, match="canonical source reports"
    ):
        decision_workflow(arguments)

def test_terminal_decision_workflow_publishes_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = []
    for source in _sources((-0.001, 0.0, 0.003)):
        path = tmp_path / f"seed-{source['seed_index']}.json"
        path.write_text(json.dumps(source["report"]), encoding="utf-8")
        arguments.extend(("--seed-report", f"{source['seed_index']}={path}"))
    output = tmp_path / "decision"
    arguments.extend(("--output-directory", str(output)))

    assert decision_workflow(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    artifact = json.loads((output / "terminal-decision.json").read_text())
    assert summary["decision"] == "NO_GO"
    assert summary["decision_sha256"] == artifact["decision_sha256"]
    with pytest.raises(FileExistsError):
        decision_workflow(arguments)
