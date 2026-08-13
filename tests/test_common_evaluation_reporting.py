from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from evaluation.common_reporting import (
    CACHED_PROTOCOL_INPUT_SCHEMA_VERSION,
    ROTATIONAL_CAVEAT,
    CommonEvaluationError,
    ImmutableEvaluationReport,
    build_master_results_table,
    evaluate_cached_protocol,
)
from foundation.provenance import content_sha256


def _vector(primary: int, secondary: int | None = None) -> list[float]:
    vector = [0.0] * 128
    vector[primary] = 1.0
    if secondary is not None:
        vector[secondary] = 0.05
    return vector


def _payload(protocol_kind: str = "OFFICIAL_IDENTITY") -> dict:
    official = protocol_kind == "OFFICIAL_IDENTITY"
    labels = ("dog-a", "dog-a", "dog-b", "dog-b")
    instances = ("instance-a", "instance-a", "instance-b", "instance-b")
    samples = []
    for index, token in enumerate(("ga", "qa", "gb", "qb")):
        samples.append(
            {
                "sample_token": token,
                "embedding": _vector(0 if index < 2 else 1, 2 + index),
                "identity_token": labels[index] if official else None,
                "instance_token": instances[index],
            }
        )
    cohorts = []
    for index, (region, scope) in enumerate(
        zip(
            ("Full", "Face", "Nose"),
            ("isolated", "unified_balanced", "instance"),
            strict=True,
        )
    ):
        cohorts.append(
            {
                "cohort_id": f"cohort-{index}",
                "status": "AVAILABLE",
                "reason": None,
                "query_sample_tokens": ["qa", "qb"],
                "gallery_sample_tokens": ["ga", "gb"],
                "region": region,
                "view": "frontal",
                "quality": "usable",
                "gallery_scope": scope,
            }
        )
    observations = [
        {
            "category": category,
            "name": f"{category}-intervention",
            "context": None,
            "dimension": None,
            "metric_name": None,
            "values": [0.1, 0.2],
        }
        for category in ("mask", "box", "background")
    ]
    attribution = [
        {
            "category": "input_channel_intervention",
            "name": "red-zeroed",
            "context": None,
            "dimension": None,
            "metric_name": None,
            "values": [0.1, 0.2],
        },
        {
            "category": "classical_feature_group",
            "name": "texture",
            "context": None,
            "dimension": None,
            "metric_name": None,
            "values": [0.2, 0.3],
        },
        *[
            {
                "category": "feature_channel_sensitivity",
                "name": "channel-0",
                "context": context,
                "dimension": None,
                "metric_name": None,
                "values": [0.05, 0.1],
            }
            for context in ("within", "between", "background")
        ],
        {
            "category": "output_dimension_ablation",
            "name": "prefix-curve",
            "context": None,
            "dimension": 64,
            "metric_name": "Rank-1",
            "values": [1.0, 0.0],
        },
        {
            "category": "output_dimension_ablation",
            "name": "prefix-curve",
            "context": None,
            "dimension": 128,
            "metric_name": "Rank-1",
            "values": [1.0, 1.0],
        },
    ]
    return {
        "schema_version": CACHED_PROTOCOL_INPUT_SCHEMA_VERSION,
        "run_id": f"run-{protocol_kind.lower()}",
        "dataset": "synthetic-contract-fixture",
        "protocol_id": f"protocol-{protocol_kind.lower()}",
        "protocol_kind": protocol_kind,
        "cache_family_sha256": "a" * 64,
        "cache_manifest_sha256": "b" * 64,
        "embedding_dimension": 128,
        "requires_identity_disjoint_fitting": official,
        "fitting_identity_tokens": ["fit-only"] if official else [],
        "declared_strata": {
            "regions": ["Full", "Face", "Nose"],
            "views": ["frontal"],
            "qualities": ["usable"],
        },
        "samples": samples,
        "cohorts": cohorts,
        "sensitivity_observations": observations,
        "attribution_observations": attribution,
    }


def _rehash(envelope: dict) -> dict:
    envelope["report_sha256"] = content_sha256(envelope["report"])
    return envelope


def test_official_and_identity_free_protocols_use_distinct_retrieval_semantics() -> (
    None
):
    official = evaluate_cached_protocol(
        _payload(), bootstrap_resamples=20, bootstrap_seed=7
    ).report
    assert official["retrieval_modes"]["official_identity"]["status"] == "AVAILABLE"
    assert (
        official["retrieval_modes"]["instance_invariance"]["status"] == "NOT_APPLICABLE"
    )
    assert (
        official["retrieval_modes"]["official_identity"]["ranking_unit"]
        == "gallery_identity"
    )
    assert {
        row["rank_k"]
        for row in official["retrieval_modes"]["official_identity"]["metrics"]
    } == {
        1,
        3,
        5,
        None,
    }
    assert {
        row["region"]
        for row in official["retrieval_modes"]["official_identity"]["metrics"]
    } == {
        "Full",
        "Face",
        "Nose",
    }
    assert all(hook["status"] == "AVAILABLE" for hook in official["bootstrap_hooks"])

    identity_free = evaluate_cached_protocol(
        _payload("IDENTITY_FREE"), bootstrap_resamples=5
    ).report
    assert (
        identity_free["retrieval_modes"]["instance_invariance"]["status"] == "AVAILABLE"
    )
    identity_mode = identity_free["retrieval_modes"]["official_identity"]
    assert identity_mode["status"] == "NOT_APPLICABLE"
    assert "identity labels" in identity_mode["reason"]
    assert not identity_mode["metrics"]
    assert all(
        hook["status"] == "NOT_APPLICABLE" and hook["cluster_unit"] is None
        for hook in identity_free["bootstrap_hooks"]
    )


def test_common_coverage_health_sensitivity_and_attribution_axes_are_emitted() -> None:
    report = evaluate_cached_protocol(_payload(), bootstrap_resamples=5).report
    assert report["embedding_health"]["nominal_dimension"] == 128
    assert {
        row["stratum"]
        for row in report["availability_coverage"]
        if row["axis"] == "region"
    } == {
        "Full",
        "Face",
        "Nose",
    }
    assert {
        row["stratum"]
        for row in report["availability_coverage"]
        if row["axis"] == "gallery_scope"
    } == {
        "isolated",
        "unified_balanced",
        "instance",
    }
    assert {row["category"] for row in report["spatial_sensitivity"]} == {
        "mask",
        "box",
        "background",
    }
    attribution = report["attribution_summaries"]
    assert attribution["rotational_caveat"] == ROTATIONAL_CAVEAT
    assert {
        row["category"]
        for row in attribution["summaries"]
        if row["status"] == "AVAILABLE"
    } == {
        "input_channel_intervention",
        "classical_feature_group",
        "feature_channel_sensitivity",
        "output_dimension_ablation",
    }
    assert {
        row["context"]
        for row in attribution["summaries"]
        if row["category"] == "feature_channel_sensitivity"
    } == {"within", "between", "background"}


def test_rejects_mixed_protocol_and_ranking_units() -> None:
    envelope = evaluate_cached_protocol(_payload(), bootstrap_resamples=5).to_dict()
    envelope["report"]["retrieval_modes"]["official_identity"]["ranking_unit"] = (
        "gallery_instance"
    )
    with pytest.raises(CommonEvaluationError, match="mixed protocol/ranking units"):
        ImmutableEvaluationReport.from_dict(_rehash(envelope))


def test_rejects_query_gallery_and_partition_leakage() -> None:
    sample_leak = _payload()
    sample_leak["cohorts"][0]["gallery_sample_tokens"].append("qa")
    with pytest.raises(CommonEvaluationError, match="sample leakage"):
        evaluate_cached_protocol(sample_leak, bootstrap_resamples=5)

    identity_leak = _payload()
    identity_leak["fitting_identity_tokens"] = ["dog-a"]
    with pytest.raises(CommonEvaluationError, match="identity leakage"):
        evaluate_cached_protocol(identity_leak, bootstrap_resamples=5)


def test_rejects_missing_denominators_and_nonfinite_values() -> None:
    envelope = evaluate_cached_protocol(_payload(), bootstrap_resamples=5).to_dict()
    del envelope["report"]["retrieval_modes"]["official_identity"]["metrics"][0][
        "denominator"
    ]
    with pytest.raises(CommonEvaluationError, match="fields differ|denominator"):
        ImmutableEvaluationReport.from_dict(_rehash(envelope))

    nonfinite = _payload()
    nonfinite["samples"][0]["embedding"][0] = float("nan")
    with pytest.raises(CommonEvaluationError, match="finite"):
        evaluate_cached_protocol(nonfinite, bootstrap_resamples=5)


def test_report_tampering_is_detected() -> None:
    envelope = evaluate_cached_protocol(_payload(), bootstrap_resamples=5).to_dict()
    envelope["report"]["retrieval_modes"]["official_identity"]["metrics"][0][
        "value"
    ] = 0.0
    with pytest.raises(CommonEvaluationError, match="tampered"):
        ImmutableEvaluationReport.from_dict(envelope)


def test_master_table_is_deterministic_report_only_and_carries_hashes() -> None:
    first = evaluate_cached_protocol(_payload(), bootstrap_resamples=5)
    second_payload = _payload("IDENTITY_FREE")
    second_payload["cache_manifest_sha256"] = "c" * 64
    second = evaluate_cached_protocol(second_payload, bootstrap_resamples=5)

    table_a = build_master_results_table([second.to_dict(), first.to_dict()])
    table_b = build_master_results_table([first, second])
    assert table_a == table_b
    assert table_a["source_report_sha256s"] == sorted(
        [first.report_sha256, second.report_sha256]
    )
    assert {row["report_sha256"] for row in table_a["rows"]} == {
        first.report_sha256,
        second.report_sha256,
    }
    assert (
        content_sha256(
            {key: value for key, value in table_a.items() if key != "table_sha256"}
        )
        == table_a["table_sha256"]
    )

    raw_report = copy.deepcopy(first.report)
    with pytest.raises(CommonEvaluationError):
        build_master_results_table([raw_report])


def test_workflows_publish_complete_suite_then_report_only_master_table(
    tmp_path,
) -> None:
    inputs = []
    expected = []
    for protocol_kind in ("OFFICIAL_IDENTITY", "IDENTITY_FREE"):
        payload = _payload(protocol_kind)
        path = tmp_path / f"{protocol_kind.lower()}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        inputs.extend(("--protocol-input", str(path)))
        expected.extend(
            (
                "--expected-protocol",
                f"{payload['dataset']}:{payload['protocol_id']}",
            )
        )
    output_directory = tmp_path / "reports"
    output_directory.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "workflows.evaluate_cached_embedding_family",
            *inputs,
            *expected,
            "--output-directory",
            str(output_directory),
            "--bootstrap-resamples",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cached_embedding_family_evaluated" in completed.stdout
    report_paths = sorted(output_directory.glob("report-*.json"))
    assert len(report_paths) == 2

    table_path = tmp_path / "master-results.json"
    report_arguments = [
        argument for path in report_paths for argument in ("--report", str(path))
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "workflows.build_master_results_table",
            *report_arguments,
            "--output",
            str(table_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    table = json.loads(table_path.read_text(encoding="utf-8"))
    assert len(table["source_report_sha256s"]) == 2
    assert table["rows"]
