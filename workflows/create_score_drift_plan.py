"""Freeze the exact inputs for a score-drift admission run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.control_scoring import (
    ControlScoringInventory,
    EmbeddingCachePolicy,
)
from operations.embedding_producer import EmbeddingProducerConfig
from operations.embedding_production_runner import read_embedding_production_outer_bundle
from evaluation.numerical_admission import NumericalAdmissionReceipt, NumericalDriftPolicy
from foundation.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_object,
    write_private_json_bundle,
)
from evaluation.score_drift_admission import (
    FrozenScoreMarginBoundary,
    RetrievalScoreWorkload,
    ScoreDriftPrecommitment,
    ScoreDriftPolicy,
    build_score_drift_admission_plan,
)


def _payload(path: Path, name: str) -> dict[str, Any]:
    payload = read_strict_json_object(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    return payload


def _receipt(
    path: Path,
    *,
    schema_version: str,
) -> dict[str, Any]:
    return read_content_hashed_json_bundle(
        path,
        schema_version=schema_version,
        payload_field="receipt",
        sha256_field="receipt_sha256",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--reference-production", required=True, type=Path)
    parser.add_argument("--candidate-production", required=True, type=Path)
    parser.add_argument("--expected-reference-production-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-reference-completed-attempt-ledger-head-sha256",
        required=True,
    )
    parser.add_argument("--expected-candidate-production-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-candidate-completed-attempt-ledger-head-sha256",
        required=True,
    )
    parser.add_argument("--reference-producer-config", required=True, type=Path)
    parser.add_argument("--candidate-producer-config", required=True, type=Path)
    parser.add_argument("--numerical-admission", required=True, type=Path)
    parser.add_argument("--numerical-policy", required=True, type=Path)
    parser.add_argument("--frozen-boundary", required=True, type=Path)
    parser.add_argument("--score-drift-policy", required=True, type=Path)
    parser.add_argument("--cache-policy", required=True, type=Path)
    parser.add_argument("--precommitment", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()

    plan = build_score_drift_admission_plan(
        workload=RetrievalScoreWorkload.from_dict(
            _payload(args.workload, "retrieval workload")
        ),
        inventory=ControlScoringInventory.from_dict(
            _payload(args.inventory, "scoring inventory")
        ),
        reference_production=read_embedding_production_outer_bundle(
            args.reference_production,
            expected_receipt_sha256=(
                args.expected_reference_production_receipt_sha256
            ),
            expected_completed_attempt_ledger_head_sha256=(
                args.expected_reference_completed_attempt_ledger_head_sha256
            ),
        ).production_receipt,
        candidate_production=read_embedding_production_outer_bundle(
            args.candidate_production,
            expected_receipt_sha256=(
                args.expected_candidate_production_receipt_sha256
            ),
            expected_completed_attempt_ledger_head_sha256=(
                args.expected_candidate_completed_attempt_ledger_head_sha256
            ),
        ).production_receipt,
        reference_config=EmbeddingProducerConfig.from_dict(
            _payload(args.reference_producer_config, "reference producer config")
        ),
        candidate_config=EmbeddingProducerConfig.from_dict(
            _payload(args.candidate_producer_config, "candidate producer config")
        ),
        numerical_admission=NumericalAdmissionReceipt.from_dict(
            _receipt(
                args.numerical_admission,
                schema_version="cvi.numerical_admission_bundle.v1",
            )
        ),
        numerical_policy=NumericalDriftPolicy.from_dict(
            _payload(args.numerical_policy, "numerical drift policy")
        ),
        boundary=FrozenScoreMarginBoundary.from_dict(
            _payload(args.frozen_boundary, "frozen score-margin boundary")
        ),
        policy=ScoreDriftPolicy.from_dict(
            _payload(args.score_drift_policy, "score drift policy")
        ),
        cache_policy=EmbeddingCachePolicy.from_dict(
            _payload(args.cache_policy, "embedding cache policy")
        ),
        precommitment=ScoreDriftPrecommitment.from_dict(
            read_content_hashed_json_bundle(
                args.precommitment,
                schema_version="cvi.score_drift_precommitment_bundle.v1",
                payload_field="precommitment",
                sha256_field="precommitment_sha256",
            )
        ),
    )
    output = {
        "schema_version": "cvi.score_drift_admission_plan_bundle.v2",
        "plan_sha256": plan.plan_sha256,
        "plan": plan.to_dict(),
    }
    write_private_json_bundle(((args.plan, output),))
    print(
        json.dumps(
            {
                "status": "CREATED",
                "plan_sha256": plan.plan_sha256,
                "workload_sha256": plan.workload_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
