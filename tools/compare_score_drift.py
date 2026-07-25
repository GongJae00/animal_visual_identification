"""Admit label-blind score, rank, and frozen-boundary stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cvi.control_scoring import (
    ControlScoringInventory,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    EmbeddingCacheVerification,
)
from cvi.embedding_producer import EmbeddingProducerConfig
from cvi.embedding_production_runner import read_embedding_production_outer_bundle
from cvi.numerical_admission import NumericalAdmissionReceipt, NumericalDriftPolicy
from cvi.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_object,
    write_private_json_bundle,
)
from cvi.score_drift_admission import (
    FrozenScoreMarginBoundary,
    RetrievalScoreWorkload,
    ScoreDriftPolicy,
    ScoreDriftAdmissionPlan,
    compare_score_rank_threshold_drift,
)


def _payload(path: Path, name: str) -> dict[str, Any]:
    payload = read_strict_json_object(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    return payload


def _numerical_receipt(path: Path) -> NumericalAdmissionReceipt:
    return NumericalAdmissionReceipt.from_dict(
        read_content_hashed_json_bundle(
            path,
            schema_version="cvi.numerical_admission_bundle.v1",
            payload_field="receipt",
            sha256_field="receipt_sha256",
        )
    )


def _admission_plan(path: Path) -> ScoreDriftAdmissionPlan:
    return ScoreDriftAdmissionPlan.from_dict(
        read_content_hashed_json_bundle(
            path,
            schema_version="cvi.score_drift_admission_plan_bundle.v2",
            payload_field="plan",
            sha256_field="plan_sha256",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--reference-cache-directory", required=True, type=Path)
    parser.add_argument("--candidate-cache-directory", required=True, type=Path)
    parser.add_argument("--reference-cache-manifest", required=True, type=Path)
    parser.add_argument("--candidate-cache-manifest", required=True, type=Path)
    parser.add_argument("--reference-cache-verification", required=True, type=Path)
    parser.add_argument("--candidate-cache-verification", required=True, type=Path)
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
    parser.add_argument("--admission-plan", required=True, type=Path)
    parser.add_argument("--expected-precommitment-sha256", required=True)
    parser.add_argument("--expected-admission-plan-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    receipt = compare_score_rank_threshold_drift(
        workload=RetrievalScoreWorkload.from_dict(
            _payload(args.workload, "retrieval workload")
        ),
        inventory=ControlScoringInventory.from_dict(
            _payload(args.inventory, "scoring inventory")
        ),
        reference_root=args.reference_cache_directory,
        candidate_root=args.candidate_cache_directory,
        reference_manifest=EmbeddingCacheManifest.from_dict(
            _payload(args.reference_cache_manifest, "reference cache manifest")
        ),
        candidate_manifest=EmbeddingCacheManifest.from_dict(
            _payload(args.candidate_cache_manifest, "candidate cache manifest")
        ),
        reference_verification=EmbeddingCacheVerification.from_dict(
            _payload(
                args.reference_cache_verification,
                "reference cache verification",
            )
        ),
        candidate_verification=EmbeddingCacheVerification.from_dict(
            _payload(
                args.candidate_cache_verification,
                "candidate cache verification",
            )
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
        numerical_admission=_numerical_receipt(args.numerical_admission),
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
        admission_plan=_admission_plan(args.admission_plan),
        expected_precommitment_sha256=(
            args.expected_precommitment_sha256
        ),
        expected_admission_plan_sha256=(
            args.expected_admission_plan_sha256
        ),
    )
    output = {
        "schema_version": "cvi.score_drift_admission_bundle.v2",
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
    }
    write_private_json_bundle(((args.receipt, output),))
    print(
        json.dumps(
            {
                "status": "CREATED",
                "decision": receipt.decision.value,
                "promotion_decision": receipt.promotion_decision.value,
                "receipt_sha256": receipt.receipt_sha256,
                "requests": receipt.summary.requests,
                "queries": receipt.summary.queries,
                "rank_inversions": receipt.summary.rank_inversions,
                "threshold_decision_flips": (
                    receipt.summary.threshold_decision_flips
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
