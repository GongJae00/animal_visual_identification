"""Admit label-blind score, rank, and frozen-boundary stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.controls.control_scoring import (
    ControlScoringInventory,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    EmbeddingCacheVerification,
)
from systems.inference.embedding_producer import EmbeddingProducerConfig
from systems.workers.embedding_production_runner import read_embedding_production_outer_bundle
from evaluation.integrity.numerical_admission import NumericalAdmissionReceipt, NumericalDriftPolicy
from foundation.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_object,
    write_private_json_bundle,
)
from evaluation.integrity.score_drift_admission import (
    FrozenScoreMarginBoundary,
    RetrievalScoreWorkload,
    ScoreDriftPolicy,
    ScoreDriftAdmissionPlan,
    compare_score_rank_threshold_drift,
)


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
            read_strict_json_object(args.workload)
        ),
        inventory=ControlScoringInventory.from_dict(
            read_strict_json_object(args.inventory)
        ),
        reference_root=args.reference_cache_directory,
        candidate_root=args.candidate_cache_directory,
        reference_manifest=EmbeddingCacheManifest.from_dict(
            read_strict_json_object(args.reference_cache_manifest)
        ),
        candidate_manifest=EmbeddingCacheManifest.from_dict(
            read_strict_json_object(args.candidate_cache_manifest)
        ),
        reference_verification=EmbeddingCacheVerification.from_dict(
            read_strict_json_object(args.reference_cache_verification)
        ),
        candidate_verification=EmbeddingCacheVerification.from_dict(
            read_strict_json_object(args.candidate_cache_verification)
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
            read_strict_json_object(args.reference_producer_config)
        ),
        candidate_config=EmbeddingProducerConfig.from_dict(
            read_strict_json_object(args.candidate_producer_config)
        ),
        numerical_admission=_numerical_receipt(args.numerical_admission),
        numerical_policy=NumericalDriftPolicy.from_dict(
            read_strict_json_object(args.numerical_policy)
        ),
        boundary=FrozenScoreMarginBoundary.from_dict(
            read_strict_json_object(args.frozen_boundary)
        ),
        policy=ScoreDriftPolicy.from_dict(
            read_strict_json_object(args.score_drift_policy)
        ),
        cache_policy=EmbeddingCachePolicy.from_dict(
            read_strict_json_object(args.cache_policy)
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
