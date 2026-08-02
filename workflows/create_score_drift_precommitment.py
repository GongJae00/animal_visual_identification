"""Freeze score-drift inputs before candidate embedding production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.control_scoring import ControlScoringInventory, EmbeddingCachePolicy
from operations.embedding_producer import EmbeddingProducerConfig
from operations.embedding_production_runner import read_embedding_production_outer_bundle
from evaluation.numerical_admission import NumericalDriftPolicy
from foundation.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)
from evaluation.score_drift_admission import (
    FrozenScoreMarginBoundary,
    RetrievalScoreWorkload,
    ScoreDriftPolicy,
    build_score_drift_precommitment,
)


def _payload(path: Path, name: str) -> dict[str, Any]:
    payload = read_strict_json_object(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--reference-production", required=True, type=Path)
    parser.add_argument("--expected-reference-production-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-reference-completed-attempt-ledger-head-sha256",
        required=True,
    )
    parser.add_argument("--reference-producer-config", required=True, type=Path)
    parser.add_argument("--candidate-producer-config", required=True, type=Path)
    parser.add_argument("--numerical-policy", required=True, type=Path)
    parser.add_argument("--frozen-boundary", required=True, type=Path)
    parser.add_argument("--score-drift-policy", required=True, type=Path)
    parser.add_argument("--cache-policy", required=True, type=Path)
    parser.add_argument("--prior-attempt-ledger-sha256", required=True)
    parser.add_argument("--candidate-attempt-token", required=True)
    parser.add_argument("--precommitment-sequence", required=True, type=int)
    parser.add_argument("--precommitment", required=True, type=Path)
    args = parser.parse_args()

    reference_receipt = read_embedding_production_outer_bundle(
        args.reference_production,
        expected_receipt_sha256=(
            args.expected_reference_production_receipt_sha256
        ),
        expected_completed_attempt_ledger_head_sha256=(
            args.expected_reference_completed_attempt_ledger_head_sha256
        ),
    ).production_receipt
    precommitment = build_score_drift_precommitment(
        workload=RetrievalScoreWorkload.from_dict(
            _payload(args.workload, "retrieval workload")
        ),
        inventory=ControlScoringInventory.from_dict(
            _payload(args.inventory, "scoring inventory")
        ),
        reference_production=reference_receipt,
        reference_config=EmbeddingProducerConfig.from_dict(
            _payload(args.reference_producer_config, "reference producer config")
        ),
        candidate_config=EmbeddingProducerConfig.from_dict(
            _payload(args.candidate_producer_config, "candidate producer config")
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
        prior_attempt_ledger_sha256=args.prior_attempt_ledger_sha256,
        candidate_attempt_token=args.candidate_attempt_token,
        precommitment_sequence=args.precommitment_sequence,
    )
    output = {
        "schema_version": "cvi.score_drift_precommitment_bundle.v1",
        "precommitment_sha256": precommitment.precommitment_sha256,
        "precommitment": precommitment.to_dict(),
    }
    write_private_json_bundle(((args.precommitment, output),))
    print(
        json.dumps(
            {
                "status": "CREATED",
                "precommitment_sha256": precommitment.precommitment_sha256,
                "candidate_attempt_token": precommitment.candidate_attempt_token,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
