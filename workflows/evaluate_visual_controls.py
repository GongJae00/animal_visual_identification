"""Join sealed labels and evaluate pair-matched visual-control scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.control_evaluation import (
    ControlEvaluationPolicy,
    control_evaluation_bindings_from_payload,
    evaluate_sealed_control_scores,
)
from evaluation.control_scoring import (
    ControlBlindScoreReceipt,
    EmbeddingCacheManifest,
)
from evaluation import (
    ClusterBootstrapConfig,
    FrozenVerificationThreshold,
)
from evaluation.pairing import PairingPolicy, pair_construction_from_bundle_payloads
from foundation.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-scoring-requests", required=True, type=Path)
    parser.add_argument("--pair-artifact-bindings", required=True, type=Path)
    parser.add_argument("--pair-ground-truth", required=True, type=Path)
    parser.add_argument("--pair-summary", required=True, type=Path)
    parser.add_argument("--pairing-policy", required=True, type=Path)
    parser.add_argument("--control-evaluation-bindings", required=True, type=Path)
    parser.add_argument("--blind-score-receipt", required=True, type=Path)
    parser.add_argument("--embedding-cache-manifest", required=True, type=Path)
    parser.add_argument("--frozen-threshold", required=True, type=Path)
    parser.add_argument("--bootstrap-config", required=True, type=Path)
    parser.add_argument("--evaluation-policy", required=True, type=Path)
    parser.add_argument("--evaluation-output", required=True, type=Path)
    args = parser.parse_args()

    construction = pair_construction_from_bundle_payloads(
        read_strict_json_object(args.pair_scoring_requests),
        read_strict_json_object(args.pair_artifact_bindings),
        read_strict_json_object(args.pair_ground_truth),
        read_strict_json_object(args.pair_summary),
    )
    plan_sha256, pair_set_sha256, bindings, summaries = (
        control_evaluation_bindings_from_payload(
            read_strict_json_object(args.control_evaluation_bindings)
        )
    )
    result = evaluate_sealed_control_scores(
        construction=construction,
        pairing_policy=PairingPolicy.from_dict(
            read_strict_json_object(args.pairing_policy)
        ),
        plan_sha256=plan_sha256,
        pair_set_sha256=pair_set_sha256,
        bindings=bindings,
        panel_summaries=summaries,
        blind_scores=ControlBlindScoreReceipt.from_dict(
            read_strict_json_object(args.blind_score_receipt)
        ),
        embedding_cache_manifest=EmbeddingCacheManifest.from_dict(
            read_strict_json_object(args.embedding_cache_manifest)
        ),
        threshold=FrozenVerificationThreshold.from_dict(
            read_strict_json_object(args.frozen_threshold)
        ),
        bootstrap=ClusterBootstrapConfig.from_dict(
            read_strict_json_object(args.bootstrap_config)
        ),
        policy=ControlEvaluationPolicy.from_dict(
            read_strict_json_object(args.evaluation_policy)
        ),
    )
    write_private_json_bundle(
        ((args.evaluation_output, result.to_dict()),)
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
                "plan_sha256": result.plan_sha256,
                "pair_set_sha256": result.pair_set_sha256,
                "evaluation_receipt_sha256": result.receipt_sha256,
                "panels": len(result.panels),
                "bindings_joined": result.cost.bindings_joined,
                "auc_sort_items": result.cost.auc_sort_items,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
