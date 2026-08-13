from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from identity.splits.tracklet_split import EvaluationStage
from evaluation import (
    ClusterBootstrapConfig,
    ClusterUnit,
    FrozenVerificationThreshold,
    VerificationDirection,
)
from evaluation.controls.control_evaluation import (
    ControlEvaluationPolicy,
    control_evaluation_bindings_from_payload,
    evaluate_sealed_control_scores,
)
from evaluation.controls.control_scoring import (
    ArtifactCacheBinding,
    ControlBlindScore,
    ControlBlindScoreReceipt,
    ControlScoreCost,
    EmbeddingCacheEntry,
    EmbeddingCacheManifest,
    embedding_cache_key,
)
from evaluation.controls.policy import (
    ControlEvaluationBinding,
    ControlPanelSummary,
    ControlStratumCount,
    VisualControlKind,
)
from evaluation.controls.pairing import (
    NegativeQuota,
    PairArtifactBinding,
    PairConstructionResult,
    PairGroundTruth,
    PairingPolicy,
    PairScoringRequest,
    PairStratum,
)
from workflows.evaluate_visual_controls import main

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _pairing_policy() -> PairingPolicy:
    return PairingPolicy(
        name="control-rgb-to-rgb",
        stage=EvaluationStage.TEST,
        direction=VerificationDirection.RGB_TO_RGB,
        positive_pairs_per_query=1,
        negative_quotas=(NegativeQuota(PairStratum.RANDOM, 1),),
        maximum_queries_per_dog=2,
        maximum_pairs_per_query=2,
        maximum_candidate_scans_per_stratum=10,
        minimum_breed_confidence=0.8,
        seed=7,
    )


def _construction() -> PairConstructionResult:
    truths = (
        PairGroundTruth(
            "pair-1",
            "dog-1",
            "dog-1",
            "query-session-1",
            "reference-session-1",
            PairStratum.POSITIVE,
        ),
        PairGroundTruth(
            "pair-2",
            "dog-2",
            "dog-2",
            "query-session-2",
            "reference-session-2",
            PairStratum.POSITIVE,
        ),
        PairGroundTruth(
            "pair-3",
            "dog-1",
            "dog-3",
            "query-session-1",
            "reference-session-3",
            PairStratum.RANDOM,
        ),
        PairGroundTruth(
            "pair-4",
            "dog-2",
            "dog-4",
            "query-session-2",
            "reference-session-4",
            PairStratum.RANDOM,
        ),
    )
    requests = tuple(
        PairScoringRequest(
            truth.pair_id,
            f"query-{truth.pair_id}",
            f"reference-{truth.pair_id}",
        )
        for truth in truths
    )
    return PairConstructionResult(
        split_manifest_sha256=HASH_A,
        pairing_policy_sha256=_pairing_policy().policy_sha256,
        attributes_sha256=HASH_C,
        eligible_query_count=2,
        selected_query_count=2,
        dropped_query_count=0,
        scoring_requests=requests,
        artifact_bindings=tuple(
            PairArtifactBinding(token, f"sample-{token}")
            for request in requests
            for token in (
                request.query_artifact_token,
                request.reference_artifact_token,
            )
        ),
        ground_truth=truths,
        quotas=(),
    )


def _cache_manifest() -> EmbeddingCacheManifest:
    key = embedding_cache_key(
        artifact_content_sha256=HASH_A,
        model_sha256=HASH_A,
        inference_config_sha256=HASH_B,
        dependency_lock_sha256=HASH_C,
        code_revision="fixture",
        precision="fp32",
        vector_dimension=2,
    )
    return EmbeddingCacheManifest(
        scoring_inventory_sha256=HASH_B,
        model_sha256=HASH_A,
        inference_config_sha256=HASH_B,
        dependency_lock_sha256=HASH_C,
        code_revision="fixture",
        precision="fp32",
        vector_dimension=2,
        normalization_tolerance=1e-6,
        bindings=(ArtifactCacheBinding("artifact", HASH_A, key),),
        entries=(
            EmbeddingCacheEntry(
                key,
                f"{key}.f32le",
                HASH_D,
                8,
            ),
        ),
    )


def _bindings() -> tuple[ControlEvaluationBinding, ...]:
    return tuple(
        ControlEvaluationBinding(
            request_id=f"{kind.value}-{pair_id}",
            panel_id="background",
            control_kind=kind,
            base_pair_id=pair_id,
        )
        for kind in (
            VisualControlKind.ORIGINAL,
            VisualControlKind.BACKGROUND_ONLY,
        )
        for pair_id in ("pair-1", "pair-2", "pair-3", "pair-4")
    )


def _summary() -> ControlPanelSummary:
    return ControlPanelSummary(
        panel_id="background",
        required_mask_roles=(),
        total_pairs=4,
        eligible_pairs=4,
        selected_pairs=4,
        ineligible_pairs=0,
        cap_applied=False,
        minimum_met=True,
        exclusions=(),
        strata=(
            ControlStratumCount(PairStratum.POSITIVE, 2, 2),
            ControlStratumCount(PairStratum.RANDOM, 2, 2),
        ),
    )


def _blind_scores(
    cache: EmbeddingCacheManifest,
) -> ControlBlindScoreReceipt:
    values = {
        "ORIGINAL-pair-1": 0.9,
        "ORIGINAL-pair-2": 0.8,
        "ORIGINAL-pair-3": 0.1,
        "ORIGINAL-pair-4": 0.2,
        "BACKGROUND_ONLY-pair-1": 0.7,
        "BACKGROUND_ONLY-pair-2": 0.6,
        "BACKGROUND_ONLY-pair-3": 0.6,
        "BACKGROUND_ONLY-pair-4": 0.5,
    }
    return ControlBlindScoreReceipt(
        plan_sha256=HASH_A,
        scoring_requests_sha256=HASH_B,
        scoring_inventory_sha256=HASH_C,
        embedding_cache_manifest_sha256=cache.manifest_sha256,
        embedding_cache_verification_sha256=HASH_E,
        gallery_sha256=HASH_D,
        score_policy_sha256=HASH_A,
        scores=tuple(
            ControlBlindScore(request_id, score)
            for request_id, score in values.items()
        ),
        cost=ControlScoreCost(
            scoring_requests=8,
            dot_product_scalar_products=16,
            cache_verification_square_terms=4,
            dot_product_bytes_read=128,
            cache_verification_bytes_read=16,
            total_file_bytes_read=144,
            unique_artifacts=8,
            unique_embedding_vectors=1,
            neural_embedding_calls_saved=15,
            peak_raw_chunk_bytes=16,
        ),
    )


class ControlEvaluationTests(unittest.TestCase):
    def test_matched_panel_joins_labels_only_at_evaluation(self) -> None:
        construction = _construction()
        cache = _cache_manifest()
        result = evaluate_sealed_control_scores(
            construction=construction,
            pairing_policy=_pairing_policy(),
            plan_sha256=HASH_A,
            pair_set_sha256=construction.result_sha256,
            bindings=_bindings(),
            panel_summaries=(_summary(),),
            blind_scores=_blind_scores(cache),
            embedding_cache_manifest=cache,
            threshold=FrozenVerificationThreshold(
                score_threshold=0.5,
                target_fmr=0.01,
                confidence_level=0.95,
                direction=VerificationDirection.RGB_TO_RGB,
                model_sha256=HASH_A,
                gallery_sha256=HASH_D,
                calibration_manifest_sha256=HASH_E,
            ),
            bootstrap=ClusterBootstrapConfig(
                ClusterUnit.QUERY_DOG,
                1_000,
                7,
                0.95,
            ),
            policy=ControlEvaluationPolicy(
                maximum_bindings=8,
                maximum_panels=1,
                maximum_total_auc_sort_items=8,
            ),
        )
        panel = result.panels[0]
        original, background = panel.controls
        self.assertEqual(original.control_kind, VisualControlKind.ORIGINAL)
        self.assertEqual(original.separation.roc_auc, 1.0)
        self.assertEqual(
            original.threshold_evaluation.false_match_rate.events,
            0,
        )
        self.assertEqual(background.separation.roc_auc, 0.875)
        self.assertEqual(
            background.threshold_evaluation.false_match_rate.events,
            2,
        )
        self.assertAlmostEqual(
            background.paired_delta.positive_mean_control_minus_original,
            -0.2,
        )
        self.assertAlmostEqual(
            background.paired_delta.negative_mean_control_minus_original,
            0.4,
        )
        self.assertEqual(result.cost.bindings_joined, 8)
        self.assertEqual(result.cost.auc_sort_items, 8)
        self.assertIn(
            "descriptive point estimates",
            " ".join(result.limitations),
        )
        self.assertNotIn("dog-1", str(_blind_scores(cache).to_dict()))

    def test_binding_parser_is_strict_and_round_trips(self) -> None:
        payload = {
            "schema_version": (
                "cvi.visual_control_evaluation_bindings.v1"
            ),
            "plan_sha256": HASH_A,
            "pair_set_sha256": _construction().result_sha256,
            "bindings": [binding.to_dict() for binding in _bindings()],
            "panel_summaries": [_summary().to_dict()],
        }
        plan, pair_set, bindings, summaries = (
            control_evaluation_bindings_from_payload(payload)
        )
        self.assertEqual(plan, HASH_A)
        self.assertEqual(pair_set, _construction().result_sha256)
        self.assertEqual(bindings, _bindings())
        self.assertEqual(summaries, (_summary(),))
        payload["identity"] = "leak"
        with self.assertRaisesRegex(ValueError, "unknown"):
            control_evaluation_bindings_from_payload(payload)

    def test_unmatched_or_wrong_stratum_panel_fails_closed(self) -> None:
        construction = _construction()
        cache = _cache_manifest()
        scores = _blind_scores(cache)
        bindings = _bindings()[:-1]
        shortened_scores = replace(
            scores,
            scores=scores.scores[:-1],
            cost=replace(scores.cost, scoring_requests=7),
        )
        common = {
            "construction": construction,
            "pairing_policy": _pairing_policy(),
            "plan_sha256": HASH_A,
            "pair_set_sha256": construction.result_sha256,
            "embedding_cache_manifest": cache,
            "threshold": FrozenVerificationThreshold(
                0.5,
                0.01,
                0.95,
                VerificationDirection.RGB_TO_RGB,
                HASH_A,
                HASH_D,
                HASH_E,
            ),
            "bootstrap": ClusterBootstrapConfig(
                ClusterUnit.QUERY_DOG,
                1_000,
                7,
                0.95,
            ),
            "policy": ControlEvaluationPolicy(),
        }
        with self.assertRaisesRegex(ValueError, "not pair-matched"):
            evaluate_sealed_control_scores(
                bindings=bindings,
                panel_summaries=(_summary(),),
                blind_scores=shortened_scores,
                **common,
            )
        wrong_summary = replace(
            _summary(),
            strata=(
                ControlStratumCount(PairStratum.POSITIVE, 1, 1),
                ControlStratumCount(PairStratum.RANDOM, 3, 3),
            ),
        )
        with self.assertRaisesRegex(ValueError, "stratum"):
            evaluate_sealed_control_scores(
                bindings=_bindings(),
                panel_summaries=(wrong_summary,),
                blind_scores=scores,
                **common,
            )

    def test_pairing_policy_and_threshold_direction_are_bound(self) -> None:
        construction = _construction()
        cache = _cache_manifest()
        common = {
            "construction": construction,
            "plan_sha256": HASH_A,
            "pair_set_sha256": construction.result_sha256,
            "bindings": _bindings(),
            "panel_summaries": (_summary(),),
            "blind_scores": _blind_scores(cache),
            "embedding_cache_manifest": cache,
            "bootstrap": ClusterBootstrapConfig(
                ClusterUnit.QUERY_DOG,
                1_000,
                7,
                0.95,
            ),
            "policy": ControlEvaluationPolicy(),
        }
        threshold = FrozenVerificationThreshold(
            0.5,
            0.01,
            0.95,
            VerificationDirection.RGB_TO_RGB,
            HASH_A,
            HASH_D,
            HASH_E,
        )
        with self.assertRaisesRegex(ValueError, "pair construction"):
            evaluate_sealed_control_scores(
                pairing_policy=replace(
                    _pairing_policy(),
                    direction=VerificationDirection.IR_TO_IR,
                ),
                threshold=threshold,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "direction"):
            evaluate_sealed_control_scores(
                pairing_policy=_pairing_policy(),
                threshold=replace(
                    threshold,
                    direction=VerificationDirection.IR_TO_IR,
                ),
                **common,
            )

    def test_cli_writes_private_sealed_evaluation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            construction = _construction()
            cache = _cache_manifest()
            payloads = {
                "pair-requests": construction.scoring_payload(),
                "pair-bindings": construction.artifact_binding_payload(),
                "pair-truth": construction.ground_truth_payload(),
                "pair-summary": construction.summary_payload(),
                "pairing-policy": _pairing_policy().to_dict(),
                "control-bindings": {
                    "schema_version": (
                        "cvi.visual_control_evaluation_bindings.v1"
                    ),
                    "plan_sha256": HASH_A,
                    "pair_set_sha256": construction.result_sha256,
                    "bindings": [
                        binding.to_dict() for binding in _bindings()
                    ],
                    "panel_summaries": [_summary().to_dict()],
                },
                "scores": _blind_scores(cache).to_dict(),
                "cache": cache.to_dict(),
                "threshold": FrozenVerificationThreshold(
                    0.5,
                    0.01,
                    0.95,
                    VerificationDirection.RGB_TO_RGB,
                    HASH_A,
                    HASH_D,
                    HASH_E,
                ).to_dict(),
                "bootstrap": ClusterBootstrapConfig(
                    ClusterUnit.QUERY_DOG,
                    1_000,
                    7,
                    0.95,
                ).to_dict(),
                "policy": ControlEvaluationPolicy(
                    maximum_bindings=8,
                    maximum_panels=1,
                    maximum_total_auc_sort_items=8,
                ).to_dict(),
            }
            paths = {}
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            output = root / "evaluation.json"
            argv = [
                "evaluate_visual_controls.py",
                "--pair-scoring-requests",
                str(paths["pair-requests"]),
                "--pair-artifact-bindings",
                str(paths["pair-bindings"]),
                "--pair-ground-truth",
                str(paths["pair-truth"]),
                "--pair-summary",
                str(paths["pair-summary"]),
                "--pairing-policy",
                str(paths["pairing-policy"]),
                "--control-evaluation-bindings",
                str(paths["control-bindings"]),
                "--blind-score-receipt",
                str(paths["scores"]),
                "--embedding-cache-manifest",
                str(paths["cache"]),
                "--frozen-threshold",
                str(paths["threshold"]),
                "--bootstrap-config",
                str(paths["bootstrap"]),
                "--evaluation-policy",
                str(paths["policy"]),
                "--evaluation-output",
                str(output),
            ]
            stdout = StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                main()
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "CREATED")
            self.assertEqual(summary["bindings_joined"], 8)
            self.assertTrue(output.is_file())
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
