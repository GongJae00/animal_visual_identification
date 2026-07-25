from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.evaluation import (
    ClusterBootstrapConfig,
    ClusterUnit,
    FrozenVerificationThreshold,
    VerificationDirection,
)
from cvi.dataset import EvaluationStage
from cvi.pairing import (
    NegativeQuota,
    PairArtifactBinding,
    PairConstructionResult,
    PairGroundTruth,
    PairingPolicy,
    PairScoringRequest,
    PairStratum,
)
from cvi.provenance import content_sha256
from cvi.scoring import (
    BlindPairScore,
    BlindScoreReceipt,
    PairArtifactEntry,
    PairArtifactManifest,
    evaluate_blind_score_receipt,
    join_blind_scores,
    verify_pair_artifact_files,
)
from cvi.acquisition import sha256_file

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64
HASH_0 = "0" * 64


def pairing_policy() -> PairingPolicy:
    return PairingPolicy(
        name="test-rgb-to-rgb",
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


def construction() -> PairConstructionResult:
    requests = tuple(
        PairScoringRequest(
            pair_id=f"pair-{index}",
            query_artifact_token=f"query-token-{index}",
            reference_artifact_token=f"reference-token-{index}",
        )
        for index in range(4)
    )
    truth = (
        PairGroundTruth(
            "pair-0", "dog-1", "dog-1", "q-session-1", "r-session-1",
            PairStratum.POSITIVE,
        ),
        PairGroundTruth(
            "pair-1", "dog-2", "dog-2", "q-session-2", "r-session-2",
            PairStratum.POSITIVE,
        ),
        PairGroundTruth(
            "pair-2", "dog-3", "dog-4", "q-session-3", "r-session-3",
            PairStratum.SAME_BREED,
        ),
        PairGroundTruth(
            "pair-3", "dog-5", "dog-6", "q-session-5", "r-session-5",
            PairStratum.RANDOM,
        ),
    )
    bindings = tuple(
        PairArtifactBinding(
            artifact_token=token,
            sample_id=f"protected-{token}",
        )
        for request in requests
        for token in (
            request.query_artifact_token,
            request.reference_artifact_token,
        )
    )
    return PairConstructionResult(
        split_manifest_sha256=HASH_A,
        pairing_policy_sha256=pairing_policy().policy_sha256,
        attributes_sha256=HASH_C,
        eligible_query_count=4,
        selected_query_count=4,
        dropped_query_count=0,
        scoring_requests=requests,
        artifact_bindings=bindings,
        ground_truth=truth,
        quotas=(),
    )


def artifact_manifest(
    pairs: PairConstructionResult,
) -> PairArtifactManifest:
    return PairArtifactManifest(
        pair_set_sha256=pairs.result_sha256,
        artifact_bindings_sha256=content_sha256(
            pairs.artifact_binding_payload()
        ),
        entries=tuple(
            PairArtifactEntry(
                artifact_token=binding.artifact_token,
                relative_path=f"{binding.artifact_token}.png",
                content_sha256=content_sha256(
                    {"token": binding.artifact_token}
                ),
                byte_size=100,
                media_type="image/png",
            )
            for binding in pairs.artifact_bindings
        ),
    )


def receipt(
    pairs: PairConstructionResult,
    artifacts: PairArtifactManifest,
) -> BlindScoreReceipt:
    return BlindScoreReceipt(
        pair_set_sha256=pairs.result_sha256,
        scoring_requests_sha256=content_sha256(pairs.scoring_payload()),
        artifact_manifest_sha256=artifacts.manifest_sha256,
        model_sha256=HASH_E,
        gallery_sha256=HASH_F,
        inference_config_sha256=HASH_0,
        dependency_lock_sha256=HASH_A,
        code_revision="dirty:test-revision",
        scorer_version="oracle-scorer-v1",
        precision="fp32",
        device="cpu-test",
        scores=(
            BlindPairScore("pair-0", 0.9),
            BlindPairScore("pair-1", 0.7),
            BlindPairScore("pair-2", 0.9),
            BlindPairScore("pair-3", 0.1),
        ),
    )


class ScoringTests(unittest.TestCase):
    def test_blind_join_requires_exact_pair_set(self) -> None:
        pairs = construction()
        artifacts = artifact_manifest(pairs)
        joined = join_blind_scores(pairs, artifacts, receipt(pairs, artifacts))
        self.assertEqual(tuple(item.pair_id for item in joined), (
            "pair-0", "pair-1", "pair-2", "pair-3"
        ))
        self.assertEqual(joined[0].query_track_id, "query-token-0")
        self.assertEqual(joined[0].query_dog_id, "dog-1")

    def test_missing_or_extra_score_is_rejected(self) -> None:
        pairs = construction()
        artifacts = artifact_manifest(pairs)
        base = receipt(pairs, artifacts)
        with self.assertRaisesRegex(ValueError, "missing"):
            join_blind_scores(
                pairs,
                artifacts,
                replace(base, scores=base.scores[:-1]),
            )
        with self.assertRaisesRegex(ValueError, "extra"):
            join_blind_scores(
                pairs,
                artifacts,
                replace(
                    base,
                    scores=base.scores + (
                        BlindPairScore("pair-extra", 0.5),
                    ),
                ),
            )

    def test_stale_request_hash_is_rejected(self) -> None:
        pairs = construction()
        artifacts = artifact_manifest(pairs)
        with self.assertRaisesRegex(ValueError, "request hash"):
            join_blind_scores(
                pairs,
                artifacts,
                replace(
                    receipt(pairs, artifacts),
                    scoring_requests_sha256=HASH_B,
                ),
            )

    def test_receipt_parser_rejects_label_fields(self) -> None:
        pairs = construction()
        payload = receipt(pairs, artifact_manifest(pairs)).to_dict()
        payload["query_dog_id"] = "leak"
        with self.assertRaisesRegex(ValueError, "unknown"):
            BlindScoreReceipt.from_dict(payload)

    def test_threshold_model_and_gallery_are_bound(self) -> None:
        pairs = construction()
        artifacts = artifact_manifest(pairs)
        score_receipt = receipt(pairs, artifacts)
        threshold = FrozenVerificationThreshold(
            score_threshold=0.8,
            target_fmr=1e-3,
            confidence_level=0.95,
            direction=VerificationDirection.RGB_TO_RGB,
            model_sha256=HASH_E,
            gallery_sha256=HASH_F,
            calibration_manifest_sha256=HASH_C,
        )
        bootstrap = ClusterBootstrapConfig(
            cluster_unit=ClusterUnit.QUERY_DOG,
            resamples=1_000,
            seed=11,
            confidence_level=0.95,
        )
        evaluated = evaluate_blind_score_receipt(
            pairs,
            artifacts,
            score_receipt,
            pairing_policy=pairing_policy(),
            threshold=threshold,
            test_manifest_sha256=HASH_D,
            bootstrap=bootstrap,
        )
        self.assertEqual(
            evaluated.verification.false_match_rate.events,
            1,
        )
        with self.assertRaisesRegex(ValueError, "model"):
            evaluate_blind_score_receipt(
                pairs,
                artifacts,
                replace(score_receipt, model_sha256=HASH_A),
                pairing_policy=pairing_policy(),
                threshold=threshold,
                test_manifest_sha256=HASH_D,
                bootstrap=bootstrap,
            )

    def test_evaluation_binds_pairing_policy_and_direction(self) -> None:
        pairs = construction()
        artifacts = artifact_manifest(pairs)
        score_receipt = receipt(pairs, artifacts)
        threshold = FrozenVerificationThreshold(
            0.8,
            1e-3,
            0.95,
            VerificationDirection.RGB_TO_RGB,
            HASH_E,
            HASH_F,
            HASH_C,
        )
        bootstrap = ClusterBootstrapConfig(
            ClusterUnit.QUERY_DOG,
            1_000,
            11,
            0.95,
        )
        wrong_policy = replace(
            pairing_policy(),
            direction=VerificationDirection.IR_TO_IR,
        )
        with self.assertRaisesRegex(ValueError, "pair construction"):
            evaluate_blind_score_receipt(
                pairs,
                artifacts,
                score_receipt,
                pairing_policy=wrong_policy,
                threshold=threshold,
                test_manifest_sha256=HASH_D,
                bootstrap=bootstrap,
            )
        mismatched_threshold = replace(
            threshold,
            direction=VerificationDirection.IR_TO_IR,
        )
        with self.assertRaisesRegex(ValueError, "direction"):
            evaluate_blind_score_receipt(
                pairs,
                artifacts,
                score_receipt,
                pairing_policy=pairing_policy(),
                threshold=mismatched_threshold,
                test_manifest_sha256=HASH_D,
                bootstrap=bootstrap,
            )

    def test_artifact_manifest_requires_exact_binding_and_tokens(self) -> None:
        pairs = construction()
        artifacts = artifact_manifest(pairs)
        score_receipt = receipt(pairs, artifacts)
        with self.assertRaisesRegex(ValueError, "binding hash"):
            join_blind_scores(
                pairs,
                replace(artifacts, artifact_bindings_sha256=HASH_B),
                score_receipt,
            )
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            join_blind_scores(
                pairs,
                replace(artifacts, entries=artifacts.entries[:-1]),
                score_receipt,
            )

    def test_artifact_files_are_rehashed_and_extra_files_are_rejected(self) -> None:
        pairs = construction()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for index, binding in enumerate(pairs.artifact_bindings):
                path = root / f"{binding.artifact_token}.png"
                path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
                entries.append(
                    PairArtifactEntry(
                        artifact_token=binding.artifact_token,
                        relative_path=path.name,
                        content_sha256=sha256_file(path),
                        byte_size=path.stat().st_size,
                        media_type="image/png",
                    )
                )
            manifest = PairArtifactManifest(
                pair_set_sha256=pairs.result_sha256,
                artifact_bindings_sha256=content_sha256(
                    pairs.artifact_binding_payload()
                ),
                entries=tuple(entries),
            )
            verified = verify_pair_artifact_files(root, manifest)
            self.assertEqual(verified.verified_files, len(entries))
            (root / "identity-leak.txt").write_text(
                "dog-1",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "entries mismatch"):
                verify_pair_artifact_files(root, manifest)

    def test_artifact_root_symlink_is_rejected(self) -> None:
        pairs = construction()
        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "artifacts"
            root.mkdir()
            entries = []
            for index, binding in enumerate(pairs.artifact_bindings):
                path = root / f"{binding.artifact_token}.png"
                path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
                entries.append(
                    PairArtifactEntry(
                        artifact_token=binding.artifact_token,
                        relative_path=path.name,
                        content_sha256=sha256_file(path),
                        byte_size=path.stat().st_size,
                        media_type="image/png",
                    )
                )
            manifest = PairArtifactManifest(
                pair_set_sha256=pairs.result_sha256,
                artifact_bindings_sha256=content_sha256(
                    pairs.artifact_binding_payload()
                ),
                entries=tuple(entries),
            )
            linked_root = parent / "linked"
            linked_root.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "root"):
                verify_pair_artifact_files(linked_root, manifest)


if __name__ == "__main__":
    unittest.main()
