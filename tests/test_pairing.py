from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace

from contracts.contracts import Modality
from identity.splits.tracklet_split import (
    EvaluationStage,
    SplitManifest,
    SplitPolicy,
    SplitRole,
    TrackletRecord,
)
from evaluation import VerificationDirection
from evaluation.controls.pairing import (
    DogAttributes,
    NegativeQuota,
    PairingPolicy,
    PairStratum,
    construct_verification_pairs,
    dog_attributes_from_payload,
    pair_construction_from_bundle_payloads,
)


def record(
    sample_id: str,
    role: SplitRole,
    dog_id: str,
    session_id: str,
    cage_id: str,
    *,
    modality: Modality = Modality.RGB,
) -> TrackletRecord:
    return TrackletRecord(
        sample_id=sample_id,
        role=role,
        registered_dog_id=dog_id,
        identity_verification_source="microchip",
        source_id=f"source-{sample_id}",
        site_id="site-1",
        camera_id=f"camera-{sample_id}",
        cage_id=cage_id,
        session_id=session_id,
        occupancy_episode_id=f"episode-{sample_id}",
        track_id=f"track-{sample_id}",
        start_timestamp_ns=1,
        end_timestamp_ns=2,
        modality=modality,
    )


def attributes(
    dog_id: str,
    *,
    breed: str | None = None,
    confidence: float | None = None,
    mixed: bool = False,
    colors: tuple[str, ...] = (),
    patterns: tuple[str, ...] = (),
    size: str | None = None,
) -> DogAttributes:
    return DogAttributes(
        registered_dog_id=dog_id,
        breed_primary=breed,
        breed_confidence=confidence,
        mixed_breed=mixed,
        coat_colors=colors,
        coat_patterns=patterns,
        size_class=size,
    )


def policy(
    *quotas: NegativeQuota,
    maximum_queries_per_dog: int = 1,
) -> PairingPolicy:
    return PairingPolicy(
        name="calibration-rgb",
        stage=EvaluationStage.CALIBRATION,
        direction=VerificationDirection.RGB_TO_RGB,
        positive_pairs_per_query=1,
        negative_quotas=quotas,
        maximum_queries_per_dog=maximum_queries_per_dog,
        maximum_pairs_per_query=1 + sum(
            quota.pairs_per_query for quota in quotas
        ),
        maximum_candidate_scans_per_stratum=100,
        minimum_breed_confidence=0.8,
        seed=17,
    )


def manifest(*records: TrackletRecord) -> SplitManifest:
    return SplitManifest(
        policy=SplitPolicy(
            name="pairing-test",
            required_roles=(
                SplitRole.CALIBRATION_GALLERY,
                SplitRole.CALIBRATION_KNOWN_QUERY,
            ),
            require_train_evaluation_identity_disjoint=False,
            require_calibration_test_identity_disjoint=False,
        ),
        admitted_source_ids=tuple(record.source_id for record in records),
        records=records,
    )


class PairingTests(unittest.TestCase):
    def test_priority_strata_do_not_reuse_negative_identity(self) -> None:
        split = manifest(
            record("g1", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("g2a", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2a", "c2"),
            record("g2b", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2b", "c2"),
            record("g3", SplitRole.CALIBRATION_GALLERY, "dog-3", "s-g3", "c3"),
            record("g4", SplitRole.CALIBRATION_GALLERY, "dog-4", "s-g4", "c4"),
        )
        result = construct_verification_pairs(
            split,
            attributes=(
                attributes("dog-1", breed="breed-a", confidence=0.9, colors=("black",)),
                attributes("dog-2", breed="breed-a", confidence=0.9, colors=("black",)),
                attributes("dog-3", breed="breed-b", confidence=0.9, colors=("black",)),
                attributes("dog-4", breed="breed-c", confidence=0.9, colors=("white",)),
            ),
            policy=policy(
                NegativeQuota(PairStratum.SAME_BREED, 1),
                NegativeQuota(PairStratum.SAME_COAT, 1),
                NegativeQuota(PairStratum.RANDOM, 1),
            ),
        )
        negatives = [
            pair
            for pair in result.ground_truth
            if pair.stratum is not PairStratum.POSITIVE
        ]
        self.assertEqual(len(negatives), 3)
        self.assertEqual(len({pair.reference_dog_id for pair in negatives}), 3)
        self.assertEqual(
            {pair.stratum for pair in negatives},
            {
                PairStratum.SAME_BREED,
                PairStratum.SAME_COAT,
                PairStratum.RANDOM,
            },
        )
        self.assertEqual(
            sum(pair.reference_dog_id == "dog-2" for pair in negatives),
            1,
        )

    def test_quota_shortfall_is_not_silently_backfilled(self) -> None:
        split = manifest(
            record("g1", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("g2", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2", "c2"),
        )
        result = construct_verification_pairs(
            split,
            attributes=(
                attributes("dog-1", breed="breed-a", confidence=0.9),
                attributes("dog-2", breed="breed-b", confidence=0.9),
            ),
            policy=policy(
                NegativeQuota(PairStratum.SAME_BREED, 2),
                NegativeQuota(PairStratum.RANDOM, 1),
            ),
        )
        breed_quota = next(
            quota
            for quota in result.quotas
            if quota.stratum is PairStratum.SAME_BREED
        )
        self.assertEqual(breed_quota.produced, 0)
        self.assertEqual(breed_quota.shortfall, 2)
        self.assertEqual(
            sum(
                pair.stratum is PairStratum.RANDOM
                for pair in result.ground_truth
            ),
            1,
        )

    def test_candidate_uses_only_its_earliest_matching_stratum(self) -> None:
        split = manifest(
            record("g1", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("g2", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2", "c2"),
            record("g3", SplitRole.CALIBRATION_GALLERY, "dog-3", "s-g3", "c3"),
        )
        result = construct_verification_pairs(
            split,
            attributes=(
                attributes("dog-1", breed="breed-a", confidence=0.9, colors=("black",)),
                attributes("dog-2", breed="breed-a", confidence=0.9, colors=("black",)),
                attributes("dog-3", breed="breed-a", confidence=0.9, colors=("black",)),
            ),
            policy=policy(
                NegativeQuota(PairStratum.SAME_BREED, 1),
                NegativeQuota(PairStratum.SAME_COAT, 1),
            ),
        )
        self.assertEqual(
            sum(
                pair.stratum is PairStratum.SAME_COAT
                for pair in result.ground_truth
            ),
            0,
        )
        coat_quota = next(
            quota
            for quota in result.quotas
            if quota.stratum is PairStratum.SAME_COAT
        )
        self.assertEqual(coat_quota.shortfall, 1)

    def test_candidate_scan_cap_is_reported(self) -> None:
        split = manifest(
            record("g1", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            *(
                record(
                    f"g{index}",
                    SplitRole.CALIBRATION_GALLERY,
                    f"dog-{index}",
                    f"s-g{index}",
                    f"c{index}",
                )
                for index in range(2, 8)
            ),
        )
        base_policy = policy(NegativeQuota(PairStratum.RANDOM, 5))
        capped_policy = replace(
            base_policy,
            maximum_candidate_scans_per_stratum=2,
        )
        result = construct_verification_pairs(
            split,
            attributes=tuple(
                attributes(f"dog-{index}")
                for index in range(1, 8)
            ),
            policy=capped_policy,
        )
        quota = next(
            item
            for item in result.quotas
            if item.stratum is PairStratum.RANDOM
        )
        self.assertEqual(quota.candidates_scanned, 2)
        self.assertTrue(quota.scan_limit_reached)

    def test_mixed_breed_label_is_not_used_as_same_breed(self) -> None:
        split = manifest(
            record("g1", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("g2", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2", "c2"),
        )
        result = construct_verification_pairs(
            split,
            attributes=(
                attributes(
                    "dog-1",
                    breed="mixed",
                    confidence=1.0,
                    mixed=True,
                ),
                attributes(
                    "dog-2",
                    breed="mixed",
                    confidence=1.0,
                    mixed=True,
                ),
            ),
            policy=policy(
                NegativeQuota(PairStratum.SAME_BREED, 1),
            ),
        )
        quota = next(
            quota
            for quota in result.quotas
            if quota.stratum is PairStratum.SAME_BREED
        )
        self.assertEqual(quota.produced, 0)

    def test_query_and_positive_templates_are_session_bounded(self) -> None:
        split = manifest(
            record("g1a", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("g1b", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1a", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("q1b", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("q2", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q2", "c9"),
            record("g2", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2", "c2"),
        )
        result = construct_verification_pairs(
            split,
            attributes=(
                attributes("dog-1"),
                attributes("dog-2"),
            ),
            policy=policy(
                NegativeQuota(PairStratum.RANDOM, 1),
                maximum_queries_per_dog=2,
            ),
        )
        self.assertEqual(result.eligible_query_count, 3)
        self.assertEqual(result.selected_query_count, 2)
        self.assertEqual(result.dropped_query_count, 1)
        positives = [
            pair
            for pair in result.ground_truth
            if pair.stratum is PairStratum.POSITIVE
        ]
        self.assertEqual(len(positives), 2)
        self.assertTrue(
            all(
                pair.query_session_id != pair.reference_session_id
                for pair in positives
            )
        )

    def test_result_is_deterministic_and_content_addressed(self) -> None:
        split = manifest(
            record("g1", SplitRole.CALIBRATION_GALLERY, "dog-1", "s-g1", "c1"),
            record("q1", SplitRole.CALIBRATION_KNOWN_QUERY, "dog-1", "s-q1", "c9"),
            record("g2", SplitRole.CALIBRATION_GALLERY, "dog-2", "s-g2", "c2"),
        )
        dog_attributes = (attributes("dog-1"), attributes("dog-2"))
        pairing_policy = policy(NegativeQuota(PairStratum.RANDOM, 1))
        first = construct_verification_pairs(
            split,
            attributes=dog_attributes,
            policy=pairing_policy,
        )
        second = construct_verification_pairs(
            split,
            attributes=dog_attributes,
            policy=pairing_policy,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.result_sha256), 64)
        scoring_text = str(first.scoring_payload())
        self.assertNotIn("sample_id", scoring_text)
        self.assertNotIn("query_dog_id", scoring_text)
        self.assertNotIn("reference_dog_id", scoring_text)
        self.assertNotIn("stratum", scoring_text)
        self.assertIn("query_dog_id", str(first.ground_truth_payload()))
        self.assertIn("sample_id", str(first.artifact_binding_payload()))
        self.assertEqual(
            PairingPolicy.from_dict(pairing_policy.to_dict()),
            pairing_policy,
        )
        attributes_payload = {
            "schema_version": "cvi.dog_attributes.v1",
            "dogs": [item.to_dict() for item in dog_attributes],
        }
        self.assertEqual(
            dog_attributes_from_payload(attributes_payload),
            dog_attributes,
        )
        rebuilt = pair_construction_from_bundle_payloads(
            first.scoring_payload(),
            first.artifact_binding_payload(),
            first.ground_truth_payload(),
            first.summary_payload(),
        )
        self.assertEqual(rebuilt, first)

        tampered_summary = deepcopy(first.summary_payload())
        tampered_summary["selected_query_count"] += 1
        with self.assertRaisesRegex(ValueError, "query counts"):
            pair_construction_from_bundle_payloads(
                first.scoring_payload(),
                first.artifact_binding_payload(),
                first.ground_truth_payload(),
                tampered_summary,
            )

        tampered_bindings = deepcopy(first.artifact_binding_payload())
        tampered_bindings["bindings"][0]["sample_id"] = "substituted"
        with self.assertRaisesRegex(ValueError, "content hash"):
            pair_construction_from_bundle_payloads(
                first.scoring_payload(),
                tampered_bindings,
                first.ground_truth_payload(),
                first.summary_payload(),
            )

    def test_pairing_policy_parser_rejects_unknown_fields(self) -> None:
        payload = policy(
            NegativeQuota(PairStratum.RANDOM, 1)
        ).to_dict()
        payload["breed_hard_routing"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            PairingPolicy.from_dict(payload)

    def test_blocked_split_manifest_is_refused(self) -> None:
        bad = SplitManifest(
            policy=SplitPolicy(
                name="blocked",
                required_roles=(
                    SplitRole.CALIBRATION_GALLERY,
                    SplitRole.CALIBRATION_KNOWN_QUERY,
                ),
                require_train_evaluation_identity_disjoint=False,
                require_calibration_test_identity_disjoint=False,
            ),
            admitted_source_ids=("shared-source",),
            records=(
                TrackletRecord(
                    **{
                        **record(
                            "g1",
                            SplitRole.CALIBRATION_GALLERY,
                            "dog-1",
                            "s-g1",
                            "c1",
                        ).to_dict(),
                        "source_id": "shared-source",
                        "role": SplitRole.CALIBRATION_GALLERY,
                        "modality": Modality.RGB,
                    }
                ),
                TrackletRecord(
                    **{
                        **record(
                            "q1",
                            SplitRole.CALIBRATION_KNOWN_QUERY,
                            "dog-1",
                            "s-q1",
                            "c1",
                        ).to_dict(),
                        "source_id": "shared-source",
                        "role": SplitRole.CALIBRATION_KNOWN_QUERY,
                        "modality": Modality.RGB,
                    }
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "split manifest is blocked"):
            construct_verification_pairs(
                bad,
                attributes=(attributes("dog-1"),),
                policy=policy(NegativeQuota(PairStratum.RANDOM, 1)),
            )


if __name__ == "__main__":
    unittest.main()
