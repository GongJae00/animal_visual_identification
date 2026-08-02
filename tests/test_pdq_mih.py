from __future__ import annotations

import hashlib
import itertools
import json
import random
import unittest
from dataclasses import fields, replace
from pathlib import Path

from identity_methods.classical.pdq_contracts import (
    PDQ_D4_ORIENTATIONS,
    PDQ_ELIGIBLE_SEARCHED,
    PDQ_INELIGIBLE_LOW_QUALITY,
    PDQ_NOT_IN_AUDIT,
    PDQ_QUALITY_THRESHOLD_STATUS,
    PDQFingerprint,
    PDQNearDuplicateCandidate,
    PDQSearchPolicy,
)
from identity_methods.classical.pdq_mih import (
    MIH_KEYS_PER_ORIENTATION,
    PDQCapacityExceeded,
    estimate_compact_mih_storage,
    estimate_random_like_raw_posting_visits,
    find_pdq_near_duplicate_candidates,
)


def _opaque(index: int) -> str:
    return hashlib.sha256(f"pdq-fixture-{index}".encode()).hexdigest()


def _hex(value: int) -> str:
    return f"{value:064x}"


def _fingerprint(
    index: int,
    hashes: tuple[int, ...] | None = None,
    *,
    quality: int = 100,
) -> PDQFingerprint:
    values = (0,) * 8 if hashes is None else hashes
    return PDQFingerprint(
        opaque_sample_id=_opaque(index),
        d4_hashes=tuple(_hex(value) for value in values),
        quality=quality,
    )


def _balanced_difference(distance: int) -> int:
    result = 0
    for index in range(distance):
        result |= 1 << ((index % 16) * 16 + index // 16)
    return result


def _brute_force(
    fingerprints: list[PDQFingerprint], radius: int, quality_threshold: int = 50
) -> tuple[tuple[str, str, int, str, str], ...]:
    eligible = sorted(
        (item for item in fingerprints if item.quality >= quality_threshold),
        key=lambda item: item.opaque_sample_id,
    )
    output: list[tuple[str, str, int, str, str]] = []
    for left, right in itertools.combinations(eligible, 2):
        witness = min(
            (
                (left_hash ^ right_hash).bit_count(),
                left_orientation,
                right_orientation,
            )
            for left_orientation, left_hash in enumerate(left.hash_integers)
            for right_orientation, right_hash in enumerate(right.hash_integers)
        )
        if witness[0] <= radius:
            output.append(
                (
                    left.opaque_sample_id,
                    right.opaque_sample_id,
                    witness[0],
                    PDQ_D4_ORIENTATIONS[witness[1]],
                    PDQ_D4_ORIENTATIONS[witness[2]],
                )
            )
    return tuple(sorted(output))


def _result_tuple(result: object) -> tuple[tuple[str, str, int, str, str], ...]:
    return tuple(
        (
            item.left_opaque_sample_id,
            item.right_opaque_sample_id,
            item.minimum_hamming_distance,
            item.left_orientation,
            item.right_orientation,
        )
        for item in result.candidates  # type: ignore[attr-defined]
    )


class PDQContractTests(unittest.TestCase):
    def test_exact_d4_order_and_lowercase_256_bit_semantics_are_retained(self) -> None:
        descending = tuple(_hex(255 - index) for index in range(8))
        fingerprint = PDQFingerprint(
            opaque_sample_id=_opaque(1), d4_hashes=descending, quality=0
        )
        self.assertEqual(fingerprint.d4_hashes, descending)
        self.assertEqual(
            PDQ_D4_ORIENTATIONS,
            (
                "ORIGINAL",
                "ROT90CCW",
                "ROT180",
                "ROT270CCW",
                "FLIP_X",
                "FLIP_Y",
                "FLIP_PLUS_DIAGONAL",
                "FLIP_MINUS_DIAGONAL",
            ),
        )
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            _fingerprint(2, (0,) * 7)
        with self.assertRaisesRegex(ValueError, "lowercase hex"):
            PDQFingerprint(
                opaque_sample_id=_opaque(2),
                d4_hashes=("A" * 64,) + descending[1:],
                quality=50,
            )
        with self.assertRaisesRegex(ValueError, "0..100"):
            _fingerprint(3, quality=101)

    def test_fingerprint_and_policy_json_schemas_are_exact(self) -> None:
        fingerprint = _fingerprint(1)
        self.assertEqual(PDQFingerprint.from_dict(fingerprint.to_dict()), fingerprint)
        contaminated = fingerprint.to_dict()
        contaminated["dog_id"] = "dog-001"
        with self.assertRaisesRegex(ValueError, "strict schema"):
            PDQFingerprint.from_dict(contaminated)

        path = (
            Path(__file__).parents[1]
            / "experiments"
            / "configs"
            / "contracts"
            / "public_canine_pdq_policy.example.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy = PDQSearchPolicy.from_dict(payload)
        self.assertEqual(policy, PDQSearchPolicy())
        self.assertEqual(policy.to_dict(), payload)
        self.assertEqual(policy.quality_threshold, 50)
        self.assertEqual(
            policy.quality_threshold_status, PDQ_QUALITY_THRESHOLD_STATUS
        )
        payload["split"] = "train"
        with self.assertRaisesRegex(ValueError, "strict schema"):
            PDQSearchPolicy.from_dict(payload)

    def test_candidate_contract_has_no_semantic_provenance_fields(self) -> None:
        names = {field.name for field in fields(PDQNearDuplicateCandidate)}
        self.assertEqual(
            names,
            {
                "left_opaque_sample_id",
                "right_opaque_sample_id",
                "minimum_hamming_distance",
                "left_orientation",
                "right_orientation",
                "left_quality",
                "right_quality",
                "minimum_quality",
                "distance_threshold",
                "quality_threshold",
                "schema_version",
            },
        )
        for forbidden in (
            "dog_id",
            "identity",
            "role",
            "source",
            "dataset",
            "split",
            "camera",
            "cage",
            "path",
        ):
            self.assertNotIn(forbidden, names)

        left, right = sorted((_opaque(11), _opaque(12)))
        candidate = PDQNearDuplicateCandidate(
            left_opaque_sample_id=left,
            right_opaque_sample_id=right,
            minimum_hamming_distance=7,
            left_orientation="ORIGINAL",
            right_orientation="FLIP_X",
            left_quality=60,
            right_quality=90,
            minimum_quality=60,
            distance_threshold=31,
            quality_threshold=50,
        )
        self.assertEqual(
            PDQNearDuplicateCandidate.from_dict(candidate.to_dict()), candidate
        )
        contaminated = candidate.to_dict()
        contaminated["role"] = "test"
        with self.assertRaisesRegex(ValueError, "strict schema"):
            PDQNearDuplicateCandidate.from_dict(contaminated)


class PDQMultiIndexHammingTests(unittest.TestCase):
    def test_matches_brute_force_at_required_radii(self) -> None:
        generator = random.Random(7341)
        fingerprints = [
            _fingerprint(
                index,
                tuple(generator.getrandbits(256) for _ in range(8)),
            )
            for index in range(14)
        ]
        anchor = generator.getrandbits(256)
        for offset, distance in enumerate((0, 1, 15, 16, 30, 31, 32)):
            hashes = [generator.getrandbits(256) for _ in range(8)]
            hashes[(offset + 3) % 8] = anchor ^ _balanced_difference(distance)
            fingerprints[offset] = _fingerprint(offset, tuple(hashes))
        anchor_hashes = [generator.getrandbits(256) for _ in range(8)]
        anchor_hashes[6] = anchor
        fingerprints[13] = _fingerprint(13, tuple(anchor_hashes))

        for radius in (0, 15, 16, 31):
            with self.subTest(radius=radius):
                policy = replace(PDQSearchPolicy(), distance_threshold=radius)
                result = find_pdq_near_duplicate_candidates(
                    fingerprints, policy=policy
                )
                self.assertEqual(
                    _result_tuple(result), _brute_force(fingerprints, radius)
                )

    def test_boundary_31_is_included_and_32_is_excluded(self) -> None:
        fingerprints = [
            _fingerprint(0, (0,) * 8),
            _fingerprint(1, (_balanced_difference(31),) * 8),
            _fingerprint(2, (_balanced_difference(32),) * 8),
        ]
        result = find_pdq_near_duplicate_candidates(fingerprints)
        anchor = _opaque(0)
        distances = {
            (item.left_opaque_sample_id, item.right_opaque_sample_id): (
                item.minimum_hamming_distance
            )
            for item in result.candidates
        }
        pair31 = tuple(sorted((anchor, _opaque(1))))
        pair32 = tuple(sorted((anchor, _opaque(2))))
        self.assertEqual(distances[pair31], 31)
        self.assertNotIn(pair32, distances)

    def test_non_original_witness_and_tie_break_follow_d4_order(self) -> None:
        pattern_a = int("aa" * 32, 16)
        pattern_b = int("55" * 32, 16)
        left_hashes = [pattern_a] * 8
        right_hashes = [pattern_b] * 8
        left_hashes[5] = 0
        right_hashes[2] = 0b111
        left = _fingerprint(1, tuple(left_hashes), quality=73)
        right = _fingerprint(2, tuple(right_hashes), quality=91)
        result = find_pdq_near_duplicate_candidates(
            [right, left],
            policy=replace(PDQSearchPolicy(), distance_threshold=3),
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        # Left/right are defined by opaque-token order, so orient the expected
        # witness by that deterministic sample order as well.
        if left.opaque_sample_id < right.opaque_sample_id:
            self.assertEqual(
                (candidate.left_orientation, candidate.right_orientation),
                ("FLIP_Y", "ROT180"),
            )
            self.assertEqual(
                (candidate.left_quality, candidate.right_quality), (73, 91)
            )
        else:
            self.assertEqual(
                (candidate.left_orientation, candidate.right_orientation),
                ("ROT180", "FLIP_Y"),
            )
            self.assertEqual(
                (candidate.left_quality, candidate.right_quality), (91, 73)
            )
        self.assertEqual(candidate.minimum_hamming_distance, 3)
        self.assertEqual(candidate.minimum_quality, 73)

    def test_input_permutation_does_not_change_candidates_or_witnesses(self) -> None:
        generator = random.Random(92)
        base = [
            _fingerprint(
                index, tuple(generator.getrandbits(256) for _ in range(8))
            )
            for index in range(9)
        ]
        shared = generator.getrandbits(256)
        first = list(base[0].hash_integers)
        second = list(base[1].hash_integers)
        first[7] = shared
        second[4] = shared ^ 0xFF
        base[0] = _fingerprint(0, tuple(first))
        base[1] = _fingerprint(1, tuple(second))
        expected = find_pdq_near_duplicate_candidates(base)
        for seed in range(6):
            shuffled = list(base)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(find_pdq_near_duplicate_candidates(shuffled), expected)

    def test_low_quality_is_explicitly_ineligible_not_a_nonduplicate(self) -> None:
        eligible = _fingerprint(1, quality=50)
        low = _fingerprint(2, quality=49)
        result = find_pdq_near_duplicate_candidates([low, eligible])
        self.assertEqual(result.candidates, ())
        self.assertEqual(
            result.sample_status(low.opaque_sample_id), PDQ_INELIGIBLE_LOW_QUALITY
        )
        self.assertEqual(
            result.sample_status(eligible.opaque_sample_id), PDQ_ELIGIBLE_SEARCHED
        )
        self.assertEqual(result.sample_status(_opaque(999)), PDQ_NOT_IN_AUDIT)
        self.assertEqual(
            result.quality_threshold_status,
            "INITIALIZATION_ONLY_NOT_CALIBRATION_ADMISSION",
        )

    def test_sample_orientation_and_work_caps_fail_closed(self) -> None:
        identical = [_fingerprint(index) for index in range(3)]
        with self.assertRaisesRegex(PDQCapacityExceeded, "sample cap"):
            find_pdq_near_duplicate_candidates(
                identical[:2],
                policy=replace(
                    PDQSearchPolicy(), maximum_samples=1, maximum_orientations=8
                ),
            )
        with self.assertRaisesRegex(PDQCapacityExceeded, "orientation cap"):
            find_pdq_near_duplicate_candidates(
                identical[:2],
                policy=replace(
                    PDQSearchPolicy(), maximum_samples=2, maximum_orientations=15
                ),
            )
        with self.assertRaisesRegex(PDQCapacityExceeded, "preflight cap"):
            find_pdq_near_duplicate_candidates(
                identical[:2],
                policy=replace(PDQSearchPolicy(), maximum_raw_posting_visits=100),
            )
        with self.assertRaisesRegex(PDQCapacityExceeded, "inspection cap"):
            find_pdq_near_duplicate_candidates(
                identical[:2],
                policy=replace(
                    PDQSearchPolicy(), maximum_unique_orientation_inspections=1
                ),
            )
        with self.assertRaisesRegex(PDQCapacityExceeded, "candidate cap"):
            find_pdq_near_duplicate_candidates(
                identical,
                policy=replace(
                    PDQSearchPolicy(), maximum_accepted_sample_candidates=1
                ),
            )

    def test_duplicate_ids_types_and_policy_bounds_fail_closed(self) -> None:
        fingerprint = _fingerprint(1)
        with self.assertRaisesRegex(ValueError, "duplicate opaque"):
            find_pdq_near_duplicate_candidates([fingerprint, fingerprint])
        with self.assertRaisesRegex(TypeError, "PDQFingerprint"):
            find_pdq_near_duplicate_candidates([object()])  # type: ignore[list-item]
        for radius in (-1, 32):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "0..31"):
                    replace(PDQSearchPolicy(), distance_threshold=radius)
        with self.assertRaisesRegex(ValueError, "safety ceiling"):
            replace(PDQSearchPolicy(), maximum_samples=50_001)

    def test_50k_random_like_plan_has_fixed_linear_packed_storage(self) -> None:
        estimate = estimate_compact_mih_storage(50_000)
        self.assertEqual(estimate.orientation_count, 400_000)
        self.assertEqual(estimate.packed_hash_bytes, 12_800_000)
        self.assertEqual(estimate.posting_bytes, 25_600_000)
        self.assertEqual(estimate.offset_bytes, 4_194_308)
        self.assertEqual(estimate.generation_stamp_bytes, 1_600_000)
        self.assertLess(estimate.peak_build_auxiliary_bytes, 50_000_000)
        self.assertLess(estimate.steady_search_auxiliary_bytes, 50_000_000)
        self.assertEqual(MIH_KEYS_PER_ORIENTATION, 272)
        self.assertLess(
            estimate_random_like_raw_posting_visits(50_000),
            PDQSearchPolicy().maximum_raw_posting_visits,
        )


if __name__ == "__main__":
    unittest.main()
