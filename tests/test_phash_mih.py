from __future__ import annotations

import itertools
import random
import unittest
from dataclasses import fields

from identity_methods.classical.phash_mih import (
    MAXIMUM_EXACT_RADIUS,
    PHASH_COEFFICIENTS,
    CandidateLimitExceeded,
    FingerprintLimitExceeded,
    NearDuplicateCandidate,
    PHashFingerprint,
    find_near_duplicate_candidates,
    fingerprint_luma32,
    opaque_sample_id,
)


def _opaque(index: int) -> str:
    return opaque_sample_id(f"fixture-{index}")


def _fingerprint(index: int, original: int, flipped: int | None = None) -> PHashFingerprint:
    return PHashFingerprint(
        opaque_sample_id=_opaque(index),
        original_hash=original,
        horizontal_flip_hash=original if flipped is None else flipped,
    )


def _brute_force(
    fingerprints: list[PHashFingerprint], radius: int
) -> tuple[tuple[str, str, int], ...]:
    output: list[tuple[str, str, int]] = []
    for left, right in itertools.combinations(
        sorted(fingerprints, key=lambda item: item.opaque_sample_id), 2
    ):
        distance = min(
            (left_hash ^ right_hash).bit_count()
            for left_hash in left.unique_hashes
            for right_hash in right.unique_hashes
        )
        if distance <= radius:
            output.append((left.opaque_sample_id, right.opaque_sample_id, distance))
    return tuple(output)


def _result_tuple(
    result: tuple[NearDuplicateCandidate, ...],
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            item.left_opaque_sample_id,
            item.right_opaque_sample_id,
            item.hamming_distance,
        )
        for item in result
    )


def _balanced_difference(distance: int) -> int:
    """Spread successive bits round-robin over all four 16-bit MIH blocks."""

    result = 0
    for index in range(distance):
        result |= 1 << ((index % 4) * 16 + index // 4)
    return result


class PHashTests(unittest.TestCase):
    def test_coefficient_contract_is_fixed_64_ac_positions(self) -> None:
        self.assertEqual(len(PHASH_COEFFICIENTS), 64)
        self.assertEqual(len(set(PHASH_COEFFICIENTS)), 64)
        self.assertNotIn((0, 0), PHASH_COEFFICIENTS)
        self.assertEqual(PHASH_COEFFICIENTS[:5], (
            (0, 1), (1, 0), (2, 0), (1, 1), (0, 2)
        ))
        self.assertEqual(PHASH_COEFFICIENTS[-1], (1, 9))

    def test_flat_rasters_have_zero_tie_bits_and_identical_orientation(self) -> None:
        for level in (0, 1, 127, 255):
            result = fingerprint_luma32(
                opaque_id=_opaque(level), luma_pixels=bytes([level]) * 1024
            )
            self.assertEqual(result.original_hash, 0)
            self.assertEqual(result.horizontal_flip_hash, 0)

    def test_horizontal_flip_hash_matches_an_explicitly_flipped_raster(self) -> None:
        pixels = bytes((row * 17 + column * 29 + row * column) % 256
                       for row in range(32) for column in range(32))
        flipped = b"".join(
            pixels[row * 32:(row + 1) * 32][::-1] for row in range(32)
        )
        original_result = fingerprint_luma32(
            opaque_id=_opaque(1), luma_pixels=pixels
        )
        flipped_result = fingerprint_luma32(
            opaque_id=_opaque(2), luma_pixels=flipped
        )
        self.assertEqual(
            original_result.horizontal_flip_hash, flipped_result.original_hash
        )
        self.assertEqual(
            original_result.original_hash, flipped_result.horizontal_flip_hash
        )

    def test_input_contract_rejects_semantic_or_malformed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            fingerprint_luma32(opaque_id="train/dog_001.jpg", luma_pixels=b"\0" * 1024)
        with self.assertRaisesRegex(ValueError, "32 x 32"):
            fingerprint_luma32(opaque_id=_opaque(1), luma_pixels=b"\0" * 1023)
        with self.assertRaisesRegex(TypeError, "immutable bytes"):
            fingerprint_luma32(opaque_id=_opaque(1), luma_pixels=bytearray(1024))  # type: ignore[arg-type]
        self.assertEqual(
            {field.name for field in fields(NearDuplicateCandidate)},
            {
                "left_opaque_sample_id",
                "right_opaque_sample_id",
                "hamming_distance",
                "schema_version",
            },
        )


class MultiIndexHammingTests(unittest.TestCase):
    def test_matches_brute_force_for_seeded_random_orientation_fixtures(self) -> None:
        for seed in range(8):
            generator = random.Random(seed)
            fingerprints = [
                _fingerprint(
                    index,
                    generator.getrandbits(64),
                    generator.getrandbits(64),
                )
                for index in range(36)
            ]
            # Inject boundary-near clusters; purely random 64-bit values otherwise
            # almost never exercise accepted pairs at radius ten.
            anchor = generator.getrandbits(64)
            for offset, distance in enumerate(range(0, 11)):
                fingerprints[offset] = _fingerprint(
                    offset, anchor ^ _balanced_difference(distance)
                )
            expected = _brute_force(fingerprints, MAXIMUM_EXACT_RADIUS)
            actual = find_near_duplicate_candidates(
                fingerprints,
                radius=MAXIMUM_EXACT_RADIUS,
                maximum_pair_inspections=10_000,
                maximum_accepted_candidates=10_000,
            )
            self.assertEqual(_result_tuple(actual), expected)

    def test_every_radius_boundary_zero_through_ten_is_exact(self) -> None:
        fingerprints = [_fingerprint(0, 0)] + [
            _fingerprint(distance + 1, _balanced_difference(distance))
            for distance in range(0, 12)
        ]
        for radius in range(0, 11):
            with self.subTest(radius=radius):
                expected = _brute_force(fingerprints, radius)
                actual = find_near_duplicate_candidates(
                    fingerprints,
                    radius=radius,
                    maximum_pair_inspections=1_000,
                    maximum_accepted_candidates=1_000,
                )
                self.assertEqual(_result_tuple(actual), expected)

    def test_original_and_horizontal_flip_fingerprints_both_participate(self) -> None:
        fingerprints = [
            _fingerprint(1, 0x0123456789ABCDEF, 0xFEDCBA9876543210),
            _fingerprint(2, 0xFEDCBA9876543210 ^ 0b111, 0xAAAAAAAAAAAAAAAA),
        ]
        result = find_near_duplicate_candidates(
            fingerprints,
            radius=3,
            maximum_pair_inspections=10,
            maximum_accepted_candidates=10,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].hamming_distance, 3)

    def test_input_order_does_not_change_pairs_or_distances(self) -> None:
        generator = random.Random(917)
        base = [_fingerprint(index, generator.getrandbits(64)) for index in range(15)]
        base[1] = _fingerprint(1, base[0].original_hash ^ 0x3F)
        expected = find_near_duplicate_candidates(
            base,
            radius=10,
            maximum_pair_inspections=1_000,
            maximum_accepted_candidates=1_000,
        )
        for seed in range(10):
            shuffled = list(base)
            random.Random(seed).shuffle(shuffled)
            self.assertEqual(
                find_near_duplicate_candidates(
                    shuffled,
                    radius=10,
                    maximum_pair_inspections=1_000,
                    maximum_accepted_candidates=1_000,
                ),
                expected,
            )

    def test_flat_collision_bucket_fails_closed_at_inspection_cap(self) -> None:
        fingerprints = [_fingerprint(index, 0) for index in range(20)]
        with self.assertRaisesRegex(
            CandidateLimitExceeded, "pair-inspection cap exceeded"
        ):
            find_near_duplicate_candidates(
                fingerprints,
                radius=10,
                maximum_pair_inspections=50,
                maximum_accepted_candidates=1_000,
            )

    def test_accepted_candidate_memory_has_an_independent_cap(self) -> None:
        fingerprints = [_fingerprint(index, 0) for index in range(3)]
        with self.assertRaisesRegex(
            CandidateLimitExceeded, "accepted-candidate cap exceeded"
        ):
            find_near_duplicate_candidates(
                fingerprints,
                radius=10,
                maximum_pair_inspections=10,
                maximum_accepted_candidates=1,
            )

    def test_invalid_radius_duplicate_id_and_cap_fail_closed(self) -> None:
        fingerprint = _fingerprint(1, 0)
        for radius in (-1, 11):
            with self.assertRaisesRegex(ValueError, "0..10"):
                find_near_duplicate_candidates(
                    [fingerprint],
                    radius=radius,
                    maximum_pair_inspections=1,
                    maximum_accepted_candidates=1,
                )
        with self.assertRaisesRegex(ValueError, "duplicate opaque"):
            find_near_duplicate_candidates(
                [fingerprint, fingerprint],
                radius=10,
                maximum_pair_inspections=1,
                maximum_accepted_candidates=1,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            find_near_duplicate_candidates(
                [fingerprint],
                radius=10,
                maximum_pair_inspections=0,
                maximum_accepted_candidates=1,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            find_near_duplicate_candidates(
                [fingerprint],
                radius=10,
                maximum_pair_inspections=1,
                maximum_accepted_candidates=0,
            )
        with self.assertRaisesRegex(FingerprintLimitExceeded, "fingerprint cap"):
            find_near_duplicate_candidates(
                [fingerprint, _fingerprint(2, 1)],
                radius=10,
                maximum_pair_inspections=1,
                maximum_accepted_candidates=1,
                maximum_fingerprints=1,
            )


if __name__ == "__main__":
    unittest.main()
