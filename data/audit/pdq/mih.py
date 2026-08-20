"""Exact-complete bounded MIH search for ordered 256-bit PDQ D4 hashes.

Each 256-bit orientation is split into sixteen 16-bit slots.  A query probes
the exact value and all sixteen one-bit flips in every slot (272 bucket keys
per orientation).  If two hashes have total Hamming distance at most 31, at
least one slot has distance at most one, so the subsequent full 256-bit check
is exact-complete for every requested radius in ``0..31``.

The index is CSR-like: uint32 offsets and postings replace Python bucket lists.
Hashes are packed as four uint64 words.  Posting deduplication uses one uint32
generation-stamp array per query orientation; there is no global provisional
pair set.  Sample-pair minima are retained only for the current query sample.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Sequence

from data.audit.pdq.contracts import (
    PDQ_D4_ORIENTATIONS,
    PDQ_ORIENTATION_COUNT,
    PDQFingerprint,
    PDQNearDuplicateCandidate,
    PDQSearchPolicy,
    PDQSearchResult,
)


MIH_SLOT_COUNT = 16
MIH_SLOT_BITS = 16
MIH_SLOT_VALUES = 1 << MIH_SLOT_BITS
MIH_BUCKET_COUNT = MIH_SLOT_COUNT * MIH_SLOT_VALUES
MIH_KEYS_PER_ORIENTATION = MIH_SLOT_COUNT * (1 + MIH_SLOT_BITS)
_SLOT_MASK = MIH_SLOT_VALUES - 1
_UINT64_MASK = (1 << 64) - 1


class PDQCapacityExceeded(RuntimeError):
    """Raised before a declared sample, orientation, work, or output cap."""


@dataclass(frozen=True, slots=True)
class PDQCompactStorageEstimate:
    sample_count: int
    orientation_count: int
    packed_hash_bytes: int
    posting_bytes: int
    offset_bytes: int
    generation_stamp_bytes: int
    build_cursor_bytes: int
    steady_search_auxiliary_bytes: int
    peak_build_auxiliary_bytes: int


def estimate_compact_mih_storage(sample_count: int) -> PDQCompactStorageEstimate:
    """Return deterministic packed-array sizes, excluding caller/output objects."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 0 <= sample_count <= 50_000
    ):
        raise ValueError("sample_count must be an integer inside 0..50000")
    orientations = sample_count * PDQ_ORIENTATION_COUNT
    packed_hash_bytes = orientations * 32
    posting_bytes = orientations * MIH_SLOT_COUNT * 4
    offset_bytes = (MIH_BUCKET_COUNT + 1) * 4
    stamp_bytes = orientations * 4
    cursor_bytes = MIH_BUCKET_COUNT * 4
    steady = packed_hash_bytes + posting_bytes + offset_bytes + stamp_bytes
    peak = packed_hash_bytes + posting_bytes + offset_bytes + cursor_bytes
    return PDQCompactStorageEstimate(
        sample_count=sample_count,
        orientation_count=orientations,
        packed_hash_bytes=packed_hash_bytes,
        posting_bytes=posting_bytes,
        offset_bytes=offset_bytes,
        generation_stamp_bytes=stamp_bytes,
        build_cursor_bytes=cursor_bytes,
        steady_search_auxiliary_bytes=steady,
        peak_build_auxiliary_bytes=peak,
    )


def estimate_random_like_raw_posting_visits(sample_count: int) -> float:
    """Expected preflight work for independent uniform orientation hashes.

    This is a planning estimate, not an admission decision.  Actual bucket
    counts are always measured exactly before allocating the postings array.
    """

    estimate = estimate_compact_mih_storage(sample_count)
    orientations = estimate.orientation_count
    if orientations == 0:
        return 0.0
    expected_bucket_visits = 1.0 + (
        (orientations - 1) * (1 + MIH_SLOT_BITS) / MIH_SLOT_VALUES
    )
    return orientations * MIH_SLOT_COUNT * expected_bucket_visits


def find_pdq_near_duplicate_candidates(
    fingerprints: Sequence[PDQFingerprint] | Iterable[PDQFingerprint],
    *,
    policy: PDQSearchPolicy | None = None,
) -> PDQSearchResult:
    """Return every eligible sample pair within the policy distance.

    Samples below the quality initialization threshold are explicitly returned
    as ``PDQ_INELIGIBLE_LOW_QUALITY`` evidence and are not indexed.  Their
    absence from ``candidates`` must never be interpreted as a nonduplicate
    decision.
    """

    effective_policy = PDQSearchPolicy() if policy is None else policy
    if not isinstance(effective_policy, PDQSearchPolicy):
        raise TypeError("policy must be a PDQSearchPolicy")

    materialized: list[PDQFingerprint] = []
    for item in fingerprints:
        if not isinstance(item, PDQFingerprint):
            raise TypeError("all inputs must be PDQFingerprint instances")
        if len(materialized) >= effective_policy.maximum_samples:
            raise PDQCapacityExceeded("PDQ sample cap exceeded")
        materialized.append(item)
    ordered = tuple(sorted(materialized, key=lambda item: item.opaque_sample_id))
    if len({item.opaque_sample_id for item in ordered}) != len(ordered):
        raise ValueError("fingerprints contain duplicate opaque sample IDs")
    total_orientations = len(ordered) * PDQ_ORIENTATION_COUNT
    if total_orientations > effective_policy.maximum_orientations:
        raise PDQCapacityExceeded("PDQ orientation cap exceeded")

    eligible_items = tuple(
        item for item in ordered if item.quality >= effective_policy.quality_threshold
    )
    low_quality_ids = tuple(
        item.opaque_sample_id
        for item in ordered
        if item.quality < effective_policy.quality_threshold
    )
    eligible_ids = tuple(item.opaque_sample_id for item in eligible_items)
    eligible_count = len(eligible_items)
    orientation_count = eligible_count * PDQ_ORIENTATION_COUNT
    if orientation_count == 0:
        return PDQSearchResult(
            candidates=(),
            eligible_sample_ids=(),
            ineligible_low_quality_sample_ids=low_quality_ids,
            preflight_raw_posting_visits=0,
            unique_orientation_inspections=0,
            indexed_orientation_count=0,
            distance_threshold=effective_policy.distance_threshold,
            quality_threshold=effective_policy.quality_threshold,
        )

    sample_qualities = array("B", (item.quality for item in eligible_items))
    packed_hashes = _pack_hashes(eligible_items)
    # Search retains only compact hashes, qualities, and opaque-ID references.
    # Generator-fed fingerprint objects and their eight hex strings can now be
    # released before allocating the CSR postings.
    del materialized, ordered, eligible_items
    bucket_counts = _count_buckets(packed_hashes, orientation_count)
    preflight_work = _preflight_raw_posting_visits(
        bucket_counts,
        maximum=effective_policy.maximum_raw_posting_visits,
    )
    offsets, postings = _build_csr_index(
        packed_hashes, orientation_count, bucket_counts
    )
    del bucket_counts

    candidates: list[PDQNearDuplicateCandidate] = []
    stamps = array("I", [0]) * orientation_count
    generation = 0
    unique_inspections = 0
    raw_posting_visits = 0

    for query_sample_index in range(eligible_count):
        # other_sample_index -> (distance, left orientation, right orientation)
        sample_minima: dict[int, tuple[int, int, int]] = {}
        for query_orientation_index in range(PDQ_ORIENTATION_COUNT):
            query_orientation_id = (
                query_sample_index * PDQ_ORIENTATION_COUNT
                + query_orientation_index
            )
            generation += 1
            for slot_index in range(MIH_SLOT_COUNT):
                slot_value = _packed_slot(
                    packed_hashes, query_orientation_id, slot_index
                )
                for neighbor_value in _exact_and_one_bit_flips(slot_value):
                    bucket = slot_index * MIH_SLOT_VALUES + neighbor_value
                    start = offsets[bucket]
                    end = offsets[bucket + 1]
                    # CSR postings are filled in ascending orientation order.
                    # Binary-searching the directional boundary avoids reading
                    # this sample and every future sample merely to reject it.
                    directional_end = bisect_left(
                        postings,
                        query_sample_index * PDQ_ORIENTATION_COUNT,
                        start,
                        end,
                    )
                    raw_posting_visits += directional_end - start
                    for posting_index in range(start, directional_end):
                        other_orientation_id = postings[posting_index]
                        other_sample_index = (
                            other_orientation_id // PDQ_ORIENTATION_COUNT
                        )
                        if stamps[other_orientation_id] == generation:
                            continue
                        stamps[other_orientation_id] = generation
                        if (
                            unique_inspections
                            >= effective_policy.maximum_unique_orientation_inspections
                        ):
                            raise PDQCapacityExceeded(
                                "PDQ unique-orientation inspection cap exceeded"
                            )
                        unique_inspections += 1
                        distance = _packed_hamming_distance(
                            packed_hashes,
                            other_orientation_id,
                            query_orientation_id,
                        )
                        if distance > effective_policy.distance_threshold:
                            continue
                        other_orientation_index = (
                            other_orientation_id % PDQ_ORIENTATION_COUNT
                        )
                        witness = (
                            distance,
                            other_orientation_index,
                            query_orientation_index,
                        )
                        previous = sample_minima.get(other_sample_index)
                        if previous is None or witness < previous:
                            sample_minima[other_sample_index] = witness

        for other_sample_index in sorted(sample_minima):
            if (
                len(candidates)
                >= effective_policy.maximum_accepted_sample_candidates
            ):
                raise PDQCapacityExceeded(
                    "PDQ accepted sample-candidate cap exceeded"
                )
            distance, left_orientation_index, right_orientation_index = (
                sample_minima[other_sample_index]
            )
            left_quality = sample_qualities[other_sample_index]
            right_quality = sample_qualities[query_sample_index]
            candidates.append(
                PDQNearDuplicateCandidate(
                    left_opaque_sample_id=eligible_ids[other_sample_index],
                    right_opaque_sample_id=eligible_ids[query_sample_index],
                    minimum_hamming_distance=distance,
                    left_orientation=PDQ_D4_ORIENTATIONS[left_orientation_index],
                    right_orientation=PDQ_D4_ORIENTATIONS[right_orientation_index],
                    left_quality=left_quality,
                    right_quality=right_quality,
                    minimum_quality=min(left_quality, right_quality),
                    distance_threshold=effective_policy.distance_threshold,
                    quality_threshold=effective_policy.quality_threshold,
                )
            )

    if raw_posting_visits > preflight_work:
        raise AssertionError("PDQ traversed work exceeds its conservative preflight")
    candidates.sort(
        key=lambda item: (
            item.left_opaque_sample_id,
            item.right_opaque_sample_id,
        )
    )
    return PDQSearchResult(
        candidates=tuple(candidates),
        eligible_sample_ids=eligible_ids,
        ineligible_low_quality_sample_ids=low_quality_ids,
        preflight_raw_posting_visits=preflight_work,
        unique_orientation_inspections=unique_inspections,
        indexed_orientation_count=orientation_count,
        distance_threshold=effective_policy.distance_threshold,
        quality_threshold=effective_policy.quality_threshold,
    )


def _pack_hashes(fingerprints: tuple[PDQFingerprint, ...]) -> array:
    packed = array("Q")
    for fingerprint in fingerprints:
        for hash_value in fingerprint.hash_integers:
            packed.extend(
                (hash_value >> (64 * word_index)) & _UINT64_MASK
                for word_index in range(4)
            )
    if packed.itemsize != 8:
        raise RuntimeError("platform uint64 array representation is unsupported")
    return packed


def _count_buckets(packed_hashes: array, orientation_count: int) -> array:
    counts = array("I", [0]) * MIH_BUCKET_COUNT
    if counts.itemsize != 4:
        raise RuntimeError("platform uint32 array representation is unsupported")
    for orientation_id in range(orientation_count):
        for slot_index in range(MIH_SLOT_COUNT):
            bucket = (
                slot_index * MIH_SLOT_VALUES
                + _packed_slot(packed_hashes, orientation_id, slot_index)
            )
            counts[bucket] += 1
    return counts


def _preflight_raw_posting_visits(counts: array, *, maximum: int) -> int:
    total = 0
    for slot_index in range(MIH_SLOT_COUNT):
        base = slot_index * MIH_SLOT_VALUES
        for value in range(MIH_SLOT_VALUES):
            count = counts[base + value]
            if count == 0:
                continue
            total += count * count
            for bit_index in range(MIH_SLOT_BITS):
                total += count * counts[base + (value ^ (1 << bit_index))]
            if total > maximum:
                raise PDQCapacityExceeded(
                    "PDQ raw-posting preflight cap exceeded"
                )
    return total


def _build_csr_index(
    packed_hashes: array, orientation_count: int, counts: array
) -> tuple[array, array]:
    offsets = array("I", [0])
    running = 0
    for count in counts:
        running += count
        offsets.append(running)
    expected_postings = orientation_count * MIH_SLOT_COUNT
    if running != expected_postings:
        raise AssertionError("PDQ bucket counts do not cover every orientation slot")
    postings = array("I", [0]) * expected_postings
    # Reuse the count array as one write cursor per bucket; no second cursor
    # allocation is retained at peak build memory.
    for bucket in range(MIH_BUCKET_COUNT):
        counts[bucket] = offsets[bucket]
    for orientation_id in range(orientation_count):
        for slot_index in range(MIH_SLOT_COUNT):
            bucket = (
                slot_index * MIH_SLOT_VALUES
                + _packed_slot(packed_hashes, orientation_id, slot_index)
            )
            position = counts[bucket]
            postings[position] = orientation_id
            counts[bucket] = position + 1
    return offsets, postings


def _packed_slot(packed_hashes: array, orientation_id: int, slot_index: int) -> int:
    word = packed_hashes[orientation_id * 4 + slot_index // 4]
    return (word >> ((slot_index % 4) * MIH_SLOT_BITS)) & _SLOT_MASK


def _packed_hamming_distance(
    packed_hashes: array, left_orientation_id: int, right_orientation_id: int
) -> int:
    left_base = left_orientation_id * 4
    right_base = right_orientation_id * 4
    return sum(
        (packed_hashes[left_base + offset] ^ packed_hashes[right_base + offset])
        .bit_count()
        for offset in range(4)
    )


def _exact_and_one_bit_flips(value: int) -> Iterable[int]:
    yield value
    for bit_index in range(MIH_SLOT_BITS):
        yield value ^ (1 << bit_index)


assert MIH_SLOT_COUNT * MIH_SLOT_BITS == 256
assert MIH_KEYS_PER_ORIENTATION == 272
