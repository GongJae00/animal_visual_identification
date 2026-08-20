"""Label-blind perceptual hashes and exact-complete bounded MIH search.

This module deliberately operates below the semantic-manifest layer.  A caller
must replace its source sample identifier with an opaque SHA-256 token and must
provide an already decoded, deterministically resized 32 x 32 luma raster.
Identity, split, sequence, camera, cage, and path fields are not accepted.

The pHash is defined by 64 fixed low-frequency AC coefficients of an
orthonormal 32 x 32 DCT-II.  The coefficient order is the literal zig-zag table
``PHASH_COEFFICIENTS``; DC ``(0, 0)`` is absent.  Pixels are mean-centred before
the DCT to make DC exclusion numerically stable.  A bit is one exactly when its
coefficient is strictly greater than the median; median ties therefore encode
as zero.  The horizontal-flip hash is derived from the DCT identity
``F_flip(u, v) = (-1)**u F(u, v)`` instead of running a second transform.

No NumPy or SciPy dependency is required.  The standard-library implementation
uses a separable transform and evaluates only the horizontal frequencies used
by the 64-bit descriptor.  Fingerprinting is O(32^2 * U + 32 * 64), where U is
the eleven distinct horizontal frequencies in the fixed table, and requires
O(32 * U) working floats.

Candidate search indexes original and horizontal-flip hashes in four 16-bit
blocks.  Each query enumerates all 137 block values at Hamming distance at most
two.  For any two 64-bit values at distance at most ten, the pigeonhole
principle guarantees that at least one of four blocks differs by at most two;
the subsequent full-distance check therefore makes the search exact-complete
for radii 0..10.  Sparse-bucket work is O(1096n + C), while adversarial bucket
collisions remain O(n^2) in the worst case.  Pair deduplication is query-local:
an unordered pair can only be inspected while its lexicographically larger ID
is the query, so a global provisional-pair set is unnecessary.  Memory is
O(n + A), where ``A`` is the accepted-candidate cap, plus one O(n) query-local
set; work fails closed at a separate integer-only inspection cap.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


LUMA_SIDE = 32
PHASH_BITS = 64
MIH_BLOCKS = 4
MIH_BLOCK_BITS = 16
MIH_PER_BLOCK_RADIUS = 2
MAXIMUM_EXACT_RADIUS = 10

# Coordinates are (horizontal frequency u, vertical frequency v).  This is the
# first 64 non-DC positions of one fixed diagonal zig-zag traversal.  Keeping
# the table literal makes a coefficient-order change an explicit schema change.
PHASH_COEFFICIENTS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1),
    (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5), (1, 4),
    (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3),
    (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4), (4, 3),
    (5, 2), (6, 1), (7, 0), (8, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (0, 8), (0, 9), (1, 8), (2, 7), (3, 6),
    (4, 5), (5, 4), (6, 3), (7, 2), (8, 1), (9, 0), (10, 0), (9, 1),
    (8, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (2, 8), (1, 9),
)

_OPAQUE_ID = re.compile(r"[0-9a-f]{64}\Z")
_UINT64_LIMIT = 1 << PHASH_BITS
_BLOCK_MASK = (1 << MIH_BLOCK_BITS) - 1
_USED_HORIZONTAL_FREQUENCIES = tuple(
    sorted({coordinate[0] for coordinate in PHASH_COEFFICIENTS})
)


def _dct_basis(frequency: int) -> tuple[float, ...]:
    scale = math.sqrt(1.0 / LUMA_SIDE) if frequency == 0 else math.sqrt(
        2.0 / LUMA_SIDE
    )
    return tuple(
        scale
        * math.cos(math.pi * (2 * position + 1) * frequency / (2 * LUMA_SIDE))
        for position in range(LUMA_SIDE)
    )


_DCT_BASIS: tuple[tuple[float, ...], ...] = tuple(
    _dct_basis(frequency) for frequency in range(LUMA_SIDE)
)


class CandidateLimitExceeded(RuntimeError):
    """Raised before search exceeds its declared work or output-memory budget."""


class FingerprintLimitExceeded(RuntimeError):
    """Raised before the linear MIH index can exceed its input-count budget."""


@dataclass(frozen=True, slots=True)
class PHashFingerprint:
    """Two orientation fingerprints joined only to an opaque sample token."""

    opaque_sample_id: str
    original_hash: int
    horizontal_flip_hash: int
    schema_version: str = "data.phash_fingerprint.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "data.phash_fingerprint.v1":
            raise ValueError("unsupported pHash fingerprint schema")
        _require_opaque_id(self.opaque_sample_id, "opaque sample ID")
        _require_uint64(self.original_hash, "original_hash")
        _require_uint64(self.horizontal_flip_hash, "horizontal_flip_hash")

    @property
    def unique_hashes(self) -> tuple[int, ...]:
        """Return sorted unique orientations to bound redundant index entries."""

        return tuple(sorted({self.original_hash, self.horizontal_flip_hash}))


@dataclass(frozen=True, slots=True)
class NearDuplicateCandidate:
    """Opaque pair evidence only; semantic metadata must be joined elsewhere."""

    left_opaque_sample_id: str
    right_opaque_sample_id: str
    hamming_distance: int
    schema_version: str = "data.phash_near_duplicate_candidate.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "data.phash_near_duplicate_candidate.v1":
            raise ValueError("unsupported near-duplicate candidate schema")
        for value in (self.left_opaque_sample_id, self.right_opaque_sample_id):
            _require_opaque_id(value, "candidate sample ID")
        if self.left_opaque_sample_id >= self.right_opaque_sample_id:
            raise ValueError("candidate IDs must be strictly increasing")
        if (
            isinstance(self.hamming_distance, bool)
            or not isinstance(self.hamming_distance, int)
            or not 0 <= self.hamming_distance <= MAXIMUM_EXACT_RADIUS
        ):
            raise ValueError("candidate Hamming distance is outside 0..10")


def opaque_sample_id(value: str | bytes) -> str:
    """Domain-separate a private source identifier before label-blind search."""

    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, bytes):
        payload = value
    else:
        raise TypeError("source identifier must be str or bytes")
    return hashlib.sha256(b"PHASH_OPAQUE_SAMPLE_V1\0" + payload).hexdigest()


def fingerprint_luma32(
    *, opaque_id: str, luma_pixels: bytes
) -> PHashFingerprint:
    """Compute original and horizontal-flip pHashes from 1024 luma bytes."""

    _require_opaque_id(opaque_id, "opaque sample ID")
    if not isinstance(luma_pixels, bytes):
        raise TypeError("luma pixels must be immutable bytes")
    if len(luma_pixels) != LUMA_SIDE * LUMA_SIDE:
        raise ValueError("luma raster must contain exactly 32 x 32 bytes")

    mean = math.fsum(luma_pixels) / len(luma_pixels)
    horizontal: dict[tuple[int, int], float] = {}
    for row in range(LUMA_SIDE):
        offset = row * LUMA_SIDE
        for frequency in _USED_HORIZONTAL_FREQUENCIES:
            basis = _DCT_BASIS[frequency]
            horizontal[(row, frequency)] = math.fsum(
                (luma_pixels[offset + column] - mean) * basis[column]
                for column in range(LUMA_SIDE)
            )

    coefficients = tuple(
        math.fsum(
            horizontal[(row, horizontal_frequency)]
            * _DCT_BASIS[vertical_frequency][row]
            for row in range(LUMA_SIDE)
        )
        for horizontal_frequency, vertical_frequency in PHASH_COEFFICIENTS
    )
    flipped_coefficients = tuple(
        -value if horizontal_frequency % 2 else value
        for value, (horizontal_frequency, _) in zip(
            coefficients, PHASH_COEFFICIENTS, strict=True
        )
    )
    return PHashFingerprint(
        opaque_sample_id=opaque_id,
        original_hash=_threshold_coefficients(coefficients),
        horizontal_flip_hash=_threshold_coefficients(flipped_coefficients),
    )


def hamming_distance(left: int, right: int) -> int:
    """Return the exact distance between two validated 64-bit hashes."""

    _require_uint64(left, "left")
    _require_uint64(right, "right")
    return (left ^ right).bit_count()


def find_near_duplicate_candidates(
    fingerprints: Sequence[PHashFingerprint] | Iterable[PHashFingerprint],
    *,
    radius: int = MAXIMUM_EXACT_RADIUS,
    maximum_pair_inspections: int,
    maximum_accepted_candidates: int,
    maximum_fingerprints: int = 100_000,
) -> tuple[NearDuplicateCandidate, ...]:
    """Find every orientation-aware pair within ``radius``, or fail closed.

    ``maximum_pair_inspections`` bounds unique sample pairs emitted by the MIH
    buckets before full Hamming filtering.  Only an integer work counter is
    global; duplicate bucket hits are removed by one query-local ID set.
    ``maximum_accepted_candidates`` separately bounds result-list memory.
    """

    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("radius must be an integer")
    if not 0 <= radius <= MAXIMUM_EXACT_RADIUS:
        raise ValueError("radius must be inside the exact-complete range 0..10")
    if (
        isinstance(maximum_pair_inspections, bool)
        or not isinstance(maximum_pair_inspections, int)
        or maximum_pair_inspections <= 0
    ):
        raise ValueError("maximum pair inspections must be a positive integer")
    if (
        isinstance(maximum_accepted_candidates, bool)
        or not isinstance(maximum_accepted_candidates, int)
        or maximum_accepted_candidates <= 0
    ):
        raise ValueError("maximum accepted candidates must be a positive integer")
    if (
        isinstance(maximum_fingerprints, bool)
        or not isinstance(maximum_fingerprints, int)
        or maximum_fingerprints <= 0
    ):
        raise ValueError("maximum fingerprints must be a positive integer")

    materialized_list: list[PHashFingerprint] = []
    for item in fingerprints:
        if not isinstance(item, PHashFingerprint):
            raise TypeError("all inputs must be PHashFingerprint instances")
        if len(materialized_list) >= maximum_fingerprints:
            raise FingerprintLimitExceeded("pHash fingerprint cap exceeded")
        materialized_list.append(item)
    materialized = tuple(materialized_list)
    ordered = tuple(sorted(materialized, key=lambda item: item.opaque_sample_id))
    by_id = {item.opaque_sample_id: item for item in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("fingerprints contain duplicate opaque sample IDs")

    index: dict[tuple[int, int], list[str]] = {}
    for item in ordered:
        for fingerprint in item.unique_hashes:
            for block_index, block_value in enumerate(_blocks(fingerprint)):
                index.setdefault((block_index, block_value), []).append(
                    item.opaque_sample_id
                )

    accepted: list[NearDuplicateCandidate] = []
    inspection_count = 0
    for item in ordered:
        seen_other_ids: set[str] = set()
        for fingerprint in item.unique_hashes:
            for block_index, block_value in enumerate(_blocks(fingerprint)):
                for neighbor in _block_neighbors_within_two(block_value):
                    for other_id in index.get((block_index, neighbor), ()):
                        if other_id >= item.opaque_sample_id:
                            continue
                        if other_id in seen_other_ids:
                            continue
                        if inspection_count >= maximum_pair_inspections:
                            raise CandidateLimitExceeded(
                                "pHash MIH pair-inspection cap exceeded"
                            )
                        inspection_count += 1
                        seen_other_ids.add(other_id)
                        distance = _minimum_orientation_distance(
                            by_id[other_id], item
                        )
                        if distance <= radius:
                            if len(accepted) >= maximum_accepted_candidates:
                                raise CandidateLimitExceeded(
                                    "pHash MIH accepted-candidate cap exceeded"
                                )
                            accepted.append(
                                NearDuplicateCandidate(
                                    left_opaque_sample_id=other_id,
                                    right_opaque_sample_id=item.opaque_sample_id,
                                    hamming_distance=distance,
                                )
                            )
    return tuple(sorted(
        accepted,
        key=lambda item: (
            item.left_opaque_sample_id,
            item.right_opaque_sample_id,
        ),
    ))


def _threshold_coefficients(coefficients: tuple[float, ...]) -> int:
    if len(coefficients) != PHASH_BITS:
        raise ValueError("pHash requires exactly 64 DCT coefficients")
    ordered = sorted(coefficients)
    median = (ordered[31] + ordered[32]) / 2.0
    result = 0
    for bit_index, value in enumerate(coefficients):
        if value > median:
            result |= 1 << bit_index
    return result


def _blocks(fingerprint: int) -> tuple[int, int, int, int]:
    return tuple(
        (fingerprint >> (block_index * MIH_BLOCK_BITS)) & _BLOCK_MASK
        for block_index in range(MIH_BLOCKS)
    )  # type: ignore[return-value]


def _block_neighbors_within_two(value: int) -> Iterable[int]:
    yield value
    for first in range(MIH_BLOCK_BITS):
        yield value ^ (1 << first)
    for first in range(MIH_BLOCK_BITS):
        for second in range(first + 1, MIH_BLOCK_BITS):
            yield value ^ (1 << first) ^ (1 << second)


def _minimum_orientation_distance(
    left: PHashFingerprint, right: PHashFingerprint
) -> int:
    return min(
        (left_hash ^ right_hash).bit_count()
        for left_hash in left.unique_hashes
        for right_hash in right.unique_hashes
    )


def _require_uint64(value: int, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < _UINT64_LIMIT
    ):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")


def _require_opaque_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _OPAQUE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 token")


assert len(PHASH_COEFFICIENTS) == PHASH_BITS
assert len(set(PHASH_COEFFICIENTS)) == PHASH_BITS
assert (0, 0) not in PHASH_COEFFICIENTS
assert MIH_BLOCKS * MIH_BLOCK_BITS == PHASH_BITS
