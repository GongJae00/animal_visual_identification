from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import identity_retrieval.gallery as gallery_module
from identity_governance.generated_identity_registry import create_provisional_identity
from identity_governance.identity_registry import compute_registered_dog_id
from identity_retrieval.gallery import (
    GalleryEnrollment,
    IdentityGallery,
    IdentityRegistryPolicy,
)
from identity_retrieval.qkv import (
    SCORER_ALGORITHM,
    AvailableIntersectionScorer,
    EnrollmentRank,
    EvidenceChannelSpec,
    GalleryKey,
    GalleryValue,
    IdentityEvidenceKind,
    QueryExclusions,
    QueryKeyScore,
    RetrievalQuery,
    ScoredGalleryValue,
    aggregate_identity_matches,
    canonical_channel_weights,
)

VectorSetConstructor = Callable[
    [dict[str, np.ndarray], dict[str, bool]], RetrievalQuery | GalleryKey
]


def _query(vectors: dict[str, np.ndarray], availability: dict[str, bool]) -> RetrievalQuery:
    return RetrievalQuery(vectors=vectors, availability=availability)


def _key(vectors: dict[str, np.ndarray], availability: dict[str, bool]) -> GalleryKey:
    return GalleryKey(template_row=0, vectors=vectors, availability=availability)


@pytest.mark.parametrize(
    ("channel_count", "values"),
    [
        (True, None),
        (0, None),
        (2, [1.0]),
        (2, [1.0, -1.0]),
        (2, [1.0, float("nan")]),
        (2, [1.0, float("inf")]),
        (2, [0.0, 0.0]),
    ],
)
def test_canonical_channel_weights_reject_invalid_inputs(
    channel_count: int, values: list[float] | None
) -> None:
    with pytest.raises(ValueError):
        canonical_channel_weights(channel_count, values)


def test_canonical_channel_weights_normalize_exactly_to_float32() -> None:
    weights = canonical_channel_weights(3, [1.0, 2.0, 3.0])

    assert weights.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        weights,
        np.asarray([1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0], dtype=np.float32),
    )


@pytest.mark.parametrize("constructor", [_query, _key])
@pytest.mark.parametrize(
    ("vectors", "availability"),
    [
        ({"shape": np.asarray([1.0, 0.0], dtype=np.float32)}, {"shape": False}),
        ({}, {"shape": True}),
    ],
)
def test_query_and_key_reject_availability_vector_mismatch(
    constructor: VectorSetConstructor,
    vectors: dict[str, np.ndarray],
    availability: dict[str, bool],
) -> None:
    with pytest.raises(ValueError, match="vectors must exactly match available channels"):
        constructor(vectors, availability)


def test_gallery_key_rejects_non_unit_vectors() -> None:
    with pytest.raises(ValueError, match="must be a unit float32 vector"):
        _key(
            {"shape": np.asarray([2.0, 0.0], dtype=np.float32)},
            {"shape": True},
        )


def test_retrieval_query_canonicalizes_vectors_once() -> None:
    query = _query(
        {"shape": np.asarray([2.0, 0.0], dtype=np.float32)},
        {"shape": True},
    )
    np.testing.assert_array_equal(
        query.vectors["shape"], np.asarray([1.0, 0.0], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        query.vectors["shape"].setflags(write=True)


def test_retrieval_query_snapshots_stateful_mappings_once() -> None:
    class ChangingVectors(dict[str, np.ndarray]):
        calls = 0

        def items(self):
            self.calls += 1
            if self.calls > 1:
                return {"shape": np.asarray([np.nan, np.nan], np.float32)}.items()
            return super().items()

    vectors = ChangingVectors(
        {"shape": np.asarray([1.0, 0.0], dtype=np.float32)}
    )
    query = _query(vectors, {"shape": True})
    assert vectors.calls == 1
    np.testing.assert_array_equal(
        query.vectors["shape"], np.asarray([1.0, 0.0], dtype=np.float32)
    )


def test_available_intersection_scorer_computes_weighted_cosine_and_availability() -> None:
    scorer = AvailableIntersectionScorer(
        (
            EvidenceChannelSpec("shape", dimension=2, optional=False, weight=2.0),
            EvidenceChannelSpec("coat", dimension=2, optional=True, weight=1.0),
            EvidenceChannelSpec("nose", dimension=2, optional=True, weight=4.0),
        )
    )
    query = RetrievalQuery(
        vectors={
            "shape": np.asarray([1.0, 0.0], dtype=np.float32),
            "coat": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        availability={"shape": True, "coat": True, "nose": False},
    )
    key = GalleryKey(
        template_row=7,
        vectors={
            "shape": np.asarray([0.0, 1.0], dtype=np.float32),
            "coat": np.asarray([1.0, 0.0], dtype=np.float32),
            "nose": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        availability={"shape": True, "coat": True, "nose": True},
    )

    result = scorer.score(query, key)

    assert result.similarity == 1.0 / 3.0
    assert result.evidence == {"shape": 0.0, "coat": 1.0}
    assert result.evidence_availability == {
        "shape": True,
        "coat": True,
        "nose": False,
    }


def test_scorer_hash_is_deterministic_and_channel_order_sensitive() -> None:
    channels = (
        EvidenceChannelSpec("shape", dimension=2, optional=False, weight=2.0),
        EvidenceChannelSpec("coat", dimension=3, optional=True, weight=1.0),
    )

    scorer_hash = AvailableIntersectionScorer(channels).scorer_hash

    assert scorer_hash == AvailableIntersectionScorer(channels).scorer_hash
    assert scorer_hash != AvailableIntersectionScorer(tuple(reversed(channels))).scorer_hash


def test_scorer_hash_preserves_historical_v4_gallery_contract() -> None:
    channels = (
        EvidenceChannelSpec(
            "shape", dimension=2, optional=False, weight=float(np.float32(2 / 3))
        ),
        EvidenceChannelSpec(
            "coat", dimension=3, optional=True, weight=float(np.float32(1 / 3))
        ),
    )

    assert AvailableIntersectionScorer(channels).scorer_hash == (
        "346cf22136a4221d74f6bacb77fbea2f060453cec9cbd2962851dccfded3778b"
    )


def _candidate(
    *, identity_suffix: int, template_digit: str, score: float, template_row: int
) -> ScoredGalleryValue:
    identity_id = f"00000000-0000-5000-8000-{identity_suffix:012d}"
    return ScoredGalleryValue(
        value=GalleryValue(
            template_row=template_row,
            registered_identity_id=identity_id,
            template_id=template_digit * 64,
            content_sha256=f"{template_row + 1:064x}",
            idempotency_key=f"cvi.test.qkv.idempotency.{template_row:04d}",
            template_schema="cvi.test.gallery_template.v1",
            breed="test-breed",
            metadata={"fixture": "retrieval-qkv"},
        ),
        query_key_score=QueryKeyScore(
            similarity=score,
            evidence={"shape": score},
            evidence_availability={"shape": True},
        ),
        template_availability={"shape": True},
    )


def test_aggregate_identity_matches_selects_templates_and_orders_identities() -> None:
    candidates = [
        _candidate(identity_suffix=2, template_digit="c", score=0.8, template_row=0),
        _candidate(identity_suffix=2, template_digit="b", score=0.9, template_row=1),
        _candidate(identity_suffix=1, template_digit="d", score=0.9, template_row=2),
        _candidate(identity_suffix=3, template_digit="e", score=0.95, template_row=3),
        _candidate(identity_suffix=2, template_digit="a", score=0.9, template_row=4),
        _candidate(identity_suffix=4, template_digit="f", score=0.7, template_row=5),
    ]

    matches = aggregate_identity_matches(candidates, top_k=3)

    assert [match.value.registered_identity_id for match in matches] == [
        "00000000-0000-5000-8000-000000000003",
        "00000000-0000-5000-8000-000000000001",
        "00000000-0000-5000-8000-000000000002",
    ]
    assert matches[2].value.template_id == "a" * 64


def _full128_contract() -> dict[str, object]:
    return {
        "schema_version": "cvi.gallery_embedding_contract.v1",
        "kind": "Full128",
        "dimension": 128,
        "channels": [
            {"name": "Full128", "dimension": 128, "optional": False}
        ],
        "fusion": {"type": SCORER_ALGORITHM, "weights": [1.0]},
    }


def _registered(label: str) -> str:
    return compute_registered_dog_id(f"fixture:v1:qkv-gallery:{label}")


def test_full128_exact_identity_search_matches_brute_force_with_imbalance_and_ties(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(71)
    gallery = IdentityGallery(tmp_path, dim=128, embedding_contract=_full128_contract())
    rows: list[tuple[str, np.ndarray, int]] = []
    try:
        for identity_number, template_count in enumerate((1, 2, 7, 19)):
            identity_id = _registered(f"parity-{identity_number}")
            for template_number in range(template_count):
                vector = rng.normal(size=128).astype(np.float32)
                row = gallery.enroll(
                    vector,
                    identity_id,
                    content_sha256=f"{len(rows) + 1:064x}",
                )
                rows.append((identity_id, vector / np.linalg.norm(vector), row))

        tied_ids = (_registered("tie-a"), _registered("tie-b"))
        tied_vector = np.zeros(128, dtype=np.float32)
        tied_vector[0] = 1.0
        for offset, identity_id in enumerate(reversed(tied_ids), start=1):
            row = gallery.enroll(
                tied_vector,
                identity_id,
                content_sha256=f"{len(rows) + offset:064x}",
            )
            rows.append((identity_id, tied_vector, row))

        query = tied_vector.copy()
        brute: dict[str, tuple[float, int]] = {}
        for identity_id, vector, row in rows:
            score = float(np.dot(query, vector))
            current = brute.get(identity_id)
            if current is None or score > current[0]:
                brute[identity_id] = (score, row)
        expected = sorted(brute, key=lambda value: (-brute[value][0], value))

        results = gallery.search(query, top_k=len(expected))

        assert [result[2]["registered_dog_id"] for result in results] == expected
        assert [
            gallery.rank_of_identity(query, identity_id)
            for identity_id in expected
        ] == list(range(1, len(expected) + 1))
        np.testing.assert_allclose(
            [result[1] for result in results],
            [brute[identity_id][0] for identity_id in expected],
            atol=1e-6,
        )
        assert [result[2]["registered_dog_id"] for result in results[:2]] == sorted(
            tied_ids
        )
        assert all(result[0] == result[2]["_winning_template_row"] for result in results)
    finally:
        gallery.close()


def test_query_excludes_template_content_and_duplicate_groups_before_identity_max(
    tmp_path: Path,
) -> None:
    gallery = IdentityGallery(tmp_path, dim=128, embedding_contract=_full128_contract())
    query = np.ones(128, dtype=np.float32)
    first_id = _registered("excluded-first")
    second_id = _registered("excluded-second")
    try:
        first_row = gallery.enroll(
            query,
            first_id,
            content_sha256="1" * 64,
            enrollment_rank=EnrollmentRank.K1,
            enrollment_view="front",
            duplicate_group_ids=("duplicate:one",),
        )
        second_row = gallery.enroll(
            query * 0.9 + np.eye(1, 128, 1, dtype=np.float32)[0],
            first_id,
            content_sha256="2" * 64,
            enrollment_rank=EnrollmentRank.K3,
            enrollment_view="left",
        )
        gallery.enroll(
            -query,
            second_id,
            content_sha256="3" * 64,
            enrollment_rank=EnrollmentRank.K5,
            enrollment_view="right",
        )

        template_excluded = gallery.search(
            query,
            top_k=2,
            exclusions=QueryExclusions(
                template_ids=frozenset({gallery._metadata[first_row]["template_id"]})
            ),
        )
        assert template_excluded[0][0] == second_row

        all_first_excluded = gallery.search(
            query,
            top_k=2,
            exclusions=QueryExclusions(
                content_sha256s=frozenset({"2" * 64}),
                duplicate_group_ids=frozenset({"duplicate:one"}),
            ),
        )
        assert [row[2]["registered_dog_id"] for row in all_first_excluded] == [
            second_id
        ]
        assert gallery.rank_of_identity(
            query,
            first_id,
            exclusions=QueryExclusions(
                content_sha256s=frozenset({"2" * 64}),
                duplicate_group_ids=frozenset({"duplicate:one"}),
            ),
        ) is None
        assert gallery.rank_of_identity(
            query,
            second_id,
            exclusions=QueryExclusions(
                content_sha256s=frozenset({"2" * 64}),
                duplicate_group_ids=frozenset({"duplicate:one"}),
            ),
        ) == 1
    finally:
        gallery.close()


def test_registered_only_policy_rejects_explicit_and_registry_known_genids(
    tmp_path: Path,
) -> None:
    provisional = create_provisional_identity("fixture-generator", "cluster-1", 3)
    policy = IdentityRegistryPolicy(
        provisional_generated_identity_ids=frozenset(
            {provisional.generated_identity_id}
        )
    )
    gallery = IdentityGallery(
        tmp_path, dim=128, embedding_contract=_full128_contract(), registry_policy=policy
    )
    vector = np.ones(128, dtype=np.float32)
    try:
        with pytest.raises(ValueError, match="provisional GenID"):
            gallery.enroll(
                vector,
                provisional.generated_identity_id,
                identity_evidence_kind=IdentityEvidenceKind.PROVISIONAL_GENID,
            )
        with pytest.raises(ValueError, match="registry-known provisional GenID"):
            gallery.enroll(vector, provisional.generated_identity_id)
        assert gallery.size == 0
    finally:
        gallery.close()


def test_enrollment_uniqueness_uses_indexes_instead_of_metadata_scan(
    tmp_path: Path,
) -> None:
    class NoItemsDict(dict[int, dict[str, object]]):
        def items(self):
            raise AssertionError("enrollment must not scan all metadata rows")

    gallery = IdentityGallery(tmp_path, dim=128, embedding_contract=_full128_contract())
    try:
        gallery.enroll(
            np.eye(1, 128, 0, dtype=np.float32)[0],
            _registered("indexed-1"),
            content_sha256="1" * 64,
        )
        gallery._metadata = NoItemsDict(gallery._metadata)
        gallery.enroll(
            np.eye(1, 128, 1, dtype=np.float32)[0],
            _registered("indexed-2"),
            content_sha256="2" * 64,
        )
        assert gallery.size == 2
    finally:
        gallery.close()


def test_bulk_k_view_generation_and_block_size_are_batch_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        GalleryEnrollment(
            embedding=np.eye(1, 128, index, dtype=np.float32)[0],
            registered_identity_id=_registered(f"bulk-{index % 2}"),
            idempotency_key=f"bulk-request-{index}",
            content_sha256=f"{index + 1:064x}",
            enrollment_rank=(
                EnrollmentRank.K1,
                EnrollmentRank.K3,
                EnrollmentRank.K5,
            )[index % 3],
            enrollment_view=f"view-{index}",
        )
        for index in range(6)
    ]
    first = IdentityGallery.build(
        tmp_path / "first", records, dim=128, embedding_contract=_full128_contract()
    )
    second = IdentityGallery.build(
        tmp_path / "second",
        list(reversed(records)),
        dim=128,
        embedding_contract=_full128_contract(),
    )
    try:
        first_manifest = json.loads(
            (tmp_path / "first" / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        second_manifest = json.loads(
            (tmp_path / "second" / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        assert first_manifest["schema_version"] == "cvi.gallery_manifest.v5"
        assert {
            key: value["sha256"] for key, value in first_manifest["files"].items()
        } == {
            key: value["sha256"] for key, value in second_manifest["files"].items()
        }

        query = np.ones(128, dtype=np.float32)
        monkeypatch.setattr(gallery_module, "_SEARCH_BLOCK_ROWS", 1)
        one_row_blocks = first.search(query, top_k=2)
        monkeypatch.setattr(gallery_module, "_SEARCH_BLOCK_ROWS", 65_536)
        large_blocks = first.search(query, top_k=2)
        assert one_row_blocks == large_blocks
        assert {
            row[2]["_enrollment_rank"] for row in large_blocks
        } <= {"K1", "K3", "K5"}
    finally:
        first.close()
        second.close()


def test_bulk_build_failure_publishes_no_partial_gallery(tmp_path: Path) -> None:
    destination = tmp_path / "failed-build"
    invalid = GalleryEnrollment(
        embedding=np.ones(128, dtype=np.float32),
        registered_identity_id=_registered("invalid-bulk-kind"),
        identity_evidence_kind=IdentityEvidenceKind.PROVISIONAL_GENID,
    )

    with pytest.raises(ValueError, match="provisional GenID"):
        IdentityGallery.build(
            destination,
            [invalid],
            dim=128,
            embedding_contract=_full128_contract(),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-build.build-*"))


def test_v5_corruption_cannot_change_registered_evidence_to_provisional(
    tmp_path: Path,
) -> None:
    gallery = IdentityGallery(tmp_path, dim=128, embedding_contract=_full128_contract())
    gallery.enroll(
        np.ones(128, dtype=np.float32),
        _registered("corruption"),
        enrollment_rank=EnrollmentRank.K1,
        enrollment_view="front",
    )
    gallery.save()
    gallery.close()

    manifest_path = tmp_path / "gallery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = tmp_path / manifest["files"]["metadata"]["name"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["0"]["identity_evidence_kind"] = "PROVISIONAL_GENID"
    payload = (
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    metadata_path.write_bytes(payload)
    manifest["files"]["metadata"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="registry policy"):
        IdentityGallery(
            tmp_path,
            dim=128,
            embedding_contract=_full128_contract(),
            read_only=True,
        )
