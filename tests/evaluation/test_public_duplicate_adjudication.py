from __future__ import annotations

import unittest
from unittest.mock import PropertyMock, patch

from evaluation.splits.public_duplicate_adjudication import (
    AdjudicationLedger,
    AdjudicationMode,
    CandidateAdjudication,
    CandidateOutcome,
    assemble_frozen_evidence_graph,
    build_adjudication_chunk,
    build_exact_duplicate_graph,
    build_review_queue,
    merge_adjudication_chunks,
)
from evaluation.splits import public_duplicate_adjudication as adjudication_module
from shared.foundation.provenance import content_sha256
from evaluation.splits.protected_public_split import (
    PublicSplitSample,
    PublicSplitSourceBundle,
)
from shared.contracts.pdq import (
    PDQNearDuplicateCandidate,
    PDQSearchPolicy,
    PDQSearchResult,
)
from data.audit.pdq.contracts import (
    PDQSearchPolicy as LegacyPDQSearchPolicy,
)

def _token(value: int) -> str:
    return f"{value:064x}"

def test_legacy_pdq_contract_import_preserves_type_identity() -> None:
    assert LegacyPDQSearchPolicy is PDQSearchPolicy

def _source() -> PublicSplitSourceBundle:
    bindings = tuple(sorted((name, _token(index + 100)) for index, name in enumerate((
        "exact_duplicate_graph_sha256",
        "geometric_verifier_sha256",
        "image_content_receipts_sha256",
        "pdq_candidates_sha256",
        "phash_candidates_sha256",
        "review_adjudication_sha256",
        "semantic_receipts_sha256",
    ))))
    samples = tuple(
        PublicSplitSample(
            sample_token=_token(index + 1),
            identity_token=_token(index + 10),
            sequence_token=_token(index + 20),
            source_sample_id=f"mpdd:v1:sample:{index}",
            dataset_identity_id=f"mpdd:v1:identity:{index}",
            dataset_name="mpdd",
            source_variant="original",
            original_split=None,
            raw_frame_index=0,
            paired_source_sample_id=None,
            in_no_mono_subset=None,
            region="DOG_CROP",
        )
        for index in range(3)
    )
    return PublicSplitSourceBundle(bindings, samples)

def _binding(source: PublicSplitSourceBundle) -> dict[str, object]:
    rows = [
        {
            "opaque_sample_id": _token(index + 50),
            "dataset_name": "mpdd",
            "source_sample_id": sample.source_sample_id,
        }
        for index, sample in enumerate(source.samples)
    ]
    binding = {"binding_count": len(rows), "bindings": rows}
    return {
        "schema_version": "cvi.public_canine_phash_binding_bundle.v1",
        "binding": binding,
        "binding_sha256": content_sha256(binding),
    }

def _images(source: PublicSplitSourceBundle) -> dict[str, object]:
    records = [
        {
            "source_sample_id": sample.source_sample_id,
            "dataset_name": "mpdd",
            "pixel_sha256": _token(500 if index < 2 else 501),
            "member_path": f"images/{index}.jpg",
            "container_member_path": None,
        }
        for index, sample in enumerate(source.samples)
    ]
    receipt = {
        "decision": "PASS_IMAGE_CONTENT_AUDIT",
        "interpretation": (
            "DECODE_AND_PIXEL_EXACT_DUPLICATE_EVIDENCE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION"
        ),
        "records": records,
        "exact_duplicate_groups": [{
            "schema_version": "cvi.pixel_exact_duplicate_group.v1",
            "pixel_sha256": _token(500),
            "source_sample_ids": sorted(
                sample.source_sample_id for sample in source.samples[:2]
            ),
        }],
    }
    policy = {"fixture": True}
    provenance = {"fixture": True}
    return {"mpdd": {
        "schema_version": "cvi.image_content_audit_bundle.v1",
        "semantic_receipt_sha256": _token(600),
        "policy": policy,
        "policy_sha256": content_sha256(policy),
        "receipt": receipt,
        "receipt_sha256": content_sha256(receipt),
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
    }}

def _phash() -> dict[str, object]:
    candidates = [
        {
            "schema_version": "cvi.phash_near_duplicate_candidate.v1",
            "left_opaque_sample_id": _token(50),
            "right_opaque_sample_id": _token(51),
            "hamming_distance": 0,
        },
        {
            "schema_version": "cvi.phash_near_duplicate_candidate.v1",
            "left_opaque_sample_id": _token(51),
            "right_opaque_sample_id": _token(52),
            "hamming_distance": 8,
        },
    ]
    evidence = {
        "fingerprint_count": 3,
        "candidate_count": 2,
        "candidates": candidates,
    }
    return {
        "schema_version": "cvi.public_canine_phash_evidence_bundle.v1",
        "evidence": evidence,
        "evidence_sha256": content_sha256(evidence),
    }

def _pdq() -> dict[str, object]:
    candidate = PDQNearDuplicateCandidate(
        left_opaque_sample_id=_token(50),
        right_opaque_sample_id=_token(52),
        minimum_hamming_distance=7,
        left_orientation="ORIGINAL",
        right_orientation="ROT90CCW",
        left_quality=80,
        right_quality=90,
        minimum_quality=80,
        distance_threshold=31,
        quality_threshold=50,
    )
    search = PDQSearchResult(
        candidates=(candidate,),
        eligible_sample_ids=tuple(_token(index) for index in range(50, 53)),
        ineligible_low_quality_sample_ids=(),
        preflight_raw_posting_visits=10,
        unique_orientation_inspections=5,
        indexed_orientation_count=24,
        distance_threshold=31,
        quality_threshold=50,
    )
    policy = PDQSearchPolicy().to_dict()
    evidence = {
        "schema_version": "cvi.public_canine_pdq_evidence.v1",
        "search_result": search.to_dict(),
        "sample_ids_sha256": content_sha256(list(search.eligible_sample_ids)),
        "fingerprint_count": 3,
        "fingerprint_manifest_sha256": _token(701),
        "source_spec_sha256": _token(702),
        "source_receipt_bindings_sha256": _token(703),
        "native_build_receipt_sha256": _token(704),
        "native_binary_sha256": _token(705),
        "official_regression_receipt_sha256": _token(706),
        "policy": policy,
        "policy_sha256": content_sha256(policy),
        "decision": "PASS_BOUNDED_LABEL_BLIND_PDQ_CANDIDATE_GENERATION",
        "interpretation": (
            "PDQ_SIMILARITY_CANDIDATES_ONLY_NOT_DUPLICATE_NONDUPLICATE_"
            "SPLIT_OR_MODEL_ADMISSION"
        ),
    }
    return {
        "schema_version": "cvi.public_canine_pdq_evidence_bundle.v1",
        "evidence": evidence,
        "evidence_sha256": content_sha256(evidence),
    }

class PublicDuplicateAdjudicationTests(unittest.TestCase):
    def test_candidate_set_streaming_hash_matches_canonical_list_hash(self) -> None:
        pair_channels = {
            (_token(1), _token(2)): {"PDQ": _token(10), "PHASH": _token(11)},
            (_token(2), _token(3)): {"EXACT": _token(12)},
        }
        ordered_pairs = tuple(sorted(pair_channels))
        expected = content_sha256([
            {
                "left_sample_token": pair[0],
                "right_sample_token": pair[1],
                "candidate_channels": sorted(pair_channels[pair]),
                "candidate_evidence_tokens": sorted(pair_channels[pair].values()),
            }
            for pair in ordered_pairs
        ])

        self.assertEqual(
            adjudication_module._candidate_set_sha256(ordered_pairs, pair_channels),
            expected,
        )

    def test_standard_graph_hashes_the_ledger_once(self) -> None:
        source = _source()
        evidence = _token(900)
        record = CandidateAdjudication(
            left_sample_token=source.samples[0].sample_token,
            right_sample_token=source.samples[1].sample_token,
            candidate_channels=("EXACT",),
            candidate_evidence_tokens=(evidence,),
            outcome=CandidateOutcome.EXACT_CONFIRMED,
            reason="AUTHENTICATED_CANONICAL_RGB_DIGEST_EQUAL",
            decision_evidence_tokens=(evidence,),
        )
        ledger = AdjudicationLedger(
            source_bundle_sha256=source.bundle_sha256,
            candidate_set_sha256=_token(901),
            evidence_bindings=source.evidence_bindings,
            records=(record,),
            outcome_counts=(("EXACT_CONFIRMED", 1),),
            global_blockers=(),
            promotion_status="READY_FOR_GRAPH_PROMOTION",
        )
        with patch.object(
            AdjudicationLedger,
            "ledger_sha256",
            new_callable=PropertyMock,
            return_value=_token(902),
        ) as ledger_sha256:
            graph = assemble_frozen_evidence_graph(source=source, ledger=ledger)

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(ledger_sha256.call_count, 1)

    def test_admitted_dinov2_filters_only_below_threshold_phash_only_pair(self) -> None:
        base = _source()
        binding = _binding(base)
        images = _images(base)
        bindings = dict(base.evidence_bindings)
        bindings["image_content_receipts_sha256"] = content_sha256(images)
        base = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        exact = build_exact_duplicate_graph(
            source=base,
            image_receipts=images,
            opaque_binding_bundle=binding,
        )
        phash = _phash()
        pdq = _pdq()
        bindings = dict(base.evidence_bindings)
        bindings.update({
            "exact_duplicate_graph_sha256": _token(800),
            "phash_candidates_sha256": content_sha256(phash),
            "pdq_candidates_sha256": content_sha256(pdq),
        })
        source = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        dino_rows = {
            (_token(50), _token(51)): (
                "CANDIDATE_COMPONENT_DEPENDENCY", None, _token(991)
            ),
            (_token(51), _token(52)): (
                "LEAKAGE_FILTER_REJECTION", 0.4, _token(992)
            ),
        }
        with patch(
            "evaluation.splits.public_duplicate_adjudication.validate_dinov2_filter_for_corpus",
            return_value=(_token(990), dino_rows),
        ):
            chunk = build_adjudication_chunk(
                source=source,
                exact_graph=exact,
                exact_graph_artifact_sha256=_token(800),
                phash_evidence_bundle=phash,
                pdq_evidence_bundle=pdq,
                dinov2_filter_evidence={"fixture": True},
                opaque_binding_bundle=binding,
                start_index=0,
                maximum_candidates=10,
                mode=AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER,
            )
        self.assertEqual(chunk.global_blockers, ())
        outcomes = {item.candidate_channels: item for item in chunk.records}
        self.assertEqual(
            outcomes[("PHASH",)].outcome,
            CandidateOutcome.PHASH_ONLY_REJECTED_BY_ADMITTED_DINOV2,
        )
        self.assertEqual(
            outcomes[("PDQ",)].outcome,
            CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY,
        )
        ledger = merge_adjudication_chunks((chunk,))
        graph = assemble_frozen_evidence_graph(source=source, ledger=ledger)
        self.assertEqual(len(graph.edges), 2)

    def test_dinov2_above_threshold_or_invalid_pair_remains_dependency(self) -> None:
        base = _source()
        binding = _binding(base)
        images = _images(base)
        bindings = dict(base.evidence_bindings)
        bindings["image_content_receipts_sha256"] = content_sha256(images)
        base = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        exact = build_exact_duplicate_graph(
            source=base,
            image_receipts=images,
            opaque_binding_bundle=binding,
        )
        phash, pdq = _phash(), _pdq()
        bindings = dict(base.evidence_bindings)
        bindings.update({
            "exact_duplicate_graph_sha256": _token(800),
            "phash_candidates_sha256": content_sha256(phash),
            "pdq_candidates_sha256": content_sha256(pdq),
        })
        source = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        rows = {
            (_token(50), _token(51)): (
                "CANDIDATE_COMPONENT_DEPENDENCY", None, _token(991)
            ),
            (_token(51), _token(52)): (
                "CANDIDATE_COMPONENT_DEPENDENCY", None, _token(992)
            ),
        }
        with patch(
            "evaluation.splits.public_duplicate_adjudication.validate_dinov2_filter_for_corpus",
            return_value=(_token(990), rows),
        ):
            chunk = build_adjudication_chunk(
                source=source,
                exact_graph=exact,
                exact_graph_artifact_sha256=_token(800),
                phash_evidence_bundle=phash,
                pdq_evidence_bundle=pdq,
                dinov2_filter_evidence={"fixture": True},
                opaque_binding_bundle=binding,
                start_index=0,
                maximum_candidates=10,
                mode=AdjudicationMode.DINOV2_TRANSFORM_FAMILY_FILTER,
            )
        phash_only = next(
            item for item in chunk.records if item.candidate_channels == ("PHASH",)
        )
        self.assertEqual(
            phash_only.outcome, CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY
        )
        graph = assemble_frozen_evidence_graph(
            source=source, ledger=merge_adjudication_chunks((chunk,))
        )
        self.assertEqual(len(graph.edges), 3)

    def test_admitted_pdq_negative_filters_only_phash_only_pairs(self) -> None:
        base = _source()
        binding = _binding(base)
        images = _images(base)
        bindings = dict(base.evidence_bindings)
        bindings["image_content_receipts_sha256"] = content_sha256(images)
        base = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        exact = build_exact_duplicate_graph(
            source=base,
            image_receipts=images,
            opaque_binding_bundle=binding,
        )
        phash = _phash()
        pdq = _pdq()
        bindings = dict(base.evidence_bindings)
        bindings.update({
            "exact_duplicate_graph_sha256": _token(800),
            "phash_candidates_sha256": content_sha256(phash),
            "pdq_candidates_sha256": content_sha256(pdq),
        })
        source = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        with patch(
            "evaluation.splits.public_duplicate_adjudication.validate_admission_for_corpus",
            return_value=_token(990),
        ):
            chunk = build_adjudication_chunk(
                source=source,
                exact_graph=exact,
                exact_graph_artifact_sha256=_token(800),
                phash_evidence_bundle=phash,
                pdq_evidence_bundle=pdq,
                pdq_transform_admission={"fixture": True},
                opaque_binding_bundle=binding,
                start_index=0,
                maximum_candidates=10,
                mode=AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER,
            )
        self.assertEqual(chunk.global_blockers, ())
        outcomes = {item.candidate_channels: item for item in chunk.records}
        self.assertEqual(
            outcomes[("PHASH",)].outcome,
            CandidateOutcome.PHASH_ONLY_REJECTED_BY_ADMITTED_PDQ,
        )
        self.assertEqual(outcomes[("PHASH",)].reason, "PDQ_COMPLETE_NEGATIVE")
        self.assertEqual(outcomes[("PDQ",)].outcome, CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY)
        self.assertEqual(outcomes[("EXACT", "PHASH")].outcome, CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY)
        ledger = merge_adjudication_chunks((chunk,))
        graph = assemble_frozen_evidence_graph(source=source, ledger=ledger)
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual({item.relation.value for item in graph.edges}, {"DEPENDENCY"})

    def test_pdq_negative_filter_fails_closed_without_admission(self) -> None:
        base = _source()
        binding = _binding(base)
        images = _images(base)
        bindings = dict(base.evidence_bindings)
        bindings["image_content_receipts_sha256"] = content_sha256(images)
        base = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        exact = build_exact_duplicate_graph(
            source=base,
            image_receipts=images,
            opaque_binding_bundle=binding,
        )
        phash = _phash()
        pdq = _pdq()
        bindings = dict(base.evidence_bindings)
        bindings.update({
            "exact_duplicate_graph_sha256": _token(800),
            "phash_candidates_sha256": content_sha256(phash),
            "pdq_candidates_sha256": content_sha256(pdq),
        })
        source = PublicSplitSourceBundle(tuple(sorted(bindings.items())), base.samples)
        chunk = build_adjudication_chunk(
            source=source,
            exact_graph=exact,
            exact_graph_artifact_sha256=_token(800),
            phash_evidence_bundle=phash,
            pdq_evidence_bundle=pdq,
            opaque_binding_bundle=binding,
            start_index=0,
            maximum_candidates=10,
            mode=AdjudicationMode.PDQ_COMPLETE_NEGATIVE_FILTER,
        )
        self.assertIn("PDQ_TRANSFORM_ADMISSION_MISSING", chunk.global_blockers)
        phash_only = next(item for item in chunk.records if item.candidate_channels == ("PHASH",))
        self.assertEqual(phash_only.outcome, CandidateOutcome.UNRESOLVED)
        self.assertEqual(merge_adjudication_chunks((chunk,)).promotion_status, "BLOCKED")

    def test_explicit_conservative_mode_closes_all_candidates_as_dependencies(self) -> None:
        base = _source()
        binding = _binding(base)
        images = _images(base)
        image_sha256 = content_sha256(images)
        base_bindings = dict(base.evidence_bindings)
        base_bindings["image_content_receipts_sha256"] = image_sha256
        base = PublicSplitSourceBundle(
            tuple(sorted(base_bindings.items())), base.samples
        )
        exact = build_exact_duplicate_graph(
            source=base,
            image_receipts=images,
            opaque_binding_bundle=binding,
        )
        phash = _phash()
        pdq = _pdq()
        final_bindings = dict(base.evidence_bindings)
        final_bindings.update({
            "exact_duplicate_graph_sha256": _token(800),
            "phash_candidates_sha256": content_sha256(phash),
            "pdq_candidates_sha256": content_sha256(pdq),
        })
        source = PublicSplitSourceBundle(
            tuple(sorted(final_bindings.items())), base.samples
        )
        chunk = build_adjudication_chunk(
            source=source,
            exact_graph=exact,
            exact_graph_artifact_sha256=_token(800),
            phash_evidence_bundle=phash,
            pdq_evidence_bundle=pdq,
            opaque_binding_bundle=binding,
            start_index=0,
            maximum_candidates=10,
            mode=(
                AdjudicationMode.LEAKAGE_CONSERVATIVE_CANDIDATE_COMPONENT_CLOSURE
            ),
        )
        self.assertEqual(chunk.global_blockers, ())
        self.assertEqual(chunk.unbound_candidate_count, 0)
        self.assertEqual(len(chunk.records), 3)
        self.assertEqual(
            {item.outcome for item in chunk.records},
            {CandidateOutcome.CANDIDATE_COMPONENT_DEPENDENCY},
        )
        ledger = merge_adjudication_chunks((chunk,))
        self.assertEqual(ledger.promotion_status, "READY_FOR_GRAPH_PROMOTION")
        graph = assemble_frozen_evidence_graph(source=source, ledger=ledger)
        self.assertEqual(len(graph.edges), 3)
        self.assertEqual({item.relation.value for item in graph.edges}, {"DEPENDENCY"})
        standard = build_adjudication_chunk(
            source=source,
            exact_graph=exact,
            exact_graph_artifact_sha256=_token(800),
            phash_evidence_bundle=phash,
            pdq_evidence_bundle=pdq,
            opaque_binding_bundle=binding,
            start_index=0,
            maximum_candidates=10,
        )
        self.assertIn("GEOMETRIC_EVIDENCE_MISSING", standard.global_blockers)
        self.assertIn(
            "REVIEW_ADJUDICATION_RECEIPT_MISSING", standard.global_blockers
        )

    def test_exact_receipts_and_candidate_chunks_are_complete_and_resumable(self) -> None:
        source = _source()
        binding = _binding(source)
        exact = build_exact_duplicate_graph(
            source=source,
            image_receipts=_images(source),
            opaque_binding_bundle=binding,
        )
        self.assertEqual(len(exact.pairs), 1)
        first = build_adjudication_chunk(
            source=source,
            exact_graph=exact,
            exact_graph_artifact_sha256=_token(800),
            phash_evidence_bundle=_phash(),
            opaque_binding_bundle=binding,
            start_index=0,
            maximum_candidates=1,
        )
        second = build_adjudication_chunk(
            source=source,
            exact_graph=exact,
            exact_graph_artifact_sha256=_token(800),
            phash_evidence_bundle=_phash(),
            opaque_binding_bundle=binding,
            start_index=1,
            maximum_candidates=1,
        )
        ledger = merge_adjudication_chunks((second, first))
        self.assertEqual(len(ledger.records), 2)
        self.assertEqual(ledger.records[0].outcome, CandidateOutcome.EXACT_CONFIRMED)
        self.assertEqual(ledger.records[1].outcome, CandidateOutcome.UNRESOLVED)
        self.assertEqual(ledger.promotion_status, "BLOCKED")
        self.assertIn("PDQ_CORPUS_EVIDENCE_MISSING", ledger.global_blockers)
        self.assertIn("GEOMETRIC_EVIDENCE_MISSING", ledger.global_blockers)
        queue = build_review_queue(
            source=source,
            ledger=ledger,
            image_receipts=_images(source),
            phash_evidence_bundle=_phash(),
            opaque_binding_bundle=binding,
        )
        self.assertEqual(queue["record_count"], 1)
        self.assertEqual(queue["records"][0]["phash_hamming_distance"], 8)
        self.assertEqual(queue["decision"], "REVIEW_REQUIRED_NO_OUTCOMES_ASSIGNED")
        with self.assertRaisesRegex(RuntimeError, "zero unresolved"):
            assemble_frozen_evidence_graph(source=source, ledger=ledger)

    def test_merge_rejects_gap_and_promotion_rechecks_source_bindings(self) -> None:
        source = _source()
        binding = _binding(source)
        exact = build_exact_duplicate_graph(
            source=source,
            image_receipts=_images(source),
            opaque_binding_bundle=binding,
        )
        second = build_adjudication_chunk(
            source=source,
            exact_graph=exact,
            exact_graph_artifact_sha256=_token(800),
            phash_evidence_bundle=_phash(),
            opaque_binding_bundle=binding,
            start_index=1,
            maximum_candidates=1,
        )
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            merge_adjudication_chunks((second,))

        evidence = _token(900)
        record = CandidateAdjudication(
            left_sample_token=source.samples[0].sample_token,
            right_sample_token=source.samples[1].sample_token,
            candidate_channels=("EXACT",),
            candidate_evidence_tokens=(evidence,),
            outcome=CandidateOutcome.EXACT_CONFIRMED,
            reason="AUTHENTICATED_CANONICAL_RGB_DIGEST_EQUAL",
            decision_evidence_tokens=(evidence,),
        )
        ledger = AdjudicationLedger(
            source_bundle_sha256=source.bundle_sha256,
            candidate_set_sha256=_token(901),
            evidence_bindings=tuple(sorted((
                ("exact_duplicate_graph_sha256", _token(999)),
                ("phash_candidates_sha256", dict(source.evidence_bindings)["phash_candidates_sha256"]),
                ("pdq_candidates_sha256", dict(source.evidence_bindings)["pdq_candidates_sha256"]),
                ("review_adjudication_sha256", dict(source.evidence_bindings)["review_adjudication_sha256"]),
            ))),
            records=(record,),
            outcome_counts=(("EXACT_CONFIRMED", 1),),
            global_blockers=(),
            promotion_status="READY_FOR_GRAPH_PROMOTION",
        )
        with self.assertRaisesRegex(ValueError, "exact_duplicate_graph"):
            assemble_frozen_evidence_graph(source=source, ledger=ledger)

if __name__ == "__main__":
    unittest.main()
