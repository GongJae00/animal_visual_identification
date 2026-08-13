from __future__ import annotations

import json
import unittest
from pathlib import Path

from contracts.contracts import (
    CandidateScore,
    ConflictStatus,
    DecisionRecord,
    DecisionStatus,
    EvidenceFrameRef,
    EvidenceStatus,
    Modality,
    OccupancyStatus,
    OperationalContext,
    StreamStatus,
    TrackKey,
    VisualIdentityResult,
    VisualIdentityStatus,
)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = OperationalContext(
            camera_id="cam-01",
            cage_id="cage-01",
            expected_dog_id="dog-001",
        )
        self.track = TrackKey("cam-01", "session-01", "track-01")
        self.refs = (
            EvidenceFrameRef("cam-01", "det-10", 100, Modality.RGB),
            EvidenceFrameRef("cam-01", "det-12", 120, Modality.RGB),
        )
        self.candidates = (
            CandidateScore("dog-001", 0.91),
            CandidateScore("dog-002", 0.73),
        )

    def visual(
        self,
        status: VisualIdentityStatus = VisualIdentityStatus.KNOWN,
        predicted_dog_id: str | None = "dog-001",
    ) -> VisualIdentityResult:
        return VisualIdentityResult(
            track=self.track,
            status=status,
            modality=Modality.RGB,
            input_quality=0.82,
            candidates=self.candidates,
            top1_top2_margin=0.18,
            evidence_frames=self.refs,
            model_version="model-sha256:abc",
            gallery_version="gallery-sha256:def",
            predicted_dog_id=predicted_dog_id,
        )

    def test_known_result_preserves_visual_only_evidence(self) -> None:
        decision = DecisionRecord(
            timestamp_ns=150,
            context=self.context,
            stream=StreamStatus.OK,
            occupancy=OccupancyStatus.SINGLE_DOG,
            evidence=EvidenceStatus.USABLE,
            visual_status=VisualIdentityStatus.KNOWN,
            visual_result=self.visual(),
        )
        self.assertEqual(decision.status, DecisionStatus.KNOWN)
        self.assertFalse(hasattr(decision.visual_result, "cage_id"))
        self.assertFalse(hasattr(decision.visual_result, "expected_dog_id"))

    def test_known_result_matches_golden_api_fixture(self) -> None:
        decision = DecisionRecord(
            timestamp_ns=150,
            context=self.context,
            stream=StreamStatus.OK,
            occupancy=OccupancyStatus.SINGLE_DOG,
            evidence=EvidenceStatus.USABLE,
            visual_status=VisualIdentityStatus.KNOWN,
            visual_result=self.visual(),
        )
        fixture_path = Path(__file__).parent / "fixtures" / "known_decision.json"
        expected = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(decision.to_dict(), expected)

    def test_conflict_does_not_replace_visual_prediction(self) -> None:
        context = OperationalContext("cam-01", "cage-01", "dog-099")
        decision = DecisionRecord(
            timestamp_ns=150,
            context=context,
            stream=StreamStatus.OK,
            occupancy=OccupancyStatus.SINGLE_DOG,
            evidence=EvidenceStatus.USABLE,
            visual_status=VisualIdentityStatus.KNOWN,
            conflict=ConflictStatus.IDENTITY_CONFLICT,
            visual_result=self.visual(),
        )
        self.assertEqual(decision.status, DecisionStatus.IDENTITY_CONFLICT)
        self.assertEqual(decision.visual_result.predicted_dog_id, "dog-001")

    def test_expected_mismatch_cannot_hide_without_conflict(self) -> None:
        context = OperationalContext("cam-01", "cage-01", "dog-099")
        with self.assertRaisesRegex(ValueError, "require conflict"):
            DecisionRecord(
                timestamp_ns=150,
                context=context,
                stream=StreamStatus.OK,
                occupancy=OccupancyStatus.SINGLE_DOG,
                evidence=EvidenceStatus.USABLE,
                visual_status=VisualIdentityStatus.KNOWN,
                visual_result=self.visual(),
            )

    def test_failed_stream_cannot_claim_no_dog(self) -> None:
        with self.assertRaisesRegex(ValueError, "blocks occupancy"):
            DecisionRecord(
                timestamp_ns=150,
                context=self.context,
                stream=StreamStatus.FAILED,
                occupancy=OccupancyStatus.NO_DOG,
                evidence=EvidenceStatus.NOT_EVALUATED,
                visual_status=VisualIdentityStatus.NOT_EVALUATED,
            )

    def test_failed_stream_has_no_identity_projection(self) -> None:
        decision = DecisionRecord(
            timestamp_ns=150,
            context=self.context,
            stream=StreamStatus.FAILED,
            occupancy=OccupancyStatus.NOT_EVALUATED,
            evidence=EvidenceStatus.NOT_EVALUATED,
            visual_status=VisualIdentityStatus.NOT_EVALUATED,
        )
        self.assertIsNone(decision.status)

    def test_multiple_dogs_block_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot evaluate evidence"):
            DecisionRecord(
                timestamp_ns=150,
                context=self.context,
                stream=StreamStatus.OK,
                occupancy=OccupancyStatus.MULTIPLE_DOGS,
                evidence=EvidenceStatus.USABLE,
                visual_status=VisualIdentityStatus.KNOWN,
                visual_result=self.visual(),
            )

    def test_no_evidence_maps_without_forcing_identity(self) -> None:
        decision = DecisionRecord(
            timestamp_ns=150,
            context=self.context,
            stream=StreamStatus.DEGRADED,
            occupancy=OccupancyStatus.SINGLE_DOG,
            evidence=EvidenceStatus.NO_USABLE_EVIDENCE,
            visual_status=VisualIdentityStatus.NOT_EVALUATED,
        )
        self.assertEqual(decision.status, DecisionStatus.NO_USABLE_EVIDENCE)

    def test_single_dog_pending_maps_to_dog_present(self) -> None:
        decision = DecisionRecord(
            timestamp_ns=150,
            context=self.context,
            stream=StreamStatus.OK,
            occupancy=OccupancyStatus.SINGLE_DOG,
            evidence=EvidenceStatus.NOT_EVALUATED,
            visual_status=VisualIdentityStatus.PENDING,
        )
        self.assertEqual(decision.status, DecisionStatus.DOG_PRESENT)

    def test_unknown_cannot_assign_a_registered_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "only KNOWN"):
            self.visual(VisualIdentityStatus.UNKNOWN, "dog-001")

    def test_candidate_order_and_margin_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "descending"):
            VisualIdentityResult(
                track=self.track,
                status=VisualIdentityStatus.REVIEW_REQUIRED,
                modality=Modality.RGB,
                input_quality=0.8,
                candidates=tuple(reversed(self.candidates)),
                top1_top2_margin=-0.18,
                evidence_frames=self.refs,
                model_version="m1",
                gallery_version="g1",
            )

    def test_future_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "future"):
            DecisionRecord(
                timestamp_ns=110,
                context=self.context,
                stream=StreamStatus.OK,
                occupancy=OccupancyStatus.SINGLE_DOG,
                evidence=EvidenceStatus.USABLE,
                visual_status=VisualIdentityStatus.KNOWN,
                visual_result=self.visual(),
            )


if __name__ == "__main__":
    unittest.main()
