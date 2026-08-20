from __future__ import annotations

import json
import unittest

from shared.contracts.contracts import Modality
from evaluation.splits.tracklet_split import (
    PresenceState,
    RoleModalityRule,
    SplitManifest,
    SplitPolicy,
    SplitRole,
    TrackletRecord,
)

class DatasetSplitTests(unittest.TestCase):
    def record(
        self,
        role: SplitRole,
        dog_id: str,
        index: int,
        *,
        source_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        start_timestamp_ns: int | None = None,
        modality: Modality | None = None,
        collar: PresenceState = PresenceState.UNKNOWN,
    ) -> TrackletRecord:
        return TrackletRecord(
            sample_id=f"sample-{index}",
            role=role,
            registered_dog_id=dog_id,
            identity_verification_source="microchip",
            source_id=source_id or f"source-{index}",
            site_id=f"site-{index}",
            camera_id=camera_id or f"camera-{index}",
            cage_id=f"cage-{index}",
            session_id=session_id or f"session-{index}",
            occupancy_episode_id=f"episode-{index}",
            track_id=f"track-{index}",
            start_timestamp_ns=(
                start_timestamp_ns
                if start_timestamp_ns is not None
                else index * 1_000
            ),
            end_timestamp_ns=(
                start_timestamp_ns + 100
                if start_timestamp_ns is not None
                else index * 1_000 + 100
            ),
            modality=(
                modality
                if modality is not None
                else (Modality.RGB if index % 2 == 0 else Modality.IR)
            ),
            collar=collar,
        )

    def valid_records(self) -> tuple[TrackletRecord, ...]:
        return (
            self.record(SplitRole.TRAIN, "dog-train", 1),
            self.record(SplitRole.DEVELOPMENT, "dog-dev", 2),
            self.record(
                SplitRole.CALIBRATION_GALLERY,
                "dog-cal-known",
                3,
            ),
            self.record(
                SplitRole.CALIBRATION_KNOWN_QUERY,
                "dog-cal-known",
                4,
            ),
            self.record(
                SplitRole.CALIBRATION_UNKNOWN_QUERY,
                "dog-cal-unknown",
                5,
            ),
            self.record(SplitRole.TEST_GALLERY, "dog-test-known", 6),
            self.record(SplitRole.TEST_KNOWN_QUERY, "dog-test-known", 7),
            self.record(
                SplitRole.TEST_UNKNOWN_QUERY,
                "dog-test-unknown",
                8,
            ),
        )

    def manifest(
        self,
        records: tuple[TrackletRecord, ...] | None = None,
        policy: SplitPolicy | None = None,
    ) -> SplitManifest:
        selected = records or self.valid_records()
        return SplitManifest(
            policy=policy or SplitPolicy("unseen-identity-open-set-v1"),
            admitted_source_ids=tuple(
                dict.fromkeys(record.source_id for record in selected)
            ),
            records=selected,
        )

    def test_valid_unseen_identity_open_set_split_passes(self) -> None:
        manifest = self.manifest()
        self.assertEqual(manifest.gate_blockers(), ())
        round_tripped = SplitManifest.from_dict(
            json.loads(json.dumps(manifest.to_dict()))
        )
        self.assertEqual(round_tripped.to_dict(), manifest.to_dict())
        self.assertEqual(round_tripped.manifest_sha256, manifest.manifest_sha256)

    def test_same_source_cannot_cross_roles(self) -> None:
        records = list(self.valid_records())
        records[1] = self.record(
            SplitRole.DEVELOPMENT,
            "dog-dev",
            2,
            source_id=records[0].source_id,
        )
        blockers = self.manifest(tuple(records)).gate_blockers()
        self.assertTrue(
            any(blocker.startswith("cross_role_leak:source_id") for blocker in blockers)
        )

    def test_same_session_cannot_cross_gallery_and_query(self) -> None:
        records = list(self.valid_records())
        records[3] = self.record(
            SplitRole.CALIBRATION_KNOWN_QUERY,
            "dog-cal-known",
            4,
            camera_id=records[2].camera_id,
            session_id=records[2].session_id,
        )
        blockers = self.manifest(tuple(records)).gate_blockers()
        self.assertTrue(
            any(
                blocker.startswith("cross_role_leak:session_key")
                for blocker in blockers
            )
        )

    def test_unknown_identity_cannot_exist_in_gallery(self) -> None:
        records = list(self.valid_records())
        records[7] = self.record(
            SplitRole.TEST_UNKNOWN_QUERY,
            "dog-test-known",
            8,
        )
        blockers = self.manifest(tuple(records)).gate_blockers()
        self.assertIn("test:unknown_identity_leak:dog-test-known", blockers)

    def test_train_identity_cannot_reappear_in_final_test(self) -> None:
        records = list(self.valid_records())
        records[7] = self.record(
            SplitRole.TEST_UNKNOWN_QUERY,
            "dog-train",
            8,
        )
        blockers = self.manifest(tuple(records)).gate_blockers()
        self.assertIn("train_evaluation_identity_leak:dog-train", blockers)

    def test_camera_disjoint_policy_is_stage_scoped(self) -> None:
        records = list(self.valid_records())
        records[5] = self.record(
            SplitRole.TEST_GALLERY,
            "dog-test-known",
            6,
            camera_id=records[0].camera_id,
        )
        policy = SplitPolicy(
            "camera-disjoint-v1",
            stage_disjoint_keys=("camera_id",),
        )
        blockers = self.manifest(tuple(records), policy).gate_blockers()
        self.assertTrue(
            any(
                blocker.startswith("cross_stage_leak:camera_id")
                for blocker in blockers
            )
        )

    def test_chronological_policy_rejects_earlier_test(self) -> None:
        records = list(self.valid_records())
        records[5] = self.record(
            SplitRole.TEST_GALLERY,
            "dog-test-known",
            6,
            start_timestamp_ns=50,
        )
        policy = SplitPolicy(
            "future-test-v1",
            require_chronological_test=True,
        )
        blockers = self.manifest(tuple(records), policy).gate_blockers()
        self.assertIn(
            "chronological_leak:test_not_strictly_after_pretest",
            blockers,
        )

    def test_role_modality_rule_blocks_wrong_cross_modal_direction(self) -> None:
        policy = SplitPolicy(
            "rgb-query-ir-gallery-v1",
            modality_rules=(
                RoleModalityRule(
                    SplitRole.TEST_GALLERY,
                    (Modality.IR,),
                ),
                RoleModalityRule(
                    SplitRole.TEST_KNOWN_QUERY,
                    (Modality.RGB,),
                ),
            ),
        )
        blockers = self.manifest(policy=policy).gate_blockers()
        self.assertIn(
            "modality_violation:sample-6:TEST_GALLERY:RGB",
            blockers,
        )
        self.assertIn(
            "modality_violation:sample-7:TEST_KNOWN_QUERY:IR",
            blockers,
        )

    def test_accessory_change_protocol_rejects_unresolved_metadata(self) -> None:
        policy = SplitPolicy(
            "accessory-change-v1",
            require_known_query_accessory_change=True,
        )
        blockers = self.manifest(policy=policy).gate_blockers()
        self.assertIn(
            "calibration:accessory_state_unresolved:dog-cal-known",
            blockers,
        )
        self.assertIn(
            "test:accessory_state_unresolved:dog-test-known",
            blockers,
        )

    def test_longitudinal_policy_requires_predeclared_gallery_query_gap(self) -> None:
        policy = SplitPolicy(
            "longitudinal-v1",
            minimum_gallery_query_gap_seconds=0.000001,
        )
        blockers = self.manifest(policy=policy).gate_blockers()
        self.assertIn(
            "calibration:insufficient_gallery_query_gap:sample-4",
            blockers,
        )
        self.assertIn(
            "test:insufficient_gallery_query_gap:sample-7",
            blockers,
        )

if __name__ == "__main__":
    unittest.main()
