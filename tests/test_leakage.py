from __future__ import annotations

import unittest

from cvi.contracts import Modality
from cvi.dataset import SplitRole, TrackletRecord
from cvi.leakage import association_audit


def record(
    index: int,
    dog_id: str,
    cage_id: str,
    *,
    episode_id: str | None = None,
    track_id: str | None = None,
) -> TrackletRecord:
    return TrackletRecord(
        sample_id=f"sample-{index}",
        role=SplitRole.TRAIN,
        registered_dog_id=dog_id,
        identity_verification_source="microchip",
        source_id=f"source-{index}",
        site_id="site-1",
        camera_id=f"camera-{cage_id}",
        cage_id=cage_id,
        session_id=f"session-{index}",
        occupancy_episode_id=episode_id or f"episode-{index}",
        track_id=track_id or f"track-{index}",
        start_timestamp_ns=index * 1_000,
        end_timestamp_ns=index * 1_000 + 100,
        modality=Modality.RGB,
    )


class LeakageAuditTests(unittest.TestCase):
    def test_one_to_one_dog_cage_mapping_is_maximally_associated(self) -> None:
        records = tuple(
            record(index, f"dog-{index}", f"cage-{index}")
            for index in range(4)
        )
        audit = association_audit(records, "cage_id")
        self.assertAlmostEqual(audit.normalized_mutual_information or 0.0, 1.0)
        self.assertEqual(audit.domain_to_identity_majority_accuracy, 1.0)
        self.assertEqual(audit.global_identity_majority_accuracy, 0.25)
        self.assertEqual(audit.identity_to_domain_concentration, 1.0)

    def test_balanced_cross_cage_use_has_zero_association(self) -> None:
        records = (
            record(1, "dog-a", "cage-a"),
            record(2, "dog-a", "cage-b"),
            record(3, "dog-b", "cage-a"),
            record(4, "dog-b", "cage-b"),
        )
        audit = association_audit(records, "cage_id")
        self.assertAlmostEqual(audit.normalized_mutual_information or 0.0, 0.0)
        self.assertEqual(audit.domain_to_identity_majority_accuracy, 0.5)
        self.assertEqual(audit.global_identity_majority_accuracy, 0.5)
        self.assertEqual(audit.identity_to_domain_concentration, 0.5)

    def test_multiple_tracklets_from_one_episode_receive_one_vote(self) -> None:
        first = record(
            1,
            "dog-a",
            "cage-a",
            episode_id="episode-shared",
            track_id="track-shared",
        )
        second = TrackletRecord(
            sample_id="sample-2",
            role=first.role,
            registered_dog_id=first.registered_dog_id,
            identity_verification_source=first.identity_verification_source,
            source_id=first.source_id,
            site_id=first.site_id,
            camera_id=first.camera_id,
            cage_id=first.cage_id,
            session_id=first.session_id,
            occupancy_episode_id=first.occupancy_episode_id,
            track_id=first.track_id,
            start_timestamp_ns=first.end_timestamp_ns,
            end_timestamp_ns=first.end_timestamp_ns + 100,
            modality=first.modality,
        )
        audit = association_audit((first, second), "cage_id")
        self.assertEqual(audit.episode_identity_events, 1)
        self.assertIsNone(audit.normalized_mutual_information)


if __name__ == "__main__":
    unittest.main()
