from __future__ import annotations

import unittest

from embedding.methods.nose.training.sampler import CrossSessionPKBatchSampler


class NoseIDSamplerTests(unittest.TestCase):
    def test_pk_batch_prefers_cross_session_and_inserts_hard_neighbor(self) -> None:
        identity_names = [f"id-{index:02d}" for index in range(18)]
        identities = [identity for identity in identity_names for _ in range(4)]
        sessions = [session for _ in identity_names for session in ("s1", "s1", "s2", "s2")]
        sampler = CrossSessionPKBatchSampler(
            identities,
            sessions,
            hard_neighbors={identity: ("id-17",) for identity in identity_names[:-1]},
            seed=3,
        )
        batch = next(iter(sampler))
        selected_identities = [identities[index] for index in batch]
        self.assertEqual(len(batch), 64)
        self.assertEqual(len(set(selected_identities)), 16)
        self.assertEqual({selected_identities.count(identity) for identity in set(selected_identities)}, {4})
        for identity in set(selected_identities):
            selected_sessions = {sessions[index] for index in batch if identities[index] == identity}
            self.assertEqual(selected_sessions, {"s1", "s2"})
        self.assertIn("id-17", selected_identities)

    def test_single_session_identity_is_rejected(self) -> None:
        identities = [f"id-{identity}" for identity in range(16) for _ in range(4)]
        sessions = ["only-session"] * len(identities)
        with self.assertRaisesRegex(ValueError, "two sessions"):
            CrossSessionPKBatchSampler(identities, sessions)


if __name__ == "__main__":
    unittest.main()
