from __future__ import annotations

import unittest

from identity.splits.duplicate_graph_capacity import analyze_duplicate_graph_capacity
from identity.splits.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    ProtectedPublicSplitPolicy,
    PublicSplitEvidenceEdge,
    PublicSplitSample,
    PublicSplitSourceBundle,
)
from foundation.provenance import content_sha256


def _token(value: int) -> str:
    return f"{value:064x}"


class DuplicateGraphCapacityTests(unittest.TestCase):
    def test_cross_lane_block_is_quarantined_without_allocation(self) -> None:
        names = (
            "exact_duplicate_graph_sha256",
            "geometric_verifier_sha256",
            "image_content_receipts_sha256",
            "pdq_candidates_sha256",
            "phash_candidates_sha256",
            "review_adjudication_sha256",
            "semantic_receipts_sha256",
        )
        bindings = tuple((name, _token(index + 100)) for index, name in enumerate(names))
        samples = (
            PublicSplitSample(
                _token(1), _token(11), _token(21), "yt:1", "yt:identity:1",
                "yt-bb-dog", "original", "train", 0, None, None, "DOG_CROP",
            ),
            PublicSplitSample(
                _token(2), _token(12), _token(22), "dog:1", "dog:identity:1",
                "dogfacenet224", "original", "train", 0, None, None, "FACE",
            ),
        )
        source = PublicSplitSourceBundle(bindings, samples)
        edge = PublicSplitEvidenceEdge(
            _token(1), _token(2), EvidenceRelation.DEPENDENCY,
            content_sha256([_token(1), _token(2)]),
        )
        graph = FrozenPublicSplitEvidenceGraph(bindings, (edge,))
        report = analyze_duplicate_graph_capacity(
            source=source, graph=graph, policy=ProtectedPublicSplitPolicy()
        )
        self.assertEqual(report["largest_cross_lane_block_identity_count"], 2)
        self.assertEqual(report["quarantined_identity_count"], 2)
        self.assertEqual(report["available_identity_upper_bounds_by_lane"], {})
        self.assertEqual(report["status"], "COMPONENT_CAPACITY_UPPER_BOUND_FAILED")


if __name__ == "__main__":
    unittest.main()
