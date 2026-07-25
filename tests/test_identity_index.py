from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvi.identity_index import (
    EvidenceBreakdown,
    IdentityIndex,
    IndexedIdentity,
    SearchResult,
)


class IdentityIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._d = tempfile.mkdtemp()
        self._idx = IdentityIndex(
            Path(self._d) / "index.faiss",
            Path(self._d) / "metadata.json",
        )

    def tearDown(self) -> None:
        self._idx.close()

    def _emb(self) -> np.ndarray:
        e = np.random.randn(640).astype(np.float32)
        return e / max(np.linalg.norm(e), 1e-8)

    def test_empty_index(self) -> None:
        self.assertEqual(self._idx.size, 0)
        results = self._idx.search(self._emb(), top_k=5)
        self.assertEqual(len(results), 0)

    def test_enroll_single(self) -> None:
        e = self._emb()
        pos = self._idx.enroll(e, "dog-uuid-abc", "yt-bb-dog", {"breed": "poodle"})
        self.assertEqual(pos, 0)
        self.assertEqual(self._idx.size, 1)

    def test_enroll_and_search(self) -> None:
        e1 = self._emb()
        e2 = self._emb()
        self._idx.enroll(e1, "dog-1", "yt-bb-dog")
        self._idx.enroll(e2, "dog-2", "yt-bb-dog")
        results = self._idx.search(e1, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].registered_dog_id, "dog-1")

    def test_enroll_batch(self) -> None:
        embs = np.stack([self._emb() for _ in range(5)])
        rids = [f"dog-{i}" for i in range(5)]
        dsn = ["yt-bb-dog"] * 5
        indices = self._idx.enroll_batch(embs, rids, dsn)
        self.assertEqual(len(indices), 5)
        self.assertEqual(self._idx.size, 5)

    def test_persistence(self) -> None:
        e = self._emb()
        self._idx.enroll(e, "dog-persist", "yt-bb-dog")
        self._idx.save()

        idx2 = IdentityIndex(
            Path(self._d) / "index.faiss",
            Path(self._d) / "metadata.json",
        )
        self.assertEqual(idx2.size, 1)
        results = idx2.search(e, top_k=1)
        self.assertEqual(results[0].registered_dog_id, "dog-persist")
        idx2.close()

    def test_search_with_fusion_weights(self) -> None:
        e1 = self._emb()
        e2 = self._emb()
        self._idx.enroll(e1, "dog-1")
        self._idx.enroll(e2, "dog-2")
        results = self._idx.search(e1, top_k=1, fusion_weights=(0.0, 1.0, 0.0))
        self.assertEqual(len(results), 1)

    def test_search_result_has_evidence(self) -> None:
        e1 = self._emb()
        self._idx.enroll(e1, "dog-ev", "yt-bb-dog")
        results = self._idx.search(e1, top_k=1)
        self.assertIsNotNone(results[0].evidence)
        self.assertIsInstance(results[0].evidence.visual, float)
        self.assertIsInstance(results[0].evidence.texture, float)
        self.assertIsInstance(results[0].evidence.structural, float)

    def test_enroll_invalid_dim(self) -> None:
        with self.assertRaises(ValueError):
            self._idx.enroll(np.random.randn(128).astype(np.float32), "bad-dim")

    def test_reject_reconstruct(self) -> None:
        e = self._emb()
        self._idx.enroll(e, "dog-recon", "yt-bb-dog")
        results = self._idx.search(e, top_k=1)
        self.assertGreater(results[0].similarity, 0.9)


class EvidenceBreakdownTests(unittest.TestCase):
    def test_to_dict(self) -> None:
        eb = EvidenceBreakdown(visual=0.9, texture=0.5, structural=0.7)
        d = eb.to_dict()
        self.assertAlmostEqual(d["visual"], 0.9)
        self.assertAlmostEqual(d["texture"], 0.5)

    def test_all_positive(self) -> None:
        eb = EvidenceBreakdown(0.8, 0.6, 0.4)
        for v in [eb.visual, eb.texture, eb.structural]:
            self.assertGreaterEqual(v, 0.0)


class SearchResultTests(unittest.TestCase):
    def test_to_dict(self) -> None:
        sr = SearchResult("uuid", 0.85, EvidenceBreakdown(0.9, 0.5, 0.7))
        d = sr.to_dict()
        self.assertEqual(d["registered_dog_id"], "uuid")


class IndexedIdentityTests(unittest.TestCase):
    def test_default_metadata_empty(self) -> None:
        ii = IndexedIdentity("uuid", "yt-bb-dog", "2024-01-01", {})
        self.assertEqual(ii.metadata, {})


if __name__ == "__main__":
    unittest.main()
