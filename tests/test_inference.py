from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvi.inference import (
    Gallery,
    InferenceConfig,
    IdentityLookup,
    MatchResult,
    QueryResult,
)


class InferenceConfigTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        c = InferenceConfig(
            model_path="model.onnx",
            gallery_embeddings_path="gallery.npy",
            registry_db_path="registry.db",
            top_k=10,
            similarity_threshold=0.5,
        )
        d = c.to_dict()
        restored = InferenceConfig.from_dict(d)
        self.assertEqual(restored.top_k, 10)
        self.assertEqual(restored.similarity_threshold, 0.5)


class GalleryTests(unittest.TestCase):
    def test_empty_gallery_returns_empty_results(self) -> None:
        g = Gallery(np.empty((0, 64)), [])
        q = np.random.randn(64).astype(np.float32)
        q = q / np.linalg.norm(q)
        results = g.search(q, top_k=5)
        self.assertEqual(len(results), 0)

    def test_single_identity_search(self) -> None:
        emb = np.random.randn(1, 64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        g = Gallery(emb, ["uuid-dog-1"])
        q = emb[0].copy()
        results = g.search(q, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].registered_dog_id, "uuid-dog-1")
        self.assertAlmostEqual(results[0].similarity, 1.0, places=5)

    def test_top_k_ordering(self) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = [f"dog-{i}" for i in range(10)]
        g = Gallery(embeddings, labels)
        query = embeddings[5].copy()
        results = g.search(query, top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].registered_dog_id, "dog-5")
        self.assertGreaterEqual(results[0].similarity, results[1].similarity)

    def test_threshold_filters(self) -> None:
        emb = np.eye(64, dtype=np.float32)[:1]
        g = Gallery(emb, ["dog-0"])
        q = -emb[0]
        results = g.search(q, top_k=5, threshold=0.5)
        self.assertEqual(len(results), 0)

    def test_save_load_roundtrip(self) -> None:
        emb = np.random.randn(3, 64).astype(np.float32)
        labels = ["a", "b", "c"]
        g1 = Gallery(emb, labels)
        with tempfile.NamedTemporaryFile(suffix=".npy") as f_emb:
            np.save(f_emb, emb)
            f_emb.flush()
            with tempfile.NamedTemporaryFile(
                suffix=".json", mode="w"
            ) as f_lbl:
                json.dump(labels, f_lbl)
                f_lbl.flush()
                g2 = Gallery.load(Path(f_emb.name), Path(f_lbl.name))
                self.assertEqual(g2.size, 3)
                self.assertEqual(g2.search(emb[0])[0].registered_dog_id, "a")


class MatchResultTests(unittest.TestCase):
    def test_to_dict(self) -> None:
        m = MatchResult("uuid-1", 0.95, "yt-bb-dog")
        d = m.to_dict()
        self.assertEqual(d["registered_dog_id"], "uuid-1")
        self.assertAlmostEqual(d["similarity"], 0.95)


class QueryResultTests(unittest.TestCase):
    def test_to_dict(self) -> None:
        m = (MatchResult("uuid-1", 0.95, "yt-bb-dog"),)
        q = QueryResult("query-1", m, 12.5)
        d = q.to_dict()
        self.assertEqual(d["query_token"], "query-1")
        self.assertEqual(len(d["matches"]), 1)


class IdentityLookupTests(unittest.TestCase):
    def _db(self) -> Path:
        fd, path = tempfile.mkstemp(suffix=".db", prefix="cvi_inf_test_")
        return Path(path)

    def test_lookup_missing(self) -> None:
        db = self._db()
        from cvi.identity_registry import create_registry_database, register_records
        create_registry_database(db)
        register_records(db, ["yt-bb-dog:v1:video-track:1"])

        lookup = IdentityLookup(db)
        self.assertIsNone(lookup.lookup("f" * 64))
        lookup.close()
        db.unlink()

    def test_lookup_by_token(self) -> None:
        db = self._db()
        from cvi.identity_registry import (
            compute_identity_token,
            create_registry_database,
            register_records,
        )
        did = "yt-bb-dog:v1:video-track:42"
        create_registry_database(db)
        register_records(db, [did])

        token = compute_identity_token(did)
        lookup = IdentityLookup(db)
        rec = lookup.lookup(token)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["registered_dog_id"], rec["registered_dog_id"])
        lookup.close()
        db.unlink()

    def test_lookup_by_label(self) -> None:
        db = self._db()
        from cvi.identity_registry import (
            compute_registered_dog_id,
            create_registry_database,
            register_records,
        )
        did = "yt-bb-dog:v1:video-track:99"
        create_registry_database(db)
        register_records(db, [did])

        rid = compute_registered_dog_id(did)
        lookup = IdentityLookup(db)
        rec = lookup.lookup_by_label(rid)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["registered_dog_id"], rid)
        lookup.close()
        db.unlink()




if __name__ == "__main__":
    unittest.main()
