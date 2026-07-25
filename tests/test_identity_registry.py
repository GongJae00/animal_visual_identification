from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cvi.identity_registry import (
    CVI_REGISTERED_DOG_NAMESPACE,
    IdentityRegistry,
    IdentityRegistryRecord,
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
    compute_sequence_token,
    create_registry_database,
    extract_dataset_name,
    load_registry_manifest,
    lookup_by_identity_token,
    lookup_registered_dog_id,
    register_identity,
    register_records,
)


class IdentityTokenTests(unittest.TestCase):
    def test_identity_token_is_deterministic(self) -> None:
        did = "yt-bb-dog:v1:video-track:1234"
        self.assertEqual(compute_identity_token(did), compute_identity_token(did))

    def test_identity_token_differs_by_payload(self) -> None:
        self.assertNotEqual(
            compute_identity_token("yt-bb-dog:v1:video-track:1234"),
            compute_identity_token("yt-bb-dog:v1:video-track:5678"),
        )

    def test_sample_token_differs_from_identity_token(self) -> None:
        sid = "yt-bb-dog:v1:video-track:1234:frame:42"
        self.assertNotEqual(compute_sample_token(sid), compute_identity_token(sid))

    def test_sequence_token_falls_back_hashes_identity_token(self) -> None:
        import hashlib
        token = compute_identity_token("dogfacenet224:v1:web-folder:231")
        expected = hashlib.sha256(b"sequence\x00" + token.encode("utf-8")).hexdigest()
        self.assertEqual(compute_sequence_token(None, token), expected)

    def test_sequence_token_differs_when_given(self) -> None:
        token = compute_identity_token("mpdd:v1:device-capture:42")
        seq = compute_sequence_token("mpdd:v1:filename-sequence-token:3", token)
        self.assertNotEqual(seq, token)


class RegisteredDogIdTests(unittest.TestCase):
    def test_uuidv5_is_deterministic(self) -> None:
        did = "yt-bb-dog:v1:video-track:1234"
        self.assertEqual(
            compute_registered_dog_id(did), compute_registered_dog_id(did)
        )

    def test_different_dataset_identity_ids_differ(self) -> None:
        self.assertNotEqual(
            compute_registered_dog_id("yt-bb-dog:v1:video-track:1234"),
            compute_registered_dog_id("yt-bb-dog:v1:video-track:5678"),
        )

    def test_cross_dataset_collision_avoidance(self) -> None:
        yt_id = compute_registered_dog_id("yt-bb-dog:v1:video-track:1")
        dogface_id = compute_registered_dog_id("dogfacenet224:v1:web-folder:1")
        self.assertNotEqual(yt_id, dogface_id)

    def test_output_is_valid_uuid(self) -> None:
        import uuid
        rid = compute_registered_dog_id("sibetan:v1:gt-json:dog_ABC")
        parsed = uuid.UUID(rid)
        self.assertEqual(parsed.version, 5)

    def test_namespace_is_stable(self) -> None:
        self.assertEqual(
            str(CVI_REGISTERED_DOG_NAMESPACE),
            "877d96de-ba43-542d-9523-5c20213bfc09",
        )


class ExtractDatasetNameTests(unittest.TestCase):
    def test_yt_bb_dog(self) -> None:
        self.assertEqual(
            extract_dataset_name("yt-bb-dog:v1:video-track:1234"),
            "yt-bb-dog",
        )

    def test_dogfacenet(self) -> None:
        self.assertEqual(
            extract_dataset_name("dogfacenet224:v1:web-folder:231"),
            "dogfacenet224",
        )

    def test_mpdd(self) -> None:
        self.assertEqual(
            extract_dataset_name("mpdd:v1:device-capture:42"),
            "mpdd",
        )

    def test_sibetan(self) -> None:
        self.assertEqual(
            extract_dataset_name("sibetan:v1:gt-json:dog_ABC"),
            "sibetan",
        )


class IdentityRegistryRecordTests(unittest.TestCase):
    def test_round_trip_dict(self) -> None:
        rec = IdentityRegistryRecord(
            identity_token="a" * 64,
            dataset_identity_id="yt-bb-dog:v1:video-track:1",
            registered_dog_id="877d96de-ba43-542d-9523-5c20213bfc09",
            dataset_name="yt-bb-dog",
            image_count=42,
        )
        d = rec.to_dict()
        restored = IdentityRegistryRecord.from_dict(d)
        self.assertEqual(rec, restored)


class IdentityRegistryContractTests(unittest.TestCase):
    def test_empty_registry(self) -> None:
        registry = IdentityRegistry(records=())
        self.assertEqual(len(registry.records), 0)

    def test_registry_rejects_duplicate_tokens(self) -> None:
        rec = IdentityRegistryRecord(
            identity_token="a" * 64,
            dataset_identity_id="yt-bb-dog:v1:video-track:1",
            registered_dog_id="877d96de-ba43-542d-9523-5c20213bfc09",
            dataset_name="yt-bb-dog",
            image_count=1,
        )
        with self.assertRaises(ValueError):
            IdentityRegistry(records=(rec, rec))

    def test_registry_rejects_duplicate_registered_ids(self) -> None:
        with self.assertRaises(ValueError):
            IdentityRegistry(records=(
                IdentityRegistryRecord(
                    identity_token="a" * 64,
                    dataset_identity_id="yt-bb-dog:v1:video-track:1",
                    registered_dog_id="877d96de-ba43-542d-9523-5c20213bfc09",
                    dataset_name="yt-bb-dog",
                    image_count=1,
                ),
                IdentityRegistryRecord(
                    identity_token="b" * 64,
                    dataset_identity_id="yt-bb-dog:v1:video-track:2",
                    registered_dog_id="877d96de-ba43-542d-9523-5c20213bfc09",
                    dataset_name="yt-bb-dog",
                    image_count=1,
                ),
            ))

    def test_round_trip_json(self) -> None:
        recs = (
            IdentityRegistryRecord(
                identity_token="a" * 64,
                dataset_identity_id="yt-bb-dog:v1:video-track:1",
                registered_dog_id=compute_registered_dog_id(
                    "yt-bb-dog:v1:video-track:1"
                ),
                dataset_name="yt-bb-dog",
                image_count=10,
            ),
            IdentityRegistryRecord(
                identity_token="b" * 64,
                dataset_identity_id="dogfacenet224:v1:web-folder:231",
                registered_dog_id=compute_registered_dog_id(
                    "dogfacenet224:v1:web-folder:231"
                ),
                dataset_name="dogfacenet224",
                image_count=5,
            ),
        )
        registry = IdentityRegistry(records=recs)
        d = registry.to_dict()
        restored = IdentityRegistry.from_dict(d)
        self.assertEqual(len(restored.records), 2)
        self.assertEqual(restored.records[0], recs[0])
        self.assertEqual(restored.records[1], recs[1])


class SqliteRegistryTests(unittest.TestCase):
    _db: Path

    def setUp(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".db", prefix="cvi_reg_test_")
        self._db = Path(path)
        create_registry_database(self._db)

    def tearDown(self) -> None:
        self._db.unlink(missing_ok=True)

    def test_create_tables(self) -> None:
        conn = sqlite3.connect(str(self._db))
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            self.assertIn(("identity_registry",), tables)
        finally:
            conn.close()

    def test_register_and_lookup(self) -> None:
        did = "yt-bb-dog:v1:video-track:42"
        rid = register_identity(self._db, did)
        expected = compute_registered_dog_id(did)
        self.assertEqual(rid, expected)

        token = compute_identity_token(did)
        looked_up = lookup_registered_dog_id(self._db, token)
        self.assertEqual(looked_up, expected)

    def test_register_idempotent_image_count(self) -> None:
        did = "yt-bb-dog:v1:video-track:1"
        rid1 = register_identity(self._db, did)
        rid2 = register_identity(self._db, did)
        self.assertEqual(rid1, rid2)

        conn = sqlite3.connect(str(self._db))
        try:
            row = conn.execute(
                "SELECT image_count FROM identity_registry WHERE identity_token = ?",
                (compute_identity_token(did),),
            ).fetchone()
            self.assertEqual(row[0], 2)
        finally:
            conn.close()

    def test_register_records_bulk(self) -> None:
        dids = [
            "yt-bb-dog:v1:video-track:1",
            "yt-bb-dog:v1:video-track:2",
            "dogfacenet224:v1:web-folder:231",
        ]
        mapping = register_records(self._db, dids)
        self.assertEqual(len(mapping), 3)
        for did in dids:
            token = compute_identity_token(did)
            self.assertIn(token, mapping)
            self.assertEqual(mapping[token], compute_registered_dog_id(did))

    def test_lookup_by_identity_token_full_record(self) -> None:
        did = "mpdd:v1:device-capture:42"
        register_identity(self._db, did)
        token = compute_identity_token(did)
        rec = lookup_by_identity_token(self._db, token)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.dataset_identity_id, did)
        self.assertEqual(rec.registered_dog_id, compute_registered_dog_id(did))
        self.assertEqual(rec.dataset_name, "mpdd")

    def test_lookup_nonexistent(self) -> None:
        self.assertIsNone(
            lookup_registered_dog_id(self._db, "f" * 64)
        )
        self.assertIsNone(
            lookup_by_identity_token(self._db, "f" * 64)
        )

    def test_load_registry_manifest(self) -> None:
        dids = [
            "sibetan:v1:gt-json:dog_A",
            "sibetan:v1:gt-json:dog_B",
        ]
        register_records(self._db, dids)
        registry = load_registry_manifest(self._db)
        self.assertEqual(len(registry.records), 2)
        sorted_dids = sorted(dids)
        self.assertEqual(registry.records[0].dataset_identity_id, sorted_dids[0])
        self.assertEqual(registry.records[1].dataset_identity_id, sorted_dids[1])


class RealWorldIdentityMappingTests(unittest.TestCase):
    """Test registered_dog_id stability for real dataset_identity_id examples."""

    def test_yt_bb_dog_stable_id(self) -> None:
        did = "yt-bb-dog:v1:video-track:1234"
        rid = compute_registered_dog_id(did)
        self.assertIsInstance(rid, str)
        self.assertEqual(len(rid), 36)

    def test_dogfacenet_stable_id(self) -> None:
        did = "dogfacenet224:v1:web-folder:231"
        rid = compute_registered_dog_id(did)
        self.assertEqual(len(rid), 36)

    def test_mpdd_stable_id(self) -> None:
        did = "mpdd:v1:device-capture:42"
        rid = compute_registered_dog_id(did)
        self.assertEqual(len(rid), 36)

    def test_sibetan_stable_id(self) -> None:
        did = "sibetan:v1:gt-json:dog_ABC"
        rid = compute_registered_dog_id(did)
        self.assertEqual(len(rid), 36)

    def test_deterministic_across_sessions(self) -> None:
        did = "yt-bb-dog:v1:video-track:999"
        first = compute_registered_dog_id(did)
        second = compute_registered_dog_id(did)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
