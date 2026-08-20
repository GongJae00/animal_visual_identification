from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import tempfile
import unittest
import uuid
from copy import deepcopy
from pathlib import Path

from enrollment.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    IdentityRegistry,
    IdentityRegistryRecord,
    compute_identity_token,
    compute_public_subject_token,
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

def _record(dataset_identity_id: str, image_count: int = 1) -> IdentityRegistryRecord:
    return IdentityRegistryRecord(
        identity_token=compute_identity_token(dataset_identity_id),
        dataset_identity_id=dataset_identity_id,
        registered_dog_id=compute_registered_dog_id(dataset_identity_id),
        dataset_name=extract_dataset_name(dataset_identity_id),
        image_count=image_count,
    )

def _registry_payload(*dataset_identity_ids: str) -> dict:
    records = sorted(
        (_record(dataset_identity_id) for dataset_identity_id in dataset_identity_ids),
        key=lambda record: (record.dataset_name, record.dataset_identity_id),
    )
    return {
        "schema_version": "cvi.identity_registry.v1",
        "generated_at": "2026-07-26T00:00:00+00:00",
        "namespace_uuid": str(REGISTERED_DOG_NAMESPACE),
        "registrations": [record.to_dict() for record in records],
    }

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

    def test_public_subject_token_is_canonical_and_domain_separated(self) -> None:
        did = "yt-bb-dog:v1:video-track:1234"
        expected = hashlib.sha256(b"public-subject\0" + did.encode()).hexdigest()
        self.assertEqual(compute_public_subject_token(did), expected)
        self.assertNotEqual(
            compute_public_subject_token(did), compute_identity_token(did)
        )

    def test_sequence_token_hashes_identity_when_sequence_absent(self) -> None:
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
        rid = compute_registered_dog_id("sibetan:v1:gt-json:dog_ABC")
        parsed = uuid.UUID(rid)
        self.assertEqual(parsed.version, 5)

    def test_namespace_is_stable(self) -> None:
        self.assertEqual(
            str(REGISTERED_DOG_NAMESPACE),
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

    def test_from_dict_requires_exact_keys_and_types(self) -> None:
        payload = _record("yt-bb-dog:v1:video-track:1").to_dict()
        for changed in (
            {**payload, "unknown": "field"},
            {key: value for key, value in payload.items() if key != "dataset_name"},
            {**payload, "identity_token": 1},
            {**payload, "image_count": True},
        ):
            with self.subTest(changed=changed), self.assertRaises(
                (TypeError, ValueError)
            ):
                IdentityRegistryRecord.from_dict(changed)

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
            _record("dogfacenet224:v1:web-folder:231", 5),
            _record("yt-bb-dog:v1:video-track:1", 10),
        )
        registry = IdentityRegistry(records=recs)
        d = registry.to_dict()
        restored = IdentityRegistry.from_dict(d)
        self.assertEqual(len(restored.records), 2)
        self.assertEqual(restored.records[0], recs[0])
        self.assertEqual(restored.records[1], recs[1])

    def test_manifest_rejects_forged_contract_fields(self) -> None:
        base = _registry_payload("yt-bb-dog:v1:video-track:1")
        invalid_manifests = (
            {**base, "schema_version": "cvi.identity_registry.v2"},
            {**base, "namespace_uuid": str(uuid.uuid4())},
            {**base, "generated_at": 1},
            {**base, "generated_at": "x" * 65},
            {**base, "generated_at": "2026-07-26T00:00:00"},
            {**base, "registrations": tuple(base["registrations"])},
            {**base, "unknown": True},
        )
        for payload in invalid_manifests:
            with self.subTest(payload=payload), self.assertRaises(
                (TypeError, ValueError)
            ):
                IdentityRegistry.from_dict(payload)

    def test_manifest_recomputes_every_registration_field(self) -> None:
        base = _registry_payload("yt-bb-dog:v1:video-track:1")
        registered_id = base["registrations"][0]["registered_dog_id"]
        forged_rehash = compute_registered_dog_id(registered_id)
        mutations = (
            ("identity_token", "forged token", "f" * 64),
            ("dataset_name", "forged dataset", "forged-dataset"),
            ("registered_dog_id", "rehashed UUID", forged_rehash),
            ("registered_dog_id", "UUID4", str(uuid.uuid4())),
            ("registered_dog_id", "uppercase UUID", registered_id.upper()),
            ("image_count", "zero count", 0),
            ("image_count", "negative count", -1),
            ("image_count", "oversized count", 2**63),
        )
        for field, name, value in mutations:
            payload = deepcopy(base)
            payload["registrations"][0][field] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                IdentityRegistry.from_dict(payload)

    def test_manifest_rejects_unknown_record_fields_duplicates_and_order(self) -> None:
        base = _registry_payload(
            "dogfacenet224:v1:web-folder:231",
            "yt-bb-dog:v1:video-track:1",
        )
        unknown = deepcopy(base)
        unknown["registrations"][0]["unknown"] = True
        duplicate = deepcopy(base)
        duplicate["registrations"].append(deepcopy(duplicate["registrations"][0]))
        out_of_order = deepcopy(base)
        out_of_order["registrations"].reverse()
        for payload in (unknown, duplicate, out_of_order):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                IdentityRegistry.from_dict(payload)

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

    def test_sqlite_lookups_and_manifest_reject_forged_rows(self) -> None:
        did = "yt-bb-dog:v1:video-track:forged"
        register_identity(self._db, did)
        token = compute_identity_token(did)
        forged_id = compute_registered_dog_id(compute_registered_dog_id(did))
        conn = sqlite3.connect(str(self._db))
        try:
            conn.execute(
                "UPDATE identity_registry SET registered_dog_id = ? "
                "WHERE identity_token = ?",
                (forged_id, token),
            )
            conn.commit()
        finally:
            conn.close()

        with self.assertRaisesRegex(ValueError, "not deterministic"):
            lookup_registered_dog_id(self._db, token)
        with self.assertRaisesRegex(ValueError, "not deterministic"):
            lookup_by_identity_token(self._db, token)
        with self.assertRaisesRegex(ValueError, "not deterministic"):
            load_registry_manifest(self._db)

    def test_register_rejects_and_rolls_back_malformed_existing_row(self) -> None:
        did = "mpdd:v1:device-capture:malformed"
        register_identity(self._db, did)
        token = compute_identity_token(did)
        conn = sqlite3.connect(str(self._db))
        try:
            conn.execute(
                "UPDATE identity_registry SET dataset_name = ?, image_count = ? "
                "WHERE identity_token = ?",
                ("forged", 7, token),
            )
            conn.commit()
        finally:
            conn.close()

        with self.assertRaisesRegex(ValueError, "dataset_name"):
            register_identity(self._db, did)
        conn = sqlite3.connect(str(self._db))
        try:
            count = conn.execute(
                "SELECT image_count FROM identity_registry WHERE identity_token = ?",
                (token,),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 7)

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

class RegistryBuilderBoundaryTests(unittest.TestCase):
    def test_builder_refuses_existing_database_before_source_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "registry.db"
            database.write_bytes(b"existing")
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "evaluation.commands.evaluate",
                    "registry-build",
                    "--source-bundle",
                    str(root / "missing-source.json"),
                    "--db-output",
                    str(database),
                    "--manifest-output",
                    str(root / "manifest.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("refusing to extend or overwrite", completed.stderr)
            self.assertEqual(database.read_bytes(), b"existing")

if __name__ == "__main__":
    unittest.main()
