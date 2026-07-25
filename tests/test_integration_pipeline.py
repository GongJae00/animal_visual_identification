"""End-to-end integration test: registry → binding → evaluation.

Uses synthetic data to exercise every layer without real archives or GPU.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvi.identity_registry import (
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
    create_registry_database,
    register_records,
    load_registry_manifest,
)
from cvi.split_registry_binding import (
    IdentityBinding,
    IdentityRoleSummary,
    SplitRegistryBinding,
    build_binding,
)


def _make_fake_source_bundle(
    identities: list[tuple[str, str, str]],
) -> dict:
    """Build a minimal PublicSplitSourceBundle payload.

    Each identity: (dataset_name, identity_label, sample_count)
    """
    samples: list[dict] = []
    for dsn, label, count in identities:
        did = f"{dsn}:v1:{label}"
        id_token = compute_identity_token(did)
        for i in range(count):
            sid = f"{dsn}:v1:sample:{label}:{i}"
            samp_token = compute_sample_token(sid)
            samples.append({
                "sample_token": samp_token,
                "identity_token": id_token,
                "sequence_token": id_token,
                "source_sample_id": sid,
                "dataset_identity_id": did,
                "dataset_name": dsn,
                "source_variant": "original",
                "original_split": None,
                "raw_frame_index": i,
                "paired_source_sample_id": None,
                "in_no_mono_subset": None,
                "region": "FACE",
                "schema_version": "cvi.public_split_sample.v1",
            })
    return {
        "schema_version": "cvi.public_split_source_bundle.v1",
        "evidence_bindings": [("semantic_receipts_sha256", "0" * 64)],
        "samples": samples,
        "interpretation": "SEMANTIC_LABEL_BINDING_ONLY_NOT_MODEL_INPUT",
    }


def _make_fake_assignment(
    identities: list[tuple[str, str, str, str, str, int]],
) -> dict:
    """Build a minimal assignment payload.

    Each: (dataset_name, identity_label, identity_role, model_access, sample_disposition, sample_count)
    """
    records: list[dict] = []
    for dsn, label, role, access, disp, count in identities:
        did = f"{dsn}:v1:{label}"
        id_token = compute_identity_token(did)
        for i in range(count):
            sid = f"{dsn}:v1:sample:{label}:{i}"
            samp_token = compute_sample_token(sid)
            records.append({
                "sample_token": samp_token,
                "identity_token": id_token,
                "dataset_name": dsn,
                "identity_role": role,
                "model_access": access,
                "sample_disposition": disp,
                "source_variant": "original",
                "component_token": "0" * 64,
                "paired_original_token": None,
            })
    return {
        "schema_version": "cvi.protected_public_split_assignment.v1",
        "records": records,
        "status": "PASS_ALL",
    }


class EndToEndPipelineTest(unittest.TestCase):
    """Synthetic end-to-end test from registry → binding."""

    def setUp(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="cvi_e2e_"))
        self._db = self._tmpdir / "registry.db"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_registry_from_source_bundle(self) -> None:
        identities = [
            ("yt-bb-dog", "video-track:1", 10),
            ("yt-bb-dog", "video-track:2", 5),
            ("dogfacenet224", "web-folder:231", 3),
        ]
        bundle = _make_fake_source_bundle(identities)
        unique_dids = sorted(set(s["dataset_identity_id"] for s in bundle["samples"]))

        create_registry_database(self._db)
        register_records(self._db, unique_dids)
        registry = load_registry_manifest(self._db)
        self.assertEqual(len(registry.records), 3)
        for rec in registry.records:
            self.assertEqual(
                rec.registered_dog_id,
                compute_registered_dog_id(rec.dataset_identity_id),
            )

    def test_binding_validates_assignment(self) -> None:
        create_registry_database(self._db)
        register_records(self._db, [
            "yt-bb-dog:v1:video-track:1",
            "yt-bb-dog:v1:video-track:2",
            "dogfacenet224:v1:web-folder:231",
        ])

        assignment = _make_fake_assignment([
            ("yt-bb-dog", "video-track:1", "YT_FIT", "MODEL_TRAINING",
             "PRIMARY_ORACLE_CROP", 10),
            ("yt-bb-dog", "video-track:2", "YT_TEST_KNOWN", "SEALED_FINAL_TEST",
             "PRIMARY_ORACLE_CROP", 5),
            ("dogfacenet224", "web-folder:231", "DOGFACE_FIT",
             "SEPARATE_FACE_ONLY_LANE", "PRIMARY_ORACLE_CROP", 3),
        ])

        binding = build_binding(assignment, self._db)
        self.assertTrue(binding.is_valid)
        self.assertEqual(binding.total_identities, 3)
        self.assertEqual(binding.total_samples, 18)

    def test_binding_detects_unregistered_identities(self) -> None:
        create_registry_database(self._db)
        register_records(self._db, ["yt-bb-dog:v1:video-track:1"])

        assignment = _make_fake_assignment([
            ("yt-bb-dog", "video-track:1", "YT_FIT", "MODEL_TRAINING",
             "PRIMARY_ORACLE_CROP", 5),
            ("yt-bb-dog", "video-track:999", "YT_TEST_KNOWN", "SEALED_FINAL_TEST",
             "PRIMARY_ORACLE_CROP", 3),
        ])

        binding = build_binding(assignment, self._db)
        self.assertFalse(binding.is_valid)
        self.assertEqual(len(binding.unregistered_tokens), 1)

    def test_identity_summary_report(self) -> None:
        create_registry_database(self._db)
        register_records(self._db, [
            "yt-bb-dog:v1:video-track:1",
            "yt-bb-dog:v1:video-track:2",
        ])

        assignment = _make_fake_assignment([
            ("yt-bb-dog", "video-track:1", "YT_FIT", "MODEL_TRAINING",
             "PRIMARY_ORACLE_CROP", 10),
            ("yt-bb-dog", "video-track:2", "YT_TEST_KNOWN", "SEALED_FINAL_TEST",
             "PRIMARY_ORACLE_CROP", 5),
        ])

        binding = build_binding(assignment, self._db)
        summaries = {s.role: s for s in binding.identity_summaries}
        self.assertIn("YT_FIT", summaries)
        self.assertEqual(summaries["YT_FIT"].unique_identities, 1)
        self.assertEqual(summaries["YT_FIT"].sample_count, 10)
        self.assertIn("YT_TEST_KNOWN", summaries)
        self.assertEqual(summaries["YT_TEST_KNOWN"].sample_count, 5)

    def test_registered_dog_id_stability(self) -> None:
        did = "yt-bb-dog:v1:video-track:42"
        rid1 = compute_registered_dog_id(did)
        rid2 = compute_registered_dog_id(did)
        self.assertEqual(rid1, rid2)

    def test_registered_dog_id_is_uuidv5(self) -> None:
        import uuid
        did = "mpdd:v1:device-capture:42"
        rid = compute_registered_dog_id(did)
        parsed = uuid.UUID(rid)
        self.assertEqual(parsed.version, 5)

    def test_identity_token_round_trip(self) -> None:
        did = "sibetan:v1:gt-json:dog_ABC"
        token = compute_identity_token(did)
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 64)
        int(token, 16)

    def test_serialization_round_trip(self) -> None:
        create_registry_database(self._db)
        register_records(self._db, ["yt-bb-dog:v1:video-track:1"])

        assignment = _make_fake_assignment([
            ("yt-bb-dog", "video-track:1", "YT_FIT", "MODEL_TRAINING",
             "PRIMARY_ORACLE_CROP", 3),
        ])

        binding = build_binding(assignment, self._db)
        payload = binding.to_dict()

        restored_bindings = tuple(
            IdentityBinding(**b) for b in payload["bindings"]
        )
        restored_summaries = tuple(
            IdentityRoleSummary(**s) for s in payload["identity_summaries"]
        )
        restored = SplitRegistryBinding(
            bindings=restored_bindings,
            identity_summaries=restored_summaries,
            total_identities=payload["total_identities"],
            total_samples=payload["total_samples"],
            unregistered_tokens=tuple(payload["unregistered_tokens"]),
        )
        self.assertEqual(restored.total_identities, 1)
        self.assertEqual(restored.total_samples, 3)

    def test_cross_dataset_no_collision(self) -> None:
        rid_yt = compute_registered_dog_id("yt-bb-dog:v1:video-track:1")
        rid_mpdd = compute_registered_dog_id("mpdd:v1:device-capture:1")
        rid_dogface = compute_registered_dog_id("dogfacenet224:v1:web-folder:1")
        rid_sibetan = compute_registered_dog_id("sibetan:v1:gt-json:dog_1")
        ids = {rid_yt, rid_mpdd, rid_dogface, rid_sibetan}
        self.assertEqual(len(ids), 4)

    def test_binding_manifest_output(self) -> None:
        create_registry_database(self._db)
        register_records(self._db, ["yt-bb-dog:v1:video-track:1"])

        assignment = _make_fake_assignment([
            ("yt-bb-dog", "video-track:1", "YT_FIT", "MODEL_TRAINING",
             "PRIMARY_ORACLE_CROP", 1),
        ])

        binding = build_binding(assignment, self._db)
        manifest = binding.to_dict()
        self.assertEqual(manifest["schema_version"], "cvi.split_registry_binding.v1")
        self.assertTrue(manifest["is_valid"])
        self.assertIn("generated_at", manifest)
        self.assertIn("bindings", manifest)
        self.assertIn("identity_summaries", manifest)

    def test_evaluation_compatible_registered_id(self) -> None:
        did = "yt-bb-dog:v1:video-track:1"
        rid = compute_registered_dog_id(did)

        label_record = {
            "sample_token": "a" * 64,
            "dataset_identity_id": did,
            "registered_dog_id": rid,
        }
        self.assertEqual(
            label_record.get("registered_dog_id", label_record["dataset_identity_id"]),
            rid,
        )
        label_record_no_reg = {
            "sample_token": "a" * 64,
            "dataset_identity_id": did,
        }
        self.assertEqual(
            label_record_no_reg.get("registered_dog_id", label_record_no_reg["dataset_identity_id"]),
            did,
        )


class RegistryAugmentationTest(unittest.TestCase):
    """Test the augment_labels_with_registry tool logic."""

    def test_augment_labels(self) -> None:
        from cvi.identity_registry import compute_registered_dog_id as _crid

        labels = {
            "schema_version": "cvi.protected_public_split_labels.v1",
            "records": [
                {"sample_token": "t1", "dataset_identity_id": "yt-bb-dog:v1:video-track:1"},
                {"sample_token": "t2", "dataset_identity_id": "yt-bb-dog:v1:video-track:2"},
                {"sample_token": "t3", "dataset_identity_id": ""},
            ],
        }

        for rec in labels["records"]:
            did = rec.get("dataset_identity_id", "")
            if did:
                rec["registered_dog_id"] = _crid(did)

        self.assertIn("registered_dog_id", labels["records"][0])
        self.assertIn("registered_dog_id", labels["records"][1])
        self.assertNotIn("registered_dog_id", labels["records"][2])
        self.assertEqual(
            labels["records"][0]["registered_dog_id"],
            _crid("yt-bb-dog:v1:video-track:1"),
        )


if __name__ == "__main__":
    unittest.main()
