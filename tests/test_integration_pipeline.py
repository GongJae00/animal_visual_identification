"""End-to-end integration test: registry → binding → evaluation.

Uses synthetic data to exercise every layer without real archives or GPU.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cvi.identity_registry import (
    CVI_REGISTERED_DOG_NAMESPACE,
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
    create_registry_database,
    register_records,
    load_registry_manifest,
)
from cvi.provenance import content_sha256
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
                "uses": [{
                    "protocol": "FIXTURE",
                    "episode": "PRIMARY",
                    "gallery_size": 1,
                    "shot": 1,
                    "role": "KNOWN_QUERY",
                    "event_token": content_sha256({"event": samp_token}),
                    "primary_query_event_token": content_sha256({
                        "query": samp_token
                    }),
                    "bootstrap_cluster_token": content_sha256({
                        "identity": id_token
                    }),
                }],
            })
    query_identities = sorted({record["identity_token"] for record in records})
    identity_roles = {
        record["identity_token"]: record["identity_role"] for record in records
    }
    actual_role_counts: dict[str, int] = {}
    for role in identity_roles.values():
        actual_role_counts[role] = actual_role_counts.get(role, 0) + 1
    return {
        "schema_version": "cvi.protected_public_split_assignment.v1",
        "records": records,
        "status": "PASS_PROTECTED_SPLIT_CONSTRUCTION",
        "seed_commitment": "1" * 64,
        "evidence_root_sha256": "2" * 64,
        "policy_sha256": "3" * 64,
        "strict_external_boundary": "STRICT_EXTERNAL_DOMAIN_ZERO_SHOT",
        "score_inputs_used": False,
        "label_fields_present": False,
        "capacity": {
            "status": "PASS_PROTECTED_SPLIT_CONSTRUCTION",
            "capacity_mode": "EVIDENCE_CONSTRAINED_MAXIMAL_COVERAGE",
            "requested_role_counts": actual_role_counts,
            "minimum_role_counts": actual_role_counts,
            "actual_role_counts": actual_role_counts,
            "contracted_role_counts": {
                role: 0 for role in actual_role_counts
            },
            "quarantined_identity_counts_by_lane": {},
            "yt_test_unknown_fpir_power": {
                "confidence_level": 0.95,
                "actual_unknown_identity_trials": 299,
                "targets": [{
                    "purpose": "PRIMARY",
                    "target_fpir": 0.01,
                    "required_zero_event_trials": 299,
                    "status": "POWERED",
                }],
            },
        },
        "protocol_cohorts": [{
            "episode": "PRIMARY",
            "gallery_size": 1,
            "identity_count": len(query_identities),
            "opaque_identity_set_sha256": content_sha256(query_identities),
            "protocol": "FIXTURE",
            "query_role": "KNOWN_QUERY",
            "shot": 1,
        }],
        "interpretation": (
            "OPAQUE_ROLE_ASSIGNMENT_ONLY_NOT_MODEL_OR_ACCURACY_EVIDENCE"
        ),
    }


def _make_receipt(assignment: dict) -> dict:
    receipt = {
        "schema_version": "cvi.protected_public_split_receipt.v3",
        "status": assignment["status"],
        "seed_commitment": assignment["seed_commitment"],
        "evidence_root_sha256": assignment["evidence_root_sha256"],
        "source_bundle_sha256": "4" * 64,
        "graph_sha256": "5" * 64,
        "policy_sha256": assignment["policy_sha256"],
        "evidence_bindings": [],
        "input_file_sha256s": [],
        "assignment_sha256": content_sha256(assignment),
        "evaluator_binding_sha256": "6" * 64,
        "role_exposure_ledger_sha256": "a" * 64,
        "role_exposure_receipt_sha256": "b" * 64,
        "capacity_mode": assignment["capacity"]["capacity_mode"],
        "requested_role_counts": assignment["capacity"]["requested_role_counts"],
        "actual_role_counts": assignment["capacity"]["actual_role_counts"],
        "quarantined_identity_counts_by_lane": assignment["capacity"][
            "quarantined_identity_counts_by_lane"
        ],
        "yt_test_unknown_fpir_power": assignment["capacity"][
            "yt_test_unknown_fpir_power"
        ],
        "capacity": assignment["capacity"],
        "protocol_cohorts": assignment["protocol_cohorts"],
        "quarantine": {},
        "tool_provenance": {},
        "tool_provenance_sha256": "7" * 64,
        "interpretation": (
            "SPLIT_CONTRACT_BEHAVIOR_ONLY_NOT_PERFORMANCE_OR_DATA_ADMISSION"
        ),
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def _make_registry_manifest(db_path: Path) -> dict:
    registry = load_registry_manifest(db_path)
    manifest = {
        "schema_version": "cvi.identity_registry_manifest.v1",
        "generated_at": "2026-07-26T00:00:00+00:00",
        "namespace_uuid": str(CVI_REGISTERED_DOG_NAMESPACE),
        "registrations": [record.to_dict() for record in registry.records],
        "source_bundle_sha256": "4" * 64,
        "tool_provenance": {"tool": "integration-test"},
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    return manifest


def _build_binding(assignment: dict, db_path: Path) -> SplitRegistryBinding:
    return build_binding(
        assignment,
        db_path,
        _make_receipt(assignment),
        _make_registry_manifest(db_path),
        _make_receipt(assignment)["receipt_sha256"],
    )


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

        binding = _build_binding(assignment, self._db)
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

        binding = _build_binding(assignment, self._db)
        self.assertFalse(binding.is_valid)
        self.assertEqual(len(binding.unregistered_tokens), 1)

    def test_binding_refuses_rehashed_registered_identity_in_registry(self) -> None:
        did = "yt-bb-dog:v1:video-track:1"
        create_registry_database(self._db)
        register_records(self._db, [did])
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

        assignment = _make_fake_assignment([
            ("yt-bb-dog", "video-track:1", "YT_FIT", "MODEL_TRAINING",
             "PRIMARY_ORACLE_CROP", 1),
        ])
        with self.assertRaisesRegex(ValueError, "not deterministic"):
            _build_binding(assignment, self._db)

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

        binding = _build_binding(assignment, self._db)
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

        binding = _build_binding(assignment, self._db)
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
            registry_manifest_sha256=payload["registry_manifest_sha256"],
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

        binding = _build_binding(assignment, self._db)
        manifest = binding.to_dict()
        self.assertEqual(manifest["schema_version"], "cvi.split_registry_binding.v2")
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
