from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from enrollment.registry.identity_registry import (
    REGISTERED_DOG_NAMESPACE,
    compute_identity_token,
    create_registry_database,
    load_registry_manifest,
    register_records,
)
from shared.foundation.provenance import content_sha256
from shared.foundation.protected_io import read_strict_json_object
import evaluation.splits.split_registry_binding as split_binding_module
from evaluation.splits.split_registry_binding import (
    IdentityBinding,
    IdentityRoleSummary,
    SplitRegistryBinding,
    build_binding,
)

def _make_assignment(records: list[dict]) -> dict:
    normalized: list[dict] = []
    for index, value in enumerate(records):
        if not value:
            normalized.append(value)
            continue
        record = dict(value)
        sample_token = record["sample_token"]
        if len(sample_token) != 64:
            sample_token = content_sha256({"sample": sample_token})
        record["sample_token"] = sample_token
        record.setdefault("component_token", content_sha256({"component": index}))
        record.setdefault("source_variant", "original")
        record.setdefault("paired_original_token", None)
        record.setdefault("uses", [{
            "protocol": "FIXTURE",
            "episode": "PRIMARY",
            "gallery_size": 1,
            "shot": 1,
            "role": "KNOWN_QUERY",
            "event_token": content_sha256({"event": sample_token}),
            "primary_query_event_token": content_sha256({"query": sample_token}),
            "bootstrap_cluster_token": content_sha256({
                "identity": record["identity_token"]
            }),
        }])
        normalized.append(record)
    query_identities = sorted({
        record["identity_token"] for record in normalized if record
    })
    cohorts = ([{
        "episode": "PRIMARY",
        "gallery_size": 1,
        "identity_count": len(query_identities),
        "opaque_identity_set_sha256": content_sha256(query_identities),
        "protocol": "FIXTURE",
        "query_role": "KNOWN_QUERY",
        "shot": 1,
    }] if query_identities else [{
        "episode": "PRIMARY",
        "gallery_size": 1,
        "identity_count": 1,
        "opaque_identity_set_sha256": "8" * 64,
        "protocol": "FIXTURE",
        "query_role": "KNOWN_QUERY",
        "shot": 1,
    }])
    identity_roles = {
        record["identity_token"]: record["identity_role"]
        for record in normalized
        if record
    }
    actual_role_counts: dict[str, int] = {}
    for role in identity_roles.values():
        actual_role_counts[role] = actual_role_counts.get(role, 0) + 1
    return {
        "schema_version": "evaluation.protected_public_split_assignment.v1",
        "records": normalized,
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
        "protocol_cohorts": cohorts,
        "interpretation": (
            "OPAQUE_ROLE_ASSIGNMENT_ONLY_NOT_MODEL_OR_ACCURACY_EVIDENCE"
        ),
    }

def _make_receipt(assignment: dict, *, version: int = 3) -> dict:
    receipt = {
        "schema_version": f"evaluation.protected_public_split_receipt.v{version}",
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
        "capacity": assignment["capacity"],
        "protocol_cohorts": assignment["protocol_cohorts"],
        "quarantine": {},
        "tool_provenance": {},
        "tool_provenance_sha256": "7" * 64,
        "interpretation": (
            "SPLIT_CONTRACT_BEHAVIOR_ONLY_NOT_PERFORMANCE_OR_DATA_ADMISSION"
        ),
    }
    if version >= 2:
        receipt["role_exposure_ledger_sha256"] = "a" * 64
        receipt["role_exposure_receipt_sha256"] = "b" * 64
    if version == 3:
        for field in (
            "capacity_mode",
            "requested_role_counts",
            "actual_role_counts",
            "quarantined_identity_counts_by_lane",
            "yt_test_unknown_fpir_power",
        ):
            receipt[field] = assignment["capacity"][field]
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt

def _make_registry(db_path: Path, identity_ids: list[str]) -> None:
    create_registry_database(db_path)
    register_records(db_path, identity_ids)

def _make_registry_manifest(db_path: Path, source_bundle_sha256: str = "4" * 64) -> dict:
    registry = load_registry_manifest(db_path)
    manifest = {
        "schema_version": "enrollment.registry_manifest.v1",
        "generated_at": "2026-07-26T00:00:00+00:00",
        "namespace_uuid": str(REGISTERED_DOG_NAMESPACE),
        "registrations": [record.to_dict() for record in registry.records],
        "source_bundle_sha256": source_bundle_sha256,
        "tool_provenance": {"tool": "unit-test"},
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    return manifest

def _build_binding(
    assignment: dict,
    db_path: Path,
    receipt: dict | None = None,
    registry_manifest: dict | None = None,
) -> SplitRegistryBinding:
    return build_binding(
        assignment,
        db_path,
        receipt if receipt is not None else _make_receipt(assignment),
        (
            registry_manifest
            if registry_manifest is not None
            else _make_registry_manifest(db_path)
        ),
        (receipt if receipt is not None else _make_receipt(assignment))[
            "receipt_sha256"
        ],
    )

class BuildBindingTests(unittest.TestCase):
    def _db(self) -> Path:
        fd, path = tempfile.mkstemp(suffix=".db", prefix="split_test_")
        return Path(path)

    def test_empty_assignment(self) -> None:
        db = self._db()
        _make_registry(db, [])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            assignment = _make_assignment([])
            _build_binding(assignment, db, registry_manifest={})
        db.unlink()

    def test_rejects_fabricated_pass_status(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        assignment["status"] = "PASS_NOT_A_REAL_GATE"
        with self.assertRaisesRegex(ValueError, "did not pass"):
            _build_binding(assignment, db, registry_manifest={})
        db.unlink()

    def test_rejects_missing_blindness_attestation(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        assignment.pop("score_inputs_used")
        with self.assertRaisesRegex(ValueError, "schema keys"):
            _build_binding(assignment, db, registry_manifest={})
        db.unlink()

    def test_single_identity_bound(self) -> None:
        db = self._db()
        did = "yt-bb-dog:v1:video-track:1"
        _make_registry(db, [did])

        assignment = _make_assignment([
            {
                "identity_token": "a" * 64,
                "dataset_name": "yt-bb-dog",
                "identity_role": "YT_FIT",
                "model_access": "MODEL_TRAINING",
                "sample_disposition": "PRIMARY_ORACLE_CROP",
                "sample_token": "b" * 64,
            }
        ])
        binding = _build_binding(assignment, db)
        self.assertFalse(
            binding.is_valid,
            "should be invalid: identity_token not in registry",
        )
        self.assertEqual(len(binding.unregistered_tokens), 1)
        db.unlink()

    def test_full_real_data_integration(self) -> None:
        db = self._db()
        did1 = "yt-bb-dog:v1:video-track:1"
        did2 = "yt-bb-dog:v1:video-track:2"
        did3 = "dogfacenet224:v1:web-folder:231"
        _make_registry(db, [did1, did2, did3])

        t1 = compute_identity_token(did1)
        t2 = compute_identity_token(did2)
        t3 = compute_identity_token(did3)

        assignment = _make_assignment([
            {"identity_token": t1, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_FIT", "model_access": "MODEL_TRAINING",
             "sample_disposition": "PRIMARY_ORACLE_CROP",
             "sample_token": "s1"},
            {"identity_token": t1, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_FIT", "model_access": "MODEL_TRAINING",
             "sample_disposition": "PRIMARY_ORACLE_CROP",
             "sample_token": "s2"},
            {"identity_token": t2, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_TEST_KNOWN",
             "model_access": "SEALED_FINAL_TEST",
             "sample_disposition": "PRIMARY_ORACLE_CROP",
             "sample_token": "s3"},
            {"identity_token": t3, "dataset_name": "dogfacenet224",
             "identity_role": "DOGFACE_FIT",
             "model_access": "SEPARATE_FACE_ONLY_LANE",
             "sample_disposition": "PRIMARY_ORACLE_CROP",
             "sample_token": "s4"},
        ])
        binding = _build_binding(assignment, db)
        self.assertTrue(binding.is_valid)
        self.assertEqual(binding.total_identities, 3)
        self.assertEqual(binding.total_samples, 4)

        summaries = {s.role: s for s in binding.identity_summaries}
        self.assertIn("YT_FIT", summaries)
        self.assertEqual(summaries["YT_FIT"].unique_identities, 1)
        self.assertEqual(summaries["YT_FIT"].sample_count, 2)

        self.assertEqual(len(binding.unregistered_tokens), 0)
        db.unlink()

    def test_some_unregistered_tokens(self) -> None:
        db = self._db()
        _make_registry(db, ["yt-bb-dog:v1:video-track:1"])
        t_reg = compute_identity_token("yt-bb-dog:v1:video-track:1")

        assignment = _make_assignment([
            {"identity_token": t_reg, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_FIT", "model_access": "MODEL_TRAINING",
             "sample_disposition": "PRIMARY_ORACLE_CROP",
             "sample_token": "s1"},
            {"identity_token": "f" * 64, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_FIT", "model_access": "MODEL_TRAINING",
             "sample_disposition": "PRIMARY_ORACLE_CROP",
             "sample_token": "s2"},
        ])
        binding = _build_binding(assignment, db)
        self.assertFalse(binding.is_valid)
        self.assertEqual(len(binding.unregistered_tokens), 1)
        self.assertIn("f" * 64, binding.unregistered_tokens)
        db.unlink()

    def test_rejects_receipt_mismatch(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        receipt = _make_receipt(assignment)
        receipt["assignment_sha256"] = "0" * 64
        receipt["receipt_sha256"] = content_sha256({
            key: value for key, value in receipt.items()
            if key != "receipt_sha256"
        })
        with self.assertRaisesRegex(ValueError, "bind the assignment"):
            _build_binding(assignment, db, receipt, registry_manifest={})
        db.unlink()

    def test_receipt_v3_authenticates_exposure_and_capacity_bindings(self) -> None:
        db = self._db()
        did = "yt-bb-dog:v1:video-track:1"
        _make_registry(db, [did])
        assignment = _make_assignment([{
            "identity_token": compute_identity_token(did),
            "dataset_name": "yt-bb-dog",
            "identity_role": "YT_FIT",
            "model_access": "MODEL_TRAINING",
            "sample_disposition": "PRIMARY_ORACLE_CROP",
            "sample_token": "s1",
        }])
        receipt = _make_receipt(assignment, version=3)
        binding = _build_binding(assignment, db, receipt)
        self.assertTrue(binding.is_valid)

        changed_power = deepcopy(receipt)
        changed_power["yt_test_unknown_fpir_power"]["targets"][0]["status"] = (
            "UNDERPOWERED"
        )
        changed_power["receipt_sha256"] = content_sha256({
            key: value for key, value in changed_power.items()
            if key != "receipt_sha256"
        })
        with self.assertRaisesRegex(
            ValueError, "yt_test_unknown_fpir_power differs"
        ):
            _build_binding(assignment, db, changed_power)

        changed = deepcopy(receipt)
        changed.pop("role_exposure_ledger_sha256")
        changed["receipt_sha256"] = content_sha256({
            key: value for key, value in changed.items()
            if key != "receipt_sha256"
        })
        with self.assertRaisesRegex(ValueError, "schema keys differ"):
            _build_binding(assignment, db, changed)
        db.unlink()

    def test_synthetic_receipt_v2_is_not_accepted_as_new_input(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        receipt = _make_receipt(assignment, version=2)
        with self.assertRaisesRegex(ValueError, "not a persisted artifact"):
            _build_binding(assignment, db, receipt, registry_manifest={})
        db.unlink()

    def test_synthetic_receipt_v1_is_not_accepted_as_new_input(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        receipt = _make_receipt(assignment, version=1)
        with self.assertRaisesRegex(ValueError, "not a persisted artifact"):
            _build_binding(assignment, db, receipt, registry_manifest={})
        db.unlink()

    def test_rejects_revoked_receipt(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        receipt = _make_receipt(assignment)
        with patch(
            "evaluation.splits.split_registry_binding._REVOKED_RECEIPT_SHA256S",
            {receipt["receipt_sha256"]},
        ):
            with self.assertRaisesRegex(ValueError, "revoked"):
                _build_binding(assignment, db, receipt, registry_manifest={})
        db.unlink()

    def test_rejects_revoked_assignment_after_receipt_rehash(self) -> None:
        db = self._db()
        assignment = _make_assignment([{}])
        receipt = _make_receipt(assignment)
        receipt["tool_provenance"] = {"rewritten": True}
        receipt["tool_provenance_sha256"] = content_sha256(
            receipt["tool_provenance"]
        )
        receipt["receipt_sha256"] = content_sha256({
            key: value for key, value in receipt.items()
            if key != "receipt_sha256"
        })
        with patch(
            "evaluation.splits.split_registry_binding._REVOKED_RECEIPT_SHA256S", set()
        ), patch(
            "evaluation.splits.split_registry_binding._REVOKED_ASSIGNMENT_SHA256S",
            {receipt["assignment_sha256"]},
        ):
            with self.assertRaisesRegex(ValueError, "revoked"):
                _build_binding(assignment, db, receipt, registry_manifest={})
        db.unlink()

    def test_production_revocation_anchors_are_pinned(self) -> None:
        self.assertIn(
            "b381813ab2ca4d981cfdb73aa6bc103bcd2e129b58e29af9f3eb9020b3ad2c88",
            split_binding_module._REVOKED_RECEIPT_SHA256S,
        )
        self.assertIn(
            "51acf6533e32d6ad69eefee6b3cc5df06ef9f934e979c74116334eace621ab1b",
            split_binding_module._REVOKED_RECEIPT_SHA256S,
        )
        self.assertIn(
            "27e77203764153b52b2a3a207249970624c41fc3b98342d8f0d63c40d4be164d",
            split_binding_module._REVOKED_ASSIGNMENT_SHA256S,
        )

    def test_external_receipt_pin_and_exact_record_contract_fail_closed(self) -> None:
        db = self._db()
        did = "yt-bb-dog:v1:video-track:1"
        _make_registry(db, [did])
        assignment = _make_assignment([{
            "identity_token": compute_identity_token(did),
            "dataset_name": "yt-bb-dog",
            "identity_role": "YT_TEST_KNOWN",
            "model_access": "SEALED_FINAL_TEST",
            "sample_disposition": "PRIMARY_ORACLE_CROP",
            "sample_token": "s1",
        }])
        receipt = _make_receipt(assignment)
        manifest = _make_registry_manifest(db)
        with self.assertRaisesRegex(ValueError, "external pin"):
            build_binding(assignment, db, receipt, manifest, "9" * 64)

        tampered = deepcopy(assignment)
        tampered["records"][0]["model_access"] = "MODEL_TRAINING"
        tampered_receipt = _make_receipt(tampered)
        with self.assertRaisesRegex(ValueError, "role and model access"):
            build_binding(
                tampered,
                db,
                tampered_receipt,
                manifest,
                tampered_receipt["receipt_sha256"],
            )

        tampered = deepcopy(assignment)
        tampered["records"][0]["dataset_identity_id"] = did
        tampered_receipt = _make_receipt(tampered)
        with self.assertRaisesRegex(ValueError, "record schema"):
            build_binding(
                tampered,
                db,
                tampered_receipt,
                manifest,
                tampered_receipt["receipt_sha256"],
            )
        db.unlink()

    def test_registry_manifest_is_bound_to_db_and_split_source(self) -> None:
        db = self._db()
        did = "yt-bb-dog:v1:video-track:1"
        _make_registry(db, [did])
        assignment = _make_assignment([{
            "identity_token": compute_identity_token(did),
            "dataset_name": "yt-bb-dog",
            "identity_role": "YT_FIT",
            "model_access": "MODEL_TRAINING",
            "sample_disposition": "PRIMARY_ORACLE_CROP",
            "sample_token": "s1",
        }])
        manifest = _make_registry_manifest(db)
        binding = _build_binding(assignment, db, registry_manifest=manifest)
        self.assertEqual(
            binding.registry_manifest_sha256, manifest["manifest_sha256"]
        )

        changed = dict(manifest)
        changed["source_bundle_sha256"] = "9" * 64
        changed["manifest_sha256"] = content_sha256({
            key: value for key, value in changed.items()
            if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "source bundle differs"):
            _build_binding(assignment, db, registry_manifest=changed)

        changed = dict(manifest)
        changed["registrations"] = []
        changed["manifest_sha256"] = content_sha256({
            key: value for key, value in changed.items()
            if key != "manifest_sha256"
        })
        with self.assertRaisesRegex(ValueError, "differs from database"):
            _build_binding(assignment, db, registry_manifest=changed)
        db.unlink()

    def test_rejects_identity_in_conflicting_access_lanes(self) -> None:
        db = self._db()
        did = "yt-bb-dog:v1:video-track:1"
        _make_registry(db, [did])
        token = compute_identity_token(did)
        assignment = _make_assignment([
            {"identity_token": token, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_FIT", "model_access": "MODEL_TRAINING",
             "sample_disposition": "PRIMARY_ORACLE_CROP", "sample_token": "s1"},
            {"identity_token": token, "dataset_name": "yt-bb-dog",
             "identity_role": "YT_DEVELOPMENT", "model_access": "MODEL_SELECTION",
             "sample_disposition": "PRIMARY_ORACLE_CROP", "sample_token": "s2"},
        ])
        with self.assertRaisesRegex(ValueError, "conflicting roles"):
            _build_binding(assignment, db)
        db.unlink()

class IdentityBindingContractTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        b = IdentityBinding(
            identity_token="a" * 64,
            registered_dog_id="877d96de-ba43-542d-9523-5c20213bfc09",
            dataset_name="yt-bb-dog",
            identity_role="YT_FIT",
            model_access="MODEL_TRAINING",
            sample_disposition="PRIMARY_ORACLE_CROP",
            sample_count=5,
        )
        d = b.to_dict()
        for k, v in d.items():
            expected = getattr(b, k)
            if isinstance(expected, tuple):
                expected = list(expected)
            self.assertEqual(expected, v, f"field {k} differs")

class IdentityRoleSummaryContractTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        s = IdentityRoleSummary(
            role="YT_FIT",
            access="MODEL_TRAINING",
            unique_identities=100,
            sample_count=500,
        )
        d = s.to_dict()
        for k, v in d.items():
            self.assertEqual(getattr(s, k), v, f"field {k} differs")

class SplitRegistryBindingContractTests(unittest.TestCase):
    def test_valid_checks_unregistered(self) -> None:
        binding = SplitRegistryBinding(
            bindings=(),
            identity_summaries=(),
            total_identities=0,
            total_samples=0,
            unregistered_tokens=(),
            registry_manifest_sha256="a" * 64,
        )
        self.assertFalse(binding.is_valid)

    def test_invalid_with_unregistered(self) -> None:
        binding = SplitRegistryBinding(
            bindings=(),
            identity_summaries=(),
            total_identities=0,
            total_samples=0,
            unregistered_tokens=("f" * 64,),
            registry_manifest_sha256="a" * 64,
        )
        self.assertFalse(binding.is_valid)

    def test_serialize_deserialize(self) -> None:
        binding = SplitRegistryBinding(
            bindings=(
                IdentityBinding(
                    identity_token="a" * 64,
                    registered_dog_id="877d96de-ba43-542d-9523-5c20213bfc09",
                    dataset_name="yt-bb-dog",
                    identity_role="YT_FIT",
                    model_access="MODEL_TRAINING",
                    sample_disposition="PRIMARY_ORACLE_CROP",
                    sample_count=10,
                ),
            ),
            identity_summaries=(
                IdentityRoleSummary(
                    role="YT_FIT", access="MODEL_TRAINING",
                    unique_identities=1, sample_count=10,
                ),
            ),
            total_identities=1,
            total_samples=10,
            unregistered_tokens=(),
            registry_manifest_sha256="a" * 64,
        )
        d = binding.to_dict()
        self.assertEqual(d["schema_version"], "evaluation.split_registry_binding.v2")
        self.assertTrue(d["is_valid"])
        self.assertEqual(len(d["bindings"]), 1)
        self.assertEqual(len(d["identity_summaries"]), 1)

class StrictJsonBoundaryTests(unittest.TestCase):
    def test_reader_rejects_duplicate_keys_symlink_and_non_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_strict_json_object(duplicate)

            valid = root / "valid.json"
            valid.write_text('{"a": 1}', encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(valid)
            with self.assertRaisesRegex(ValueError, "symlink"):
                read_strict_json_object(link)

            with self.assertRaisesRegex(ValueError, "regular file"):
                read_strict_json_object(root)

if __name__ == "__main__":
    unittest.main()
