from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cvi.identity_registry import create_registry_database, register_records
from cvi.split_registry_binding import (
    IdentityBinding,
    IdentityRoleSummary,
    SplitRegistryBinding,
    build_binding,
)


def _make_assignment(records: list[dict]) -> dict:
    return {
        "schema_version": "cvi.protected_public_split_assignment.v1",
        "records": records,
        "status": "PASS_ALL",
    }


def _make_registry(db_path: Path, identity_ids: list[str]) -> None:
    create_registry_database(db_path)
    register_records(db_path, identity_ids)


class BuildBindingTests(unittest.TestCase):
    def _db(self) -> Path:
        fd, path = tempfile.mkstemp(suffix=".db", prefix="cvi_split_test_")
        return Path(path)

    def test_empty_assignment(self) -> None:
        db = self._db()
        _make_registry(db, [])
        binding = build_binding(_make_assignment([]), db)
        self.assertTrue(binding.is_valid)
        self.assertEqual(binding.total_identities, 0)
        self.assertEqual(binding.total_samples, 0)
        db.unlink()

    def test_single_identity_bound(self) -> None:
        db = self._db()
        did = "yt-bb-dog:v1:video-track:1"
        _make_registry(db, [did])
        token = "a" * 64

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
        binding = build_binding(assignment, db)
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

        from cvi.identity_registry import compute_identity_token
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
        binding = build_binding(assignment, db)
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
        from cvi.identity_registry import compute_identity_token
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
        binding = build_binding(assignment, db)
        self.assertFalse(binding.is_valid)
        self.assertEqual(len(binding.unregistered_tokens), 1)
        self.assertIn("f" * 64, binding.unregistered_tokens)
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
        )
        self.assertTrue(binding.is_valid)

    def test_invalid_with_unregistered(self) -> None:
        binding = SplitRegistryBinding(
            bindings=(),
            identity_summaries=(),
            total_identities=0,
            total_samples=0,
            unregistered_tokens=("f" * 64,),
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
        )
        d = binding.to_dict()
        self.assertEqual(d["schema_version"], "cvi.split_registry_binding.v1")
        self.assertTrue(d["is_valid"])
        self.assertEqual(len(d["bindings"]), 1)
        self.assertEqual(len(d["identity_summaries"]), 1)


if __name__ == "__main__":
    unittest.main()
