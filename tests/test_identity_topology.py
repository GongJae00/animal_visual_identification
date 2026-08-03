from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from experiments.identity_topology import (
    IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
    IDENTITY_TOPOLOGY_REPORT_SCHEMA_VERSION,
    IdentityTopologyConfig,
    IdentityTopologyError,
    audit_identity_topology,
)
from foundation.provenance import content_sha256
from workflows.audit_identity_topology import REPORT_BUNDLE_SCHEMA_VERSION, main


def _record(
    sample: str,
    identity: str,
    session: str,
    branch: str,
    embedding: list[float] | None,
    *,
    available: bool = True,
    quality: float | None = 0.9,
    rank: int | None = None,
) -> dict:
    result = {
        "sample_token": sample,
        "identity_token": identity,
        "session_token": session,
        "branch": branch,
        "quality": quality,
        "available": available,
        "embedding": embedding,
    }
    if rank is not None:
        result["rank_label"] = rank
    return result


def _manifest(*records: dict) -> dict:
    return {
        "schema_version": IDENTITY_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        "records": list(records),
    }


def _two_branch_manifest() -> dict:
    records = []
    ranks = {
        "base": {"a-1": 2, "a-2": 1, "b-1": 1, "b-2": 1},
        "candidate": {"a-1": 1, "a-2": 2, "b-1": 1, "b-2": 1},
    }
    samples = (
        ("a-1", "identity-a", "session-1", [1.0, 0.0]),
        ("a-2", "identity-a", "session-2", [1.0, 0.0]),
        ("b-1", "identity-b", "session-1", [0.0, 1.0]),
        ("b-2", "identity-b", "session-2", [0.0, 1.0]),
    )
    for branch in ("base", "candidate"):
        for sample, identity, session, embedding in samples:
            records.append(
                _record(
                    sample,
                    identity,
                    session,
                    branch,
                    embedding,
                    rank=ranks[branch][sample],
                )
            )
    return _manifest(*records)


class IdentityTopologyMetricTests(unittest.TestCase):
    def test_exact_geometry_connectivity_hubness_and_rank_outcomes(self) -> None:
        report = audit_identity_topology(
            _two_branch_manifest(),
            config=IdentityTopologyConfig(
                connectivity_cosine_distance_threshold=0.01,
                hubness_k=1,
            ),
        )

        self.assertEqual(
            report["schema_version"], IDENTITY_TOPOLOGY_REPORT_SCHEMA_VERSION
        )
        self.assertIn("NOT_PHYSICAL_NOSE_TOPOLOGY", report["interpretation"])
        base = report["branches"]["base"]
        identity_a = base["identities"]["identity-a"]
        self.assertEqual(identity_a["available_sample_count"], 2)
        self.assertEqual(identity_a["session_count"], 2)
        self.assertEqual(identity_a["normalized_prototype_stability"], 1.0)
        self.assertEqual(identity_a["intra_identity_cosine_diameter"], 0.0)
        self.assertEqual(
            identity_a["leave_one_out_prototype_drift"][
                "prototype_cosine_distance"
            ]["maximum"],
            0.0,
        )
        connectivity = identity_a["cross_session_connectivity"]
        self.assertTrue(connectivity["available"])
        self.assertFalse(connectivity["fragmented"])
        self.assertEqual(connectivity["component_count"], 1)
        self.assertEqual(
            identity_a["nearest_impostor"]["prototype_cosine_distance"], 1.0
        )
        self.assertEqual(
            identity_a["nearest_impostor"]["diameter_adjusted_margin"], 1.0
        )
        self.assertEqual(identity_a["hubness"]["neighbor_occurrence_count"], 1)
        self.assertFalse(base["aggregate"]["same_track_only"])

        outcomes = report["branch_rank_outcomes"]
        self.assertTrue(outcomes["available"])
        comparison = outcomes["comparisons"][0]
        self.assertEqual(comparison["paired_sample_count"], 4)
        self.assertEqual(comparison["left_to_right"]["rescue_count"], 1)
        self.assertEqual(comparison["left_to_right"]["break_count"], 1)

    def test_fragmentation_and_same_track_only_are_explicit(self) -> None:
        fragmented = audit_identity_topology(
            _manifest(
                _record("a-1", "a", "s1", "branch", [1.0, 0.0]),
                _record("a-2", "a", "s2", "branch", [0.0, 1.0]),
                _record("b-1", "b", "s1", "branch", [-1.0, 0.0]),
                _record("b-2", "b", "s2", "branch", [0.0, -1.0]),
            ),
            config=IdentityTopologyConfig(
                connectivity_cosine_distance_threshold=0.5
            ),
        )
        cross_session = fragmented["branches"]["branch"]["identities"]["a"][
            "cross_session_connectivity"
        ]
        self.assertTrue(cross_session["fragmented"])
        self.assertEqual(cross_session["component_count"], 2)

        same_track = audit_identity_topology(
            _manifest(
                _record("a-1", "a", "track-a", "branch", [1.0, 0.0]),
                _record("a-2", "a", "track-a", "branch", [1.0, 0.0]),
                _record("b-1", "b", "track-b", "branch", [0.0, 1.0]),
                _record("b-2", "b", "track-b", "branch", [0.0, 1.0]),
            )
        )["branches"]["branch"]
        self.assertTrue(same_track["aggregate"]["same_track_only"])
        self.assertFalse(same_track["aggregate"]["cross_session_connectivity"]["available"])
        self.assertEqual(
            same_track["identities"]["a"]["cross_session_connectivity"]["reason"],
            "AT_LEAST_TWO_SESSIONS_REQUIRED",
        )

    def test_report_is_deterministic_and_json_finite(self) -> None:
        manifest = _two_branch_manifest()
        first = audit_identity_topology(manifest)
        second = audit_identity_topology(manifest)
        self.assertEqual(first, second)
        self.assertEqual(first["provenance"]["input_sha256"], content_sha256(manifest))
        self.assertEqual(
            first["provenance"]["config_sha256"],
            content_sha256(IdentityTopologyConfig().to_dict()),
        )
        json.dumps(first, sort_keys=True, allow_nan=False)

    def test_unavailable_samples_are_counted_but_not_used(self) -> None:
        report = audit_identity_topology(
            _manifest(
                _record("a-1", "a", "s1", "branch", [1.0, 0.0]),
                _record(
                    "a-2",
                    "a",
                    "s2",
                    "branch",
                    None,
                    available=False,
                    quality=None,
                ),
                _record("b-1", "b", "s1", "branch", [0.0, 1.0]),
            )
        )["branches"]["branch"]
        identity_a = report["identities"]["a"]
        self.assertEqual(identity_a["record_count"], 2)
        self.assertEqual(identity_a["available_sample_count"], 1)
        self.assertEqual(identity_a["unavailable_sample_count"], 1)
        self.assertTrue(identity_a["cross_session_connectivity"]["same_track_only"])

    def test_fully_unavailable_branch_and_identity_are_reported(self) -> None:
        report = audit_identity_topology(
            _manifest(
                _record("a", "a", "s1", "available", [1.0, 0.0]),
                _record(
                    "a",
                    "a",
                    "s1",
                    "missing",
                    None,
                    available=False,
                    quality=None,
                ),
                _record(
                    "b",
                    "b",
                    "s2",
                    "available",
                    None,
                    available=False,
                    quality=0.2,
                ),
            )
        )
        missing = report["branches"]["missing"]
        self.assertEqual(missing["aggregate"]["topology_identity_count"], 0)
        self.assertEqual(missing["aggregate"]["available_sample_count"], 0)
        self.assertFalse(
            missing["aggregate"]["normalized_prototype_stability"]["available"]
        )
        unavailable_b = report["branches"]["available"]["identities"]["b"]
        self.assertEqual(unavailable_b["available_sample_count"], 0)
        self.assertFalse(unavailable_b["nearest_impostor"]["available"])


class IdentityTopologyAdversarialTests(unittest.TestCase):
    def test_rejects_nonfinite_unnormalized_and_incompatible_vectors(self) -> None:
        bad_manifests = (
            _manifest(_record("a", "a", "s", "b", [np.nan, 0.0])),
            _manifest(_record("a", "a", "s", "b", [2.0, 0.0])),
            _manifest(
                _record("a", "a", "s", "b", [1.0, 0.0]),
                _record("b", "b", "s", "b", [0.0, 1.0, 0.0]),
            ),
        )
        for manifest in bad_manifests:
            with self.subTest(manifest=manifest), self.assertRaises(
                IdentityTopologyError
            ):
                audit_identity_topology(manifest)

    def test_rejects_duplicate_or_inconsistent_sample_bindings(self) -> None:
        duplicate = _record("a", "identity-a", "s1", "branch", [1.0, 0.0])
        inconsistent = _manifest(
            duplicate,
            _record("a", "identity-b", "s1", "other", [1.0, 0.0]),
        )
        for manifest in (_manifest(duplicate, dict(duplicate)), inconsistent):
            with self.subTest(manifest=manifest), self.assertRaises(
                IdentityTopologyError
            ):
                audit_identity_topology(manifest)

    def test_rejects_missing_or_contradictory_availability_bindings(self) -> None:
        missing = _record("a", "a", "s", "b", None)
        contradictory = _record(
            "a", "a", "s", "b", [1.0, 0.0], available=False
        )
        unavailable_rank = _record(
            "a", "a", "s", "b", None, available=False, rank=1
        )
        for record in (missing, contradictory, unavailable_rank):
            with self.subTest(record=record), self.assertRaises(
                IdentityTopologyError
            ):
                audit_identity_topology(_manifest(record))

    def test_degenerate_identity_prototypes_fail_closed(self) -> None:
        with self.assertRaisesRegex(IdentityTopologyError, "degenerate"):
            audit_identity_topology(
                _manifest(
                    _record("a-1", "a", "s1", "b", [1.0, 0.0]),
                    _record("a-2", "a", "s2", "b", [-1.0, 0.0]),
                )
            )


class IdentityTopologyCliTests(unittest.TestCase):
    def test_cli_writes_bound_canonical_bundle_and_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            output_path = root / "audit.json"
            manifest_path.write_text(
                json.dumps(_two_branch_manifest(), sort_keys=True), encoding="utf-8"
            )
            argv = [
                "audit_identity_topology.py",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
                "--connectivity-threshold",
                "0.1",
            ]
            with patch.object(sys, "argv", argv):
                main()

            raw = output_path.read_text(encoding="utf-8")
            bundle = json.loads(raw)
            self.assertEqual(bundle["schema_version"], REPORT_BUNDLE_SCHEMA_VERSION)
            self.assertEqual(bundle["report_sha256"], content_sha256(bundle["report"]))
            provenance = bundle["report"]["provenance"]
            self.assertEqual(len(provenance["input_sha256"]), 64)
            self.assertEqual(len(provenance["input_file_sha256"]), 64)
            self.assertEqual(len(provenance["config_sha256"]), 64)
            self.assertTrue(
                {
                    "experiments/identity_topology.py",
                    "workflows/audit_identity_topology.py",
                }.issubset(provenance["code_sha256s"]),
            )
            self.assertTrue(raw.endswith("\n"))
            self.assertLess(raw.index('"report"'), raw.index('"report_sha256"'))
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
