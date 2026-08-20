from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from evaluation.splits.protected_public_split import PublicSplitSample, PublicSplitSourceBundle
from archive.shared_helpers.commands import evaluate_external_appearance as tool

def _token(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def _use(protocol: str, episode: str, shot: int, role: str, value: str) -> dict:
    return {
        "protocol": protocol,
        "episode": episode,
        "gallery_size": 0,
        "shot": shot,
        "role": role,
        "event_token": _token(f"event:{value}"),
        "primary_query_event_token": (
            _token(f"primary:{value}") if role == "KNOWN_QUERY" else None
        ),
        "bootstrap_cluster_token": (
            _token(f"cluster:{value.split(':')[0]}") if role == "KNOWN_QUERY" else None
        ),
    }

def _assignment_fixture() -> dict:
    records: dict[str, dict] = {}

    def add(
        protocol: str,
        dataset: str,
        identity: str,
        sample: str,
        episode: str,
        shot: int,
        role: str,
    ) -> None:
        sample_token = _token(f"sample:{sample}")
        record = records.setdefault(
            sample_token,
            {
                "sample_token": sample_token,
                "identity_token": _token(f"identity:{identity}"),
                "dataset_name": dataset,
                "uses": [],
            },
        )
        record["uses"].append(
            _use(
                protocol,
                episode,
                shot,
                role,
                f"{identity}:{sample}:{episode}:{shot}:{role}",
            )
        )

    for protocol, dataset in (
        ("DOGFACE_CLOSED_SET", "dogfacenet224"),
        ("MPDD_CLOSED_SET", "mpdd"),
    ):
        for identity in ("one", "two"):
            add(
                protocol,
                dataset,
                identity,
                f"{protocol}-{identity}-g",
                "PRIMARY",
                1,
                "GALLERY",
            )
            add(
                protocol,
                dataset,
                identity,
                f"{protocol}-{identity}-q",
                "PRIMARY",
                1,
                "KNOWN_QUERY",
            )

    protocol = "SIBETAN_CROSS_SEQUENCE"
    for identity in ("one", "two"):
        for episode in ("PRIMARY", "REPEAT"):
            add(
                protocol,
                "sibetan",
                identity,
                f"sib-{identity}-{episode}-g1",
                episode,
                1,
                "GALLERY",
            )
            add(
                protocol,
                "sibetan",
                identity,
                f"sib-{identity}-{episode}-q",
                episode,
                1,
                "KNOWN_QUERY",
            )
        add(
            protocol,
            "sibetan",
            identity,
            f"sib-{identity}-PRIMARY-g1",
            "PRIMARY",
            2,
            "GALLERY",
        )
        add(
            protocol,
            "sibetan",
            identity,
            f"sib-{identity}-PRIMARY-g2",
            "PRIMARY",
            2,
            "GALLERY",
        )
        add(
            protocol,
            "sibetan",
            identity,
            f"sib-{identity}-PRIMARY-q2",
            "PRIMARY",
            2,
            "KNOWN_QUERY",
        )
    return {"records": list(records.values())}

def _source_bundle() -> PublicSplitSourceBundle:
    source_id = "mpdd:v1:device-capture:1:gallery:c1:s1:image:1"
    identity_id = "mpdd:v1:device-capture:1"
    identity_token = tool._opaque_token("identity", identity_id)
    bindings = tuple(
        (name, _token(name))
        for name in (
            "exact_duplicate_graph_sha256",
            "geometric_verifier_sha256",
            "image_content_receipts_sha256",
            "pdq_candidates_sha256",
            "phash_candidates_sha256",
            "review_adjudication_sha256",
            "semantic_receipts_sha256",
        )
    )
    return PublicSplitSourceBundle(
        evidence_bindings=bindings,
        samples=(
            PublicSplitSample(
                sample_token=tool._opaque_token("sample", source_id),
                identity_token=identity_token,
                sequence_token=identity_token,
                source_sample_id=source_id,
                dataset_identity_id=identity_id,
                dataset_name="mpdd",
                source_variant="original",
                original_split="gallery",
                raw_frame_index=1,
                paired_source_sample_id=None,
                in_no_mono_subset=None,
                region="FACE",
            ),
        ),
    )

class ExternalAppearanceToolTests(unittest.TestCase):
    def test_cli_sha256_parser_accepts_exact_lowercase_digest(self) -> None:
        self.assertEqual(tool._parse_sha256("a" * 64), "a" * 64)
        with self.assertRaisesRegex(ValueError, "command-line SHA-256"):
            tool._parse_sha256("a" * 63)

    def test_population_isolation_preserves_episode_shot_and_role(self) -> None:
        populations = tool._build_populations(_assignment_fixture())
        keys = {population.key: population for population in populations}
        self.assertEqual(len(populations), 5)
        primary_one = keys[
            tool.PopulationKey("SIBETAN_CROSS_SEQUENCE", "PRIMARY", 0, 1)
        ]
        primary_two = keys[
            tool.PopulationKey("SIBETAN_CROSS_SEQUENCE", "PRIMARY", 0, 2)
        ]
        repeat_one = keys[tool.PopulationKey("SIBETAN_CROSS_SEQUENCE", "REPEAT", 0, 1)]
        self.assertEqual((len(primary_one.gallery), len(primary_one.queries)), (2, 2))
        self.assertEqual((len(primary_two.gallery), len(primary_two.queries)), (4, 2))
        self.assertEqual((len(repeat_one.gallery), len(repeat_one.queries)), (2, 2))
        self.assertTrue(
            {member.sample_token for member in primary_one.gallery}.isdisjoint(
                member.sample_token for member in repeat_one.gallery
            )
        )

    def test_population_rejects_cross_dataset_protocol_record(self) -> None:
        assignment = _assignment_fixture()
        record = next(
            item
            for item in assignment["records"]
            if any(use["protocol"] == "MPDD_CLOSED_SET" for use in item["uses"])
        )
        record["dataset_name"] = "sibetan"
        with self.assertRaisesRegex(ValueError, "dataset boundary"):
            tool._build_populations(assignment)

    def test_wrong_source_hash_and_schema_fail_closed(self) -> None:
        source = _source_bundle()
        payload = source.to_dict()
        receipt = {
            "source_bundle_sha256": "0" * 64,
            "evidence_bindings": [list(item) for item in source.evidence_bindings],
            "input_file_sha256s": [
                ["source_bundle_payload_sha256", source.bundle_sha256]
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not bind"):
            tool._validate_source_bundle_receipt(payload, receipt)
        with self.assertRaisesRegex(ValueError, "unsupported.*schema"):
            tool._source_spec_from_payload(
                {
                    "schema_version": "cvi.external_appearance_source_spec.v0",
                    "sources": [],
                }
            )
        with self.assertRaisesRegex(ValueError, "assignment schema"):
            tool._validate_split_documents(
                {"schema_version": "wrong"},
                {},
                {},
                payload,
                expected_receipt_sha256="0" * 64,
            )

    def test_source_receipt_uses_semantic_canonical_bundle_hash(self) -> None:
        source = _source_bundle()
        payload = source.to_dict()
        payload["samples"] = list(reversed(payload["samples"]))
        receipt = {
            "source_bundle_sha256": source.bundle_sha256,
            "evidence_bindings": [list(item) for item in source.evidence_bindings],
            "input_file_sha256s": [
                ["source_bundle_payload_sha256", source.bundle_sha256]
            ],
        }
        self.assertEqual(
            tool._validate_source_bundle_receipt(payload, receipt).bundle_sha256,
            source.bundle_sha256,
        )

    def test_wrong_checkpoint_hash_fails_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp).resolve() / "checkpoint.pt"
            checkpoint.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                tool._verify_file_sha256(checkpoint, "0" * 64)

    def test_zip_member_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.jpg", b"unsafe")
            record = SimpleNamespace(
                member_path="../escape.jpg",
                member_crc32=0,
                member_uncompressed_bytes=6,
            )
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    tool._read_member_bytes(archive, record)

    def test_paired_identity_clustered_delta_is_trained_minus_frozen(self) -> None:
        frozen = []
        trained = []
        for identity, frozen_values, trained_values in (
            ("a", (0.0, 0.0), (1.0, 1.0)),
            ("b", (1.0, 1.0), (1.0, 1.0)),
        ):
            for index, (frozen_value, trained_value) in enumerate(
                zip(frozen_values, trained_values, strict=True)
            ):
                shared = {
                    "query_event_token": _token(f"query:{identity}:{index}"),
                    "query_identity_id": identity,
                    "bootstrap_cluster_id": _token(f"cluster:{identity}"),
                }
                frozen.append({**shared, "Rank-1": frozen_value})
                trained.append({**shared, "Rank-1": trained_value})
        first = tool._paired_identity_clustered_delta_ci(
            frozen, trained, metric="Rank-1", resamples=200, seed=7
        )
        second = tool._paired_identity_clustered_delta_ci(
            frozen, trained, metric="Rank-1", resamples=200, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 0.5)
        self.assertEqual(first["direction"], "trained_minus_frozen")
        self.assertTrue(first["paired_query_order_verified"])
        with self.assertRaisesRegex(ValueError, "query order differs"):
            tool._paired_identity_clustered_delta_ci(
                frozen,
                list(reversed(trained)),
                metric="Rank-1",
                resamples=20,
                seed=7,
            )

    def test_cli_help_has_no_filesystem_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = Path("archive/shared_helpers/commands/evaluate_external_appearance.py").resolve()
            completed = subprocess.run(
                [sys.executable, script, "--help"],
                check=True,
                capture_output=True,
                text=True,
                cwd=root,
            )
            self.assertIn("--checkpoint-sha256", completed.stdout)
            self.assertEqual(list(root.iterdir()), [])

if __name__ == "__main__":
    unittest.main()
