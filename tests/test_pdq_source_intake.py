from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import cvi.pdq_source_intake as pdq_intake_module
from cvi.pdq_source_intake import (
    PdqSelectedSourceMember,
    PdqSourceContract,
    PdqSourceIntakePolicy,
    PdqSourceIntakeReceipt,
    audit_pdq_source_archive,
    publish_pdq_source_bundle,
)
from cvi.pretrained_supporting_asset_intake import (
    MAXIMUM_ASSET_BYTES,
    MAXIMUM_JSON_ARRAY_LENGTH,
    MAXIMUM_JSON_DEPTH,
    MAXIMUM_JSON_KEYS,
    MAXIMUM_JSON_NODES,
    MAXIMUM_JSON_NUMBER_CHARACTERS,
    MAXIMUM_JSON_STRING_CHARACTERS,
)


_COMMIT = "1" * 40
_TREE = "2" * 40


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _member(path: str, payload: bytes) -> PdqSelectedSourceMember:
    return PdqSelectedSourceMember(
        relative_path=path,
        expected_bytes=len(payload),
        git_blob_sha1=_git_blob_sha1(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_tar(
    path: Path,
    entries: tuple[tuple[str, bytes, bytes], ...],
) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, payload, member_type in entries:
            info = tarfile.TarInfo(name)
            info.type = member_type
            if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            else:
                info.linkname = "target"
                bundle.addfile(info)


def _fixture(root: Path) -> tuple[Path, Path, Path, PdqSourceContract]:
    root.mkdir(parents=True, exist_ok=True)
    license_payload = b"synthetic BSD-3-Clause fixture\n"
    source_payload = b"int pdq_fixture() { return 1; }\n"
    selected = (
        _member("LICENSE", license_payload),
        _member("pdq/cpp/hashing/pdqhashing.cpp", source_payload),
    )
    archive_name = f"ThreatExchange-{_COMMIT}.tar.gz.partial"
    archive = root / archive_name
    archive_root = f"ThreatExchange-{_COMMIT}"
    _write_tar(
        archive,
        (
            (f"{archive_root}/LICENSE", license_payload, tarfile.REGTYPE),
            (
                f"{archive_root}/pdq/cpp/CImg.h",
                b"excluded CImg bytes",
                tarfile.REGTYPE,
            ),
            (
                f"{archive_root}/pdq/cpp/io/pdqio.cpp",
                b"excluded IO bytes",
                tarfile.REGTYPE,
            ),
            (
                f"{archive_root}/pdq/cpp/hashing/pdqhashing.cpp",
                source_payload,
                tarfile.REGTYPE,
            ),
        ),
    )
    source = PdqSourceContract(
        repository="facebook/ThreatExchange",
        commit_sha=_COMMIT,
        tree_sha=_TREE,
        official_repository_url="https://github.com/facebook/ThreatExchange",
        commit_api_url=(
            "https://api.github.com/repos/facebook/ThreatExchange/commits/"
            f"{_COMMIT}"
        ),
        tree_api_url=(
            "https://api.github.com/repos/facebook/ThreatExchange/git/trees/"
            f"{_TREE}?recursive=1"
        ),
        codeload_url=(
            "https://codeload.github.com/facebook/ThreatExchange/tar.gz/"
            f"{_COMMIT}"
        ),
        archive_filename=archive_name,
        archive_root=archive_root,
        license_path="LICENSE",
        license_id="BSD-3-Clause",
        license_classification=(
            "MANUALLY_CLASSIFIED_EXACT_ROOT_LICENSE_BSD_3_CLAUSE"
        ),
        license_git_blob_sha1=selected[0].git_blob_sha1,
        license_content_sha256=selected[0].content_sha256,
        license_bytes=selected[0].expected_bytes,
        require_verified_commit=True,
        selected_members=selected,
        forbidden_selected_paths=("pdq/cpp/CImg.h", "pdq/cpp/io"),
        policy=PdqSourceIntakePolicy(
            maximum_archive_bytes=2_000_000,
            maximum_members=100,
            maximum_total_uncompressed_bytes=2_000_000,
            maximum_member_uncompressed_bytes=1_000_000,
            maximum_expansion_ratio=100.0,
            maximum_path_utf8_bytes=512,
            maximum_total_path_utf8_bytes=10_000,
            maximum_path_depth=16,
            maximum_api_snapshot_bytes=100_000,
            read_chunk_bytes=1_024,
        ),
    )
    commit_snapshot = root / "commit-api.json.partial"
    commit_snapshot.write_text(
        json.dumps(
            {
                "sha": _COMMIT,
                "url": source.commit_api_url,
                "html_url": (
                    f"https://github.com/facebook/ThreatExchange/commit/{_COMMIT}"
                ),
                "commit": {
                    "tree": {"sha": _TREE},
                    "verification": {"verified": True, "reason": "valid"},
                },
            }
        ),
        encoding="utf-8",
    )
    tree_snapshot = root / "tree-api.json.partial"
    tree_snapshot.write_text(
        json.dumps(
            {
                "sha": _TREE,
                "truncated": False,
                "tree": [
                    {
                        "path": item.relative_path,
                        "type": "blob",
                        "sha": item.git_blob_sha1,
                        "size": item.expected_bytes,
                    }
                    for item in selected
                ],
            }
        ),
        encoding="utf-8",
    )
    return archive, commit_snapshot, tree_snapshot, source


class PdqSourceIntakeTests(unittest.TestCase):
    def test_api_json_parser_enforces_all_structural_and_numeric_bounds(self) -> None:
        bounded_cases = {
            "JSON bytes must be bounded": (
                b'{"a":"' + b"x" * MAXIMUM_ASSET_BYTES + b'"}'
            ),
            "depth exceeds limit": (
                b'{"a":'
                + b"[" * MAXIMUM_JSON_DEPTH
                + b"0"
                + b"]" * MAXIMUM_JSON_DEPTH
                + b"}"
            ),
            "node count exceeds limit": json.dumps(
                {
                    str(index): [0] * 8_000
                    for index in range(MAXIMUM_JSON_NODES // 8_000 + 1)
                }
            ).encode("utf-8"),
            "key count exceeds limit": json.dumps(
                {str(index): 0 for index in range(MAXIMUM_JSON_KEYS + 1)}
            ).encode("utf-8"),
            "array exceeds limit": json.dumps(
                {"a": [0] * (MAXIMUM_JSON_ARRAY_LENGTH + 1)}
            ).encode("utf-8"),
            "string exceeds limit": json.dumps(
                {"a": "x" * (MAXIMUM_JSON_STRING_CHARACTERS + 1)}
            ).encode("utf-8"),
            "integer token exceeds limit": (
                b'{"a":'
                + b"9" * (MAXIMUM_JSON_NUMBER_CHARACTERS + 1)
                + b"}"
            ),
            "number must be finite": b'{"a":1e999}',
            "unpaired surrogate": b'{"a":"\\ud800"}',
        }
        for message, payload in bounded_cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                pdq_intake_module._parse_strict_json_object(
                    payload,
                    "adversarial API snapshot",
                )

        parser_recursion_payload = (
            b'{"a":' + b"[" * 2_000 + b"0" + b"]" * 2_000 + b"}"
        )
        with self.assertRaisesRegex(
            ValueError,
            "parser bounds|depth exceeds limit",
        ):
            pdq_intake_module._parse_strict_json_object(
                parser_recursion_payload,
                "recursive API snapshot",
            )

    def test_fixed_source_passes_round_trips_and_excludes_cimg_and_io(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, commit_snapshot, tree_snapshot, source = _fixture(root)
            audit = audit_pdq_source_archive(
                archive_path=archive,
                commit_api_snapshot_path=commit_snapshot,
                tree_api_snapshot_path=tree_snapshot,
                source=source,
            )
            self.assertEqual(
                audit.receipt.archive_checksum_authority,
                "OBSERVED_SHA256_ONLY_NO_PUBLISHER_ARCHIVE_CHECKSUM",
            )
            self.assertTrue(audit.receipt.forbidden_selected_paths_absent)
            self.assertEqual(
                PdqSourceContract.from_dict(source.to_dict()),
                source,
            )
            self.assertEqual(
                PdqSourceIntakeReceipt.from_dict(audit.receipt.to_dict()),
                audit.receipt,
            )
            output = root / "published"
            strategy = publish_pdq_source_bundle(
                audit=audit,
                source=source,
                output_directory=output,
                tool_provenance={"fixture": True},
            )
            self.assertIn(strategy, {"RENAMEAT2_NOREPLACE", "RESERVED_EMPTY_DIRECTORY_RENAME"})
            self.assertTrue((output / "source/LICENSE").is_file())
            self.assertFalse((output / "source/pdq/cpp/CImg.h").exists())
            self.assertFalse((output / "source/pdq/cpp/io").exists())
            self.assertTrue((output / "intake-bundle.json").is_file())

    def test_contract_fixes_urls_license_selection_and_schema(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            source = fixture[3]
            with self.assertRaisesRegex(ValueError, "official URL"):
                replace(source, codeload_url="https://example.org/archive.tar.gz")
            with self.assertRaisesRegex(ValueError, "license"):
                replace(source, license_id="MIT")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                replace(
                    source,
                    selected_members=tuple(
                        sorted(
                            source.selected_members
                            + (_member("pdq/cpp/CImg.h", b"bad"),),
                            key=lambda item: item.relative_path.casefold(),
                        )
                    ),
                )
            payload = source.to_dict()
            payload["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "fields differ"):
                PdqSourceContract.from_dict(payload)

    def test_official_commit_tree_signature_and_selected_blob_are_bound(self) -> None:
        mutators = (
            ("commit SHA", lambda payload: payload.update(sha="f" * 40)),
            (
                "commit tree SHA",
                lambda payload: payload["commit"]["tree"].update(sha="f" * 40),
            ),
            (
                "verification",
                lambda payload: payload["commit"]["verification"].update(
                    verified=False, reason="unsigned"
                ),
            ),
        )
        for message, mutate in mutators:
            with self.subTest(message=message), TemporaryDirectory() as temporary:
                fixture = _fixture(Path(temporary))
                payload = json.loads(fixture[1].read_text(encoding="utf-8"))
                mutate(payload)
                fixture[1].write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    audit_pdq_source_archive(
                        archive_path=fixture[0],
                        commit_api_snapshot_path=fixture[1],
                        tree_api_snapshot_path=fixture[2],
                        source=fixture[3],
                    )

        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            payload = json.loads(fixture[2].read_text(encoding="utf-8"))
            payload["truncated"] = True
            fixture[2].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncated"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                )

        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            payload = json.loads(fixture[2].read_text(encoding="utf-8"))
            payload["tree"][0]["sha"] = "f" * 40
            fixture[2].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected tree member"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                )

    def test_traversal_links_devices_collisions_and_reserved_names_fail_closed(self) -> None:
        unsafe = (
            ("traversal", "../escape", tarfile.REGTYPE),
            ("symbolic link", "bad-link", tarfile.SYMTYPE),
            ("hard link", "bad-hardlink", tarfile.LNKTYPE),
            ("device or FIFO", "bad-fifo", tarfile.FIFOTYPE),
            ("reserved", "CON/file.cpp", tarfile.REGTYPE),
            (
                "NFC-normalized",
                "cafe\N{COMBINING ACUTE ACCENT}.cpp",
                tarfile.REGTYPE,
            ),
        )
        for message, suffix, member_type in unsafe:
            with self.subTest(message=message), TemporaryDirectory() as temporary:
                fixture = _fixture(Path(temporary))
                archive_root = fixture[3].archive_root
                _write_tar(
                    fixture[0],
                    (
                        (
                            f"{archive_root}/{suffix}",
                            b"bad",
                            member_type,
                        ),
                    ),
                )
                with self.assertRaisesRegex(ValueError, message):
                    audit_pdq_source_archive(
                        archive_path=fixture[0],
                        commit_api_snapshot_path=fixture[1],
                        tree_api_snapshot_path=fixture[2],
                        source=fixture[3],
                    )

        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            root = fixture[3].archive_root
            _write_tar(
                fixture[0],
                (
                    (f"{root}/pdq/A.cpp", b"a", tarfile.REGTYPE),
                    (f"{root}/pdq/a.cpp", b"b", tarfile.REGTYPE),
                ),
            )
            with self.assertRaisesRegex(ValueError, "path collision"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                )

    def test_caps_and_selected_exact_hashes_fail_before_publication(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            capped = replace(
                fixture[3],
                policy=replace(fixture[3].policy, maximum_members=1),
            )
            with self.assertRaisesRegex(ValueError, "member count"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=capped,
                )

        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            root = fixture[3].archive_root
            license_payload = b"synthetic BSD-3-Clause fixture\n"
            _write_tar(
                fixture[0],
                (
                    (f"{root}/LICENSE", license_payload, tarfile.REGTYPE),
                    (
                        f"{root}/pdq/cpp/hashing/pdqhashing.cpp",
                        b"same byte count but altered!!!!\n",
                        tarfile.REGTYPE,
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "byte size|Git blob|SHA-256"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                )

    def test_retained_archive_fd_parent_and_aba_checks_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            link = root / "commit-link.json.partial"
            link.symlink_to(fixture[1])
            with self.assertRaises(OSError):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=link,
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                )

        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))

            def mutate(_: str) -> None:
                fixture[0].write_bytes(b"X" * fixture[0].stat().st_size)

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=(
                        lambda phase: mutate(phase) if phase == "MEMBERS_SCANNED" else None
                    ),
                )

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            fixture = _fixture(source_root)
            parked = base / "parked"
            replacement = base / "replacement"
            replacement.mkdir()
            (replacement / fixture[0].name).write_bytes(b"replacement")

            def replace_parent(phase: str) -> None:
                if phase == "MEMBERS_SCANNED":
                    os.replace(source_root, parked)
                    os.replace(replacement, source_root)

            with self.assertRaisesRegex(RuntimeError, "parent|path changed"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=replace_parent,
                )

        with TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            parked = fixture[0].with_name("parked.tar.gz.partial")
            replacement = fixture[0].with_name("replacement.tar.gz.partial")
            replacement.write_bytes(b"replacement")

            def replace_then_restore(phase: str) -> None:
                if phase == "MEMBERS_SCANNED":
                    os.replace(fixture[0], parked)
                    os.replace(replacement, fixture[0])
                    os.replace(fixture[0], replacement)
                    os.replace(parked, fixture[0])

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pdq_source_archive(
                    archive_path=fixture[0],
                    commit_api_snapshot_path=fixture[1],
                    tree_api_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=replace_then_restore,
                )

    def test_post_rename_bundle_mutation_cannot_receive_a_successful_return(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            audit = audit_pdq_source_archive(
                archive_path=fixture[0],
                commit_api_snapshot_path=fixture[1],
                tree_api_snapshot_path=fixture[2],
                source=fixture[3],
            )
            output = root / "published"
            real_publish = pdq_intake_module.rename_directory_noreplace

            def mutate_after_publish(stage: Path, target: Path) -> str:
                strategy = real_publish(stage, target)
                (target / "intake-bundle.json").write_bytes(b"corrupt")
                return strategy

            with mock.patch.object(
                pdq_intake_module,
                "rename_directory_noreplace",
                side_effect=mutate_after_publish,
            ), self.assertRaisesRegex(ValueError, "byte size|SHA-256"):
                publish_pdq_source_bundle(
                    audit=audit,
                    source=fixture[3],
                    output_directory=output,
                    tool_provenance={"fixture": True},
                )

    def test_cli_writes_once_and_never_executes_retained_source(self) -> None:
        tool = Path(__file__).parents[1] / "tools/intake_threatexchange_pdq.py"
        help_result = subprocess.run(
            [sys.executable, str(tool), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--commit-api-snapshot", help_result.stdout)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root / "inputs")
            contract = root / "contract.json"
            contract.write_text(json.dumps(fixture[3].to_dict()), encoding="utf-8")
            output = root / "published"
            sentinel = root / "UPSTREAM_EXECUTED"
            command = [
                sys.executable,
                str(tool),
                "--source-contract",
                str(contract),
                "--archive",
                str(fixture[0]),
                "--commit-api-snapshot",
                str(fixture[1]),
                "--tree-api-snapshot",
                str(fixture[2]),
                "--output-directory",
                str(output),
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("PASS_FIXED_COMMIT", first.stdout)
            self.assertFalse(sentinel.exists())
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)

    def test_repository_contract_matches_observed_official_fixed_metadata(self) -> None:
        contract_path = (
            Path(__file__).parents[1]
            / "configs/pdq/threatexchange-pdq-baefb4ed.json"
        )
        contract = PdqSourceContract.from_dict(
            json.loads(contract_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(
            contract.commit_sha,
            "baefb4ed67b6cdc1d4c82dbaef858d50866ac424",
        )
        self.assertEqual(
            contract.tree_sha,
            "fd19aaa19d4503fe8f5107ae36116fe216d27c24",
        )
        self.assertEqual(
            contract.license_content_sha256,
            "68ecc6aafbd2a205a1077f86127030898f03091b7dae9d9017325a8702d8668f",
        )
        selected = {item.relative_path for item in contract.selected_members}
        self.assertNotIn("pdq/cpp/CImg.h", selected)
        self.assertFalse(any(path.startswith("pdq/cpp/io/") for path in selected))


if __name__ == "__main__":
    unittest.main()
