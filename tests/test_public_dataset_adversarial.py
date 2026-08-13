from __future__ import annotations

import errno
import hashlib
import os
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import data.public.public_dataset_extraction as extraction_module
import foundation.protected_publication as protected_publication
from data.public.public_dataset import (
    ArchiveSuffixCount,
    DatasetUsageLane,
    PublicDatasetArchivePolicy,
    PublicDatasetSourceContract,
    SourceChecksumAuthority,
    audit_public_dataset_zip,
)
from data.public.public_dataset_extraction import (
    ExtractedPublicDatasetFile,
    extract_audited_public_dataset_zip,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4_096), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_case(
    root: Path,
    *,
    entries: tuple[tuple[str, bytes], ...] = (
        ("dogs/a.jpg", b"image-a"),
        ("dogs/b.png", b"image-b"),
    ),
    archive_name: str = "dogs.zip",
) -> tuple[
    Path,
    Path,
    PublicDatasetSourceContract,
    PublicDatasetArchivePolicy,
]:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / archive_name
    terms = root / "terms.html"
    terms.write_bytes(b"fixture license terms")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for relative_path, payload in entries:
            bundle.writestr(relative_path, payload)
    source = PublicDatasetSourceContract(
        dataset_id="adversarial-fixture",
        dataset_version="1",
        archive_filename=archive.name,
        official_page_url="https://example.org/dataset",
        archive_url="https://example.org/dogs.zip",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        usage_lane=DatasetUsageLane.DEPLOYMENT_ELIGIBLE_CANDIDATE,
        expected_archive_bytes=archive.stat().st_size,
        checksum_authority=SourceChecksumAuthority.PUBLISHED_SHA256,
        expected_sha256=_sha256(archive),
        expected_md5=None,
        terms_snapshot_sha256=_sha256(terms),
    )
    return archive, terms, source, PublicDatasetArchivePolicy()


def _audit_case(
    root: Path,
    *,
    entries: tuple[tuple[str, bytes], ...] = (
        ("dogs/a.jpg", b"image-a"),
        ("dogs/b.png", b"image-b"),
    ),
):
    archive, terms, source, policy = _make_case(root, entries=entries)
    receipt = audit_public_dataset_zip(
        archive_path=archive,
        terms_snapshot_path=terms,
        source=source,
        policy=policy,
    )
    return archive, source, policy, receipt


class PublicDatasetAuditAdversarialTests(unittest.TestCase):
    def test_parent_directory_aba_cannot_mix_hash_and_member_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            original_root = base / "original"
            archive, terms, source, policy = _make_case(original_root)
            clean_receipt = audit_public_dataset_zip(
                archive_path=archive,
                terms_snapshot_path=terms,
                source=source,
                policy=policy,
            )

            alternate_root = base / "alternate"
            _make_case(
                alternate_root,
                entries=(("different/only.jpg", b"alternate"),),
            )
            held_root = base / "held-original"

            def swap_parent_directories(phase: str) -> None:
                if phase == "ARCHIVE_HASHED":
                    original_root.rename(held_root)
                    alternate_root.rename(original_root)
                elif phase == "MEMBERS_SCANNED":
                    original_root.rename(alternate_root)
                    held_root.rename(original_root)

            attacked_receipt = audit_public_dataset_zip(
                archive_path=archive,
                terms_snapshot_path=terms,
                source=source,
                policy=policy,
                audit_phase_callback=swap_parent_directories,
            )
            self.assertEqual(attacked_receipt, clean_receipt)

    def test_unicode_normalization_and_casefold_collisions_are_rejected(self) -> None:
        collision_sets = (
            (
                ("dogs/caf\N{LATIN SMALL LETTER E WITH ACUTE}.jpg", b"a"),
                ("dogs/cafe\N{COMBINING ACUTE ACCENT}.jpg", b"b"),
            ),
            (
                ("dogs/stra\N{LATIN SMALL LETTER SHARP S}e.jpg", b"a"),
                ("dogs/STRASSE.jpg", b"b"),
            ),
        )
        for entries in collision_sets:
            with self.subTest(entries=entries), TemporaryDirectory() as temporary:
                archive, terms, source, policy = _make_case(
                    Path(temporary),
                    entries=entries,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "path collision|not NFC-normalized",
                ):
                    audit_public_dataset_zip(
                        archive_path=archive,
                        terms_snapshot_path=terms,
                        source=source,
                        policy=policy,
                    )

    def test_drvfs_ambiguous_or_reserved_member_paths_are_rejected(self) -> None:
        unsafe_paths = (
            "dogs/a.jpg:payload.txt",
            "dogs./a.jpg",
            "dogs /a.jpg",
            "CON/a.jpg",
            "dogs/aux.jpg",
            "dogs/a<name>.jpg",
        )
        for unsafe_path in unsafe_paths:
            with (
                self.subTest(unsafe_path=unsafe_path),
                TemporaryDirectory() as temporary,
            ):
                archive, terms, source, policy = _make_case(
                    Path(temporary),
                    entries=((unsafe_path, b"image"),),
                )
                with self.assertRaisesRegex(ValueError, "path"):
                    audit_public_dataset_zip(
                        archive_path=archive,
                        terms_snapshot_path=terms,
                        source=source,
                        policy=policy,
                    )

    def test_raw_zip_name_hidden_after_nul_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, terms, source, policy = _make_case(
                root,
                entries=(("dogs/a.jpgXevil.txt", b"image"),),
            )
            raw = archive.read_bytes()
            original_name = b"dogs/a.jpgXevil.txt"
            nul_name = b"dogs/a.jpg\x00evil.txt"
            self.assertEqual(len(original_name), len(nul_name))
            self.assertEqual(raw.count(original_name), 2)
            archive.write_bytes(raw.replace(original_name, nul_name))
            changed_source = replace(
                source,
                expected_archive_bytes=archive.stat().st_size,
                expected_sha256=_sha256(archive),
            )
            with zipfile.ZipFile(archive, "r") as bundle:
                info = bundle.infolist()[0]
                self.assertEqual(info.filename, "dogs/a.jpg")
                self.assertIn("\x00", info.orig_filename)
            with self.assertRaisesRegex(ValueError, "path|NUL|malformed"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=changed_source,
                    policy=policy,
                )

    def test_path_and_expansion_caps_are_applied_before_member_reads(self) -> None:
        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = _make_case(
                Path(temporary),
                entries=(("a/b/c/d/e.jpg", b"image"),),
            )
            with self.assertRaisesRegex(ValueError, "path depth"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=source,
                    policy=replace(policy, maximum_path_depth=4),
                )

        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = _make_case(
                Path(temporary),
                entries=(("dogs/very-long-name.jpg", b"image"),),
            )
            with self.assertRaisesRegex(ValueError, "byte policy"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=source,
                    policy=replace(policy, maximum_path_utf8_bytes=12),
                )

        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = _make_case(Path(temporary))
            with mock.patch.object(
                zipfile.ZipFile,
                "open",
                side_effect=AssertionError("member must not be opened"),
            ) as open_member:
                with self.assertRaisesRegex(ValueError, "member count"):
                    audit_public_dataset_zip(
                        archive_path=archive,
                        terms_snapshot_path=terms,
                        source=source,
                        policy=replace(policy, maximum_members=1),
                    )
            open_member.assert_not_called()

    def test_drvfs_unsafe_archive_basenames_are_rejected_by_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            _, _, source, _ = _make_case(Path(temporary))
            for archive_filename in (
                "CON.zip",
                "dogs.zip ",
                "dogs.zip:payload",
                "nested\\dogs.zip",
            ):
                with self.subTest(archive_filename=archive_filename):
                    with self.assertRaisesRegex(ValueError, "archive_filename"):
                        replace(source, archive_filename=archive_filename)


class PublicDatasetExtractionAdversarialTests(unittest.TestCase):
    def test_receipt_for_another_archive_cannot_be_rebound_to_source(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, source, policy, _ = _audit_case(base / "source-a")
            alternate_archive, _, _, alternate_receipt = _audit_case(
                base / "source-b",
                entries=(
                    ("dogs/a.jpg", b"evil!!!"),
                    ("dogs/b.png", b"altered"),
                ),
            )
            forged_receipt = replace(
                alternate_receipt,
                source_contract_sha256=source.contract_sha256,
            )
            output_parent = base / "datasets"
            output_parent.mkdir()
            output = output_parent / "forged"

            with self.assertRaisesRegex(
                ValueError,
                "source|checksum|SHA-256|byte size",
            ):
                extract_audited_public_dataset_zip(
                    archive_path=alternate_archive,
                    source=source,
                    archive_policy=policy,
                    archive_receipt=forged_receipt,
                    output_directory=output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(
                list(output_parent.glob(".cvi-public-extract-*")),
                [],
            )

    def test_receipt_authority_and_md5_are_reverified(self) -> None:
        receipt_mutators = (
            (
                "checksum-authority",
                lambda receipt: replace(
                    receipt,
                    checksum_authority=SourceChecksumAuthority.PUBLISHED_MD5,
                ),
            ),
            (
                "archive-md5",
                lambda receipt: replace(receipt, archive_md5="0" * 32),
            ),
        )
        for label, mutate_receipt in receipt_mutators:
            with self.subTest(label=label), TemporaryDirectory() as temporary:
                base = Path(temporary)
                archive, source, policy, receipt = _audit_case(base / "input")
                output_parent = base / "datasets"
                output_parent.mkdir()
                output = output_parent / "rebound"
                with self.assertRaisesRegex(
                    ValueError,
                    "authority|MD5|checksum",
                ):
                    extract_audited_public_dataset_zip(
                        archive_path=archive,
                        source=source,
                        archive_policy=policy,
                        archive_receipt=mutate_receipt(receipt),
                        output_directory=output,
                    )
                self.assertFalse(output.exists())

    def test_all_receipt_member_summary_fields_are_reverified(self) -> None:
        receipt_mutators = (
            (
                "maximum-compression-ratio",
                lambda receipt: replace(
                    receipt,
                    maximum_observed_compression_ratio=0.0,
                ),
            ),
            (
                "suffix-counts",
                lambda receipt: replace(
                    receipt,
                    suffix_counts=(ArchiveSuffixCount(".jpg", 2),),
                ),
            ),
        )
        for label, mutate_receipt in receipt_mutators:
            with self.subTest(label=label), TemporaryDirectory() as temporary:
                base = Path(temporary)
                archive, source, policy, receipt = _audit_case(base / "input")
                output_parent = base / "datasets"
                output_parent.mkdir()
                output = output_parent / "forged-summary"
                with self.assertRaisesRegex(
                    ValueError,
                    "member summary|compression ratio|suffix",
                ):
                    extract_audited_public_dataset_zip(
                        archive_path=archive,
                        source=source,
                        archive_policy=policy,
                        archive_receipt=mutate_receipt(receipt),
                        output_directory=output,
                    )
                self.assertFalse(output.exists())

    def test_publication_time_mutation_cannot_receive_pass_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive, source, policy, receipt = _audit_case(base / "input")
            output_parent = base / "datasets"
            output_parent.mkdir()
            output = output_parent / "mutated"
            real_publish = protected_publication.rename_directory_noreplace

            def mutate_then_publish(source_root: Path, target: Path) -> str:
                target_file = source_root / "dogs/a.jpg"
                target_file.write_bytes(b"evil!!!")
                return real_publish(source_root, target)

            with mock.patch.object(
                extraction_module,
                "rename_directory_noreplace",
                side_effect=mutate_then_publish,
            ):
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError),
                    "changed|differs|manifest|hash",
                ):
                    extract_audited_public_dataset_zip(
                        archive_path=archive,
                        source=source,
                        archive_policy=policy,
                        archive_receipt=receipt,
                        output_directory=output,
                    )

    def test_external_file_records_reject_noncanonical_paths(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        unsafe_paths = (
            "../escape.jpg",
            "dogs/../escape.jpg",
            "dogs\\escape.jpg",
            "dogs/cafe\N{COMBINING ACUTE ACCENT}.jpg",
            "dogs/a.jpg:payload",
        )
        for relative_path in unsafe_paths:
            with self.subTest(relative_path=relative_path):
                with self.assertRaisesRegex(ValueError, "relative_path"):
                    ExtractedPublicDatasetFile(
                        relative_path=relative_path,
                        byte_size=1,
                        content_sha256=digest,
                    )

    def test_drvfs_unsafe_output_names_fail_before_staging(self) -> None:
        for output_name in ("CON", "dataset.", "dataset ", "dataset:stream"):
            with self.subTest(output_name=output_name), TemporaryDirectory() as temporary:
                base = Path(temporary)
                archive, source, policy, receipt = _audit_case(base / "input")
                output_parent = base / "datasets"
                output_parent.mkdir()
                with self.assertRaisesRegex(ValueError, "output directory"):
                    extract_audited_public_dataset_zip(
                        archive_path=archive,
                        source=source,
                        archive_policy=policy,
                        archive_receipt=receipt,
                        output_directory=output_parent / output_name,
                    )
                self.assertEqual(
                    list(output_parent.glob(".cvi-public-extract-*")),
                    [],
                )


class ProtectedPublicationAdversarialTests(unittest.TestCase):
    def test_existing_target_is_preserved_without_touching_source(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            target.mkdir()
            (source / "candidate.txt").write_bytes(b"candidate")
            (target / "incumbent.txt").write_bytes(b"incumbent")

            with self.assertRaises(FileExistsError):
                protected_publication.rename_directory_noreplace(source, target)
            self.assertEqual((source / "candidate.txt").read_bytes(), b"candidate")
            self.assertEqual((target / "incumbent.txt").read_bytes(), b"incumbent")

    def test_drvfs_fallback_cleans_reservation_on_rename_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            (source / "complete.txt").write_bytes(b"complete")

            with mock.patch.object(
                protected_publication.os,
                "rename",
                side_effect=OSError(errno.EIO, os.strerror(errno.EIO)),
            ):
                with self.assertRaises(OSError):
                    protected_publication._reserved_empty_directory_rename(
                        source,
                        target,
                    )
            self.assertFalse(target.exists())
            self.assertEqual((source / "complete.txt").read_bytes(), b"complete")

    def test_drvfs_fallback_publishes_complete_directory_once(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            target = base / "target"
            source.mkdir()
            (source / "complete.txt").write_bytes(b"complete")

            strategy = protected_publication._reserved_empty_directory_rename(
                source,
                target,
            )
            self.assertEqual(strategy, "RESERVED_EMPTY_DIRECTORY_RENAME")
            self.assertFalse(source.exists())
            self.assertEqual((target / "complete.txt").read_bytes(), b"complete")


if __name__ == "__main__":
    unittest.main()
