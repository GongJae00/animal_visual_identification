from __future__ import annotations

import hashlib
import os
import stat
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.public_dataset import (
    DatasetUsageLane,
    PublicDatasetArchivePolicy,
    PublicDatasetArchiveReceipt,
    PublicDatasetSourceContract,
    SourceChecksumAuthority,
    audit_public_dataset_zip,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


class PublicDatasetArchiveTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        entries: tuple[tuple[str, bytes], ...] = (
            ("dataset/dog-001/a.jpg", b"jpeg-a"),
            ("dataset/dog-002/b.PNG", b"png-b"),
        ),
        checksum_authority: SourceChecksumAuthority = (
            SourceChecksumAuthority.PUBLISHED_SHA256
        ),
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> tuple[Path, Path, PublicDatasetSourceContract, PublicDatasetArchivePolicy]:
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "dogs.zip"
        terms = root / "terms.html"
        terms.write_text("CC BY 4.0 fixture", encoding="utf-8")
        with zipfile.ZipFile(archive, "w", compression) as bundle:
            for name, payload in entries:
                bundle.writestr(name, payload)
        expected_sha256 = (
            _sha256(archive)
            if checksum_authority is SourceChecksumAuthority.PUBLISHED_SHA256
            else None
        )
        expected_md5 = (
            _md5(archive)
            if checksum_authority is SourceChecksumAuthority.PUBLISHED_MD5
            else None
        )
        source = PublicDatasetSourceContract(
            dataset_id="fixture-dogs",
            dataset_version="1",
            archive_filename=archive.name,
            official_page_url="https://example.org/dataset",
            archive_url="https://example.org/dogs.zip",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            usage_lane=(
                DatasetUsageLane.RESEARCH_ONLY
                if checksum_authority
                is SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
                else DatasetUsageLane.DEPLOYMENT_ELIGIBLE_CANDIDATE
            ),
            expected_archive_bytes=archive.stat().st_size,
            checksum_authority=checksum_authority,
            expected_sha256=expected_sha256,
            expected_md5=expected_md5,
            terms_snapshot_sha256=_sha256(terms),
        )
        return archive, terms, source, PublicDatasetArchivePolicy()

    def test_published_checksum_archive_is_crc_scanned_and_round_trips(self) -> None:
        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = self._fixture(Path(temporary))
            receipt = audit_public_dataset_zip(
                archive_path=archive,
                terms_snapshot_path=terms,
                source=source,
                policy=policy,
            )
            self.assertEqual(receipt.decision, "PASS_PUBLISHED_CHECKSUM")
            self.assertEqual(receipt.regular_files, 2)
            self.assertEqual(receipt.crc_verified_files, 2)
            self.assertEqual(
                [(item.suffix, item.files) for item in receipt.suffix_counts],
                [(".jpg", 1), (".png", 1)],
            )
            self.assertEqual(
                PublicDatasetSourceContract.from_dict(source.to_dict()),
                source,
            )
            self.assertEqual(
                PublicDatasetArchivePolicy.from_dict(policy.to_dict()),
                policy,
            )
            self.assertEqual(
                PublicDatasetArchiveReceipt.from_dict(receipt.to_dict()),
                receipt,
            )

    def test_md5_and_unpublished_checksum_lanes_are_explicit(self) -> None:
        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = self._fixture(
                Path(temporary),
                checksum_authority=SourceChecksumAuthority.PUBLISHED_MD5,
            )
            receipt = audit_public_dataset_zip(
                archive_path=archive,
                terms_snapshot_path=terms,
                source=source,
                policy=policy,
            )
            self.assertEqual(receipt.checksum_authority, SourceChecksumAuthority.PUBLISHED_MD5)
        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = self._fixture(
                Path(temporary),
                checksum_authority=(
                    SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
                ),
            )
            receipt = audit_public_dataset_zip(
                archive_path=archive,
                terms_snapshot_path=terms,
                source=source,
                policy=policy,
            )
            self.assertEqual(
                receipt.decision,
                "PASS_RESEARCH_INTAKE_SOURCE_CHECKSUM_UNAVAILABLE",
            )
            with self.assertRaisesRegex(ValueError, "only in the research lane"):
                replace(
                    source,
                    usage_lane=DatasetUsageLane.DEPLOYMENT_ELIGIBLE_CANDIDATE,
                )

    def test_checksum_size_terms_and_policy_mismatches_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = self._fixture(Path(temporary))
            cases = (
                (replace(source, expected_archive_bytes=archive.stat().st_size + 1), policy, "byte size"),
                (replace(source, expected_sha256="0" * 64), policy, "SHA-256"),
                (replace(source, terms_snapshot_sha256="0" * 64), policy, "terms snapshot"),
                (source, replace(policy, maximum_archive_bytes=1), "exceeds policy"),
                (source, replace(policy, maximum_members=1), "member count"),
                (source, replace(policy, maximum_total_uncompressed_bytes=1), "expansion"),
            )
            for changed_source, changed_policy, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    audit_public_dataset_zip(
                        archive_path=archive,
                        terms_snapshot_path=terms,
                        source=changed_source,
                        policy=changed_policy,
                    )

    def test_traversal_case_collision_suffix_and_symlink_are_rejected(self) -> None:
        cases: list[tuple[tuple[tuple[str, bytes], ...], str]] = [
            ((("../escape.jpg", b"x"),), "traversal"),
            (
                (("dogs/A.jpg", b"a"), ("dogs/a.JPG", b"b")),
                "path collision",
            ),
            ((("dogs/a.exe", b"x"),), "suffix is not allowed"),
            ((("dogs//a.jpg", b"x"),), "traversal"),
            (
                (("dogs.jpg", b"file"), ("dogs.jpg/a.jpg", b"image")),
                "file-directory conflict",
            ),
            (
                (("dogs.jpg/a.jpg", b"image"), ("DOGS.JPG", b"file")),
                "file-directory conflict",
            ),
        ]
        for entries, message in cases:
            with self.subTest(message=message), TemporaryDirectory() as temporary:
                archive, terms, source, policy = self._fixture(
                    Path(temporary),
                    entries=entries,
                )
                with self.assertRaisesRegex(ValueError, message):
                    audit_public_dataset_zip(
                        archive_path=archive,
                        terms_snapshot_path=terms,
                        source=source,
                        policy=policy,
                    )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "dogs.zip"
            terms = root / "terms.html"
            terms.write_bytes(b"terms")
            info = zipfile.ZipInfo("dogs/link.jpg")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(info, b"target")
            _, _, source, policy = self._fixture(root / "other")
            source = replace(
                source,
                archive_filename=archive.name,
                expected_archive_bytes=archive.stat().st_size,
                expected_sha256=_sha256(archive),
                terms_snapshot_sha256=_sha256(terms),
            )
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=source,
                    policy=policy,
                )

    def test_compression_ratio_and_member_size_are_bounded(self) -> None:
        with TemporaryDirectory() as temporary:
            archive, terms, source, policy = self._fixture(
                Path(temporary),
                entries=(("dogs/zeros.jpg", b"\x00" * 100_000),),
            )
            with self.assertRaisesRegex(ValueError, "compression ratio"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=source,
                    policy=replace(policy, maximum_compression_ratio=2.0),
                )
            with self.assertRaisesRegex(ValueError, "member exceeds"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=source,
                    policy=replace(
                        policy,
                        maximum_member_uncompressed_bytes=10,
                    ),
                )

    def test_crc_corruption_and_archive_mutation_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, terms, source, policy = self._fixture(
                root,
                compression=zipfile.ZIP_STORED,
            )
            payload = bytearray(archive.read_bytes())
            marker = payload.find(b"jpeg-a")
            self.assertGreaterEqual(marker, 0)
            payload[marker] ^= 1
            archive.write_bytes(payload)
            corrupted_source = replace(
                source,
                expected_archive_bytes=archive.stat().st_size,
                expected_sha256=_sha256(archive),
            )
            with self.assertRaises((zipfile.BadZipFile, ValueError)):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=corrupted_source,
                    policy=policy,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, terms, source, policy = self._fixture(root)

            def mutate(phase: str) -> None:
                if phase == "ARCHIVE_HASHED":
                    with archive.open("ab") as stream:
                        stream.write(b"x")
                        stream.flush()
                        os.fsync(stream.fileno())

            with self.assertRaises((RuntimeError, zipfile.BadZipFile, ValueError)):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms,
                    source=source,
                    policy=policy,
                    audit_phase_callback=mutate,
                )

    def test_archive_and_terms_symlinks_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, terms, source, policy = self._fixture(root)
            archive_link = root / "archive-link.zip"
            archive_link.symlink_to(archive)
            linked_source = replace(source, archive_filename=archive_link.name)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                audit_public_dataset_zip(
                    archive_path=archive_link,
                    terms_snapshot_path=terms,
                    source=linked_source,
                    policy=policy,
                )
            terms_link = root / "terms-link.html"
            terms_link.symlink_to(terms)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                audit_public_dataset_zip(
                    archive_path=archive,
                    terms_snapshot_path=terms_link,
                    source=source,
                    policy=policy,
                )


if __name__ == "__main__":
    unittest.main()
