from __future__ import annotations

import hashlib
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.public_dataset import (
    DatasetUsageLane,
    PublicDatasetArchivePolicy,
    PublicDatasetSourceContract,
    SourceChecksumAuthority,
    audit_public_dataset_zip,
)
from cvi.public_dataset_extraction import (
    PublicDatasetExtractionReceipt,
    extract_audited_public_dataset_zip,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicDatasetExtractionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "dogs.zip"
        terms = root / "terms.html"
        terms.write_bytes(b"fixture terms")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("dogs/", b"")
            bundle.writestr("dogs/a.jpg", b"image-a")
            bundle.writestr("dogs/nested/b.png", b"image-b")
        source = PublicDatasetSourceContract(
            dataset_id="fixture",
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
        policy = PublicDatasetArchivePolicy()
        archive_receipt = audit_public_dataset_zip(
            archive_path=archive,
            terms_snapshot_path=terms,
            source=source,
            policy=policy,
        )
        return archive, source, policy, archive_receipt

    def test_exact_receipt_bound_archive_is_published_with_file_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, source, policy, archive_receipt = self._fixture(root / "input")
            output_parent = root / "datasets"
            output_parent.mkdir()
            output = output_parent / "fixture-v1"
            receipt, files = extract_audited_public_dataset_zip(
                archive_path=archive,
                source=source,
                archive_policy=policy,
                archive_receipt=archive_receipt,
                output_directory=output,
            )
            self.assertEqual(receipt.extracted_regular_files, 2)
            self.assertEqual(receipt.extracted_bytes, 14)
            self.assertEqual(
                [(item.relative_path, item.byte_size) for item in files],
                [("dogs/a.jpg", 7), ("dogs/nested/b.png", 7)],
            )
            self.assertEqual((output / "dogs/a.jpg").read_bytes(), b"image-a")
            self.assertEqual((output / "dogs/nested/b.png").read_bytes(), b"image-b")
            self.assertEqual(
                PublicDatasetExtractionReceipt.from_dict(receipt.to_dict()),
                receipt,
            )

    def test_existing_output_and_symlink_boundaries_fail_before_publish(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, source, policy, archive_receipt = self._fixture(root / "input")
            output_parent = root / "datasets"
            output_parent.mkdir()
            existing = output_parent / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                extract_audited_public_dataset_zip(
                    archive_path=archive,
                    source=source,
                    archive_policy=policy,
                    archive_receipt=archive_receipt,
                    output_directory=existing,
                )

            linked_parent = root / "linked-datasets"
            linked_parent.symlink_to(output_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                extract_audited_public_dataset_zip(
                    archive_path=archive,
                    source=source,
                    archive_policy=policy,
                    archive_receipt=archive_receipt,
                    output_directory=linked_parent / "new",
                )

            archive_link_parent = root / "linked-archive"
            archive_link_parent.mkdir()
            archive_link = archive_link_parent / archive.name
            archive_link.symlink_to(archive)
            with self.assertRaisesRegex(ValueError, "real regular file"):
                extract_audited_public_dataset_zip(
                    archive_path=archive_link,
                    source=source,
                    archive_policy=policy,
                    archive_receipt=archive_receipt,
                    output_directory=output_parent / "new",
                )

    def test_receipt_substitution_and_post_audit_archive_change_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, source, policy, archive_receipt = self._fixture(root / "input")
            output_parent = root / "datasets"
            output_parent.mkdir()
            substituted = replace(
                archive_receipt,
                source_contract_sha256="0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "source contract differs"):
                extract_audited_public_dataset_zip(
                    archive_path=archive,
                    source=source,
                    archive_policy=policy,
                    archive_receipt=substituted,
                    output_directory=output_parent / "substituted",
                )
            self.assertFalse((output_parent / "substituted").exists())

            with archive.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "size differs from receipt"):
                extract_audited_public_dataset_zip(
                    archive_path=archive,
                    source=source,
                    archive_policy=policy,
                    archive_receipt=archive_receipt,
                    output_directory=output_parent / "changed",
                )
            self.assertFalse((output_parent / "changed").exists())
            self.assertEqual(
                list(output_parent.glob(".cvi-public-extract-*")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
