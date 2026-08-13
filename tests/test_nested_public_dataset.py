from __future__ import annotations

import hashlib
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from data.public.nested_public_dataset import (
    ParentBoundNestedArchiveReceipt,
    audit_parent_bound_nested_public_zip,
)
from data.public.public_dataset import (
    DatasetUsageLane,
    PublicDatasetArchivePolicy,
    PublicDatasetSourceContract,
    SourceChecksumAuthority,
    audit_public_dataset_zip,
)
from data.public.public_dataset_extraction import extract_audited_public_dataset_zip


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NestedPublicDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        terms = root / "terms.html"
        terms.write_bytes(b"terms")
        inner = root / "inner.zip"
        with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("dogs/1/a.jpg", b"image")
        outer = root / "outer.zip"
        with zipfile.ZipFile(outer, "w", zipfile.ZIP_STORED) as bundle:
            bundle.write(inner, "package/inner.zip")
        outer_source = PublicDatasetSourceContract(
            dataset_id="outer",
            dataset_version="1",
            archive_filename=outer.name,
            official_page_url="https://example.org/outer",
            archive_url="https://example.org/outer.zip",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            usage_lane=DatasetUsageLane.DEPLOYMENT_ELIGIBLE_CANDIDATE,
            expected_archive_bytes=outer.stat().st_size,
            checksum_authority=SourceChecksumAuthority.PUBLISHED_SHA256,
            expected_sha256=_sha256(outer),
            expected_md5=None,
            terms_snapshot_sha256=_sha256(terms),
        )
        outer_policy = replace(
            PublicDatasetArchivePolicy(),
            allowed_file_suffixes=(".zip",),
        )
        outer_receipt = audit_public_dataset_zip(
            archive_path=outer,
            terms_snapshot_path=terms,
            source=outer_source,
            policy=outer_policy,
        )
        output_parent = root / "datasets"
        output_parent.mkdir()
        output = output_parent / "outer-v1"
        extraction_receipt, files = extract_audited_public_dataset_zip(
            archive_path=outer,
            source=outer_source,
            archive_policy=outer_policy,
            archive_receipt=outer_receipt,
            output_directory=output,
        )
        nested_source = PublicDatasetSourceContract(
            dataset_id="inner",
            dataset_version="parent-bound-1",
            archive_filename=inner.name,
            official_page_url="https://example.org/outer",
            archive_url="https://example.org/outer.zip#package/inner.zip",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            usage_lane=DatasetUsageLane.RESEARCH_ONLY,
            expected_archive_bytes=inner.stat().st_size,
            checksum_authority=(
                SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
            ),
            expected_sha256=None,
            expected_md5=None,
            terms_snapshot_sha256=_sha256(terms),
        )
        return (
            terms,
            output,
            extraction_receipt,
            files,
            nested_source,
            PublicDatasetArchivePolicy(),
        )

    def test_nested_archive_is_bound_to_parent_file_content(self) -> None:
        with TemporaryDirectory() as temporary:
            values = self._fixture(Path(temporary))
            receipt = audit_parent_bound_nested_public_zip(
                parent_output_directory=values[1],
                parent_extraction_receipt=values[2],
                parent_files=values[3],
                parent_member_relative_path="package/inner.zip",
                terms_snapshot_path=values[0],
                nested_source=values[4],
                nested_policy=values[5],
            )
            self.assertEqual(receipt.decision, "PASS_PARENT_BOUND_NESTED_ARCHIVE")
            self.assertEqual(receipt.nested_archive_receipt.regular_files, 1)
            self.assertEqual(
                ParentBoundNestedArchiveReceipt.from_dict(receipt.to_dict()),
                receipt,
            )

    def test_parent_manifest_substitution_and_false_published_claim_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            values = self._fixture(Path(temporary))
            changed_files = (
                replace(values[3][0], content_sha256="0" * 64),
            )
            with self.assertRaisesRegex(ValueError, "manifest digest differs"):
                audit_parent_bound_nested_public_zip(
                    parent_output_directory=values[1],
                    parent_extraction_receipt=values[2],
                    parent_files=changed_files,
                    parent_member_relative_path="package/inner.zip",
                    terms_snapshot_path=values[0],
                    nested_source=values[4],
                    nested_policy=values[5],
                )
            with self.assertRaisesRegex(ValueError, "must not claim"):
                audit_parent_bound_nested_public_zip(
                    parent_output_directory=values[1],
                    parent_extraction_receipt=values[2],
                    parent_files=values[3],
                    parent_member_relative_path="package/inner.zip",
                    terms_snapshot_path=values[0],
                    nested_source=replace(
                        values[4],
                        checksum_authority=SourceChecksumAuthority.PUBLISHED_SHA256,
                        expected_sha256="0" * 64,
                    ),
                    nested_policy=values[5],
                )


if __name__ == "__main__":
    unittest.main()
