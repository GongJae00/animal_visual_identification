from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from data.public.public_canine_manifest import PublicCanineManifest
from data.public.public_canine_semantic_receipt import (
    PublicCanineSemanticReceipt,
    summarize_public_canine_manifest,
)
from tests.test_public_canine_manifest import _record


class PublicCanineSemanticReceiptTests(unittest.TestCase):
    def test_cli_help_executes_the_entry_point(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).parents[1]
                    / "workflows"
                    / "audit_public_canine_semantics.py"
                ),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--archive-receipt", completed.stdout)

        content_completed = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).parents[1]
                    / "workflows"
                    / "audit_public_canine_image_content.py"
                ),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--semantic-receipt", content_completed.stdout)

    def test_record_digest_is_independent_of_archive_member_order(self) -> None:
        first = _record()
        second = replace(
            first,
            source_sample_id="fixture:v1:sample:2",
            member_path="root/1/1.1.jpg",
            member_crc32=2,
        )
        left = PublicCanineManifest("fixture", "v1", "1" * 64, "2" * 64, (first, second))
        right = PublicCanineManifest("fixture", "v1", "1" * 64, "2" * 64, (second, first))
        self.assertEqual(
            summarize_public_canine_manifest(left)[0].record_manifest_sha256,
            summarize_public_canine_manifest(right)[0].record_manifest_sha256,
        )

    def test_receipt_refuses_verified_public_camera_tokens(self) -> None:
        record = _record()
        manifest = PublicCanineManifest("fixture", "v1", "1" * 64, "2" * 64, (record,))
        summary = summarize_public_canine_manifest(manifest)[0]
        with self.assertRaisesRegex(ValueError, "not verified"):
            replace(summary, verified_camera_token_count=1)

    def test_summary_rejects_lookalike_digest_and_noncanonical_counts(self) -> None:
        manifest = PublicCanineManifest(
            "fixture", "v1", "1" * 64, "2" * 64, (_record(),)
        )
        summary = summarize_public_canine_manifest(manifest)[0]
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            replace(summary, record_manifest_sha256="z" * 64)
        with self.assertRaisesRegex(ValueError, "sorted"):
            replace(summary, region_counts=(("Z", 1), ("A", 1)))

    def test_receipt_digest_is_deterministic_and_not_model_admission(self) -> None:
        manifest = PublicCanineManifest("fixture", "v1", "1" * 64, "2" * 64, (_record(),))
        summary = summarize_public_canine_manifest(manifest)[0]
        receipt = PublicCanineSemanticReceipt("fixture", (summary,), (("images", 1),))
        self.assertEqual(receipt.receipt_sha256, receipt.receipt_sha256)
        self.assertIn("NOT_SPLIT_OR_MODEL_ADMISSION", receipt.interpretation)


if __name__ == "__main__":
    unittest.main()
