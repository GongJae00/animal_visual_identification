from __future__ import annotations

import hashlib
import os
import unittest
from dataclasses import replace
from pathlib import Path

from data.public_canine_manifest import (
    DOGFACE_DATASET,
    DOGFACE_POLICY,
    MPDD_DATASET,
    MPDD_POLICY,
    SIBETAN_DATASET,
    SIBETAN_POLICY,
    YT_DATASET,
    YT_ORIGINAL_POLICY,
    YT_RANDOM_BACKGROUND_POLICY,
    ArchiveReceiptBinding,
    CanineRegion,
    IdentitySemantics,
    PublicCanineManifest,
    PublicCanineRecord,
    parse_dogfacenet224,
    parse_mpdd,
    parse_sibetan,
    parse_yt_bb_dog,
)

_SECURE_ROOT = Path(os.environ.get("CANINE_IDENTITY_PUBLIC_CANINE_SECURE_ROOT") or os.devnull)
_RECEIPT_DIGEST = hashlib.sha256(b"focused semantic parser test receipt").hexdigest()


def _binding(dataset_name: str, archive: Path) -> ArchiveReceiptBinding:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return ArchiveReceiptBinding(dataset_name, digest.hexdigest(), _RECEIPT_DIGEST)


def _record() -> PublicCanineRecord:
    return PublicCanineRecord(
        dataset_name="fixture",
        dataset_version="v1",
        source_variant="original",
        source_sample_id="fixture:v1:sample:1",
        dataset_identity_id="fixture:v1:web-folder:1",
        identity_semantics=IdentitySemantics.WEB_FOLDER,
        region=CanineRegion.FACE,
        original_split=None,
        sequence_id=None,
        camera_token=None,
        camera_token_verified=False,
        filename_identity_token="1",
        source_cluster_id=None,
        in_no_mono_subset=None,
        paired_source_sample_id=None,
        member_path="root/1/1.0.jpg",
        member_crc32=1,
        member_uncompressed_bytes=5,
        source_archive_sha256="1" * 64,
        source_archive_receipt_sha256="2" * 64,
    )


class PublicCanineManifestContractTests(unittest.TestCase):
    def test_audited_cardinalities_are_explicit_schema_policy(self) -> None:
        self.assertEqual(
            (DOGFACE_POLICY.image_count, DOGFACE_POLICY.identity_count),
            (8_363, 1_393),
        )
        self.assertEqual(
            (MPDD_POLICY.image_count, MPDD_POLICY.identity_count),
            (1_657, 191),
        )
        self.assertEqual(dict(MPDD_POLICY.split_image_counts)["query"], 104)
        self.assertEqual(
            (SIBETAN_POLICY.image_count, SIBETAN_POLICY.identity_count),
            (1_755, 59),
        )
        self.assertEqual(
            (YT_ORIGINAL_POLICY.image_count, YT_ORIGINAL_POLICY.identity_count),
            (27_036, 2_723),
        )
        self.assertEqual(
            (
                YT_RANDOM_BACKGROUND_POLICY.image_count,
                YT_RANDOM_BACKGROUND_POLICY.identity_count,
            ),
            (7_064, 723),
        )

    def test_metadata_is_not_exposed_as_visual_model_input(self) -> None:
        record = _record()
        self.assertEqual(record.visual_model_input_fields, ())
        self.assertNotEqual(record.member_path, "")
        self.assertIn("web-folder", record.dataset_identity_id)

    def test_strict_binding_namespace_and_unverified_camera_contract(self) -> None:
        record = _record()
        with self.assertRaisesRegex(ValueError, "dataset-namespaced"):
            replace(record, dataset_identity_id="registered-dog:1")
        with self.assertRaisesRegex(ValueError, "unverified"):
            replace(record, camera_token="c1", camera_token_verified=True)
        with self.assertRaisesRegex(TypeError, "IdentitySemantics"):
            replace(record, identity_semantics="WEB_FOLDER")
        with self.assertRaisesRegex(ValueError, "record differs"):
            PublicCanineManifest(
                "fixture",
                "v1",
                "3" * 64,
                "2" * 64,
                (record,),
            )


@unittest.skipUnless(
    (_SECURE_ROOT / "dogfacenet-v1" / "DogFaceNet_224resized.zip").is_file(),
    "secure public canine archives are not mounted",
)
class AuditedArchiveIntegrationTests(unittest.TestCase):
    def test_dogfacenet_exact_policy_and_published_split_files(self) -> None:
        root = _SECURE_ROOT / "dogfacenet-v1"
        archive = root / "DogFaceNet_224resized.zip"
        result = parse_dogfacenet224(
            archive_path=archive,
            binding=_binding(DOGFACE_DATASET, archive),
            classes_train_path=root / "classes_train.txt",
            classes_test_path=root / "classes_test.txt",
        )
        self.assertEqual((result.manifest.image_count, result.manifest.identity_count), (8_363, 1_393))
        self.assertEqual(result.basename_identity_mismatches, 17)
        self.assertEqual(
            (result.class_split_receipt.train_lines, result.class_split_receipt.train_identities),
            (7_666, 1_254),
        )
        self.assertEqual(
            (result.class_split_receipt.test_lines, result.class_split_receipt.test_identities),
            (697, 139),
        )
        self.assertEqual(
            {record.identity_semantics for record in result.manifest.records},
            {IdentitySemantics.WEB_FOLDER},
        )

    def test_mpdd_exact_policy_and_optional_underscore_member(self) -> None:
        archive = _SECURE_ROOT / "mpdd-v1" / "MPDD.zip"
        manifest = parse_mpdd(
            archive_path=archive,
            binding=_binding(MPDD_DATASET, archive),
        )
        self.assertEqual((manifest.image_count, manifest.identity_count), (1_657, 191))
        anomaly = [
            record
            for record in manifest.records
            if record.member_path == "MPDD/pytorch/query/146_c1_s3_1.jpg"
        ]
        self.assertEqual(len(anomaly), 1)
        self.assertEqual(anomaly[0].identity_semantics, IdentitySemantics.DEVICE_CAPTURE)
        self.assertFalse(anomaly[0].camera_token_verified)

    def test_sibetan_cluster_to_gt_identity_and_no_mono_policy(self) -> None:
        archive = _SECURE_ROOT / "sibetan-v1" / "Sibetan.zip"
        result = parse_sibetan(
            archive_path=archive,
            binding=_binding(SIBETAN_DATASET, archive),
        )
        self.assertEqual((result.manifest.image_count, result.manifest.identity_count), (1_755, 59))
        self.assertEqual(
            (
                result.cluster_count,
                result.no_mono_cluster_count,
                result.no_mono_identity_count,
                result.no_mono_image_count,
            ),
            (223, 203, 39, 1_603),
        )
        self.assertEqual(
            len({record.sequence_id for record in result.manifest.records}), 223
        )
        self.assertEqual(
            {record.identity_semantics for record in result.manifest.records},
            {IdentitySemantics.GT_JSON},
        )

    def test_yt_original_and_paired_random_background_policy(self) -> None:
        archive = _SECURE_ROOT / "yt-bb-dog-v1" / "YT-BB-dog.zip"
        result = parse_yt_bb_dog(
            archive_path=archive,
            binding=_binding(YT_DATASET, archive),
        )
        self.assertEqual((result.original.image_count, result.original.identity_count), (27_036, 2_723))
        train_ids = {
            record.dataset_identity_id
            for record in result.original.records
            if record.original_split == "train"
        }
        test_ids = {
            record.dataset_identity_id
            for record in result.original.records
            if record.original_split == "test"
        }
        self.assertEqual((len(train_ids), len(test_ids), len(train_ids & test_ids)), (2_000, 723, 0))
        self.assertEqual(
            (result.random_background.image_count, result.paired_test_images, result.missing_random_background_images),
            (7_064, 7_064, 40),
        )
        self.assertTrue(
            all(record.paired_source_sample_id for record in result.random_background.records)
        )
        self.assertEqual(
            {record.identity_semantics for record in result.original.records},
            {IdentitySemantics.VIDEO_TRACK},
        )


if __name__ == "__main__":
    unittest.main()
