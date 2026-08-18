from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from data.source_lock import SOURCE_REGISTRY, admitted_records, get_record
from data.types import DatasetAdmission, UnifiedCanidSample
from identity.registry.generated_identity_registry import create_provisional_identity
from workflows import build_canid_unified_manifest


class CanidRegistryTests(unittest.TestCase):
    def test_all_records_have_valid_canonical_names(self) -> None:
        names = {r.canonical_name for r in SOURCE_REGISTRY}
        self.assertEqual(len(names), len(SOURCE_REGISTRY))

    def test_admitted_records_exclude_blocked(self) -> None:
        for record in admitted_records():
            self.assertNotIn(
                record.admission,
                (
                    DatasetAdmission.BLOCKED_LICENSE,
                    DatasetAdmission.BLOCKED_ACCESS,
                    DatasetAdmission.REJECT_LABEL_QUALITY,
                    DatasetAdmission.REJECT_NOT_CANID,
                ),
            )

    def test_get_record_returns_registry_entry(self) -> None:
        self.assertEqual(get_record("dogfacenet224").canonical_name, "dogfacenet224")
        with self.assertRaises(KeyError):
            get_record("nonexistent")

    def test_sibetan_uses_publisher_validation_metadata(self) -> None:
        record = get_record("sibetan")
        self.assertEqual(record.admission, DatasetAdmission.ADMIT_VALIDATION_ONLY)
        self.assertEqual(record.license_id, "CC-BY-4.0")
        self.assertEqual(record.total_identities, 59)

    def test_mpdd_uses_audited_local_identity_count_and_unverified_groups(self) -> None:
        record = get_record("mpdd")
        self.assertEqual(record.total_identities, 191)
        self.assertEqual(record.capture_group_kind.value, "POSE_VIEW_CLUSTER")

    def test_oxford_is_teacher_only_and_has_no_identity_claim(self) -> None:
        record = get_record("oxford-pets-dog")
        self.assertEqual(record.admission, DatasetAdmission.ADMIT_TEACHER_ONLY)
        self.assertEqual(record.total_images, 4978)
        self.assertEqual(record.total_identities, 0)
        self.assertIn("Research-only", record.license_id)

    def test_petface_remains_blocked_pending_source_binding(self) -> None:
        record = get_record("petface-dog")
        self.assertEqual(record.admission, DatasetAdmission.BLOCKED_ACCESS)
        self.assertEqual(record.sha256_checksums, {})
        data_root = Path(
            os.environ.get(
                "CANINE_IDENTITY_DATA_DIR", Path.home() / "canine_identity_data"
            )
        )
        self.assertEqual(record.data_root, str(data_root / "datasets" / "petface"))
        self.assertNotIn(record, admitted_records())

    def test_dogfacenet_checksum_is_the_audited_archive(self) -> None:
        record = get_record("dogfacenet224")
        self.assertEqual(
            record.sha256_checksums["DogFaceNet_224resized.zip"],
            "b3b335180bfd8d18b17e13601c9b0fa9c7c92bf9c18d64fe2999597f2e71f871",
        )

    def test_acquired_dataset_paths_use_flat_content_directories(self) -> None:
        data_root = Path(
            os.environ.get(
                "CANINE_IDENTITY_DATA_DIR", Path.home() / "canine_identity_data"
            )
        )
        expected = {
            "ap10k-dog": "ap10k",
            "dogflw": "dogflw",
            "dogfacenet224": "dogfacenet224",
            "mpdd": "mpdd",
            "oxford-pets-dog": "oxford-iiit-pet",
            "sibetan": "sibetan",
            "yt-bb-dog": "yt-bb-dog",
        }
        for name, directory in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    Path(get_record(name).data_root),
                    data_root / "datasets" / directory,
                )


class CanidWorkflowRoutingTests(unittest.TestCase):
    def test_only_explicit_first_argument_selects_a_report_route(self) -> None:
        with (
            patch.object(build_canid_unified_manifest, "_build_manifest") as manifest,
            patch.object(build_canid_unified_manifest, "_inspect_datasets") as inspect,
            patch.object(
                build_canid_unified_manifest, "_audit_duplicates"
            ) as duplicates,
        ):
            build_canid_unified_manifest.main(["inspect", "ignored"])
            inspect.assert_called_once_with()
            manifest.assert_not_called()
            duplicates.assert_not_called()

        with (
            patch.object(build_canid_unified_manifest, "_build_manifest") as manifest,
            patch.object(build_canid_unified_manifest, "_inspect_datasets") as inspect,
            patch.object(
                build_canid_unified_manifest, "_audit_duplicates"
            ) as duplicates,
        ):
            arguments = ["--output-dir", "/tmp/manifests", "duplicates"]
            build_canid_unified_manifest.main(arguments)
            manifest.assert_called_once_with(arguments)
            inspect.assert_not_called()
            duplicates.assert_not_called()


class UnifiedCanidSampleTests(unittest.TestCase):
    def test_minimal_sample_passes_validation(self) -> None:
        sample = UnifiedCanidSample(
            sample_id="test-1",
            dataset_name="dogfacenet224",
            dataset_version="v1",
            source_group_id="42",
            image_path="img.jpg",
            image_sha256="a" * 64,
            width=224,
            height=224,
            raw_identity_id="42",
        )
        self.assertEqual(sample.species, "Canis lupus familiaris")
        self.assertTrue(sample.label_availability["identity"])

    def test_bbox_is_validated(self) -> None:
        UnifiedCanidSample(
            sample_id="t",
            dataset_name="d",
            dataset_version="v",
            source_group_id="g",
            image_path="p",
            image_sha256="a" * 64,
            width=1,
            height=1,
            dog_boxes_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        with self.assertRaises(ValueError):
            UnifiedCanidSample(
                sample_id="t",
                dataset_name="d",
                dataset_version="v",
                source_group_id="g",
                image_path="p",
                image_sha256="a" * 64,
                width=1,
                height=1,
                dog_boxes_xyxy=(3.0, 0.0, 1.0, 0.0),
            )

    def test_label_availability_defaults(self) -> None:
        s = UnifiedCanidSample(
            sample_id="t",
            dataset_name="d",
            dataset_version="v",
            source_group_id="g",
            image_path="p",
            image_sha256="a" * 64,
            width=1,
            height=1,
            raw_identity_id="1",
            camera_id="c1",
            nose_mask_path="mask.png",
        )
        self.assertTrue(s.label_availability["identity"])
        self.assertTrue(s.label_availability["camera"])
        self.assertTrue(s.label_availability["nose_mask"])
        self.assertFalse(s.label_availability["breed"])
        self.assertFalse(s.label_availability["dog_bbox"])

    def test_generated_identity_does_not_become_ground_truth_identity(self) -> None:
        generated = create_provisional_identity(
            "cvi.track-cluster:v1", "unlabeled-track:42", 5
        )
        sample = UnifiedCanidSample(
            sample_id="t",
            dataset_name="unlabeled-canid",
            dataset_version="v1",
            source_group_id="g",
            image_path="p",
            image_sha256="a" * 64,
            width=1,
            height=1,
            generated_identity_id=generated.generated_identity_id,
            metadata={
                "generated_identity_generator_id": generated.generator_id,
                "generated_identity_source_cluster_token": generated.source_cluster_token,
            },
        )
        self.assertTrue(sample.label_availability["generated_identity"])
        self.assertFalse(sample.label_availability["identity"])
        self.assertIsNone(sample.registered_identity_id)


if __name__ == "__main__":
    unittest.main()
