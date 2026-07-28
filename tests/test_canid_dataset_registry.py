from __future__ import annotations

import unittest

from cvi.canid_data.source_lock import SOURCE_REGISTRY, admitted_records, get_record
from cvi.canid_data.types import DatasetAdmission, UnifiedCanidSample, CaptureGroupKind


class CanidRegistryTests(unittest.TestCase):
    def test_all_records_have_valid_canonical_names(self) -> None:
        names = {r.canonical_name for r in SOURCE_REGISTRY}
        self.assertEqual(len(names), len(SOURCE_REGISTRY))

    def test_admitted_records_exclude_blocked(self) -> None:
        for record in admitted_records():
            self.assertNotIn(
                record.admission,
                (DatasetAdmission.BLOCKED_LICENSE, DatasetAdmission.BLOCKED_ACCESS,
                 DatasetAdmission.REJECT_LABEL_QUALITY, DatasetAdmission.REJECT_NOT_CANID),
            )

    def test_get_record_returns_registry_entry(self) -> None:
        self.assertEqual(get_record("dogfacenet224").canonical_name, "dogfacenet224")
        with self.assertRaises(KeyError):
            get_record("nonexistent")

    def test_sibetan_is_blocked_license(self) -> None:
        record = get_record("sibetan")
        self.assertEqual(record.admission, DatasetAdmission.BLOCKED_LICENSE)


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
            sample_id="t", dataset_name="d", dataset_version="v",
            source_group_id="g", image_path="p", image_sha256="a"*64,
            width=1, height=1, dog_boxes_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        with self.assertRaises(ValueError):
            UnifiedCanidSample(
                sample_id="t", dataset_name="d", dataset_version="v",
                source_group_id="g", image_path="p", image_sha256="a"*64,
                width=1, height=1, dog_boxes_xyxy=(3.0, 0.0, 1.0, 0.0),
            )

    def test_label_availability_defaults(self) -> None:
        s = UnifiedCanidSample(
            sample_id="t", dataset_name="d", dataset_version="v",
            source_group_id="g", image_path="p", image_sha256="a"*64,
            width=1, height=1,
            raw_identity_id="1",
            camera_id="c1",
            nose_mask_path="mask.png",
        )
        self.assertTrue(s.label_availability["identity"])
        self.assertTrue(s.label_availability["camera"])
        self.assertTrue(s.label_availability["nose_mask"])
        self.assertFalse(s.label_availability["breed"])
        self.assertFalse(s.label_availability["dog_bbox"])


if __name__ == "__main__":
    unittest.main()
