from __future__ import annotations

import unittest

from tools.evaluate_multichannel import (
    relaxed_status_from_warnings,
    validate_split_disjoint,
)


class SplitLeakageTest(unittest.TestCase):
    def test_reversed_pair_duplicate_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1}]
        test = [{"image_a": "b.jpg", "image_b": "a.jpg", "label": 0}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(any("reversed" in w for w in warnings))

    def test_image_path_cross_split_detected(self):
        cal = [{"image_a": "shared.jpg", "image_b": "other.jpg", "label": 1}]
        test = [{"image_a": "shared.jpg", "image_b": "diff.jpg", "label": 0}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(any("image_a" in w for w in warnings))

    def test_identity_cross_split_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1, "identity": "dog_1"}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0, "identity": "dog_1"}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(any("identity" in w for w in warnings))

    def test_video_group_leakage_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1, "video_id": "vid_1"}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0, "video_id": "vid_1"}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(any("video_id" in w for w in warnings))

    def test_no_leakage_clean(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1, "identity": "dog_1"}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0, "identity": "dog_2"}]
        warnings = validate_split_disjoint(cal, test)
        self.assertEqual(len(warnings), 0)

    def test_missing_metadata_no_crash(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0}]
        warnings = validate_split_disjoint(cal, test)
        self.assertIsInstance(warnings, list)

    def test_relaxed_status_verified(self):
        self.assertEqual(relaxed_status_from_warnings([], False), "VERIFIED")

    def test_relaxed_status_invalid(self):
        self.assertEqual(relaxed_status_from_warnings(["leak"], False), "INVALID")

    def test_relaxed_status_unsafe(self):
        self.assertEqual(relaxed_status_from_warnings(["leak"], True), "RELAXED_UNSAFE")


if __name__ == "__main__":
    unittest.main()
