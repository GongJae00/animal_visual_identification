from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.evaluate_multichannel import (
    validate_split_disjoint,
    load_split_manifest,
)


class SplitLeakageTest(unittest.TestCase):
    def test_reversed_pair_duplicate_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1}]
        test = [{"image_a": "b.jpg", "image_b": "a.jpg", "label": 0}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(
            any("reversed" in w for w in warnings),
            f"expected reversed pair warning, got {warnings}",
        )

    def test_image_path_cross_split_detected(self):
        cal = [{"image_a": "shared.jpg", "image_b": "other.jpg", "label": 1}]
        test = [{"image_a": "shared.jpg", "image_b": "diff.jpg", "label": 0}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(
            any("image_a" in w for w in warnings),
            f"expected image_a leakage warning, got {warnings}",
        )

    def test_identity_cross_split_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1, "identity": "dog_1"}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0, "identity": "dog_1"}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(
            any("identity" in w for w in warnings),
            f"expected identity leakage warning, got {warnings}",
        )

    def test_video_group_leakage_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1, "video_id": "vid_1"}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0, "video_id": "vid_1"}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(
            any("video_id" in w for w in warnings),
            f"expected video_id leakage warning, got {warnings}",
        )

    def test_session_group_leakage_detected(self):
        cal = [{"image_a": "a.jpg", "image_b": "b.jpg", "label": 1, "session_id": "ses_1"}]
        test = [{"image_a": "c.jpg", "image_b": "d.jpg", "label": 0, "session_id": "ses_1"}]
        warnings = validate_split_disjoint(cal, test)
        self.assertTrue(
            any("session_id" in w for w in warnings),
            f"expected session_id leakage warning, got {warnings}",
        )

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

    def test_load_split_manifest_validates_schema(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"calibration": [], "test": []}, f)
            p = Path(f.name)
        try:
            data = load_split_manifest(p)
            self.assertIn("calibration", data)
            self.assertIn("test", data)
        finally:
            p.unlink()

    def test_load_split_manifest_rejects_missing_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"calibration": []}, f)
            p = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_split_manifest(p)
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main()
