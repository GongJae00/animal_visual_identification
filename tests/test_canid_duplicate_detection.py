from __future__ import annotations

import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from data.duplicates import (
    canonical_pixel_hash,
    find_exact_duplicates,
    summarize_duplicates,
)
from data.types import UnifiedCanidSample


def _sample(sample_id: str, image_path: str) -> UnifiedCanidSample:
    return UnifiedCanidSample(
        sample_id=sample_id,
        dataset_name="test",
        dataset_version="v1",
        source_group_id="0",
        image_path=image_path,
        image_sha256="a" * 64,
        width=1,
        height=1,
    )


class CanidDuplicateTests(unittest.TestCase):
    def test_identical_pixels_are_detected_as_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), (128, 64, 32)).save(root / "a.png")
            Image.new("RGB", (2, 2), (255, 255, 255)).save(root / "b.png")

            samples = (
                _sample("s1", "a.png"),
                _sample("s2", "b.png"),
                _sample("s3", "a.png"),
            )
            duplicates = find_exact_duplicates(samples, root)
            self.assertEqual(len(duplicates), 1)
            group = list(duplicates.values())[0]
            self.assertEqual(len(group), 2)
            self.assertIn("s1", group)
            self.assertIn("s3", group)

    def test_different_pixels_are_not_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), (0, 0, 0)).save(root / "a.png")
            Image.new("RGB", (2, 2), (1, 1, 1)).save(root / "b.png")
            duplicates = find_exact_duplicates(
                (_sample("s1", "a.png"), _sample("s2", "b.png")), root
            )
            self.assertEqual(len(duplicates), 0)

    def test_summary_separates_repeated_annotations_from_distinct_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), (128, 64, 32)).save(root / "a.png")
            Image.new("RGB", (2, 2), (128, 64, 32)).save(root / "copy.png")
            samples = (
                _sample("annotation-1", "a.png"),
                _sample("annotation-2", "a.png"),
                _sample("copy", "copy.png"),
            )

            summary = summarize_duplicates(samples, root)

            self.assertEqual(summary["total_samples"], 3)
            self.assertEqual(summary["unique_image_paths"], 2)
            self.assertEqual(summary["repeated_source_path_groups"], 1)
            self.assertEqual(summary["repeated_source_path_samples"], 2)
            self.assertEqual(summary["duplicate_groups"], 1)
            self.assertEqual(summary["duplicate_samples"], 2)


if __name__ == "__main__":
    unittest.main()
