from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from data.duplicates import (
    find_exact_duplicates,
    summarize_duplicates,
)
from data.source_lock import get_record
from data.types import UnifiedCanidSample
from workflows import build_canid_unified_manifest


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
            group = next(iter(duplicates.values()))
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

    def test_inspect_route_reports_absent_roots_and_preserves_adapter_behavior(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_root = root / "missing"
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (2, 2), (128, 64, 32)).save(image_root / "a.png")
            records = (
                replace(get_record("dogfacenet224"), data_root=str(image_root)),
                replace(get_record("mpdd"), data_root=str(missing_root)),
            )
            samples = (_sample("sample", "a.png"),)
            seen_roots: list[Path] = []

            def adapter(adapter_root: Path) -> tuple[UnifiedCanidSample, ...]:
                seen_roots.append(adapter_root)
                return samples

            stdout = io.StringIO()
            with (
                patch.object(
                    build_canid_unified_manifest,
                    "admitted_records",
                    return_value=records,
                ),
                patch.object(
                    build_canid_unified_manifest,
                    "ADAPTERS",
                    {"dogfacenet224": adapter, "mpdd": adapter},
                ),
                redirect_stdout(stdout),
            ):
                build_canid_unified_manifest.main(["inspect", "ignored"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(list(report), sorted(report))
            self.assertEqual(seen_roots, [image_root])
            self.assertEqual(report["dogfacenet224"]["statistics"]["total_samples"], 1)
            self.assertEqual(
                report["dogfacenet224"]["duplicates"]["duplicate_groups"], 0
            )
            self.assertEqual(
                report["mpdd"],
                {
                    "admission": records[1].admission.value,
                    "error": "data root not found",
                    "root": str(missing_root),
                },
            )

    def test_duplicates_route_keeps_within_and_cross_calculations_distinct(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            Image.new("RGB", (2, 2), (128, 64, 32)).save(first_root / "a.png")
            Image.new("RGB", (2, 2), (128, 64, 32)).save(second_root / "b.png")
            records = (
                replace(get_record("dogfacenet224"), data_root=str(first_root)),
                replace(get_record("mpdd"), data_root=str(second_root)),
                replace(get_record("sibetan"), data_root=str(root / "missing")),
            )
            adapters = {
                "dogfacenet224": lambda _: (_sample("first", "a.png"),),
                "mpdd": lambda _: (_sample("second", "b.png"),),
                "sibetan": lambda _: (),
            }

            stdout = io.StringIO()
            with (
                patch.object(
                    build_canid_unified_manifest,
                    "admitted_records",
                    return_value=records,
                ),
                patch.object(build_canid_unified_manifest, "ADAPTERS", adapters),
                redirect_stdout(stdout),
            ):
                build_canid_unified_manifest.main(["duplicates", "ignored"])

            report = json.loads(stdout.getvalue())
            self.assertEqual(
                report["within_dataset"],
                {
                    "dogfacenet224": {"groups": 0, "total_duplicate_samples": 0},
                    "mpdd": {"groups": 0, "total_duplicate_samples": 0},
                },
            )
            self.assertEqual(
                report["cross_dataset"],
                {
                    "datasets_involved": ["dogfacenet224", "mpdd"],
                    "groups": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
