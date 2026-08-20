from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data import download_datasets

class DatasetDownloaderTests(unittest.TestCase):
    def test_all_named_selectors_fail_before_external_imports_or_writes(self) -> None:
        original_import = __import__
        blocked_roots = {
            "huggingface_hub",
            "PIL",
            "pyarrow",
            "requests",
            "tensorflow",
            "torch",
        }

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".", 1)[0] in blocked_roots:
                raise AssertionError(f"unexpected external import: {name}")
            return original_import(name, *args, **kwargs)

        for name in (
            "ap10k-dog",
            "dogflw",
            "dogfacenet",
            "yt-bb-dog",
            "sibetan",
            "mpdd",
        ):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary) / "data-root"
                with patch("builtins.__import__", side_effect=guarded_import):
                    with self.assertRaisesRegex(
                        download_datasets.ManualAcquisitionRequired,
                        "automatic acquisition is disabled",
                    ):
                        download_datasets.download_dataset(name, root)
                self.assertFalse(root.exists())

    def test_default_all_is_explicit_successful_no_op(self) -> None:
        output = StringIO()
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "data-root"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["download_datasets.py", "--data-root", str(root)],
                ),
                patch.object(download_datasets, "download_dataset") as download,
                redirect_stdout(output),
            ):
                download_datasets.main()

            download.assert_not_called()
            self.assertFalse(root.exists())
            self.assertEqual(
                output.getvalue(),
                "Intentional no-op: no dataset has an admitted automatic download; "
                "no network request or filesystem change was attempted.\n",
            )

    def test_list_reports_every_selector_as_disabled_manual(self) -> None:
        with TemporaryDirectory() as temporary:
            output = StringIO()
            with redirect_stdout(output):
                download_datasets._list_datasets(Path(temporary))

        listing = output.getvalue()
        for name in (
            "ap10k-dog",
            "dogflw",
            "dogfacenet",
            "yt-bb-dog",
            "sibetan",
            "mpdd",
        ):
            self.assertIn(f"{name}: disabled/manual acquisition only", listing)
        self.assertEqual(listing.count("disabled/manual acquisition only"), 6)
        self.assertEqual(listing.count("not present locally"), 6)
        self.assertTrue(listing.isascii())

    def test_cli_named_selectors_exit_with_manual_guidance(self) -> None:
        for name in (
            "ap10k-dog",
            "dogflw",
            "dogfacenet",
            "yt-bb-dog",
            "sibetan",
            "mpdd",
        ):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary) / "data-root"
                error = StringIO()
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "download_datasets.py",
                            "--dataset",
                            name,
                            "--data-root",
                            str(root),
                        ],
                    ),
                    redirect_stderr(error),
                    self.assertRaises(SystemExit) as raised,
                ):
                    download_datasets.main()

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("automatic acquisition is disabled", error.getvalue())
                self.assertFalse(root.exists())

    def test_destinations_use_content_names_not_admission_directories(self) -> None:
        root = Path("/tmp/data")
        self.assertEqual(
            download_datasets._dataset_destination("ap10k-dog", root),
            root / "datasets" / "ap10k",
        )
        self.assertEqual(
            download_datasets._dataset_destination("dogflw", root),
            root / "datasets" / "dogflw",
        )
        self.assertEqual(
            download_datasets._dataset_destination("dogfacenet", root),
            root / "datasets" / "dogfacenet224",
        )

if __name__ == "__main__":
    unittest.main()
