from __future__ import annotations

import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.controls.construct_pairs import _write_bundle

class PairingToolTests(unittest.TestCase):
    def test_bundle_is_private_and_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = (
                (root / "scoring.json", {"kind": "scoring"}),
                (root / "bindings.json", {"kind": "bindings"}),
                (root / "ground_truth.json", {"kind": "truth"}),
                (root / "summary.json", {"kind": "summary"}),
            )
            _write_bundle(outputs)
            self.assertTrue(all(path.is_file() for path, _ in outputs))
            for path, _ in outputs:
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertEqual(mode, 0o600)
            original = (root / "scoring.json").read_bytes()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                _write_bundle(outputs)
            self.assertEqual((root / "scoring.json").read_bytes(), original)

    def test_bundle_requires_one_output_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            with self.assertRaisesRegex(ValueError, "one protected directory"):
                _write_bundle(
                    (
                        (first / "a.json", {"a": 1}),
                        (second / "b.json", {"b": 2}),
                    )
                )

if __name__ == "__main__":
    unittest.main()
