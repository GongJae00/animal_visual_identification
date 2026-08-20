from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tests.repo_root import REPO_ROOT as ROOT

class PublicSplitEvidenceGraphToolTests(unittest.TestCase):
    def test_adjudication_and_graph_cli_help(self) -> None:
        for command, expected in (
            (
                [sys.executable, "-m", "data.commands.audit", "duplicates", "--help"],
                "{exact,chunk,source-generation,merge,review-queue}",
            ),
            (
                [sys.executable, "-m", "operations.commands.measure", "capacity", "--help"],
                "--split-policy",
            ),
            (
                [sys.executable, "-m", "data.commands.audit", "evidence-graph", "--help"],
                "--adjudication-ledger",
            ),
        ):
            with self.subTest(command=command):
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=ROOT,
                )
                self.assertIn(expected, completed.stdout)

if __name__ == "__main__":
    unittest.main()
