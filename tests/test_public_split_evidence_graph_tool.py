from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class PublicSplitEvidenceGraphToolTests(unittest.TestCase):
    def test_adjudication_and_graph_cli_help(self) -> None:
        for tool, expected in (
            (
                "adjudicate_public_duplicates.py",
                "{exact,chunk,source-generation,merge,review-queue}",
            ),
            ("analyze_duplicate_graph_capacity.py", "--split-policy"),
            ("assemble_public_split_evidence_graph.py", "--adjudication-ledger"),
        ):
            with self.subTest(tool=tool):
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / tool), "--help"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn(expected, completed.stdout)


if __name__ == "__main__":
    unittest.main()
