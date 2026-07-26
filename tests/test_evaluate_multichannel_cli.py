"""E2E CLI smoke tests for evaluate_multichannel.py (G1 gate)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _run(*args, check=True):
    result = subprocess.run(
        [sys.executable, "-m", "tools.evaluate_multichannel"] + list(args),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check:
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            result.check_returncode()
    return result


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class EvaluateMultichannelCliTest(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="cvi_e2e_")

    def _path(self, *parts):
        return Path(self.td, *parts)

    # ---- Help ----
    def test_help_exits_zero(self):
        result = _run("--help", check=False)
        self.assertEqual(result.returncode, 0)

    def test_verification_help_exits_zero(self):
        result = _run("verification", "--help", check=False)
        self.assertEqual(result.returncode, 0)

    def test_retrieval_help_exits_zero(self):
        result = _run("retrieval", "--help", check=False)
        self.assertEqual(result.returncode, 0)

    def test_open_set_help_exits_zero(self):
        result = _run("open-set", "--help", check=False)
        self.assertEqual(result.returncode, 0)

    # ---- Positive retrieval (open-set mode) ----
    def test_retrieval_happy_path(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        qry = self._path("qry.json")
        np.random.seed(0)
        _write_json(
            gal,
            {
                "embeddings": np.random.randn(5, 8).tolist(),
                "identities": [0, 0, 1, 1, 2],
            },
        )
        _write_json(
            qry,
            {
                "embeddings": np.random.randn(3, 8).tolist(),
                "identities": [0, 1, 3],
            },
        )
        result = _run(
            "retrieval",
            "--gallery", str(gal),
            "--queries", str(qry),
            "--output", str(out),
            "--open-set",
        )
        self.assertEqual(result.returncode, 0)
        report = json.loads(out.read_text())
        self.assertIn("Rank-1", report)

    # ---- Retrieval: self-match with sample IDs ----
    def test_retrieval_self_match_excluded(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        qry = self._path("qry.json")
        _write_json(
            gal,
            {
                "embeddings": [[1, 0], [0, 1], [1, 0]],
                "identities": [0, 1, 0],
                "sample_ids": ["g1", "g2", "g3"],
            },
        )
        _write_json(
            qry,
            {
                "embeddings": [[1, 0]],
                "identities": [0],
                "sample_ids": ["g1"],
            },
        )
        result = _run(
            "retrieval",
            "--gallery", str(gal),
            "--queries", str(qry),
            "--output", str(out),
            "--no-self-match",
            "--open-set",
        )
        self.assertEqual(result.returncode, 0)
        report = json.loads(out.read_text())
        self.assertIn("self_match_excluded", report)
        self.assertTrue(report["self_match_excluded"])

    # ---- Retrieval: missing sample IDs with --no-self-match ----
    def test_retrieval_no_self_match_missing_sample_ids(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        qry = self._path("qry.json")
        _write_json(gal, {"embeddings": [[1, 0]], "identities": [0]})
        _write_json(qry, {"embeddings": [[1, 0]], "identities": [0]})
        result = _run(
            "retrieval",
            "--gallery", str(gal),
            "--queries", str(qry),
            "--output", str(out),
            "--no-self-match",
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        combined = result.stdout + result.stderr
        self.assertIn("sample_ids", combined)

    # ---- Positive open-set ----
    def test_open_set_happy_path(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        cal_q = self._path("cal_q.json")
        test_q = self._path("test_q.json")
        np.random.seed(42)
        _write_json(
            gal,
            {
                "embeddings": np.random.randn(4, 8).tolist(),
                "identities": [0, 1, 2, 3],
            },
        )
        _write_json(
            cal_q,
            {
                "embeddings": np.random.randn(4, 8).tolist(),
                "identities": [0, 1, 2, 99],
            },
        )
        _write_json(
            test_q,
            {
                "embeddings": np.random.randn(4, 8).tolist(),
                "identities": [0, 1, 2, 99],
            },
        )
        result = _run(
            "open-set",
            "--gallery", str(gal),
            "--calibration-queries", str(cal_q),
            "--test-queries", str(test_q),
            "--output", str(out),
        )
        self.assertEqual(result.returncode, 0)
        report = json.loads(out.read_text())
        self.assertIn("known_detection_AUROC", report)
        self.assertIn("per_target", report)

    # ---- Open-set: without unknowns raises ----
    def test_open_set_no_unknowns_raises(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        cal_q = self._path("cal_q.json")
        test_q = self._path("test_q.json")
        np.random.seed(7)
        _write_json(
            gal,
            {
                "embeddings": np.random.randn(2, 8).tolist(),
                "identities": [0, 1],
            },
        )
        _write_json(
            cal_q,
            {
                "embeddings": np.random.randn(2, 8).tolist(),
                "identities": [0, 1],
            },
        )
        _write_json(
            test_q,
            {
                "embeddings": np.random.randn(2, 8).tolist(),
                "identities": [0, 1],
            },
        )
        _write_json(self._path("fpt.json"), [0.5])
        result = _run(
            "open-set",
            "--gallery", str(gal),
            "--calibration-queries", str(cal_q),
            "--test-queries", str(test_q),
            "--fpir-targets", str(self._path("fpt.json")),
            "--output", str(out),
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)

    # ---- Provenance checks via retrieval ----
    def test_provenance_has_git_info(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        qry = self._path("qry.json")
        _write_json(gal, {"embeddings": [[1, 0]], "identities": [0]})
        _write_json(qry, {"embeddings": [[1, 0]], "identities": [0]})
        _run(
            "retrieval",
            "--gallery", str(gal),
            "--queries", str(qry),
            "--output", str(out),
        )
        report = json.loads(out.read_text())
        prov = report["provenance"]
        self.assertIn("git_commit", prov)
        self.assertIn("git_branch", prov)
        self.assertIn("dirty_state", prov)
        self.assertIn("python_version", prov)

    # ---- Schema version via retrieval ----
    def test_schema_version_in_report(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        qry = self._path("qry.json")
        _write_json(gal, {"embeddings": [[1, 0]], "identities": [0]})
        _write_json(qry, {"embeddings": [[1, 0]], "identities": [0]})
        _run(
            "retrieval",
            "--gallery", str(gal),
            "--queries", str(qry),
            "--output", str(out),
        )
        report = json.loads(out.read_text())
        self.assertIn("schema_version", report["provenance"])
        self.assertEqual(report["provenance"]["schema_version"], "cvi.evaluation.report.v2")

    # ---- Bootstrap CIs in retrieval ----
    def test_retrieval_bootstrap_ci_present(self):
        out = self._path("report.json")
        gal = self._path("gal.json")
        qry = self._path("qry.json")
        np.random.seed(1)
        _write_json(
            gal,
            {
                "embeddings": np.random.randn(10, 4).tolist(),
                "identities": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            },
        )
        _write_json(
            qry,
            {
                "embeddings": np.random.randn(5, 4).tolist(),
                "identities": [0, 1, 2, 3, 5],
            },
        )
        _run(
            "retrieval",
            "--gallery", str(gal),
            "--queries", str(qry),
            "--output", str(out),
            "--open-set",
        )
        report = json.loads(out.read_text())
        self.assertIn("rank_bootstrap_ci", report)
        self.assertIn("mAP_bootstrap_ci", report)
