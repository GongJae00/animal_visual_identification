from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.pdq_official_regression import (
    CANONICAL_NATIVE_BINARY_SHA256,
    PDQOfficialRegressionReceipt,
    _verify_expected_output,
    publish_official_pdq_regression,
    run_official_pdq_regression,
)
from cvi.source_provenance import build_offline_tool_provenance


ROOT = Path(__file__).parents[1]
SOURCE_ROOT = Path(os.environ.get("CVI_PDQ_OFFICIAL_SOURCE_ROOT") or os.devnull)
REGRESSION_BUNDLE = SOURCE_ROOT / "pdq-regression-intake-v1"
NATIVE_WORKER = SOURCE_ROOT / "pdq-native-worker-v4"
AVAILABLE = REGRESSION_BUNDLE.is_dir() and NATIVE_WORKER.is_dir()


def _provenance() -> dict[str, object]:
    return build_offline_tool_provenance(
        ROOT / "tools/admit_native_pdq_regression.py"
    )


@unittest.skipUnless(AVAILABLE, "canonical official PDQ regression assets unavailable")
class PDQOfficialRegressionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.provenance = _provenance()
        cls.receipt = run_official_pdq_regression(
            regression_bundle_directory=REGRESSION_BUNDLE,
            native_worker_directory=NATIVE_WORKER,
            tool_provenance=cls.provenance,
        )

    def test_exact_official_bridge_and_d4_regression_passes(self) -> None:
        receipt = self.receipt
        self.assertEqual(receipt.native_binary_sha256, CANONICAL_NATIVE_BINARY_SHA256)
        self.assertEqual(receipt.decision, "PASS_EXACT_FIXED_COMMIT_OFFICIAL_REGRESSION")
        self.assertEqual(len(receipt.bridge_results), 8)
        self.assertTrue(all(item.quality == 100 for item in receipt.bridge_results))
        self.assertEqual(len(receipt.d4_hashes), 8)
        self.assertEqual(PDQOfficialRegressionReceipt.from_dict(receipt.to_dict()), receipt)

    def test_receipt_rejects_self_consistent_result_and_binding_tampering(self) -> None:
        payload = self.receipt.to_dict()
        mutations = {
            "bundle": lambda value: value.__setitem__("regression_bundle_sha256", "0" * 64),
            "native": lambda value: value.__setitem__("native_binary_sha256", "0" * 64),
            "bridge": lambda value: value["bridge_results"][0].__setitem__("original_hash", "0" * 64),
            "bridge order": lambda value: value["bridge_results"].reverse(),
            "d4": lambda value: value["d4_hashes"].__setitem__(0, "0" * 64),
            "decision": lambda value: value.__setitem__("decision", "PASS"),
            "interpretation": lambda value: value.__setitem__("interpretation", "ADMITTED"),
        }
        for name, mutate in mutations.items():
            changed = copy.deepcopy(payload)
            mutate(changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                PDQOfficialRegressionReceipt.from_dict(changed)

    def test_publication_is_bound_private_and_no_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            publish_official_pdq_regression(
                receipt=self.receipt,
                tool_provenance=self.provenance,
                output_path=output,
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(bundle["receipt_sha256"], self.receipt.receipt_sha256)
            with self.assertRaises(FileExistsError):
                publish_official_pdq_regression(
                    receipt=self.receipt,
                    tool_provenance=self.provenance,
                    output_path=output,
                )
            bad = copy.deepcopy(self.provenance)
            bad["runtime_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "provenance binding"):
                publish_official_pdq_regression(
                    receipt=self.receipt,
                    tool_provenance=bad,
                    output_path=Path(temporary) / "bad.json",
                )


class PDQOfficialExpectedParserTests(unittest.TestCase):
    def test_expected_parser_requires_exact_unique_bridge_and_d4_blocks(self) -> None:
        expected = (
            REGRESSION_BUNDLE / "source/pdq/cpp/reg_test/expected/out"
        )
        if not expected.is_file():
            self.skipTest("canonical official expected output unavailable")
        payload = expected.read_bytes()
        _verify_expected_output(payload)
        with self.assertRaisesRegex(ValueError, "bridge expected block"):
            _verify_expected_output(payload.replace(
                b"30a10efdf1c83f429013d48d0ffffc52",
                b"20a10efdf1c83f429013d48d0ffffc52",
                1,
            ))
        with self.assertRaisesRegex(ValueError, "D4 expected line"):
            marker = b",100,./reg_test/../../data/reg-test-input/dih/bridge-1-original.jpg"
            first = payload.find(marker, payload.find(b"--pdqdih-across"))
            changed = payload[:first] + payload[first:].replace(marker, b",99" + marker[4:], 1)
            _verify_expected_output(changed)


if __name__ == "__main__":
    unittest.main()
