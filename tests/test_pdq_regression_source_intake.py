from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from identity_methods.classical.pdq_regression_source_intake import (
    PDQ_REGRESSION_ASSET_PATHS,
    PDQ_REGRESSION_COMMIT_SHA,
    PDQ_REGRESSION_SELECTED_PATHS,
    PDQ_REGRESSION_TREE_SHA,
    validate_pdq_regression_source_contract,
)
from identity_methods.classical.pdq_source_intake import PdqSelectedSourceMember, PdqSourceContract


class PdqRegressionSourceIntakeTests(unittest.TestCase):
    @staticmethod
    def _contract() -> PdqSourceContract:
        path = (
            Path(__file__).parents[1]
            / "contracts/configs/pdq/threatexchange-pdq-regression-baefb4ed.json"
        )
        return PdqSourceContract.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def test_official_regression_contract_round_trips_exact_profile(self) -> None:
        source = self._contract()
        validate_pdq_regression_source_contract(source)
        self.assertEqual(PdqSourceContract.from_dict(source.to_dict()), source)
        self.assertEqual(source.commit_sha, PDQ_REGRESSION_COMMIT_SHA)
        self.assertEqual(source.tree_sha, PDQ_REGRESSION_TREE_SHA)
        self.assertEqual(len(source.selected_members), 21)
        self.assertEqual(
            tuple(item.relative_path for item in source.selected_members),
            PDQ_REGRESSION_SELECTED_PATHS,
        )

    def test_exact_nine_regression_assets_are_present_and_forbidden_are_absent(
        self,
    ) -> None:
        source = self._contract()
        selected = tuple(item.relative_path for item in source.selected_members)
        regression_assets = tuple(
            path
            for path in selected
            if path == "pdq/cpp/reg_test/expected/out"
            or path.startswith("pdq/data/reg-test-input/dih/bridge-")
        )
        self.assertEqual(regression_assets, PDQ_REGRESSION_ASSET_PATHS)
        self.assertEqual(len(regression_assets), 9)
        self.assertNotIn("pdq/cpp/CImg.h", selected)
        self.assertFalse(any(path.startswith("pdq/cpp/io/") for path in selected))

    def test_missing_or_unexpected_regression_member_fails_closed(self) -> None:
        source = self._contract()
        missing_path = "pdq/data/reg-test-input/dih/bridge-8-flip-minus-1.jpg"
        missing = replace(
            source,
            selected_members=tuple(
                item
                for item in source.selected_members
                if item.relative_path != missing_path
            ),
        )
        with self.assertRaisesRegex(ValueError, "missing=.*bridge-8"):
            validate_pdq_regression_source_contract(missing)

        extra = PdqSelectedSourceMember(
            relative_path="pdq/data/reg-test-input/dih/unexpected.jpg",
            expected_bytes=1,
            git_blob_sha1="1" * 40,
            content_sha256="2" * 64,
        )
        unexpected = replace(
            source,
            selected_members=tuple(
                sorted(
                    source.selected_members + (extra,),
                    key=lambda item: item.relative_path.casefold(),
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "unexpected=.*unexpected.jpg"):
            validate_pdq_regression_source_contract(unexpected)

    def test_source_only_contract_remains_the_exact_twelve_member_profile(self) -> None:
        source_only_path = (
            Path(__file__).parents[1]
            / "contracts/configs/pdq/threatexchange-pdq-baefb4ed.json"
        )
        source_only = PdqSourceContract.from_dict(
            json.loads(source_only_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(len(source_only.selected_members), 12)
        self.assertFalse(
            any(
                item.relative_path in PDQ_REGRESSION_ASSET_PATHS
                for item in source_only.selected_members
            )
        )


if __name__ == "__main__":
    unittest.main()
