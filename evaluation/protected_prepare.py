"""Prepare and pre-publish a protected evaluation receipt chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.contracts.source_provenance import build_offline_tool_provenance
from evaluation.protected_evaluation import prepare_protected_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--external-pins", required=True, type=Path)
    parser.add_argument("--expected-external-pins-raw-sha256", required=True)
    parser.add_argument("--split-assignment", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--exposure-ledger", required=True, type=Path)
    parser.add_argument("--exposure-receipt", required=True, type=Path)
    parser.add_argument("--gallery", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    provenance = build_offline_tool_provenance(
        Path(__file__),
        additional_paths=(
            root / "evaluation" / "protected_evaluation.py",
            root / "shared" / "foundation" / "protected_io.py",
            root / "evaluation" / "splits" / "role_exposure.py",
        ),
    )
    plan = prepare_protected_evaluation(
        policy_path=args.policy,
        external_pins_path=args.external_pins,
        expected_external_pins_raw_sha256=args.expected_external_pins_raw_sha256,
        split_assignment_path=args.split_assignment,
        split_receipt_path=args.split_receipt,
        exposure_ledger_path=args.exposure_ledger,
        exposure_receipt_path=args.exposure_receipt,
        gallery_path=args.gallery,
        queries_path=args.queries,
        output_directory=args.output_directory,
        tool_provenance=provenance,
    )
    print(json.dumps({
        "status": plan.status,
        "evaluation_token": plan.evaluation_token,
        "plan_receipt_sha256": plan.receipt_sha256,
        "advanced_exposure_declaration_sha256": (
            plan.advanced_exposure_declaration_sha256
        ),
        "output_directory": str(args.output_directory),
    }, sort_keys=True))
