"""Evaluate prepared protocols from one cached 128D embedding family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.common_reporting import (
    build_master_results_table,
    evaluate_cached_protocol,
)
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256


def _protocol_key(report: dict[str, Any]) -> str:
    return f"{report['dataset']}:{report['protocol_id']}"


def _build_table(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic long-form master table only from sealed reports."
    )
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parsed = parser.parse_args(args)

    table = build_master_results_table(
        [read_strict_json_object(path) for path in parsed.report]
    )
    write_private_json_bundle(((parsed.output, table),))
    print(
        json.dumps(
            {
                "event": "master_results_table_built",
                "output": str(parsed.output),
                "report_count": len(table["source_report_sha256s"]),
                "row_count": len(table["rows"]),
                "table_sha256": table["table_sha256"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    if sys.argv[1:2] == ["table"]:
        _build_table(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-input",
        action="append",
        required=True,
        type=Path,
        help="prepared cached-protocol JSON; repeat for every expected protocol",
    )
    parser.add_argument(
        "--expected-protocol",
        action="append",
        required=True,
        help="required dataset:protocol_id key; repeat to declare the complete suite",
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    reports = [
        evaluate_cached_protocol(
            read_strict_json_object(path),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for path in args.protocol_input
    ]
    ordered = sorted(reports, key=lambda item: _protocol_key(item.report))
    observed_keys = [_protocol_key(item.report) for item in ordered]
    expected_keys = sorted(args.expected_protocol)
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("expected protocol keys must be unique")
    if observed_keys != expected_keys:
        raise ValueError(
            "prepared protocols differ from expected complete suite: "
            f"expected={expected_keys}, observed={observed_keys}"
        )
    family_hashes = {
        item.report["cache_binding"]["cache_family_sha256"] for item in ordered
    }
    if len(family_hashes) != 1:
        raise ValueError("all protocols must use one cached embedding family")

    args.output_directory.resolve(strict=True)
    outputs: list[tuple[Path, dict[str, Any]]] = []
    entries: list[dict[str, Any]] = []
    for index, report in enumerate(ordered):
        filename = f"report-{index:04d}.json"
        outputs.append((args.output_directory / filename, report.to_dict()))
        entries.append(
            {
                "dataset": report.report["dataset"],
                "protocol_id": report.report["protocol_id"],
                "relative_path": filename,
                "report_sha256": report.report_sha256,
            }
        )
    index_payload: dict[str, Any] = {
        "schema_version": "cvi.common_evaluation_family_index.v1",
        "cache_family_sha256": next(iter(family_hashes)),
        "expected_protocols": expected_keys,
        "reports": entries,
    }
    index_payload["index_sha256"] = content_sha256(index_payload)
    outputs.append((args.output_directory / "family-index.json", index_payload))
    write_private_json_bundle(tuple(outputs))
    print(
        json.dumps(
            {
                "event": "cached_embedding_family_evaluated",
                "cache_family_sha256": next(iter(family_hashes)),
                "report_count": len(ordered),
                "output_directory": str(args.output_directory),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
