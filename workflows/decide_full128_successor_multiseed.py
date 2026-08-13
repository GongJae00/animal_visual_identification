"""Build the terminal three-seed Full128 successor promotion decision."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from evaluation.full128_successors import (
    build_multiseed_terminal_successor_decision,
)
from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_directory_bundle,
)

_PUBLIC_REPORT_JSON = {
    "maximum_bytes": 67_108_864,
    "maximum_depth": 32,
    "maximum_nodes": 1_000_000,
    "maximum_keys": 500_000,
    "maximum_array_length": 100_000,
    "maximum_string_characters": 1_048_576,
    "maximum_number_characters": 128,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-report",
        required=True,
        action="append",
        type=_seed_report,
        metavar="INDEX=PUBLIC_REPORT",
        help="repeat exactly for seed indexes 0, 1, and 2",
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)

    sources = []
    for seed_index, path in args.seed_report:
        document = read_strict_json_document(path, **_PUBLIC_REPORT_JSON)
        sources.append(
            {
                "seed_index": seed_index,
                "report": document.payload,
                "raw_sha256": document.raw_sha256,
                "canonical_payload_sha256": document.canonical_payload_sha256,
                "byte_size": document.byte_size,
            }
        )
    artifact = build_multiseed_terminal_successor_decision(sources)
    target = Path(os.path.abspath(os.fspath(args.output_directory)))
    strategy = write_private_json_directory_bundle(
        target, (("terminal-decision.json", artifact),)
    )
    output_path = target / "terminal-decision.json"
    print(
        json.dumps(
            {
                "status": "FULL128_SUCCESSOR_TERMINAL_DECISION",
                "decision": artifact["promotion_gate"]["decision"],
                "decision_sha256": artifact["decision_sha256"],
                "output_path": str(output_path),
                "publication_strategy": strategy,
            },
            sort_keys=True,
        )
    )
    return 0


def _seed_report(value: str) -> tuple[int, Path]:
    index, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("must be INDEX=PUBLIC_REPORT")
    try:
        parsed = int(index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed index must be an integer") from exc
    if parsed not in {0, 1, 2}:
        raise argparse.ArgumentTypeError("seed index must be 0, 1, or 2")
    return parsed, Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
