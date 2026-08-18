"""Validate private successor representation traces and publish safe summaries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evaluation.full_segment.full128_analysis import (
    build_public_representation_analysis,
    sanitize_representation_trace_manifest,
)
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import admit_new_external_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-trace", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    traces = [
        sanitize_representation_trace_manifest(
            read_strict_json_document(path, maximum_bytes=268_435_456).payload
        )
        for path in args.private_trace
    ]
    result = build_public_representation_analysis(traces)
    output = _new_external_output(args.output)
    write_private_json_bundle(((output, result),))
    print(
        json.dumps(
            {
                "status": "ANALYZED_FULL128_SUCCESSOR_TRACES",
                "trace_count": len(traces),
                "analysis_sha256": result["analysis_sha256"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _new_external_output(path: Path) -> Path:
    return admit_new_external_output(
        path,
        repository_root=Path(__file__).resolve().parents[1],
        repository_error="representation analysis output must remain outside repository",
        overwrite_error="refusing to overwrite representation analysis output",
    )


if __name__ == "__main__":
    raise SystemExit(main())
