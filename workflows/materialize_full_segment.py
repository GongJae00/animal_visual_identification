"""Materialize one common Full-segment record from frozen parsing or native input."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# Compatibility imports keep the established workflow CLI and test seam while
# the materialization implementation remains owned by the algorithm package.
from identity_methods.full_segment.sample_materialization import (
    REQUEST_SCHEMA,
    _read_json_object,
    run,
    run_prevalidated,
)

__all__ = ["REQUEST_SCHEMA", "run", "run_prevalidated"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    request = _read_json_object(args.request.resolve(), label="Full segment request")
    run(request, output_dir=args.output_dir.resolve())


if __name__ == "__main__":
    main()
