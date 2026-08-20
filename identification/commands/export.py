"""Identification export CLI.

Run: ``uv run python -m identification.commands.export --help``
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m identification.commands.export",
        description="Export trained encoders and verify ONNX parity.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("onnx", help="Export a training checkpoint to ONNX")
    sub.add_parser("pretrained", help="Export receipt-bound DINOv2-small to ONNX")
    sub.add_parser("parity", help="Produce or verify DINOv2 ONNX parity")
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command, rest = argv[0], argv[1:]
    if command == "onnx":
        from identification.training.appearance.onnx_export import main as run

        run(rest)
        return 0
    if command == "pretrained":
        from prototype.export.pretrained_onnx import main as run

        run(rest)
        return 0
    if command == "parity":
        from prototype.export.dinov2_parity import main as run

        run(rest)
        return 0
    parser.error(f"unknown command {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
