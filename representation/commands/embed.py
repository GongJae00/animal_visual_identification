"""Embedding-cache produce and compare CLI.

Run: ``uv run python -m representation.commands.embed --help``

Subcommands: produce (default), precommit, verify, compare.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m representation.commands.embed",
        description="Produce or compare protected embedding caches.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("produce", help="Produce a protected embedding cache")
    sub.add_parser("precommit", help="Freeze production before inference")
    sub.add_parser("verify", help="Verify a production receipt")
    sub.add_parser("compare", help="Numerical admission between caches")
    if argv and argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if argv and argv[0] == "compare":
        from operations.workers.compare_embedding_caches import main as run

        run(argv[1:])
        return 0
    from operations.workers.embedding_cache_production import main as run

    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
