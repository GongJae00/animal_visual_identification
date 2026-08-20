"""Dataset and model acquisition status CLI.

Run: ``uv run python -m data.commands.download --help``

Subcommands: ``datasets`` (default) and ``models``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m data.commands.download",
        description="Inspect dataset and model acquisition status.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=("datasets", "models"),
        help="datasets is the default when omitted.",
    )
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    target = "datasets"
    rest = argv
    if argv[0] in {"datasets", "models"}:
        target, rest = argv[0], argv[1:]
    if target == "models":
        from data.download_models import main as run_models

        previous = sys.argv
        sys.argv = [previous[0], *rest]
        try:
            run_models()
        finally:
            sys.argv = previous
        return 0
    from data.download_datasets import main as run_datasets

    run_datasets(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
