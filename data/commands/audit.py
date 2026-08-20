"""Public-corpus and pretrained-asset audit CLI.

Run: ``uv run python -m data.commands.audit --help``
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

_ABSORBED = {
    "image-content": "data.public_sources.audit_public_canine_image_content",
    "semantics": "data.public_sources.audit_public_canine_semantics",
    "phash": "data.audit.phash_candidate_audit",
    "duplicates": "evaluation.splits.adjudicate_public_duplicates",
    "extract": "data.public_sources.extract_public_dataset_archive",
    "evidence-graph": "evaluation.splits.assemble_public_split_evidence_graph",
    "pretrained-weight": "shared.contracts.intake.audit_pretrained_weight",
    "pretrained-asset": "shared.contracts.intake.audit_pretrained_supporting_asset",
    "pdq-intake": "data.audit.pdq.intake_threatexchange",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m data.commands.audit",
        description="Audit public image/semantics/phash/duplicates/extract.",
    )
    sub = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("image-content", "Decode and pixel-exact duplicate audit"),
        ("semantics", "Protected semantic receipt"),
        ("phash", "Label-blind pHash candidate audit"),
        ("duplicates", "Public duplicate adjudication ledger"),
        ("extract", "Extract an admitted public archive"),
        ("evidence-graph", "Promote adjudication to a frozen graph"),
        ("pretrained-weight", "Audit pretrained weight bytes"),
        ("pretrained-asset", "Audit pretrained supporting JSON"),
        ("pdq-intake", "Publish an audited PDQ source subset"),
    ):
        sub.add_parser(name, help=help_text)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command, rest = argv[0], argv[1:]
    module_name = _ABSORBED.get(command)
    if module_name is None:
        parser.error(f"unknown command {command}")
    module = importlib.import_module(module_name)
    result = module.main(rest) if _accepts_argv(module.main) else _run_main(module, rest)
    if isinstance(result, int):
        return result
    return 0


def _accepts_argv(func: object) -> bool:
    import inspect

    try:
        inspect.signature(func).bind([])
    except TypeError:
        return False
    return True


def _run_main(module: object, rest: list[str]) -> object:
    previous = sys.argv
    sys.argv = [previous[0], *rest]
    try:
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
