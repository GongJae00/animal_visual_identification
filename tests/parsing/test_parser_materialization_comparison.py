from __future__ import annotations

from parsing.export import compare as comparison
import subprocess
import sys

def test_transition_category_covers_rescue_and_regression() -> None:
    body = {"actual_route": "BODY_PARSING", "terminal_reason": None}
    miss = {"actual_route": "NONE", "terminal_reason": "NO_PARSED_DOG_INSTANCE"}
    ambiguous = {
        "actual_route": "NONE",
        "terminal_reason": "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS",
    }
    assert comparison._category(miss, body) == "TERMINAL_TO_BODY_PARSING"
    assert comparison._category(body, miss) == "BODY_PARSING_TO_TERMINAL"
    assert comparison._category(miss, ambiguous) == "TERMINAL_REASON_CHANGED"
    assert comparison._category(miss, miss) == "TERMINAL_REASON_UNCHANGED"

def test_parse_cli_lists_absorbed_subcommands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "parsing.commands.parse", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in ("materialize", "manifest", "panel", "compare", "three-region"):
        assert name in completed.stdout
