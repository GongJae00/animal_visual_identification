from __future__ import annotations

from workflows import compare_parser_materializations as comparison


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
