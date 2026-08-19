from __future__ import annotations

from workflows import compare_parser_materializations as review


def test_balanced_tokens_is_deterministic_across_datasets() -> None:
    rows = {
        "a1": {"dataset_name": "a"},
        "a2": {"dataset_name": "a"},
        "b1": {"dataset_name": "b"},
    }
    first = review._balanced_tokens(("a1", "a2", "b1"), rows, 3, "RESCUE")
    second = review._balanced_tokens(("b1", "a2", "a1"), rows, 3, "RESCUE")
    assert first == second
    assert {rows[token]["dataset_name"] for token in first[:2]} == {"a", "b"}
