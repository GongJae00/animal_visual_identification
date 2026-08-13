from __future__ import annotations

from workflows import summarize_parser_materialization as summary


def test_summary_schema_is_versioned() -> None:
    assert summary.REPORT_SCHEMA == "cvi.parser_materialization_summary.v1"
