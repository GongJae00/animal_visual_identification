from __future__ import annotations

from workflows import compare_parser_materializations as summary


def test_summary_schema_is_versioned() -> None:
    assert summary.SUMMARY_REPORT_SCHEMA == "cvi.parser_materialization_summary.v1"
