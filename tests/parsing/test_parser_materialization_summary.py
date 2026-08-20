from __future__ import annotations

from parsing.export import compare as summary

def test_summary_schema_is_versioned() -> None:
    assert summary.SUMMARY_REPORT_SCHEMA == "parsing.parser_materialization_summary.v1"
