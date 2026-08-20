from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from evaluation.parsing_protocol import (
    METRICS,
    PROTOCOL_SCHEMA,
    STAGES,
    protocol_document,
    visualization_trace,
)
from visualization.rendering.pipeline import STAGE_LAYOUT

from tests.repo_root import REPO_ROOT as ROOT

_SCHEMA_PATH = (
    ROOT
    / "shared"
    / "contracts"
    / "schemas"
    / "evaluation.parsing_protocol.v1.schema.json"
)


def test_stages_match_visualization_substages() -> None:
    assert tuple(stage.vis_substage for stage in STAGES) == STAGE_LAYOUT["parsing"][1]
    assert tuple(stage.stage_id for stage in STAGES) == (
        "detection",
        "segmentation",
        "regions",
        "quality",
        "crops",
    )


def test_protocol_document_is_a_catalog_not_results() -> None:
    document = protocol_document()
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)
    assert document["schema_version"] == PROTOCOL_SCHEMA
    assert document["backbone_independent"] is True
    ids = [metric.metric_id for metric in METRICS]
    assert len(ids) == len(set(ids))
    assert document["columns"] == ["stage", "data", "metric", "extraction"]
    assert len(document["figure_rows"]) == len(METRICS)
    blob = json.dumps(document)
    assert "cvi." not in blob
    assert "canine_identity." not in blob
    assert "AP@" in blob
    for row in document["figure_rows"]:
        assert set(row) == {"stage", "data", "metric", "extraction"}
        assert all(row[key] for key in row)


def test_every_metric_has_a_known_stage_and_owner() -> None:
    stage_ids = {stage.stage_id for stage in STAGES}
    for metric in METRICS:
        assert metric.stage_id in stage_ids
        assert metric.kind in {
            "supervised",
            "unsupervised",
            "census",
            "control",
            "review",
        }
        assert "." in metric.owner
        assert metric.command


def test_visualization_trace_groups_rows_by_substage() -> None:
    trace = visualization_trace()
    assert trace["stage"] == "parsing"
    substages = STAGE_LAYOUT["parsing"][1]
    assert tuple(trace["substages"]) == substages
    for name in substages:
        rows = trace["substages"][name]["metrics"]
        assert rows
        labels = {row["stage"] for row in rows}
        assert len(labels) == 1


def test_parsing_protocol_cli_writes_trace(tmp_path: Path) -> None:
    output = tmp_path / "parsing_protocol.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.commands.evaluate",
            "parsing-protocol",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["event"] == "parsing_protocol_written"
    assert payload["schema_version"] == PROTOCOL_SCHEMA
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["protocol"]["schema_version"] == PROTOCOL_SCHEMA
    assert written["stage"] == "parsing"
