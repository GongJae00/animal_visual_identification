from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from visualization.registry import FIGURE_REGISTRY
from visualization.rendering.pipeline import STAGE_LAYOUT, vis_directory

from tests.repo_root import REPO_ROOT as ROOT

def test_paper_registry_is_not_pipeline_vis_numbering() -> None:
    paper_ids = tuple(spec.figure_id for spec in FIGURE_REGISTRY)
    vis_dirs = tuple(layout[0] for layout in STAGE_LAYOUT.values())
    assert paper_ids[0] == "00_evidence_ladder"
    assert vis_dirs == (
        "00_parsing",
        "01_identification",
        "02_representation",
        "03_enrollment",
        "04_gallery",
        "05_search",
    )
    assert paper_ids[0] not in vis_dirs
    assert "00_parsing" not in paper_ids

def test_render_stage_writes_substages_and_says_activations_absent(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    from visualization.parsing import render

    written = render({"stage": "parsing", "substages": {}}, tmp_path)
    stage_root = vis_directory(tmp_path, "parsing")
    assert stage_root.name == "00_parsing"
    assert (stage_root / "00_detection" / "trace.png").is_file()
    assert (stage_root / "04_crops" / "trace.json").is_file()
    payload = json.loads(
        (stage_root / "00_detection" / "trace.json").read_text(encoding="utf-8")
    )
    assert payload["activations"] == "activations absent"
    assert all(path.is_file() for path in written)
    png = (stage_root / "00_detection" / "trace.png").read_bytes()
    assert b"heatmap" not in png.lower()

def test_render_cli_stage_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "visualization.commands.render", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--stage" in completed.stdout
    assert "00_parsing" in completed.stdout or "parsing" in completed.stdout
    assert "--paper" in completed.stdout
