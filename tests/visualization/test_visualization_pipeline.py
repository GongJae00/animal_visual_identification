from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def test_render_stage_writes_json_and_skips_empty_plates(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.parsing import render

    written = render({"stage": "parsing", "substages": {}}, tmp_path)
    stage_root = vis_directory(tmp_path, "parsing")
    assert stage_root.name == "00_parsing"
    assert (stage_root / "00_detection" / "trace.json").is_file()
    assert (stage_root / "04_crops" / "trace.json").is_file()
    assert not (stage_root / "00_detection" / "trace.png").exists()
    assert not (stage_root / "flow.png").exists()
    assert not (stage_root / "00_detection" / "table.svg").exists()
    payload = json.loads(
        (stage_root / "00_detection" / "trace.json").read_text(encoding="utf-8")
    )
    assert payload["activations"] == "activations absent"
    assert all(path.is_file() for path in written)


def test_optimization_catalog_writes_json_not_a_figure(tmp_path: Path) -> None:
    from evaluation.optimization_protocol import visualization_trace
    from visualization.enrollment import render as render_enrollment
    from visualization.gallery import render as render_gallery
    from visualization.identification import render as render_identification
    from visualization.parsing import render
    from visualization.representation import render as render_representation
    from visualization.search import render as render_search

    trace = visualization_trace()
    callers = (
        ("parsing", render),
        ("identification", render_identification),
        ("representation", render_representation),
        ("enrollment", render_enrollment),
        ("gallery", render_gallery),
        ("search", render_search),
    )
    for stage, caller in callers:
        written = caller(trace, tmp_path)
        stage_root = vis_directory(tmp_path, stage)
        path = stage_root / "optimization.json"
        assert path in written
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "evaluation.optimization_protocol.v1"
        assert payload["stage"] == stage
        assert "runtime" not in payload
        assert "title" not in payload
        assert "caption" not in payload
        assert tuple(payload["substages"]) == STAGE_LAYOUT[stage][1]
        assert not (stage_root / "optimization.png").exists()
        assert not (stage_root / "protocol.png").exists()
        if stage == "parsing":
            assert not (stage_root / "protocol.json").exists()
        assert not (stage_root / "flow.svg").exists()
        assert not (stage_root / "flow.pdf").exists()
        for name in STAGE_LAYOUT[stage][1]:
            assert (stage_root / name / "trace.png").exists() is False
    assert not (tmp_path / "vis" / "06_runtime").exists()
    assert not (tmp_path / "vis" / "06_prototype").exists()


def test_parsing_protocol_writes_json_not_a_figure(tmp_path: Path) -> None:
    from evaluation.parsing_protocol import visualization_trace
    from visualization.parsing import render

    render(visualization_trace(), tmp_path)
    stage_root = vis_directory(tmp_path, "parsing")
    protocol = json.loads((stage_root / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["schema_version"] == "evaluation.parsing_protocol.v1"
    assert not (stage_root / "protocol.png").exists()
    assert not (stage_root / "flow.svg").exists()


def test_identification_embedding_views_have_no_title(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.identification import render

    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.2, size=(24, 8))
    b = rng.normal(1.2, 0.2, size=(24, 8))
    embeddings = np.vstack((a, b)).tolist()
    identity = ["dog-a"] * 24 + ["dog-b"] * 24
    dataset = ["ap10k"] * 24 + ["oxford"] * 24
    view = (["left", "right"] * 24)
    written = render(
        {
            "stage": "identification",
            "substages": {
                "00_appearance": {
                    "embeddings": embeddings,
                    "identity": identity,
                    "dataset": dataset,
                    "view": view,
                }
            },
        },
        tmp_path,
    )
    stage_root = vis_directory(tmp_path, "identification")
    appearance = stage_root / "00_appearance"
    assert (appearance / "pca2.png").is_file()
    assert (appearance / "pca2_dataset.png").is_file()
    assert (appearance / "pca2_identity.png").is_file()
    assert (appearance / "pca2_view.png").is_file()
    assert (appearance / "pca3.png").is_file()
    assert (appearance / "cosine_identity.png").is_file()
    assert (appearance / "pca_var.png").is_file()
    assert (appearance / "dim_contrib.png").is_file()
    assert not (appearance / "pca2.svg").exists()
    assert not (appearance / "pca2.pdf").exists()
    png = (appearance / "pca2_dataset.png").read_bytes()
    assert b"Figure 1" not in png
    assert b"heatmap" not in png.lower()
    record = json.loads((appearance / "trace.json").read_text(encoding="utf-8"))
    assert record["n"] == 48
    assert record["dim"] == 8
    assert record["dim_contrib_kind"] == "identity_ratio"
    assert record["identity_cosine"]["available"] is True
    assert "embeddings" not in record
    assert all(path.is_file() for path in written)
    assert not (stage_root / "01_face" / "pca2.png").exists()


def test_representation_channel_views(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.representation import render

    matrix = np.eye(6, 4).tolist()
    render(
        {
            "stage": "representation",
            "substages": {"01_channels": {"embeddings": matrix}},
        },
        tmp_path,
    )
    channels = vis_directory(tmp_path, "representation") / "01_channels"
    assert (channels / "pca2.png").is_file()
    assert (channels / "pca_var.png").is_file()
    assert (channels / "dim_contrib.png").is_file()
    assert not (channels / "pca2_identity.png").exists()
    assert not (channels / "cosine_identity.png").exists()
    record = json.loads((channels / "trace.json").read_text(encoding="utf-8"))
    assert record["dim_contrib_kind"] == "variance_share"


def test_channel_gap_compares_named_embeddings(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.identification import render
    from visualization.representation import render as render_repr

    rng = np.random.default_rng(1)
    n = 20
    identity = ["dog-a"] * 10 + ["dog-b"] * 10
    strong = np.vstack(
        (rng.normal(0.0, 0.1, size=(10, 4)), rng.normal(1.5, 0.1, size=(10, 4)))
    )
    weak = rng.normal(0.0, 1.0, size=(n, 4))
    render(
        {
            "stage": "identification",
            "substages": {
                "00_appearance": {
                    "embeddings": strong.tolist(),
                    "identity": identity,
                },
                "01_face": {"embeddings": weak.tolist(), "identity": identity},
            },
        },
        tmp_path,
    )
    stage_root = vis_directory(tmp_path, "identification")
    assert (stage_root / "channel_gap.png").is_file()
    assert not (stage_root / "channel_gap.svg").exists()
    payload = json.loads((stage_root / "channel_gap.json").read_text(encoding="utf-8"))
    assert payload["metric"] == "same_minus_different_cosine"
    by_name = {row["channel"]: row["gap"] for row in payload["channels"]}
    assert set(by_name) == {"appearance", "face"}
    assert by_name["appearance"] > by_name["face"]

    render_repr(
        {
            "stage": "representation",
            "substages": {
                "01_channels": {
                    "channels": {
                        "appearance": strong.tolist(),
                        "nose": weak.tolist(),
                    },
                    "identity": identity,
                }
            },
        },
        tmp_path,
    )
    packed = vis_directory(tmp_path, "representation") / "01_channels"
    packed_payload = json.loads((packed / "trace.json").read_text(encoding="utf-8"))
    assert {row["channel"] for row in packed_payload["channel_gap"]} == {
        "appearance",
        "nose",
    }
    assert (packed / "channel_gap.png").is_file()


def test_paper_renderer_does_not_set_title_or_caption(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.contracts import FigureData, SourceBinding
    from visualization.privacy import PublicationScope
    from visualization.rendering.renderer import render_static_figure

    figure = FigureData.create(
        figure_id="02_census_availability",
        kind="census",
        scope=PublicationScope.PUBLIC,
        title="Must not appear on the raster",
        caption="Must not appear on the raster either.",
        limitations=("Must not appear on the raster.",),
        source_bindings=(
            SourceBinding("synthetic-contract-fixture", "fixture.v1", "a" * 64),
        ),
        payload={
            "rows": [
                {"label": "A", "count": 3, "group_index": 0},
                {"label": "B", "count": 5, "group_index": 1},
            ],
            "x_label": "count",
            "x_max": 6,
        },
    )
    render_static_figure(figure, tmp_path)
    png = (tmp_path / "figures" / "02_census_availability.png").read_bytes()
    assert b"Must not appear on the raster" not in png
    assert not (tmp_path / "figures" / "02_census_availability.svg").exists()
    assert not (tmp_path / "figures" / "02_census_availability.pdf").exists()


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
