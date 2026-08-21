from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.repo_root import REPO_ROOT as ROOT
from visualization.registry import FIGURE_REGISTRY
from visualization.rendering.pipeline import (
    STAGE_LAYOUT,
    clear_visualizations,
    vis_directory,
)


def test_paper_registry_is_not_pipeline_vis_numbering() -> None:
    paper_ids = tuple(spec.figure_id for spec in FIGURE_REGISTRY)
    vis_dirs = tuple(layout[0] for layout in STAGE_LAYOUT.values())
    assert paper_ids[0] == "00_evidence_ladder"
    assert vis_dirs == (
        "00_parsing",
        "01_representation",
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


def test_pipeline_render_clears_previous_stage_files(tmp_path: Path) -> None:
    stage_root = vis_directory(tmp_path, "parsing")
    stage_root.mkdir(parents=True)
    stale = stage_root / "stale.pdf"
    stale.write_bytes(b"stale")

    from visualization.parsing import render

    render({"stage": "parsing", "substages": {}}, tmp_path)
    assert not stale.exists()


def test_clear_visualizations_removes_previous_pipeline_tree(tmp_path: Path) -> None:
    stale = tmp_path / "vis" / "02_identification" / "stale.pdf"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    clear_visualizations(tmp_path)
    assert (tmp_path / "vis").is_dir()
    assert not tuple((tmp_path / "vis").iterdir())


def test_parsing_detection_writes_dataset_pdf(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.parsing import render

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    samples = []
    for index in range(5):
        source = asset_root / f"{index}-source.png"
        segment = asset_root / f"{index}-segment.png"
        Image.new("RGB", (16, 12), (index * 20, 80, 140)).save(source)
        Image.new("RGB", (8, 8), (140, 80, index * 20)).save(segment)
        samples.append(
            {
                "source_image": f"assets/{source.name}",
                "segment_image": f"assets/{segment.name}",
                "detector_boxes": [[2, 2, 14, 10]],
            }
        )
    written = render(
        {
            "stage": "parsing",
            "substages": {
                "00_detection": {"samples": {"yt-bb-dog": samples}},
                "01_segmentation": {
                    "segmentation_metrics": [
                        {
                            "dataset_name": "yt-bb-dog",
                            "detected_inputs": 5,
                            "parsed_instances": 5,
                            "usable": 3,
                            "review": 1,
                            "unusable": 1,
                            "mean_shape_iou": 0.8,
                            "mean_ownership_retention": 0.9,
                        }
                    ],
                    "parser_backbones": "RF-DETR + BiRefNet",
                    "samples": {
                        "yt-bb-dog": [
                            {
                                "segment_input_image": sample["segment_image"],
                                "segment_output_image": sample["segment_image"],
                                "background_image": sample["source_image"],
                            }
                            for sample in samples
                        ]
                    },
                },
            },
        },
        tmp_path,
        asset_root=asset_root.parent,
    )
    pdf = (
        vis_directory(tmp_path, "parsing")
        / "00_detection"
        / "detection_box_yt-bb-dog.pdf"
    )
    metrics = (
        vis_directory(tmp_path, "parsing") / "00_detection" / "detection_metrics.pdf"
    )
    segmentation = (
        vis_directory(tmp_path, "parsing")
        / "01_segmentation"
        / "segmentation_yt-bb-dog.pdf"
    )
    segmentation_metrics = (
        vis_directory(tmp_path, "parsing")
        / "01_segmentation"
        / "segmentation_metrics.pdf"
    )
    assert pdf.is_file()
    assert metrics.is_file()
    assert segmentation.is_file()
    assert segmentation_metrics.is_file()
    assert pdf in written
    assert metrics in written
    assert segmentation in written
    assert segmentation_metrics in written
    assert pdf.read_bytes().startswith(b"%PDF")
    assert metrics.read_bytes().startswith(b"%PDF")
    assert segmentation.read_bytes().startswith(b"%PDF")
    assert segmentation_metrics.read_bytes().startswith(b"%PDF")
    assert not pdf.with_suffix(".png").exists()


def test_optimization_catalog_writes_json_not_a_figure(tmp_path: Path) -> None:
    from evaluation.optimization_protocol import visualization_trace
    from visualization.enrollment import render as render_enrollment
    from visualization.gallery import render as render_gallery
    from visualization.parsing import render
    from visualization.representation import render as render_representation
    from visualization.search import render as render_search

    trace = visualization_trace()
    callers = (
        ("parsing", render),
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


def test_identification_has_no_observer_stage() -> None:
    assert "identification" not in STAGE_LAYOUT


def test_representation_embedding_diagnostics_are_pdf_and_labeled(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.representation import render

    rng = np.random.default_rng(0)
    embeddings = rng.normal(0.0, 0.2, size=(48, 8))
    identity = ["dog-a"] * 24 + ["dog-b"] * 24
    dataset = ["yt-bb-dog"] * 24 + ["sibetan"] * 24
    written = render(
        {
            "stage": "representation",
            "substages": {
                "01_channels": {
                    "embeddings": embeddings.tolist(),
                    "identity": identity,
                    "dataset": dataset,
                    "backbone_id": "dinov2-small:test",
                }
            },
        },
        tmp_path,
    )
    channels = vis_directory(tmp_path, "representation") / "01_channels"
    for name in (
        "embedding_heatmap.pdf",
        "pca_variance.pdf",
        "pca_components.pdf",
        "pca_identity.pdf",
    ):
        assert (channels / name).is_file()
        assert (channels / name).read_bytes().startswith(b"%PDF")
    assert not list(channels.glob("*.png"))
    assert not (channels / "cosine_identity.pdf").exists()
    assert not (channels / "pca3.pdf").exists()
    assert not (vis_directory(tmp_path, "representation") / "00_evidence").exists()
    assert not (vis_directory(tmp_path, "representation") / "02_quality").exists()
    record = json.loads((channels / "trace.json").read_text(encoding="utf-8"))
    assert record["n"] == 48
    assert record["dim"] == 8
    assert record["group"] == "identity"
    assert record["heatmap_group"] == "dataset"
    assert record["heatmap_dimensions"] == list(range(8))
    assert record["heatmap_labeled_dimensions"] == list(range(8))
    assert record["heatmap_rows"] == 48
    assert record["pca_variance_components"] == 8
    assert record["pca_component_dimensions"] == list(range(8))
    assert record["pca_component_labeled_dimensions"] == list(range(8))
    assert record["pca_group"] == "dataset"
    assert record["pca_group_counts"] == {"sibetan": 24, "yt-bb-dog": 24}
    assert record["backbone_id"] == "dinov2-small:test"
    assert set(record["group_labels"].values()) == {"dog-a", "dog-b"}
    assert len(record["pca_components_top"]) == 3
    assert "embeddings" not in record
    assert all(path.is_file() for path in written)


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
    assert (channels / "embedding_heatmap.pdf").is_file()
    assert (channels / "pca_variance.pdf").is_file()
    assert (channels / "pca_components.pdf").is_file()
    assert not (channels / "pca_identity.pdf").exists()
    record = json.loads((channels / "trace.json").read_text(encoding="utf-8"))
    assert record["norm_mean"] == pytest.approx(float(np.mean(np.linalg.norm(matrix, axis=1))))


def test_representation_heatmap_prefers_detection_groups(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.representation import render

    render(
        {
            "stage": "representation",
            "substages": {
                "01_channels": {
                    "embeddings": np.eye(6, 4).tolist(),
                    "dataset": ["sibetan"] * 6,
                    "detection": [
                        "undetected_samples",
                        "detected_samples",
                        "undetected_samples",
                        "detected_samples",
                        "detected_samples",
                        "undetected_samples",
                    ],
                }
            },
        },
        tmp_path,
    )
    record = json.loads(
        (
            vis_directory(tmp_path, "representation")
            / "01_channels"
            / "trace.json"
        ).read_text(encoding="utf-8")
    )
    assert record["heatmap_group"] == "detection"
    assert record["heatmap_group_counts"] == {
        "detected_samples": 3,
        "undetected_samples": 3,
    }


def test_embedding_heatmap_keeps_all_dimensions_in_original_order() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from visualization.rendering.embeddings import _draw_heatmap

    figure = plt.figure()
    try:
        _draw_heatmap(
            figure,
            np.arange(18, dtype=float).reshape(3, 6),
            labels=("dataset-a", "dataset-a", "dataset-b"),
            dimensions=np.asarray([4, 1]),
        )
        axis = figure.axes[0]
        assert axis.images[0].get_array().shape == (3, 6)
        assert axis.get_xlim() == pytest.approx((-0.5, 5.5))
        assert axis.get_xticks().tolist() == [0, 1, 4]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "D000",
            "D001",
            "D004",
        ]
    finally:
        plt.close(figure)


def test_pca_components_keeps_all_dimensions_in_original_order() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from visualization.rendering.embeddings import _draw_pca_components

    figure = plt.figure()
    try:
        _draw_pca_components(
            figure,
            np.arange(18, dtype=float).reshape(3, 6),
            dimensions=np.asarray([4, 1]),
        )
        axis = figure.axes[0]
        assert axis.images[0].get_array().shape == (3, 6)
        assert axis.get_xlim() == pytest.approx((-0.5, 5.5))
        assert axis.get_xticks().tolist() == [0, 1, 4]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "D000",
            "D001",
            "D004",
        ]
    finally:
        plt.close(figure)


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
