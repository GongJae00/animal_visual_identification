from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from visualization.adapters import adapt_master_results_table
from visualization.contracts import FigureContractError, FigureData, SourceBinding
from visualization.privacy import PublicationScope
from visualization.publication import publish
from visualization.recipes import validate_recipe
from visualization.registry import FIGURE_REGISTRY
from visualization.static_index import build_static_index


def _binding() -> SourceBinding:
    return SourceBinding("synthetic-contract-fixture", "fixture.v1", "a" * 64)


def _figure(
    figure_id: str,
    kind: str,
    payload: dict,
    *,
    title: str = "Contract figure",
    scope: PublicationScope = PublicationScope.PUBLIC,
) -> FigureData:
    return FigureData.create(
        figure_id=figure_id,
        kind=kind,
        scope=scope,
        title=title,
        caption="Synthetic values exercise rendering only.",
        limitations=("Synthetic fixture; this is not biometric validation.",),
        source_bindings=(_binding(),),
        payload=payload,
    )


def _payloads(image_sha256: str = "b" * 64) -> dict[str, tuple[str, dict]]:
    return {
        "architecture": (
            "03_role_dependency_closure",
            {
                "nodes": [
                    {"label": "Crop", "layer": 0, "group_index": 0},
                    {"label": "Representation", "layer": 1, "group_index": 1},
                    {"label": "Retrieval", "layer": 2, "group_index": 2},
                ],
                "edges": [
                    {"source": 0, "target": 1, "label": "extract"},
                    {"source": 1, "target": 2, "label": "score"},
                ],
            },
        ),
        "ladder": (
            "00_evidence_ladder",
            {
                "steps": [
                    {
                        "label": "Contract",
                        "detail": "content bound",
                        "status": "established",
                    },
                    {
                        "label": "Evaluation",
                        "detail": "protocol dependent",
                        "status": "conditional",
                    },
                ]
            },
        ),
        "census": (
            "02_census_availability",
            {
                "rows": [
                    {"label": "Train", "count": 12, "group_index": 0},
                    {"label": "Test", "count": 8, "group_index": 1},
                ],
                "x_label": "Samples",
                "x_max": 15,
            },
        ),
        "result_forest": (
            "13_primary_results_paired_deltas",
            {
                "rows": [
                    {"label": "Method A", "estimate": 0.7, "lower": 0.6, "upper": 0.8},
                    {
                        "label": "Method B",
                        "estimate": 0.8,
                        "lower": 0.72,
                        "upper": 0.88,
                    },
                ],
                "x_label": "Rank-1",
                "x_min": 0.0,
                "x_max": 1.0,
                "reference": 0.5,
            },
        ),
        "cosine_distribution": (
            "05_score_distributions",
            {
                "bin_edges": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "series": [
                    {"label": "Same class", "counts": [0, 1, 3, 7]},
                    {"label": "Different class", "counts": [2, 5, 2, 1]},
                ],
                "x_label": "Cosine similarity",
            },
        ),
        "embedding_spectrum": (
            "09_embedding_spectrum_pca",
            {
                "components": [1, 2, 3, 4],
                "values": [0.5, 0.25, 0.15, 0.1],
                "x_label": "Component",
                "y_label": "Explained variance ratio",
                "x_min": 1,
                "x_max": 4,
                "y_min": 0,
                "y_max": 0.6,
                "log_y": False,
            },
        ),
        "pca_projection": (
            "09_embedding_spectrum_pca",
            {
                "points": [
                    {"x": -0.5, "y": 0.1, "group": "Cohort A"},
                    {"x": 0.4, "y": -0.2, "group": "Cohort B"},
                ],
                "x_limits": [-1.0, 1.0],
                "y_limits": [-1.0, 1.0],
                "x_label": "PC1",
                "y_label": "PC2",
            },
        ),
        "embedding_topology": (
            "03_role_dependency_closure",
            {
                "nodes": [
                    {"x": -0.5, "y": 0.1, "group": "Cohort A"},
                    {"x": 0.4, "y": -0.2, "group": "Cohort B"},
                ],
                "edges": [{"source": 0, "target": 1}],
                "x_limits": [-1.0, 1.0],
                "y_limits": [-1.0, 1.0],
                "x_label": "Layout x",
                "y_label": "Layout y",
            },
        ),
        "gallery_composition": (
            "10_gallery_composition",
            {
                "rows": [
                    {"label": "One template", "value": 7, "group_index": 0},
                    {"label": "Multiple templates", "value": 3, "group_index": 1},
                ],
                "center_label": "10 identities",
            },
        ),
        "ranked_retrieval": (
            "12_private_ranked_qkv",
            {
                "query": {
                    "path": "query.png",
                    "sha256": image_sha256,
                    "label": "Query",
                },
                "candidates": [
                    {
                        "path": "candidate.png",
                        "sha256": image_sha256,
                        "label": "Candidate A",
                        "rank": 1,
                        "score": 0.75,
                        "margin": None,
                        "outcome": "relevant",
                    }
                ],
            },
        ),
        "embedding_diagnostics": (
            "09_embedding_spectrum_pca",
            {
                "series": [
                    {
                        "label": "Successor A",
                        "style_index": 0,
                        "sample_count": 4,
                        "explained_variance": [0.6, 0.25, 0.1],
                        "cumulative_variance": [0.6, 0.85, 0.95],
                    }
                ],
                "manifest": [
                    {
                        "alias": "Successor A",
                        "description": "Synthetic fixture",
                        "sample_count": 4,
                        "cache_descriptor_sha256": "c" * 64,
                        "displayed": True,
                    }
                ],
                "component_count": 3,
                "variance_y_max": 0.7,
            },
        ),
        "model_ladder": (
            "00_evidence_ladder",
            {
                "variants": [
                    {
                        "alias": "B0-FV",
                        "description": "Classical baseline",
                        "status": "GO",
                        "reported": True,
                        "column": 0,
                        "row": 0.5,
                    },
                    {
                        "alias": "B1",
                        "description": "Scratch baseline",
                        "status": "NO_GO",
                        "reported": True,
                        "column": 1,
                        "row": 0.5,
                    },
                ],
                "edges": [{"source": "B0-FV", "target": "B1"}],
                "boundaries": [
                    {"label": "DEV", "detail": "selection", "status": "GO/NO_GO"},
                    {"label": "CAL", "detail": "reporting", "status": "NO_SELECTION"},
                    {"label": "EXPOSED", "detail": "diagnostic", "status": "BOUNDARY"},
                ],
            },
        ),
        "score_rank_distributions": (
            "11_cosine_rank_distributions",
            {
                "rank_series": [
                    {
                        "label": "Successor A | DEV",
                        "ranks": [1, 5, 10],
                        "values": [0.6, 0.8, 0.9],
                    }
                ],
                "rank_ticks": [1, 5, 10],
                "rank_x_max": 10,
                "cosine_distribution": None,
            },
        ),
    }


def _save_fixture_image(path: Path) -> str:
    Image.new("RGB", (32, 24), (80, 120, 160)).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_lightweight_import_does_not_load_rendering_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import visualization; "
                "assert 'matplotlib' not in sys.modules; "
                "assert 'matplotlib.pyplot' not in sys.modules; "
                "assert 'PIL' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_registry_is_complete_ordered_and_recipe_backed() -> None:
    assert len(FIGURE_REGISTRY) == 18
    assert [spec.figure_id[:2] for spec in FIGURE_REGISTRY] == [
        f"{index:02d}" for index in range(18)
    ]
    payloads = _payloads()
    implemented = set(payloads)
    assert {spec.kind for spec in FIGURE_REGISTRY} <= implemented


def test_figure_data_detects_tampering_and_rejects_private_public_data() -> None:
    figure_id, payload = _payloads()["census"]
    figure = _figure(figure_id, "census", payload)
    bundle = figure.to_bundle()
    tampered = copy.deepcopy(bundle)
    tampered["figure_data"]["title"] = "Changed"
    with pytest.raises(FigureContractError, match="tampered"):
        FigureData.from_bundle(tampered)

    private_id = copy.deepcopy(payload)
    private_id["sample_id"] = "subject-1"
    with pytest.raises(ValueError, match="private identifier"):
        _figure(figure_id, "census", private_id)
    absolute_path = copy.deepcopy(payload)
    absolute_path["note"] = "/private/workstation/data.json"
    with pytest.raises(ValueError, match="absolute path"):
        _figure(figure_id, "census", absolute_path)


def test_scope_selection_fails_closed(tmp_path: Path) -> None:
    figure_id, payload = _payloads()["census"]
    paper = _figure(figure_id, "census", payload, scope=PublicationScope.PAPER)
    with pytest.raises(PermissionError, match="exceeds publication scope"):
        publish((paper,), tmp_path / "public", target_scope=PublicationScope.PUBLIC)


def test_static_index_escapes_text_and_has_no_remote_assets() -> None:
    figure_id, payload = _payloads()["census"]
    figure = _figure(figure_id, "census", payload, title='<A & "B">')
    page = build_static_index((figure,), target_scope="public")
    assert '<A & "B">' not in page
    assert "&lt;A &amp; &quot;B&quot;&gt;" in page
    assert "http://" not in page and "https://" not in page
    assert 'src="figures/02_census_availability.svg"' in page


def test_all_recipe_families_validate_and_render(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.renderer import render_static_figure

    digest = _save_fixture_image(tmp_path / "query.png")
    (tmp_path / "candidate.png").write_bytes((tmp_path / "query.png").read_bytes())
    for kind, (figure_id, payload) in _payloads(digest).items():
        figure = _figure(figure_id, kind, payload)
        validate_recipe(figure)
        paths = render_static_figure(figure, tmp_path / "rendered", asset_root=tmp_path)
        assert tuple(Path(path).suffix for path in paths) == (".svg", ".pdf", ".png")
        assert all((tmp_path / "rendered" / path).stat().st_size > 0 for path in paths)


def test_publication_is_byte_deterministic_and_inventory_is_ordered(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    architecture_id, architecture_payload = _payloads()["architecture"]
    census_id, census_payload = _payloads()["census"]
    figures = (
        _figure(census_id, "census", census_payload),
        _figure(architecture_id, "architecture", architecture_payload),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    publish(figures, first, target_scope=PublicationScope.PUBLIC)
    publish(figures, second, target_scope=PublicationScope.PUBLIC)
    assert _hashes(first) == _hashes(second)
    inventory = json.loads(
        (first / "output_inventory.json").read_text(encoding="utf-8")
    )
    paths = [entry["path"] for entry in inventory["entries"]]
    assert paths == [
        "figures/02_census_availability.svg",
        "figures/02_census_availability.pdf",
        "figures/02_census_availability.png",
        "figures/03_role_dependency_closure.svg",
        "figures/03_role_dependency_closure.pdf",
        "figures/03_role_dependency_closure.png",
        "index.html",
    ]
    provenance = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
    serialized = json.dumps(provenance, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert all(
        str(tmp_path).encode() not in path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    )
    assert provenance["renderer"]["fingerprint"]
    assert provenance["style"]["fingerprint"]


def test_no_overwrite_precedes_rendering_or_asset_access(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish(
            (),
            output,
            target_scope=PublicationScope.PUBLIC,
            asset_root=tmp_path / "missing",
        )


def test_contact_sheet_rejects_tampered_asset(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from visualization.renderer import render_static_figure

    digest = _save_fixture_image(tmp_path / "query.png")
    (tmp_path / "candidate.png").write_bytes((tmp_path / "query.png").read_bytes())
    figure_id, payload = _payloads(digest)["ranked_retrieval"]
    figure = _figure(figure_id, "ranked_retrieval", payload)
    (tmp_path / "candidate.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        render_static_figure(figure, tmp_path / "rendered", asset_root=tmp_path)


def test_master_results_adapter_rejects_tampering() -> None:
    without_hash = {
        "schema_version": "cvi.master_results_table.v1",
        "source_report_sha256s": ["a" * 64],
        "columns": ["section", "metric_name", "value"],
        "rows": [
            {
                "section": "retrieval",
                "metric_name": "Rank-1",
                "value": 0.8,
                "lower_bound": 0.7,
                "upper_bound": 0.9,
                "region": "Full",
                "gallery_scope": "isolated",
            }
        ],
    }
    from foundation.provenance import content_sha256

    table = {**without_hash, "table_sha256": content_sha256(without_hash)}
    assert adapt_master_results_table(table).kind == "result_forest"
    table["rows"][0]["value"] = 0.1
    with pytest.raises(FigureContractError, match="tampered"):
        adapt_master_results_table(table)


def test_workflow_renders_figure_data_bundle(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figure_id, payload = _payloads()["census"]
    input_path = tmp_path / "figure.json"
    input_path.write_text(
        json.dumps(_figure(figure_id, "census", payload).to_bundle()),
        encoding="utf-8",
    )
    output = tmp_path / "publication"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "workflows" / "render_research_visualizations.py"),
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--scope",
            "public",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["event"] == "research_visualizations_rendered"
    assert receipt["figure_ids"] == ["02_census_availability"]
    assert (output / "figures" / "02_census_availability.svg").is_file()


def test_tracked_json_schema_accepts_normalized_bundle() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    figure_id, payload = _payloads()["census"]
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "artifact_contracts"
        / "schemas"
        / "cvi.figure_data.bundle.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(
        _figure(figure_id, "census", payload).to_bundle()
    )
