from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vis.full128_visual_audit import (
    EXPECTED_FILENAMES,
    AuditSample,
    QueryOutcome,
    RankedTemplate,
    neutralized_rgb,
    normalized_neural_input,
    render_png_audit,
    select_occupancy_quantiles,
)
from workflows import render_full128_visual_audit as workflow


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sample(token: str, occupancy: int) -> AuditSample:
    rgb = np.full((224, 224, 3), 64, dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=bool)
    mask.flat[:occupancy] = True
    return AuditSample(token, "identity", "dataset", rgb, mask)


def test_occupancy_selection_is_deterministic_with_token_ties() -> None:
    values = (_sample("z", 10), _sample("b", 20), _sample("a", 20), _sample("q", 30))

    selected = select_occupancy_quantiles(values)

    assert [item.token for item in selected] == ["z", "a", "q"]
    assert select_occupancy_quantiles(tuple(reversed(values))) == selected


def test_neutralization_and_normalization_match_production_order() -> None:
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 128, 0)
    mask = np.zeros((224, 224), dtype=bool)
    mask[0, 0] = True

    neutral = neutralized_rgb(rgb, mask)
    normalized = normalized_neural_input(rgb, mask)

    np.testing.assert_allclose(neutral[0, 0], (1.0, 128 / 255, 0.0))
    np.testing.assert_allclose(neutral[0, 1], (0.485, 0.456, 0.406))
    np.testing.assert_allclose(normalized[0, 1], (0.0, 0.0, 0.0))


def test_native_routes_are_distinguished_from_foreground_masks() -> None:
    from vis.full128_visual_audit import _mask_title

    assert _mask_title("NATIVE_FACE") == "source-frame validity\n(no segmentation)"
    assert _mask_title("NATIVE_HEAD") == "source-frame validity\n(no segmentation)"
    assert _mask_title("BODY_PARSING") == "parser foreground mask"
    assert _mask_title("BODY_MASK") == "authoritative foreground mask"


def test_route_is_read_from_the_cache_bundle_bound_to_the_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb_sha, mask_sha, crop_sha, cache_sha = (
        _sha("rgb"),
        _sha("mask"),
        _sha("crop"),
        _sha("cache"),
    )
    row = {
        "artifact": {
            "full_rgb_path": "/tmp/sample/full.png",
            "full_rgb_sha256": rgb_sha,
            "full_mask_sha256": mask_sha,
            "crop_record_sha256": crop_sha,
            "full_segment_cache_sha256": cache_sha,
        }
    }
    bundle = {
        "cache_sha256": cache_sha,
        "cache": {
            "records": [
                {
                    "crop": {
                        "full_rgb_sha256": rgb_sha,
                        "full_mask_sha256": mask_sha,
                        "crop_record_sha256": crop_sha,
                        "route": "NATIVE_FACE",
                    }
                }
            ]
        },
    }
    monkeypatch.setattr(workflow, "_read", lambda path: bundle)
    monkeypatch.setattr(
        workflow,
        "validate_full_segment_cache_bundle",
        lambda value: value["cache"],
    )

    assert workflow._route(row) == "NATIVE_FACE"


def test_input_lane_selection_decodes_rgb_only_for_three_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        str(index): {
            "sample_token": str(index),
            "registered_identity_id": None,
            "dataset_name": "dogfacenet224",
            "artifact": {
                "full_rgb_path": f"{index}.png",
                "full_rgb_sha256": _sha(f"rgb:{index}"),
                "full_mask_path": f"{index}-mask.png",
                "full_mask_sha256": _sha(f"mask:{index}"),
                "crop_record_sha256": _sha(f"crop:{index}"),
            },
        }
        for index in range(5)
    }
    occupancies = iter((0.1, 0.2, 0.3, 0.4, 0.5))
    rgb_reads = []
    monkeypatch.setattr(
        workflow,
        "read_full128_mask",
        lambda sample: np.full((224, 224), next(occupancies), dtype=np.float32),
    )

    def read_crop(sample: Any) -> tuple[np.ndarray, np.ndarray]:
        rgb_reads.append(sample.sample_id)
        return (
            np.zeros((224, 224, 3), dtype=np.uint8),
            np.ones((224, 224), dtype=bool),
        )

    monkeypatch.setattr(workflow, "read_full128_crop", read_crop)
    monkeypatch.setattr(workflow, "_route", lambda row: "BODY_PARSING")

    assert [row["sample_token"] for row in workflow._select_occupancy_rows(
        records, "dogfacenet224"
    )] == ["0", "2", "4"]
    occupancies = iter((0.1, 0.2, 0.3, 0.4, 0.5))
    lanes = workflow._input_lanes(records)

    assert len(lanes["successor"][0][1]) == 3
    assert rgb_reads == ["0", "2", "4"]


def test_renderer_writes_only_the_expected_png_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_render(*args: Any, **kwargs: Any) -> None:
        path = next(value for value in args if isinstance(value, Path))
        path.write_bytes(b"png")

    monkeypatch.setattr("vis.full128_visual_audit._pyplot", lambda: object())
    monkeypatch.setattr("vis.full128_visual_audit._render_input_plate", fake_render)
    monkeypatch.setattr("vis.full128_visual_audit._render_gallery_plate", fake_render)
    monkeypatch.setattr("vis.full128_visual_audit._render_retrieval_plate", fake_render)
    monkeypatch.setattr("vis.full128_visual_audit._render_embedding_plate", fake_render)
    monkeypatch.setattr("vis.full128_visual_audit._render_trace_plate", fake_render)
    ranked = (RankedTemplate("q", 1.0, True), RankedTemplate("k", 0.0, False))
    outcome = QueryOutcome("q", ("DEV", "dataset", 1), 1, 1.0, ranked, ranked)
    samples = {"q": _sample("q", 1), "k": _sample("k", 1)}

    render_png_audit(
        output_dir=tmp_path,
        input_lanes={name: [] for name in ("successor", "auxiliary", "terminal")},
        gallery_query=outcome,
        outcomes={name: outcome for name in ("high", "middle", "low")},
        samples=samples,
        dev_population={"dataset": (("q",), ("k",))},
        b3_vectors={token: np.ones(128, dtype=np.float32) for token in samples},
        b5_vectors={token: np.ones(128, dtype=np.float32) for token in samples},
        trace={
            "private_samples": {"query_sample_token": "q", "key_sample_token": "k"},
            "available_maps": {},
        },
    )

    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == EXPECTED_FILENAMES
    assert {path.suffix for path in tmp_path.iterdir()} == {".png"}


def test_reconstructed_rank_disagreement_fails_closed() -> None:
    query, positive, negative = "q", "p", "n"
    records = {
        query: {"registered_identity_id": "a"},
        positive: {"registered_identity_id": "a"},
        negative: {"registered_identity_id": "b"},
    }
    panel = {
        "cohorts": [
            {
                "scope": "DEV",
                "dataset_name": "dataset",
                "enrollment_k": 1,
                "status": "AVAILABLE",
                "query_sample_tokens": [query],
                "gallery_sample_tokens": [positive, negative],
            }
        ]
    }
    vectors = {
        query: np.array([1, 0], dtype=np.float32),
        positive: np.array([1, 0], dtype=np.float32),
        negative: np.array([0, 1], dtype=np.float32),
    }
    report = {
        "B3": {("DEV", "dataset", 1, query): 2},
        "B5-SPATIAL": {("DEV", "dataset", 1, query): 1},
    }

    with pytest.raises(ValueError, match="reported relevant rank"):
        workflow._dev_outcomes(panel, records, vectors, vectors, report)


def test_private_trace_asset_binding_rejects_substituted_artifact() -> None:
    token = _sha("private")
    terminal_token = _sha("terminal")
    inventory = {
        "inventory": {
            "artifact_root": "/tmp",
            "successor_population": [
                {
                    "sample_token": token,
                    "artifact": {
                        "full_rgb_path": "/tmp/rgb.png",
                        "full_rgb_sha256": _sha("rgb"),
                        "full_mask_path": "/tmp/mask.png",
                        "full_mask_sha256": _sha("mask"),
                        "crop_record_sha256": _sha("crop"),
                    },
                }
            ],
            "identity_free_auxiliary_population": [],
            "terminal_exclusions": [
                {
                    "sample_token": terminal_token,
                    "artifact": {
                        "full_rgb_path": "/tmp/rgb.png",
                        "full_rgb_sha256": _sha("rgb"),
                        "full_mask_path": "/tmp/mask.png",
                        "full_mask_sha256": _sha("mask"),
                        "crop_record_sha256": _sha("crop"),
                    },
                }
            ],
        },
    }
    assert terminal_token in workflow._records(inventory, Path("/tmp"))
    trace = {
        "successor_id": "B5-SPATIAL",
        "artifact_bindings": {"evaluation_cache_descriptor_sha256": _sha("cache")},
        "private_samples": {"query_sample_token": token, "key_sample_token": token},
        "input_bindings": {
            "query": {
                "rgb_sha256": _sha("other"),
                "mask_sha256": _sha("mask"),
                "crop_record_sha256": _sha("crop"),
            },
            "key": {
                "rgb_sha256": _sha("rgb"),
                "mask_sha256": _sha("mask"),
                "crop_record_sha256": _sha("crop"),
            },
        },
    }

    with pytest.raises(ValueError, match="input binding"):
        workflow._validate_trace_binding(
            trace, inventory, {"cache_descriptor_sha256": _sha("cache")}
        )


def test_rendered_labels_do_not_include_private_token_or_path() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    from vis.full128_visual_audit import _sample_triplet

    figure, axes = pyplot.subplots(1, 3)
    private_token = "/secure/private/path/" + _sha("token")
    _sample_triplet(axes, _sample(private_token, 1), "dataset\nlow occupancy")

    displayed = " ".join(text.get_text() for axis in axes for text in axis.texts)
    displayed += " ".join(axis.get_title() + axis.get_ylabel() for axis in axes)
    assert private_token not in displayed
    assert "/secure/private/path" not in displayed
    pyplot.close(figure)
