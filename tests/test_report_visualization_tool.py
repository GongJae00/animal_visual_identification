from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from apps.report import generate as tool


def _metric(rank1: float, mrr: float, *, queries: int = 20, identities: int = 5, templates: int = 5) -> dict:
    return {
        "Rank-1": rank1,
        "MRR": mrr,
        "closed_set": True,
        "num_queries": queries,
        "num_gallery_identities": identities,
        "num_gallery_templates": templates,
    }


def _diagnostic(nominal: int, effective: float, centroid: float, norm: float, std: float) -> dict:
    return {
        "schema_version": "cvi.embedding_diagnostics.v1",
        "nominal_dimension": nominal,
        "normalized_centered_covariance_spectrum": {
            "effective_rank_entropy": effective,
        },
        "normalized_directional_geometry": {"centroid_norm": centroid},
        "raw_norm_summary": {"mean": norm, "standard_deviation": std},
    }


def _report() -> dict:
    candidates = {
        "max": _metric(0.88, 0.92, queries=14, identities=4, templates=12),
        "mean": _metric(0.82, 0.89, queries=14, identities=4, templates=12),
    }
    final = {}
    for shot, queries, identities, templates, shift in (
        ("one_shot", 20, 5, 5, 0.0),
        ("three_shot", 14, 4, 12, 0.08),
    ):
        final[shot] = {
            "channel_context": {
                "Appearance-v3-max": _metric(0.76 + shift, 0.82 + shift, queries=queries, identities=identities, templates=templates),
                "Face-v4-max": _metric(0.74 + shift, 0.81 + shift, queries=queries, identities=identities, templates=templates),
            },
            "fused": _metric(0.76 + shift, 0.82 + shift, queries=queries, identities=identities, templates=templates),
            "fused_identity_clustered_ci": {
                "Rank-1": {
                    "confidence_level": 0.95,
                    "lower_bound": 0.65 + shift,
                    "upper_bound": 0.86 + shift,
                    "cluster_count": identities,
                    "query_row_count": queries,
                }
            },
        }
    return {
        "schema_version": tool.REPORT_SCHEMA,
        "status": tool.REPORT_STATUS,
        "privacy": {
            "per_pair_values_serialized": False,
            "per_sample_ids_serialized": False,
            "per_sample_vectors_serialized": False,
            "query_rows_serialized": False,
        },
        "calibration_results": {
            "fold_protocol": {"n_splits": 5},
            "one_shot_oof": {
                "Appearance-v3": _metric(0.78, 0.84),
                "Face-v4": _metric(0.75, 0.82),
                "fused": _metric(0.78, 0.84),
            },
            "three_shot_oof_candidates": candidates,
            "simplex": {
                "channel_names": ["Appearance-v3", "Face-v4"],
                "weights": [1.0, 0.0],
            },
        },
        "frozen_selection": {
            "selected_aggregation_id": "max",
            "simplex_weights": [1.0, 0.0],
        },
        "final_results": final,
        "embedding_diagnostics": {
            "final": {
                "Appearance-v3-pre-L2-CLS": _diagnostic(384, 62.0, 0.45, 49.0, 0.7),
                "Face-v4-DINO-baseline-384D": _diagnostic(384, 61.0, 0.47, 49.1, 0.7),
                "Face-v4-final-640D": _diagnostic(640, 62.0, 0.51, 1.0, 3e-8),
                "Face-v4-regional-projection-256D": _diagnostic(256, 19.0, 0.9999, 40.9, 0.07),
            }
        },
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_image(path: Path, color: tuple[int, int, int], *, mode: str = "RGB") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L":
        image = Image.new(mode, (96, 96), 220)
    else:
        image = Image.new(mode, (160, 120), color)
    image.save(path, format="PNG", compress_level=9)
    return _sha(path)


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "secure"
    datasets = data_root / "datasets"
    colors = {
        "ap10k-dog": (180, 95, 70),
        "dogfacenet224": (70, 120, 180),
        "mpdd": (90, 165, 120),
        "sibetan": (155, 125, 75),
        "yt-bb-dog": (100, 85, 160),
    }
    samples = {}
    directories = {
        "ap10k-dog": "ap10k",
        "dogfacenet224": "dogfacenet224",
        "mpdd": "mpdd",
        "sibetan": "sibetan",
        "yt-bb-dog": "yt-bb-dog",
    }
    for name, directory in directories.items():
        path = datasets / directory / "source.png"
        digest = _save_image(path, colors[name])
        samples[name] = SimpleNamespace(image_path="source.png", image_sha256=digest)

    def adapter(name: str):
        return lambda _: (samples[name],)

    monkeypatch.setattr(
        tool,
        "DATASET_SPECS",
        (
            ("dogfacenet224", "dogfacenet224", "TRAIN", adapter("dogfacenet224")),
            ("mpdd", "mpdd", "VALIDATION", adapter("mpdd")),
            ("sibetan", "sibetan", "VALIDATION", adapter("sibetan")),
            ("yt-bb-dog", "yt-bb-dog", "TRAIN", adapter("yt-bb-dog")),
        ),
    )

    roi_root = tmp_path / "roi"
    artifact_fields = {}
    for key, field, color, mode in (
        ("dog.png", "dog_crop", (190, 100, 75), "RGB"),
        ("face.png", "face_crop", (75, 150, 190), "RGB"),
        ("nose.png", "weak_nose_crop", (210, 145, 70), "RGB"),
        ("mask.png", "source_valid_mask", (0, 0, 0), "L"),
    ):
        digest = _save_image(roi_root / key, color, mode=mode)
        artifact_fields[f"{field}_path"] = key
        artifact_fields[f"{field}_sha256"] = digest
    point_names = (
        "left_eye", "right_eye", "nose_center", "neck", "tail_base",
        "left_shoulder", "left_elbow", "left_front_paw", "right_shoulder",
        "right_elbow", "right_front_paw", "left_hip", "left_knee",
        "left_back_paw", "right_hip", "right_knee", "right_back_paw",
    )
    record = {
        "image_path": "source.png",
        "image_sha256": samples["ap10k-dog"].image_sha256,
        "dog_bbox_xyxy": [20.0, 15.0, 140.0, 110.0],
        "body_keypoints": {
            name: [35.0 + index * 5.0, 25.0 + (index % 5) * 15.0, 0.9]
            for index, name in enumerate(point_names)
        },
        **artifact_fields,
    }
    manifest = {
        "schema_version": tool.ROI_MANIFEST_SCHEMA,
        "dataset_name": "ap10k-dog",
        "records": [record],
    }
    bundle = {
        "schema_version": tool.ROI_BUNDLE_SCHEMA,
        "manifest_sha256": tool._canonical_sha256(manifest),
        "manifest": manifest,
    }
    roi_manifest = roi_root / "roi_manifest.json"
    roi_manifest.parent.mkdir(parents=True, exist_ok=True)
    roi_manifest.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(_report(), sort_keys=True), encoding="utf-8")
    return data_root, roi_manifest, report_path


def test_metric_extraction_uses_report_values() -> None:
    metrics = tool.extract_metrics(_report())
    assert metrics["folds"] == 5
    assert metrics["weights"] == [1.0, 0.0]
    assert metrics["calibration_one"]["face"]["mrr"] == pytest.approx(0.82)
    assert metrics["calibration_candidates"]["mean"]["rank1"] == pytest.approx(0.82)
    assert metrics["final"]["three_shot"]["fused"]["rank1"] == pytest.approx(0.84)
    assert metrics["diagnostics"]["regional"]["effective_rank"] == pytest.approx(19.0)


def test_report_validation_rejects_status_private_fields_and_paths() -> None:
    tool.validate_report(_report())
    bad_status = copy.deepcopy(_report())
    bad_status["status"] = "FAILED"
    with pytest.raises(ValueError, match="did not pass"):
        tool.validate_report(bad_status)
    private = copy.deepcopy(_report())
    private["query_rows"] = []
    with pytest.raises(ValueError, match="forbidden private field"):
        tool.validate_report(private)
    path_leak = copy.deepcopy(_report())
    path_leak["provenance"] = {"path": "/private/workstation/input.json"}
    with pytest.raises(ValueError, match="absolute path"):
        tool.validate_report(path_leak)


def test_svg_escape_handles_markup_and_quotes(tmp_path: Path) -> None:
    assert tool.svg_escape('<A & "B">') == "&lt;A &amp; &quot;B&quot;&gt;"
    scene = tool.Scene("F", "T", "S", "L")
    scene.text(100, 300, '<A & "B">', 20)
    output = tmp_path / "escaped.svg"
    tool.render_svg(scene, output)
    content = output.read_text(encoding="utf-8")
    assert '<A & "B">' not in content
    assert "&lt;A &amp; &quot;B&quot;&gt;" in content


def test_contain_preserves_full_image_and_coordinate_transform() -> None:
    image = Image.new("RGB", (100, 200), "white")
    rendered, scale, offset_x, offset_y = tool._contain(image, (300, 300))
    assert rendered.size == (300, 300)
    assert scale == pytest.approx(1.5)
    assert offset_x == pytest.approx(75.0)
    assert offset_y == pytest.approx(0.0)


def test_generation_is_byte_deterministic_and_provenance_is_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root, roi_manifest, report_path = _inputs(tmp_path, monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    tool.generate(data_root, roi_manifest, report_path, first)
    tool.generate(data_root, roi_manifest, report_path, second)
    first_hashes = {relative: _sha(first / relative) for relative in tool.REQUIRED_OUTPUTS}
    second_hashes = {relative: _sha(second / relative) for relative in tool.REQUIRED_OUTPUTS}
    assert first_hashes == second_hashes
    provenance = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
    serialized = json.dumps(provenance, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "sample_id" not in serialized
    assert "identity_token" not in serialized
    assert provenance["privacy"]["aggregate_only"] is True


def test_required_output_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root, roi_manifest, report_path = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "visualization"
    tool.generate(data_root, roi_manifest, report_path, output)
    inventory = tuple(
        sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    )
    assert inventory == tuple(sorted(tool.REQUIRED_OUTPUTS))
    for relative in tool.REQUIRED_OUTPUTS:
        assert (output / relative).stat().st_size > 0


def test_overwrite_refusal_occurs_before_input_access(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tool.generate(
            tmp_path / "missing-data",
            tmp_path / "missing-roi.json",
            tmp_path / "missing-report.json",
            output,
        )
