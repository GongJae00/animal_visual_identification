from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from archive.appearance_face_nose.experiments.unified_multievidence import (
    _CODE_PATHS,
    _LEGACY_CODE_PATHS,
    _PRE_EMBEDDING_CODE_PATHS,
    _PRE_NESTED_EMBEDDING_CODE_PATHS,
    _PRE_TRAINING_OWNERSHIP_CODE_PATHS,
    INTERPRETATION,
    METHODS,
    REPORT_BUNDLE_SCHEMA,
    REPORT_SCHEMA,
    _group_k5_population,
    _identity_bound_face_records,
    _identity_list_sha256,
    _metrics,
    _paired_bootstrap,
    calibrate_and_evaluate_score_fusion,
    validate_report_bundle,
)

from tests.repo_root import REPO_ROOT
from shared.foundation.provenance import content_sha256

BRANCHES = tuple(list(METHODS)[:3])

def _score_matrices(size: int) -> dict[str, np.ndarray]:
    appearance = np.eye(size, dtype=np.float64)
    face = np.roll(appearance, 1, axis=1)
    nose = np.roll(appearance, 2, axis=1)
    return dict(zip(BRANCHES, (appearance, face, nose), strict=True))

def test_dev_only_simplex_is_deterministic_and_applied_to_eval() -> None:
    dev_ids = [f"dev-{index:03d}" for index in range(12)]
    eval_ids = [f"eval-{index:03d}" for index in range(20)]

    first = calibrate_and_evaluate_score_fusion(
        dev_ids,
        eval_ids,
        _score_matrices(len(dev_ids)),
        _score_matrices(len(eval_ids)),
        resolution=10,
    )
    second = calibrate_and_evaluate_score_fusion(
        dev_ids,
        eval_ids,
        _score_matrices(len(dev_ids)),
        _score_matrices(len(eval_ids)),
        resolution=10,
    )

    assert second[0] == first[0]
    assert second[1] == first[1]
    assert first[0]["labels_used"] == "DEV_ONLY"
    assert first[0]["fusions"]["A0_plus_F0"]["selected_weights"] == {
        BRANCHES[0]: 1.0,
        BRANCHES[1]: 0.0,
    }
    assert first[0]["fusions"]["A0_plus_F0_plus_N3"]["selected_weights"] == {
        BRANCHES[0]: 1.0,
        BRANCHES[1]: 0.0,
        BRANCHES[2]: 0.0,
    }
    assert first[0]["fusions"]["A0_plus_N3"]["selected_weights"] == {
        BRANCHES[0]: 1.0,
        BRANCHES[2]: 0.0,
    }
    assert len(first[1]["A0_plus_F0_plus_N3"]) == len(eval_ids)
    assert all(row["Rank-1"] == 1.0 for row in first[1]["A0_plus_F0_plus_N3"])

def test_dev_eval_overlap_and_degenerate_scores_fail_closed() -> None:
    ids = [f"dog-{index:03d}" for index in range(10)]
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_and_evaluate_score_fusion(
            ids,
            ids,
            _score_matrices(len(ids)),
            _score_matrices(len(ids)),
        )

    degenerate = _score_matrices(len(ids))
    degenerate[BRANCHES[1]] = np.ones((len(ids), len(ids)))
    with pytest.raises(ValueError, match="near-zero"):
        calibrate_and_evaluate_score_fusion(
            ids,
            [f"eval-{index:03d}" for index in range(10)],
            degenerate,
            _score_matrices(len(ids)),
        )

def test_face_population_requires_identity_bound_source_match() -> None:
    populations = []
    roi_records = []
    for identity_index, identity in enumerate(("dog-a", "dog-b")):
        rows = []
        for frame in range(10):
            source = f"YT-BB-Dog/train/{identity_index}/{identity_index}_{frame}.jpg"
            digest = f"{identity_index}{frame:063d}"[-64:]
            rows.append(
                {
                    "sample_token": f"sample-{identity_index}-{frame}",
                    "source_archive_member": source,
                    "source_sha256": digest,
                }
            )
            roi_records.append(
                {
                    "image_path": f"YT-BB-dog/{source}",
                    "image_sha256": digest,
                    "face_crop_path": f"face/{identity_index}-{frame}.jpg",
                    "registered_identity_id": identity if identity_index == 0 else None,
                }
            )
        populations.append(
            {
                "registered_dog_id": identity,
                "gallery": rows[:5],
                "query": rows[5:],
            }
        )

    selected, exclusions, permissive_count = _identity_bound_face_records(
        populations, roi_records
    )

    assert [item["registered_dog_id"] for item in selected] == ["dog-a"]
    assert exclusions == {
        "missing_any_face_crop": 0,
        "missing_identity_bound_face_crop": 1,
        "repeated_identity_bound_face_crop": 0,
    }
    assert permissive_count == 2

def test_k5_population_rejects_cross_identity_token_and_repeated_frame() -> None:
    def records() -> list[dict[str, object]]:
        return [
            {
                "record_state": "AVAILABLE",
                "registered_dog_id": identity,
                "identity_token": f"identity-{identity}",
                "track_token": f"track-{identity}",
                "sequence_token": f"sequence-{identity}",
                "frame_index": frame,
                "sample_token": f"sample-{identity}-{frame}",
            }
            for identity in ("dog-a", "dog-b")
            for frame in range(10)
        ]

    cross_owned = records()
    for row in cross_owned:
        if row["registered_dog_id"] == "dog-b":
            row["track_token"] = "track-dog-a"
    with pytest.raises(ValueError, match="multiple registered identities"):
        _group_k5_population(cross_owned, ["dog-a", "dog-b"])

    repeated_frame = records()
    repeated_frame[9]["frame_index"] = 8
    with pytest.raises(ValueError, match="repeats a frame index"):
        _group_k5_population(repeated_frame, ["dog-a", "dog-b"])

def _valid_report_bundle() -> dict[str, object]:
    dev_ids = [f"dev-{index:03d}" for index in range(12)]
    eval_ids = [f"eval-{index:03d}" for index in range(20)]
    calibration, outcomes = calibrate_and_evaluate_score_fusion(
        dev_ids,
        eval_ids,
        _score_matrices(len(dev_ids)),
        _score_matrices(len(eval_ids)),
        resolution=10,
    )
    per_identity = [
        {
            "registered_dog_id": identity,
            "method_outcomes": {
                method: outcomes[method][index] for method in METHODS
            },
        }
        for index, identity in enumerate(eval_ids)
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_SAME_TRACK_UNIFIED_RESEARCH_DIAGNOSTIC",
        "interpretation": INTERPRETATION,
        "methods": METHODS,
        "input_bindings": {
            "native_bundle": {"content_sha256": "0" * 64},
            "native_root": "/native",
            "source_image_root": "/source",
            "roi_manifest": {"content_sha256": "1" * 64},
            "nose_lineage": {"content_sha256": "2" * 64},
            "nose_runtime_manifest": {"content_sha256": "3" * 64},
            "nose_onnx": {"sha256": "4" * 64},
            "frozen_dinov2": {"model_sha256": "5" * 64},
            "dev_pairing_sha256": "8" * 64,
            "eval_pairing_sha256": "9" * 64,
        },
        "input_sha256s": {
            "native_bundle_content": "0" * 64,
            "roi_manifest_content": "1" * 64,
            "nose_lineage_content": "2" * 64,
            "nose_runtime_manifest_content": "3" * 64,
            "nose_onnx": "4" * 64,
            "frozen_dinov2": "5" * 64,
            "dev_registered_dog_ids": _identity_list_sha256(dev_ids),
            "eval_registered_dog_ids": _identity_list_sha256(eval_ids),
            "dev_pairing": "8" * 64,
            "eval_pairing": "9" * 64,
        },
        "code_sha256s": {path: "a" * 64 for path in _CODE_PATHS},
        "protocol": {
            "population_source": "fixture",
            "face_admission": "EXACTLY_ONE_FACE_CROP_BOUND_TO_REGISTERED_DOG_ID_AND_SOURCE_SHA256",
            "localized_states": ["AVAILABLE", "LOW_QUALITY"],
            "temporal_order": ["frame_index", "sample_token"],
            "gallery_selection": "earliest_five",
            "query_selection": "latest_five",
            "frames_per_role": 5,
            "frame_overlap_allowed": False,
            "fixed_common_population_across_methods": True,
            "temporal_aggregation": "L2_NORMALIZED_UNWEIGHTED_K5_MEAN",
            "retrieval": "EXHAUSTIVE_COSINE_ONE_GALLERY_VECTOR_PER_IDENTITY",
            "fusion_labels": "DEV_ONLY",
            "evaluation_labels_used_for_weight_selection": False,
            "bootstrap": {
                "cluster_unit": "registered_dog_id",
                "resamples": 10,
                "seed": 7,
                "confidence_level": 0.95,
            },
            "limitations": ["TEST_FIXTURE"],
        },
        "population": {
            "parent_dev_identity_count": len(dev_ids),
            "parent_eval_identity_count": len(eval_ids),
            "permissive_dev_identity_count": len(dev_ids),
            "permissive_eval_identity_count": len(eval_ids),
            "selected_dev_identity_count": len(dev_ids),
            "selected_eval_identity_count": len(eval_ids),
            "selected_dev_registered_dog_ids": dev_ids,
            "selected_eval_registered_dog_ids": eval_ids,
            "selected_dev_registered_dog_ids_sha256": _identity_list_sha256(dev_ids),
            "selected_eval_registered_dog_ids_sha256": _identity_list_sha256(eval_ids),
            "dev_excluded_identity_counts": {
                "missing_any_face_crop": 0,
                "missing_identity_bound_face_crop": 0,
                "repeated_identity_bound_face_crop": 0,
            },
            "eval_excluded_identity_counts": {
                "missing_any_face_crop": 0,
                "missing_identity_bound_face_crop": 0,
                "repeated_identity_bound_face_crop": 0,
            },
        },
        "calibration": calibration,
        "evaluation": {
            "metrics": {
                method: _metrics(outcomes[method], len(eval_ids)) for method in METHODS
            },
            "per_identity": per_identity,
        },
        "paired_delta_bootstrap_cis": _paired_bootstrap(
            per_identity,
            resamples=10,
            seed=7,
            confidence_level=0.95,
        ),
    }
    return {
        "schema_version": REPORT_BUNDLE_SCHEMA,
        "report_sha256": content_sha256(report),
        "report": report,
    }

def test_report_bundle_detects_digest_and_rehashed_metric_tamper() -> None:
    bundle = _valid_report_bundle()
    assert validate_report_bundle(bundle)["schema_version"] == REPORT_SCHEMA

    digest_tamper = copy.deepcopy(bundle)
    digest_tamper["report"]["status"] = "TAMPERED"
    with pytest.raises(ValueError, match="digest differs"):
        validate_report_bundle(digest_tamper)

    metric_tamper = copy.deepcopy(bundle)
    metric_tamper["report"]["evaluation"]["metrics"][BRANCHES[0]]["Rank-1"] = 0.0
    metric_tamper["report_sha256"] = content_sha256(metric_tamper["report"])
    with pytest.raises(ValueError, match="metrics differ"):
        validate_report_bundle(metric_tamper)

def test_report_bundle_reads_only_complete_legacy_code_path_set() -> None:
    pre_training_ownership = _valid_report_bundle()
    pre_training_ownership["report"]["code_sha256s"] = {
        path: "a" * 64 for path in _PRE_TRAINING_OWNERSHIP_CODE_PATHS
    }
    pre_training_ownership["report_sha256"] = content_sha256(
        pre_training_ownership["report"]
    )
    assert validate_report_bundle(pre_training_ownership)["code_sha256s"] == (
        pre_training_ownership["report"]["code_sha256s"]
    )

    pre_nested = _valid_report_bundle()
    pre_nested["report"]["code_sha256s"] = {
        path: "a" * 64 for path in _PRE_NESTED_EMBEDDING_CODE_PATHS
    }
    pre_nested["report_sha256"] = content_sha256(pre_nested["report"])
    assert validate_report_bundle(pre_nested)["code_sha256s"] == pre_nested[
        "report"
    ]["code_sha256s"]

    pre_embedding = _valid_report_bundle()
    pre_embedding["report"]["code_sha256s"] = {
        path: "a" * 64 for path in _PRE_EMBEDDING_CODE_PATHS
    }
    pre_embedding["report_sha256"] = content_sha256(pre_embedding["report"])
    assert validate_report_bundle(pre_embedding)["code_sha256s"] == (
        pre_embedding["report"]["code_sha256s"]
    )

    legacy = _valid_report_bundle()
    legacy["report"]["code_sha256s"] = {
        path: "a" * 64 for path in _LEGACY_CODE_PATHS
    }
    legacy["report_sha256"] = content_sha256(legacy["report"])
    assert validate_report_bundle(legacy)["code_sha256s"] == legacy["report"][
        "code_sha256s"
    ]

    mixed = copy.deepcopy(legacy)
    mixed["report"]["code_sha256s"].pop("localization/roi_manifest.py")
    mixed["report"]["code_sha256s"]["parsing/export/regions/roi_manifest.py"] = "a" * 64
    mixed["report_sha256"] = content_sha256(mixed["report"])
    with pytest.raises(ValueError, match="code hash schema"):
        validate_report_bundle(mixed)

def test_cli_help() -> None:
    tool = REPO_ROOT / "archive/appearance_face_nose/commands" / "evaluate_yt_unified_multievidence.py"
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--roi-manifest-sha256" in completed.stdout
    assert "--nose-lineage-sha256" in completed.stdout
    assert "--fusion-resolution" in completed.stdout

def test_report_json_round_trip_is_finite() -> None:
    bundle = _valid_report_bundle()
    assert json.loads(json.dumps(bundle, allow_nan=False)) == bundle
