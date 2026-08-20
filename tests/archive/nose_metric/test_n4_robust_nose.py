from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from archive.appearance_face_nose.experiments.fixed_multievidence import (
    DEV_FRACTION,
    FRAMES_PER_WINDOW,
    METHODS,
    MINIMUM_DEV_IDENTITIES,
    MINIMUM_EVAL_IDENTITIES,
    PANEL_BUNDLE_SCHEMA_VERSION,
    PANEL_SCHEMA_VERSION,
    _LEGACY_PANEL_CODE_PATHS,
    _PRE_TRAINING_OWNERSHIP_PANEL_CODE_PATHS,
    SPLIT_COMMITMENT,
    partition_identities,
    validate_fixed_topology_bindings,
    validate_panel_bundle,
)

from tests.repo_root import REPO_ROOT
from archive.shared_helpers.experiments.identity_topology import FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION
from archive.nose_metric.experiments.n4_robust_nose import (
    CANDIDATES,
    N3_BRANCH,
    REPORT_BUNDLE_SCHEMA_VERSION,
    build_n4_report,
    consensus_trimmed_mean,
    evaluate_n4_robust_nose,
    medoid,
    normalized_mean,
    quality_weighted_mean,
    rank_score_rows,
    select_dev_candidate,
    two_prototype_farthest_first,
    validate_report_bundle,
)
from shared.foundation.provenance import content_sha256

_PANEL_LIMITATIONS = [
    "SAME_VIDEO_TRACK_GALLERY_AND_QUERY",
    "PUBLISHER_TEST_EXPOSED_DIAGNOSTIC",
    "PRIOR_PUBLISHER_TEST_EXPOSURE",
    "CLOSED_SET_ONLY_NO_UNKNOWN_REJECTION",
    "TRACK_IDENTITIES_NOT_LIFELONG_DOG_IDENTITIES",
    "WEAK_NOSE_ROI_INPUT_DIFFERS_FROM_NATIVE_NOSE_TRAINING_INPUT",
    "NOT_FINAL_EVALUATION",
    "NO_BIOMETRIC_OR_OPEN_SET_CLAIM",
]
_PANEL_CODE_PATHS = (
    "archive/appearance_face_nose/experiments/fixed_multievidence.py",
    "identification/export/face/checkpoint.py",
    "parsing/export/regions/roi_manifest.py",
    "identification/training/nose/embedding_consistency_training.py",
    "archive/face/commands/train_roi_face_reid.py",
    "archive/appearance_face_nose/commands/build_fixed_multievidence_panel.py",
)

def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()

def _identity_population() -> tuple[list[str], list[str], list[str]]:
    identities: list[str] = []
    for index in range(10_000):
        identities.append(
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"panel-identity-{index:05d}"))
        )
        ordered = sorted(identities)
        dev, evaluation = partition_identities(ordered)
        if (
            len(dev) >= MINIMUM_DEV_IDENTITIES
            and len(evaluation) >= MINIMUM_EVAL_IDENTITIES
        ):
            return ordered, dev, evaluation
    raise AssertionError("could not construct the fixed minimum partition")

def _synthetic_inputs() -> tuple[dict, dict]:
    identities, dev_ids, eval_ids = _identity_population()
    partition = {
        **{identity: "DEV" for identity in dev_ids},
        **{identity: "EVAL" for identity in eval_ids},
    }
    records = []
    topology_rows = []
    vector_by_sample: dict[str, list[float]] = {}
    quality_by_sample: dict[str, float] = {}
    for identity_index, identity in enumerate(identities, start=1):
        base_angle = 2.0 * np.pi * identity_index / len(identities)
        for role, frames in (("gallery", range(5)), ("query", range(10, 15))):
            for offset, frame in enumerate(frames):
                sample = f"sample-{identity}-{frame:02d}"
                quality_value = 0.55 + 0.1 * offset
                quality = {"overall": quality_value}
                face_quality = {"overall": quality_value - 0.05}
                angle = base_angle + (offset - 2) * 0.08
                vector_by_sample[sample] = [float(np.cos(angle)), float(np.sin(angle))]
                quality_by_sample[sample] = quality_value
                records.append(
                    {
                        "sample_id": sample,
                        "instance_id": _sha(f"instance-{sample}")[:32],
                        "registered_identity_id": identity,
                        "partition": partition[identity],
                        "window_role": role,
                        "publisher_frame_index": frame,
                        "split_role": "test",
                        "capture_group_id": f"track-{identity_index:05d}",
                        "capture_group_kind": "VIDEO_TRACK",
                        "source": {
                            "path": f"YT-BB-Dog/test/{identity_index}/{identity_index}_{frame}.jpg",
                            "sha256": _sha(f"source-{sample}"),
                            "quality": quality,
                        },
                        "face": {
                            "path": f"face_crops/{sample}.jpg",
                            "sha256": _sha(f"face-{sample}"),
                            "quality": face_quality,
                        },
                        "weak_nose": {
                            "path": f"weak_nose_crops/{sample}.jpg",
                            "sha256": _sha(f"nose-{sample}"),
                            "quality": quality,
                            "quality_semantics": "DOG_ROI_QUALITY_PROXY_NO_NOSE_SPECIFIC_SCORE",
                        },
                    }
                )
    panel = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": "FROZEN_EXPOSED_PUBLISHER_TEST_DIAGNOSTIC_PANEL",
        "interpretation": "SYNTHETIC_CONTRACT_FIXTURE",
        "protocol": {
            "population_source": "PUBLISHER_TEST_ROI_MANIFEST_ONLY",
            "selection_uses_model_outputs": False,
            "split_commitment": SPLIT_COMMITMENT,
            "dev_fraction": DEV_FRACTION,
            "gallery_selection": "EARLIEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
            "query_selection": "LATEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
            "frames_per_window": FRAMES_PER_WINDOW,
            "same_track_only": True,
            "limitations": _PANEL_LIMITATIONS,
        },
        "input_bindings": {
            "roi_manifest_bundle": {
                "path": "/tmp/roi.json", "raw_sha256": "7" * 64,
                "content_sha256": "8" * 64, "byte_size": 1,
            },
            "roi_manifest_sha256": "9" * 64,
            "source_image_root": "/tmp/source",
            "f5_checkpoint": {
                "path": "/tmp/f5.pt", "sha256": "1" * 64,
                "training_split_sha256": "a" * 64,
            },
            "f5_training_roi_manifest_bundle": {
                "path": "/tmp/train-roi.json", "raw_sha256": "b" * 64,
                "content_sha256": "c" * 64, "byte_size": 1,
            },
            "f5_training_roi_manifest_sha256": "d" * 64,
            "n3_lineage": {
                "path": "/tmp/lineage.json", "raw_sha256": "e" * 64,
                "content_sha256": "2" * 64, "byte_size": 1,
                "lineage_sha256": "f" * 64,
            },
        },
        "exposure_identity_lists": {
            "lists": {
                name: []
                for name in (
                    "f5_train", "f5_model_selection", "n3_parent_seen_yt",
                    "n3_parent_seen_native_ssl_train", "n3_ssl_train",
                    "n3_dev", "n3_eval",
                )
            },
            "sha256s": {
                name: content_sha256({"identity_ids": []})
                for name in (
                    "f5_train", "f5_model_selection", "n3_parent_seen_yt",
                    "n3_parent_seen_native_ssl_train", "n3_ssl_train",
                    "n3_dev", "n3_eval",
                )
            },
        },
        "population": {
            "observed_publisher_test_identity_ids": identities,
            "eligible_identity_ids": identities,
            "dev_identity_ids": dev_ids,
            "eval_identity_ids": eval_ids,
            "dev_identity_ids_sha256": content_sha256({"identity_ids": dev_ids}),
            "eval_identity_ids_sha256": content_sha256({"identity_ids": eval_ids}),
            "minimum_dev_identities": MINIMUM_DEV_IDENTITIES,
            "minimum_eval_identities": MINIMUM_EVAL_IDENTITIES,
        },
        "exclusions": {},
        "records": records,
        "code_sha256s": {path: "0" * 64 for path in _PANEL_CODE_PATHS},
    }
    bundle = {
        "schema_version": PANEL_BUNDLE_SCHEMA_VERSION,
        "panel_sha256": content_sha256(panel),
        "panel": panel,
    }
    panel_by_sample = {record["sample_id"]: record for record in records}
    for branch in reversed(METHODS):
        for sample in reversed(list(panel_by_sample)):
            record = panel_by_sample[sample]
            topology_rows.append(
                {
                    "sample_token": sample,
                    "identity_token": record["registered_identity_id"],
                    "session_token": record["capture_group_id"],
                    "branch": branch,
                    "quality": (
                        record["face"]["quality"]["overall"]
                        if branch == METHODS[1]
                        else quality_by_sample[sample]
                    ),
                    "available": True,
                    "embedding": vector_by_sample[sample],
                }
            )
    topology_bindings = {
        "panel_bundle_content_sha256": content_sha256(bundle),
        "panel_sha256": bundle["panel_sha256"],
        "frozen_dinov2_sha256": "3" * 64,
        "f5_checkpoint_sha256": "1" * 64,
        "n3_lineage_content_sha256": "2" * 64,
        "n3_runtime_manifest_content_sha256": "4" * 64,
        "n3_runtime_manifest_raw_sha256": "5" * 64,
        "n3_onnx_sha256": "6" * 64,
        "execution": {"device": "cpu", "n3_device": "cpu", "batch_size": 2},
    }
    topology = {
        "schema_version": FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        "input_bindings": topology_bindings,
        "input_bindings_sha256": content_sha256(topology_bindings),
        "records": topology_rows,
    }
    return bundle, topology

@pytest.fixture(scope="module")
def fixed_inputs() -> tuple[dict, dict]:
    return _synthetic_inputs()

def test_panel_and_topology_bindings_fail_closed(fixed_inputs: tuple[dict, dict]) -> None:
    panel, topology = fixed_inputs
    missing_exposure = copy.deepcopy(panel)
    del missing_exposure["panel"]["exposure_identity_lists"]["lists"]["n3_eval"]
    del missing_exposure["panel"]["exposure_identity_lists"]["sha256s"]["n3_eval"]
    missing_exposure["panel_sha256"] = content_sha256(missing_exposure["panel"])
    with pytest.raises(ValueError, match="exposure identity-list hashes"):
        validate_panel_bundle(missing_exposure)

    mismatched_topology = copy.deepcopy(topology)
    mismatched_topology["input_bindings"]["n3_onnx_sha256"] = "f" * 64
    mismatched_topology["input_bindings_sha256"] = content_sha256(
        mismatched_topology["input_bindings"]
    )
    with pytest.raises(ValueError, match="N3 ONNX"):
        validate_fixed_topology_bindings(
            panel,
            mismatched_topology,
            n3_onnx_sha256="6" * 64,
        )

def test_panel_reader_accepts_only_complete_legacy_code_paths(
    fixed_inputs: tuple[dict, dict],
) -> None:
    current, _ = fixed_inputs
    assert validate_panel_bundle(current) == current["panel"]

    pre_training_ownership = copy.deepcopy(current)
    pre_training_ownership["panel"]["code_sha256s"] = {
        historical: current["panel"]["code_sha256s"][present]
        for historical, present in zip(
            _PRE_TRAINING_OWNERSHIP_PANEL_CODE_PATHS,
            current["panel"]["code_sha256s"],
            strict=True,
        )
    }
    pre_training_ownership["panel_sha256"] = content_sha256(
        pre_training_ownership["panel"]
    )
    assert validate_panel_bundle(pre_training_ownership) == pre_training_ownership[
        "panel"
    ]

    legacy = copy.deepcopy(current)
    legacy_hashes = {
        historical: current["panel"]["code_sha256s"][present]
        for historical, present in zip(
            _LEGACY_PANEL_CODE_PATHS,
            current["panel"]["code_sha256s"],
            strict=True,
        )
    }
    legacy["panel"]["code_sha256s"] = legacy_hashes
    legacy["panel_sha256"] = content_sha256(legacy["panel"])
    assert validate_panel_bundle(legacy) == legacy["panel"]

    mixed = copy.deepcopy(legacy)
    mixed["panel"]["code_sha256s"].pop("localization/roi_manifest.py")
    mixed["panel"]["code_sha256s"]["parsing/export/regions/roi_manifest.py"] = "0" * 64
    mixed["panel_sha256"] = content_sha256(mixed["panel"])
    with pytest.raises(ValueError, match="code hashes"):
        validate_panel_bundle(mixed)

def _rows(identities: list[str], ranks: list[int]) -> list[dict]:
    return [
        {
            "registered_identity_id": identity,
            "rank": rank,
            "Rank-1": float(rank == 1),
            "MRR": 1.0 / rank,
            "Rank-5": float(rank <= 5),
        }
        for identity, rank in zip(identities, ranks, strict=True)
    ]

def test_candidate_algorithms_are_normalized_and_use_bound_quality() -> None:
    vectors = [[2.0, 0.0], [0.0, 3.0]]
    mean = normalized_mean(vectors)
    weighted = quality_weighted_mean(vectors, [1.0, 0.0], 2)

    assert np.linalg.norm(mean) == pytest.approx(1.0)
    assert weighted == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError, match="zero total"):
        quality_weighted_mean(vectors, [0.0, 0.0], 1)
    with pytest.raises(ValueError, match="exponent"):
        quality_weighted_mean(vectors, [1.0, 1.0], 3)

def test_medoid_and_consensus_ties_use_lexical_sample_tokens() -> None:
    tied = medoid([[1.0, 0.0], [0.0, 1.0]], ["z-sample", "a-sample"])
    assert tied == pytest.approx([0.0, 1.0])

    trimmed = consensus_trimmed_mean(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ["e", "d", "c", "b", "a"],
        2,
    )
    assert trimmed == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError, match="retain"):
        consensus_trimmed_mean([[1.0, 0.0]] * 5, list("abcde"), 5)

def test_two_prototype_farthest_first_is_order_invariant_by_token() -> None:
    vectors = [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1], [0.0, 1.0]]
    tokens = ["a", "b", "c", "d", "e"]
    first = two_prototype_farthest_first(vectors, tokens)
    order = [4, 2, 0, 3, 1]
    second = two_prototype_farthest_first(
        [vectors[index] for index in order], [tokens[index] for index in order]
    )

    assert first == pytest.approx(second)
    assert np.linalg.norm(first, axis=1) == pytest.approx([1.0, 1.0])

def test_rank_ties_are_broken_by_lexical_gallery_identity() -> None:
    rows = rank_score_rows(np.zeros((3, 3)), ["a", "b", "c"])
    assert [row["rank"] for row in rows] == [1, 2, 3]

def test_dev_selection_objective_and_complexity_tie_break() -> None:
    identities = ["a", "b", "c", "d", "e", "f"]
    tied = {
        candidate.name: _rows(identities, [1, 1, 1, 1, 1, 1])
        for candidate in CANDIDATES
    }
    assert select_dev_candidate(tied)["selected_candidate"] == "normalized_mean"

    better_mrr = copy.deepcopy(tied)
    for candidate in CANDIDATES:
        better_mrr[candidate.name] = _rows(identities, [1, 1, 2, 2, 6, 6])
    better_mrr["medoid"] = _rows(identities, [1, 1, 2, 2, 2, 2])
    selection = select_dev_candidate(better_mrr)
    assert selection["selected_candidate"] == "medoid"
    assert selection["labels_used"] == "DEV_ONLY"
    assert selection["evaluation_labels_used_for_selection"] is False

def test_exact_contract_report_is_deterministic_and_dev_selected(
    fixed_inputs: tuple[dict, dict],
) -> None:
    panel, topology = fixed_inputs
    first = build_n4_report(panel, topology, bootstrap_resamples=20, bootstrap_seed=7)
    second = build_n4_report(panel, topology, bootstrap_resamples=20, bootstrap_seed=7)

    assert first == second
    assert set(first["development"]["all_candidate_metrics"]) == {
        candidate.name for candidate in CANDIDATES
    }
    selected = first["development"]["selection"]["selected_candidate"]
    evaluated = set(first["evaluation"]["identity_bootstrap_cis"])
    assert evaluated == {"normalized_mean", selected}
    assert len(first["evaluation"]["per_identity_ranks"]) >= MINIMUM_EVAL_IDENTITIES
    assert first["protocol"]["physical_nose_topology_claim"] is False
    assert first["protocol"]["same_track_only"] is True
    assert first["protocol"]["open_set"] is False
    json.dumps(first, allow_nan=False, sort_keys=True)

@pytest.mark.parametrize("tamper", ("identity", "quality", "normalization", "coverage"))
def test_topology_join_and_vector_tampering_fail_closed(
    fixed_inputs: tuple[dict, dict], tamper: str
) -> None:
    panel, source_topology = fixed_inputs
    topology = copy.deepcopy(source_topology)
    n3_index = next(
        index
        for index, row in enumerate(topology["records"])
        if row["branch"] == N3_BRANCH
    )
    if tamper == "identity":
        topology["records"][n3_index]["identity_token"] = "wrong-identity"
    elif tamper == "quality":
        topology["records"][n3_index]["quality"] = 0.01
    elif tamper == "normalization":
        topology["records"][n3_index]["embedding"][0] *= 2.0
    else:
        del topology["records"][n3_index]

    with pytest.raises(ValueError):
        build_n4_report(panel, topology, bootstrap_resamples=2)

def test_panel_window_and_partition_tampering_fail_closed(
    fixed_inputs: tuple[dict, dict],
) -> None:
    source_panel, topology = fixed_inputs
    panel = copy.deepcopy(source_panel)
    panel["panel"]["records"].pop()
    panel["panel_sha256"] = content_sha256(panel["panel"])
    with pytest.raises(ValueError, match="record count|windows"):
        build_n4_report(panel, topology, bootstrap_resamples=2)

    panel = copy.deepcopy(source_panel)
    panel["panel"]["records"][0]["partition"] = "EVAL"
    panel["panel_sha256"] = content_sha256(panel["panel"])
    with pytest.raises(ValueError, match="partition"):
        build_n4_report(panel, topology, bootstrap_resamples=2)

def test_external_pins_canonical_output_and_no_overwrite(
    fixed_inputs: tuple[dict, dict], tmp_path: Path
) -> None:
    panel, topology = fixed_inputs
    panel_path = tmp_path / "panel.json"
    topology_path = tmp_path / "topology.json"
    output = tmp_path / "n4.json"
    panel_path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")
    topology_path.write_text(json.dumps(topology, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="external pin"):
        evaluate_n4_robust_nose(
            panel_path=panel_path,
            panel_sha256="0" * 64,
            topology_manifest_path=topology_path,
            topology_sha256=content_sha256(topology),
            n3_runtime_manifest_sha256="4" * 64,
            n3_onnx_sha256="6" * 64,
            output_path=output,
            bootstrap_resamples=2,
        )

    bundle = evaluate_n4_robust_nose(
        panel_path=panel_path,
        panel_sha256=content_sha256(panel),
        topology_manifest_path=topology_path,
        topology_sha256=content_sha256(topology),
        n3_runtime_manifest_sha256="4" * 64,
        n3_onnx_sha256="6" * 64,
        output_path=output,
        bootstrap_resamples=10,
    )
    raw = output.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert validate_report_bundle(json.loads(raw)) == bundle["report"]
    assert bundle["schema_version"] == REPORT_BUNDLE_SCHEMA_VERSION

    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_n4_robust_nose(
            panel_path=tmp_path / "missing-panel",
            panel_sha256="0" * 64,
            topology_manifest_path=tmp_path / "missing-topology",
            topology_sha256="0" * 64,
            n3_runtime_manifest_sha256="0" * 64,
            n3_onnx_sha256="0" * 64,
            output_path=output,
        )

def test_cli_help_exposes_both_required_canonical_pins() -> None:
    tool = (
        REPO_ROOT / "archive/nose_metric/commands" / "evaluate_n4_robust_nose.py"
    )
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--panel-sha256" in completed.stdout
    assert "--topology-sha256" in completed.stdout
