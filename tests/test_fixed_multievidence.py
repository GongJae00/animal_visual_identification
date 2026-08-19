from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from legacy.version.afn.experiments.fixed_multievidence import (
    ALL_METHODS,
    METHODS,
    build_fixed_panel,
    build_topology_manifest,
    calibrate_and_evaluate_fusions,
    parse_publisher_frame_index,
    partition_identities,
    read_bound_rgb,
    reconstruct_f5_training_split,
    rescue_break_against_a0,
    select_fixed_panel_population,
    validate_panel_exposure,
)
from legacy.version.common.experiments.identity_topology import (
    audit_identity_topology,
    validate_identity_topology_manifest,
)
from foundation.provenance import content_sha256


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _n3_lists(*, exposed: tuple[str, ...] = ()) -> dict[str, list[str]]:
    return {
        "parent_seen_yt": [],
        "parent_seen_native_ssl_train": [],
        "ssl_train": list(exposed),
        "dev": [],
        "eval": [],
    }


def _record(identity: str, numeric_identity: int, frame: int) -> dict:
    sample = f"sample-{identity}-{frame}"
    quality = {
        "detector_confidence": 0.9,
        "model_agreement": 0.9,
        "truncation": 0.9,
        "native_resolution": 0.9,
        "multi_dog_contamination": 0.9,
        "blur_estimate": 0.9,
        "overall": 0.9,
    }
    face_quality = {
        "landmark_confidence": 0.8,
        "anchor_visibility": 0.8,
        "yaw_roll_proxy": 0.8,
        "resolution": 0.8,
        "truncation": 0.8,
        "blur_estimate": 0.8,
        "overall": 0.8,
    }
    return {
        "sample_id": sample,
        "instance_id": _sha(sample)[:32],
        "registered_identity_id": identity,
        "split_role": "test",
        "capture_group_id": f"track-{numeric_identity}",
        "capture_group_kind": "VIDEO_TRACK",
        "image_path": (
            f"YT-BB-dog/YT-BB-Dog/test/{numeric_identity}/"
            f"{numeric_identity}_{frame}.jpg"
        ),
        "image_sha256": _sha(f"source-{sample}"),
        "quality": quality,
        "face_crop_path": f"face_crops/{sample}.jpg",
        "face_crop_sha256": _sha(f"face-{sample}"),
        "face_quality": face_quality,
        "weak_nose_crop_path": f"weak_nose_crops/{sample}.jpg",
        "weak_nose_crop_sha256": _sha(f"nose-{sample}"),
    }


def _one_dev_one_eval() -> tuple[str, str]:
    dev = None
    evaluation = None
    for index in range(1, 10_000):
        identity = f"identity-{index:05d}"
        split = partition_identities([identity])
        if split[0] and dev is None:
            dev = identity
        if split[1] and evaluation is None:
            evaluation = identity
        if dev is not None and evaluation is not None:
            return dev, evaluation
    raise AssertionError("test could not find both deterministic partitions")


def _selection_records() -> tuple[list[dict], tuple[str, str]]:
    identities = _one_dev_one_eval()
    records = [
        _record(identity, numeric_identity, frame)
        for numeric_identity, identity in enumerate(identities, start=101)
        for frame in (18, 4, 7, 1, 15, 3, 11, 9, 20, 6, 13, 2)
    ]
    return records, identities


def test_selection_uses_exact_earliest_and_latest_k5_windows() -> None:
    records, identities = _selection_records()
    selected = select_fixed_panel_population(
        records,
        f5_train_identities=(),
        f5_model_selection_identities=(),
        n3_lists=_n3_lists(),
        minimum_dev_identities=1,
        minimum_eval_identities=1,
    )

    assert selected["population"]["eligible_identity_ids"] == sorted(identities)
    for identity in identities:
        rows = [
            row for row in selected["records"] if row["registered_identity_id"] == identity
        ]
        assert [
            row["publisher_frame_index"]
            for row in rows
            if row["window_role"] == "gallery"
        ] == [1, 2, 3, 4, 6]
        assert [
            row["publisher_frame_index"]
            for row in rows
            if row["window_role"] == "query"
        ] == [11, 13, 15, 18, 20]


def test_selection_fails_closed_for_exposure_shortfall_and_duplicate_frames() -> None:
    records, identities = _selection_records()
    with pytest.raises(ValueError, match="too small"):
        select_fixed_panel_population(
            records,
            f5_train_identities=identities,
            f5_model_selection_identities=(),
            n3_lists=_n3_lists(),
            minimum_dev_identities=1,
            minimum_eval_identities=1,
        )

    duplicate = [*records, dict(records[0], sample_id="different-sample")]
    with pytest.raises(ValueError, match="repeats a source frame"):
        select_fixed_panel_population(
            duplicate,
            f5_train_identities=(),
            f5_model_selection_identities=(),
            n3_lists=_n3_lists(),
            minimum_dev_identities=1,
            minimum_eval_identities=1,
        )


@pytest.mark.parametrize(
    "path",
    (
        "YT-BB-Dog/train/7/7_4.jpg",
        "YT-BB-Dog/test/7/8_4.jpg",
        "YT-BB-Dog/test/7/7_bad.jpg",
        "../YT-BB-Dog/test/7/7_4.jpg",
        "/YT-BB-Dog/test/7/7_4.jpg",
    ),
)
def test_malformed_publisher_frame_paths_fail(path: str) -> None:
    with pytest.raises(ValueError, match="publisher|safe"):
        parse_publisher_frame_index(path)


def test_deterministic_identity_hash_split_is_order_independent() -> None:
    identities = [f"identity-{index:04d}" for index in range(500)]
    first = partition_identities(identities)
    second = partition_identities(list(identities))

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert sorted([*first[0], *first[1]]) == identities
    assert len(first[0]) >= 40
    assert len(first[1]) >= 100


def test_f5_split_reconstruction_and_checkpoint_exposure_overlap() -> None:
    records = []
    for identity_index in range(10):
        identity = f"training-{identity_index}"
        for sample_index in range(4):
            records.append(
                {
                    "sample_id": f"sample-{identity_index}-{sample_index}",
                    "registered_identity_id": identity,
                    "face_crop_path": f"faces/{identity_index}-{sample_index}.jpg",
                    "face_quality": {"overall": 0.8},
                }
            )
    split = reconstruct_f5_training_split({"records": records})
    assert split == reconstruct_f5_training_split({"records": records})
    assert split["training_split_sha256"] == content_sha256(
        {key: value for key, value in split.items() if key != "training_split_sha256"}
    )

    n3_lists = _n3_lists()
    lists = {
        "f5_train": split["train_identities"],
        "f5_model_selection": split["dev_identities"],
        **{f"n3_{name}": values for name, values in sorted(n3_lists.items())},
    }
    panel = {
        "population": {"eligible_identity_ids": [split["train_identities"][0]]},
        "exposure_identity_lists": {
            "lists": lists,
            "sha256s": {
                name: content_sha256({"identity_ids": values})
                for name, values in lists.items()
            },
        },
    }
    with pytest.raises(ValueError, match="overlaps"):
        validate_panel_exposure(panel, f5_split=split, n3_lists_value=n3_lists)


def test_bound_image_rejects_path_and_hash_tampering(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert read_bound_rgb(tmp_path, "image.png", digest).size == (8, 8)
    with pytest.raises(ValueError, match="SHA-256"):
        read_bound_rgb(tmp_path, "image.png", "0" * 64)
    with pytest.raises(ValueError, match="safe"):
        read_bound_rgb(tmp_path, "../image.png", digest)


def _scores(size: int) -> dict[str, np.ndarray]:
    appearance = np.eye(size, dtype=np.float64)
    face = np.roll(appearance, 1, axis=1)
    nose = np.roll(appearance, 2, axis=1)
    return dict(zip(METHODS, (appearance, face, nose), strict=True))


def test_fusion_selection_is_dev_only_and_deterministic() -> None:
    dev_ids = [f"dev-{index:03d}" for index in range(8)]
    eval_ids = [f"eval-{index:03d}" for index in range(12)]
    dev = _scores(len(dev_ids))
    first = calibrate_and_evaluate_fusions(
        dev_ids, eval_ids, dev, _scores(len(eval_ids)), resolution=10
    )
    changed_eval = _scores(len(eval_ids))
    changed_eval[METHODS[0]] = np.roll(np.eye(len(eval_ids)), 3, axis=1)
    second = calibrate_and_evaluate_fusions(
        dev_ids, eval_ids, dev, changed_eval, resolution=10
    )

    assert first[0] == second[0]
    assert first[0]["labels_used"] == "DEV_ONLY"
    assert first[0]["evaluation_labels_used_for_weight_selection"] is False
    assert first[0]["fusions"]["A0_plus_F5"]["selected_weights"] == {
        METHODS[0]: 1.0,
        METHODS[1]: 0.0,
    }
    assert first[1]["A0_plus_F5"] != second[1]["A0_plus_F5"]


def test_rescue_break_is_exactly_paired_against_a0() -> None:
    identities = ["a", "b", "c"]

    def rows(ranks: list[int]) -> list[dict]:
        return [
            {
                "registered_identity_id": identity,
                "rank": rank,
                "Rank-1": float(rank == 1),
                "Rank-5": float(rank <= 5),
                "MRR": 1.0 / rank,
            }
            for identity, rank in zip(identities, ranks, strict=True)
        ]

    outcomes = {method: rows([2, 1, 1]) for method in ALL_METHODS}
    outcomes[METHODS[1]] = rows([1, 2, 1])
    result = rescue_break_against_a0(outcomes)

    assert result[METHODS[1]]["rescue_count"] == 1
    assert result[METHODS[1]]["break_count"] == 1


def test_topology_manifest_is_compatible_and_same_track_only() -> None:
    records = []
    for identity_index, identity in enumerate(("identity-a", "identity-b")):
        for frame in range(2):
            row = _record(identity, identity_index + 1, frame)
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "registered_identity_id": identity,
                    "capture_group_id": row["capture_group_id"],
                    "source": {"quality": row["quality"]},
                    "face": {"quality": row["face_quality"]},
                }
            )
    vectors = np.asarray(
        ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]),
        dtype=np.float32,
    )
    manifest = build_topology_manifest(
        records,
        {method: vectors.copy() for method in METHODS},
        input_bindings={
            "panel_bundle_content_sha256": "0" * 64,
            "panel_sha256": "1" * 64,
            "frozen_dinov2_sha256": "2" * 64,
            "f5_checkpoint_sha256": "3" * 64,
            "n3_lineage_content_sha256": "4" * 64,
            "n3_runtime_manifest_content_sha256": "5" * 64,
            "n3_runtime_manifest_raw_sha256": "6" * 64,
            "n3_onnx_sha256": "7" * 64,
            "execution": {"device": "cpu", "n3_device": "cpu", "batch_size": 2},
        },
    )

    validate_identity_topology_manifest(manifest)
    report = audit_identity_topology(manifest)
    assert all(
        branch["aggregate"]["same_track_only"]
        for branch in report["branches"].values()
    )
    assert all(
        row["session_token"].startswith("track-") for row in manifest["records"]
    )


def test_panel_builder_refuses_overwrite_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "panel.json"
    output.write_text("occupied", encoding="utf-8")
    missing = tmp_path / "missing"
    with pytest.raises(FileExistsError, match="overwrite"):
        build_fixed_panel(
            roi_manifest_path=missing,
            roi_manifest_sha256="0" * 64,
            source_image_root=missing,
            f5_checkpoint_path=missing,
            f5_checkpoint_sha256="0" * 64,
            f5_training_roi_manifest_path=missing,
            f5_training_roi_manifest_sha256="0" * 64,
            n3_lineage_path=missing,
            n3_lineage_sha256="0" * 64,
            output_path=output,
        )


@pytest.mark.parametrize(
    "script, expected",
    (
        ("build_fixed_multievidence_panel.py", "--f5-training-roi-manifest-sha256"),
        ("evaluate_fixed_multievidence.py", "--topology-output"),
    ),
)
def test_cli_help(script: str, expected: str) -> None:
    tool = Path(__file__).resolve().parents[1] / "legacy/version/afn/workflows" / script
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert expected in completed.stdout
