from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
    InstanceSegmentationModelManifest,
    instance_segmentation_model_bundle,
)
from artifact_contracts.model_file_binding import ModelFileBinding
from localization.animal_instance_segmentation import (
    AnimalInstanceCandidate,
    _all_target_candidates,
    _suppress_duplicate_candidates,
)


def _binding(root: Path, name: str, payload: bytes) -> ModelFileBinding:
    path = root / name
    path.write_bytes(payload)
    return ModelFileBinding(name, len(payload), hashlib.sha256(payload).hexdigest())


def test_instance_model_artifact_revalidates_bound_files(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    manifest = InstanceSegmentationModelManifest(
        model_id="fixture/rf-detr",
        source_revision="a" * 40,
        model_family="RF_DETR_SEGMENTATION_COCO",
        training_label_space="COCO",
        license_id="Apache-2.0",
        license_url="https://example.org/license",
        files=tuple(
            sorted(
                (
                    _binding(root, "config.json", b"{}"),
                    _binding(root, "model.safetensors", b"weights"),
                    _binding(root, "preprocessor_config.json", b"{}"),
                ),
                key=lambda item: item.relative_path,
            )
        ),
    )
    bundle_path = tmp_path / "manifest.json"
    bundle_path.write_text(
        json.dumps(instance_segmentation_model_bundle(manifest)), encoding="utf-8"
    )
    artifact = InstanceSegmentationArtifact.load(
        model_directory=root, manifest_bundle_path=bundle_path
    )
    assert artifact.manifest.manifest_sha256 == manifest.manifest_sha256
    (root / "config.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte size|SHA-256"):
        artifact.revalidate_local_files()


def test_all_target_candidates_has_no_best_score_fallback() -> None:
    classes = np.asarray(
        (
            (0.1, 0.2, 0.7),
            (0.8, 0.1, 0.1),
            (0.1, 0.7, 0.2),
        ),
        dtype=np.float32,
    )
    masks = np.zeros((3, 4, 5), dtype=np.float32)
    masks[2, 1:3, 2:4] = 0.9
    candidates = _all_target_candidates(
        class_probabilities=classes,
        mask_probabilities=masks,
        class_names_by_id={1: "dog"},
        mask_threshold=0.5,
        minimum_class_score=0.25,
    )
    assert len(candidates) == 1
    assert candidates[0].query_index == 2
    assert candidates[0].source_box_xyxy == (2, 1, 4, 3)


def test_all_target_candidates_returns_empty_below_policy() -> None:
    classes = np.asarray(((0.4, 0.2, 0.4),), dtype=np.float32)
    masks = np.full((1, 3, 3), 0.9, dtype=np.float32)
    assert not _all_target_candidates(
        class_probabilities=classes,
        mask_probabilities=masks,
        class_names_by_id={1: "dog"},
        mask_threshold=0.5,
        minimum_class_score=0.25,
    )


def test_all_target_candidates_thresholds_requested_class_despite_no_object() -> None:
    classes = np.asarray(((0.25, 0.35, 0.4),), dtype=np.float32)
    masks = np.full((1, 3, 3), 0.9, dtype=np.float32)
    candidates = _all_target_candidates(
        class_probabilities=classes,
        mask_probabilities=masks,
        class_names_by_id={1: "dog"},
        mask_threshold=0.5,
        minimum_class_score=0.25,
    )
    assert len(candidates) == 1
    assert candidates[0].class_name == "dog"
    assert candidates[0].class_score == pytest.approx(0.35)


def test_all_target_candidates_preserves_source_query_indices() -> None:
    classes = np.asarray(((0.1, 0.8, 0.1), (0.1, 0.7, 0.2)), dtype=np.float32)
    masks = np.full((2, 3, 3), 0.9, dtype=np.float32)
    candidates = _all_target_candidates(
        class_probabilities=classes,
        mask_probabilities=masks,
        class_names_by_id={1: "dog"},
        mask_threshold=0.5,
        minimum_class_score=0.25,
        query_indices=np.asarray((17, 29), dtype=np.int64),
    )
    assert [candidate.query_index for candidate in candidates] == [17, 29]


def _candidate(
    query_index: int,
    score: float,
    support: tuple[slice, slice],
) -> AnimalInstanceCandidate:
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[support] = 0.9
    hard_mask = (probability >= 0.5).astype(np.uint8)
    ys, xs = np.nonzero(hard_mask)
    return AnimalInstanceCandidate(
        probability=probability,
        hard_mask=hard_mask,
        source_box_xyxy=(
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ),
        query_index=query_index,
        class_id=1,
        class_name="dog",
        class_score=score,
    )


def test_duplicate_instance_suppression_is_score_ordered_and_deterministic() -> None:
    lower = _candidate(4, 0.8, (slice(1, 5), slice(1, 5)))
    higher = _candidate(2, 0.9, (slice(1, 5), slice(1, 5)))
    separate = _candidate(1, 0.7, (slice(5, 7), slice(5, 7)))
    first = _suppress_duplicate_candidates(
        [lower, separate, higher], duplicate_mask_iou=0.8, maximum_instances=10
    )
    second = _suppress_duplicate_candidates(
        [higher, lower, separate], duplicate_mask_iou=0.8, maximum_instances=10
    )
    assert [item.query_index for item in first] == [2, 1]
    assert [item.query_index for item in second] == [2, 1]


def test_duplicate_instance_suppression_removes_contained_query() -> None:
    outer = _candidate(1, 0.9, (slice(1, 7), slice(1, 7)))
    inner = _candidate(2, 0.8, (slice(2, 6), slice(2, 6)))
    retained = _suppress_duplicate_candidates(
        [inner, outer], duplicate_mask_iou=0.8, maximum_instances=10
    )
    assert [item.query_index for item in retained] == [1]
