from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from data.types import CaptureGroupKind, UnifiedCanidSample
from identity.registry.generated_identity_registry import create_provisional_identity
from parsing.full_segment.animal_parsing import (
    AnimalIdentityCrop,
    ParsedAnimalInstance,
    ParsedAnimalQuality,
)
from workflows.evaluate_parsed_body_reid import (
    GENERATOR_ID,
    _AcceptedFrame,
    _background_plus_silhouette,
    _candidate_tracks,
    _evaluate_view,
    _identity_instance,
    _read_source_image,
    _require_external_output,
    _validate_archive_topology,
)


def _sample(identity: str, frame: int, *, split: str = "train") -> UnifiedCanidSample:
    return UnifiedCanidSample(
        sample_id=f"{identity}-{frame}",
        dataset_name="yt-bb-dog",
        dataset_version="publisher-v1-2025-10-27",
        source_group_id=identity,
        image_path=f"YT-BB-dog/YT-BB-Dog/{split}/{identity}/{identity}_{frame}.jpg",
        image_sha256=f"{frame:064x}",
        width=16,
        height=12,
        raw_identity_id=identity,
        capture_group_id=identity,
        capture_group_kind=CaptureGroupKind.VIDEO_TRACK,
        split_role=split,
    )


def _instance(*, state: str = "USABLE", class_name: str = "dog") -> ParsedAnimalInstance:
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:10, 3:13] = 1
    probability = mask.astype(np.float32)
    reasons = ("fixture",) if state == "UNUSABLE" else ()
    flags = ("fixture",) if state == "REVIEW" else ()
    return ParsedAnimalInstance(
        instance_index=0,
        query_index=0,
        class_id=1,
        class_name=class_name,
        class_score=0.9,
        detector_box_xyxy=(3, 2, 13, 10),
        refinement_box_xyxy=(2, 1, 14, 11),
        mask_box_xyxy=(3, 2, 13, 10),
        instance_probability=probability,
        foreground_probability=probability,
        ownership_probability=probability,
        hard_mask=mask,
        quality=ParsedAnimalQuality(
            state, reasons, flags, 1.0, 1.0, 80, 1, False
        ),
    )


def _frame(identity: str, frame: int) -> _AcceptedFrame:
    generated = create_provisional_identity(
        GENERATOR_ID, f"fixture\0{identity}", evidence_count=2
    )
    source = Image.new("RGB", (4, 4), (10 + frame, 20, 30))
    mask = Image.new("L", (4, 4), 255)
    return _AcceptedFrame(
        sample=_sample(identity, frame),
        generated_identity_id=generated.generated_identity_id,
        crop=AnimalIdentityCrop(
            box_rgb=source,
            masked_rgb=source,
            mask=mask,
            source_box_xyxy=(0, 0, 4, 4),
            instance_index=0,
            class_name="dog",
            parsing_quality_state="USABLE",
        ),
        quality={"state": "USABLE"},
    )


def test_candidate_tracks_use_only_train_and_keep_temporal_endpoints() -> None:
    samples = [
        *(_sample("a", frame) for frame in (0, 1, 2, 3, 10)),
        _sample("b", 0),
        _sample("b", 1),
        _sample("c", 0, split="test"),
        _sample("c", 1, split="test"),
    ]
    selected = _candidate_tracks(
        tuple(reversed(samples)), candidate_limit=3, frames_per_identity=3
    )
    assert [identity for identity, _ in selected] == ["a", "b"]
    assert [sample.image_path for sample in selected[0][1]] == [
        _sample("a", 0).image_path,
        _sample("a", 2).image_path,
        _sample("a", 10).image_path,
    ]


def test_identity_input_requires_exactly_one_usable_dog() -> None:
    usable = _instance()
    assert _identity_instance((usable,)) == (usable, None)
    assert _identity_instance((_instance(class_name="cat"),))[1] == "NO_DOG_INSTANCE"
    assert _identity_instance((usable, _instance()))[1] == "MULTIPLE_DOG_INSTANCES"
    assert _identity_instance((_instance(state="REVIEW"),))[1] == "DOG_REVIEW"


def test_background_plus_silhouette_neutralizes_foreground_and_retains_context() -> None:
    values = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    crop = AnimalIdentityCrop(
        box_rgb=Image.fromarray(values, mode="RGB"),
        masked_rgb=Image.new("RGB", (4, 4)),
        mask=Image.fromarray(mask, mode="L"),
        source_box_xyxy=(0, 0, 4, 4),
        instance_index=0,
        class_name="dog",
        parsing_quality_state="USABLE",
    )
    result = np.asarray(_background_plus_silhouette(crop))
    assert np.all(result[1:3, 1:3] == 127)
    assert np.array_equal(result[0, 0], values[0, 0])


def test_output_must_remain_outside_repository(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    inside = repository / "artifacts"
    outside = tmp_path / "external"
    outside.mkdir()
    try:
        _require_external_output(inside, repository_root=repository)
    except ValueError as exc:
        assert "outside Git" in str(exc)
    else:
        raise AssertionError("repository-local artifact output was accepted")
    assert _require_external_output(
        outside / "result", repository_root=repository
    ) == outside / "result"


def test_source_image_must_match_admitted_archive_member(tmp_path) -> None:
    root = tmp_path / "dataset"
    image_path = root / "YT-BB-dog" / "YT-BB-Dog" / "train" / "a" / "a_0.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), "blue").save(image_path, format="JPEG")
    payload = image_path.read_bytes()
    sample = replace(
        _sample("a", 0), image_sha256=hashlib.sha256(payload).hexdigest()
    )
    archive_path = tmp_path / "YT-BB-Dog.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("YT-BB-Dog/train/a/a_0.jpg", payload)
    with zipfile.ZipFile(archive_path) as archive:
        assert _read_source_image(
            sample, dataset_root=root, source_archive=archive
        ).size == (16, 12)
        _validate_archive_topology((sample,), archive)
    Image.new("RGB", (16, 12), "red").save(image_path, format="JPEG")
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(ValueError, match="SHA-256 differs"),
    ):
        _read_source_image(sample, dataset_root=root, source_archive=archive)


def test_archive_topology_must_match_complete_adapter_inventory(tmp_path) -> None:
    archive_path = tmp_path / "YT-BB-Dog.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("YT-BB-Dog/train/a/a_0.jpg", b"first")
        archive.writestr("YT-BB-Dog/train/a/a_1.jpg", b"second")
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(ValueError, match="topology differs"),
    ):
        _validate_archive_topology((_sample("a", 0),), archive)


def test_view_evaluation_uses_generated_track_ids_and_exact_cosine() -> None:
    frames = [
        (_frame("a", 0).generated_identity_id, [_frame("a", 0), _frame("a", 1)]),
        (_frame("b", 0).generated_identity_id, [_frame("b", 0), _frame("b", 1)]),
    ]
    embeddings = np.zeros((4, 384), dtype=np.float32)
    embeddings[0:2, 0] = 1.0
    embeddings[2:4, 1] = 1.0
    result = _evaluate_view(
        frames, embeddings, bootstrap_resamples=20, bootstrap_seed=7
    )
    assert result["gallery_identities"] == 2
    assert result["queries"] == 2
    assert result["Rank-1"] == 1.0
    assert all("generated_identity_id" in row for row in result["query_rows"])
