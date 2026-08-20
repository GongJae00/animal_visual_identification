from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from parsing.export.segmentation.animal_instance_segmentation import AnimalInstanceCandidate
from parsing.export.segmentation.animal_parsing import (
    AnimalParsingPolicy,
    AnimalParsingRuntime,
    ParsedAnimalInstance,
    ParsedAnimalQuality,
    _DraftInstance,
    _exclusive_ownership,
    _seeded_refinement,
    materialize_identity_crop,
)
from parsing.export.segmentation.foreground_segmentation import ForegroundSegmentationPrediction

def test_animal_parsing_policy_v6_is_dog_only_and_prior_policies_round_trip() -> None:
    policy = AnimalParsingPolicy()
    assert policy.schema_version == "cvi.animal_parsing_policy.v6"
    assert policy.class_names == ("dog",)

    legacy = replace(
        policy,
        class_names=("dog", "cat"),
        schema_version="cvi.animal_parsing_policy.v5",
    )
    restored = AnimalParsingPolicy.from_dict(legacy.to_dict())
    assert restored == legacy
    assert restored.policy_sha256 == legacy.policy_sha256

    v4 = replace(legacy, schema_version="cvi.animal_parsing_policy.v4")
    assert AnimalParsingPolicy.from_dict(v4.to_dict()) == v4

def test_animal_parsing_policy_v6_rejects_non_dog_classes() -> None:
    with pytest.raises(ValueError, match="dog-only"):
        AnimalParsingPolicy(class_names=("dog", "cat"))
    with pytest.raises(ValueError, match="schema differs"):
        AnimalParsingPolicy(schema_version="cvi.animal_parsing_policy.v3")

def _candidate(
    *,
    query_index: int,
    box: tuple[int, int, int, int],
    shape: tuple[int, int] = (12, 16),
    score: float = 0.9,
) -> AnimalInstanceCandidate:
    probability = np.zeros(shape, dtype=np.float32)
    x1, y1, x2, y2 = box
    probability[y1:y2, x1:x2] = 0.9
    return AnimalInstanceCandidate(
        probability=probability,
        hard_mask=(probability >= 0.5).astype(np.uint8),
        source_box_xyxy=box,
        query_index=query_index,
        class_id=1,
        class_name="dog",
        class_score=score,
    )

def test_seeded_refinement_expands_shape_but_rejects_disconnected_saliency() -> None:
    policy = replace(AnimalParsingPolicy(), minimum_support_dilation_pixels=1)
    instance = np.zeros((12, 16), dtype=np.float32)
    instance[3:9, 4:10] = 0.9
    foreground = np.zeros_like(instance)
    foreground[2:10, 3:11] = 0.9
    foreground[1:4, 13:16] = 0.9
    mask, _, agreement, empty = _seeded_refinement(
        foreground_probability=foreground,
        instance_probability=instance,
        policy=policy,
    )
    assert not empty
    assert mask[3:9, 4:10].all()
    assert mask[3:9, 3].all()
    assert mask[3:9, 10].all()
    assert not mask[:, 13:].any()
    assert 0.0 < agreement < 1.0

def test_seeded_refinement_never_discards_cleaned_rf_hard_support() -> None:
    policy = replace(AnimalParsingPolicy(), minimum_support_dilation_pixels=1)
    instance = np.zeros((20, 20), dtype=np.float32)
    instance[3:17, 4:16] = 0.6
    foreground = np.zeros_like(instance)
    foreground[5:15, 6:14] = 0.9
    mask, _, _, empty = _seeded_refinement(
        foreground_probability=foreground,
        instance_probability=instance,
        policy=policy,
    )
    assert not empty
    assert mask[3:17, 4:16].all()

def test_exclusive_ownership_assigns_contact_pixels_once() -> None:
    first = _candidate(query_index=0, box=(2, 2, 9, 9))
    second = _candidate(query_index=1, box=(7, 2, 14, 9))
    first_mask = first.hard_mask.copy()
    second_mask = second.hard_mask.copy()
    first_score = first.probability.copy()
    second_score = second.probability.copy()
    first_score[:, 7:9] = 0.95
    drafts = [
        _DraftInstance(
            first,
            first.source_box_xyxy,
            first.probability,
            first_score,
            first_mask,
            1.0,
            False,
        ),
        _DraftInstance(
            second,
            second.source_box_xyxy,
            second.probability,
            second_score,
            second_mask,
            1.0,
            False,
        ),
    ]
    owned = _exclusive_ownership(drafts, shape=first.hard_mask.shape)
    assert not np.any(owned[0] & owned[1])
    assert owned[0][:, 7:9].sum() > 0
    assert sum(int(mask.sum()) for mask in owned) == int(
        np.count_nonzero(first_mask | second_mask)
    )

def test_exclusive_ownership_equal_score_tie_keeps_earlier_instance() -> None:
    first = _candidate(query_index=0, box=(2, 2, 10, 10))
    second = _candidate(query_index=1, box=(6, 2, 14, 10))
    drafts = [
        _DraftInstance(
            first,
            first.source_box_xyxy,
            first.probability,
            first.probability,
            first.hard_mask,
            1.0,
            False,
        ),
        _DraftInstance(
            second,
            second.source_box_xyxy,
            second.probability,
            second.probability,
            second.hard_mask,
            1.0,
            False,
        ),
    ]
    owned = _exclusive_ownership(drafts, shape=first.hard_mask.shape)
    assert owned[0][:, 6:10].sum() == first.hard_mask[:, 6:10].sum()
    assert not owned[1][:, 6:10].any()

class _InstanceRuntime:
    def __init__(self, candidates: tuple[AnimalInstanceCandidate, ...]) -> None:
        self.candidates = candidates
        self.class_names: list[tuple[str, ...]] = []

    def predict_all(
        self, image: Image.Image, **kwargs: object
    ) -> tuple[AnimalInstanceCandidate, ...]:
        assert image.size == (16, 12)
        self.class_names.append(kwargs["class_names"])  # type: ignore[arg-type]
        return self.candidates

    def predict_all_batch(
        self, images: tuple[Image.Image, ...], **kwargs: object
    ) -> tuple[tuple[AnimalInstanceCandidate, ...], ...]:
        self.class_names.append(kwargs["class_names"])  # type: ignore[arg-type]
        return tuple(self.candidates for _ in images)

class _ForegroundRuntime:
    def predict(
        self, image: Image.Image, *, target_box_xyxy: tuple[int, int, int, int]
    ) -> ForegroundSegmentationPrediction:
        probability = np.zeros((12, 16), dtype=np.float32)
        x1, y1, x2, y2 = target_box_xyxy
        probability[y1:y2, x1:x2] = 0.9
        hard_mask = (probability >= 0.5).astype(np.uint8)
        return ForegroundSegmentationPrediction(
            probability=probability,
            hard_mask=hard_mask,
            source_box_xyxy=target_box_xyxy,
            inference_width=32,
            inference_height=32,
            threshold=0.5,
            foreground_fraction=float(hard_mask.mean()),
            border_foreground_fraction=0.0,
            state="CANDIDATE",
            reasons=(),
        )

    def predict_batch(
        self,
        images: tuple[Image.Image, ...],
        *,
        target_boxes_xyxy: tuple[tuple[int, int, int, int], ...],
        maximum_batch_size: int,
    ) -> tuple[ForegroundSegmentationPrediction, ...]:
        assert maximum_batch_size > 0
        return tuple(
            self.predict(image, target_box_xyxy=box)
            for image, box in zip(images, target_boxes_xyxy, strict=True)
        )

def test_multi_instance_parser_returns_disjoint_source_coordinate_masks() -> None:
    candidates = (
        _candidate(query_index=0, box=(1, 2, 8, 10)),
        _candidate(query_index=1, box=(7, 2, 15, 10)),
    )
    runtime = AnimalParsingRuntime(
        instance_runtime=_InstanceRuntime(candidates),  # type: ignore[arg-type]
        foreground_runtime=_ForegroundRuntime(),  # type: ignore[arg-type]
        policy=replace(AnimalParsingPolicy(), minimum_mask_pixels=4),
    )
    prediction = runtime.predict(Image.new("RGB", (16, 12), "white"))
    assert len(prediction.instances) == 2
    assert not np.any(
        prediction.instances[0].hard_mask & prediction.instances[1].hard_mask
    )
    assert all(item.hard_mask.shape == (12, 16) for item in prediction.instances)

def test_multi_instance_parser_returns_empty_result_without_target_queries() -> None:
    runtime = AnimalParsingRuntime(
        instance_runtime=_InstanceRuntime(()),  # type: ignore[arg-type]
        foreground_runtime=_ForegroundRuntime(),  # type: ignore[arg-type]
    )
    prediction = runtime.predict(Image.new("RGB", (16, 12), "white"))
    assert prediction.instances == ()

def test_default_parser_requests_only_dogs_for_single_and_batch_inference() -> None:
    instance_runtime = _InstanceRuntime(())
    runtime = AnimalParsingRuntime(
        instance_runtime=instance_runtime,  # type: ignore[arg-type]
        foreground_runtime=_ForegroundRuntime(),  # type: ignore[arg-type]
    )
    image = Image.new("RGB", (16, 12), "white")
    runtime.predict(image)
    runtime.predict_batch(
        (image,), instance_batch_size=1, foreground_batch_size=1
    )
    assert instance_runtime.class_names == [("dog",), ("dog",)]

def test_batched_parser_preserves_input_and_candidate_order() -> None:
    candidates = (
        _candidate(query_index=0, box=(1, 2, 8, 10)),
        _candidate(query_index=1, box=(7, 2, 15, 10)),
    )
    runtime = AnimalParsingRuntime(
        instance_runtime=_InstanceRuntime(candidates),  # type: ignore[arg-type]
        foreground_runtime=_ForegroundRuntime(),  # type: ignore[arg-type]
        policy=replace(AnimalParsingPolicy(), minimum_mask_pixels=4),
    )
    images = (Image.new("RGB", (16, 12), "white"),) * 3

    batched = runtime.predict_batch(
        images, instance_batch_size=3, foreground_batch_size=3
    )
    repeated = tuple(runtime.predict(image) for image in images)

    assert [
        [instance.query_index for instance in prediction.instances]
        for prediction in batched
    ] == [[0, 1], [0, 1], [0, 1]]
    for left, right in zip(batched, repeated, strict=True):
        for left_instance, right_instance in zip(
            left.instances, right.instances, strict=True
        ):
            np.testing.assert_array_equal(
                left_instance.hard_mask, right_instance.hard_mask
            )

class _EmptyForegroundRuntime:
    def predict(
        self, image: Image.Image, *, target_box_xyxy: tuple[int, int, int, int]
    ) -> ForegroundSegmentationPrediction:
        probability = np.zeros((12, 16), dtype=np.float32)
        hard_mask = np.zeros((12, 16), dtype=np.uint8)
        return ForegroundSegmentationPrediction(
            probability=probability,
            hard_mask=hard_mask,
            source_box_xyxy=target_box_xyxy,
            inference_width=32,
            inference_height=32,
            threshold=0.5,
            foreground_fraction=0.0,
            border_foreground_fraction=0.0,
            state="ABSTAIN",
            reasons=("EMPTY_FOREGROUND",),
        )

def test_empty_foreground_refinement_is_returned_but_not_identity_eligible() -> None:
    runtime = AnimalParsingRuntime(
        instance_runtime=_InstanceRuntime(
            (_candidate(query_index=0, box=(2, 2, 10, 10)),)
        ),  # type: ignore[arg-type]
        foreground_runtime=_EmptyForegroundRuntime(),  # type: ignore[arg-type]
        policy=replace(AnimalParsingPolicy(), minimum_mask_pixels=4),
    )
    instance = runtime.predict(Image.new("RGB", (16, 12), "white")).instances[0]
    assert instance.quality.state == "UNUSABLE"
    assert "FOREGROUND_REFINEMENT_EMPTY" in instance.quality.reasons
    with pytest.raises(ValueError, match="requires a usable"):
        materialize_identity_crop(Image.new("RGB", (16, 12)), instance)

def test_full_frame_parser_mask_is_not_identity_eligible() -> None:
    candidate = _candidate(query_index=0, box=(0, 0, 16, 12))
    runtime = AnimalParsingRuntime(
        instance_runtime=_InstanceRuntime((candidate,)),  # type: ignore[arg-type]
        foreground_runtime=_ForegroundRuntime(),  # type: ignore[arg-type]
        policy=replace(AnimalParsingPolicy(), minimum_mask_pixels=4),
    )

    instance = runtime.predict(Image.new("RGB", (16, 12), "white")).instances[0]

    assert instance.quality.state == "UNUSABLE"
    assert "FULL_FRAME_FOREGROUND" in instance.quality.reasons

def test_identity_crop_replaces_every_background_pixel_deterministically() -> None:
    source = Image.fromarray(
        np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3), mode="RGB"
    )
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 2:6] = 1
    probability = mask.astype(np.float32)
    quality = ParsedAnimalQuality("USABLE", (), (), 1.0, 1.0, 12, 1, False)
    instance = ParsedAnimalInstance(
        instance_index=0,
        query_index=1,
        class_id=1,
        class_name="dog",
        class_score=0.9,
        detector_box_xyxy=(2, 2, 6, 5),
        refinement_box_xyxy=(1, 1, 7, 6),
        mask_box_xyxy=(2, 2, 6, 5),
        instance_probability=probability,
        foreground_probability=probability,
        ownership_probability=probability,
        hard_mask=mask,
        quality=quality,
    )
    first = materialize_identity_crop(
        source, instance, context_fraction=0.25, background_rgb=(10, 20, 30)
    )
    second = materialize_identity_crop(
        source, instance, context_fraction=0.25, background_rgb=(10, 20, 30)
    )
    values = np.asarray(first.masked_rgb)
    mask_values = np.asarray(first.mask).astype(bool)
    assert np.all(values[~mask_values] == np.asarray((10, 20, 30)))
    np.testing.assert_array_equal(values, np.asarray(second.masked_rgb))

def test_identity_crop_rejects_review_instance_by_default() -> None:
    mask = np.ones((4, 4), dtype=np.uint8)
    probability = mask.astype(np.float32)
    instance = ParsedAnimalInstance(
        instance_index=0,
        query_index=0,
        class_id=1,
        class_name="dog",
        class_score=0.9,
        detector_box_xyxy=(0, 0, 4, 4),
        refinement_box_xyxy=(0, 0, 4, 4),
        mask_box_xyxy=(0, 0, 4, 4),
        instance_probability=probability,
        foreground_probability=probability,
        ownership_probability=probability,
        hard_mask=mask,
        quality=ParsedAnimalQuality(
            "REVIEW", (), ("SOURCE_BORDER_TRUNCATED",), 1.0, 1.0, 16, 1, True
        ),
    )
    with pytest.raises(ValueError, match="requires a usable"):
        materialize_identity_crop(Image.new("RGB", (4, 4)), instance)
