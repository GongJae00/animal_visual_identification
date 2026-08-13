from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import embedding.methods.full_segment.preparation.sample_materialization as materialize_workflow
from foundation.provenance import content_sha256
from parsing.full_segment.animal_parsing import (
    AnimalParsingPrediction,
    ParsedAnimalInstance,
    ParsedAnimalQuality,
)
from parsing.full_segment.full_segment_cache import (
    build_body_mask_observation,
    build_body_observation,
    build_full_segment_cache,
    freeze_animal_parsing_prediction,
    thaw_animal_parsing_prediction,
    validate_frozen_animal_parsing,
    validate_full_segment_cache_bundle,
)
from parsing.full_segment.full_segment_contracts import (
    AnimalAssociation,
    AssociationKind,
    BodyMaskPolicy,
    BodyMaskPolicyKind,
    FullSegmentObservation,
    FullStatus,
    ObservationRoute,
    SourceViewScope,
    TerminalObservability,
    build_native_observation,
    build_terminal_observation,
)
from parsing.full_segment.full_segment_crop import (
    materialize_body_mask_full_crop,
    materialize_full_crop,
    materialize_native_full_crop,
    verify_full_crop_artifacts,
)
from workflows.materialize_full_segment import REQUEST_SCHEMA, run


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _source_bytes() -> bytes:
    values = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    output = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(
        output, format="PNG", optimize=False, compress_level=9
    )
    return output.getvalue()


def _mask_bytes(values: np.ndarray | None = None) -> bytes:
    if values is None:
        values = np.full((8, 12), 2, dtype=np.uint8)
        values[0, :] = 3
        values[2:7, 3:9] = 1
    output = io.BytesIO()
    Image.fromarray(values, mode="L").save(
        output, format="PNG", optimize=False, compress_level=9
    )
    return output.getvalue()


def _body_mask_policy(
    permitted_labels: tuple[int, ...] | None = None,
) -> BodyMaskPolicy:
    return BodyMaskPolicy(
        BodyMaskPolicyKind.OXFORD_IIIT_PET_TRIMAP,
        permitted_labels=permitted_labels,
    )


def _instance(
    index: int = 0,
    *,
    box: tuple[int, int, int, int] = (3, 2, 9, 7),
    state: str = "USABLE",
) -> ParsedAnimalInstance:
    mask = np.zeros((8, 12), dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    probability = np.ascontiguousarray(mask * np.float32(0.9), dtype=np.float32)
    reasons = ("MASK_SUPPORT_BELOW_POLICY",) if state == "UNUSABLE" else ()
    flags = ("SOURCE_BORDER_TRUNCATED",) if state == "REVIEW" else ()
    return ParsedAnimalInstance(
        instance_index=index,
        query_index=index + 10,
        class_id=1,
        class_name="dog",
        class_score=0.9,
        detector_box_xyxy=box,
        refinement_box_xyxy=(
            max(0, x1 - 1),
            max(0, y1 - 1),
            min(12, x2 + 1),
            min(8, y2 + 1),
        ),
        mask_box_xyxy=box,
        instance_probability=probability,
        foreground_probability=probability.copy(),
        ownership_probability=probability.copy(),
        hard_mask=mask,
        quality=ParsedAnimalQuality(
            state=state,
            reasons=reasons,
            flags=flags,
            semantic_shape_iou=0.9,
            ownership_retention=1.0,
            foreground_pixels=int(mask.sum()),
            component_count=1,
            touches_source_border=state == "REVIEW",
        ),
    )


def _prediction(
    instances: tuple[ParsedAnimalInstance, ...] | None = None,
) -> AnimalParsingPrediction:
    return AnimalParsingPrediction(
        source_width=12,
        source_height=8,
        instances=instances if instances is not None else (_instance(),),
        policy_sha256=_sha("parser-policy"),
    )


def _exactly_one() -> AnimalAssociation:
    return AnimalAssociation(AssociationKind.EXACTLY_ONE, 0)


def test_scope_status_and_observability_values_are_explicit() -> None:
    assert {item.value for item in SourceViewScope} == {
        "BODY_AVAILABLE",
        "BODY_TRUNCATED",
        "FACE_NATIVE",
        "HEAD_NATIVE",
        "AMBIGUOUS",
        "UNAVAILABLE",
    }
    assert {item.value for item in FullStatus} == {
        "USABLE",
        "REVIEW",
        "UNUSABLE",
        "AMBIGUOUS",
    }
    assert {item.value for item in TerminalObservability} == {
        "NOT_RUN",
        "NOT_DETECTED",
        "REVIEW",
        "USABLE",
        "NATIVE",
    }
    assert {item.value for item in ObservationRoute} == {
        "BODY_PARSING",
        "BODY_MASK",
        "NATIVE_FACE",
        "NATIVE_HEAD",
        "NONE",
    }


def test_body_mask_policy_is_versioned_exact_and_content_bound() -> None:
    policy = _body_mask_policy((1, 2))
    assert BodyMaskPolicy.from_dict(policy.to_dict()) == policy
    assert policy.to_dict() == _body_mask_policy((1, 2)).to_dict()

    changed = policy.to_dict()
    changed["excluded_labels"] = [2]
    changed["policy_sha256"] = content_sha256(
        {key: item for key, item in changed.items() if key != "policy_sha256"}
    )
    with pytest.raises(ValueError, match="label semantics"):
        BodyMaskPolicy.from_dict(changed)


def test_frozen_parsing_round_trips_complete_arrays_deterministically() -> None:
    prediction = _prediction()
    first = freeze_animal_parsing_prediction(prediction)
    second = freeze_animal_parsing_prediction(prediction)
    assert first == second
    restored = thaw_animal_parsing_prediction(first)
    assert restored.policy_sha256 == prediction.policy_sha256
    np.testing.assert_array_equal(
        restored.instances[0].hard_mask, prediction.instances[0].hard_mask
    )
    np.testing.assert_array_equal(
        restored.instances[0].instance_probability,
        prediction.instances[0].instance_probability,
    )


def test_frozen_parsing_rejects_content_and_packed_array_tampering() -> None:
    frozen = freeze_animal_parsing_prediction(_prediction())
    changed = deepcopy(frozen)
    changed["prediction"]["policy_sha256"] = _sha("different")
    with pytest.raises(ValueError, match="prediction digest"):
        validate_frozen_animal_parsing(changed)

    changed = deepcopy(frozen)
    packed = changed["prediction"]["instances"][0]["hard_mask"]
    packed["raw_sha256"] = _sha("wrong-array")
    changed["prediction_sha256"] = content_sha256(changed["prediction"])
    with pytest.raises(ValueError, match="digest or length"):
        validate_frozen_animal_parsing(changed)


def test_body_route_requires_exactly_one_or_authoritative_association() -> None:
    first = _instance(0, box=(1, 1, 5, 7))
    second = _instance(1, box=(7, 1, 11, 7))
    frozen = freeze_animal_parsing_prediction(_prediction((first, second)))
    with pytest.raises(ValueError, match="exactly one prediction"):
        build_body_observation(
            source_id="sample",
            source_sha256=_sha("source"),
            source_view_scope=SourceViewScope.BODY_AVAILABLE,
            frozen_parsing=frozen,
            association=_exactly_one(),
        )

    observation = build_body_observation(
        source_id="sample",
        source_sha256=_sha("source"),
        source_view_scope=SourceViewScope.BODY_AVAILABLE,
        frozen_parsing=frozen,
        association=AnimalAssociation(
            AssociationKind.AUTHORITATIVE, 1, _sha("association-receipt")
        ),
        face_observability=TerminalObservability.NOT_DETECTED,
        nose_observability=TerminalObservability.NOT_RUN,
    )
    assert observation.association is not None
    assert observation.association.instance_index == 1
    assert FullSegmentObservation.from_dict(observation.to_dict()) == observation


def test_native_route_is_explicit_and_cannot_claim_whole_body_parsing() -> None:
    observation = build_native_observation(
        source_id="native-face",
        source_sha256=_sha("native-face"),
        source_width=96,
        source_height=80,
        source_view_scope=SourceViewScope.FACE_NATIVE,
        native_artifact_sha256=_sha("native-face"),
        full_rgb_sha256=_sha("native-full-crop"),
        nose_observability=TerminalObservability.NOT_DETECTED,
    )
    assert observation.route is ObservationRoute.NATIVE_FACE
    assert observation.full_status is FullStatus.USABLE
    assert observation.face_observability is TerminalObservability.NATIVE
    assert observation.parsing_prediction_sha256 is None
    assert observation.association is None

    changed = observation.to_dict()
    changed["parsing_prediction_sha256"] = _sha("false-parsing")
    changed["observation_sha256"] = content_sha256(
        {key: value for key, value in changed.items() if key != "observation_sha256"}
    )
    with pytest.raises(ValueError, match="observation values differ"):
        FullSegmentObservation.from_dict(changed)


def test_native_route_materializes_square_full_rgb_and_mask() -> None:
    source = _source_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    crop = materialize_native_full_crop(
        source,
        expected_source_sha256=source_sha256,
        route=ObservationRoute.NATIVE_FACE,
        target_size=32,
        background_rgb=(101, 102, 103),
    )

    assert crop.record["route"] == "NATIVE_FACE"
    assert crop.record["parsing_prediction_sha256"] is None
    assert crop.record["association"] is None
    assert crop.record["parsing_quality_state"] == "NATIVE"
    verify_full_crop_artifacts(crop)
    with Image.open(io.BytesIO(crop.full_mask_png)) as mask:
        values = np.asarray(mask)
        assert set(np.unique(values)) == {0, 255}
        assert np.all(values[5:27] == 255)


def test_terminal_observation_preserves_ambiguous_and_unavailable_states() -> None:
    ambiguous = build_terminal_observation(
        source_id="ambiguous",
        source_sha256=_sha("ambiguous"),
        source_width=10,
        source_height=8,
        source_view_scope=SourceViewScope.AMBIGUOUS,
        face_observability=TerminalObservability.REVIEW,
        nose_observability=TerminalObservability.NOT_RUN,
    )
    unavailable = build_terminal_observation(
        source_id="unavailable",
        source_sha256=_sha("unavailable"),
        source_width=10,
        source_height=8,
        source_view_scope=SourceViewScope.UNAVAILABLE,
        face_observability=TerminalObservability.NOT_DETECTED,
        nose_observability=TerminalObservability.NOT_DETECTED,
    )
    assert ambiguous.full_status is FullStatus.AMBIGUOUS
    assert unavailable.full_status is FullStatus.UNUSABLE


def test_full_crop_is_square_binary_neutral_and_content_bound() -> None:
    source = _source_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    frozen = freeze_animal_parsing_prediction(_prediction())
    first = materialize_full_crop(
        source,
        expected_source_sha256=source_sha256,
        frozen_parsing=frozen,
        association=_exactly_one(),
        target_size=32,
        context_fraction=0.25,
        background_rgb=(101, 102, 103),
    )
    second = materialize_full_crop(
        source,
        expected_source_sha256=source_sha256,
        frozen_parsing=frozen,
        association=_exactly_one(),
        target_size=32,
        context_fraction=0.25,
        background_rgb=(101, 102, 103),
    )
    assert first == second
    assert first.record["source_sha256"] == source_sha256
    assert (
        first.record["full_rgb_sha256"]
        == hashlib.sha256(first.full_rgb_png).hexdigest()
    )
    assert (
        first.record["full_mask_sha256"]
        == hashlib.sha256(first.full_mask_png).hexdigest()
    )
    verify_full_crop_artifacts(first)
    with (
        Image.open(io.BytesIO(first.full_rgb_png)) as rgb,
        Image.open(io.BytesIO(first.full_mask_png)) as mask,
    ):
        rgb_values = np.asarray(rgb)
        mask_values = np.asarray(mask)
        assert rgb.size == mask.size == (32, 32)
        assert set(np.unique(mask_values)).issubset({0, 255})
        assert np.all(rgb_values[mask_values == 0] == (101, 102, 103))

    with pytest.raises(ValueError, match="source digest"):
        materialize_full_crop(
            source,
            expected_source_sha256=_sha("wrong-source"),
            frozen_parsing=frozen,
            association=_exactly_one(),
        )


def test_authoritative_body_mask_crop_is_deterministic_and_not_parser_output() -> None:
    source = _source_bytes()
    mask = _mask_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    mask_sha256 = hashlib.sha256(mask).hexdigest()
    policy = _body_mask_policy()
    first = materialize_body_mask_full_crop(
        source,
        mask,
        expected_source_sha256=source_sha256,
        expected_authoritative_mask_sha256=mask_sha256,
        policy=policy,
        target_size=32,
        context_fraction=0.25,
        background_rgb=(101, 102, 103),
    )
    second = materialize_body_mask_full_crop(
        source,
        mask,
        expected_source_sha256=source_sha256,
        expected_authoritative_mask_sha256=mask_sha256,
        policy=policy,
        target_size=32,
        context_fraction=0.25,
        background_rgb=(101, 102, 103),
    )

    assert first == second
    assert first.record["route"] == "BODY_MASK"
    assert first.record["authoritative_mask_sha256"] == mask_sha256
    assert first.record["mask_policy_sha256"] == policy.policy_sha256
    assert first.record["parsing_prediction_sha256"] is None
    assert first.record["association"] is None
    assert first.record["instance_index"] is None
    assert first.record["parsing_quality_state"] is None
    assert first.record["source_crop_box_xyxy"] == [1, 0, 11, 8]
    verify_full_crop_artifacts(first)
    with (
        Image.open(io.BytesIO(first.full_rgb_png)) as rgb,
        Image.open(io.BytesIO(first.full_mask_png)) as output_mask,
    ):
        rgb_values = np.asarray(rgb)
        mask_values = np.asarray(output_mask)
        assert set(np.unique(mask_values)) == {0, 255}
        assert np.all(rgb_values[mask_values == 0] == (101, 102, 103))

    observation = build_body_mask_observation(
        source_id="oxford-pet",
        source_sha256=source_sha256,
        source_width=12,
        source_height=8,
        source_view_scope=SourceViewScope.BODY_AVAILABLE,
        authoritative_mask_sha256=mask_sha256,
        mask_policy_sha256=policy.policy_sha256,
        full_rgb_sha256=first.record["full_rgb_sha256"],
        face_observability=TerminalObservability.NOT_DETECTED,
    )
    assert observation.full_status is FullStatus.USABLE
    assert observation.parsing_prediction_sha256 is None
    assert observation.association is None
    assert observation.roles[0].producer_sha256 == policy.policy_sha256
    assert FullSegmentObservation.from_dict(observation.to_dict()) == observation

    truncated = build_body_mask_observation(
        source_id="oxford-pet-truncated",
        source_sha256=source_sha256,
        source_width=12,
        source_height=8,
        source_view_scope=SourceViewScope.BODY_TRUNCATED,
        authoritative_mask_sha256=mask_sha256,
        mask_policy_sha256=policy.policy_sha256,
        full_rgb_sha256=first.record["full_rgb_sha256"],
    )
    assert truncated.full_status is FullStatus.REVIEW


def test_authoritative_body_mask_rejects_tamper_invalid_content_and_empty_support() -> (
    None
):
    source = _source_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    policy = _body_mask_policy()
    mask = _mask_bytes()
    with pytest.raises(ValueError, match="mask digest"):
        materialize_body_mask_full_crop(
            source,
            mask,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=_sha("substituted-mask"),
            policy=policy,
        )

    invalid = np.full((8, 12), 2, dtype=np.uint8)
    invalid[2:7, 3:9] = 1
    invalid[0, 0] = 4
    invalid_bytes = _mask_bytes(invalid)
    with pytest.raises(ValueError, match="invalid Oxford trimap labels"):
        materialize_body_mask_full_crop(
            source,
            invalid_bytes,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=hashlib.sha256(
                invalid_bytes
            ).hexdigest(),
            policy=policy,
        )

    all_background = _mask_bytes(np.full((8, 12), 2, dtype=np.uint8))
    with pytest.raises(ValueError, match="all-background"):
        materialize_body_mask_full_crop(
            source,
            all_background,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=hashlib.sha256(
                all_background
            ).hexdigest(),
            policy=_body_mask_policy((1, 2)),
        )

    missing_label = np.full((8, 12), 2, dtype=np.uint8)
    missing_label[2:7, 3:9] = 1
    missing_label_bytes = _mask_bytes(missing_label)
    with pytest.raises(ValueError, match="missing expected Oxford labels"):
        materialize_body_mask_full_crop(
            source,
            missing_label_bytes,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=hashlib.sha256(
                missing_label_bytes
            ).hexdigest(),
            policy=policy,
        )
    materialize_body_mask_full_crop(
        source,
        missing_label_bytes,
        expected_source_sha256=source_sha256,
        expected_authoritative_mask_sha256=hashlib.sha256(
            missing_label_bytes
        ).hexdigest(),
        policy=_body_mask_policy((1, 2)),
    )


def test_authoritative_body_mask_rejects_size_non_image_and_multichannel() -> None:
    source = _source_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    policy = _body_mask_policy()
    wrong_size = _mask_bytes(np.ones((7, 12), dtype=np.uint8))
    with pytest.raises(ValueError, match="dimensions differ"):
        materialize_body_mask_full_crop(
            source,
            wrong_size,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=hashlib.sha256(wrong_size).hexdigest(),
            policy=policy,
        )
    with pytest.raises(ValueError, match="not a supported image"):
        materialize_body_mask_full_crop(
            source,
            b"not-an-image",
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=hashlib.sha256(
                b"not-an-image"
            ).hexdigest(),
            policy=policy,
        )
    with pytest.raises(ValueError, match="single-channel"):
        materialize_body_mask_full_crop(
            source,
            source,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=source_sha256,
            policy=policy,
        )


def test_cache_binds_observation_parsing_and_crop() -> None:
    source = _source_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    frozen = freeze_animal_parsing_prediction(_prediction())
    crop = materialize_full_crop(
        source,
        expected_source_sha256=source_sha256,
        frozen_parsing=frozen,
        association=_exactly_one(),
    )
    observation = build_body_observation(
        source_id="sample",
        source_sha256=source_sha256,
        source_view_scope=SourceViewScope.BODY_AVAILABLE,
        frozen_parsing=frozen,
        association=_exactly_one(),
        full_rgb_sha256=crop.record["full_rgb_sha256"],
    )
    bundle = build_full_segment_cache(
        (
            {
                "source_id": "sample",
                "observation": observation.to_dict(),
                "frozen_parsing": frozen,
                "crop": crop.record,
            },
        )
    )
    validate_full_segment_cache_bundle(bundle)
    changed = deepcopy(bundle)
    changed["cache"]["records"][0]["source_id"] = "substituted"
    changed["cache_sha256"] = content_sha256(changed["cache"])
    with pytest.raises(ValueError, match="differs from observation"):
        validate_full_segment_cache_bundle(changed)

    missing = deepcopy(bundle)
    missing["cache"]["records"][0]["crop"] = None
    missing["cache_sha256"] = content_sha256(missing["cache"])
    with pytest.raises(ValueError, match="missing Full crop"):
        validate_full_segment_cache_bundle(missing)


def test_cache_binds_authoritative_mask_policy_crop_and_artifact_hashes() -> None:
    source = _source_bytes()
    mask = _mask_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    mask_sha256 = hashlib.sha256(mask).hexdigest()
    policy = _body_mask_policy()
    crop = materialize_body_mask_full_crop(
        source,
        mask,
        expected_source_sha256=source_sha256,
        expected_authoritative_mask_sha256=mask_sha256,
        policy=policy,
    )
    observation = build_body_mask_observation(
        source_id="oxford-pet",
        source_sha256=source_sha256,
        source_width=12,
        source_height=8,
        source_view_scope=SourceViewScope.BODY_AVAILABLE,
        authoritative_mask_sha256=mask_sha256,
        mask_policy_sha256=policy.policy_sha256,
        full_rgb_sha256=crop.record["full_rgb_sha256"],
    )
    bundle = build_full_segment_cache(
        (
            {
                "source_id": "oxford-pet",
                "observation": observation.to_dict(),
                "frozen_parsing": None,
                "crop": crop.record,
            },
        )
    )
    validate_full_segment_cache_bundle(bundle)

    changed = deepcopy(bundle)
    changed_crop = changed["cache"]["records"][0]["crop"]
    changed_crop["authoritative_mask_sha256"] = _sha("substituted-mask")
    changed_crop["crop_record_sha256"] = content_sha256(
        {key: item for key, item in changed_crop.items() if key != "crop_record_sha256"}
    )
    changed["cache_sha256"] = content_sha256(changed["cache"])
    with pytest.raises(ValueError, match="cache authoritative mask differs"):
        validate_full_segment_cache_bundle(changed)

    with pytest.raises(ValueError, match="full_mask artifact byte size differs"):
        verify_full_crop_artifacts(
            crop.record,
            crop.full_rgb_png,
            crop.full_mask_png + b"tampered",
        )


def test_single_record_workflow_materializes_real_image_and_frozen_prediction(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    source_path.write_bytes(_source_bytes())
    frozen_path = tmp_path / "frozen.json"
    frozen = freeze_animal_parsing_prediction(_prediction())
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    request = {
        "schema_version": REQUEST_SCHEMA,
        "source_id": "real-compatible-sample",
        "source_image_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_view_scope": "BODY_AVAILABLE",
        "route": "BODY_PARSING",
        "frozen_parsing_path": str(frozen_path),
        "association": _exactly_one().to_dict(),
        "face_observability": "NOT_RUN",
        "nose_observability": "NOT_DETECTED",
        "target_size": 40,
        "context_fraction": 0.1,
        "background_rgb": [127, 127, 127],
    }
    output = tmp_path / "output"
    bundle = run(request, output_dir=output)
    assert sorted(path.name for path in output.iterdir()) == [
        "full-mask.png",
        "full-segment-cache.json",
        "full-segment-observation.json",
        "full.png",
    ]
    assert json.loads((output / "full-segment-cache.json").read_text()) == bundle
    validate_full_segment_cache_bundle(bundle)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(request, output_dir=output)


def test_prevalidated_workflow_is_byte_identical_to_file_backed_workflow(
    tmp_path: Path,
) -> None:
    from foundation.protected_io import json_document_bytes
    from workflows.materialize_full_segment import run_prevalidated

    source = _source_bytes()
    source_path = tmp_path / "source.png"
    source_path.write_bytes(source)
    frozen = freeze_animal_parsing_prediction(_prediction())
    frozen_bytes = json_document_bytes(frozen)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_bytes(frozen_bytes)
    request = {
        "schema_version": REQUEST_SCHEMA,
        "source_id": "prevalidated-parity",
        "source_image_path": str(source_path),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_view_scope": "BODY_AVAILABLE",
        "route": "BODY_PARSING",
        "frozen_parsing_path": str(frozen_path),
        "association": _exactly_one().to_dict(),
        "face_observability": "NOT_RUN",
        "nose_observability": "NOT_RUN",
        "target_size": 40,
        "context_fraction": 0.1,
        "background_rgb": [127, 127, 127],
    }
    file_backed = tmp_path / "file-backed"
    prevalidated = tmp_path / "prevalidated"

    run(request, output_dir=file_backed)
    run_prevalidated(
        request,
        output_dir=prevalidated,
        source_bytes=source,
        frozen_parsing=frozen,
        frozen_json_sha256=hashlib.sha256(frozen_bytes).hexdigest(),
    )

    assert {path.name: path.read_bytes() for path in file_backed.iterdir()} == {
        path.name: path.read_bytes() for path in prevalidated.iterdir()
    }


def test_single_record_workflow_materializes_authoritative_body_mask(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.png"
    source_path.write_bytes(_source_bytes())
    mask_path = tmp_path / "trimap.png"
    mask_path.write_bytes(_mask_bytes())
    policy = _body_mask_policy()
    request = {
        "schema_version": REQUEST_SCHEMA,
        "source_id": "oxford-pet-sample",
        "source_image_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_view_scope": "BODY_AVAILABLE",
        "route": "BODY_MASK",
        "frozen_parsing_path": None,
        "association": None,
        "authoritative_mask_path": str(mask_path),
        "authoritative_mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "body_mask_policy": policy.to_dict(),
        "face_observability": "NOT_DETECTED",
        "nose_observability": "NOT_RUN",
        "target_size": 40,
        "context_fraction": 0.1,
        "background_rgb": [127, 127, 127],
    }
    output = tmp_path / "body-mask-output"
    first = run(request, output_dir=output)
    assert sorted(path.name for path in output.iterdir()) == [
        "full-mask.png",
        "full-segment-cache.json",
        "full-segment-observation.json",
        "full.png",
    ]
    assert json.loads((output / "full-segment-cache.json").read_text()) == first
    record = first["cache"]["records"][0]
    observation = FullSegmentObservation.from_dict(record["observation"])
    assert observation.route is ObservationRoute.BODY_MASK
    assert observation.full_status is FullStatus.USABLE
    assert observation.authoritative_mask_sha256 == request["authoritative_mask_sha256"]
    assert observation.mask_policy_sha256 == policy.policy_sha256
    assert observation.parsing_prediction_sha256 is None
    assert observation.association is None
    assert record["frozen_parsing"] is None
    verify_full_crop_artifacts(
        record["crop"],
        (output / "full.png").read_bytes(),
        (output / "full-mask.png").read_bytes(),
    )

    missing_field = deepcopy(request)
    del missing_field["body_mask_policy"]
    with pytest.raises(ValueError, match="request schema differs"):
        run(missing_field, output_dir=tmp_path / "invalid-request-output")


def test_body_mask_workflow_rejects_missing_and_symlink_mask(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    source_path.write_bytes(_source_bytes())
    missing_path = tmp_path / "missing-trimap.png"
    request = {
        "schema_version": REQUEST_SCHEMA,
        "source_id": "missing-mask-sample",
        "source_image_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_view_scope": "BODY_AVAILABLE",
        "route": "BODY_MASK",
        "frozen_parsing_path": None,
        "association": None,
        "authoritative_mask_path": str(missing_path),
        "authoritative_mask_sha256": _sha("missing-mask"),
        "body_mask_policy": _body_mask_policy().to_dict(),
        "face_observability": "NOT_RUN",
        "nose_observability": "NOT_RUN",
        "target_size": 40,
        "context_fraction": 0.1,
        "background_rgb": [127, 127, 127],
    }
    with pytest.raises(
        FileNotFoundError, match="authoritative body mask does not exist"
    ):
        run(request, output_dir=tmp_path / "missing-output")

    real_mask_path = tmp_path / "real-trimap.png"
    real_mask_path.write_bytes(_mask_bytes())
    symlink_path = tmp_path / "linked-trimap.png"
    symlink_path.symlink_to(real_mask_path)
    request["authoritative_mask_path"] = str(symlink_path)
    request["authoritative_mask_sha256"] = hashlib.sha256(
        real_mask_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="absolute non-symlink file"):
        run(request, output_dir=tmp_path / "symlink-output")


def test_regular_file_read_detects_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trimap.png"
    path.write_bytes(_mask_bytes())
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(_mask_bytes(np.ones((8, 12), dtype=np.uint8)))
    original_inode = path.stat().st_ino
    original_read = materialize_workflow.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, size)
        if (
            payload
            and not replaced
            and materialize_workflow.os.fstat(descriptor).st_ino == original_inode
        ):
            materialize_workflow.os.replace(replacement, path)
            replaced = True
        return payload

    monkeypatch.setattr(materialize_workflow.os, "read", replacing_read)
    with pytest.raises(RuntimeError, match="changed while being read"):
        materialize_workflow._read_regular_file(
            path,
            maximum_bytes=1_000_000,
            label="authoritative body mask",
        )


def test_single_record_workflow_materializes_native_full_input(tmp_path: Path) -> None:
    source_path = tmp_path / "native.png"
    source_path.write_bytes(_source_bytes())
    request = {
        "schema_version": REQUEST_SCHEMA,
        "source_id": "native-face-sample",
        "source_image_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_view_scope": "FACE_NATIVE",
        "route": "NATIVE_FACE",
        "frozen_parsing_path": None,
        "association": None,
        "face_observability": "NATIVE",
        "nose_observability": "NOT_DETECTED",
        "target_size": 40,
        "context_fraction": 0.1,
        "background_rgb": [127, 127, 127],
    }

    output = tmp_path / "native-output"
    bundle = run(request, output_dir=output)
    assert sorted(path.name for path in output.iterdir()) == [
        "full-mask.png",
        "full-segment-cache.json",
        "full-segment-observation.json",
        "full.png",
    ]
    observation = FullSegmentObservation.from_dict(
        bundle["cache"]["records"][0]["observation"]
    )
    assert observation.full_status is FullStatus.USABLE
    assert (
        observation.roles[0].artifact_sha256
        == bundle["cache"]["records"][0]["crop"]["full_rgb_sha256"]
    )
