from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from data.source_lock import get_record
from data.types import CaptureGroupKind, UnifiedCanidSample
from data.full_segment import route_plan
from data.full_segment.route_plan import CANONICAL_DATASETS, build_full128_route_plan
from shared.foundation.protected_io import json_document_bytes
from shared.foundation.provenance import content_sha256
from enrollment.registry.generated_identity_registry import GENERATED_DOG_NAMESPACE
from enrollment.registry.identity_registry import compute_registered_dog_id
from archive.full128.methods.preparation import inventory, materialization
from archive.full128.methods.preparation.inventory import (
    validate_full128_experiment_inventory_bundle,
)
from archive.full128.methods.preparation.materialization import (
    BoundParserRuntime,
    assemble_full128_materialization,
    materialize_full128_route_plan,
    migrate_full128_compact_sample_caches,
)
from parsing.export.segmentation.animal_parsing import (
    AnimalParsingPrediction,
    ParsedAnimalInstance,
    ParsedAnimalQuality,
)
from parsing.export.segmentation.full_segment_cache import (
    CACHE_SCHEMA,
    FROZEN_PARSING_BINDING_SCHEMA,
    LEGACY_CACHE_BUNDLE_SCHEMA,
    LEGACY_CACHE_SCHEMA,
)
from archive.full128.commands.materialize_full128_route_plan import main as materialization_main

_RUNTIME_SHA = hashlib.sha256(b"full128-fake-parser-runtime").hexdigest()
_POLICY_SHA = hashlib.sha256(b"full128-fake-parser-policy").hexdigest()

def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()

def _write_image(
    root: Path,
    relative: str,
    *,
    color: int,
    size: tuple[int, int] = (32, 24),
) -> tuple[str, int, int]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(color, color, color)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), *size

def _write_trimap(
    root: Path,
    relative: str,
    values: np.ndarray,
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)

def _sample(
    dataset: str,
    root: Path,
    relative: str,
    *,
    label: str,
    color: int,
    split: str = "train",
    registered: bool = False,
    raw_identity: str | None = None,
    face_box: tuple[float, float, float, float] | None = None,
    dog_box: tuple[float, float, float, float] | None = None,
    head_box: tuple[float, float, float, float] | None = None,
    trimap: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> UnifiedCanidSample:
    digest, width, height = _write_image(root, relative, color=color)
    identity = raw_identity or label
    dataset_identity = {
        "dogfacenet224": f"dogfacenet224:v1:web-folder:{identity}",
        "mpdd": f"mpdd:v1:device-capture:{identity}",
        "sibetan": f"sibetan:v1:gt-json:{identity}",
        "yt-bb-dog": f"yt-bb-dog:v1:video-track:{identity}",
    }.get(dataset)
    return UnifiedCanidSample(
        sample_id=_sha(f"sample:{label}"),
        dataset_name=dataset,
        dataset_version=get_record(dataset).version,
        source_group_id=f"source:{identity}",
        image_path=relative,
        image_sha256=digest,
        width=width,
        height=height,
        registered_identity_id=(
            compute_registered_dog_id(dataset_identity) if registered else None
        ),
        raw_identity_id=identity if registered else None,
        face_box_xyxy=face_box,
        dog_boxes_xyxy=dog_box,
        head_roi_xyxy=head_box,
        foreground_mask_path=trimap,
        capture_group_id=f"capture:{identity}",
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        split_role=split,
        metadata={} if metadata is None else metadata,
    )

def _instance(
    index: int,
    *,
    class_name: str = "dog",
    box: tuple[int, int, int, int],
    state: str = "USABLE",
    touches_border: bool = False,
) -> ParsedAnimalInstance:
    height, width = 24, 32
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    probability = mask.astype(np.float32) * np.float32(0.9)
    return ParsedAnimalInstance(
        instance_index=index,
        query_index=index,
        class_id=17 if class_name == "dog" else 16,
        class_name=class_name,
        class_score=0.95,
        detector_box_xyxy=box,
        refinement_box_xyxy=box,
        mask_box_xyxy=box,
        instance_probability=probability.copy(),
        foreground_probability=probability.copy(),
        ownership_probability=probability.copy(),
        hard_mask=mask,
        quality=ParsedAnimalQuality(
            state=state,
            reasons=() if state != "UNUSABLE" else ("FIXTURE_UNUSABLE",),
            flags=("TOUCHES_SOURCE_BORDER",) if state == "REVIEW" else (),
            semantic_shape_iou=0.9,
            ownership_retention=0.95,
            foreground_pixels=int(mask.sum()),
            component_count=1,
            touches_source_border=touches_border,
        ),
    )

class _FakeParser:
    def __init__(self) -> None:
        self.calls: Counter[int] = Counter()

    def predict(self, image: Image.Image) -> AnimalParsingPrediction:
        color = int(image.convert("RGB").getpixel((0, 0))[0])
        self.calls[color] += 1
        if color == 10:
            instances = (
                _instance(0, box=(1, 1, 11, 11)),
                _instance(1, box=(18, 2, 30, 15)),
            )
        elif color == 60:
            instances = (
                _instance(0, box=(1, 1, 10, 12)),
                _instance(1, box=(18, 2, 30, 14)),
            )
        elif color == 70:
            instances = (_instance(0, class_name="cat", box=(4, 3, 20, 18)),)
        elif color == 40:
            instances = (
                _instance(
                    0,
                    box=(0, 2, 18, 23),
                    state="REVIEW",
                    touches_border=True,
                ),
            )
        elif color == 20:
            instances = (
                _instance(0, box=(1, 1, 8, 8)),
                _instance(1, box=(10, 2, 30, 22)),
            )
        elif color == 30:
            instances = (
                _instance(0, class_name="cat", box=(1, 1, 8, 8)),
                _instance(1, box=(10, 2, 30, 22)),
            )
        else:
            instances = (_instance(0, box=(3, 2, 25, 21)),)
        return AnimalParsingPrediction(
            source_width=image.width,
            source_height=image.height,
            instances=instances,
            policy_sha256=_POLICY_SHA,
        )

    def predict_batch(
        self,
        images: tuple[Image.Image, ...],
        *,
        instance_batch_size: int,
        foreground_batch_size: int,
    ) -> tuple[AnimalParsingPrediction, ...]:
        assert instance_batch_size > 0
        assert foreground_batch_size > 0
        return tuple(self.predict(image) for image in images)

def _bound_runtime(fake: _FakeParser) -> BoundParserRuntime:
    return BoundParserRuntime(
        runtime=fake,
        parser_runtime_manifest_sha256=_RUNTIME_SHA,
        parser_runtime_bundle_raw_sha256=_sha("runtime-bundle"),
        parser_policy_sha256=_POLICY_SHA,
        foreground_model_manifest_sha256=_sha("foreground-manifest"),
        foreground_model_bundle_raw_sha256=_sha("foreground-bundle"),
        instance_model_manifest_sha256=_sha("instance-manifest"),
        instance_model_bundle_raw_sha256=_sha("instance-bundle"),
        device="cpu",
    )

@pytest.fixture
def batch_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Path], _FakeParser]:
    roots = {name: tmp_path / f"dataset-{name}" for name in CANONICAL_DATASETS}
    for root in roots.values():
        root.mkdir()
    records = {
        name: replace(get_record(name), data_root=str(roots[name]))
        for name in CANONICAL_DATASETS
    }
    real_get_record = get_record

    def fixture_record(name: str) -> Any:
        return records.get(name, real_get_record(name))

    monkeypatch.setattr(route_plan, "get_record", fixture_record)
    monkeypatch.setattr(materialization, "get_record", fixture_record)
    monkeypatch.setattr(inventory, "get_record", fixture_record)

    ap_root = roots["ap10k-dog"]
    ap_first = _sample(
        "ap10k-dog",
        ap_root,
        "ap-10k/data/shared.png",
        label="ap-first",
        color=10,
        dog_box=(1.0, 1.0, 11.0, 11.0),
        metadata={"annotation_id": 101, "image_id": 7},
    )
    ap_second = replace(
        ap_first,
        sample_id=_sha("sample:ap-second"),
        dog_boxes_xyxy=(18.0, 2.0, 30.0, 15.0),
        metadata={"annotation_id": 102, "image_id": 7},
    )
    annotation_path = ap_root / "ap-10k/annotations/ap10k-train-split1.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "shared.png"}],
                "annotations": [
                    {
                        "id": 101,
                        "image_id": 7,
                        "category_id": 8,
                        "bbox": [1, 1, 10, 10],
                    },
                    {
                        "id": 102,
                        "image_id": 7,
                        "category_id": 8,
                        "bbox": [18, 2, 12, 13],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    dogflw_root = roots["dogflw"]
    dogflw_face = _sample(
        "dogflw",
        dogflw_root,
        "DogFLW/train/images/face.png",
        label="dogflw-face",
        color=20,
        face_box=(2.2, 1.1, 25.4, 20.2),
    )
    dogflw_parse = _sample(
        "dogflw",
        dogflw_root,
        "DogFLW/train/images/parse.png",
        label="dogflw-parse",
        color=21,
    )
    label_root = dogflw_root / "DogFLW/train/labels"
    label_root.mkdir(parents=True)
    (label_root / "face.json").write_text(
        json.dumps({"bounding_boxes": [2.2, 1.1, 25.4, 20.2]}),
        encoding="utf-8",
    )
    (label_root / "parse.json").write_text(
        json.dumps({"bounding_boxes": ["", 1, 20, 20]}),
        encoding="utf-8",
    )

    dogface = _sample(
        "dogfacenet224",
        roots["dogfacenet224"],
        "after_4_bis/17/native.png",
        label="dogface",
        color=30,
        split="UNASSIGNED",
        registered=True,
        raw_identity="17",
    )
    mpdd = _sample(
        "mpdd",
        roots["mpdd"],
        "MPDD/pytorch/query/1_c1_s1_1.png",
        label="mpdd",
        color=40,
        split="query",
        registered=True,
        raw_identity="1",
        metadata={
            "unverified_camera_token": "c1",
            "unverified_sequence_token": "s1",
        },
    )

    oxford_root = roots["oxford-pets-dog"]
    oxford_mask = _sample(
        "oxford-pets-dog",
        oxford_root,
        "images/beagle_1.jpg",
        label="oxford-mask",
        color=50,
        split="trainval",
        trimap="annotations/trimaps/beagle_1.png",
    )
    trimap = np.full((24, 32), 2, dtype=np.uint8)
    trimap[0:18, 3:24] = 1
    trimap[18:, :] = 3
    _write_trimap(oxford_root, "annotations/trimaps/beagle_1.png", trimap)
    oxford_head = _sample(
        "oxford-pets-dog",
        oxford_root,
        "images/beagle_2.jpg",
        label="oxford-head",
        color=51,
        split="test",
        head_box=(3.0, 2.0, 25.0, 21.0),
        trimap="annotations/trimaps/beagle_2.png",
    )
    _write_trimap(
        oxford_root,
        "annotations/trimaps/beagle_2.png",
        np.full((24, 32), 2, dtype=np.uint8),
    )
    xml_path = oxford_root / "annotations/xmls/beagle_2.xml"
    xml_path.parent.mkdir(parents=True)
    xml_path.write_text(
        "<annotation><filename>beagle_2.jpg</filename><object><name>dog</name>"
        "<bndbox><xmin>3</xmin><ymin>2</ymin><xmax>25</xmax><ymax>21</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    sibetan = _sample(
        "sibetan",
        roots["sibetan"],
        "Sibetan/1/body.png",
        label="sibetan",
        color=60,
        split="UNASSIGNED",
        registered=True,
        raw_identity="dog-1",
    )
    yt = _sample(
        "yt-bb-dog",
        roots["yt-bb-dog"],
        "YT-BB-dog/YT-BB-Dog/test/track-9/frame.png",
        label="yt",
        color=70,
        split="test",
        registered=True,
        raw_identity="track-9",
    )
    samples = {
        "ap10k-dog": (ap_second, ap_first),
        "dogflw": (dogflw_parse, dogflw_face),
        "dogfacenet224": (dogface,),
        "mpdd": (mpdd,),
        "oxford-pets-dog": (oxford_head, oxford_mask),
        "sibetan": (sibetan,),
        "yt-bb-dog": (yt,),
    }
    bundle = build_full128_route_plan(
        parser_runtime_manifest_sha256=_RUNTIME_SHA,
        parser_policy_sha256=_POLICY_SHA,
        samples_by_dataset=samples,
    )
    return bundle, roots, _FakeParser()

def _sample_receipt(output: Path, token: str) -> dict[str, Any]:
    return json.loads(
        (output / "samples" / token / "execution-receipt.json").read_text(
            encoding="utf-8"
        )
    )

def _replace_cache_and_resign(sample_dir: Path, cache_bundle: dict[str, Any]) -> None:
    cache_bytes = json_document_bytes(cache_bundle)
    (sample_dir / "full-segment-cache.json").write_bytes(cache_bytes)
    receipt_path = sample_dir / "execution-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["outputs"]["full_segment_cache_file_sha256"] = hashlib.sha256(
        cache_bytes
    ).hexdigest()
    receipt["outputs"]["full_segment_cache_sha256"] = cache_bundle["cache_sha256"]
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = content_sha256(payload)
    receipt_path.write_bytes(json_document_bytes(receipt))

def _make_legacy_sample(output: Path, row: dict[str, Any]) -> Path:
    sample_dir = output / "samples" / row["sample_token"]
    cache_path = sample_dir / "full-segment-cache.json"
    cache_bundle = json.loads(cache_path.read_text(encoding="utf-8"))
    frozen = json.loads(
        (output / "parser-cache" / row["parser_cache_key"] / "frozen.json").read_text(
            encoding="utf-8"
        )
    )
    cache_bundle["schema_version"] = LEGACY_CACHE_BUNDLE_SCHEMA
    cache_bundle["cache"]["schema_version"] = LEGACY_CACHE_SCHEMA
    cache_bundle["cache"]["records"][0]["frozen_parsing"] = frozen
    cache_bundle["cache_sha256"] = content_sha256(cache_bundle["cache"])
    _replace_cache_and_resign(sample_dir, cache_bundle)
    return sample_dir

def _decision_fixture(
    dataset: str, instances: tuple[ParsedAnimalInstance, ...]
) -> Any:
    row = {
        "schema_version": "archive.full128.route_plan_record.v3",
        "sample_token": _sha(f"decision:{dataset}"),
        "dataset_name": dataset,
        "record_sha256": _sha(f"record:{dataset}"),
    }
    prediction = AnimalParsingPrediction(
        source_width=32,
        source_height=24,
        instances=instances,
        policy_sha256=_POLICY_SHA,
    )
    cache_receipt = {
        "parser_cache_key": _sha("parser-cache"),
        "receipt_sha256": _sha("parser-receipt"),
        "prediction_sha256": _sha("prediction"),
        "runtime": {"parser_policy_sha256": _POLICY_SHA},
    }
    return materialization._parser_decisions(
        (row,), prediction=prediction, cache_receipt=cache_receipt
    )[row["sample_token"]]

def test_auxiliary_selection_fails_when_all_dogs_are_unusable() -> None:
    decision = _decision_fixture(
        "dogflw", (_instance(0, box=(1, 1, 12, 12), state="UNUSABLE"),)
    )
    assert decision.terminal_reason == "NO_VALID_PARSED_DOG_INSTANCE"
    assert decision.parser_lineage["selection"]["post_suppression_dog_count"] == 1
    assert decision.parser_lineage["selection"][
        "valid_post_suppression_dog_count"
    ] == 0

def test_auxiliary_selection_breaks_equal_area_ties_by_instance_index() -> None:
    decision = _decision_fixture(
        "oxford-pets-dog",
        (
            _instance(0, box=(1, 1, 6, 6)),
            _instance(1, box=(10, 1, 15, 6)),
        ),
    )
    assert decision.association.instance_index == 0
    assert decision.parser_lineage["selection"]["selected_foreground_pixels"] == 25

def test_identity_selection_rejects_sole_unusable_dog() -> None:
    decision = _decision_fixture(
        "mpdd", (_instance(0, box=(1, 1, 12, 12), state="UNUSABLE"),)
    )
    assert decision.terminal_reason == "SELECTED_DOG_PARSING_UNUSABLE"
    assert decision.parser_lineage["selection"]["selected_instance_count"] == 1

def test_all_routes_ap_global_matching_terminals_and_resume(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-output"
    summary = materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    assert summary["created_sample_count"] == 10
    assert summary["terminal_sample_count"] == 2
    assert sum(fake.calls.values()) == 9

    compact_records = []
    for row in bundle["plan"]["records"]:
        receipt = _sample_receipt(output, row["sample_token"])
        if receipt["actual_route"] != "BODY_PARSING":
            continue
        cache = json.loads(
            (
                output / "samples" / row["sample_token"] / "full-segment-cache.json"
            ).read_text(encoding="utf-8")
        )
        compact_records.append(cache["cache"]["records"][0]["frozen_parsing"])
        assert cache["cache"]["schema_version"] == CACHE_SCHEMA
    assert compact_records
    assert all(
        record["schema_version"] == FROZEN_PARSING_BINDING_SCHEMA
        and "prediction" not in record
        for record in compact_records
    )

    rows = bundle["plan"]["records"]
    receipts = {
        row["sample_token"]: _sample_receipt(output, row["sample_token"])
        for row in rows
    }
    assert {receipt["route_intent"] for receipt in receipts.values()} == {
        "BODY_PARSING"
    }
    for dataset in ("dogfacenet224", "dogflw", "oxford-pets-dog"):
        dataset_receipts = [
            receipts[row["sample_token"]]
            for row in rows
            if row["dataset_name"] == dataset
        ]
        assert all(
            receipt["actual_route"] == "BODY_PARSING" for receipt in dataset_receipts
        )
        assert all(
            receipt["parser_lineage"] is not None for receipt in dataset_receipts
        )
        assert all(receipt["derived_lineage"] is None for receipt in dataset_receipts)
    ap_rows = [row for row in rows if row["dataset_name"] == "ap10k-dog"]
    ap_receipts = [receipts[row["sample_token"]] for row in ap_rows]
    assert len({row["parser_cache_key"] for row in ap_rows}) == 1
    assert {
        receipt["parser_lineage"]["association"]["instance_index"]
        for receipt in ap_receipts
    } == {0, 1}
    assert all(
        receipt["parser_lineage"]["association"]["kind"] == "AUTHORITATIVE"
        and receipt["parser_lineage"]["association_authority"]["match_iou"] >= 0.5
        for receipt in ap_receipts
    )

    by_dataset = {row["dataset_name"]: receipts[row["sample_token"]] for row in rows}
    assert by_dataset["sibetan"]["terminal_reason"] == (
        "PARSER_DISTINCT_DOG_CARDINALITY_AMBIGUOUS"
    )
    assert by_dataset["yt-bb-dog"]["terminal_reason"] == "NO_PARSED_DOG_INSTANCE"
    for dataset in ("sibetan", "yt-bb-dog"):
        directory = output / "samples" / by_dataset[dataset]["sample_token"]
        assert {path.suffix for path in directory.iterdir()} == {".json"}
    mpdd_receipt = by_dataset["mpdd"]
    mpdd_observation = json.loads(
        (
            output
            / "samples"
            / mpdd_receipt["sample_token"]
            / "full-segment-observation.json"
        ).read_text(encoding="utf-8")
    )
    assert mpdd_observation["source_view_scope"] == "BODY_TRUNCATED"
    assert mpdd_observation["full_status"] == "REVIEW"
    dogface_lineage = by_dataset["dogfacenet224"]["parser_lineage"]
    assert dogface_lineage["association"]["instance_index"] == 1
    assert dogface_lineage["selection"] == {
        "schema_version": "archive.full128.parser_selection_lineage.v1",
        "rule": "REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG",
        "prediction_instance_count": 2,
        "post_suppression_dog_count": 1,
        "valid_post_suppression_dog_count": 1,
        "selected_instance_count": 1,
        "selected_instance_index": 1,
        "selected_foreground_pixels": 400,
        "terminal_reason": None,
    }
    dogflw_receipt = next(
        receipts[row["sample_token"]]
        for row in rows
        if row["dataset_name"] == "dogflw" and row["source_path"].endswith("face.png")
    )
    assert dogflw_receipt["parser_lineage"]["association"]["instance_index"] == 1
    assert dogflw_receipt["parser_lineage"]["selection"][
        "post_suppression_dog_count"
    ] == 2

    calls_before_resume = fake.calls.copy()
    resumed = materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    assert resumed["created_sample_count"] == 0
    assert resumed["skipped_sample_count"] == 10
    assert fake.calls == calls_before_resume

def test_batched_materialization_is_fixed_ordered_and_resumable(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-batched"
    runtime = replace(
        _bound_runtime(fake),
        job_batch_size=4,
        instance_batch_size=4,
        foreground_batch_size=4,
        publication_workers=4,
    )

    summary = materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=runtime,
        verify_plan_files_upfront=False,
    )

    assert summary["created_sample_count"] == len(bundle["plan"]["records"])
    assert sum(fake.calls.values()) == 9
    calls_before_resume = fake.calls.copy()
    resumed = materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=runtime,
        verify_plan_files_upfront=False,
    )
    assert resumed["skipped_sample_count"] == len(bundle["plan"]["records"])
    assert fake.calls == calls_before_resume

    with pytest.raises(ValueError, match="align to the parser job batch size"):
        materialize_full128_route_plan(
            bundle,
            output_root=tmp_path / "unaligned-batch",
            parser_runtime=runtime,
            verify_plan_files_upfront=False,
            maximum_jobs=2,
        )

def test_batched_job_units_share_one_prediction_for_duplicate_source_bytes() -> None:
    from archive.full128.methods.preparation.materialization import _batched_job_units

    cache_key = _sha("shared-cache")
    jobs = (
        (cache_key, ({"sample_token": _sha("first")},)),
        (cache_key, ({"sample_token": _sha("second")},)),
        (_sha("other-cache"), ({"sample_token": _sha("other")},)),
    )

    units = _batched_job_units(jobs)

    assert len(units) == 2
    assert units[0] == jobs[:2]
    assert units[1] == jobs[2:]

def test_partial_outputs_source_and_artifact_tampering_fail_closed(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, roots, fake = batch_case
    output = tmp_path / "full128-tamper"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    parsed_face = next(
        row
        for row in bundle["plan"]["records"]
        if row["dataset_name"] == "dogfacenet224"
    )
    sample_dir = output / "samples" / parsed_face["sample_token"]
    full_path = sample_dir / "full.png"
    full_bytes = full_path.read_bytes()
    full_path.write_bytes(full_bytes + b"tampered")
    with pytest.raises(ValueError, match="fast-resume full.png digest differs"):
        materialize_full128_route_plan(
            bundle,
            output_root=output,
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
        )
    full_path.write_bytes(full_bytes)

    ap = next(
        row for row in bundle["plan"]["records"] if row["dataset_name"] == "ap10k-dog"
    )
    frozen_path = output / "parser-cache" / ap["parser_cache_key"] / "frozen.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen_path.write_bytes(frozen_bytes + b"tampered")
    with pytest.raises(ValueError, match="frozen JSON digest differs"):
        materialize_full128_route_plan(
            bundle,
            output_root=output,
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
        )
    frozen_path.write_bytes(frozen_bytes)

    cache_path = sample_dir / "full-segment-cache.json"
    cache_bytes = cache_path.read_bytes()
    cache_path.write_bytes(cache_bytes + b"tampered")
    with pytest.raises(ValueError, match="sample cache digest differs"):
        materialize_full128_route_plan(
            bundle,
            output_root=output,
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
        )
    cache_path.write_bytes(cache_bytes)

    (sample_dir / "execution-receipt.json").unlink()
    with pytest.raises(ValueError, match="partial|unexpected"):
        materialize_full128_route_plan(
            bundle,
            output_root=output,
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
        )

    source = roots["dogfacenet224"] / parsed_face["source_path"]
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="route-plan binding"):
        materialize_full128_route_plan(
            bundle,
            output_root=tmp_path / "source-tamper",
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
        )

def test_legacy_cache_resumes_fast_and_migrates_by_directory_exchange(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-legacy"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    row = next(
        row
        for row in bundle["plan"]["records"]
        if _sample_receipt(output, row["sample_token"])["actual_route"]
        == "BODY_PARSING"
    )
    sample_dir = _make_legacy_sample(output, row)
    legacy_size = (sample_dir / "full-segment-cache.json").stat().st_size

    resumed = materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    assert resumed["skipped_sample_count"] == len(bundle["plan"]["records"])
    assert (
        json.loads(
            (sample_dir / "full-segment-cache.json").read_text(encoding="utf-8")
        )["cache"]["schema_version"]
        == LEGACY_CACHE_SCHEMA
    )

    migrated = migrate_full128_compact_sample_caches(
        bundle,
        output_root=output,
        maximum_samples=1,
        verify_plan_files_upfront=False,
    )
    assert migrated["atomic_exchange_available"] is True
    assert migrated["migrated_sample_count"] == 1
    assert migrated["legacy_cache_bytes"] == legacy_size
    assert migrated["compact_cache_bytes"] < legacy_size
    assert (
        json.loads(
            (sample_dir / "full-segment-cache.json").read_text(encoding="utf-8")
        )["cache"]["schema_version"]
        == CACHE_SCHEMA
    )

def test_fast_resume_rejects_receipt_tamper_without_deep_decoding(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-fast"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    monkeypatch.setattr(
        materialization,
        "thaw_animal_parsing_prediction",
        lambda value: pytest.fail("fast resume decoded frozen parser arrays"),
    )
    monkeypatch.setattr(
        materialization,
        "_image_from_bytes",
        lambda payload, label: pytest.fail("fast resume decoded a PNG"),
    )
    resumed = materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    assert resumed["skipped_sample_count"] == len(bundle["plan"]["records"])

    row = bundle["plan"]["records"][0]
    receipt_path = output / "samples" / row["sample_token"] / "execution-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["plan_sha256"] = _sha("tampered-plan")
    receipt_path.write_bytes(json_document_bytes(receipt))
    with pytest.raises(ValueError, match="execution receipt digest differs"):
        materialize_full128_route_plan(
            bundle,
            output_root=output,
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
        )

def test_deep_assembly_rejects_resigned_compact_parser_binding_tamper(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-binding-tamper"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    row = next(
        row
        for row in bundle["plan"]["records"]
        if _sample_receipt(output, row["sample_token"])["actual_route"]
        == "BODY_PARSING"
    )
    sample_dir = output / "samples" / row["sample_token"]
    cache_bundle = json.loads(
        (sample_dir / "full-segment-cache.json").read_text(encoding="utf-8")
    )
    cache_bundle["cache"]["records"][0]["frozen_parsing"]["policy_sha256"] = _sha(
        "substituted-parser-policy"
    )
    cache_bundle["cache_sha256"] = content_sha256(cache_bundle["cache"])
    _replace_cache_and_resign(sample_dir, cache_bundle)

    with pytest.raises(ValueError, match="binding differs from parser cache"):
        assemble_full128_materialization(
            bundle,
            output_root=output,
            verify_plan_files_upfront=False,
        )

def test_assembly_reads_each_materialized_artifact_once(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-one-pass"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    strict_reads: Counter[Path] = Counter()
    regular_reads: Counter[Path] = Counter()
    real_strict_read = materialization.read_strict_json_document
    real_regular_read = materialization._read_regular_absolute
    real_inventory_read = inventory.read_strict_json_document

    def counted_strict_read(path: Path, **kwargs: Any) -> Any:
        strict_reads[path] += 1
        return real_strict_read(path, **kwargs)

    def counted_regular_read(path: Path, **kwargs: Any) -> bytes:
        regular_reads[path] += 1
        return real_regular_read(path, **kwargs)

    def reject_inventory_artifact_read(path: Path, **kwargs: Any) -> Any:
        if path.is_relative_to(output):
            pytest.fail(f"optimized inventory reopened {path.name}")
        return real_inventory_read(path, **kwargs)

    monkeypatch.setattr(
        materialization, "read_strict_json_document", counted_strict_read
    )
    monkeypatch.setattr(materialization, "_read_regular_absolute", counted_regular_read)
    monkeypatch.setattr(
        inventory, "read_strict_json_document", reject_inventory_artifact_read
    )

    assemble_full128_materialization(bundle, output_root=output)

    assert strict_reads
    assert regular_reads
    assert max(strict_reads.values()) == 1
    assert max(regular_reads.values()) == 1

def test_assembly_accepts_mixed_legacy_and_compact_shared_parser_caches(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-mixed-cache-schemas"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    shared_rows = [
        row for row in bundle["plan"]["records"] if row["dataset_name"] == "ap10k-dog"
    ]
    assert len(shared_rows) == 2
    assert len({row["parser_cache_key"] for row in shared_rows}) == 1
    _make_legacy_sample(output, shared_rows[0])

    assembly = assemble_full128_materialization(bundle, output_root=output)

    assert assembly["sample_count"] == len(bundle["plan"]["records"])
    cache_schemas = {
        json.loads(
            (
                output / "samples" / row["sample_token"] / "full-segment-cache.json"
            ).read_text(encoding="utf-8")
        )["cache"]["schema_version"]
        for row in shared_rows
    }
    assert cache_schemas == {CACHE_SCHEMA, LEGACY_CACHE_SCHEMA}

def test_assembly_detects_source_parser_crop_and_lineage_tamper(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, roots, fake = batch_case
    output = tmp_path / "full128-assembly-tamper"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    parser_row = next(
        row for row in bundle["plan"]["records"] if row["parser_cache_key"] is not None
    )
    crop_row = next(
        row
        for row in bundle["plan"]["records"]
        if _sample_receipt(output, row["sample_token"])["actual_route"] != "NONE"
    )
    assert all(
        _sample_receipt(output, row["sample_token"])["derived_lineage"] is None
        for row in bundle["plan"]["records"]
    )
    source_row = bundle["plan"]["records"][0]
    cases = (
        (
            roots[source_row["dataset_name"]] / source_row["source_path"],
            "route-plan binding",
        ),
        (
            output / "parser-cache" / parser_row["parser_cache_key"] / "frozen.json",
            "Extra data|frozen JSON",
        ),
        (
            output / "samples" / crop_row["sample_token"] / "full.png",
            "full_rgb artifact",
        ),
    )
    for path, match in cases:
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        try:
            with pytest.raises(ValueError, match=match):
                assemble_full128_materialization(bundle, output_root=output)
        finally:
            path.write_bytes(original)

def test_shards_are_stable_and_maximum_jobs_is_bounded(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-shards"
    for shard_index in range(3):
        materialize_full128_route_plan(
            bundle,
            output_root=output,
            parser_runtime=_bound_runtime(fake),
            verify_plan_files_upfront=False,
            shard_count=3,
            shard_index=shard_index,
        )
    assert {path.name for path in (output / "samples").iterdir()} == {
        row["sample_token"] for row in bundle["plan"]["records"]
    }
    for row in bundle["plan"]["records"]:
        selection = _sample_receipt(output, row["sample_token"])["shard_selection"]
        key = row["parser_cache_key"] or row["source_sha256"]
        assert selection["assigned_shard"] == int(key, 16) % 3
        assert selection["executed_shard"] == selection["assigned_shard"]

    bounded_output = tmp_path / "full128-bounded"
    bounded = materialize_full128_route_plan(
        bundle,
        output_root=bounded_output,
        parser_runtime=_bound_runtime(_FakeParser()),
        verify_plan_files_upfront=False,
        maximum_jobs=2,
    )
    assert bounded["selected_job_count"] == 2
    assert 1 <= len(tuple((bounded_output / "samples").iterdir())) <= 3

def test_complete_assembly_generates_identity_roles_and_inventory_lineage(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-assembly"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    progress: list[tuple[int, int]] = []
    assembly = assemble_full128_materialization(
        bundle,
        output_root=output,
        allocation_name="full128-fixture",
        verify_plan_files_upfront=False,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    assert progress == [
        (index, progress[-1][1]) for index in range(1, len(progress) + 1)
    ]
    inventory_bundle = assembly["inventory_bundle"]
    assert validate_full128_experiment_inventory_bundle(inventory_bundle) == (
        inventory_bundle
    )
    observations = assembly["unified_full_split"]["manifest"]["observations"]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_dataset.setdefault(observation["dataset_name"], []).append(observation)
    for dataset in ("ap10k-dog", "dogflw", "oxford-pets-dog"):
        assert all(
            row["identity_evidence_kind"] == "NONE" for row in by_dataset[dataset]
        )
        assert all(row["terminal_role"] == "AUXILIARY" for row in by_dataset[dataset])
    yt = by_dataset["yt-bb-dog"][0]
    assert yt["identity_evidence_kind"] == "GENERATED"
    assert yt["identity_namespace_uuid"] == str(GENERATED_DOG_NAMESPACE)
    assert yt["terminal_role"] == "EVAL"
    assert by_dataset["mpdd"][0]["terminal_role"] == "EVAL"
    assert by_dataset["sibetan"][0]["terminal_role"] == "EVAL"
    assert all(
        row["terminal_role"] != "FIT"
        for dataset in ("mpdd", "sibetan")
        for row in by_dataset[dataset]
    )
    assert all(
        not values
        for values in assembly["unified_full_split"]["census"][
            "overlap_report"
        ].values()
    )
    inventory_rows = inventory_bundle["inventory"]["records"]
    derived = [row for row in inventory_rows if row["lineage_receipt_path"] is not None]
    assert derived == []
    assert all(
        row["original_source_sha256"] == row["effective_source_sha256"]
        and row["lineage_receipt_sha256"] is None
        for row in inventory_rows
    )
    crop_record = next(row for row in inventory_rows if row["crop_artifacts_present"])
    crop_path = output / crop_record["full_rgb_path"]
    crop_bytes = crop_path.read_bytes()
    crop_path.write_bytes(crop_bytes + b"tampered")
    with pytest.raises(ValueError, match="full_rgb artifact"):
        validate_full128_experiment_inventory_bundle(inventory_bundle)

def test_assembly_workflow_progress_is_deterministic(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "full128-progress"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
    )
    route_plan_path = tmp_path / "route-plan.json"
    route_plan_path.write_bytes(json_document_bytes(bundle))
    assembly_path = tmp_path / "assembly.json"

    assert (
        materialization_main(
            [
                "assemble",
                "--route-plan",
                str(route_plan_path),
                "--output-root",
                str(output),
                "--output",
                str(assembly_path),
                "--progress-every-jobs",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    progress = [json.loads(line) for line in captured.err.splitlines()]
    assert progress
    assert all(row["status"] == "FULL128_ASSEMBLY_PROGRESS" for row in progress)
    total = progress[-1]["total_jobs"]
    assert [row["completed_jobs"] for row in progress] == [
        *range(2, total, 2),
        total,
    ]
    assert all(row["total_jobs"] == total for row in progress)

def test_assembly_requires_complete_coverage(
    batch_case: tuple[dict[str, Any], dict[str, Path], _FakeParser],
    tmp_path: Path,
) -> None:
    bundle, _, fake = batch_case
    output = tmp_path / "incomplete"
    materialize_full128_route_plan(
        bundle,
        output_root=output,
        parser_runtime=_bound_runtime(fake),
        verify_plan_files_upfront=False,
        maximum_jobs=1,
    )
    with pytest.raises(
        ValueError, match="complete parser-cache coverage|complete sample coverage"
    ):
        assemble_full128_materialization(
            bundle,
            output_root=output,
            verify_plan_files_upfront=False,
        )
