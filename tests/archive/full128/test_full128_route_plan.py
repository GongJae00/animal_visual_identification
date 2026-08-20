from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from data.source_lock import get_record
from data.types import CaptureGroupKind, UnifiedCanidSample
from data.full_segment import route_plan
from data.full_segment.route_plan import (
    CANONICAL_DATASETS,
    ROUTE_PLAN_BUNDLE_SCHEMA,
    ROUTE_PLAN_RECORD_SCHEMA,
    ROUTE_PLAN_SCHEMA,
    ROUTE_POLICY_SCHEMA,
    build_full128_route_plan,
    build_parser_cache_key,
    validate_full128_route_plan_bundle,
)
from archive.full128.commands import build_full128_route_plan as route_workflow

_RUNTIME_SHA = hashlib.sha256(b"parser-runtime").hexdigest()
_POLICY_SHA = hashlib.sha256(b"parser-policy").hexdigest()

def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()

def _write_image(
    root: Path,
    relative: str,
    *,
    color: int,
    mode: str = "RGB",
    size: tuple[int, int] = (20, 16),
) -> tuple[str, int, int]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), size[0], size[1]

def _sample(
    dataset: str,
    root: Path,
    relative: str,
    *,
    label: str,
    split: str = "train",
    face_box: tuple[float, float, float, float] | None = None,
    dog_box: tuple[float, float, float, float] | None = None,
    head_box: tuple[float, float, float, float] | None = None,
    trimap: str | None = None,
    metadata: dict[str, Any] | None = None,
    color: int = 10,
) -> UnifiedCanidSample:
    digest, width, height = _write_image(root, relative, color=color)
    record = get_record(dataset)
    return UnifiedCanidSample(
        sample_id=_sha(f"sample:{label}"),
        dataset_name=dataset,
        dataset_version=record.version,
        source_group_id=f"source:{label}",
        image_path=relative,
        image_sha256=digest,
        width=width,
        height=height,
        raw_identity_id=f"identity:{label}",
        face_box_xyxy=face_box,
        dog_boxes_xyxy=dog_box,
        head_roi_xyxy=head_box,
        foreground_mask_path=trimap,
        capture_group_id=f"capture:{label}",
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        split_role=split,
        metadata={} if metadata is None else metadata,
    )

def _write_oxford_xml(root: Path, stem: str) -> None:
    path = root / "annotations" / "xmls" / f"{stem}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<annotation>"
        f"<filename>{stem}.jpg</filename>"
        "<object><name>dog</name><bndbox>"
        "<xmin>2</xmin><ymin>3</ymin><xmax>14</xmax><ymax>13</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )

@pytest.fixture
def route_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, tuple[UnifiedCanidSample, ...]], dict[str, Path]]:
    roots = {name: tmp_path / name for name in CANONICAL_DATASETS}
    for root in roots.values():
        root.mkdir()
    source_records = {
        name: replace(get_record(name), data_root=str(roots[name]))
        for name in CANONICAL_DATASETS
    }
    original_get_record = get_record

    def fixture_record(name: str) -> Any:
        if name in source_records:
            return source_records[name]
        return original_get_record(name)

    monkeypatch.setattr(route_plan, "get_record", fixture_record)

    ap_root = roots["ap10k-dog"]
    ap_source = _sample(
        "ap10k-dog",
        ap_root,
        "ap-10k/data/shared.jpg",
        label="ap-first",
        dog_box=(1.0, 2.0, 11.0, 12.0),
        metadata={"annotation_id": 11, "image_id": 1},
    )
    ap_second = replace(
        ap_source,
        sample_id=_sha("sample:ap-second"),
        dog_boxes_xyxy=(3.0, 4.0, 9.0, 10.0),
        metadata={"annotation_id": 12, "image_id": 1},
    )
    ap_annotation = ap_root / "ap-10k" / "annotations" / "ap10k-train-split1.json"
    ap_annotation.parent.mkdir(parents=True)
    ap_annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "shared.jpg"}],
                "annotations": [
                    {
                        "id": 11,
                        "image_id": 1,
                        "category_id": 8,
                        "bbox": [1, 2, 10, 10],
                    },
                    {
                        "id": 12,
                        "image_id": 1,
                        "category_id": 8,
                        "bbox": [3, 4, 6, 6],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    dogflw_root = roots["dogflw"]
    dogflw_valid = _sample(
        "dogflw",
        dogflw_root,
        "DogFLW/train/images/valid.png",
        label="dogflw-valid",
        face_box=(2.0, 2.0, 16.0, 14.0),
    )
    dogflw_invalid = _sample(
        "dogflw",
        dogflw_root,
        "DogFLW/train/images/invalid.png",
        label="dogflw-invalid",
    )
    label_root = dogflw_root / "DogFLW" / "train" / "labels"
    label_root.mkdir(parents=True)
    (label_root / "valid.json").write_text(
        json.dumps({"landmarks": [float("nan")], "bounding_boxes": [2, 2, 16, 14]}),
        encoding="utf-8",
    )
    (label_root / "invalid.json").write_text(
        json.dumps({"landmarks": [], "bounding_boxes": ["", 2, 16, 14]}),
        encoding="utf-8",
    )

    dogface = _sample(
        "dogfacenet224",
        roots["dogfacenet224"],
        "after_4_bis/7/native.png",
        label="dogface",
    )
    mpdd = _sample("mpdd", roots["mpdd"], "MPDD/pytorch/train/body.png", label="mpdd")

    oxford_root = roots["oxford-pets-dog"]
    oxford_valid = _sample(
        "oxford-pets-dog",
        oxford_root,
        "images/beagle_1.jpg",
        label="oxford-mask",
        split="trainval",
        trimap="annotations/trimaps/beagle_1.png",
    )
    _write_image(
        oxford_root,
        "annotations/trimaps/beagle_1.png",
        color=1,
        mode="L",
    )
    oxford_head = _sample(
        "oxford-pets-dog",
        oxford_root,
        "images/beagle_2.jpg",
        label="oxford-head",
        split="test",
        head_box=(2.0, 3.0, 14.0, 13.0),
        trimap="annotations/trimaps/beagle_2.png",
    )
    _write_image(
        oxford_root,
        "annotations/trimaps/beagle_2.png",
        color=9,
        mode="L",
    )
    _write_oxford_xml(oxford_root, "beagle_2")
    oxford_parse = _sample(
        "oxford-pets-dog",
        oxford_root,
        "images/beagle_3.jpg",
        label="oxford-parse",
        split="test",
        trimap="annotations/trimaps/beagle_3.png",
    )
    _write_image(
        oxford_root,
        "annotations/trimaps/beagle_3.png",
        color=2,
        mode="L",
    )

    sibetan = _sample(
        "sibetan", roots["sibetan"], "Sibetan/0/body.jpg", label="sibetan"
    )
    yt = _sample(
        "yt-bb-dog",
        roots["yt-bb-dog"],
        "YT-BB-dog/YT-BB-Dog/train/1/body.jpg",
        label="yt",
    )
    samples = {
        "ap10k-dog": (ap_second, ap_source),
        "dogflw": (dogflw_invalid, dogflw_valid),
        "dogfacenet224": (dogface,),
        "mpdd": (mpdd,),
        "oxford-pets-dog": (oxford_parse, oxford_head, oxford_valid),
        "sibetan": (sibetan,),
        "yt-bb-dog": (yt,),
    }
    return samples, roots

def _build(
    samples: Mapping[str, tuple[UnifiedCanidSample, ...]],
    *,
    maximum: int | None = None,
) -> dict[str, Any]:
    return build_full128_route_plan(
        parser_runtime_manifest_sha256=_RUNTIME_SHA,
        parser_policy_sha256=_POLICY_SHA,
        maximum_samples_per_dataset=maximum,
        samples_by_dataset=samples,
    )

def test_all_routes_are_content_bound_and_order_deterministic(
    route_fixture: tuple[dict[str, tuple[UnifiedCanidSample, ...]], dict[str, Path]],
) -> None:
    samples, _ = route_fixture
    first = _build(samples)
    second = _build({name: tuple(reversed(rows)) for name, rows in samples.items()})

    assert first == second
    assert first["content_kind"] == "METADATA_ONLY"
    assert first["schema_version"] == ROUTE_PLAN_BUNDLE_SCHEMA
    assert first["plan"]["schema_version"] == ROUTE_PLAN_SCHEMA
    assert first["route_policy"]["schema_version"] == ROUTE_POLICY_SCHEMA
    assert first["executes_image_crop_or_animal_parsing"] is False
    records = first["plan"]["records"]
    assert [row["sample_token"] for row in records] == sorted(
        row["sample_token"] for row in records
    )
    assert {row["route_intent"] for row in records} == {"BODY_PARSING"}
    assert all(row["parser_cache_key"] is not None for row in records)
    assert all(row["target_size"] == 224 for row in records)
    assert all(row["context_fraction"] == 0.05 for row in records)
    assert all(row["background_rgb"] == [127, 127, 127] for row in records)
    assert all(row["schema_version"] == ROUTE_PLAN_RECORD_SCHEMA for row in records)
    assert validate_full128_route_plan_bundle(first) == first

    oxford_mask = next(
        row
        for row in records
        if row["dataset_name"] == "oxford-pets-dog"
        and row["route_evidence"]["trimap_state"] == "VALID_FOREGROUND"
    )
    assert oxford_mask["route_evidence"]["label_policy"] == {
        "foreground": 1,
        "excluded": [2, 3],
    }
    assert oxford_mask["route_evidence"]["observed_labels"] == [1]
    assert oxford_mask["route_evidence"]["annotation_usage"] == (
        "AUDIT_AND_FUTURE_PARSER_DEVELOPMENT_ONLY"
    )
    dogflw_face = next(
        row
        for row in records
        if row["dataset_name"] == "dogflw"
        and row["route_evidence"]["bbox_state"] == "VALID_WITHIN_SOURCE"
    )
    assert dogflw_face["route_evidence"]["normalized_nonstandard_constants"] == {
        "NaN": 1
    }
    assert dogflw_face["route_evidence"]["annotation_usage"] == (
        "AUDIT_AND_FUTURE_PARSER_DEVELOPMENT_ONLY"
    )
    intents = {
        row["dataset_name"]: row["route_evidence"]["association_intent"]
        for row in records
        if row["dataset_name"] != "ap10k-dog"
    }
    assert intents["dogflw"] == "SELECT_LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS"
    assert intents["oxford-pets-dog"] == (
        "SELECT_LARGEST_VALID_DOG_BY_FOREGROUND_PIXELS"
    )
    assert all(
        intents[name] == "REQUIRE_EXACTLY_ONE_POST_SUPPRESSION_DOG"
        for name in ("dogfacenet224", "mpdd", "sibetan", "yt-bb-dog")
    )

def test_ap10k_annotations_share_parser_cache_but_keep_bbox_authority(
    route_fixture: tuple[dict[str, tuple[UnifiedCanidSample, ...]], dict[str, Path]],
) -> None:
    samples, _ = route_fixture
    rows = [
        row
        for row in _build(samples)["plan"]["records"]
        if row["dataset_name"] == "ap10k-dog"
    ]

    assert len(rows) == 2
    assert rows[0]["parser_cache_key"] == rows[1]["parser_cache_key"]
    assert rows[0]["duplicate_component"] == rows[1]["duplicate_component"]
    assert {row["route_evidence"]["annotation_id"] for row in rows} == {11, 12}
    assert (
        len(
            {
                row["route_evidence"]["association_intent"]["authority_sha256"]
                for row in rows
            }
        )
        == 2
    )

def test_bounded_selection_is_recorded_without_changing_selected_rows(
    route_fixture: tuple[dict[str, tuple[UnifiedCanidSample, ...]], dict[str, Path]],
) -> None:
    samples, _ = route_fixture
    complete = _build(samples)
    bounded = _build(samples, maximum=1)

    selection = bounded["plan"]["selection"]
    assert selection["mode"] == "DETERMINISTIC_MAXIMUM_PER_DATASET"
    assert selection["maximum_samples_per_dataset"] == 1
    assert len(bounded["plan"]["records"]) == len(CANONICAL_DATASETS)
    complete_by_token = {
        row["sample_token"]: row for row in complete["plan"]["records"]
    }
    assert all(
        row == complete_by_token[row["sample_token"]]
        for row in bounded["plan"]["records"]
    )

class _ExplodingAdapters(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError(f"adapter was accessed: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("adapter registry was iterated")

    def __len__(self) -> int:
        raise AssertionError("adapter registry length was accessed")

def test_blocked_petface_and_uppercase_hash_fail_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(route_plan, "ADAPTERS", _ExplodingAdapters())
    with pytest.raises(ValueError, match="blocked by source_lock"):
        build_full128_route_plan(
            parser_runtime_manifest_sha256=_RUNTIME_SHA,
            parser_policy_sha256=_POLICY_SHA,
            dataset_names=("petface-dog",),
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_parser_cache_key(
            _sha("source").upper(),
            parser_runtime_manifest_sha256=_RUNTIME_SHA,
            parser_policy_sha256=_POLICY_SHA,
        )

def test_missing_and_tampered_route_evidence_fail_closed(
    route_fixture: tuple[dict[str, tuple[UnifiedCanidSample, ...]], dict[str, Path]],
) -> None:
    samples, roots = route_fixture
    label = roots["dogflw"] / "DogFLW" / "train" / "labels" / "valid.json"
    label_bytes = label.read_bytes()
    label.unlink()
    with pytest.raises(FileNotFoundError, match="DogFLW label artifact"):
        _build(samples)

    label.write_bytes(label_bytes)
    bundle = _build(samples)
    trimap = roots["oxford-pets-dog"] / "annotations" / "trimaps" / "beagle_1.png"
    trimap.write_bytes(trimap.read_bytes() + b"tampered")
    with pytest.raises((ValueError, RuntimeError), match="artifact|trimap"):
        validate_full128_route_plan_bundle(bundle)

def test_source_traversal_and_symlink_are_rejected(
    route_fixture: tuple[dict[str, tuple[UnifiedCanidSample, ...]], dict[str, Path]],
) -> None:
    samples, roots = route_fixture
    dogface = samples["dogfacenet224"][0]
    traversing = replace(dogface, image_path="../escape.png")
    changed = {**samples, "dogfacenet224": (traversing,)}
    with pytest.raises(ValueError, match="unsafe Full128 source image"):
        _build(changed)

    source = roots["dogfacenet224"] / dogface.image_path
    link = source.with_name("link.png")
    link.symlink_to(source)
    linked = replace(
        dogface, image_path=link.relative_to(roots["dogfacenet224"]).as_posix()
    )
    changed = {**samples, "dogfacenet224": (linked,)}
    with pytest.raises(ValueError, match="symlink"):
        _build(changed)

def test_dataset_root_symlink_spelling_resolves_to_canonical_storage(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    link = tmp_path / "dataset"
    link.symlink_to(storage, target_is_directory=True)

    assert route_plan._canonical_dataset_root(link, "fixture") == storage

def test_workflow_publishes_private_json_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "route-plan.json"
    fake_bundle = {
        "plan_sha256": _sha("plan"),
        "plan": {
            "selection": {"mode": "COMPLETE_DATASETS"},
            "records": [],
        },
    }
    monkeypatch.setenv("CANINE_IDENTITY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        route_workflow, "build_full128_route_plan", lambda **_: fake_bundle
    )
    argv = [
        "--parser-runtime-manifest-sha256",
        _RUNTIME_SHA,
        "--parser-policy-sha256",
        _POLICY_SHA,
        "--output",
        str(output),
    ]

    assert route_workflow.main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == fake_bundle
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        route_workflow.main(argv)
