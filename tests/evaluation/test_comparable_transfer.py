from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator
import pytest

from evaluation.comparable_transfer import (
    REPORT_SCHEMA,
    assert_backbone_only_comparison,
    load_split,
    main,
    run_smoke,
    score_comparable_transfer,
    smoke_embeddings,
    smoke_samples,
    write_split,
)
from evaluation.splits.comparable_transfer import (
    METRICS,
    PARSER_POLICY_SCHEMA,
    SPLIT_SEED,
    ComparableTransferSplit,
    bind_crops,
    freeze_comparable_transfer as freeze_split,
)
from tests.repo_root import REPO_ROOT as ROOT

_SCHEMA_PATH = (
    ROOT
    / "shared"
    / "contracts"
    / "schemas"
    / "evaluation.comparable_transfer.v1.schema.json"
)


def _bound(split: ComparableTransferSplit) -> ComparableTransferSplit:
    crops = {
        row.sample_id: f"{index:064x}"
        for index, row in enumerate(
            (*split.train_samples, *split.gallery, *split.query), start=1
        )
    }
    return bind_crops(split, crops, include_train=True)


def test_freeze_uses_train_ids_only_and_is_identity_disjoint() -> None:
    train, eval_samples = smoke_samples()
    split = freeze_split(train, eval_samples)
    assert split.split_seed == SPLIT_SEED
    assert split.comparable is True
    assert split.parser_policy_schema == PARSER_POLICY_SCHEMA
    assert all(item.dataset_name == "yt-bb-dog" for item in split.train_identities)
    assert all(row.dataset_name == "sibetan" for row in (*split.gallery, *split.query))
    assert "test-hold" not in {item.raw_identity_id for item in split.train_identities}
    train_ids = {item.identity_id for item in split.train_identities}
    eval_ids = {row.identity_id for row in split.gallery}
    assert train_ids.isdisjoint(eval_ids)
    assert eval_ids == {row.identity_id for row in split.query}
    gallery_seq = {(row.identity_id, row.sequence_id) for row in split.gallery}
    query_seq = {(row.identity_id, row.sequence_id) for row in split.query}
    assert not gallery_seq & query_seq
    assert tuple(row.sample_id for row in split.gallery) == tuple(
        sorted(row.sample_id for row in split.gallery)
    )


def test_same_seed_same_lists_and_schema_round_trip(tmp_path: Path) -> None:
    train, eval_samples = smoke_samples()
    left = freeze_split(train, eval_samples, split_seed=SPLIT_SEED)
    right = freeze_split(train, eval_samples, split_seed=SPLIT_SEED)
    assert left.split_sha256 == right.split_sha256
    assert left.gallery_list_sha256 == right.gallery_list_sha256
    assert left.query_list_sha256 == right.query_list_sha256
    payload = left.to_dict()
    Draft202012Validator(json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))).validate(
        payload
    )
    assert payload["metrics"] == list(METRICS)
    assert payload["comparison_variable"] == "backbone"
    assert "cvi." not in json.dumps(payload)
    path = tmp_path / "split.json"
    write_split(path, left)
    loaded = load_split(path)
    assert loaded.split_sha256 == left.split_sha256
    assert ComparableTransferSplit.from_dict(payload).split_sha256 == left.split_sha256


def test_seed_change_is_not_the_frozen_panel() -> None:
    train, eval_samples = smoke_samples()
    frozen = freeze_split(train, eval_samples, split_seed=SPLIT_SEED)
    other = freeze_split(train, eval_samples, split_seed=1)
    assert other.comparable is False
    assert other.split_sha256 != frozen.split_sha256


def test_score_emits_rank_and_map_and_backbone_is_the_only_variable() -> None:
    train, eval_samples = smoke_samples()
    split = _bound(freeze_split(train, eval_samples))
    embeddings = smoke_embeddings(split)
    left = score_comparable_transfer(split, embeddings, backbone_id="backbone-a")
    right = score_comparable_transfer(split, embeddings, backbone_id="backbone-b")
    assert left["schema_version"] == REPORT_SCHEMA
    assert left["metrics"].keys() == {"Rank-1", "Rank-5", "mAP"}
    assert left["metrics"]["Rank-1"] == pytest.approx(1.0)
    assert left["metrics"]["Rank-5"] == pytest.approx(1.0)
    assert 0.0 <= left["metrics"]["mAP"] <= 1.0
    assert_backbone_only_comparison(left, right)
    mutated = dict(left)
    mutated["gallery_list_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="gallery_list_sha256"):
        assert_backbone_only_comparison(mutated, right)


def test_score_requires_bound_crops() -> None:
    train, eval_samples = smoke_samples()
    split = freeze_split(train, eval_samples)
    with pytest.raises(ValueError, match="crop binding"):
        score_comparable_transfer(split, smoke_embeddings(split), backbone_id="x")


def test_bind_crops_fails_closed_when_a_frozen_sample_is_missing() -> None:
    train, eval_samples = smoke_samples()
    split = freeze_split(train, eval_samples)
    crops = {split.gallery[0].sample_id: "a" * 64}
    with pytest.raises(ValueError, match="missing"):
        bind_crops(split, crops)


def test_smoke_cli_writes_split_report_and_traces(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    result = run_smoke(output)
    assert result["comparable"] is True
    assert result["metrics"]["Rank-1"] == pytest.approx(1.0)
    split = load_split(output / "split.json")
    assert split.crop_binding_status == "bound"
    assert (output / "report.json").is_file()
    for stage in (
        "parsing",
        "identification",
        "representation",
        "enrollment",
        "gallery",
        "search",
    ):
        assert (output / "traces" / f"{stage}.json").is_file()
    assert main(["smoke", "--output-dir", str(tmp_path / "cli-smoke")]) == 0
