"""Freeze the YT-BB-Dog → Sibetan comparable-transfer split.

Train identities are the official yt-bb-dog train split only. Evaluation is
Sibetan, identity-disjoint from training. Gallery and query lists are sequence
disjoint, built with a fixed seed, and hashed so later backbone swaps cannot
quietly change the panel.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.types import UnifiedCanidSample
from shared.foundation.provenance import content_sha256

PROTOCOL_SCHEMA = "evaluation.comparable_transfer.v1"
SPLIT_SEED = 0
TRAIN_DATASET = "yt-bb-dog"
TRAIN_SPLIT_ROLE = "train"
EVAL_DATASET = "sibetan"
PARSER_POLICY_SCHEMA = "parsing.policy.v6"
METRICS = ("Rank-1", "Rank-5", "mAP")
COMPARISON_VARIABLE = "backbone"
MIN_EVAL_SEQUENCES = 2
INTERPRETATION = (
    "YT_BB_DOG_TRAIN_IDS_TO_SIBETAN_HELD_OUT_IDENTITY_DISJOINT_TRANSFER;"
    "GALLERY_AND_QUERY_LISTS_FROZEN;"
    "BACKBONE_IS_THE_ONLY_COMPARISON_VARIABLE;"
    "PARSER_POLICY_V6_CROPS;"
    "NOT_BIFOR_SEQUENCE_MEAN_CLAIM;"
    "NOT_BIOMETRIC_VALIDATION"
)

_HEX64 = 64


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != _HEX64 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    int(value, 16)
    return value


def _require_text(value: object, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty text")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds {maximum} bytes")
    return value


def _identity_rng(split_seed: int, identity_id: str) -> np.random.Generator:
    material = f"{split_seed}\0{identity_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(seed))


@dataclass(frozen=True, slots=True)
class ComparableTransferRow:
    sample_id: str
    identity_id: str
    raw_identity_id: str
    dataset_name: str
    sequence_id: str
    image_path: str
    image_sha256: str
    crop_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.sample_id, "sample_id")
        _require_text(self.identity_id, "identity_id")
        _require_text(self.raw_identity_id, "raw_identity_id", maximum=256)
        _require_text(self.dataset_name, "dataset_name", maximum=64)
        _require_text(self.sequence_id, "sequence_id", maximum=256)
        _require_text(self.image_path, "image_path")
        _require_sha256(self.image_sha256, "image_sha256")
        if self.crop_sha256 is not None:
            _require_sha256(self.crop_sha256, "crop_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "identity_id": self.identity_id,
            "raw_identity_id": self.raw_identity_id,
            "dataset_name": self.dataset_name,
            "sequence_id": self.sequence_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "crop_sha256": self.crop_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComparableTransferRow:
        expected = {
            "sample_id",
            "identity_id",
            "raw_identity_id",
            "dataset_name",
            "sequence_id",
            "image_path",
            "image_sha256",
            "crop_sha256",
        }
        if set(payload) != expected:
            raise ValueError("comparable-transfer row keys differ")
        return cls(
            sample_id=str(payload["sample_id"]),
            identity_id=str(payload["identity_id"]),
            raw_identity_id=str(payload["raw_identity_id"]),
            dataset_name=str(payload["dataset_name"]),
            sequence_id=str(payload["sequence_id"]),
            image_path=str(payload["image_path"]),
            image_sha256=str(payload["image_sha256"]),
            crop_sha256=(
                None
                if payload["crop_sha256"] is None
                else str(payload["crop_sha256"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ComparableTransferIdentity:
    identity_id: str
    raw_identity_id: str
    dataset_name: str

    def __post_init__(self) -> None:
        _require_text(self.identity_id, "identity_id")
        _require_text(self.raw_identity_id, "raw_identity_id", maximum=256)
        _require_text(self.dataset_name, "dataset_name", maximum=64)

    def to_dict(self) -> dict[str, str]:
        return {
            "identity_id": self.identity_id,
            "raw_identity_id": self.raw_identity_id,
            "dataset_name": self.dataset_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComparableTransferIdentity:
        if set(payload) != {"identity_id", "raw_identity_id", "dataset_name"}:
            raise ValueError("comparable-transfer identity keys differ")
        return cls(
            identity_id=str(payload["identity_id"]),
            raw_identity_id=str(payload["raw_identity_id"]),
            dataset_name=str(payload["dataset_name"]),
        )


@dataclass(frozen=True, slots=True)
class ComparableTransferSplit:
    train_identities: tuple[ComparableTransferIdentity, ...]
    train_samples: tuple[ComparableTransferRow, ...]
    gallery: tuple[ComparableTransferRow, ...]
    query: tuple[ComparableTransferRow, ...]
    split_seed: int = SPLIT_SEED
    parser_policy_schema: str = PARSER_POLICY_SCHEMA
    crop_binding_status: str = "unbound"
    schema_version: str = PROTOCOL_SCHEMA
    interpretation: str = INTERPRETATION

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ValueError("unsupported comparable-transfer schema")
        if self.parser_policy_schema != PARSER_POLICY_SCHEMA:
            raise ValueError("comparable-transfer crops must declare parsing.policy.v6")
        if self.crop_binding_status not in {"unbound", "bound"}:
            raise ValueError("crop_binding_status must be unbound or bound")
        if not isinstance(self.split_seed, int) or isinstance(self.split_seed, bool):
            raise ValueError("split_seed must be an int")
        if not self.train_identities or not self.train_samples:
            raise ValueError("train identities and samples must not be empty")
        if not self.gallery or not self.query:
            raise ValueError("gallery and query lists must not be empty")
        _validate_split_lists(self)

    @property
    def comparable(self) -> bool:
        return (
            self.split_seed == SPLIT_SEED
            and self.parser_policy_schema == PARSER_POLICY_SCHEMA
            and self.interpretation == INTERPRETATION
        )

    @property
    def train_identity_sha256(self) -> str:
        return content_sha256([item.to_dict() for item in self.train_identities])

    @property
    def gallery_list_sha256(self) -> str:
        return content_sha256([item.to_dict() for item in self.gallery])

    @property
    def query_list_sha256(self) -> str:
        return content_sha256([item.to_dict() for item in self.query])

    @property
    def train_sample_sha256(self) -> str:
        return content_sha256([item.to_dict() for item in self.train_samples])

    @property
    def split_sha256(self) -> str:
        return content_sha256(self.to_dict(include_hashes=False))

    def to_dict(self, *, include_hashes: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "interpretation": self.interpretation,
            "split_seed": self.split_seed,
            "comparable": self.comparable,
            "parser_policy_schema": self.parser_policy_schema,
            "crop_binding_status": self.crop_binding_status,
            "metrics": list(METRICS),
            "comparison_variable": COMPARISON_VARIABLE,
            "train": {
                "dataset": TRAIN_DATASET,
                "split_role": TRAIN_SPLIT_ROLE,
                "identities": [item.to_dict() for item in self.train_identities],
                "samples": [item.to_dict() for item in self.train_samples],
            },
            "eval": {
                "dataset": EVAL_DATASET,
                "gallery": [item.to_dict() for item in self.gallery],
                "query": [item.to_dict() for item in self.query],
            },
        }
        if include_hashes:
            payload["train_identity_sha256"] = self.train_identity_sha256
            payload["train_sample_sha256"] = self.train_sample_sha256
            payload["gallery_list_sha256"] = self.gallery_list_sha256
            payload["query_list_sha256"] = self.query_list_sha256
            payload["split_sha256"] = content_sha256(
                self.to_dict(include_hashes=False)
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComparableTransferSplit:
        train = payload.get("train")
        eval_block = payload.get("eval")
        if not isinstance(train, Mapping) or not isinstance(eval_block, Mapping):
            raise ValueError("comparable-transfer train/eval blocks differ")
        split = cls(
            train_identities=tuple(
                ComparableTransferIdentity.from_dict(item)
                for item in train["identities"]
            ),
            train_samples=tuple(
                ComparableTransferRow.from_dict(item) for item in train["samples"]
            ),
            gallery=tuple(
                ComparableTransferRow.from_dict(item) for item in eval_block["gallery"]
            ),
            query=tuple(
                ComparableTransferRow.from_dict(item) for item in eval_block["query"]
            ),
            split_seed=int(payload["split_seed"]),
            parser_policy_schema=str(payload["parser_policy_schema"]),
            crop_binding_status=str(payload["crop_binding_status"]),
            schema_version=str(payload["schema_version"]),
            interpretation=str(payload["interpretation"]),
        )
        if payload.get("metrics") != list(METRICS):
            raise ValueError("comparable-transfer metrics differ")
        if payload.get("comparison_variable") != COMPARISON_VARIABLE:
            raise ValueError("comparison_variable must be backbone")
        if train.get("dataset") != TRAIN_DATASET or train.get("split_role") != TRAIN_SPLIT_ROLE:
            raise ValueError("train dataset/split_role differ")
        if eval_block.get("dataset") != EVAL_DATASET:
            raise ValueError("eval dataset differs")
        if bool(payload.get("comparable")) != split.comparable:
            raise ValueError("comparable flag disagrees with seed and interpretation")
        return split


def _row_from_sample(sample: UnifiedCanidSample, *, dataset: str) -> ComparableTransferRow:
    if sample.dataset_name != dataset:
        raise ValueError(f"sample dataset {sample.dataset_name!r} is not {dataset}")
    if sample.raw_identity_id is None or sample.registered_identity_id is None:
        raise ValueError(f"{dataset} sample is missing identity fields")
    if sample.capture_group_id is None:
        raise ValueError(f"{dataset} sample is missing capture_group_id")
    if len(sample.image_sha256) != _HEX64:
        raise ValueError(f"{dataset} image_sha256 must be SHA-256")
    return ComparableTransferRow(
        sample_id=sample.sample_id,
        identity_id=sample.registered_identity_id,
        raw_identity_id=sample.raw_identity_id,
        dataset_name=dataset,
        sequence_id=sample.capture_group_id,
        image_path=sample.image_path,
        image_sha256=sample.image_sha256,
    )


def _validate_split_lists(split: ComparableTransferSplit) -> None:
    train_ids = tuple(item.identity_id for item in split.train_identities)
    if tuple(sorted(train_ids)) != train_ids:
        raise ValueError("train identities must be sorted by identity_id")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("train identities must be unique")
    if any(item.dataset_name != TRAIN_DATASET for item in split.train_identities):
        raise ValueError("train identities must be yt-bb-dog")
    train_samples = split.train_samples
    if tuple(sorted(train_samples, key=lambda row: row.sample_id)) != train_samples:
        raise ValueError("train samples must be sorted by sample_id")
    train_sample_ids = [row.sample_id for row in train_samples]
    if len(set(train_sample_ids)) != len(train_sample_ids):
        raise ValueError("train sample_ids must be unique")
    if any(row.dataset_name != TRAIN_DATASET for row in train_samples):
        raise ValueError("train samples must be yt-bb-dog")
    admitted = set(train_ids)
    if {row.identity_id for row in train_samples} != admitted:
        raise ValueError("train samples and train identities differ")
    train_bound = {row.crop_sha256 is not None for row in train_samples}
    if len(train_bound) != 1:
        raise ValueError("train crop binding must be uniform")
    if tuple(sorted(split.gallery, key=lambda row: row.sample_id)) != split.gallery:
        raise ValueError("gallery must be sorted by sample_id")
    if tuple(sorted(split.query, key=lambda row: row.sample_id)) != split.query:
        raise ValueError("query must be sorted by sample_id")
    for name, rows in (("gallery", split.gallery), ("query", split.query)):
        ids = [row.sample_id for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} sample_ids must be unique")
        if any(row.dataset_name != EVAL_DATASET for row in rows):
            raise ValueError(f"{name} rows must be sibetan")
        bound = {row.crop_sha256 is not None for row in rows}
        if split.crop_binding_status == "bound" and bound != {True}:
            raise ValueError(f"{name} crops must be bound")
        if split.crop_binding_status == "unbound" and bound != {False}:
            raise ValueError(f"{name} crops must be unbound")
    gallery_ids = {row.sample_id for row in split.gallery}
    query_ids = {row.sample_id for row in split.query}
    if gallery_ids & query_ids:
        raise ValueError("gallery and query sample_ids overlap")
    gallery_sequences = {(row.identity_id, row.sequence_id) for row in split.gallery}
    query_sequences = {(row.identity_id, row.sequence_id) for row in split.query}
    if gallery_sequences & query_sequences:
        raise ValueError("gallery and query sequences overlap")
    eval_ids = {row.identity_id for row in split.gallery} | {
        row.identity_id for row in split.query
    }
    if eval_ids != {row.identity_id for row in split.gallery} or eval_ids != {
        row.identity_id for row in split.query
    }:
        raise ValueError("closed-set eval requires the same identities in gallery and query")
    train_registered = {item.identity_id for item in split.train_identities}
    train_raw = {(item.dataset_name, item.raw_identity_id) for item in split.train_identities}
    eval_raw = {(row.dataset_name, row.raw_identity_id) for row in split.gallery}
    if train_registered & eval_ids:
        raise ValueError("train and eval registered identities overlap")
    if train_raw & eval_raw:
        raise ValueError("train and eval raw identities overlap")


def freeze_comparable_transfer(
    train_samples: Sequence[UnifiedCanidSample],
    eval_samples: Sequence[UnifiedCanidSample],
    *,
    split_seed: int = SPLIT_SEED,
) -> ComparableTransferSplit:
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("split_seed must be an int")
    train_rows: list[ComparableTransferRow] = []
    for sample in train_samples:
        if sample.dataset_name != TRAIN_DATASET:
            raise ValueError("train samples must be yt-bb-dog")
        if sample.split_role != TRAIN_SPLIT_ROLE:
            continue
        train_rows.append(_row_from_sample(sample, dataset=TRAIN_DATASET))
    if not train_rows:
        raise ValueError("yt-bb-dog train split has no identity samples")
    train_rows.sort(key=lambda row: row.sample_id)
    identities = {
        (row.identity_id, row.raw_identity_id, row.dataset_name) for row in train_rows
    }
    train_identities = tuple(
        ComparableTransferIdentity(*item) for item in sorted(identities)
    )
    by_identity: dict[str, dict[str, list[UnifiedCanidSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in eval_samples:
        if sample.dataset_name != EVAL_DATASET:
            raise ValueError("eval samples must be sibetan")
        row = _row_from_sample(sample, dataset=EVAL_DATASET)
        by_identity[row.identity_id][row.sequence_id].append(sample)
    gallery: list[ComparableTransferRow] = []
    query: list[ComparableTransferRow] = []
    for identity_id in sorted(by_identity):
        sequences = tuple(sorted(by_identity[identity_id]))
        if len(sequences) < MIN_EVAL_SEQUENCES:
            continue
        order = _identity_rng(split_seed, identity_id).permutation(len(sequences))
        gallery_sequence = sequences[int(order[0])]
        query_sequences = {sequences[int(index)] for index in order[1:]}
        for sample in by_identity[identity_id][gallery_sequence]:
            gallery.append(_row_from_sample(sample, dataset=EVAL_DATASET))
        for sequence_id in sorted(query_sequences):
            for sample in by_identity[identity_id][sequence_id]:
                query.append(_row_from_sample(sample, dataset=EVAL_DATASET))
    if not gallery or not query:
        raise ValueError(
            "sibetan held-out panel needs identities with at least two sequences"
        )
    gallery.sort(key=lambda row: row.sample_id)
    query.sort(key=lambda row: row.sample_id)
    return ComparableTransferSplit(
        train_identities=train_identities,
        train_samples=tuple(train_rows),
        gallery=tuple(gallery),
        query=tuple(query),
        split_seed=split_seed,
    )


def bind_crops(
    split: ComparableTransferSplit,
    crop_sha256_by_sample: Mapping[str, str],
    *,
    include_train: bool = False,
) -> ComparableTransferSplit:
    required = [row.sample_id for row in (*split.gallery, *split.query)]
    if include_train:
        required.extend(row.sample_id for row in split.train_samples)
    missing = [sample_id for sample_id in required if sample_id not in crop_sha256_by_sample]
    if missing:
        raise ValueError(
            "parser v6 crop binding is missing "
            f"{len(missing)} frozen sample(s); first={missing[0]}"
        )

    def _bind(row: ComparableTransferRow, *, required_row: bool) -> ComparableTransferRow:
        digest = crop_sha256_by_sample.get(row.sample_id)
        if required_row:
            if digest is None:
                raise ValueError(f"crop binding missing {row.sample_id}")
            _require_sha256(digest, "crop_sha256")
            return ComparableTransferRow(
                sample_id=row.sample_id,
                identity_id=row.identity_id,
                raw_identity_id=row.raw_identity_id,
                dataset_name=row.dataset_name,
                sequence_id=row.sequence_id,
                image_path=row.image_path,
                image_sha256=row.image_sha256,
                crop_sha256=digest,
            )
        if digest is None:
            return row
        _require_sha256(digest, "crop_sha256")
        return ComparableTransferRow(
            sample_id=row.sample_id,
            identity_id=row.identity_id,
            raw_identity_id=row.raw_identity_id,
            dataset_name=row.dataset_name,
            sequence_id=row.sequence_id,
            image_path=row.image_path,
            image_sha256=row.image_sha256,
            crop_sha256=digest,
        )

    train_bound = include_train
    return ComparableTransferSplit(
        train_identities=split.train_identities,
        train_samples=tuple(
            _bind(row, required_row=train_bound) for row in split.train_samples
        ),
        gallery=tuple(_bind(row, required_row=True) for row in split.gallery),
        query=tuple(_bind(row, required_row=True) for row in split.query),
        split_seed=split.split_seed,
        parser_policy_schema=split.parser_policy_schema,
        crop_binding_status="bound",
        schema_version=split.schema_version,
        interpretation=split.interpretation,
    )


def assert_same_panel(left: ComparableTransferSplit, right: ComparableTransferSplit) -> None:
    if left.split_sha256 != right.split_sha256:
        raise ValueError("comparable-transfer panels differ; backbone is the only allowed change")
    if left.gallery_list_sha256 != right.gallery_list_sha256:
        raise ValueError("gallery list changed")
    if left.query_list_sha256 != right.query_list_sha256:
        raise ValueError("query list changed")
    if left.train_identity_sha256 != right.train_identity_sha256:
        raise ValueError("train identity list changed")
