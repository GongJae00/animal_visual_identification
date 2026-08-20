"""Deterministic receipts for audited public canine label semantics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from data.public_sources.public_canine_manifest import PublicCanineManifest, PublicCanineRecord
from shared.foundation.provenance import content_sha256


@dataclass(frozen=True, slots=True)
class PublicCanineVariantSummary:
    dataset_name: str
    dataset_version: str
    source_variant: str
    source_archive_sha256: str
    source_archive_receipt_sha256: str
    image_count: int
    identity_count: int
    record_manifest_sha256: str
    identity_semantics_counts: tuple[tuple[str, int], ...]
    region_counts: tuple[tuple[str, int], ...]
    split_image_counts: tuple[tuple[str, int], ...]
    split_identity_counts: tuple[tuple[str, int], ...]
    verified_camera_token_count: int
    schema_version: str = "data.public_canine_variant_summary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "data.public_canine_variant_summary.v1":
            raise ValueError("unsupported public canine variant summary")
        _require_token(self.dataset_name, "dataset_name")
        _require_token(self.dataset_version, "dataset_version")
        _require_token(self.source_variant, "source_variant")
        _require_sha256(self.source_archive_sha256, "source_archive_sha256")
        _require_sha256(
            self.source_archive_receipt_sha256,
            "source_archive_receipt_sha256",
        )
        if (
            isinstance(self.image_count, bool)
            or not isinstance(self.image_count, int)
            or isinstance(self.identity_count, bool)
            or not isinstance(self.identity_count, int)
            or self.image_count <= 0
            or self.identity_count <= 0
        ):
            raise ValueError("semantic summary cardinalities must be positive")
        if (
            isinstance(self.verified_camera_token_count, bool)
            or self.verified_camera_token_count != 0
        ):
            raise ValueError("public camera filename tokens are not verified")
        _require_sha256(self.record_manifest_sha256, "record_manifest_sha256")
        for name in (
            "identity_semantics_counts",
            "region_counts",
            "split_image_counts",
            "split_identity_counts",
        ):
            _require_sorted_counts(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_variant": self.source_variant,
            "source_archive_sha256": self.source_archive_sha256,
            "source_archive_receipt_sha256": (
                self.source_archive_receipt_sha256
            ),
            "image_count": self.image_count,
            "identity_count": self.identity_count,
            "record_manifest_sha256": self.record_manifest_sha256,
            "identity_semantics_counts": [list(item) for item in self.identity_semantics_counts],
            "region_counts": [list(item) for item in self.region_counts],
            "split_image_counts": [list(item) for item in self.split_image_counts],
            "split_identity_counts": [list(item) for item in self.split_identity_counts],
            "verified_camera_token_count": self.verified_camera_token_count,
        }


@dataclass(frozen=True, slots=True)
class PublicCanineSemanticReceipt:
    dataset_name: str
    variants: tuple[PublicCanineVariantSummary, ...]
    audited_facts: tuple[tuple[str, int], ...]
    interpretation: str = "SEMANTIC_INTAKE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION"
    schema_version: str = "data.public_canine_semantic_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "data.public_canine_semantic_receipt.v1":
            raise ValueError("unsupported public canine semantic receipt")
        if not self.variants:
            raise ValueError("semantic receipt must contain a variant")
        _require_token(self.dataset_name, "dataset_name")
        if any(item.dataset_name != self.dataset_name for item in self.variants):
            raise ValueError("semantic receipt mixes datasets")
        variant_names = tuple(item.source_variant for item in self.variants)
        if variant_names != tuple(sorted(variant_names)) or len(variant_names) != len(
            set(variant_names)
        ):
            raise ValueError("semantic variants must be sorted and unique")
        fact_names = tuple(name for name, _ in self.audited_facts)
        if fact_names != tuple(sorted(fact_names)) or len(fact_names) != len(
            set(fact_names)
        ):
            raise ValueError("audited facts must be sorted and unique")
        for name, value in self.audited_facts:
            _require_token(name, "audited fact name")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("audited fact values must be nonnegative integers")
        if self.interpretation != "SEMANTIC_INTAKE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION":
            raise ValueError("semantic receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "variants": [item.to_dict() for item in self.variants],
            "audited_facts": [list(item) for item in self.audited_facts],
            "interpretation": self.interpretation,
        }


def summarize_public_canine_manifest(
    manifest: PublicCanineManifest,
) -> tuple[PublicCanineVariantSummary, ...]:
    """Summarize each source variant with an order-independent record digest."""

    by_variant: dict[str, list[PublicCanineRecord]] = defaultdict(list)
    for record in manifest.records:
        by_variant[record.source_variant].append(record)
    summaries: list[PublicCanineVariantSummary] = []
    for variant, records in sorted(by_variant.items()):
        rows = sorted((_record_row(record) for record in records), key=lambda row: row[0])
        semantics = Counter(record.identity_semantics.value for record in records)
        regions = Counter(record.region.value for record in records)
        split_images = Counter(record.original_split or "UNASSIGNED" for record in records)
        split_ids: dict[str, set[str]] = defaultdict(set)
        for record in records:
            split_ids[record.original_split or "UNASSIGNED"].add(
                record.dataset_identity_id
            )
        summaries.append(
            PublicCanineVariantSummary(
                dataset_name=manifest.dataset_name,
                dataset_version=manifest.dataset_version,
                source_variant=variant,
                source_archive_sha256=manifest.source_archive_sha256,
                source_archive_receipt_sha256=manifest.source_archive_receipt_sha256,
                image_count=len(records),
                identity_count=len({record.dataset_identity_id for record in records}),
                record_manifest_sha256=content_sha256(rows),
                identity_semantics_counts=tuple(sorted(semantics.items())),
                region_counts=tuple(sorted(regions.items())),
                split_image_counts=tuple(sorted(split_images.items())),
                split_identity_counts=tuple(
                    sorted((split, len(identities)) for split, identities in split_ids.items())
                ),
                verified_camera_token_count=sum(
                    record.camera_token_verified for record in records
                ),
            )
        )
    return tuple(summaries)


def _record_row(record: PublicCanineRecord) -> tuple[Any, ...]:
    return (
        record.source_sample_id,
        record.dataset_identity_id,
        record.identity_semantics.value,
        record.region.value,
        record.source_variant,
        record.original_split,
        record.sequence_id,
        record.camera_token,
        record.camera_token_verified,
        record.filename_identity_token,
        record.source_cluster_id,
        record.in_no_mono_subset,
        record.paired_source_sample_id,
        record.member_path,
        record.member_crc32,
        record.member_uncompressed_bytes,
        record.container_member_path,
        record.container_member_crc32,
        record.container_member_uncompressed_bytes,
    )


def _require_token(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{name} must be bounded non-empty text")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_sorted_counts(values: object, name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    keys: list[str] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name} entries must be pairs")
        key, count = item
        _require_token(key, f"{name} key")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name} counts must be positive integers")
        keys.append(key)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} must be sorted with unique keys")
