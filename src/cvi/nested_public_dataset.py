"""Bind a nested public ZIP to its audited parent extraction manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvi.provenance import content_sha256
from cvi.public_dataset import (
    PublicDatasetArchivePolicy,
    PublicDatasetArchiveReceipt,
    PublicDatasetSourceContract,
    SourceChecksumAuthority,
    _validate_sha256,
    audit_public_dataset_zip,
)
from cvi.public_dataset_extraction import (
    ExtractedPublicDatasetFile,
    PublicDatasetExtractionReceipt,
)


@dataclass(frozen=True, slots=True)
class ParentBoundNestedArchiveReceipt:
    parent_extraction_receipt_sha256: str
    parent_file_content_manifest_sha256: str
    parent_member_relative_path: str
    parent_member_content_sha256: str
    parent_member_bytes: int
    nested_source_contract_sha256: str
    nested_archive_receipt_sha256: str
    nested_archive_receipt: PublicDatasetArchiveReceipt
    decision: str = "PASS_PARENT_BOUND_NESTED_ARCHIVE"
    interpretation: str = "NESTED_ARCHIVE_INTAKE_ONLY_NOT_DATASET_ADMISSION"
    schema_version: str = "cvi.parent_bound_nested_archive_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.parent_bound_nested_archive_receipt.v1":
            raise ValueError("unsupported parent-bound nested archive receipt")
        for name in (
            "parent_extraction_receipt_sha256",
            "parent_file_content_manifest_sha256",
            "parent_member_content_sha256",
            "nested_source_contract_sha256",
            "nested_archive_receipt_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if (
            not isinstance(self.parent_member_relative_path, str)
            or not self.parent_member_relative_path
        ):
            raise ValueError("nested parent member path is empty")
        if (
            isinstance(self.parent_member_bytes, bool)
            or not isinstance(self.parent_member_bytes, int)
            or self.parent_member_bytes <= 0
        ):
            raise ValueError("nested parent member bytes differ")
        if (
            self.nested_archive_receipt.receipt_sha256
            != self.nested_archive_receipt_sha256
        ):
            raise ValueError("nested archive receipt digest differs")
        if (
            self.nested_archive_receipt.archive_sha256
            != self.parent_member_content_sha256
            or self.nested_archive_receipt.archive_bytes != self.parent_member_bytes
        ):
            raise ValueError("nested archive differs from parent member")
        if self.decision != "PASS_PARENT_BOUND_NESTED_ARCHIVE":
            raise ValueError("nested archive decision differs")
        if self.interpretation != "NESTED_ARCHIVE_INTAKE_ONLY_NOT_DATASET_ADMISSION":
            raise ValueError("nested archive interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_extraction_receipt_sha256": (
                self.parent_extraction_receipt_sha256
            ),
            "parent_file_content_manifest_sha256": (
                self.parent_file_content_manifest_sha256
            ),
            "parent_member_relative_path": self.parent_member_relative_path,
            "parent_member_content_sha256": self.parent_member_content_sha256,
            "parent_member_bytes": self.parent_member_bytes,
            "nested_source_contract_sha256": self.nested_source_contract_sha256,
            "nested_archive_receipt_sha256": self.nested_archive_receipt_sha256,
            "nested_archive_receipt": self.nested_archive_receipt.to_dict(),
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> ParentBoundNestedArchiveReceipt:
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("parent-bound nested archive receipt fields differ")
        values = dict(payload)
        nested = values["nested_archive_receipt"]
        if not isinstance(nested, dict):
            raise TypeError("nested archive receipt must be an object")
        values["nested_archive_receipt"] = PublicDatasetArchiveReceipt.from_dict(
            nested
        )
        return cls(**values)


def audit_parent_bound_nested_public_zip(
    *,
    parent_output_directory: Path,
    parent_extraction_receipt: PublicDatasetExtractionReceipt,
    parent_files: tuple[ExtractedPublicDatasetFile, ...],
    parent_member_relative_path: str,
    terms_snapshot_path: Path,
    nested_source: PublicDatasetSourceContract,
    nested_policy: PublicDatasetArchivePolicy,
) -> ParentBoundNestedArchiveReceipt:
    """Audit a nested ZIP whose bytes are authenticated by the parent manifest."""

    if nested_source.checksum_authority is not (
        SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
    ):
        raise ValueError("nested source must not claim a publisher checksum")
    if parent_output_directory.is_symlink():
        raise ValueError("parent extraction directory must not be a symlink")
    parent_root = parent_output_directory.resolve(strict=True)
    if not parent_root.is_dir():
        raise ValueError("parent extraction directory must be a directory")
    if parent_root.name != parent_extraction_receipt.output_directory_name:
        raise ValueError("parent extraction directory name differs")
    ordered = tuple(sorted(parent_files, key=lambda item: item.relative_path))
    if content_sha256([item.to_dict() for item in ordered]) != (
        parent_extraction_receipt.file_content_manifest_sha256
    ):
        raise ValueError("parent file manifest digest differs")
    matches = tuple(
        item for item in ordered if item.relative_path == parent_member_relative_path
    )
    if len(matches) != 1:
        raise ValueError("nested archive parent member is not unique")
    member = matches[0]
    if nested_source.archive_filename != Path(member.relative_path).name:
        raise ValueError("nested source filename differs from parent member")
    if nested_source.expected_archive_bytes != member.byte_size:
        raise ValueError("nested source byte size differs from parent member")
    archive_path = parent_root.joinpath(*member.relative_path.split("/"))
    receipt = audit_public_dataset_zip(
        archive_path=archive_path,
        terms_snapshot_path=terms_snapshot_path,
        source=nested_source,
        policy=nested_policy,
    )
    if receipt.archive_sha256 != member.content_sha256:
        raise ValueError("nested archive hash differs from parent manifest")
    return ParentBoundNestedArchiveReceipt(
        parent_extraction_receipt_sha256=(
            parent_extraction_receipt.receipt_sha256
        ),
        parent_file_content_manifest_sha256=(
            parent_extraction_receipt.file_content_manifest_sha256
        ),
        parent_member_relative_path=member.relative_path,
        parent_member_content_sha256=member.content_sha256,
        parent_member_bytes=member.byte_size,
        nested_source_contract_sha256=nested_source.contract_sha256,
        nested_archive_receipt_sha256=receipt.receipt_sha256,
        nested_archive_receipt=receipt,
    )
