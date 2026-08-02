"""Receipt-bound, no-overwrite extraction for audited public ZIP datasets."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from typing import Any, BinaryIO

from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from data_pipeline.public_dataset import (
    PublicDatasetArchivePolicy,
    PublicDatasetArchiveReceipt,
    PublicDatasetSourceContract,
    _canonical_zip_path,
    _stat_identity,
    _validate_portable_component,
    _validate_sha256,
    _validate_zip_members,
)


@dataclass(frozen=True, slots=True)
class ExtractedPublicDatasetFile:
    relative_path: str
    byte_size: int
    content_sha256: str

    def __post_init__(self) -> None:
        _validate_manifest_relative_path(self.relative_path)
        if isinstance(self.byte_size, bool) or not isinstance(self.byte_size, int):
            raise ValueError("extracted byte_size must be an integer")
        if self.byte_size < 0:
            raise ValueError("extracted byte_size must be non-negative")
        _validate_sha256(self.content_sha256, "content_sha256")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtractedPublicDatasetFile:
        if not isinstance(payload, dict) or set(payload) != {
            "relative_path",
            "byte_size",
            "content_sha256",
        }:
            raise ValueError("extracted file fields differ")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PublicDatasetExtractionReceipt:
    source_contract_sha256: str
    archive_policy_sha256: str
    archive_receipt_sha256: str
    archive_sha256: str
    output_directory_name: str
    file_content_manifest_sha256: str
    extracted_regular_files: int
    extracted_bytes: int
    publication_strategy: str
    decision: str = "PASS_EXTRACTED_CONTENT_MANIFEST"
    interpretation: str = "EXTRACTION_ONLY_NOT_DATASET_OR_MODEL_ADMISSION"
    schema_version: str = "cvi.public_dataset_extraction_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_dataset_extraction_receipt.v1":
            raise ValueError("unsupported public dataset extraction receipt")
        for name in (
            "source_contract_sha256",
            "archive_policy_sha256",
            "archive_receipt_sha256",
            "archive_sha256",
            "file_content_manifest_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if (
            not isinstance(self.output_directory_name, str)
            or Path(self.output_directory_name).name != self.output_directory_name
            or self.output_directory_name in {"", ".", ".."}
        ):
            raise ValueError("extraction output directory name must be a basename")
        _validate_portable_component(
            self.output_directory_name,
            "extraction output directory name",
        )
        for name in ("extracted_regular_files", "extracted_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.publication_strategy not in {
            "RENAMEAT2_NOREPLACE",
            "RESERVED_EMPTY_DIRECTORY_RENAME",
            "PLATFORM_NOREPLACE_RENAME",
        }:
            raise ValueError("extraction publication strategy differs")
        if self.decision != "PASS_EXTRACTED_CONTENT_MANIFEST":
            raise ValueError("extraction receipt decision differs")
        if self.interpretation != "EXTRACTION_ONLY_NOT_DATASET_OR_MODEL_ADMISSION":
            raise ValueError("extraction receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "source_contract_sha256": self.source_contract_sha256,
            "archive_policy_sha256": self.archive_policy_sha256,
            "archive_receipt_sha256": self.archive_receipt_sha256,
            "archive_sha256": self.archive_sha256,
            "output_directory_name": self.output_directory_name,
            "file_content_manifest_sha256": self.file_content_manifest_sha256,
            "extracted_regular_files": self.extracted_regular_files,
            "extracted_bytes": self.extracted_bytes,
            "publication_strategy": self.publication_strategy,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicDatasetExtractionReceipt:
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("extraction receipt fields differ")
        return cls(**payload)


def extract_audited_public_dataset_zip(
    *,
    archive_path: Path,
    source: PublicDatasetSourceContract,
    archive_policy: PublicDatasetArchivePolicy,
    archive_receipt: PublicDatasetArchiveReceipt,
    output_directory: Path,
) -> tuple[PublicDatasetExtractionReceipt, tuple[ExtractedPublicDatasetFile, ...]]:
    """Extract the exact audited archive to a new atomically published directory."""

    _verify_receipt_bindings(source, archive_policy, archive_receipt)
    if archive_path.name != source.archive_filename:
        raise ValueError("public dataset archive filename differs from source")
    if output_directory.name in {"", ".", ".."}:
        raise ValueError("extraction output directory must have a safe basename")
    try:
        _validate_portable_component(
            output_directory.name,
            "extraction output directory",
        )
    except ValueError as error:
        raise ValueError("extraction output directory is not portable") from error
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    parent = output_directory.parent.resolve(strict=True)
    if not parent.is_dir() or output_directory.parent.is_symlink():
        raise ValueError("extraction output parent must be a real directory")
    final_target = parent / output_directory.name
    required_free_bytes = (
        archive_receipt.total_uncompressed_member_bytes
        + archive_policy.minimum_free_bytes_after_extraction
    )
    if shutil.disk_usage(parent).free < required_free_bytes:
        raise OSError("insufficient free space for protected dataset extraction")

    archive = archive_path.resolve(strict=True)
    if archive_path.is_symlink() or not archive.is_file():
        raise ValueError("public dataset archive must be a real regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(archive, flags)
    temporary = Path(mkdtemp(prefix=".cvi-public-extract-", dir=parent))
    published = False
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("public dataset archive must be a regular file")
        if initial.st_size != archive_receipt.archive_bytes:
            raise ValueError("public dataset archive size differs from receipt")
        (
            observed_archive_sha256,
            observed_archive_md5,
        ) = _hash_descriptor(descriptor, archive_policy.read_chunk_bytes)
        if observed_archive_sha256 != archive_receipt.archive_sha256:
            raise ValueError("public dataset archive hash differs from receipt")
        if observed_archive_md5 != archive_receipt.archive_md5:
            raise ValueError("public dataset archive MD5 differs from receipt")
        os.lseek(descriptor, 0, os.SEEK_SET)

        records: list[ExtractedPublicDatasetFile] = []
        expected_directories: set[str] = set()
        extracted_bytes = 0
        with os.fdopen(descriptor, "rb", closefd=False) as archive_stream:
            with zipfile.ZipFile(archive_stream, "r") as bundle:
                infos = bundle.infolist()
                summary = _validate_zip_members(infos, archive_policy)
                _verify_archive_member_summary(summary, archive_receipt, len(infos))
                for info in infos:
                    canonical = _canonical_zip_path(info.filename, archive_policy)
                    parts = PurePosixPath(canonical).parts
                    expected_directories.update(
                        "/".join(parts[:depth])
                        for depth in range(1, len(parts))
                    )
                    destination = temporary.joinpath(*canonical.split("/"))
                    if info.is_dir():
                        expected_directories.add(canonical)
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    observed = 0
                    with bundle.open(info, "r") as source_stream:
                        with destination.open("xb") as target_stream:
                            observed = _copy_hashed(
                                source_stream,
                                target_stream,
                                digest,
                                archive_policy.read_chunk_bytes,
                                info.file_size,
                            )
                            target_stream.flush()
                            os.fsync(target_stream.fileno())
                    os.chmod(destination, 0o600)
                    if observed != info.file_size:
                        raise ValueError("extracted member byte size differs")
                    extracted_bytes += observed
                    records.append(
                        ExtractedPublicDatasetFile(
                            relative_path=canonical,
                            byte_size=observed,
                            content_sha256=digest.hexdigest(),
                        )
                    )

        final = os.fstat(descriptor)
        if _stat_identity(initial) != _stat_identity(final):
            raise RuntimeError("public dataset archive changed during extraction")
        named = os.stat(archive, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (final.st_dev, final.st_ino):
            raise RuntimeError("public dataset archive path changed during extraction")
        if len(records) != archive_receipt.regular_files:
            raise RuntimeError("extracted regular file count differs from receipt")
        if extracted_bytes != archive_receipt.total_uncompressed_member_bytes:
            raise RuntimeError("extracted byte count differs from receipt")
        ordered = tuple(sorted(records, key=lambda item: item.relative_path))
        manifest_sha256 = content_sha256([item.to_dict() for item in ordered])
        _verify_extracted_tree(
            temporary,
            files=ordered,
            expected_directories=expected_directories,
            chunk_bytes=archive_policy.read_chunk_bytes,
        )
        fsync_directory(temporary)
        staged_identity = temporary.stat()
        publication_strategy = rename_directory_noreplace(temporary, final_target)
        published = True
        published_identity = final_target.stat()
        if (staged_identity.st_dev, staged_identity.st_ino) != (
            published_identity.st_dev,
            published_identity.st_ino,
        ):
            raise RuntimeError("published extraction directory identity differs")
        _verify_extracted_tree(
            final_target,
            files=ordered,
            expected_directories=expected_directories,
            chunk_bytes=archive_policy.read_chunk_bytes,
        )
        fsync_directory(final_target)
        fsync_directory(parent)
    except BaseException:
        if not published and temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    finally:
        os.close(descriptor)

    receipt = PublicDatasetExtractionReceipt(
        source_contract_sha256=source.contract_sha256,
        archive_policy_sha256=archive_policy.policy_sha256,
        archive_receipt_sha256=archive_receipt.receipt_sha256,
        archive_sha256=archive_receipt.archive_sha256,
        output_directory_name=final_target.name,
        file_content_manifest_sha256=manifest_sha256,
        extracted_regular_files=len(ordered),
        extracted_bytes=extracted_bytes,
        publication_strategy=publication_strategy,
    )
    return receipt, ordered


def _verify_receipt_bindings(
    source: PublicDatasetSourceContract,
    policy: PublicDatasetArchivePolicy,
    receipt: PublicDatasetArchiveReceipt,
) -> None:
    if receipt.source_contract_sha256 != source.contract_sha256:
        raise ValueError("archive receipt source contract differs")
    if receipt.archive_policy_sha256 != policy.policy_sha256:
        raise ValueError("archive receipt policy differs")
    if receipt.checksum_authority is not source.checksum_authority:
        raise ValueError("archive receipt checksum authority differs")
    if receipt.archive_bytes != source.expected_archive_bytes:
        raise ValueError("archive receipt byte size differs from source")
    if (
        source.expected_sha256 is not None
        and receipt.archive_sha256 != source.expected_sha256
    ):
        raise ValueError("archive receipt SHA-256 differs from source")
    if source.expected_md5 is not None and receipt.archive_md5 != source.expected_md5:
        raise ValueError("archive receipt MD5 differs from source")
    expected_crc_files = receipt.regular_files if policy.verify_member_crc else 0
    if receipt.crc_verified_files != expected_crc_files:
        raise ValueError("archive receipt CRC verification differs from policy")
    expected_decision = (
        "PASS_RESEARCH_INTAKE_SOURCE_CHECKSUM_UNAVAILABLE"
        if source.checksum_authority.value == "SOURCE_CHECKSUM_UNAVAILABLE"
        else "PASS_PUBLISHED_CHECKSUM"
    )
    if receipt.decision != expected_decision:
        raise ValueError("archive receipt decision differs from source")


def _verify_archive_member_summary(
    summary: dict[str, Any],
    receipt: PublicDatasetArchiveReceipt,
    member_count: int,
) -> None:
    observed = (
        summary["manifest_sha256"],
        member_count,
        summary["regular_files"],
        summary["directories"],
        summary["compressed_bytes"],
        summary["uncompressed_bytes"],
        summary["maximum_member_bytes"],
        summary["path_utf8_bytes"],
        summary["maximum_ratio"],
        summary["suffix_counts"],
    )
    expected = (
        receipt.member_manifest_sha256,
        receipt.total_members,
        receipt.regular_files,
        receipt.directories,
        receipt.total_compressed_member_bytes,
        receipt.total_uncompressed_member_bytes,
        receipt.maximum_member_uncompressed_bytes,
        receipt.total_path_utf8_bytes,
        receipt.maximum_observed_compression_ratio,
        receipt.suffix_counts,
    )
    if observed != expected:
        raise ValueError("archive member summary differs from receipt")


def _hash_descriptor(descriptor: int, chunk_bytes: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        payload = os.read(descriptor, chunk_bytes)
        if not payload:
            break
        sha256.update(payload)
        md5.update(payload)
    return sha256.hexdigest(), md5.hexdigest()


def _copy_hashed(
    source: BinaryIO,
    target: BinaryIO,
    digest: Any,
    chunk_bytes: int,
    maximum_bytes: int,
) -> int:
    observed = 0
    while True:
        payload = source.read(chunk_bytes)
        if not payload:
            break
        observed += len(payload)
        if observed > maximum_bytes:
            raise ValueError("ZIP member expands beyond declared size")
        target.write(payload)
        digest.update(payload)
    return observed


def _verify_extracted_tree(
    root: Path,
    *,
    files: tuple[ExtractedPublicDatasetFile, ...],
    expected_directories: set[str],
    chunk_bytes: int,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("extracted dataset root is not a real directory")
    expected_files = {item.relative_path: item for item in files}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("extracted dataset contains an unsafe directory")
            relative = path.relative_to(root).as_posix()
            _validate_manifest_relative_path(relative)
            observed_directories.add(relative)
        for name in file_names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("extracted dataset contains an unsafe file")
            relative = path.relative_to(root).as_posix()
            _validate_manifest_relative_path(relative)
            observed_files.add(relative)
    if observed_files != set(expected_files):
        raise RuntimeError("extracted dataset file set differs from manifest")
    if observed_directories != expected_directories:
        raise RuntimeError("extracted dataset directory set differs from archive")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for relative in sorted(expected_files):
        expected = expected_files[relative]
        path = root.joinpath(*relative.split("/"))
        descriptor = os.open(path, flags)
        try:
            initial = os.fstat(descriptor)
            if not stat.S_ISREG(initial.st_mode) or initial.st_size != expected.byte_size:
                raise RuntimeError("extracted dataset file size or type differs")
            digest, _ = _hash_descriptor(descriptor, chunk_bytes)
            final = os.fstat(descriptor)
            if _stat_identity(initial) != _stat_identity(final):
                raise RuntimeError("extracted dataset file changed during verification")
            named = os.stat(path, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (final.st_dev, final.st_ino):
                raise RuntimeError("extracted dataset path changed during verification")
            if digest != expected.content_sha256:
                raise RuntimeError("extracted dataset content differs from manifest")
        finally:
            os.close(descriptor)


def _validate_manifest_relative_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > 1_024
    ):
        raise ValueError("extracted relative_path is unsafe")
    parts = PurePosixPath(value).parts
    if (
        not parts
        or len(parts) > 16
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("extracted relative_path is unsafe")
    for part in parts:
        try:
            _validate_portable_component(part, "extracted relative_path")
        except ValueError as error:
            raise ValueError("extracted relative_path is Windows-ambiguous")
