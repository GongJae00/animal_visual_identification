"""License-bound, extraction-blind intake audit for public ZIP datasets."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from foundation.provenance import content_sha256


class DatasetUsageLane(StrEnum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPLOYMENT_ELIGIBLE_CANDIDATE = "DEPLOYMENT_ELIGIBLE_CANDIDATE"


class SourceChecksumAuthority(StrEnum):
    PUBLISHED_SHA256 = "PUBLISHED_SHA256"
    PUBLISHED_MD5 = "PUBLISHED_MD5"
    SOURCE_CHECKSUM_UNAVAILABLE = "SOURCE_CHECKSUM_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PublicDatasetSourceContract:
    dataset_id: str
    dataset_version: str
    archive_filename: str
    official_page_url: str
    archive_url: str
    license_id: str
    license_url: str
    usage_lane: DatasetUsageLane
    expected_archive_bytes: int
    checksum_authority: SourceChecksumAuthority
    expected_sha256: str | None
    expected_md5: str | None
    terms_snapshot_sha256: str
    schema_version: str = "cvi.public_dataset_source_contract.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_dataset_source_contract.v1":
            raise ValueError("unsupported public dataset source contract")
        for name in ("dataset_id", "dataset_version", "license_id"):
            _require_bounded_text(getattr(self, name), name, maximum=256)
        _require_safe_filename(self.archive_filename)
        _require_https_url(self.official_page_url, "official_page_url")
        _require_https_url(self.archive_url, "archive_url")
        _require_https_url(self.license_url, "license_url")
        _require_positive_int(
            self.expected_archive_bytes,
            "expected_archive_bytes",
        )
        _validate_sha256(self.terms_snapshot_sha256, "terms_snapshot_sha256")
        if self.expected_sha256 is not None:
            _validate_sha256(self.expected_sha256, "expected_sha256")
        if self.expected_md5 is not None:
            _validate_md5(self.expected_md5, "expected_md5")
        if self.checksum_authority is SourceChecksumAuthority.PUBLISHED_SHA256:
            if self.expected_sha256 is None or self.expected_md5 is not None:
                raise ValueError("published SHA-256 contract fields differ")
        elif self.checksum_authority is SourceChecksumAuthority.PUBLISHED_MD5:
            if self.expected_md5 is None or self.expected_sha256 is not None:
                raise ValueError("published MD5 contract fields differ")
        elif self.expected_sha256 is not None or self.expected_md5 is not None:
            raise ValueError("unpublished checksum contract must not claim a digest")
        if (
            self.checksum_authority
            is SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
            and self.usage_lane is not DatasetUsageLane.RESEARCH_ONLY
        ):
            raise ValueError(
                "source-checksum gap is permitted only in the research lane"
            )

    @property
    def contract_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "archive_filename": self.archive_filename,
            "official_page_url": self.official_page_url,
            "archive_url": self.archive_url,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "usage_lane": self.usage_lane.value,
            "expected_archive_bytes": self.expected_archive_bytes,
            "checksum_authority": self.checksum_authority.value,
            "expected_sha256": self.expected_sha256,
            "expected_md5": self.expected_md5,
            "terms_snapshot_sha256": self.terms_snapshot_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PublicDatasetSourceContract:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "source contract")
        values = dict(payload)
        values["usage_lane"] = DatasetUsageLane(values["usage_lane"])
        values["checksum_authority"] = SourceChecksumAuthority(
            values["checksum_authority"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PublicDatasetArchivePolicy:
    maximum_archive_bytes: int = 5_000_000_000
    maximum_members: int = 200_000
    maximum_total_uncompressed_bytes: int = 20_000_000_000
    maximum_member_uncompressed_bytes: int = 500_000_000
    maximum_compression_ratio: float = 200.0
    maximum_path_utf8_bytes: int = 1_024
    maximum_total_path_utf8_bytes: int = 100_000_000
    maximum_path_depth: int = 16
    read_chunk_bytes: int = 1_048_576
    minimum_free_bytes_after_extraction: int = 1_000_000_000
    allowed_file_suffixes: tuple[str, ...] = (
        ".bmp",
        ".csv",
        ".jpeg",
        ".jpg",
        ".json",
        ".md",
        ".png",
        ".txt",
    )
    verify_member_crc: bool = True
    schema_version: str = "cvi.public_dataset_archive_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_dataset_archive_policy.v1":
            raise ValueError("unsupported public dataset archive policy")
        for name in (
            "maximum_archive_bytes",
            "maximum_members",
            "maximum_total_uncompressed_bytes",
            "maximum_member_uncompressed_bytes",
            "maximum_path_utf8_bytes",
            "maximum_total_path_utf8_bytes",
            "maximum_path_depth",
            "read_chunk_bytes",
            "minimum_free_bytes_after_extraction",
        ):
            _require_positive_int(getattr(self, name), name)
        _require_finite_positive(
            self.maximum_compression_ratio,
            "maximum_compression_ratio",
        )
        if not self.allowed_file_suffixes:
            raise ValueError("allowed_file_suffixes must not be empty")
        normalized = tuple(sorted(set(self.allowed_file_suffixes)))
        if normalized != self.allowed_file_suffixes or any(
            not isinstance(item, str)
            or not item.startswith(".")
            or item != item.lower()
            for item in self.allowed_file_suffixes
        ):
            raise ValueError("allowed_file_suffixes must be sorted lowercase suffixes")
        if not isinstance(self.verify_member_crc, bool):
            raise TypeError("verify_member_crc must be boolean")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "maximum_archive_bytes": self.maximum_archive_bytes,
            "maximum_members": self.maximum_members,
            "maximum_total_uncompressed_bytes": (
                self.maximum_total_uncompressed_bytes
            ),
            "maximum_member_uncompressed_bytes": (
                self.maximum_member_uncompressed_bytes
            ),
            "maximum_compression_ratio": self.maximum_compression_ratio,
            "maximum_path_utf8_bytes": self.maximum_path_utf8_bytes,
            "maximum_total_path_utf8_bytes": self.maximum_total_path_utf8_bytes,
            "maximum_path_depth": self.maximum_path_depth,
            "read_chunk_bytes": self.read_chunk_bytes,
            "minimum_free_bytes_after_extraction": (
                self.minimum_free_bytes_after_extraction
            ),
            "allowed_file_suffixes": list(self.allowed_file_suffixes),
            "verify_member_crc": self.verify_member_crc,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PublicDatasetArchivePolicy:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "archive policy")
        values = dict(payload)
        suffixes = values["allowed_file_suffixes"]
        if not isinstance(suffixes, list):
            raise TypeError("allowed_file_suffixes must be a list")
        values["allowed_file_suffixes"] = tuple(suffixes)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ArchiveSuffixCount:
    suffix: str
    files: int

    def __post_init__(self) -> None:
        if not isinstance(self.suffix, str) or not self.suffix.startswith("."):
            raise ValueError("archive suffix must be explicit")
        _require_positive_int(self.files, "files")

    def to_dict(self) -> dict[str, str | int]:
        return {"suffix": self.suffix, "files": self.files}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArchiveSuffixCount:
        _require_exact_keys(payload, {"suffix", "files"}, "suffix count")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PublicDatasetArchiveReceipt:
    source_contract_sha256: str
    archive_policy_sha256: str
    archive_sha256: str
    archive_md5: str
    archive_bytes: int
    member_manifest_sha256: str
    total_members: int
    regular_files: int
    directories: int
    total_compressed_member_bytes: int
    total_uncompressed_member_bytes: int
    maximum_member_uncompressed_bytes: int
    total_path_utf8_bytes: int
    maximum_observed_compression_ratio: float
    crc_verified_files: int
    suffix_counts: tuple[ArchiveSuffixCount, ...]
    checksum_authority: SourceChecksumAuthority
    decision: str
    interpretation: str = "ARCHIVE_INTAKE_ONLY_NOT_DATASET_OR_MODEL_ADMISSION"
    schema_version: str = "cvi.public_dataset_archive_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_dataset_archive_receipt.v1":
            raise ValueError("unsupported public dataset archive receipt")
        for name in (
            "source_contract_sha256",
            "archive_policy_sha256",
            "archive_sha256",
            "member_manifest_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _validate_md5(self.archive_md5, "archive_md5")
        for name in (
            "archive_bytes",
            "total_members",
            "regular_files",
            "directories",
            "total_compressed_member_bytes",
            "total_uncompressed_member_bytes",
            "maximum_member_uncompressed_bytes",
            "total_path_utf8_bytes",
            "crc_verified_files",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        _require_finite_nonnegative(
            self.maximum_observed_compression_ratio,
            "maximum_observed_compression_ratio",
        )
        if self.total_members != self.regular_files + self.directories:
            raise ValueError("archive member accounting differs")
        if self.crc_verified_files not in {0, self.regular_files}:
            raise ValueError("archive CRC accounting differs")
        suffixes = tuple(item.suffix for item in self.suffix_counts)
        if suffixes != tuple(sorted(suffixes)) or len(suffixes) != len(set(suffixes)):
            raise ValueError("archive suffix counts must be canonical")
        expected_decision = (
            "PASS_PUBLISHED_CHECKSUM"
            if self.checksum_authority
            is not SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
            else "PASS_RESEARCH_INTAKE_SOURCE_CHECKSUM_UNAVAILABLE"
        )
        if self.decision != expected_decision:
            raise ValueError("archive receipt decision differs")
        if self.interpretation != (
            "ARCHIVE_INTAKE_ONLY_NOT_DATASET_OR_MODEL_ADMISSION"
        ):
            raise ValueError("archive receipt interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract_sha256": self.source_contract_sha256,
            "archive_policy_sha256": self.archive_policy_sha256,
            "archive_sha256": self.archive_sha256,
            "archive_md5": self.archive_md5,
            "archive_bytes": self.archive_bytes,
            "member_manifest_sha256": self.member_manifest_sha256,
            "total_members": self.total_members,
            "regular_files": self.regular_files,
            "directories": self.directories,
            "total_compressed_member_bytes": self.total_compressed_member_bytes,
            "total_uncompressed_member_bytes": self.total_uncompressed_member_bytes,
            "maximum_member_uncompressed_bytes": (
                self.maximum_member_uncompressed_bytes
            ),
            "total_path_utf8_bytes": self.total_path_utf8_bytes,
            "maximum_observed_compression_ratio": (
                self.maximum_observed_compression_ratio
            ),
            "crc_verified_files": self.crc_verified_files,
            "suffix_counts": [item.to_dict() for item in self.suffix_counts],
            "checksum_authority": self.checksum_authority.value,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> PublicDatasetArchiveReceipt:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "archive receipt")
        values = dict(payload)
        suffixes = values["suffix_counts"]
        if not isinstance(suffixes, list):
            raise TypeError("suffix_counts must be a list")
        values["suffix_counts"] = tuple(
            ArchiveSuffixCount.from_dict(item) for item in suffixes
        )
        values["checksum_authority"] = SourceChecksumAuthority(
            values["checksum_authority"]
        )
        return cls(**values)


def audit_public_dataset_zip(
    *,
    archive_path: Path,
    terms_snapshot_path: Path,
    source: PublicDatasetSourceContract,
    policy: PublicDatasetArchivePolicy,
    audit_phase_callback: Callable[[str], None] | None = None,
) -> PublicDatasetArchiveReceipt:
    """Hash and fully CRC-scan a ZIP without extracting any member."""

    if archive_path.name != source.archive_filename:
        raise ValueError("public dataset archive filename differs")
    archive = _real_regular_file(archive_path, "public dataset archive")
    terms = _real_regular_file(terms_snapshot_path, "license terms snapshot")
    terms_sha256, _, _ = _hash_file(terms, policy.read_chunk_bytes)
    if terms_sha256 != source.terms_snapshot_sha256:
        raise ValueError("license terms snapshot hash differs")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(archive, flags)
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("public dataset archive must be a regular file")
        if initial.st_size != source.expected_archive_bytes:
            raise ValueError("public dataset archive byte size differs from source")
        if initial.st_size > policy.maximum_archive_bytes:
            raise ValueError("public dataset archive exceeds policy")
        archive_sha256, archive_md5, hashed_bytes = _hash_descriptor(
            descriptor,
            policy.read_chunk_bytes,
        )
        if hashed_bytes != initial.st_size:
            raise RuntimeError("public dataset archive hash byte count differs")
        if (
            source.expected_sha256 is not None
            and archive_sha256 != source.expected_sha256
        ):
            raise ValueError("public dataset published SHA-256 differs")
        if source.expected_md5 is not None and archive_md5 != source.expected_md5:
            raise ValueError("public dataset published MD5 differs")
        if audit_phase_callback is not None:
            audit_phase_callback("ARCHIVE_HASHED")

        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as archive_stream:
            with zipfile.ZipFile(archive_stream, "r") as bundle:
                infos = bundle.infolist()
                member_summary = _validate_zip_members(infos, policy)
                crc_verified = 0
                if policy.verify_member_crc:
                    for info in infos:
                        if info.is_dir():
                            continue
                        observed = 0
                        with bundle.open(info, "r") as member:
                            while True:
                                chunk = member.read(policy.read_chunk_bytes)
                                if not chunk:
                                    break
                                observed += len(chunk)
                                if observed > info.file_size:
                                    raise ValueError(
                                        "ZIP member expands beyond declared size"
                                    )
                        if observed != info.file_size:
                            raise ValueError("ZIP member extracted byte size differs")
                        crc_verified += 1
                if audit_phase_callback is not None:
                    audit_phase_callback("MEMBERS_SCANNED")

        final = os.fstat(descriptor)
        if _stat_identity(initial) != _stat_identity(final):
            raise RuntimeError("public dataset archive changed during audit")
        named_final = os.stat(archive, follow_symlinks=False)
        if (named_final.st_dev, named_final.st_ino) != (final.st_dev, final.st_ino):
            raise RuntimeError("public dataset archive path changed during audit")
    finally:
        os.close(descriptor)

    decision = (
        "PASS_PUBLISHED_CHECKSUM"
        if source.checksum_authority
        is not SourceChecksumAuthority.SOURCE_CHECKSUM_UNAVAILABLE
        else "PASS_RESEARCH_INTAKE_SOURCE_CHECKSUM_UNAVAILABLE"
    )
    return PublicDatasetArchiveReceipt(
        source_contract_sha256=source.contract_sha256,
        archive_policy_sha256=policy.policy_sha256,
        archive_sha256=archive_sha256,
        archive_md5=archive_md5,
        archive_bytes=initial.st_size,
        member_manifest_sha256=member_summary["manifest_sha256"],
        total_members=len(infos),
        regular_files=member_summary["regular_files"],
        directories=member_summary["directories"],
        total_compressed_member_bytes=member_summary["compressed_bytes"],
        total_uncompressed_member_bytes=member_summary["uncompressed_bytes"],
        maximum_member_uncompressed_bytes=member_summary["maximum_member_bytes"],
        total_path_utf8_bytes=member_summary["path_utf8_bytes"],
        maximum_observed_compression_ratio=member_summary["maximum_ratio"],
        crc_verified_files=crc_verified,
        suffix_counts=member_summary["suffix_counts"],
        checksum_authority=source.checksum_authority,
        decision=decision,
    )


def _validate_zip_members(
    infos: list[zipfile.ZipInfo],
    policy: PublicDatasetArchivePolicy,
) -> dict[str, Any]:
    if not infos:
        raise ValueError("public dataset ZIP must not be empty")
    if len(infos) > policy.maximum_members:
        raise ValueError("public dataset ZIP member count exceeds policy")
    canonical_paths: set[str] = set()
    casefold_paths: set[str] = set()
    casefold_file_paths: set[str] = set()
    required_casefold_directories: set[str] = set()
    suffix_counts: Counter[str] = Counter()
    compressed_bytes = 0
    uncompressed_bytes = 0
    path_utf8_bytes = 0
    maximum_member_bytes = 0
    maximum_ratio = 0.0
    regular_files = 0
    directories = 0
    manifest_rows: list[tuple[Any, ...]] = []
    for info in infos:
        if info.orig_filename != info.filename or "\x00" in info.orig_filename:
            raise ValueError("public dataset ZIP raw path is ambiguous")
        canonical = _canonical_zip_path(info.filename, policy)
        path_utf8_bytes += len(canonical.encode("utf-8"))
        if path_utf8_bytes > policy.maximum_total_path_utf8_bytes:
            raise ValueError("public dataset ZIP aggregate path bytes exceed policy")
        collision_key = canonical.casefold()
        if canonical in canonical_paths or collision_key in casefold_paths:
            raise ValueError("public dataset ZIP contains a path collision")
        parts = PurePosixPath(canonical).parts
        parent_keys = {
            "/".join(parts[:depth]).casefold()
            for depth in range(1, len(parts))
        }
        if parent_keys & casefold_file_paths:
            raise ValueError("public dataset ZIP contains a file-directory conflict")
        if not info.is_dir() and collision_key in required_casefold_directories:
            raise ValueError("public dataset ZIP contains a file-directory conflict")
        canonical_paths.add(canonical)
        casefold_paths.add(collision_key)
        required_casefold_directories.update(parent_keys)
        if not info.is_dir():
            casefold_file_paths.add(collision_key)
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
            raise ValueError("public dataset ZIP contains a symbolic link")
        if info.flag_bits & 0x1:
            raise ValueError("public dataset ZIP contains an encrypted member")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError("public dataset ZIP compression method is unsupported")
        if info.file_size > policy.maximum_member_uncompressed_bytes:
            raise ValueError("public dataset ZIP member exceeds size policy")
        ratio = (
            0.0
            if info.file_size == 0
            else info.file_size / max(1, info.compress_size)
        )
        if ratio > policy.maximum_compression_ratio:
            raise ValueError("public dataset ZIP compression ratio exceeds policy")
        compressed_bytes += info.compress_size
        uncompressed_bytes += info.file_size
        if uncompressed_bytes > policy.maximum_total_uncompressed_bytes:
            raise ValueError("public dataset ZIP expansion exceeds policy")
        maximum_member_bytes = max(maximum_member_bytes, info.file_size)
        maximum_ratio = max(maximum_ratio, ratio)
        if info.is_dir():
            directories += 1
        else:
            regular_files += 1
            suffix = PurePosixPath(canonical).suffix.lower()
            if suffix not in policy.allowed_file_suffixes:
                raise ValueError(f"public dataset ZIP suffix is not allowed: {suffix!r}")
            suffix_counts[suffix] += 1
        manifest_rows.append(
            (
                canonical,
                info.CRC,
                info.compress_type,
                info.compress_size,
                info.file_size,
                info.is_dir(),
            )
        )
    if regular_files == 0:
        raise ValueError("public dataset ZIP contains no regular files")
    return {
        "manifest_sha256": content_sha256(manifest_rows),
        "regular_files": regular_files,
        "directories": directories,
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": uncompressed_bytes,
        "maximum_member_bytes": maximum_member_bytes,
        "path_utf8_bytes": path_utf8_bytes,
        "maximum_ratio": maximum_ratio,
        "suffix_counts": tuple(
            ArchiveSuffixCount(suffix, suffix_counts[suffix])
            for suffix in sorted(suffix_counts)
        ),
    }


def _canonical_zip_path(
    raw_name: str,
    policy: PublicDatasetArchivePolicy,
) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise ValueError("public dataset ZIP path is malformed")
    if "\\" in raw_name or raw_name.startswith("/"):
        raise ValueError("public dataset ZIP path is not relative POSIX")
    if any(ord(character) < 32 for character in raw_name):
        raise ValueError("public dataset ZIP path contains a control character")
    trimmed = raw_name[:-1] if raw_name.endswith("/") else raw_name
    raw_parts = trimmed.split("/")
    if not trimmed or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("public dataset ZIP path contains traversal")
    canonical = unicodedata.normalize("NFC", trimmed)
    if canonical != trimmed:
        raise ValueError("public dataset ZIP path is not NFC-normalized")
    if len(canonical.encode("utf-8")) > policy.maximum_path_utf8_bytes:
        raise ValueError("public dataset ZIP path exceeds byte policy")
    path = PurePosixPath(canonical)
    parts = path.parts
    if not parts or len(parts) > policy.maximum_path_depth:
        raise ValueError("public dataset ZIP path depth differs")
    for part in parts:
        _validate_portable_component(part, "public dataset ZIP path")
    return canonical


def _hash_file(path: Path, chunk_bytes: int) -> tuple[str, str, int]:
    with path.open("rb") as stream:
        return _hash_descriptor(stream.fileno(), chunk_bytes)


def _hash_descriptor(descriptor: int, chunk_bytes: int) -> tuple[str, str, int]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    observed = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, chunk_bytes)
        if not chunk:
            break
        sha256.update(chunk)
        md5.update(chunk)
        observed += len(chunk)
    return sha256.hexdigest(), md5.hexdigest(), observed


def _real_regular_file(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_safe_filename(value: str) -> None:
    _require_bounded_text(value, "archive_filename", maximum=255)
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("archive_filename must be a basename")
    _validate_portable_component(value, "archive_filename")
    if not value.casefold().endswith(".zip"):
        raise ValueError("archive_filename must be a ZIP basename")


def _validate_portable_component(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
        or any(character in '<>:"/\\|?*' for character in value)
        or value.endswith((".", " "))
    ):
        raise ValueError(f"{name} is not a portable path component")
    stem = value.split(".", 1)[0].casefold()
    if stem in {"con", "prn", "aux", "nul"} or (
        len(stem) == 4
        and stem[:3] in {"com", "lpt"}
        and stem[3] in "123456789"
    ):
        raise ValueError(f"{name} uses a Windows reserved name")


def _require_https_url(value: str, name: str) -> None:
    _require_bounded_text(value, name, maximum=4_096)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise ValueError(f"{name} must be an anonymous HTTPS URL")


def _require_bounded_text(value: str, name: str, *, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty and bounded")


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{name} fields differ")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_sha256(value: str, name: str) -> None:
    _validate_hex_digest(value, name, length=64)


def _validate_md5(value: str, name: str) -> None:
    _validate_hex_digest(value, name, length=32)


def _validate_hex_digest(value: str, name: str, *, length: int) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
