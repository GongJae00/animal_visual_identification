"""Fixed-commit, source-only intake for Meta ThreatExchange PDQ.

The boundary validates official GitHub commit/tree snapshots and a codeload
tarball, then retains only an explicitly allowlisted source subset.  It never
imports, builds, links, or executes upstream code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tarfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from typing import Any, BinaryIO
from urllib.parse import urlsplit

from cvi.protected_publication import fsync_directory, rename_directory_noreplace
from cvi.pretrained_supporting_asset_intake import (
    parse_bounded_strict_json_object,
)
from cvi.provenance import content_sha256
from cvi.retained_file import read_retained_regular_file


_ARCHIVE_AUTHORITY = "OBSERVED_SHA256_ONLY_NO_PUBLISHER_ARCHIVE_CHECKSUM"
_LICENSE_CLASSIFICATION = "MANUALLY_CLASSIFIED_EXACT_ROOT_LICENSE_BSD_3_CLAUSE"
_INTERPRETATION = (
    "FIXED_COMMIT_SOURCE_INTAKE_ONLY_NOT_BUILD_EXECUTION_ALGORITHM_OR_"
    "PERFORMANCE_ADMISSION"
)


@dataclass(frozen=True, slots=True)
class PdqSelectedSourceMember:
    relative_path: str
    expected_bytes: int
    git_blob_sha1: str
    content_sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path, "selected source path")
        _require_nonnegative_int(self.expected_bytes, "expected_bytes")
        _validate_hex_digest(self.git_blob_sha1, 40, "git_blob_sha1")
        _validate_hex_digest(self.content_sha256, 64, "content_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "expected_bytes": self.expected_bytes,
            "git_blob_sha1": self.git_blob_sha1,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PdqSelectedSourceMember:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "source member")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PdqSourceIntakePolicy:
    maximum_archive_bytes: int = 500_000_000
    maximum_members: int = 5_000
    maximum_total_uncompressed_bytes: int = 1_000_000_000
    maximum_member_uncompressed_bytes: int = 50_000_000
    maximum_expansion_ratio: float = 20.0
    maximum_path_utf8_bytes: int = 1_024
    maximum_total_path_utf8_bytes: int = 5_000_000
    maximum_path_depth: int = 24
    maximum_api_snapshot_bytes: int = 4_000_000
    read_chunk_bytes: int = 1_048_576
    schema_version: str = "cvi.pdq_source_intake_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_source_intake_policy.v1":
            raise ValueError("unsupported PDQ source intake policy")
        for name in (
            "maximum_archive_bytes",
            "maximum_members",
            "maximum_total_uncompressed_bytes",
            "maximum_member_uncompressed_bytes",
            "maximum_path_utf8_bytes",
            "maximum_total_path_utf8_bytes",
            "maximum_path_depth",
            "maximum_api_snapshot_bytes",
            "read_chunk_bytes",
        ):
            _require_positive_int(getattr(self, name), name)
        if (
            isinstance(self.maximum_expansion_ratio, bool)
            or not isinstance(self.maximum_expansion_ratio, (int, float))
            or not math.isfinite(self.maximum_expansion_ratio)
            or self.maximum_expansion_ratio <= 0
        ):
            raise ValueError("maximum_expansion_ratio must be finite and positive")

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
            "maximum_expansion_ratio": self.maximum_expansion_ratio,
            "maximum_path_utf8_bytes": self.maximum_path_utf8_bytes,
            "maximum_total_path_utf8_bytes": self.maximum_total_path_utf8_bytes,
            "maximum_path_depth": self.maximum_path_depth,
            "maximum_api_snapshot_bytes": self.maximum_api_snapshot_bytes,
            "read_chunk_bytes": self.read_chunk_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PdqSourceIntakePolicy:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "intake policy")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PdqSourceContract:
    repository: str
    commit_sha: str
    tree_sha: str
    official_repository_url: str
    commit_api_url: str
    tree_api_url: str
    codeload_url: str
    archive_filename: str
    archive_root: str
    license_path: str
    license_id: str
    license_classification: str
    license_git_blob_sha1: str
    license_content_sha256: str
    license_bytes: int
    require_verified_commit: bool
    selected_members: tuple[PdqSelectedSourceMember, ...]
    forbidden_selected_paths: tuple[str, ...]
    policy: PdqSourceIntakePolicy
    schema_version: str = "cvi.pdq_source_contract.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_source_contract.v1":
            raise ValueError("unsupported PDQ source contract")
        if self.repository != "facebook/ThreatExchange":
            raise ValueError("PDQ source repository differs")
        _validate_hex_digest(self.commit_sha, 40, "commit_sha")
        _validate_hex_digest(self.tree_sha, 40, "tree_sha")
        expected_urls = {
            "official_repository_url": "https://github.com/facebook/ThreatExchange",
            "commit_api_url": (
                "https://api.github.com/repos/facebook/ThreatExchange/commits/"
                f"{self.commit_sha}"
            ),
            "tree_api_url": (
                "https://api.github.com/repos/facebook/ThreatExchange/git/trees/"
                f"{self.tree_sha}?recursive=1"
            ),
            "codeload_url": (
                "https://codeload.github.com/facebook/ThreatExchange/tar.gz/"
                f"{self.commit_sha}"
            ),
        }
        for name, expected in expected_urls.items():
            _require_anonymous_https_url(getattr(self, name), name)
            if getattr(self, name) != expected:
                raise ValueError(f"{name} differs from fixed official URL")
        _validate_portable_component(self.archive_filename, "archive_filename")
        if not self.archive_filename.endswith(".tar.gz.partial"):
            raise ValueError("archive_filename must retain the .tar.gz.partial suffix")
        _validate_portable_component(self.archive_root, "archive_root")
        if self.archive_root != f"ThreatExchange-{self.commit_sha}":
            raise ValueError("archive_root differs from fixed codeload root")
        _validate_relative_path(self.license_path, "license_path")
        if self.license_path != "LICENSE" or self.license_id != "BSD-3-Clause":
            raise ValueError("PDQ root license identity differs")
        if self.license_classification != _LICENSE_CLASSIFICATION:
            raise ValueError("PDQ license classification differs")
        _validate_hex_digest(
            self.license_git_blob_sha1, 40, "license_git_blob_sha1"
        )
        _validate_hex_digest(
            self.license_content_sha256, 64, "license_content_sha256"
        )
        _require_positive_int(self.license_bytes, "license_bytes")
        if not isinstance(self.require_verified_commit, bool):
            raise TypeError("require_verified_commit must be boolean")
        if not isinstance(self.policy, PdqSourceIntakePolicy):
            raise TypeError("policy must be a PdqSourceIntakePolicy")
        if not self.selected_members:
            raise ValueError("selected_members must not be empty")
        paths = tuple(item.relative_path for item in self.selected_members)
        if paths != tuple(sorted(paths, key=str.casefold)) or len(paths) != len(
            set(path.casefold() for path in paths)
        ):
            raise ValueError("selected_members must be casefold-sorted and unique")
        license_members = tuple(
            item for item in self.selected_members if item.relative_path == "LICENSE"
        )
        if len(license_members) != 1:
            raise ValueError("selected_members must contain root LICENSE exactly once")
        license_member = license_members[0]
        if (
            license_member.expected_bytes != self.license_bytes
            or license_member.git_blob_sha1 != self.license_git_blob_sha1
            or license_member.content_sha256 != self.license_content_sha256
        ):
            raise ValueError("root LICENSE member binding differs")
        if self.forbidden_selected_paths != (
            "pdq/cpp/CImg.h",
            "pdq/cpp/io",
        ):
            raise ValueError("forbidden_selected_paths differs from frozen boundary")
        if any(
            _path_is_equal_or_below(path, forbidden)
            for path in paths
            for forbidden in self.forbidden_selected_paths
        ):
            raise ValueError("selected_members includes a forbidden PDQ path")

    @property
    def contract_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "official_repository_url": self.official_repository_url,
            "commit_api_url": self.commit_api_url,
            "tree_api_url": self.tree_api_url,
            "codeload_url": self.codeload_url,
            "archive_filename": self.archive_filename,
            "archive_root": self.archive_root,
            "license_path": self.license_path,
            "license_id": self.license_id,
            "license_classification": self.license_classification,
            "license_git_blob_sha1": self.license_git_blob_sha1,
            "license_content_sha256": self.license_content_sha256,
            "license_bytes": self.license_bytes,
            "require_verified_commit": self.require_verified_commit,
            "selected_members": [item.to_dict() for item in self.selected_members],
            "forbidden_selected_paths": list(self.forbidden_selected_paths),
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PdqSourceContract:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "source contract")
        values = dict(payload)
        if not isinstance(values["selected_members"], list):
            raise TypeError("selected_members must be a list")
        if not isinstance(values["forbidden_selected_paths"], list):
            raise TypeError("forbidden_selected_paths must be a list")
        values["selected_members"] = tuple(
            PdqSelectedSourceMember.from_dict(item)
            for item in values["selected_members"]
        )
        values["forbidden_selected_paths"] = tuple(
            values["forbidden_selected_paths"]
        )
        values["policy"] = PdqSourceIntakePolicy.from_dict(values["policy"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PdqRetainedSourceMember:
    relative_path: str
    byte_size: int
    git_blob_sha1: str
    content_sha256: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path, "retained source path")
        _require_nonnegative_int(self.byte_size, "byte_size")
        _validate_hex_digest(self.git_blob_sha1, 40, "git_blob_sha1")
        _validate_hex_digest(self.content_sha256, 64, "content_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "git_blob_sha1": self.git_blob_sha1,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PdqRetainedSourceMember:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "retained member")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PdqSourceIntakeReceipt:
    source_contract_sha256: str
    intake_policy_sha256: str
    commit_sha: str
    tree_sha: str
    commit_api_snapshot_sha256: str
    commit_api_snapshot_bytes: int
    tree_api_snapshot_sha256: str
    tree_api_snapshot_bytes: int
    commit_signature_verified: bool
    commit_signature_reason: str
    archive_sha256: str
    archive_bytes: int
    archive_checksum_authority: str
    total_archive_members: int
    archive_regular_files: int
    archive_directories: int
    total_uncompressed_member_bytes: int
    maximum_member_uncompressed_bytes: int
    total_path_utf8_bytes: int
    retained_members: tuple[PdqRetainedSourceMember, ...]
    retained_source_bytes: int
    retained_source_aggregate_sha256: str
    license_id: str
    license_content_sha256: str
    license_classification: str
    forbidden_selected_paths: tuple[str, ...]
    forbidden_selected_paths_absent: bool
    decision: str = "PASS_FIXED_COMMIT_BSD_3_CLAUSE_SOURCE_ONLY"
    interpretation: str = _INTERPRETATION
    schema_version: str = "cvi.pdq_source_intake_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_source_intake_receipt.v1":
            raise ValueError("unsupported PDQ source intake receipt")
        for name in (
            "source_contract_sha256",
            "intake_policy_sha256",
            "commit_api_snapshot_sha256",
            "tree_api_snapshot_sha256",
            "archive_sha256",
            "retained_source_aggregate_sha256",
            "license_content_sha256",
        ):
            _validate_hex_digest(getattr(self, name), 64, name)
        _validate_hex_digest(self.commit_sha, 40, "commit_sha")
        _validate_hex_digest(self.tree_sha, 40, "tree_sha")
        for name in (
            "commit_api_snapshot_bytes",
            "tree_api_snapshot_bytes",
            "archive_bytes",
            "total_archive_members",
            "archive_regular_files",
            "archive_directories",
            "total_uncompressed_member_bytes",
            "maximum_member_uncompressed_bytes",
            "total_path_utf8_bytes",
            "retained_source_bytes",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        if self.total_archive_members != (
            self.archive_regular_files + self.archive_directories
        ):
            raise ValueError("PDQ archive member accounting differs")
        if not isinstance(self.commit_signature_verified, bool):
            raise TypeError("commit_signature_verified must be boolean")
        _require_canonical_text(
            self.commit_signature_reason,
            "commit_signature_reason",
            maximum=128,
        )
        if self.archive_checksum_authority != _ARCHIVE_AUTHORITY:
            raise ValueError("PDQ archive checksum authority differs")
        if self.license_id != "BSD-3-Clause":
            raise ValueError("PDQ receipt license differs")
        if self.license_classification != _LICENSE_CLASSIFICATION:
            raise ValueError("PDQ receipt license classification differs")
        if self.forbidden_selected_paths != (
            "pdq/cpp/CImg.h",
            "pdq/cpp/io",
        ) or self.forbidden_selected_paths_absent is not True:
            raise ValueError("PDQ forbidden source exclusion evidence differs")
        paths = tuple(item.relative_path for item in self.retained_members)
        if paths != tuple(sorted(paths, key=str.casefold)) or len(paths) != len(
            set(path.casefold() for path in paths)
        ):
            raise ValueError("PDQ retained members must be sorted and unique")
        if self.retained_source_bytes != sum(
            item.byte_size for item in self.retained_members
        ):
            raise ValueError("PDQ retained source byte accounting differs")
        if self.retained_source_aggregate_sha256 != content_sha256(
            [item.to_dict() for item in self.retained_members]
        ):
            raise ValueError("PDQ retained source aggregate differs")
        if self.decision != "PASS_FIXED_COMMIT_BSD_3_CLAUSE_SOURCE_ONLY":
            raise ValueError("PDQ source intake decision differs")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("PDQ source intake interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract_sha256": self.source_contract_sha256,
            "intake_policy_sha256": self.intake_policy_sha256,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "commit_api_snapshot_sha256": self.commit_api_snapshot_sha256,
            "commit_api_snapshot_bytes": self.commit_api_snapshot_bytes,
            "tree_api_snapshot_sha256": self.tree_api_snapshot_sha256,
            "tree_api_snapshot_bytes": self.tree_api_snapshot_bytes,
            "commit_signature_verified": self.commit_signature_verified,
            "commit_signature_reason": self.commit_signature_reason,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "archive_checksum_authority": self.archive_checksum_authority,
            "total_archive_members": self.total_archive_members,
            "archive_regular_files": self.archive_regular_files,
            "archive_directories": self.archive_directories,
            "total_uncompressed_member_bytes": (
                self.total_uncompressed_member_bytes
            ),
            "maximum_member_uncompressed_bytes": (
                self.maximum_member_uncompressed_bytes
            ),
            "total_path_utf8_bytes": self.total_path_utf8_bytes,
            "retained_members": [item.to_dict() for item in self.retained_members],
            "retained_source_bytes": self.retained_source_bytes,
            "retained_source_aggregate_sha256": (
                self.retained_source_aggregate_sha256
            ),
            "license_id": self.license_id,
            "license_content_sha256": self.license_content_sha256,
            "license_classification": self.license_classification,
            "forbidden_selected_paths": list(self.forbidden_selected_paths),
            "forbidden_selected_paths_absent": self.forbidden_selected_paths_absent,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PdqSourceIntakeReceipt:
        _require_exact_keys(payload, set(cls.__dataclass_fields__), "intake receipt")
        values = dict(payload)
        if not isinstance(values["retained_members"], list):
            raise TypeError("retained_members must be a list")
        values["retained_members"] = tuple(
            PdqRetainedSourceMember.from_dict(item)
            for item in values["retained_members"]
        )
        if not isinstance(values["forbidden_selected_paths"], list):
            raise TypeError("forbidden_selected_paths must be a list")
        values["forbidden_selected_paths"] = tuple(
            values["forbidden_selected_paths"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PdqSourceAuditResult:
    receipt: PdqSourceIntakeReceipt
    retained_payloads: tuple[tuple[str, bytes], ...]


def audit_pdq_source_archive(
    *,
    archive_path: Path,
    commit_api_snapshot_path: Path,
    tree_api_snapshot_path: Path,
    source: PdqSourceContract,
    audit_phase_callback: Callable[[str], None] | None = None,
) -> PdqSourceAuditResult:
    """Audit fixed metadata and source bytes without loading upstream code."""

    if archive_path.name != source.archive_filename:
        raise ValueError("PDQ codeload archive filename differs")
    commit_read = read_retained_regular_file(
        commit_api_snapshot_path,
        maximum_bytes=source.policy.maximum_api_snapshot_bytes,
        capture_payload=True,
        subject="PDQ commit API snapshot",
    )
    tree_read = read_retained_regular_file(
        tree_api_snapshot_path,
        maximum_bytes=source.policy.maximum_api_snapshot_bytes,
        capture_payload=True,
        subject="PDQ tree API snapshot",
    )
    assert commit_read.payload is not None and tree_read.payload is not None
    commit_payload = _parse_strict_json_object(commit_read.payload, "commit snapshot")
    tree_payload = _parse_strict_json_object(tree_read.payload, "tree snapshot")
    verified, signature_reason = _validate_official_api_snapshots(
        commit_payload,
        tree_payload,
        source,
    )
    archive_summary = _audit_retained_tar_archive(
        archive_path,
        source,
        audit_phase_callback,
    )
    retained = tuple(
        PdqRetainedSourceMember(
            relative_path=path,
            byte_size=len(payload),
            git_blob_sha1=_git_blob_sha1(payload),
            content_sha256=hashlib.sha256(payload).hexdigest(),
        )
        for path, payload in archive_summary["retained_payloads"]
    )
    receipt = PdqSourceIntakeReceipt(
        source_contract_sha256=source.contract_sha256,
        intake_policy_sha256=source.policy.policy_sha256,
        commit_sha=source.commit_sha,
        tree_sha=source.tree_sha,
        commit_api_snapshot_sha256=commit_read.sha256,
        commit_api_snapshot_bytes=commit_read.byte_count,
        tree_api_snapshot_sha256=tree_read.sha256,
        tree_api_snapshot_bytes=tree_read.byte_count,
        commit_signature_verified=verified,
        commit_signature_reason=signature_reason,
        archive_sha256=archive_summary["archive_sha256"],
        archive_bytes=archive_summary["archive_bytes"],
        archive_checksum_authority=_ARCHIVE_AUTHORITY,
        total_archive_members=archive_summary["total_members"],
        archive_regular_files=archive_summary["regular_files"],
        archive_directories=archive_summary["directories"],
        total_uncompressed_member_bytes=archive_summary["uncompressed_bytes"],
        maximum_member_uncompressed_bytes=archive_summary["maximum_member_bytes"],
        total_path_utf8_bytes=archive_summary["path_utf8_bytes"],
        retained_members=retained,
        retained_source_bytes=sum(item.byte_size for item in retained),
        retained_source_aggregate_sha256=content_sha256(
            [item.to_dict() for item in retained]
        ),
        license_id=source.license_id,
        license_content_sha256=source.license_content_sha256,
        license_classification=source.license_classification,
        forbidden_selected_paths=source.forbidden_selected_paths,
        forbidden_selected_paths_absent=True,
    )
    return PdqSourceAuditResult(
        receipt=receipt,
        retained_payloads=archive_summary["retained_payloads"],
    )


def publish_pdq_source_bundle(
    *,
    audit: PdqSourceAuditResult,
    source: PdqSourceContract,
    output_directory: Path,
    tool_provenance: dict[str, Any],
) -> str:
    """Atomically publish selected source and its receipt as one directory."""

    if audit.receipt.source_contract_sha256 != source.contract_sha256:
        raise ValueError("PDQ audit result source contract differs")
    expected_rows = tuple(item.to_dict() for item in audit.receipt.retained_members)
    observed_rows = tuple(
        {
            "relative_path": path,
            "byte_size": len(payload),
            "git_blob_sha1": _git_blob_sha1(payload),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in audit.retained_payloads
    )
    if observed_rows != expected_rows:
        raise ValueError("PDQ retained payloads differ from receipt")
    _validate_output_target(output_directory)
    parent = output_directory.parent.resolve(strict=True)
    target = parent / output_directory.name
    stage = Path(mkdtemp(prefix=".cvi-pdq-source-", dir=parent))
    try:
        source_root = stage / "source"
        source_root.mkdir(mode=0o700)
        for relative_path, payload in audit.retained_payloads:
            destination = source_root / PurePosixPath(relative_path)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_new_file(destination, payload)
        bundle = {
            "schema_version": "cvi.pdq_source_intake_bundle.v1",
            "source_contract_sha256": source.contract_sha256,
            "source_contract": source.to_dict(),
            "receipt_sha256": audit.receipt.receipt_sha256,
            "receipt": audit.receipt.to_dict(),
            "tool_provenance": tool_provenance,
            "tool_provenance_sha256": content_sha256(tool_provenance),
            "publication_guarantee": "ATOMIC_DIRECTORY_NOREPLACE",
        }
        bundle_payload = (
            json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        bundle_sha256 = hashlib.sha256(bundle_payload).hexdigest()
        _write_new_file(stage / "intake-bundle.json", bundle_payload)
        _fsync_tree(stage)
        _verify_published_source_tree(stage / "source", audit.receipt)
        _verify_bundle_file(stage, len(bundle_payload), bundle_sha256)
        strategy = rename_directory_noreplace(stage, target)
        _verify_published_source_tree(target / "source", audit.receipt)
        _verify_bundle_file(target, len(bundle_payload), bundle_sha256)
        fsync_directory(parent)
        return strategy
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def _validate_official_api_snapshots(
    commit_payload: dict[str, Any],
    tree_payload: dict[str, Any],
    source: PdqSourceContract,
) -> tuple[bool, str]:
    if commit_payload.get("sha") != source.commit_sha:
        raise ValueError("official GitHub commit SHA differs")
    if commit_payload.get("url") != source.commit_api_url:
        raise ValueError("official GitHub commit API URL differs")
    if commit_payload.get("html_url") != (
        f"{source.official_repository_url}/commit/{source.commit_sha}"
    ):
        raise ValueError("official GitHub commit HTML URL differs")
    commit = commit_payload.get("commit")
    if not isinstance(commit, dict):
        raise ValueError("official GitHub commit object is missing")
    tree = commit.get("tree")
    if not isinstance(tree, dict) or tree.get("sha") != source.tree_sha:
        raise ValueError("official GitHub commit tree SHA differs")
    verification = commit.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("official GitHub commit verification is missing")
    verified = verification.get("verified")
    reason = verification.get("reason")
    if not isinstance(verified, bool):
        raise ValueError("official GitHub commit verification flag differs")
    _require_canonical_text(reason, "commit verification reason", maximum=128)
    if source.require_verified_commit and (not verified or reason != "valid"):
        raise ValueError("fixed PDQ commit lacks a valid GitHub verification")

    if tree_payload.get("sha") != source.tree_sha:
        raise ValueError("official GitHub recursive tree SHA differs")
    if tree_payload.get("truncated") is not False:
        raise ValueError("official GitHub recursive tree is truncated")
    entries = tree_payload.get("tree")
    if not isinstance(entries, list):
        raise ValueError("official GitHub recursive tree entries are missing")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("official GitHub tree entry is malformed")
        path = entry.get("path")
        if not isinstance(path, str) or path in indexed:
            raise ValueError("official GitHub tree path is malformed or duplicated")
        indexed[path] = entry
    for member in source.selected_members:
        entry = indexed.get(member.relative_path)
        if (
            entry is None
            or entry.get("type") != "blob"
            or entry.get("sha") != member.git_blob_sha1
            or entry.get("size") != member.expected_bytes
        ):
            raise ValueError(
                f"official GitHub selected tree member differs: {member.relative_path}"
            )
    return verified, reason


def _audit_retained_tar_archive(
    archive_path: Path,
    source: PdqSourceContract,
    phase_callback: Callable[[str], None] | None,
) -> dict[str, Any]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("PDQ source intake requires O_NOFOLLOW support")
    absolute = Path(os.path.abspath(os.fspath(archive_path)))
    parent_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_CLOEXEC"):
        parent_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    parent_fd = os.open(absolute.parent, parent_flags)
    try:
        parent_initial = os.fstat(parent_fd)
        descriptor = os.open(absolute.name, file_flags, dir_fd=parent_fd)
        try:
            initial = os.fstat(descriptor)
            if not stat.S_ISREG(initial.st_mode):
                raise ValueError("PDQ codeload archive must be a regular file")
            if initial.st_size <= 0 or initial.st_size > source.policy.maximum_archive_bytes:
                raise ValueError("PDQ codeload archive size exceeds policy")
            archive_sha256, observed = _hash_descriptor(
                descriptor, source.policy.read_chunk_bytes
            )
            if observed != initial.st_size:
                raise RuntimeError("PDQ codeload archive hash byte count differs")
            if phase_callback is not None:
                phase_callback("ARCHIVE_HASHED")
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                summary = _scan_tar_stream(stream, initial.st_size, source)
            if phase_callback is not None:
                phase_callback("MEMBERS_SCANNED")
            final = os.fstat(descriptor)
            named_final = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
            parent_final = os.fstat(parent_fd)
            named_parent_final = os.stat(absolute.parent, follow_symlinks=False)
            if _stat_identity(initial) != _stat_identity(final):
                raise RuntimeError("PDQ codeload archive changed during intake")
            if not stat.S_ISREG(named_final.st_mode) or (
                named_final.st_dev,
                named_final.st_ino,
            ) != (final.st_dev, final.st_ino):
                raise RuntimeError("PDQ codeload archive path changed during intake")
            if _directory_identity(parent_initial) != _directory_identity(parent_final):
                raise RuntimeError("PDQ codeload archive parent changed during intake")
            if not stat.S_ISDIR(named_parent_final.st_mode) or (
                named_parent_final.st_dev,
                named_parent_final.st_ino,
            ) != (parent_final.st_dev, parent_final.st_ino):
                raise RuntimeError("PDQ codeload archive parent path changed during intake")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)
    summary["archive_sha256"] = archive_sha256
    summary["archive_bytes"] = observed
    return summary


def _scan_tar_stream(
    stream: BinaryIO,
    archive_bytes: int,
    source: PdqSourceContract,
) -> dict[str, Any]:
    expected = {item.relative_path: item for item in source.selected_members}
    retained: dict[str, bytes] = {}
    canonical_paths: set[str] = set()
    casefold_paths: set[str] = set()
    casefold_files: set[str] = set()
    required_directories: set[str] = set()
    total_members = regular_files = directories = 0
    uncompressed_bytes = maximum_member_bytes = total_path_bytes = 0
    try:
        bundle = tarfile.open(fileobj=stream, mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        raise ValueError("PDQ codeload archive is not a valid gzip tar") from error
    with bundle:
        try:
            for member in bundle:
                total_members += 1
                if total_members > source.policy.maximum_members:
                    raise ValueError("PDQ tar member count exceeds policy")
                canonical = _canonical_tar_path(member.name, source.policy)
                total_path_bytes += len(canonical.encode("utf-8"))
                if total_path_bytes > source.policy.maximum_total_path_utf8_bytes:
                    raise ValueError("PDQ tar aggregate path bytes exceed policy")
                collision_key = canonical.casefold()
                if canonical in canonical_paths or collision_key in casefold_paths:
                    raise ValueError("PDQ tar contains a path collision")
                parts = PurePosixPath(canonical).parts
                parent_keys = {
                    "/".join(parts[:depth]).casefold()
                    for depth in range(1, len(parts))
                }
                if parent_keys & casefold_files or (
                    not member.isdir() and collision_key in required_directories
                ):
                    raise ValueError("PDQ tar contains a file-directory conflict")
                canonical_paths.add(canonical)
                casefold_paths.add(collision_key)
                required_directories.update(parent_keys)
                if not member.isdir():
                    casefold_files.add(collision_key)
                if member.issym():
                    raise ValueError("PDQ tar contains a symbolic link")
                if member.islnk():
                    raise ValueError("PDQ tar contains a hard link")
                if member.isdev() or member.isfifo():
                    raise ValueError("PDQ tar contains a device or FIFO")
                if member.sparse is not None or any(
                    key.startswith("GNU.sparse") for key in member.pax_headers
                ):
                    raise ValueError("PDQ tar contains a sparse member")
                if not (member.isfile() or member.isdir()):
                    raise ValueError("PDQ tar contains an unsupported member type")
                relative = _strip_archive_root(canonical, source.archive_root)
                if member.isdir():
                    directories += 1
                    continue
                regular_files += 1
                _require_nonnegative_int(member.size, "tar member size")
                if member.size > source.policy.maximum_member_uncompressed_bytes:
                    raise ValueError("PDQ tar member exceeds size policy")
                uncompressed_bytes += member.size
                if uncompressed_bytes > source.policy.maximum_total_uncompressed_bytes:
                    raise ValueError("PDQ tar expansion exceeds policy")
                maximum_member_bytes = max(maximum_member_bytes, member.size)
                expected_member = expected.get(relative)
                if expected_member is not None:
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        raise ValueError("PDQ selected tar member cannot be read")
                    payload = _read_exact_bounded_member(
                        extracted,
                        expected_member.expected_bytes,
                        source.policy.read_chunk_bytes,
                    )
                    if _git_blob_sha1(payload) != expected_member.git_blob_sha1:
                        raise ValueError(
                            f"PDQ selected member Git blob differs: {relative}"
                        )
                    if hashlib.sha256(payload).hexdigest() != (
                        expected_member.content_sha256
                    ):
                        raise ValueError(
                            f"PDQ selected member SHA-256 differs: {relative}"
                        )
                    retained[relative] = payload
        except (tarfile.TarError, EOFError, OSError) as error:
            raise ValueError("PDQ codeload tar member scan failed") from error
    ratio = uncompressed_bytes / max(1, archive_bytes)
    if ratio > source.policy.maximum_expansion_ratio:
        raise ValueError("PDQ tar expansion ratio exceeds policy")
    missing = tuple(sorted(set(expected) - set(retained), key=str.casefold))
    if missing:
        raise ValueError("PDQ selected tar members are missing: " + ", ".join(missing))
    return {
        "total_members": total_members,
        "regular_files": regular_files,
        "directories": directories,
        "uncompressed_bytes": uncompressed_bytes,
        "maximum_member_bytes": maximum_member_bytes,
        "path_utf8_bytes": total_path_bytes,
        "retained_payloads": tuple(
            (path, retained[path]) for path in sorted(retained, key=str.casefold)
        ),
    }


def _read_exact_bounded_member(
    stream: BinaryIO,
    expected_bytes: int,
    chunk_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            break
        observed += len(chunk)
        if observed > expected_bytes:
            raise ValueError("PDQ selected tar member expands beyond expected size")
        chunks.append(chunk)
    if observed != expected_bytes:
        raise ValueError("PDQ selected tar member byte size differs")
    return b"".join(chunks)


def _parse_strict_json_object(payload: bytes, subject: str) -> dict[str, Any]:
    """Apply the shared explicit byte and JSON-structure resource bounds."""

    try:
        return parse_bounded_strict_json_object(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"PDQ {subject} violates strict JSON bounds: {error}") from error


def _canonical_tar_path(raw_name: str, policy: PdqSourceIntakePolicy) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise ValueError("PDQ tar path is malformed")
    if "\\" in raw_name or raw_name.startswith("/"):
        raise ValueError("PDQ tar path is not relative POSIX")
    if any(ord(character) < 32 for character in raw_name):
        raise ValueError("PDQ tar path contains a control character")
    trimmed = raw_name[:-1] if raw_name.endswith("/") else raw_name
    raw_parts = trimmed.split("/")
    if not trimmed or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("PDQ tar path contains traversal")
    if unicodedata.normalize("NFC", trimmed) != trimmed:
        raise ValueError("PDQ tar path is not NFC-normalized")
    if len(trimmed.encode("utf-8")) > policy.maximum_path_utf8_bytes:
        raise ValueError("PDQ tar path exceeds byte policy")
    parts = PurePosixPath(trimmed).parts
    if not parts or len(parts) > policy.maximum_path_depth:
        raise ValueError("PDQ tar path depth differs")
    for part in parts:
        _validate_portable_component(part, "PDQ tar path")
    return trimmed


def _strip_archive_root(path: str, root: str) -> str:
    if path == root:
        return ""
    prefix = root + "/"
    if not path.startswith(prefix):
        raise ValueError("PDQ tar member escapes or differs from codeload root")
    return path[len(prefix) :]


def _validate_output_target(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("PDQ output directory must not be a symlink")
    _validate_portable_component(path.name, "PDQ output directory")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)


def _write_new_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("PDQ protected source write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        fsync_directory(directory)
    fsync_directory(root)


def _verify_published_source_tree(
    source_root: Path,
    receipt: PdqSourceIntakeReceipt,
) -> None:
    expected = {item.relative_path: item for item in receipt.retained_members}
    observed_paths: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("published PDQ source contains a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(source_root).as_posix()
        observed_paths.add(relative)
        expected_member = expected.get(relative)
        if expected_member is None:
            raise RuntimeError("published PDQ source contains an unexpected file")
        read = read_retained_regular_file(
            path,
            expected_bytes=expected_member.byte_size,
            expected_sha256=expected_member.content_sha256,
            capture_payload=False,
            subject="published PDQ source member",
        )
        if read.byte_count != expected_member.byte_size:
            raise RuntimeError("published PDQ source member byte size differs")
    if observed_paths != set(expected):
        raise RuntimeError("published PDQ source member set differs")


def _verify_bundle_file(root: Path, expected_bytes: int, expected_sha256: str) -> None:
    read_retained_regular_file(
        root / "intake-bundle.json",
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        capture_payload=False,
        subject="published PDQ intake bundle",
    )


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _hash_descriptor(descriptor: int, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    observed = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, chunk_bytes)
        if not chunk:
            break
        digest.update(chunk)
        observed += len(chunk)
    return digest.hexdigest(), observed


def _validate_relative_path(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise ValueError(f"{name} must be a relative POSIX path")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must be NFC-normalized")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} contains traversal")
    for part in parts:
        _validate_portable_component(part, name)


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


def _require_anonymous_https_url(value: str, name: str) -> None:
    _require_canonical_text(value, name, maximum=4_096)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} must be an anonymous HTTPS URL")


def _path_is_equal_or_below(path: str, prefix: str) -> bool:
    return path.casefold() == prefix.casefold() or path.casefold().startswith(
        prefix.casefold() + "/"
    )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"PDQ {name} fields differ")


def _require_canonical_text(value: str, name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be canonical bounded text")


def _validate_hex_digest(value: str, length: int, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
