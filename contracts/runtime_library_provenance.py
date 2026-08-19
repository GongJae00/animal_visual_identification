"""Phase-aware provenance for file-backed executable process mappings."""

from __future__ import annotations

import hashlib
import os
import platform
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from foundation.provenance import content_sha256


class RuntimeLibraryPhase(StrEnum):
    DEPENDENCIES_IMPORTED = "DEPENDENCIES_IMPORTED"
    SESSION_READY = "SESSION_READY"
    FIRST_OUTPUT_READY = "FIRST_OUTPUT_READY"
    FINAL_OUTPUT_READY = "FINAL_OUTPUT_READY"


_PHASES = tuple(RuntimeLibraryPhase)


@dataclass(frozen=True, slots=True)
class ExpectedRuntimeBinary:
    resolved_path: str
    byte_size: int
    content_sha256: str
    schema_version: str = "cvi.expected_runtime_binary.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.expected_runtime_binary.v1":
            raise ValueError("unsupported expected runtime binary schema")
        if not Path(self.resolved_path).is_absolute():
            raise ValueError("expected runtime binary path must be absolute")
        _positive_int(self.byte_size, "runtime binary byte_size")
        _sha256(self.content_sha256, "runtime binary content_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpectedRuntimeBinary:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("expected runtime binary keys mismatch")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class RuntimeLibraryPolicy:
    expected_binaries: tuple[ExpectedRuntimeBinary, ...]
    discovery_binary_set_sha256: str | None = None
    maximum_maps_bytes: int = 8_388_608
    maximum_maps_lines: int = 100_000
    maximum_executable_identities: int = 256
    maximum_individual_binary_bytes: int = 2_147_483_648
    maximum_total_binary_bytes: int = 8_589_934_592
    hash_chunk_bytes: int = 1_048_576
    allow_wsl_driver_projection_device_mismatch: bool = False
    allow_discovery_only: bool = False
    schema_version: str = "cvi.runtime_library_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.runtime_library_policy.v1":
            raise ValueError("unsupported runtime library policy schema")
        for name in (
            "maximum_maps_bytes", "maximum_maps_lines",
            "maximum_executable_identities", "maximum_individual_binary_bytes",
            "maximum_total_binary_bytes", "hash_chunk_bytes",
        ):
            _positive_int(getattr(self, name), name)
        paths = tuple(item.resolved_path for item in self.expected_binaries)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("expected runtime binaries must be unique path-sorted")
        if not self.expected_binaries and self.allow_discovery_only is not True:
            raise ValueError("strict runtime policy requires expected binaries")
        expected_set_sha256 = content_sha256([
            (item.resolved_path, item.byte_size, item.content_sha256)
            for item in self.expected_binaries
        ])
        if self.expected_binaries:
            if self.discovery_binary_set_sha256 is None:
                raise ValueError(
                    "strict runtime policy requires discovery binary-set lineage"
                )
            _sha256(
                self.discovery_binary_set_sha256,
                "discovery binary set hash",
            )
            if self.discovery_binary_set_sha256 != expected_set_sha256:
                raise ValueError(
                    "strict runtime policy differs from discovery binary set"
                )
        elif self.discovery_binary_set_sha256 is not None:
            raise ValueError(
                "discovery-only policy cannot claim binary-set lineage"
            )
        if not isinstance(
            self.allow_wsl_driver_projection_device_mismatch,
            bool,
        ):
            raise TypeError("WSL driver projection policy must be boolean")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_binaries": [item.to_dict() for item in self.expected_binaries],
            "discovery_binary_set_sha256": (
                self.discovery_binary_set_sha256
            ),
            "maximum_maps_bytes": self.maximum_maps_bytes,
            "maximum_maps_lines": self.maximum_maps_lines,
            "maximum_executable_identities": self.maximum_executable_identities,
            "maximum_individual_binary_bytes": self.maximum_individual_binary_bytes,
            "maximum_total_binary_bytes": self.maximum_total_binary_bytes,
            "hash_chunk_bytes": self.hash_chunk_bytes,
            "allow_wsl_driver_projection_device_mismatch": (
                self.allow_wsl_driver_projection_device_mismatch
            ),
            "allow_discovery_only": self.allow_discovery_only,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeLibraryPolicy:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("runtime library policy keys mismatch")
        binaries = payload["expected_binaries"]
        if not isinstance(binaries, list):
            raise TypeError("expected runtime binaries must be a list")
        if any(not isinstance(item, dict) for item in binaries):
            raise TypeError("expected runtime binary must be an object")
        values = dict(payload)
        values["expected_binaries"] = tuple(
            ExpectedRuntimeBinary.from_dict(item) for item in binaries
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RuntimeBinaryEntry:
    resolved_path: str
    device_major: int
    device_minor: int
    inode: int
    byte_size: int
    content_sha256: str
    first_seen_phase: RuntimeLibraryPhase
    last_seen_phase: RuntimeLibraryPhase
    schema_version: str = "cvi.runtime_binary_entry.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.runtime_binary_entry.v1":
            raise ValueError("unsupported runtime binary entry schema")
        if not Path(self.resolved_path).is_absolute():
            raise ValueError("runtime binary path must be absolute")
        for name in ("device_major", "device_minor", "inode"):
            _nonnegative_int(getattr(self, name), name)
        _positive_int(self.byte_size, "runtime binary byte_size")
        _sha256(self.content_sha256, "runtime binary content_sha256")
        if _PHASES.index(self.first_seen_phase) > _PHASES.index(self.last_seen_phase):
            raise ValueError("runtime binary phase interval is reversed")

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["first_seen_phase"] = self.first_seen_phase.value
        payload["last_seen_phase"] = self.last_seen_phase.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeBinaryEntry:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("runtime binary entry keys mismatch")
        values = dict(payload)
        values["first_seen_phase"] = RuntimeLibraryPhase(
            payload["first_seen_phase"]
        )
        values["last_seen_phase"] = RuntimeLibraryPhase(
            payload["last_seen_phase"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RuntimeLibraryManifest:
    policy_sha256: str
    entries: tuple[RuntimeBinaryEntry, ...]
    binary_set_sha256: str
    maps_snapshots: int
    maps_bytes_read: int
    binary_bytes_hashed: int
    provenance_wall_time_ns: int
    decision: str
    hard_failures: tuple[str, ...]
    schema_version: str = "cvi.runtime_library_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.runtime_library_manifest.v1":
            raise ValueError("unsupported runtime library manifest schema")
        _sha256(self.policy_sha256, "runtime policy hash")
        _sha256(self.binary_set_sha256, "runtime binary set hash")
        if tuple(item.resolved_path for item in self.entries) != tuple(
            sorted(item.resolved_path for item in self.entries)
        ):
            raise ValueError("runtime entries must be path-sorted")
        for name in (
            "maps_snapshots", "maps_bytes_read", "binary_bytes_hashed",
            "provenance_wall_time_ns",
        ):
            _nonnegative_int(getattr(self, name), name)
        if self.decision not in {"PASS", "FAIL", "DISCOVERY_ONLY"}:
            raise ValueError("runtime provenance decision is invalid")
        if self.hard_failures != tuple(sorted(set(self.hard_failures))):
            raise ValueError("runtime provenance failures must be sorted unique")
        if (self.decision == "FAIL") != bool(self.hard_failures):
            raise ValueError("runtime provenance decision/failures differ")
        expected_set_sha256 = content_sha256([
            (item.resolved_path, item.byte_size, item.content_sha256)
            for item in self.entries
        ])
        if expected_set_sha256 != self.binary_set_sha256:
            raise ValueError("runtime binary set hash differs from entries")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_sha256": self.policy_sha256,
            "entries": [item.to_dict() for item in self.entries],
            "binary_set_sha256": self.binary_set_sha256,
            "maps_snapshots": self.maps_snapshots,
            "maps_bytes_read": self.maps_bytes_read,
            "binary_bytes_hashed": self.binary_bytes_hashed,
            "provenance_wall_time_ns": self.provenance_wall_time_ns,
            "decision": self.decision,
            "hard_failures": list(self.hard_failures),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeLibraryManifest:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("runtime library manifest keys mismatch")
        entries = payload["entries"]
        failures = payload["hard_failures"]
        if not isinstance(entries, list) or not isinstance(failures, list):
            raise TypeError("runtime manifest collections must be lists")
        if any(not isinstance(item, dict) for item in entries):
            raise TypeError("runtime manifest entry must be an object")
        values = dict(payload)
        values["entries"] = tuple(
            RuntimeBinaryEntry.from_dict(item) for item in entries
        )
        values["hard_failures"] = tuple(failures)
        return cls(**values)


@dataclass(slots=True)
class _Observed:
    path: Path
    device: int
    inode: int
    fd: int
    initial_stat: os.stat_result
    first: RuntimeLibraryPhase
    last: RuntimeLibraryPhase


class RuntimeLibraryTracker:
    """Retain FDs at phase boundaries and hash only after timed work."""

    def __init__(self, policy: RuntimeLibraryPolicy, maps_path: Path = Path("/proc/self/maps")):
        self.policy = policy
        self.maps_path = maps_path
        self._observed: dict[tuple[int, int], _Observed] = {}
        self._phase_index = -1
        self._snapshots = 0
        self._maps_bytes = 0
        self._started = perf_counter_ns()
        self._closed = False

    def capture(self, phase: RuntimeLibraryPhase) -> None:
        if self._closed or _PHASES.index(phase) != self._phase_index + 1:
            raise ValueError("runtime library phases must be captured exactly once in order")
        payload = self.maps_path.read_bytes()
        if len(payload) > self.policy.maximum_maps_bytes:
            raise ValueError("process maps exceed runtime policy")
        mappings = parse_executable_mappings(
            payload, maximum_lines=self.policy.maximum_maps_lines
        )
        if len(mappings) > self.policy.maximum_executable_identities:
            raise ValueError("executable mapping count exceeds runtime policy")
        for device_major, device_minor, inode, raw_path in mappings:
            device = os.makedev(device_major, device_minor)
            key = (device, inode)
            existing = self._observed.get(key)
            if existing is not None:
                if existing.path != Path(raw_path):
                    raise ValueError("one executable inode has multiple path aliases")
                existing.last = phase
                continue
            path = Path(raw_path)
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            observed_stat = os.fstat(fd)
            if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_nlink == 0:
                os.close(fd)
                raise ValueError("executable mapping is not a linked regular file")
            exact_identity = (
                observed_stat.st_dev == device
                and observed_stat.st_ino == inode
            )
            projected_wsl_identity = (
                self.policy.allow_wsl_driver_projection_device_mismatch
                and observed_stat.st_ino == inode
                and _is_wsl_driver_projection(path, device, observed_stat.st_dev)
            )
            if not exact_identity and not projected_wsl_identity:
                os.close(fd)
                raise ValueError(
                    "executable mapping path identity differs from maps: "
                    f"{path}"
                )
            self._observed[key] = _Observed(
                path=path, device=device, inode=inode, fd=fd,
                initial_stat=observed_stat, first=phase, last=phase,
            )
        self._phase_index += 1
        self._snapshots += 1
        self._maps_bytes += len(payload)

    def finalize(self) -> RuntimeLibraryManifest:
        if self._closed or self._phase_index != len(_PHASES) - 1:
            raise ValueError("all runtime library phases must precede finalize")
        entries: list[RuntimeBinaryEntry] = []
        total_bytes = 0
        try:
            for observed in self._observed.values():
                before = os.fstat(observed.fd)
                identity_fields = (
                    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"
                )
                if any(
                    getattr(observed.initial_stat, name) != getattr(before, name)
                    for name in identity_fields
                ):
                    raise RuntimeError("runtime binary changed after first observation")
                if before.st_size <= 0 or before.st_size > self.policy.maximum_individual_binary_bytes:
                    raise ValueError("runtime binary size exceeds policy")
                digest = hashlib.sha256()
                os.lseek(observed.fd, 0, os.SEEK_SET)
                while True:
                    chunk = os.read(observed.fd, self.policy.hash_chunk_bytes)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(observed.fd)
                named = observed.path.stat()
                if any(getattr(before, name) != getattr(after, name) for name in identity_fields) or (
                    named.st_dev != after.st_dev or named.st_ino != after.st_ino
                ):
                    raise RuntimeError("runtime binary changed while hashing")
                total_bytes += after.st_size
                entries.append(RuntimeBinaryEntry(
                    resolved_path=str(observed.path),
                    device_major=os.major(after.st_dev),
                    device_minor=os.minor(after.st_dev),
                    inode=after.st_ino,
                    byte_size=after.st_size,
                    content_sha256=digest.hexdigest(),
                    first_seen_phase=observed.first,
                    last_seen_phase=observed.last,
                ))
            if total_bytes > self.policy.maximum_total_binary_bytes:
                raise ValueError("runtime binary bytes exceed policy")
            final_maps = self.maps_path.read_bytes()
            if len(final_maps) > self.policy.maximum_maps_bytes:
                raise ValueError("final process maps exceed runtime policy")
            final_mappings = parse_executable_mappings(
                final_maps,
                maximum_lines=self.policy.maximum_maps_lines,
            )
            final_keys = {
                (os.makedev(major, minor), inode)
                for major, minor, inode, _ in final_mappings
            }
            if final_keys != set(self._observed):
                raise RuntimeError("executable mapping set changed after final phase")
            self._snapshots += 1
            self._maps_bytes += len(final_maps)
        finally:
            for observed in self._observed.values():
                os.close(observed.fd)
            self._closed = True
        ordered = tuple(sorted(entries, key=lambda item: item.resolved_path))
        expected = {
            (item.resolved_path, item.byte_size, item.content_sha256)
            for item in self.policy.expected_binaries
        }
        actual = {
            (item.resolved_path, item.byte_size, item.content_sha256)
            for item in ordered
        }
        failures = () if expected == actual else ("RUNTIME_BINARY_SET_DIFFERS",)
        if self.policy.allow_discovery_only and not expected:
            decision, failures = "DISCOVERY_ONLY", ()
        else:
            decision = "FAIL" if failures else "PASS"
        return RuntimeLibraryManifest(
            policy_sha256=self.policy.policy_sha256,
            entries=ordered,
            binary_set_sha256=content_sha256([
                (item.resolved_path, item.byte_size, item.content_sha256)
                for item in ordered
            ]),
            maps_snapshots=self._snapshots,
            maps_bytes_read=self._maps_bytes,
            binary_bytes_hashed=total_bytes,
            provenance_wall_time_ns=perf_counter_ns() - self._started,
            decision=decision,
            hard_failures=failures,
        )


def parse_executable_mappings(
    payload: bytes,
    *,
    maximum_lines: int,
) -> tuple[tuple[int, int, int, str], ...]:
    """Return unique executable file identities from Linux proc maps bytes."""

    lines = payload.splitlines()
    if len(lines) > maximum_lines:
        raise ValueError("process maps line count exceeds policy")
    found: dict[tuple[int, int, int], str] = {}
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) < 5:
            raise ValueError("malformed process maps record")
        permissions = fields[1]
        if b"x" not in permissions:
            continue
        if len(fields) != 6:
            raise ValueError("anonymous executable mapping is forbidden")
        path_bytes = fields[5]
        if path_bytes in {b"[vdso]", b"[vsyscall]"}:
            continue
        if path_bytes.startswith(b"[") or path_bytes.endswith(b" (deleted)"):
            raise ValueError("special or deleted executable mapping is forbidden")
        try:
            device_major, device_minor = (
                int(value, 16) for value in fields[3].split(b":", 1)
            )
            inode = int(fields[4])
        except (ValueError, TypeError) as error:
            raise ValueError("invalid process maps device or inode") from error
        if inode <= 0 or not path_bytes.startswith(b"/"):
            raise ValueError("executable mapping must be an absolute file identity")
        path = os.fsdecode(path_bytes)
        key = (device_major, device_minor, inode)
        previous = found.setdefault(key, path)
        if previous != path:
            raise ValueError("one executable identity has multiple map paths")
    return tuple((*key, path) for key, path in sorted(found.items()))


def freeze_runtime_library_policy(
    discovery_policy: RuntimeLibraryPolicy,
    manifests: tuple[RuntimeLibraryManifest, ...],
) -> RuntimeLibraryPolicy:
    """Convert consistent discovery manifests into one strict candidate policy."""

    if (
        discovery_policy.expected_binaries
        or discovery_policy.allow_discovery_only is not True
        or discovery_policy.discovery_binary_set_sha256 is not None
    ):
        raise ValueError("runtime policy freeze requires discovery-only policy")
    if not manifests:
        raise ValueError("runtime policy freeze requires discovery manifests")
    reference_entries = tuple(
        (item.resolved_path, item.byte_size, item.content_sha256)
        for item in manifests[0].entries
    )
    if not reference_entries:
        raise ValueError("runtime discovery manifest is empty")
    for manifest in manifests:
        entries = tuple(
            (item.resolved_path, item.byte_size, item.content_sha256)
            for item in manifest.entries
        )
        if manifest.policy_sha256 != discovery_policy.policy_sha256:
            raise ValueError("runtime discovery manifest policy differs")
        if manifest.decision != "DISCOVERY_ONLY" or manifest.hard_failures:
            raise ValueError("runtime discovery manifest is not discovery-only")
        if (
            entries != reference_entries
            or manifest.binary_set_sha256
            != manifests[0].binary_set_sha256
        ):
            raise ValueError("runtime discovery binary sets differ")
    expected = tuple(
        ExpectedRuntimeBinary(
            resolved_path=path,
            byte_size=byte_size,
            content_sha256=digest,
        )
        for path, byte_size, digest in reference_entries
    )
    return RuntimeLibraryPolicy(
        expected_binaries=expected,
        discovery_binary_set_sha256=manifests[0].binary_set_sha256,
        maximum_maps_bytes=discovery_policy.maximum_maps_bytes,
        maximum_maps_lines=discovery_policy.maximum_maps_lines,
        maximum_executable_identities=(
            discovery_policy.maximum_executable_identities
        ),
        maximum_individual_binary_bytes=(
            discovery_policy.maximum_individual_binary_bytes
        ),
        maximum_total_binary_bytes=(
            discovery_policy.maximum_total_binary_bytes
        ),
        hash_chunk_bytes=discovery_policy.hash_chunk_bytes,
        allow_wsl_driver_projection_device_mismatch=(
            discovery_policy.allow_wsl_driver_projection_device_mismatch
        ),
        allow_discovery_only=False,
    )


def _is_wsl_driver_projection(
    path: Path,
    mapped_device: int,
    file_device: int,
) -> bool:
    release = platform.release().lower()
    try:
        within_driver_root = path.is_relative_to(Path("/usr/lib/wsl/lib"))
    except ValueError:
        within_driver_root = False
    return (
        "microsoft" in release
        and within_driver_root
        and os.major(mapped_device) == 0
        and os.major(file_device) == 0
    )


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
