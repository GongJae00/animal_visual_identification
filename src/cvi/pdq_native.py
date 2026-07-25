"""Bounded portable native worker and provenance-bound build for Meta PDQ."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import struct
import subprocess
import threading
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from time import monotonic, sleep
from typing import Any, BinaryIO, Iterable

from cvi.pdq_contracts import PDQ_D4_ORIENTATIONS
from cvi.pdq_source_intake import PdqSourceContract, PdqSourceIntakeReceipt
from cvi.protected_publication import fsync_directory, rename_directory_noreplace
from cvi.provenance import content_sha256
from cvi.source_provenance import build_offline_tool_provenance


CANONICAL_INTAKE_BUNDLE_SHA256 = (
    "aaf78c4fa4575ec6cadee519abd4b528063e82b4472f85440e1eb867ae45b9dd"
)
CANONICAL_SOURCE_RECEIPT_SHA256 = (
    "f68b7c966a2010fd95ac35d38c087f420fdab9ae6eb8f24297b007e12d80bcf7"
)
CANONICAL_SOURCE_CONTRACT_SHA256 = (
    "c77d8dc2068a5ab819ad840e88ec71bfa515a8a3505a447ae9f9840dba28b50c"
)
CANONICAL_TOOL_PROVENANCE_SHA256 = (
    "92d5599d59bc128f8482a33cbab602cf889f60d477ee5f8eafb24cbf873b12c6"
)
CANONICAL_RETAINED_AGGREGATE_SHA256 = (
    "ec769191e30740010e45b0e53b1ec0d87b7d1de75fd9492eea99e53b9fb45fd3"
)
CANONICAL_COMMIT_SHA = "baefb4ed67b6cdc1d4c82dbaef858d50866ac424"
CANONICAL_WORKER_SOURCE_SHA256 = (
    "710fa7390fe26951867ac204a48626610b1a5dca23b1e14ed479f0cd92079830"
)
CANONICAL_COMPILER_REALPATH = "/usr/bin/x86_64-linux-gnu-g++-13"
CANONICAL_COMPILER_SHA256 = (
    "1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769"
)
CANONICAL_COMPILER_VERSION_SHA256 = (
    "39553616934606f58b27c04bed0cdbbe5b00c92c831a9aaadebd7ceda61d1334"
)
CANONICAL_COMPILER_VERSION_FIRST_LINE = (
    "x86_64-linux-gnu-g++-13 (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
)
CANONICAL_BINARY_BYTES = 28_400
CANONICAL_BINARY_SHA256 = (
    "b4774cfba578f6dd235b3892e6f880fad5f5c078076eda89bd618cd773aa0ad6"
)
CANONICAL_INTERPRETATION = (
    "PORTABLE_PROTOCOL_AND_METAMORPHIC_SMOKE_ONLY_NOT_PDQ_ACCURACY_"
    "PERFORMANCE_OR_DEPLOYMENT_ADMISSION"
)
CANONICAL_RETAINED_MEMBERS: tuple[tuple[str, int, str], ...] = (
    ("LICENSE", 1522, "68ecc6aafbd2a205a1077f86127030898f03091b7dae9d9017325a8702d8668f"),
    ("pdq/cpp/common/pdqbasetypes.h", 495, "2f047049969b07351c935e41da1b4c7000af19a354929a3df28d9c4f22b99b4b"),
    ("pdq/cpp/common/pdqhamming.cpp", 898932, "91864deda101e7e0b4405b6ce050cb8a945f644c3fe9815bcf08474cd89ef1af"),
    ("pdq/cpp/common/pdqhamming.h", 2068, "c3314937e3edc01a7d20531e9baf4007049d6ea06a93ec25b6457025aaa84c62"),
    ("pdq/cpp/common/pdqhashtypes.cpp", 4861, "8c5712de136d2fdcde15d165c2bca982015ffd5bc089393c010662f690ae48a3"),
    ("pdq/cpp/common/pdqhashtypes.h", 5199, "06ae688a78cdb17fc0aeb88fec2abdf1cb35d73a38ed49a629af03bf07ae8d72"),
    ("pdq/cpp/downscaling/downscaling.cpp", 15855, "4f7cc18baaebf27650166b4c1569ef8b3faf615abc861d18250573995ba91688"),
    ("pdq/cpp/downscaling/downscaling.h", 5399, "1238e00188a29035b3440d4405f64fdbe933c5a042cdfef572e8a5ef07e5923a"),
    ("pdq/cpp/hashing/pdqhashing.cpp", 15546, "2013379e258d8d0eb19c46049498aa72705db7a79014d3312b1f11b247fc0c51"),
    ("pdq/cpp/hashing/pdqhashing.h", 5078, "2c9abab5bb03119d3c17410570d13ab0a3e496118a28849196a8c079dac3e6c4"),
    ("pdq/cpp/hashing/torben.cpp", 1388, "64f1e95edb168a2e07216fae2f7da2f64b3246f6d8c752657eead3142353f09e"),
    ("pdq/cpp/hashing/torben.h", 626, "d58b25e10d30de4137e53ea869008f911687851a19b5d25c091e56ec5030acf3"),
)

REQUEST_MAGIC = b"CVIPDQ02"
RESPONSE_MAGIC = b"CVIPDQR2"
PROTOCOL_VERSION = 2
REQUEST_HEADER = struct.Struct("<8sIQIII32s")
RESPONSE_PREFIX = struct.Struct("<8sIIIQ32s")
RESPONSE_BYTES = RESPONSE_PREFIX.size + 8 * 32
MAXIMUM_DIMENSION = 16_384
MAXIMUM_PIXELS = 33_554_432
MAXIMUM_RGB_BYTES = MAXIMUM_PIXELS * 3
MAXIMUM_BATCH_REQUESTS = 1_024
MAXIMUM_BATCH_BYTES = 134_217_728
MAXIMUM_STDERR_BYTES = 8_192
MAXIMUM_BATCH_TIMEOUT_SECONDS = 300.0
PDQIO_RESIZE_DIMENSION = 512
OVERSIZE_PREPROCESSING = (
    "IF_EITHER_DIMENSION_GT_512_FORCE_512X512_NEAREST_FLOOR_COORDINATES"
)

_COMPILED_UPSTREAM_UNITS = (
    "pdq/cpp/downscaling/downscaling.cpp",
    "pdq/cpp/hashing/pdqhashing.cpp",
    "pdq/cpp/hashing/torben.cpp",
)
_PORTABLE_FLAGS = (
    "-std=c++17",
    "-O2",
    "-DNDEBUG",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-ident",
    "-Wl,--build-id=none",
)


@dataclass(frozen=True, slots=True)
class CanonicalRGBRequest:
    width: int
    height: int
    rgb: bytes
    request_sequence: int
    request_token: str

    def __post_init__(self) -> None:
        _validate_rgb_geometry(self.width, self.height, self.rgb)
        if (
            isinstance(self.request_sequence, bool)
            or not isinstance(self.request_sequence, int)
            or not 0 <= self.request_sequence <= (1 << 64) - 1
        ):
            raise ValueError("native PDQ request sequence must be an unsigned 64-bit integer")
        _validate_sha256(self.request_token)


@dataclass(frozen=True, slots=True)
class PDQNativeResult:
    request_sequence: int
    request_token: str
    d4_hashes: tuple[str, ...]
    quality: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.request_sequence, bool)
            or not isinstance(self.request_sequence, int)
            or not 0 <= self.request_sequence <= (1 << 64) - 1
        ):
            raise ValueError("native PDQ result sequence must be an unsigned 64-bit integer")
        _validate_sha256(self.request_token)
        if len(self.d4_hashes) != len(PDQ_D4_ORIENTATIONS):
            raise ValueError("native PDQ result must contain eight ordered D4 hashes")
        for value in self.d4_hashes:
            if len(value) != 64 or value.lower() != value:
                raise ValueError("native PDQ hash must be 64 lowercase hex digits")
            int(value, 16)
        if isinstance(self.quality, bool) or not isinstance(self.quality, int):
            raise TypeError("native PDQ quality must be an integer")
        if not 0 <= self.quality <= 100:
            raise ValueError("native PDQ quality must be between zero and 100")


@dataclass(frozen=True, slots=True)
class BuilderSourceProvenanceRow:
    relative_path: str
    content_sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or self.relative_path.startswith("/")
            or ".." in PurePosixPath(self.relative_path).parts
            or "\\" in self.relative_path
        ):
            raise ValueError("builder provenance source path differs")
        _validate_sha256(self.content_sha256)
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 0
        ):
            raise ValueError("builder provenance source byte size differs")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "content_sha256": self.content_sha256,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BuilderSourceProvenanceRow":
        if set(payload) != {"relative_path", "content_sha256", "byte_size"}:
            raise ValueError("builder provenance source row fields differ")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BuilderRuntimeProvenance:
    python_implementation: str
    python_version: str
    python_cache_tag: str
    platform_system: str
    platform_release: str
    os_name: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(f"builder runtime {name} differs")

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BuilderRuntimeProvenance":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("builder runtime fields differ")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class BuilderToolProvenance:
    code_source_manifest_sha256: str
    code_source_files: tuple[BuilderSourceProvenanceRow, ...]
    runtime: BuilderRuntimeProvenance
    runtime_sha256: str
    schema_version: str = "cvi.offline_tool_provenance.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.offline_tool_provenance.v1":
            raise ValueError("builder tool provenance schema differs")
        _validate_sha256(self.code_source_manifest_sha256)
        _validate_sha256(self.runtime_sha256)
        if not isinstance(self.code_source_files, tuple) or not self.code_source_files:
            raise ValueError("builder source provenance must not be empty")
        paths = tuple(row.relative_path for row in self.code_source_files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("builder source provenance rows must be sorted and unique")
        cli_rows = tuple(
            path for path in paths if path == "tools/build_native_pdq_worker.py"
        )
        package_rows = tuple(path for path in paths if path.startswith("src/cvi/"))
        if len(cli_rows) != 1 or len(package_rows) != len(paths) - 1:
            raise ValueError("builder source provenance path policy differs")
        if any(
            len(PurePosixPath(path).parts) != 3 or not path.endswith(".py")
            for path in package_rows
        ):
            raise ValueError("builder source provenance must contain top-level CVI Python files")
        if any(row.byte_size > 10_000_000 for row in self.code_source_files):
            raise ValueError("builder source provenance row exceeds the byte bound")
        required = {
            "src/cvi/pdq_native.py",
            "tools/build_native_pdq_worker.py",
        }
        if not required.issubset(paths):
            raise ValueError("builder provenance omits a required implementation source")
        source_payload = [row.to_dict() for row in self.code_source_files]
        if content_sha256(source_payload) != self.code_source_manifest_sha256:
            raise ValueError("builder source provenance manifest hash differs")
        if content_sha256(self.runtime.to_dict()) != self.runtime_sha256:
            raise ValueError("builder runtime provenance hash differs")

    @property
    def provenance_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_source_manifest_sha256": self.code_source_manifest_sha256,
            "code_source_files": [row.to_dict() for row in self.code_source_files],
            "runtime": self.runtime.to_dict(),
            "runtime_sha256": self.runtime_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BuilderToolProvenance":
        expected = {
            "schema_version",
            "code_source_manifest_sha256",
            "code_source_files",
            "runtime",
            "runtime_sha256",
        }
        if set(payload) != expected:
            raise ValueError("builder tool provenance fields differ")
        rows = payload["code_source_files"]
        runtime = payload["runtime"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise TypeError("builder tool provenance source rows differ")
        if not isinstance(runtime, dict):
            raise TypeError("builder tool provenance runtime differs")
        return cls(
            schema_version=payload["schema_version"],
            code_source_manifest_sha256=payload["code_source_manifest_sha256"],
            code_source_files=tuple(
                BuilderSourceProvenanceRow.from_dict(row) for row in rows
            ),
            runtime=BuilderRuntimeProvenance.from_dict(runtime),
            runtime_sha256=payload["runtime_sha256"],
        )


@dataclass(frozen=True, slots=True)
class PdqNativeBuildReceipt:
    intake_bundle_sha256: str
    source_contract_sha256: str
    source_receipt_sha256: str
    source_tool_provenance_sha256: str
    retained_source_aggregate_sha256: str
    builder_tool_provenance: BuilderToolProvenance
    builder_tool_provenance_sha256: str
    builder_code_source_manifest_sha256: str
    builder_runtime_sha256: str
    upstream_commit_sha: str
    retained_members: tuple[tuple[str, int, str], ...]
    compiled_upstream_units: tuple[str, ...]
    worker_source_sha256: str
    compiler_realpath: str
    compiler_sha256: str
    compiler_version_sha256: str
    compiler_version_first_line: str
    build_lane: str
    compiler_flags: tuple[str, ...]
    protocol_version: int
    maximum_dimension: int
    maximum_pixels: int
    maximum_rgb_bytes: int
    pdqio_resize_dimension: int
    oversize_preprocessing: str
    binary_filename: str
    binary_bytes: int
    binary_sha256: str
    license_content_sha256: str
    publication_guarantee: str = "ATOMIC_DIRECTORY_NOREPLACE"
    interpretation: str = CANONICAL_INTERPRETATION
    schema_version: str = "cvi.pdq_native_build_receipt.v4"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_native_build_receipt.v4":
            raise ValueError("unsupported native PDQ build receipt")
        for digest in (
            self.intake_bundle_sha256,
            self.source_contract_sha256,
            self.source_receipt_sha256,
            self.source_tool_provenance_sha256,
            self.retained_source_aggregate_sha256,
            self.builder_tool_provenance_sha256,
            self.builder_code_source_manifest_sha256,
            self.builder_runtime_sha256,
            self.worker_source_sha256,
            self.compiler_sha256,
            self.compiler_version_sha256,
            self.binary_sha256,
            self.license_content_sha256,
        ):
            _validate_sha256(digest)
        if not isinstance(self.builder_tool_provenance, BuilderToolProvenance):
            raise TypeError("builder tool provenance type differs")
        if (
            self.builder_tool_provenance_sha256
            != self.builder_tool_provenance.provenance_sha256
            or self.builder_code_source_manifest_sha256
            != self.builder_tool_provenance.code_source_manifest_sha256
            or self.builder_runtime_sha256
            != self.builder_tool_provenance.runtime_sha256
        ):
            raise ValueError("builder tool provenance receipt binding differs")
        if (
            self.intake_bundle_sha256 != CANONICAL_INTAKE_BUNDLE_SHA256
            or self.source_contract_sha256 != CANONICAL_SOURCE_CONTRACT_SHA256
            or self.source_receipt_sha256 != CANONICAL_SOURCE_RECEIPT_SHA256
            or self.source_tool_provenance_sha256
            != CANONICAL_TOOL_PROVENANCE_SHA256
            or self.retained_source_aggregate_sha256
            != CANONICAL_RETAINED_AGGREGATE_SHA256
        ):
            raise ValueError("native PDQ canonical source binding differs")
        if self.upstream_commit_sha != CANONICAL_COMMIT_SHA:
            raise ValueError("native build upstream commit differs")
        if not isinstance(self.retained_members, tuple) or len(self.retained_members) != 12:
            raise ValueError("native PDQ retained member inventory differs")
        retained_paths: list[str] = []
        for item in self.retained_members:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError("native PDQ retained member row differs")
            path, byte_size, digest = item
            if (
                not isinstance(path, str)
                or not path
                or path.startswith("/")
                or ".." in PurePosixPath(path).parts
                or "\\" in path
            ):
                raise ValueError("native PDQ retained member path differs")
            if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
                raise ValueError("native PDQ retained member byte size differs")
            _validate_sha256(digest)
            retained_paths.append(path)
        if retained_paths != sorted(retained_paths, key=str.casefold) or len(
            set(path.casefold() for path in retained_paths)
        ) != len(retained_paths):
            raise ValueError("native PDQ retained member ordering differs")
        if any(
            path == "pdq/cpp/CImg.h" or path.startswith("pdq/cpp/io/")
            for path in retained_paths
        ):
            raise ValueError("native PDQ retained inventory crosses the source boundary")
        if self.retained_members != CANONICAL_RETAINED_MEMBERS:
            raise ValueError("native PDQ retained member bytes differ from canonical v3")
        if self.compiled_upstream_units != _COMPILED_UPSTREAM_UNITS:
            raise ValueError("compiled upstream translation units differ")
        if self.build_lane != "PORTABLE_CPU_REFERENCE":
            raise ValueError("native PDQ build lane differs")
        if (
            self.worker_source_sha256 != CANONICAL_WORKER_SOURCE_SHA256
            or self.compiler_realpath != CANONICAL_COMPILER_REALPATH
            or self.compiler_sha256 != CANONICAL_COMPILER_SHA256
            or self.compiler_version_sha256 != CANONICAL_COMPILER_VERSION_SHA256
            or self.compiler_version_first_line
            != CANONICAL_COMPILER_VERSION_FIRST_LINE
        ):
            raise ValueError("native PDQ compiler identity differs")
        if self.compiler_flags != _PORTABLE_FLAGS:
            raise ValueError("portable native PDQ flags differ")
        if (self.protocol_version, self.maximum_dimension, self.maximum_pixels,
            self.maximum_rgb_bytes, self.pdqio_resize_dimension) != (
                PROTOCOL_VERSION, MAXIMUM_DIMENSION, MAXIMUM_PIXELS,
                MAXIMUM_RGB_BYTES, PDQIO_RESIZE_DIMENSION):
            raise ValueError("native PDQ protocol bounds differ")
        if self.oversize_preprocessing != OVERSIZE_PREPROCESSING:
            raise ValueError("native PDQ oversize preprocessing differs")
        if (
            self.binary_filename != "pdq-native-worker"
            or isinstance(self.binary_bytes, bool)
            or not isinstance(self.binary_bytes, int)
            or not 0 < self.binary_bytes <= 20_000_000
            or self.binary_bytes != CANONICAL_BINARY_BYTES
            or self.binary_sha256 != CANONICAL_BINARY_SHA256
        ):
            raise ValueError("native PDQ binary identity differs")
        if self.license_content_sha256 != (
            "68ecc6aafbd2a205a1077f86127030898f03091b7dae9d9017325a8702d8668f"
        ):
            raise ValueError("native PDQ license binding differs")
        if self.publication_guarantee != "ATOMIC_DIRECTORY_NOREPLACE":
            raise ValueError("native PDQ publication guarantee differs")
        if self.interpretation != CANONICAL_INTERPRETATION:
            raise ValueError("native PDQ evidence interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            field: (
                value.to_dict()
                if field == "builder_tool_provenance"
                else [list(item) for item in value]
                if field == "retained_members"
                else list(value) if isinstance(value, tuple) else value
            )
            for field, value in (
                (name, getattr(self, name)) for name in self.__dataclass_fields__
            )
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PdqNativeBuildReceipt":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("native PDQ build receipt keys mismatch")
        values = dict(payload)
        values["retained_members"] = tuple(
            tuple(item) for item in values["retained_members"]
        )
        if not isinstance(values["builder_tool_provenance"], dict):
            raise TypeError("builder tool provenance receipt field differs")
        values["builder_tool_provenance"] = BuilderToolProvenance.from_dict(
            values["builder_tool_provenance"]
        )
        values["compiled_upstream_units"] = tuple(values["compiled_upstream_units"])
        values["compiler_flags"] = tuple(values["compiler_flags"])
        return cls(**values)


def build_native_pdq_worker(
    *, source_bundle_directory: Path, worker_source: Path,
    output_directory: Path, compiler: Path,
    builder_tool_provenance: dict[str, Any],
) -> tuple[PdqNativeBuildReceipt, str]:
    """Build and atomically publish the fixed portable worker without overwrite."""

    if not isinstance(builder_tool_provenance, dict) or not builder_tool_provenance:
        raise ValueError("builder tool provenance must be a nonempty object")
    builder_provenance = BuilderToolProvenance.from_dict(builder_tool_provenance)
    if builder_provenance.to_dict() != builder_tool_provenance:
        raise ValueError("builder tool provenance normalization differs")
    if _current_builder_tool_provenance() != builder_provenance.to_dict():
        raise ValueError("builder tool provenance differs before build")
    bundle_path = source_bundle_directory / "intake-bundle.json"
    bundle_bytes = _read_regular_file(bundle_path, 4_000_000)
    if hashlib.sha256(bundle_bytes).hexdigest() != CANONICAL_INTAKE_BUNDLE_SHA256:
        raise ValueError("PDQ intake bundle differs from canonical v3")
    bundle = json.loads(bundle_bytes)
    if not isinstance(bundle, dict):
        raise ValueError("PDQ intake bundle root must be an object")
    source_contract = PdqSourceContract.from_dict(bundle["source_contract"])
    source_receipt = PdqSourceIntakeReceipt.from_dict(bundle["receipt"])
    if (
        bundle["source_contract_sha256"] != source_contract.contract_sha256
        or bundle["receipt_sha256"] != source_receipt.receipt_sha256
        or bundle["receipt_sha256"] != CANONICAL_SOURCE_RECEIPT_SHA256
        or bundle["tool_provenance_sha256"] != CANONICAL_TOOL_PROVENANCE_SHA256
        or source_receipt.retained_source_aggregate_sha256
        != CANONICAL_RETAINED_AGGREGATE_SHA256
    ):
        raise ValueError("PDQ canonical source bindings differ")
    source_root = source_bundle_directory / "source"
    retained_payloads: list[tuple[str, bytes, str]] = []
    for member in source_receipt.retained_members:
        payload = _read_regular_file(
            source_root / PurePosixPath(member.relative_path), member.byte_size
        )
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != member.byte_size or digest != member.content_sha256:
            raise ValueError("retained PDQ source bytes differ")
        retained_payloads.append((member.relative_path, payload, digest))
    observed_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if any(
        path.is_symlink() or (not path.is_dir() and not path.is_file())
        for path in source_root.rglob("*")
    ):
        raise ValueError("retained PDQ source tree must not contain symlinks")
    if observed_paths != {item[0] for item in retained_payloads}:
        raise ValueError("retained PDQ source tree membership differs")
    worker_payload = _read_regular_file(worker_source, 2_000_000)
    compiler_realpath = compiler.resolve(strict=True)
    compiler_payload = _read_regular_file(compiler_realpath, 100_000_000)
    version = subprocess.run(
        (str(compiler_realpath), "--version"), stdin=subprocess.DEVNULL,
        capture_output=True, check=True, timeout=10.0, shell=False,
    )
    if len(version.stdout) > 64_000 or version.stderr:
        raise RuntimeError("compiler version output is not bounded and clean")
    version_text = version.stdout.decode("utf-8", errors="strict")
    if not version_text.strip():
        raise RuntimeError("compiler version output is empty")

    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    parent = output_directory.parent.resolve(strict=True)
    stage = Path(mkdtemp(prefix=".cvi-pdq-build-", dir=parent))
    try:
        staged_source = stage / "source"
        for relative_path, payload, _ in retained_payloads:
            destination = staged_source / PurePosixPath(relative_path)
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            destination.write_bytes(payload)
        staged_worker = stage / "worker-main.cpp"
        staged_worker.write_bytes(worker_payload)
        binary = stage / "pdq-native-worker"
        command = (
            str(compiler_realpath), *_PORTABLE_FLAGS, "-I", str(staged_source),
            str(staged_worker),
            *(str(staged_source / path) for path in _COMPILED_UPSTREAM_UNITS),
            "-o", str(binary),
        )
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=120.0, shell=False,
        )
        if len(completed.stdout) > 1_000_000 or len(completed.stderr) > 1_000_000:
            raise RuntimeError("compiler output exceeds the fixed limit")
        if completed.returncode != 0:
            raise RuntimeError(
                "native PDQ build failed: "
                + completed.stderr.decode("utf-8", errors="replace")
            )
        if completed.stdout or completed.stderr:
            raise RuntimeError("portable native PDQ build emitted diagnostics")
        binary_payload = _read_regular_file(binary, 20_000_000)
        os.chmod(binary, 0o500)
        license_payload = next(
            payload for path, payload, _ in retained_payloads if path == "LICENSE"
        )
        license_path = stage / "LICENSE"
        if license_path.exists():
            license_path.unlink()
        license_path.write_bytes(license_payload)
        receipt = PdqNativeBuildReceipt(
            intake_bundle_sha256=CANONICAL_INTAKE_BUNDLE_SHA256,
            source_contract_sha256=source_contract.contract_sha256,
            source_receipt_sha256=source_receipt.receipt_sha256,
            source_tool_provenance_sha256=bundle["tool_provenance_sha256"],
            retained_source_aggregate_sha256=(
                source_receipt.retained_source_aggregate_sha256
            ),
            builder_tool_provenance=builder_provenance,
            builder_tool_provenance_sha256=builder_provenance.provenance_sha256,
            builder_code_source_manifest_sha256=(
                builder_provenance.code_source_manifest_sha256
            ),
            builder_runtime_sha256=builder_provenance.runtime_sha256,
            upstream_commit_sha=source_receipt.commit_sha,
            retained_members=tuple(
                (path, len(payload), digest)
                for path, payload, digest in retained_payloads
            ),
            compiled_upstream_units=_COMPILED_UPSTREAM_UNITS,
            worker_source_sha256=hashlib.sha256(worker_payload).hexdigest(),
            compiler_realpath=str(compiler_realpath),
            compiler_sha256=hashlib.sha256(compiler_payload).hexdigest(),
            compiler_version_sha256=hashlib.sha256(version.stdout).hexdigest(),
            compiler_version_first_line=version_text.splitlines()[0],
            build_lane="PORTABLE_CPU_REFERENCE",
            compiler_flags=_PORTABLE_FLAGS,
            protocol_version=PROTOCOL_VERSION,
            maximum_dimension=MAXIMUM_DIMENSION,
            maximum_pixels=MAXIMUM_PIXELS,
            maximum_rgb_bytes=MAXIMUM_RGB_BYTES,
            pdqio_resize_dimension=PDQIO_RESIZE_DIMENSION,
            oversize_preprocessing=OVERSIZE_PREPROCESSING,
            binary_filename="pdq-native-worker",
            binary_bytes=len(binary_payload),
            binary_sha256=hashlib.sha256(binary_payload).hexdigest(),
            license_content_sha256=hashlib.sha256(license_payload).hexdigest(),
        )
        receipt_payload = (
            json.dumps(receipt.to_dict(), sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        (stage / "build-receipt.json").write_bytes(receipt_payload)
        shutil.rmtree(staged_source)
        staged_worker.unlink()
        if _current_builder_tool_provenance() != builder_provenance.to_dict():
            raise RuntimeError("builder tool provenance changed during build")
        _fsync_tree(stage)
        strategy = rename_directory_noreplace(stage, output_directory)
        fsync_directory(parent)
        verify_native_pdq_build(output_directory, receipt)
        return receipt, strategy
    except BaseException:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def verify_native_pdq_build(root: Path, receipt: PdqNativeBuildReceipt) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native PDQ build root must be a real directory")
    if {path.name for path in root.iterdir()} != {
        "LICENSE", "build-receipt.json", "pdq-native-worker"
    }:
        raise ValueError("native PDQ build file membership differs")
    binary = _read_regular_file(root / receipt.binary_filename, 20_000_000)
    if len(binary) != receipt.binary_bytes or hashlib.sha256(binary).hexdigest() != receipt.binary_sha256:
        raise ValueError("native PDQ binary bytes differ from receipt")
    license_payload = _read_regular_file(root / "LICENSE", 100_000)
    if hashlib.sha256(license_payload).hexdigest() != receipt.license_content_sha256:
        raise ValueError("native PDQ license bytes differ from receipt")
    receipt_payload = json.loads(_read_regular_file(root / "build-receipt.json", 1_000_000))
    if PdqNativeBuildReceipt.from_dict(receipt_payload) != receipt:
        raise ValueError("native PDQ receipt file differs")


def hash_rgb_batch(
    requests: Iterable[CanonicalRGBRequest], *, binary_path: Path,
    expected_binary_sha256: str, timeout_seconds: float = 30.0,
) -> tuple[PDQNativeResult, ...]:
    """Hash one bounded batch in one fresh, shell-free worker process."""

    rows = tuple(requests)
    if not rows or len(rows) > MAXIMUM_BATCH_REQUESTS:
        raise ValueError("native PDQ batch size is outside the fixed bounds")
    if any(not isinstance(row, CanonicalRGBRequest) for row in rows):
        raise TypeError("native PDQ batch requires CanonicalRGBRequest rows")
    sequences = tuple(row.request_sequence for row in rows)
    if len(set(sequences)) != len(sequences):
        raise ValueError("native PDQ request sequences must be unique within a batch")
    tokens = tuple(row.request_token for row in rows)
    if len(set(tokens)) != len(tokens):
        raise ValueError("native PDQ request tokens must be unique within a batch")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > MAXIMUM_BATCH_TIMEOUT_SECONDS
    ):
        raise ValueError("native PDQ timeout is outside the fixed bound")
    total = sum(REQUEST_HEADER.size + len(row.rgb) for row in rows)
    if total > MAXIMUM_BATCH_BYTES:
        raise ValueError("native PDQ batch bytes exceed the fixed bound")
    input_payload = bytearray(total)
    write_offset = 0
    for row in rows:
        REQUEST_HEADER.pack_into(
            input_payload, write_offset, REQUEST_MAGIC, PROTOCOL_VERSION,
            row.request_sequence, row.width, row.height, len(row.rgb),
            bytes.fromhex(row.request_token),
        )
        write_offset += REQUEST_HEADER.size
        input_payload[write_offset:write_offset + len(row.rgb)] = row.rgb
        write_offset += len(row.rgb)
    expected_stdout = len(rows) * RESPONSE_BYTES
    descriptor = os.open(binary_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & 0o111:
            raise ValueError("native PDQ worker must be an executable regular file")
        observed_sha256 = _sha256_descriptor(descriptor)
        _validate_sha256(expected_binary_sha256)
        if observed_sha256 != expected_binary_sha256:
            raise ValueError("native PDQ worker binary differs from expected hash")
        command = (f"/proc/self/fd/{descriptor}",)
        result = _run_bounded_worker(
            command, input_payload, timeout_seconds=timeout_seconds,
            maximum_stdout_bytes=expected_stdout,
            maximum_stderr_bytes=MAXIMUM_STDERR_BYTES,
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    if result[0] != 0 or result[2]:
        detail = result[2].decode("utf-8", errors="replace")
        raise RuntimeError(f"native PDQ worker failed closed: {detail}")
    stdout = result[1]
    if len(stdout) != expected_stdout:
        raise RuntimeError("native PDQ worker response length differs")
    parsed: list[PDQNativeResult] = []
    for row_index, offset in enumerate(range(0, len(stdout), RESPONSE_BYTES)):
        response = stdout[offset:offset + RESPONSE_BYTES]
        magic, version, status_code, quality, sequence, token = RESPONSE_PREFIX.unpack_from(response)
        if magic != RESPONSE_MAGIC or version != PROTOCOL_VERSION or status_code != 0:
            raise RuntimeError("native PDQ worker response header differs")
        if sequence != rows[row_index].request_sequence:
            raise RuntimeError("native PDQ worker response sequence differs")
        if token.hex() != rows[row_index].request_token:
            raise RuntimeError("native PDQ worker response token differs")
        hash_payload = response[RESPONSE_PREFIX.size:]
        hashes = tuple(
            hash_payload[index:index + 32].hex()
            for index in range(0, len(hash_payload), 32)
        )
        parsed.append(PDQNativeResult(sequence, token.hex(), hashes, quality))
    return tuple(parsed)


def hash_rgb(
    request: CanonicalRGBRequest, *, binary_path: Path,
    expected_binary_sha256: str, timeout_seconds: float = 30.0,
) -> PDQNativeResult:
    return hash_rgb_batch(
        (request,), binary_path=binary_path,
        expected_binary_sha256=expected_binary_sha256,
        timeout_seconds=timeout_seconds,
    )[0]


def _run_bounded_worker(
    command: tuple[str, ...], input_payload: bytes | bytearray, *, timeout_seconds: float,
    maximum_stdout_bytes: int, maximum_stderr_bytes: int,
    pass_fds: tuple[int, ...],
) -> tuple[int, bytes, bytes]:
    if timeout_seconds <= 0:
        raise ValueError("native PDQ timeout must be positive")
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, shell=False, start_new_session=True,
        pass_fds=pass_fds, env={"LC_ALL": "C", "LANG": "C"},
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    limit_exceeded = threading.Event()
    errors: list[BaseException] = []
    stdout = bytearray()
    stderr = bytearray()

    def writer() -> None:
        try:
            view = memoryview(input_payload)
            offset = 0
            while offset < len(view):
                count = process.stdin.write(view[offset:offset + 1_048_576])
                if count is None or count <= 0:
                    raise RuntimeError("native PDQ worker stdin made no progress")
                offset += count
            process.stdin.close()
        except BrokenPipeError:
            pass
        except BaseException as error:
            errors.append(error)

    def reader(stream: BinaryIO, target: bytearray, maximum: int) -> None:
        try:
            while True:
                chunk = stream.read(65_536)
                if not chunk:
                    return
                remaining = maximum - len(target)
                target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    limit_exceeded.set()
                    return
        except BaseException as error:
            errors.append(error)

    threads = (
        threading.Thread(target=writer, daemon=True),
        threading.Thread(target=reader, args=(process.stdout, stdout, maximum_stdout_bytes), daemon=True),
        threading.Thread(target=reader, args=(process.stderr, stderr, maximum_stderr_bytes), daemon=True),
    )
    for thread in threads:
        thread.start()
    deadline = monotonic() + timeout_seconds
    timed_out = False
    while process.poll() is None:
        if limit_exceeded.is_set() or monotonic() >= deadline:
            timed_out = monotonic() >= deadline
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            break
        sleep(0.005)
    return_code = process.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    for stream in (process.stdin, process.stdout, process.stderr):
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("native PDQ worker pipe did not reach EOF")
    if errors:
        raise RuntimeError("native PDQ worker pipe handling failed") from errors[0]
    if timed_out:
        raise TimeoutError("native PDQ worker timed out")
    if limit_exceeded.is_set():
        raise RuntimeError("native PDQ worker output exceeded the fixed bound")
    return return_code, bytes(stdout), bytes(stderr)


def _validate_rgb_geometry(width: int, height: int, rgb: bytes) -> None:
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"RGB {name} must be an integer")
        if value <= 0 or value > MAXIMUM_DIMENSION:
            raise ValueError(f"RGB {name} is outside the fixed bound")
    if not isinstance(rgb, bytes):
        raise TypeError("canonical RGB payload must be immutable bytes")
    pixels = width * height
    if pixels > MAXIMUM_PIXELS:
        raise ValueError("RGB pixel count exceeds the fixed bound")
    if len(rgb) != pixels * 3 or len(rgb) > MAXIMUM_RGB_BYTES:
        raise ValueError("RGB payload length differs from image geometry")


def _current_builder_tool_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    return build_offline_tool_provenance(
        repository_root / "tools/build_native_pdq_worker.py"
    )


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_bytes:
            raise ValueError(f"file is not a bounded regular file: {path}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes:
            raise ValueError(f"file exceeds the fixed byte bound: {path}")
        final_metadata = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )
        if identity(metadata) != identity(final_metadata) or len(payload) != metadata.st_size:
            raise RuntimeError(f"file changed while it was read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1_048_576)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected a lowercase SHA-256 digest")
    int(value, 16)
    if value.lower() != value:
        raise ValueError("expected a lowercase SHA-256 digest")


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            fsync_directory(path)
    fsync_directory(root)
