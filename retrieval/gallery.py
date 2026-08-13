from __future__ import annotations

import fcntl
import hashlib
import heapq
import json
import os
import shutil
import stat
import zipfile
from array import array
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import faiss
import numpy as np

from foundation.protected_publication import rename_directory_noreplace
from retrieval.qkv import (
    FULL128_CHANNEL,
    SCORER_ALGORITHM,
    AvailableIntersectionScorer,
    EnrollmentRank,
    EvidenceChannelSpec,
    GalleryKey,
    GalleryValue,
    IdentityEvidenceKind,
    QueryExclusions,
    RetrievalQuery,
    ScoredGalleryValue,
)

_MANIFEST_SCHEMA_V4 = "cvi.gallery_manifest.v4"
_MANIFEST_SCHEMA_V5 = "cvi.gallery_manifest.v5"
_TEMPLATE_SCHEMA_V1 = "cvi.gallery_template.v1"
_TEMPLATE_SCHEMA_V2 = "cvi.gallery_template.v2"
_TEMPLATE_ID_DOMAIN = _TEMPLATE_SCHEMA_V1
_IDENTITY_AGGREGATION = "max"
_IDENTITY_POLICY_SCHEMA = "cvi.gallery_identity_policy.v1"
_REGISTERED_ONLY = "REGISTERED_ONLY"
_SEARCH_BLOCK_ROWS = 65_536
_MAXIMUM_MANIFEST_BYTES = 2 * 1024 * 1024
_MAXIMUM_SIDECAR_JSON_BYTES = 64 * 1024 * 1024
_MAXIMUM_METADATA_BYTES = 64 * 1024
_MAXIMUM_IDEMPOTENCY_KEY_BYTES = _MAXIMUM_METADATA_BYTES
_MAXIMUM_PROVENANCE_TEXT_BYTES = 4 * 1024
_MAXIMUM_GALLERY_TEMPLATES = 1_000_000
_MAXIMUM_BINARY_FILE_BYTES = 64 * 1024 * 1024 * 1024
_BINARY_FORMAT_OVERHEAD_BYTES = 1024 * 1024
_OPTIONAL_CHANNEL_OVERHEAD_BYTES = 64 * 1024
_RESERVED_METADATA_KEYS = {
    "template_id", "content_sha256", "idempotency_key", "template_schema"
}


@dataclass(frozen=True, slots=True)
class IdentityRegistryPolicy:
    """Fail-closed admission policy for registered-only gallery identities."""

    registered_identity_ids: frozenset[str] | None = None
    provisional_generated_identity_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        registered = self.registered_identity_ids
        if registered is not None and not isinstance(registered, frozenset):
            registered = frozenset(registered)
            object.__setattr__(self, "registered_identity_ids", registered)
        provisional = self.provisional_generated_identity_ids
        if not isinstance(provisional, frozenset):
            provisional = frozenset(provisional)
            object.__setattr__(self, "provisional_generated_identity_ids", provisional)
        for values in (registered, provisional):
            if values is not None and any(not _is_canonical_uuid5(value) for value in values):
                raise ValueError("identity registry policy IDs must be canonical UUIDv5")
        if registered is not None and registered & provisional:
            raise ValueError("identity registry policy namespaces overlap")

    @property
    def descriptor(self) -> dict[str, str | None]:
        digest: str | None = None
        if self.registered_identity_ids is not None or self.provisional_generated_identity_ids:
            payload = {
                "registered": sorted(self.registered_identity_ids or ()),
                "provisional_genid": sorted(self.provisional_generated_identity_ids),
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        return {
            "schema_version": _IDENTITY_POLICY_SCHEMA,
            "mode": _REGISTERED_ONLY,
            "registry_sha256": digest,
        }

    def validate(self, identity_id: str, kind: IdentityEvidenceKind) -> None:
        if kind is not IdentityEvidenceKind.REGISTERED:
            raise ValueError("registered-only gallery rejects provisional GenID evidence")
        if identity_id in self.provisional_generated_identity_ids:
            raise ValueError("registered-only gallery rejects a registry-known provisional GenID")
        if (
            self.registered_identity_ids is not None
            and identity_id not in self.registered_identity_ids
        ):
            raise ValueError("registered identity is absent from the configured registry")


@dataclass(frozen=True, slots=True)
class GalleryEnrollment:
    embedding: np.ndarray | dict[str, np.ndarray]
    registered_identity_id: str
    breed: str = "unknown"
    metadata: dict[str, Any] | None = None
    idempotency_key: str | None = None
    content_sha256: str | None = None
    availability: dict[str, bool] | None = None
    identity_evidence_kind: IdentityEvidenceKind = IdentityEvidenceKind.REGISTERED
    enrollment_rank: EnrollmentRank | None = None
    enrollment_view: str | None = None
    duplicate_group_ids: tuple[str, ...] = ()


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    try:
        rename_directory_noreplace(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            "bulk gallery build destination already exists"
        ) from exc


def _open_regular_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int | None = None,
):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - the supported release target is Linux
        raise RuntimeError("secure gallery reads require O_NOFOLLOW support")
    try:
        descriptor = os.open(path, flags | no_follow)
    except OSError as exc:
        raise RuntimeError(
            f"unable to read {label} {str(path)!r} without following links"
        ) from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"{label} must be a regular file")
        if maximum_bytes is not None and file_stat.st_size > maximum_bytes:
            raise RuntimeError(f"{label} exceeds its byte limit")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _sha256_file(
    path: Path,
    label: str = "gallery file",
    *,
    maximum_bytes: int | None = None,
) -> str:
    with _open_regular_file(path, label, maximum_bytes=maximum_bytes) as stream:
        return _sha256_stream(stream)


def _checked_size_product(label: str, *values: int) -> int:
    result = 1
    for value in values:
        if value < 0 or (result and value > _MAXIMUM_BINARY_FILE_BYTES // result):
            raise RuntimeError(f"{label} exceeds the supported binary byte limit")
        result *= value
    return result


def _checked_size_sum(label: str, *values: int) -> int:
    result = 0
    for value in values:
        if value < 0 or value > _MAXIMUM_BINARY_FILE_BYTES - result:
            raise RuntimeError(f"{label} exceeds the supported binary byte limit")
        result += value
    return result


def _binary_file_limits(
    template_count: int,
    required_dimension: int,
    optional_channels: tuple[EvidenceChannelSpec, ...],
) -> dict[str, int]:
    required_payload = _checked_size_product(
        "gallery required index",
        template_count,
        required_dimension,
        np.dtype(np.float32).itemsize,
    )
    required_limit = _checked_size_sum(
        "gallery required index",
        required_payload,
        _BINARY_FORMAT_OVERHEAD_BYTES,
    )
    optional_limit = _BINARY_FORMAT_OVERHEAD_BYTES
    for channel in optional_channels:
        channel_payload = _checked_size_sum(
            "gallery optional vectors",
            _checked_size_product(
                "gallery optional vectors",
                template_count,
                np.dtype(np.int64).itemsize,
            ),
            _checked_size_product(
                "gallery optional vectors",
                template_count,
                channel.dimension,
                np.dtype(np.float32).itemsize,
            ),
            _OPTIONAL_CHANNEL_OVERHEAD_BYTES,
        )
        optional_limit = _checked_size_sum(
            "gallery optional vectors", optional_limit, channel_payload
        )
    return {
        "required_index": required_limit,
        "optional_vectors": optional_limit,
    }


def _optional_member_limit(
    template_count: int,
    channel: EvidenceChannelSpec,
    member_kind: str,
) -> int:
    if member_kind == "rows":
        payload = _checked_size_product(
            "gallery optional row member",
            template_count,
            np.dtype(np.int64).itemsize,
        )
    else:
        payload = _checked_size_product(
            "gallery optional vector member",
            template_count,
            channel.dimension,
            np.dtype(np.float32).itemsize,
        )
    return _checked_size_sum(
        f"gallery optional {member_kind} member",
        payload,
        _OPTIONAL_CHANNEL_OVERHEAD_BYTES,
    )


def _deflate_compressed_limit(uncompressed_limit: int) -> int:
    block_overhead = _checked_size_product(
        "gallery optional compressed member",
        (uncompressed_limit + 16_383) // 16_384,
        5,
    )
    return _checked_size_sum(
        "gallery optional compressed member",
        uncompressed_limit,
        block_overhead,
        64,
    )


def _preflight_optional_vectors_npz(
    stream,
    template_count: int,
    optional_channels: tuple[EvidenceChannelSpec, ...],
) -> None:
    expected: dict[str, tuple[EvidenceChannelSpec, str]] = {}
    member_limits: dict[str, int] = {}
    for channel_index, channel in enumerate(optional_channels):
        for member_kind in ("rows", "vectors"):
            name = f"c{channel_index}_{member_kind}.npy"
            expected[name] = (channel, member_kind)
            member_limits[name] = _optional_member_limit(
                template_count, channel, member_kind
            )
    total_uncompressed_limit = _checked_size_sum(
        "gallery optional vectors", *member_limits.values()
    )

    try:
        with zipfile.ZipFile(stream, mode="r") as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if (
                len(members) != len(expected)
                or len(member_names) != len(set(member_names))
                or set(member_names) != set(expected)
            ):
                raise RuntimeError(
                    "gallery sparse optional vector archive schema is invalid"
                )
            total_uncompressed = 0
            header_shapes: dict[str, tuple[int, ...]] = {}
            for member in members:
                name = member.filename
                channel, member_kind = expected[name]
                limit = member_limits[name]
                compressed_limit = _deflate_compressed_limit(limit)
                if (
                    member.orig_filename != name
                    or member.is_dir()
                    or member.flag_bits & 0x1
                    or member.compress_type != zipfile.ZIP_DEFLATED
                    or not 0 < member.compress_size <= compressed_limit
                    or not 0 < member.file_size <= limit
                ):
                    raise RuntimeError(
                        f"gallery optional vector member {name!r} is invalid"
                    )
                total_uncompressed = _checked_size_sum(
                    "gallery optional vectors", total_uncompressed, member.file_size
                )
                if total_uncompressed > total_uncompressed_limit:
                    raise RuntimeError(
                        "gallery optional vector members exceed their byte limit"
                    )

                with archive.open(member, mode="r") as member_stream:
                    version = np.lib.format.read_magic(member_stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(
                            member_stream
                        )
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(
                            member_stream
                        )
                    else:
                        raise RuntimeError(
                            f"gallery optional vector member {name!r} has an "
                            "unsupported NPY version"
                        )
                    if member_kind == "rows":
                        valid_shape = len(shape) == 1 and 0 <= shape[0] <= template_count
                        valid_dtype = dtype == np.dtype(np.int64)
                        payload_bytes = (
                            shape[0] * np.dtype(np.int64).itemsize
                            if valid_shape
                            else 0
                        )
                    else:
                        valid_shape = (
                            len(shape) == 2
                            and 0 <= shape[0] <= template_count
                            and shape[1] == channel.dimension
                        )
                        valid_dtype = dtype == np.dtype(np.float32)
                        payload_bytes = (
                            shape[0]
                            * channel.dimension
                            * np.dtype(np.float32).itemsize
                            if valid_shape
                            else 0
                        )
                    if not valid_shape or not valid_dtype:
                        raise RuntimeError(
                            f"gallery optional vector member {name!r} has an invalid "
                            "dtype or shape"
                        )
                    if member_stream.tell() + payload_bytes != member.file_size:
                        raise RuntimeError(
                            f"gallery optional vector member {name!r} has an invalid "
                            "payload size"
                        )
                    header_shapes[name] = shape

            for channel_index in range(len(optional_channels)):
                rows = header_shapes[f"c{channel_index}_rows.npy"]
                vectors = header_shapes[f"c{channel_index}_vectors.npy"]
                if vectors[0] != rows[0]:
                    raise RuntimeError(
                        "gallery sparse optional vector row counts are inconsistent"
                    )
    except RuntimeError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise RuntimeError(
            "gallery sparse optional vector archive is invalid"
        ) from exc
    finally:
        stream.seek(0)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite JSON number is not accepted: {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise RuntimeError(f"non-finite JSON number is not accepted: {value}")
    return parsed


def _parse_strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    if not payload:
        raise RuntimeError(f"{label} must not be empty")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")  # noqa: TRY004
    return value


def _read_strict_json_object(
    path: Path,
    label: str,
    *,
    maximum_bytes: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    with _open_regular_file(path, label, maximum_bytes=maximum_bytes) as stream:
        payload = stream.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise RuntimeError(f"{label} exceeds its byte limit")
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise RuntimeError(f"{label} is missing or corrupted")
    return _parse_strict_json_object(payload, label)


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _write_json_file(
    path: Path,
    value: object,
    label: str,
    *,
    maximum_bytes: int,
) -> None:
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be finite UTF-8 JSON") from exc
    if len(payload) > maximum_bytes:
        raise RuntimeError(f"{label} exceeds its byte limit")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class IdentityGallery:
    """Generation-published K/V gallery with exact sparse-channel QK scoring."""

    def __init__(
        self,
        gallery_directory: Path,
        dim: int = 640,
        embedding_contract: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
        registry_policy: IdentityRegistryPolicy | None = None,
    ) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("gallery dimension must be a positive integer")
        self._dim = dim
        self._embedding_contract = _canonical_embedding_contract(
            embedding_contract, dim
        )
        self._channels = _contract_channels(self._embedding_contract, dim)
        self._scorer = AvailableIntersectionScorer(self._channels)
        self._required_channels = tuple(
            channel for channel in self._channels if not channel.optional
        )
        self._optional_channels = tuple(
            channel for channel in self._channels if channel.optional
        )
        if not self._required_channels:
            raise ValueError("gallery contract requires at least one required channel")
        self._required_dim = sum(channel.dimension for channel in self._required_channels)
        self._scorer_hash = self._scorer.scorer_hash
        self._full128_fast_path = (
            len(self._channels) == 1
            and self._channels[0].name == FULL128_CHANNEL
            and self._channels[0].dimension == 128
            and not self._channels[0].optional
        )
        if registry_policy is None:
            registry_policy = IdentityRegistryPolicy()
        if not isinstance(registry_policy, IdentityRegistryPolicy):
            raise TypeError("registry_policy must be an IdentityRegistryPolicy")
        self._registry_policy = registry_policy
        self._base_dir = gallery_directory
        if _path_entry_exists(self._base_dir) and self._base_dir.is_symlink():
            raise RuntimeError("gallery root must not be a symbolic link")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        if self._base_dir.is_symlink() or not self._base_dir.is_dir():
            raise RuntimeError("gallery root must be a non-symlink directory")
        self._metadata: dict[int, dict[str, Any]] = {}
        self._breed_index: dict[int, str] = {}
        self._optional_vectors: dict[str, dict[int, np.ndarray]] = {
            channel.name: {} for channel in self._optional_channels
        }
        self._availability: dict[int, dict[str, bool]] = {}
        self._identity_kinds: dict[int, IdentityEvidenceKind] = {}
        self._enrollment_ranks: dict[int, EnrollmentRank | None] = {}
        self._enrollment_views: dict[int, str | None] = {}
        self._duplicate_group_ids: dict[int, tuple[str, ...]] = {}
        self._template_id_index: dict[str, int] = {}
        self._content_sha256_index: dict[str, int] = {}
        self._idempotency_key_index: dict[str, int] = {}
        self._duplicate_group_index: dict[str, set[int]] = {}
        self._identity_rows: dict[str, list[int]] = {}
        self._identity_ordinals: dict[str, int] = {}
        self._identities_by_ordinal: list[str] = []
        self._row_identity_ordinals: array[int] = array("q")
        self._loaded_manifest_schema: str | None = None
        self._read_only = read_only
        self._lock_stream = None
        if not read_only:
            lock_path = self._base_dir / ".gallery-writer.lock"
            flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
            no_follow = getattr(os, "O_NOFOLLOW", None)
            if no_follow is None:  # pragma: no cover - supported target is Linux
                raise RuntimeError("gallery writer locking requires O_NOFOLLOW support")
            try:
                descriptor = os.open(lock_path, flags | no_follow, 0o600)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    os.close(descriptor)
                    raise RuntimeError("gallery writer lock must be a regular file")
                self._lock_stream = os.fdopen(descriptor, "a+b")
            except OSError as exc:
                raise RuntimeError(
                    "gallery writer lock cannot be opened without following links"
                ) from exc
            try:
                fcntl.flock(
                    self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                self._lock_stream.close()
                self._lock_stream = None
                raise RuntimeError(
                    f"gallery {str(self._base_dir)!r} already has an active writer"
                ) from exc

        try:
            manifest_path = self._base_dir / "gallery_manifest.json"
            unversioned_paths = (
                self._base_dir / "master.idx",
                self._base_dir / "metadata.json",
                self._base_dir / "breed_index.json",
            )
            if _path_entry_exists(manifest_path):
                self._load_snapshot(manifest_path)
            elif any(_path_entry_exists(path) for path in unversioned_paths):
                raise RuntimeError(
                    "unversioned gallery files are not accepted; rebuild the gallery"
                )
            else:
                if read_only:
                    raise RuntimeError("read-only gallery does not exist")
                self._index = faiss.IndexFlatIP(self._required_dim)
                self._rebuild_runtime_indexes()
        except Exception:
            self.close()
            raise

    @property
    def scorer_hash(self) -> str:
        return self._scorer_hash

    def _load_snapshot(self, manifest_path: Path) -> None:
        manifest = _read_strict_json_object(
            manifest_path,
            "gallery manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
        common = {
            "schema_version", "dimension", "required_dimension",
            "embedding_contract", "count", "template_count", "identity_count",
            "identity_aggregation", "scorer", "files",
        }
        schema = manifest.get("schema_version")
        expected = common if schema == _MANIFEST_SCHEMA_V4 else common | {"identity_policy"}
        if schema not in {_MANIFEST_SCHEMA_V4, _MANIFEST_SCHEMA_V5} or set(manifest) != expected:
            raise RuntimeError(
                "gallery is not an exact supported v4/v5 manifest; migrate v3 into "
                "a new output directory"
            )
        if (
            schema == _MANIFEST_SCHEMA_V5
            and manifest["identity_policy"] != self._registry_policy.descriptor
        ):
            raise RuntimeError("gallery identity registry policy differs from runtime")
        self._loaded_manifest_schema = schema
        for field in ("count", "template_count", "identity_count"):
            if not _is_cardinality(
                manifest[field], maximum=_MAXIMUM_GALLERY_TEMPLATES
            ):
                raise RuntimeError(f"gallery manifest {field} is invalid")
        if manifest["count"] != manifest["template_count"]:
            raise RuntimeError("gallery manifest count is inconsistent")
        if manifest["identity_count"] > manifest["template_count"]:
            raise RuntimeError("gallery manifest identity count is inconsistent")
        if manifest["identity_aggregation"] != _IDENTITY_AGGREGATION:
            raise RuntimeError("gallery identity aggregation contract is invalid")
        if manifest["dimension"] != self._dim:
            raise RuntimeError(
                f"gallery dimension {manifest['dimension']} does not match {self._dim}"
            )
        if manifest["required_dimension"] != self._required_dim:
            raise RuntimeError("gallery required dimension differs from runtime")
        if manifest["embedding_contract"] != self._embedding_contract:
            raise RuntimeError("gallery embedding contract differs from runtime")
        if manifest["scorer"] != {
            "algorithm": SCORER_ALGORITHM,
            "hash": self._scorer_hash,
            "exact": True,
        }:
            raise RuntimeError("gallery scorer contract differs from runtime")
        files = manifest["files"]
        expected_files = {
            "required_index", "optional_vectors", "availability", "metadata", "breeds"
        }
        if not isinstance(files, dict) or set(files) != expected_files:
            raise RuntimeError("invalid gallery file manifest")
        resolved: dict[str, Path] = {}
        for kind, entry in files.items():
            if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
                raise RuntimeError(f"invalid gallery {kind} entry")
            name = entry["name"]
            if not isinstance(name, str) or Path(name).name != name:
                raise RuntimeError(f"invalid gallery {kind} filename")
            if not _is_sha256(entry["sha256"]):
                raise RuntimeError(f"invalid gallery {kind} digest")
            path = self._base_dir / name
            resolved[kind] = path

        metadata = _read_strict_json_object(
            resolved["metadata"],
            "gallery metadata sidecar",
            maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            expected_sha256=files["metadata"]["sha256"],
        )
        breeds = _read_strict_json_object(
            resolved["breeds"],
            "gallery breeds sidecar",
            maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            expected_sha256=files["breeds"]["sha256"],
        )
        availability = _read_strict_json_object(
            resolved["availability"],
            "gallery availability sidecar",
            maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            expected_sha256=files["availability"]["sha256"],
        )
        template_count = manifest["template_count"]
        if any(
            len(sidecar) != template_count
            for sidecar in (metadata, breeds, availability)
        ):
            raise RuntimeError("gallery manifest template count is inconsistent")
        if any(
            str(index) not in sidecar
            for sidecar in (metadata, breeds, availability)
            for index in range(template_count)
        ):
            raise RuntimeError("gallery index and sidecar cardinality are inconsistent")
        binary_limits = _binary_file_limits(
            template_count, self._required_dim, self._optional_channels
        )
        with _open_regular_file(
            resolved["required_index"],
            "gallery required index",
            maximum_bytes=binary_limits["required_index"],
        ) as stream:
            if _sha256_stream(stream) != files["required_index"]["sha256"]:
                raise RuntimeError("gallery required_index file is missing or corrupted")
            descriptor_path = Path("/proc/self/fd") / str(stream.fileno())
            if not descriptor_path.exists():  # pragma: no cover - Linux release target
                raise RuntimeError("secure FAISS gallery reads require /proc/self/fd")
            self._index = faiss.read_index(str(descriptor_path))
        if int(self._index.ntotal) != template_count:
            raise RuntimeError("gallery index and manifest cardinality are inconsistent")
        self._metadata = {}
        self._identity_kinds = {}
        self._enrollment_ranks = {}
        self._enrollment_views = {}
        self._duplicate_group_ids = {}
        for key, value in metadata.items():
            row = int(key)
            if not isinstance(value, dict):
                raise RuntimeError(  # noqa: TRY004
                    "gallery metadata row has an invalid schema"
                )
            if schema == _MANIFEST_SCHEMA_V4:
                self._metadata[row] = value
                self._identity_kinds[row] = IdentityEvidenceKind.REGISTERED
                self._enrollment_ranks[row] = None
                self._enrollment_views[row] = None
                self._duplicate_group_ids[row] = ()
                continue
            v5_fields = {
                "registered_dog_id", "template_id", "content_sha256",
                "idempotency_key", "template_schema", "metadata",
                "identity_evidence_kind", "enrollment_rank", "enrollment_view",
                "duplicate_group_ids",
            }
            if set(value) != v5_fields:
                raise RuntimeError("gallery v5 metadata row has an invalid schema")
            try:
                kind = IdentityEvidenceKind(value["identity_evidence_kind"])
                rank = (
                    None
                    if value["enrollment_rank"] is None
                    else EnrollmentRank(value["enrollment_rank"])
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("gallery v5 enrollment provenance is invalid") from exc
            duplicate_group_ids = value["duplicate_group_ids"]
            if not isinstance(duplicate_group_ids, list):
                raise RuntimeError(  # noqa: TRY004
                    "gallery v5 duplicate-group provenance must be an array"
                )
            self._metadata[row] = {
                name: value[name]
                for name in (
                    "registered_dog_id", "template_id", "content_sha256",
                    "idempotency_key", "template_schema", "metadata",
                )
            }
            self._identity_kinds[row] = kind
            self._enrollment_ranks[row] = rank
            self._enrollment_views[row] = value["enrollment_view"]
            self._duplicate_group_ids[row] = tuple(duplicate_group_ids)
        self._breed_index = {int(key): value for key, value in breeds.items()}
        self._availability = {
            int(key): value for key, value in availability.items()
        }
        self._optional_vectors = {
            channel.name: {} for channel in self._optional_channels
        }
        with _open_regular_file(
            resolved["optional_vectors"],
            "gallery optional vectors",
            maximum_bytes=binary_limits["optional_vectors"],
        ) as stream:
            if _sha256_stream(stream) != files["optional_vectors"]["sha256"]:
                raise RuntimeError(
                    "gallery optional_vectors file is missing or corrupted"
                )
            _preflight_optional_vectors_npz(
                stream, template_count, self._optional_channels
            )
            sparse_context = np.load(stream, allow_pickle=False)
            with sparse_context as sparse:
                expected_arrays = {
                    key
                    for index in range(len(self._optional_channels))
                    for key in (f"c{index}_rows", f"c{index}_vectors")
                }
                if set(sparse.files) != expected_arrays:
                    raise RuntimeError("gallery sparse optional vector schema is invalid")
                for channel_index, channel in enumerate(self._optional_channels):
                    rows = sparse[f"c{channel_index}_rows"]
                    vectors = sparse[f"c{channel_index}_vectors"]
                    if (
                        rows.dtype != np.int64
                        or rows.ndim != 1
                        or len(rows) > template_count
                        or vectors.dtype != np.float32
                        or vectors.shape != (len(rows), channel.dimension)
                        or len(set(rows.tolist())) != len(rows)
                    ):
                        raise RuntimeError(
                            f"gallery optional channel {channel.name!r} is invalid"
                        )
                    self._optional_vectors[channel.name] = {
                        int(row): vectors[offset].copy()
                        for offset, row in enumerate(rows.tolist())
                    }
        self._validate_state(
            expected_template_count=manifest["template_count"],
            expected_identity_count=manifest["identity_count"],
        )
        self._rebuild_runtime_indexes()

    def _validate_state(
        self,
        *,
        expected_template_count: int | None = None,
        expected_identity_count: int | None = None,
    ) -> None:
        count = int(self._index.ntotal)
        if self._index.d != self._required_dim:
            raise RuntimeError("gallery required-vector dimension is inconsistent")
        if expected_template_count is not None and expected_template_count != count:
            raise RuntimeError("gallery manifest template count is inconsistent")
        sidecars = (
            self._metadata,
            self._breed_index,
            self._availability,
            self._identity_kinds,
            self._enrollment_ranks,
            self._enrollment_views,
            self._duplicate_group_ids,
        )
        if any(len(sidecar) != count for sidecar in sidecars) or any(
            index not in sidecar for sidecar in sidecars for index in range(count)
        ):
            raise RuntimeError("gallery index and sidecar cardinality are inconsistent")
        channel_names = {channel.name for channel in self._channels}
        required_names = {channel.name for channel in self._required_channels}
        ids: set[str] = set()
        template_ids: set[str] = set()
        content_hashes: set[str] = set()
        idempotency_keys: set[str] = set()
        for idx in range(count):
            meta = self._metadata[idx]
            if not isinstance(meta, dict) or set(meta) != {
                "registered_dog_id", "template_id", "content_sha256",
                "idempotency_key", "template_schema", "metadata",
            }:
                raise RuntimeError("gallery metadata row has an invalid schema")
            registered_id = meta["registered_dog_id"]
            if not _is_canonical_uuid5(registered_id):
                raise RuntimeError(
                    "gallery contains a non-canonical UUIDv5 registered_dog_id"
                )
            if not isinstance(meta["metadata"], dict):
                raise RuntimeError(  # noqa: TRY004
                    "gallery identity metadata must be an object"
                )
            try:
                _canonical_metadata(meta["metadata"])
            except ValueError as exc:
                raise RuntimeError("gallery identity metadata is invalid") from exc
            template_id = meta["template_id"]
            content_sha256 = meta["content_sha256"]
            idempotency_key = meta["idempotency_key"]
            if meta["template_schema"] not in {
                _TEMPLATE_SCHEMA_V1, _TEMPLATE_SCHEMA_V2
            }:
                raise RuntimeError("gallery template metadata has an invalid schema")
            if (
                self._loaded_manifest_schema == _MANIFEST_SCHEMA_V4
                and meta["template_schema"] != _TEMPLATE_SCHEMA_V1
            ):
                raise RuntimeError("gallery v4 contains a non-v1 template schema")
            if not _is_sha256(content_sha256):
                raise RuntimeError("gallery contains an invalid template content hash")
            if template_id != _template_id(content_sha256):
                raise RuntimeError("gallery contains a non-deterministic template ID")
            if not _is_bounded_utf8_text(
                idempotency_key,
                maximum_bytes=_MAXIMUM_IDEMPOTENCY_KEY_BYTES,
                allow_empty=False,
            ):
                raise RuntimeError("gallery contains an invalid idempotency key")
            if set(meta["metadata"]) & _RESERVED_METADATA_KEYS:
                raise RuntimeError("gallery user metadata contains reserved fields")
            if not isinstance(self._breed_index[idx], str):
                raise RuntimeError(  # noqa: TRY004
                    "gallery breed metadata must be a string"
                )
            kind = self._identity_kinds[idx]
            if not isinstance(kind, IdentityEvidenceKind):
                raise RuntimeError(  # noqa: TRY004
                    "gallery identity evidence kind is invalid"
                )
            try:
                self._registry_policy.validate(registered_id, kind)
            except ValueError as exc:
                raise RuntimeError("gallery identity violates its registry policy") from exc
            rank = self._enrollment_ranks[idx]
            view = self._enrollment_views[idx]
            if (rank is None) != (view is None) or (
                rank is not None and not isinstance(rank, EnrollmentRank)
            ) or (
                view is not None
                and not _is_bounded_utf8_text(
                    view,
                    maximum_bytes=_MAXIMUM_PROVENANCE_TEXT_BYTES,
                    allow_empty=False,
                )
            ):
                raise RuntimeError("gallery enrollment rank/view provenance is invalid")
            duplicate_group_ids = self._duplicate_group_ids[idx]
            if (
                not isinstance(duplicate_group_ids, tuple)
                or tuple(sorted(set(duplicate_group_ids))) != duplicate_group_ids
                or any(
                    not _is_bounded_utf8_text(
                        value,
                        maximum_bytes=_MAXIMUM_PROVENANCE_TEXT_BYTES,
                        allow_empty=False,
                    )
                    for value in duplicate_group_ids
                )
            ):
                raise RuntimeError("gallery duplicate-group provenance is invalid")
            row_availability = self._availability[idx]
            if (
                not isinstance(row_availability, dict)
                or set(row_availability) != channel_names
                or any(not isinstance(value, bool) for value in row_availability.values())
                or not all(row_availability[name] for name in required_names)
            ):
                raise RuntimeError("gallery availability sidecar is invalid")
            required_vector = self._index.reconstruct(idx)
            offset = 0
            for channel in self._required_channels:
                _validate_unit_vector(
                    required_vector[offset:offset + channel.dimension],
                    channel.dimension,
                    channel.name,
                )
                offset += channel.dimension
            if template_id in template_ids:
                raise RuntimeError("gallery contains duplicate template IDs")
            if content_sha256 in content_hashes:
                raise RuntimeError("gallery contains duplicate template content")
            if idempotency_key in idempotency_keys:
                raise RuntimeError("gallery contains duplicate idempotency keys")
            ids.add(registered_id)
            template_ids.add(template_id)
            content_hashes.add(content_sha256)
            idempotency_keys.add(idempotency_key)
        for channel in self._optional_channels:
            vectors = self._optional_vectors[channel.name]
            available_count = sum(
                self._availability[idx][channel.name] for idx in range(count)
            )
            if len(vectors) != available_count or any(
                (idx in vectors) is not self._availability[idx][channel.name]
                for idx in range(count)
            ):
                raise RuntimeError(
                    f"gallery optional channel {channel.name!r} disagrees with availability"
                )
            for vector in vectors.values():
                _validate_unit_vector(vector, channel.dimension, channel.name)
        identity_count = len(ids)
        if expected_identity_count is not None and expected_identity_count != identity_count:
            raise RuntimeError("gallery manifest identity count is inconsistent")

    def _rebuild_runtime_indexes(self) -> None:
        self._template_id_index = {}
        self._content_sha256_index = {}
        self._idempotency_key_index = {}
        self._duplicate_group_index = {}
        self._identity_rows = {}
        self._identity_ordinals = {}
        self._identities_by_ordinal = []
        row_ordinals: list[int] = []
        for row in range(int(self._index.ntotal)):
            metadata = self._metadata[row]
            self._template_id_index[metadata["template_id"]] = row
            self._content_sha256_index[metadata["content_sha256"]] = row
            self._idempotency_key_index[metadata["idempotency_key"]] = row
            identity_id = metadata["registered_dog_id"]
            if identity_id not in self._identity_ordinals:
                self._identity_ordinals[identity_id] = len(self._identity_ordinals)
                self._identity_rows[identity_id] = []
                self._identities_by_ordinal.append(identity_id)
            ordinal = self._identity_ordinals[identity_id]
            self._identity_rows[identity_id].append(row)
            row_ordinals.append(ordinal)
            for duplicate_group_id in self._duplicate_group_ids[row]:
                self._duplicate_group_index.setdefault(duplicate_group_id, set()).add(row)
        self._row_identity_ordinals = array("q", row_ordinals)

    def enroll(
        self,
        embedding: np.ndarray | dict[str, np.ndarray],
        registered_dog_id: str,
        metadata: dict | None = None,
        idempotency_key: str | None = None,
        content_sha256: str | None = None,
        *,
        availability: dict[str, bool] | None = None,
        identity_evidence_kind: IdentityEvidenceKind = IdentityEvidenceKind.REGISTERED,
        enrollment_rank: EnrollmentRank | None = None,
        enrollment_view: str | None = None,
        duplicate_group_ids: tuple[str, ...] = (),
    ) -> int:
        return self.enroll_with_breed(
            embedding, registered_dog_id, "unknown", metadata, idempotency_key,
            content_sha256, availability=availability,
            identity_evidence_kind=identity_evidence_kind,
            enrollment_rank=enrollment_rank,
            enrollment_view=enrollment_view,
            duplicate_group_ids=duplicate_group_ids,
        )

    def enroll_with_breed(
        self,
        embedding: np.ndarray | dict[str, np.ndarray],
        dog_id: str,
        breed: str,
        metadata: dict | None = None,
        idempotency_key: str | None = None,
        content_sha256: str | None = None,
        *,
        availability: dict[str, bool] | None = None,
        identity_evidence_kind: IdentityEvidenceKind = IdentityEvidenceKind.REGISTERED,
        enrollment_rank: EnrollmentRank | None = None,
        enrollment_view: str | None = None,
        duplicate_group_ids: tuple[str, ...] = (),
    ) -> int:
        return self._enroll_with_breed(
            embedding,
            dog_id,
            breed,
            metadata,
            idempotency_key,
            content_sha256,
            availability=availability,
            identity_evidence_kind=identity_evidence_kind,
            enrollment_rank=enrollment_rank,
            enrollment_view=enrollment_view,
            duplicate_group_ids=duplicate_group_ids,
        )

    def _enroll_with_breed(
        self,
        embedding: np.ndarray | dict[str, np.ndarray],
        dog_id: str,
        breed: str,
        metadata: dict | None = None,
        idempotency_key: str | None = None,
        content_sha256: str | None = None,
        *,
        availability: dict[str, bool] | None = None,
        identity_evidence_kind: IdentityEvidenceKind = IdentityEvidenceKind.REGISTERED,
        enrollment_rank: EnrollmentRank | None = None,
        enrollment_view: str | None = None,
        duplicate_group_ids: tuple[str, ...] = (),
        prepared_vectors: dict[str, np.ndarray] | None = None,
    ) -> int:
        if not _is_canonical_uuid5(dog_id):
            raise ValueError(
                "registered_dog_id must be a canonical lowercase UUIDv5 string"
            )
        self._ensure_writer()
        if not isinstance(identity_evidence_kind, IdentityEvidenceKind):
            raise TypeError("identity_evidence_kind must be an IdentityEvidenceKind")
        self._registry_policy.validate(dog_id, identity_evidence_kind)
        if (enrollment_rank is None) != (enrollment_view is None):
            raise ValueError("enrollment rank and view must be provided together")
        if enrollment_rank is not None and not isinstance(enrollment_rank, EnrollmentRank):
            raise TypeError("enrollment_rank must be K1, K3, or K5")
        if enrollment_view is not None and not _is_bounded_utf8_text(
            enrollment_view,
            maximum_bytes=_MAXIMUM_PROVENANCE_TEXT_BYTES,
            allow_empty=False,
        ):
            raise ValueError("enrollment_view must be bounded non-empty UTF-8 text")
        if not isinstance(duplicate_group_ids, tuple):
            duplicate_group_ids = tuple(duplicate_group_ids)
        duplicate_group_ids = tuple(sorted(set(duplicate_group_ids)))
        if any(
            not _is_bounded_utf8_text(
                value,
                maximum_bytes=_MAXIMUM_PROVENANCE_TEXT_BYTES,
                allow_empty=False,
            )
            for value in duplicate_group_ids
        ):
            raise ValueError("duplicate_group_ids must contain bounded non-empty text")
        if int(self._index.ntotal) >= _MAXIMUM_GALLERY_TEMPLATES:
            raise RuntimeError("gallery template cardinality limit has been reached")
        if not isinstance(breed, str):
            raise TypeError("breed must be a string")
        canonical_metadata = _canonical_metadata(metadata)
        vectors = (
            self._canonical_vectors(embedding)
            if prepared_vectors is None
            else prepared_vectors
        )
        if content_sha256 is None:
            content_sha256 = self._vector_content_sha256(vectors)
        elif not _is_sha256(content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        template_id = _template_id(content_sha256)
        if idempotency_key is None:
            idempotency_key = template_id
        elif not _is_bounded_utf8_text(
            idempotency_key,
            maximum_bytes=_MAXIMUM_IDEMPOTENCY_KEY_BYTES,
            allow_empty=False,
        ):
            raise ValueError("idempotency_key must be a bounded non-empty UTF-8 string")

        derived_availability = {
            channel.name: channel.name in vectors for channel in self._channels
        }
        if availability is not None and availability != derived_availability:
            raise ValueError("gallery-key availability differs from its vectors")
        availability = derived_availability
        existing_content_row = self._content_sha256_index.get(content_sha256)
        existing_idempotency_row = self._idempotency_key_index.get(idempotency_key)
        if existing_content_row is not None:
            existing = self._metadata[existing_content_row]
            if existing["registered_dog_id"] != dog_id:
                raise ValueError(
                    f"template/content {template_id!r} is already bound to different "
                    f"registered identity {existing['registered_dog_id']!r}"
                )
            exact_retry = (
                existing["idempotency_key"] == idempotency_key
                and existing["metadata"] == canonical_metadata
                and self._breed_index[existing_content_row] == breed
                and self._availability[existing_content_row] == availability
                and self._identity_kinds[existing_content_row] is identity_evidence_kind
                and self._enrollment_ranks[existing_content_row] is enrollment_rank
                and self._enrollment_views[existing_content_row] == enrollment_view
                and self._duplicate_group_ids[existing_content_row] == duplicate_group_ids
                and self._vectors_equal(existing_content_row, vectors)
            )
            if exact_retry:
                return int(existing_content_row)
            if existing["idempotency_key"] == idempotency_key:
                raise ValueError(
                    f"idempotency key {idempotency_key!r} conflicts with an existing enrollment"
                )
            else:
                raise ValueError(
                    f"template/content {template_id!r} is already enrolled with different "
                    "immutable evidence or metadata"
                )
        if existing_idempotency_row is not None:
            raise ValueError(
                f"idempotency key {idempotency_key!r} conflicts with an existing enrollment"
            )

        idx = int(self._index.ntotal)
        key = GalleryKey(idx, vectors, availability)
        value = GalleryValue(
            template_row=idx,
            registered_identity_id=dog_id,
            template_id=template_id,
            content_sha256=content_sha256,
            idempotency_key=idempotency_key,
            template_schema=(
                _TEMPLATE_SCHEMA_V2
                if enrollment_rank is not None or duplicate_group_ids
                else _TEMPLATE_SCHEMA_V1
            ),
            breed=breed,
            metadata=canonical_metadata,
            identity_evidence_kind=identity_evidence_kind,
            enrollment_rank=enrollment_rank,
            enrollment_view=enrollment_view,
            duplicate_group_ids=duplicate_group_ids,
        )
        required = np.concatenate(
            [key.vectors[channel.name] for channel in self._required_channels]
        ).astype(np.float32, copy=False)
        self._index.add(required.reshape(1, -1))
        for channel in self._optional_channels:
            if channel.name in key.vectors:
                self._optional_vectors[channel.name][idx] = key.vectors[channel.name]
        self._availability[idx] = dict(key.availability)
        self._metadata[idx] = {
            "registered_dog_id": value.registered_identity_id,
            "template_id": value.template_id,
            "content_sha256": value.content_sha256,
            "idempotency_key": value.idempotency_key,
            "template_schema": value.template_schema,
            "metadata": value.metadata,
        }
        self._breed_index[idx] = value.breed
        self._identity_kinds[idx] = value.identity_evidence_kind
        self._enrollment_ranks[idx] = value.enrollment_rank
        self._enrollment_views[idx] = value.enrollment_view
        self._duplicate_group_ids[idx] = value.duplicate_group_ids
        self._template_id_index[value.template_id] = idx
        self._content_sha256_index[value.content_sha256] = idx
        self._idempotency_key_index[value.idempotency_key] = idx
        identity_id = value.registered_identity_id
        if identity_id not in self._identity_ordinals:
            self._identity_ordinals[identity_id] = len(self._identity_ordinals)
            self._identity_rows[identity_id] = []
            self._identities_by_ordinal.append(identity_id)
        self._identity_rows[identity_id].append(idx)
        self._row_identity_ordinals.append(self._identity_ordinals[identity_id])
        for duplicate_group_id in value.duplicate_group_ids:
            self._duplicate_group_index.setdefault(duplicate_group_id, set()).add(idx)
        return idx

    def enroll_many(self, enrollments: list[GalleryEnrollment]) -> list[int]:
        """Enroll a deterministic bulk without publishing intermediate rows."""

        self._ensure_writer()
        if not isinstance(enrollments, list) or any(
            not isinstance(value, GalleryEnrollment) for value in enrollments
        ):
            raise TypeError("enrollments must be a list of GalleryEnrollment values")
        ordered: list[
            tuple[
                tuple[str, str, str, str],
                GalleryEnrollment,
                dict[str, np.ndarray],
                str,
            ]
        ] = []
        for enrollment in enrollments:
            vectors = self._canonical_vectors(enrollment.embedding)
            content_sha256 = enrollment.content_sha256
            if content_sha256 is None:
                content_sha256 = self._vector_content_sha256(vectors)
            ordered.append((
                (
                    enrollment.registered_identity_id,
                    enrollment.enrollment_rank.value
                    if isinstance(enrollment.enrollment_rank, EnrollmentRank)
                    else "",
                    enrollment.enrollment_view or "",
                    content_sha256,
                ),
                enrollment,
                vectors,
                content_sha256,
            ))
        rows: list[int] = []
        for _, enrollment, vectors, content_sha256 in sorted(
            ordered, key=lambda item: item[0]
        ):
            rows.append(self._enroll_with_breed(
                enrollment.embedding,
                enrollment.registered_identity_id,
                enrollment.breed,
                enrollment.metadata,
                enrollment.idempotency_key,
                content_sha256,
                availability=enrollment.availability,
                identity_evidence_kind=enrollment.identity_evidence_kind,
                enrollment_rank=enrollment.enrollment_rank,
                enrollment_view=enrollment.enrollment_view,
                duplicate_group_ids=enrollment.duplicate_group_ids,
                prepared_vectors=vectors,
            ))
        return rows

    def _vector_content_sha256(self, vectors: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for channel in self._channels:
            digest.update(channel.name.encode("utf-8"))
            digest.update(b"\0")
            if channel.name in vectors:
                digest.update(
                    vectors[channel.name].astype("<f4", copy=False).tobytes()
                )
        return digest.hexdigest()

    @classmethod
    def build(
        cls,
        gallery_directory: Path,
        enrollments: list[GalleryEnrollment],
        dim: int = 640,
        embedding_contract: dict[str, Any] | None = None,
        *,
        registry_policy: IdentityRegistryPolicy | None = None,
    ) -> IdentityGallery:
        """Build and atomically publish one immutable content-addressed generation."""

        destination = Path(os.path.abspath(gallery_directory))
        parent = destination.parent
        if not parent.is_dir() or parent.is_symlink():
            raise RuntimeError("bulk gallery build parent must be a non-symlink directory")
        if _path_entry_exists(destination):
            raise FileExistsError("bulk gallery build destination already exists")
        staging = parent / f".{destination.name}.build-{uuid4().hex}"
        gallery: IdentityGallery | None = None
        published = False
        try:
            gallery = cls(
                staging,
                dim=dim,
                embedding_contract=embedding_contract,
                registry_policy=registry_policy,
            )
            gallery.enroll_many(enrollments)
            gallery.save()
            gallery.close()
            _publish_directory_no_replace(staging, destination)
            published = True
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return cls(
                destination,
                dim=dim,
                embedding_contract=embedding_contract,
                read_only=True,
                registry_policy=registry_policy,
            )
        except Exception:
            if gallery is not None:
                gallery.close()
            raise
        finally:
            if not published and _path_entry_exists(staging):
                shutil.rmtree(staging)

    def _canonical_vectors(
        self,
        embedding: np.ndarray | dict[str, np.ndarray],
        *,
        normalize: bool = True,
    ) -> dict[str, np.ndarray]:
        if isinstance(embedding, np.ndarray):
            if len(self._channels) != 1:
                raise ValueError("multi-channel gallery input must be a channel mapping")
            raw = {self._channels[0].name: embedding}
        elif isinstance(embedding, dict):
            raw = embedding
        else:
            raise TypeError("embedding must be a vector or channel mapping")
        names = {channel.name for channel in self._channels}
        if any(not isinstance(name, str) for name in raw) or not set(raw) <= names:
            raise ValueError("embedding contains unknown channel names")
        missing_required = {
            channel.name for channel in self._required_channels
        } - set(raw)
        if missing_required:
            raise ValueError(
                f"required embedding channels are missing: {sorted(missing_required)}"
            )
        vectors: dict[str, np.ndarray] = {}
        dimensions = {channel.name: channel.dimension for channel in self._channels}
        for name, value in raw.items():
            vector = np.asarray(value, dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] != dimensions[name]:
                raise ValueError(
                    f"channel {name!r} must be a {dimensions[name]}-d vector"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"channel {name!r} must contain only finite values")
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-8:
                raise ValueError(f"channel {name!r} must have non-zero finite norm")
            vectors[name] = (
                np.asarray(vector / norm, dtype=np.float32)
                if normalize
                else vector.copy()
            )
        return vectors

    def _vectors_equal(self, index: int, vectors: dict[str, np.ndarray]) -> bool:
        existing = self._gallery_key(index).vectors
        return set(existing) == set(vectors) and all(
            np.array_equal(existing[name], vectors[name]) for name in existing
        )

    def _gallery_key(self, index: int) -> GalleryKey:
        required = self._index.reconstruct(index)
        vectors: dict[str, np.ndarray] = {}
        offset = 0
        for channel in self._required_channels:
            vectors[channel.name] = required[offset:offset + channel.dimension].copy()
            offset += channel.dimension
        for channel in self._optional_channels:
            vector = self._optional_vectors[channel.name].get(index)
            if vector is not None:
                vectors[channel.name] = vector
        return GalleryKey(index, vectors, dict(self._availability[index]))

    def _gallery_value(self, index: int) -> GalleryValue:
        metadata = self._metadata[index]
        return GalleryValue(
            template_row=index,
            registered_identity_id=metadata["registered_dog_id"],
            template_id=metadata["template_id"],
            content_sha256=metadata["content_sha256"],
            idempotency_key=metadata["idempotency_key"],
            template_schema=metadata["template_schema"],
            breed=self._breed_index[index],
            metadata=metadata["metadata"],
            identity_evidence_kind=self._identity_kinds[index],
            enrollment_rank=self._enrollment_ranks[index],
            enrollment_view=self._enrollment_views[index],
            duplicate_group_ids=self._duplicate_group_ids[index],
        )

    def prepare_query(
        self,
        vectors: np.ndarray | dict[str, np.ndarray],
        availability: dict[str, bool] | None = None,
        exclusions: QueryExclusions | None = None,
    ) -> RetrievalQuery:
        canonical = self._canonical_vectors(vectors, normalize=False)
        derived = {
            channel.name: channel.name in canonical for channel in self._channels
        }
        if availability is not None and availability != derived:
            raise ValueError("query availability differs from its vectors")
        return RetrievalQuery(canonical, derived, exclusions or QueryExclusions())

    def _validated_query(
        self, query: np.ndarray | dict[str, np.ndarray] | RetrievalQuery
    ) -> RetrievalQuery:
        if not isinstance(query, RetrievalQuery):
            return self.prepare_query(query)
        expected_names = {channel.name for channel in self._channels}
        if set(query.availability) != expected_names:
            raise ValueError("query availability differs from gallery channels")
        missing_required = {
            channel.name for channel in self._required_channels
        } - set(query.vectors)
        if missing_required:
            raise ValueError(
                f"required embedding channels are missing: {sorted(missing_required)}"
            )
        dimensions = {channel.name: channel.dimension for channel in self._channels}
        for name, vector in query.vectors.items():
            if name not in dimensions or vector.shape != (dimensions[name],):
                raise ValueError(f"query channel {name!r} dimension differs")
        return query

    def search_filtered(
        self,
        query: np.ndarray | dict[str, np.ndarray] | RetrievalQuery,
        allowed_breeds: list[str] | None = None,
        top_k: int = 5,
        *,
        exclusions: QueryExclusions | None = None,
    ) -> list[tuple[int, float, dict]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self._index.ntotal == 0:
            return []
        prepared = self._query_with_exclusions(query, exclusions)
        eligible = self._eligible_rows(prepared.exclusions, allowed_breeds)
        scores = self._all_template_scores(prepared, eligible)
        return [
            self._result_tuple(prepared, candidate)
            for candidate in self._ranked_candidates(prepared, scores, top_k)
        ]

    def search(
        self,
        query: np.ndarray | dict[str, np.ndarray] | RetrievalQuery,
        top_k: int = 5,
        *,
        exclusions: QueryExclusions | None = None,
    ) -> list[tuple[int, float, dict]]:
        return self.search_filtered(query, None, top_k, exclusions=exclusions)

    def rank_of_identity(
        self,
        query: np.ndarray | dict[str, np.ndarray] | RetrievalQuery,
        registered_dog_id: str,
        *,
        exclusions: QueryExclusions | None = None,
    ) -> int | None:
        """Return one identity's exact rank without materializing search results."""

        if not _is_canonical_uuid5(registered_dog_id):
            raise ValueError("registered_dog_id must be a canonical lowercase UUIDv5 string")
        target_ordinal = self._identity_ordinals.get(registered_dog_id)
        if target_ordinal is None or self._index.ntotal == 0:
            return None
        prepared = self._query_with_exclusions(query, exclusions)
        eligible = self._eligible_rows(prepared.exclusions, None)
        scores = self._all_template_scores(prepared, eligible)
        identity_scores = np.full(
            len(self._identity_ordinals), -np.inf, dtype=scores.dtype
        )
        np.maximum.at(
            identity_scores,
            self._row_identity_ordinal_array(),
            scores,
        )
        target_score = identity_scores[target_ordinal]
        if not np.isfinite(target_score):
            return None
        precedes_tied_target = sum(
            identity_scores[ordinal] == target_score
            and identity_id < registered_dog_id
            for ordinal, identity_id in enumerate(self._identities_by_ordinal)
        )
        return 1 + int(
            np.count_nonzero(identity_scores > target_score)
            + precedes_tied_target
        )

    def explain_identity(
        self,
        query: np.ndarray | dict[str, np.ndarray] | RetrievalQuery,
        registered_dog_id: str,
        *,
        exclusions: QueryExclusions | None = None,
    ) -> tuple[int, float, dict] | None:
        prepared = self._query_with_exclusions(query, exclusions)
        rows = self._identity_rows.get(registered_dog_id)
        if not rows:
            return None
        eligible = self._eligible_rows(prepared.exclusions, None)
        scores = self._all_template_scores(prepared, eligible)
        eligible_rows = [row for row in rows if np.isfinite(scores[row])]
        if not eligible_rows:
            return None
        best_score = max(float(scores[row]) for row in eligible_rows)
        winning_row = min(
            (row for row in eligible_rows if float(scores[row]) == best_score),
            key=lambda row: self._metadata[row]["template_id"],
        )
        return self._result_tuple(
            prepared, self._scored_candidate(prepared, winning_row)
        )

    def _query_with_exclusions(
        self,
        query: np.ndarray | dict[str, np.ndarray] | RetrievalQuery,
        exclusions: QueryExclusions | None,
    ) -> RetrievalQuery:
        prepared = self._validated_query(query)
        if exclusions is None:
            return prepared
        if not isinstance(exclusions, QueryExclusions):
            raise TypeError("exclusions must be a QueryExclusions value")
        if prepared.exclusions != QueryExclusions():
            raise ValueError("query and search call both specify exclusions")
        return RetrievalQuery(prepared.vectors, prepared.availability, exclusions)

    def _eligible_rows(
        self,
        exclusions: QueryExclusions,
        allowed_breeds: list[str] | None,
    ) -> np.ndarray | None:
        breed_set = set(allowed_breeds or ())
        if not breed_set and exclusions == QueryExclusions():
            return None
        eligible = np.ones(int(self._index.ntotal), dtype=np.bool_)
        if breed_set:
            for row, breed in self._breed_index.items():
                if breed not in breed_set:
                    eligible[row] = False
        for template_id in exclusions.template_ids:
            row = self._template_id_index.get(template_id)
            if row is not None:
                eligible[row] = False
        for content_sha256 in exclusions.content_sha256s:
            row = self._content_sha256_index.get(content_sha256)
            if row is not None:
                eligible[row] = False
        for duplicate_group_id in exclusions.duplicate_group_ids:
            rows = self._duplicate_group_index.get(duplicate_group_id)
            if rows:
                eligible[np.fromiter(rows, dtype=np.int64)] = False
        return eligible

    def _all_template_scores(
        self, query: RetrievalQuery, eligible: np.ndarray | None
    ) -> np.ndarray:
        count = int(self._index.ntotal)
        if self._full128_fast_path:
            scores = np.empty(count, dtype=np.float32)
            query_vector = query.vectors[FULL128_CHANNEL]
            for start in range(0, count, _SEARCH_BLOCK_ROWS):
                length = min(_SEARCH_BLOCK_ROWS, count - start)
                matrix = self._index.reconstruct_n(start, length)
                scores[start:start + length] = matrix @ query_vector
            if eligible is not None:
                scores[~eligible] = -np.inf
            return scores

        numerator = np.zeros(count, dtype=np.float64)
        required_weight = sum(channel.weight for channel in self._required_channels)
        active_optional_channels = tuple(
            channel
            for channel in self._optional_channels
            if query.availability[channel.name] and channel.weight != 0.0
        )
        denominator = (
            np.full(count, required_weight, dtype=np.float64)
            if active_optional_channels
            else None
        )
        for start in range(0, count, _SEARCH_BLOCK_ROWS):
            length = min(_SEARCH_BLOCK_ROWS, count - start)
            matrix = self._index.reconstruct_n(start, length)
            offset = 0
            for channel in self._required_channels:
                section = matrix[:, offset:offset + channel.dimension]
                numerator[start:start + length] += (
                    channel.weight * (section @ query.vectors[channel.name])
                )
                offset += channel.dimension
        for channel in active_optional_channels:
            query_vector = query.vectors[channel.name]
            for row, vector in self._optional_vectors[channel.name].items():
                numerator[row] += channel.weight * float(np.dot(query_vector, vector))
                assert denominator is not None
                denominator[row] += channel.weight
        np.divide(
            numerator,
            required_weight if denominator is None else denominator,
            out=numerator,
        )
        scores = numerator
        if eligible is not None:
            scores[~eligible] = -np.inf
        return scores

    def _ranked_candidates(
        self, query: RetrievalQuery, scores: np.ndarray, top_k: int
    ) -> list[ScoredGalleryValue]:
        identity_count = len(self._identity_ordinals)
        identity_scores = np.full(identity_count, -np.inf, dtype=scores.dtype)
        np.maximum.at(identity_scores, self._row_identity_ordinal_array(), scores)
        ranked_ordinals = heapq.nsmallest(
            top_k,
            (
                ordinal
                for ordinal, score in enumerate(identity_scores)
                if np.isfinite(score)
            ),
            key=lambda ordinal: (
                -float(identity_scores[ordinal]),
                self._identities_by_ordinal[ordinal],
            ),
        )
        candidates: list[ScoredGalleryValue] = []
        for ordinal in ranked_ordinals:
            identity_id = self._identities_by_ordinal[ordinal]
            best_score = float(identity_scores[ordinal])
            winning_row = min(
                (
                    row
                    for row in self._identity_rows[identity_id]
                    if float(scores[row]) == best_score
                ),
                key=lambda row: self._metadata[row]["template_id"],
            )
            candidates.append(self._scored_candidate(query, winning_row))
        return candidates

    def _row_identity_ordinal_array(self) -> np.ndarray:
        return np.frombuffer(self._row_identity_ordinals, dtype=np.int64)

    def _scored_candidate(
        self, query: RetrievalQuery, row: int
    ) -> ScoredGalleryValue:
        key = self._gallery_key(row)
        return ScoredGalleryValue(
            value=self._gallery_value(row),
            query_key_score=self._scorer.score(query, key),
            template_availability=dict(key.availability),
        )

    def _result_tuple(
        self, query: RetrievalQuery, candidate: ScoredGalleryValue
    ) -> tuple[int, float, dict[str, Any]]:
        value = candidate.value
        score = candidate.query_key_score
        meta = {
            "registered_dog_id": value.registered_identity_id,
            "template_id": value.template_id,
            "content_sha256": value.content_sha256,
            "idempotency_key": value.idempotency_key,
            "template_schema": value.template_schema,
            "metadata": deepcopy(value.metadata),
        }
        meta["_evidence"] = dict(score.evidence)
        meta["_evidence_availability"] = dict(score.evidence_availability)
        meta["_query_availability"] = dict(query.availability)
        meta["_template_availability"] = dict(candidate.template_availability)
        meta["_scorer_hash"] = self._scorer_hash
        meta["_exact"] = True
        meta["_identity_evidence_kind"] = value.identity_evidence_kind.value
        meta["_enrollment_rank"] = (
            value.enrollment_rank.value if value.enrollment_rank is not None else None
        )
        meta["_enrollment_view"] = value.enrollment_view
        meta["_duplicate_group_ids"] = list(value.duplicate_group_ids)
        meta["_winning_template_row"] = value.template_row
        return value.template_row, score.similarity, meta

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def close(self) -> None:
        if self._lock_stream is not None:
            try:
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_stream.close()
                self._lock_stream = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            return

    def _ensure_writer(self) -> None:
        if self._read_only or self._lock_stream is None:
            raise RuntimeError("gallery was opened read-only")

    def save(self) -> None:
        self._ensure_writer()
        self._validate_state()
        template_count = int(self._index.ntotal)
        enhanced = (
            self._loaded_manifest_schema == _MANIFEST_SCHEMA_V5
            or self._registry_policy.descriptor["registry_sha256"] is not None
            or any(rank is not None for rank in self._enrollment_ranks.values())
            or any(self._duplicate_group_ids.values())
        )
        manifest_schema = _MANIFEST_SCHEMA_V5 if enhanced else _MANIFEST_SCHEMA_V4
        serialized_metadata: dict[int, dict[str, Any]] = {}
        for row, metadata in self._metadata.items():
            serialized_metadata[row] = dict(metadata)
            if enhanced:
                serialized_metadata[row].update({
                    "identity_evidence_kind": self._identity_kinds[row].value,
                    "enrollment_rank": (
                        self._enrollment_ranks[row].value
                        if self._enrollment_ranks[row] is not None
                        else None
                    ),
                    "enrollment_view": self._enrollment_views[row],
                    "duplicate_group_ids": list(self._duplicate_group_ids[row]),
                })
        binary_limits = _binary_file_limits(
            template_count, self._required_dim, self._optional_channels
        )
        token = uuid4().hex
        temporary = {
            "required_index": self._base_dir / f".required-{token}.tmp",
            "optional_vectors": self._base_dir / f".optional-{token}.tmp",
            "availability": self._base_dir / f".availability-{token}.tmp",
            "metadata": self._base_dir / f".metadata-{token}.tmp",
            "breeds": self._base_dir / f".breeds-{token}.tmp",
        }
        manifest_tmp = self._base_dir / f".manifest-{token}.tmp"
        published_destinations: list[Path] = []
        manifest_published = False
        try:
            faiss.write_index(self._index, str(temporary["required_index"]))
            with temporary["required_index"].open("rb") as stream:
                os.fsync(stream.fileno())
            sparse: dict[str, np.ndarray] = {}
            for channel_index, channel in enumerate(self._optional_channels):
                values = self._optional_vectors[channel.name]
                rows = np.asarray(sorted(values), dtype=np.int64)
                vectors = (
                    np.stack([values[int(row)] for row in rows]).astype(np.float32)
                    if len(rows)
                    else np.empty((0, channel.dimension), dtype=np.float32)
                )
                sparse[f"c{channel_index}_rows"] = rows
                sparse[f"c{channel_index}_vectors"] = vectors
            with temporary["optional_vectors"].open("xb") as stream:
                np.savez_compressed(stream, **sparse)
                stream.flush()
                os.fsync(stream.fileno())
            _write_json_file(
                temporary["availability"],
                self._availability,
                "gallery availability sidecar",
                maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            )
            _write_json_file(
                temporary["metadata"],
                serialized_metadata,
                "gallery metadata sidecar",
                maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            )
            _write_json_file(
                temporary["breeds"],
                self._breed_index,
                "gallery breeds sidecar",
                maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            )
            file_limits = {
                **binary_limits,
                "availability": _MAXIMUM_SIDECAR_JSON_BYTES,
                "metadata": _MAXIMUM_SIDECAR_JSON_BYTES,
                "breeds": _MAXIMUM_SIDECAR_JSON_BYTES,
            }
            hashes = {
                kind: _sha256_file(
                    path,
                    f"generated gallery {kind} file",
                    maximum_bytes=file_limits[kind],
                )
                for kind, path in temporary.items()
            }
            generation = hashlib.sha256(
                "".join(hashes[kind] for kind in sorted(hashes)).encode("ascii")
            ).hexdigest()
            suffixes = {
                "required_index": "idx",
                "optional_vectors": "npz",
                "availability": "json",
                "metadata": "json",
                "breeds": "json",
            }
            destinations = {
                kind: self._base_dir / f"{kind}-{generation}.{suffixes[kind]}"
                for kind in temporary
            }
            for kind, destination in destinations.items():
                if not _path_entry_exists(destination):
                    continue
                if _sha256_file(
                    destination,
                    f"gallery {kind} generation file",
                    maximum_bytes=file_limits[kind],
                ) != hashes[kind]:
                    raise RuntimeError(
                        f"existing content-addressed gallery {kind} generation "
                        "file is corrupted; refusing to overwrite it"
                    )
            manifest = {
                "schema_version": manifest_schema,
                "dimension": self._dim,
                "required_dimension": self._required_dim,
                "embedding_contract": self._embedding_contract,
                "count": template_count,
                "template_count": template_count,
                "identity_count": len({
                    meta["registered_dog_id"] for meta in self._metadata.values()
                }),
                "identity_aggregation": _IDENTITY_AGGREGATION,
                "scorer": {
                    "algorithm": SCORER_ALGORITHM,
                    "hash": self._scorer_hash,
                    "exact": True,
                },
                "files": {
                    kind: {"name": path.name, "sha256": hashes[kind]}
                    for kind, path in destinations.items()
                },
            }
            if enhanced:
                manifest["identity_policy"] = self._registry_policy.descriptor
            manifest_path = self._base_dir / "gallery_manifest.json"
            if _path_entry_exists(manifest_path):
                manifest_stat = manifest_path.lstat()
                if stat.S_ISLNK(manifest_stat.st_mode):
                    raise RuntimeError("gallery manifest must not be a symbolic link")
                if not stat.S_ISREG(manifest_stat.st_mode):
                    raise RuntimeError("gallery manifest must be a regular file")
            _write_json_file(
                manifest_tmp,
                manifest,
                "gallery manifest",
                maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
            )
            if _read_strict_json_object(
                manifest_tmp,
                "generated gallery manifest",
                maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
            ) != manifest:
                raise RuntimeError("generated gallery manifest failed preflight")
            for kind, source in temporary.items():
                destination = destinations[kind]
                if _path_entry_exists(destination):
                    source.unlink()
                else:
                    os.replace(source, destination)
                    published_destinations.append(destination)
            os.replace(manifest_tmp, manifest_path)
            manifest_published = True
            self._loaded_manifest_schema = manifest_schema
            directory_fd = os.open(self._base_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if not manifest_published:
                for path in published_destinations:
                    if _path_entry_exists(path):
                        path.unlink()
            for path in (*temporary.values(), manifest_tmp):
                if _path_entry_exists(path):
                    path.unlink()


def _canonical_embedding_contract(
    contract: dict[str, Any] | None,
    dimension: int,
) -> dict[str, Any]:
    value = contract or {
        "schema_version": "cvi.gallery_embedding_contract.v1",
        "kind": "opaque",
        "dimension": dimension,
    }
    if not isinstance(value, dict):
        raise TypeError("embedding contract must be an object")
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding contract must be finite JSON") from exc
    canonical = json.loads(encoded)
    if canonical.get("schema_version") != "cvi.gallery_embedding_contract.v1":
        raise ValueError("unsupported gallery embedding contract schema")
    if canonical.get("dimension") != dimension:
        raise ValueError("embedding contract dimension differs from index")
    return canonical


def _contract_channels(
    contract: dict[str, Any], dimension: int
) -> tuple[EvidenceChannelSpec, ...]:
    raw_channels = contract.get("channels")
    fusion = contract.get("fusion")
    if not isinstance(raw_channels, list) or not raw_channels or not all(
        isinstance(value, dict) for value in raw_channels
    ):
        return (EvidenceChannelSpec("__opaque__", dimension, False, 1.0),)
    if not isinstance(fusion, dict) or fusion.get("type") not in {
        "weighted_concatenated_cosine", SCORER_ALGORITHM
    }:
        raise ValueError("gallery contract requires an exact fusion contract")
    weights = fusion.get("weights")
    if not isinstance(weights, list) or len(weights) != len(raw_channels):
        raise ValueError("gallery fusion weights must match channels")
    channels: list[EvidenceChannelSpec] = []
    for index, payload in enumerate(raw_channels):
        name = payload.get("name")
        channel_dim = payload.get("dimension")
        optional = payload.get("optional", False)
        weight = weights[index]
        if (
            not isinstance(name, str) or not name
            or not isinstance(channel_dim, int) or isinstance(channel_dim, bool)
            or channel_dim <= 0
            or not isinstance(optional, bool)
            or isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not np.isfinite(weight) or weight < 0.0
        ):
            raise ValueError("gallery channel contract is invalid")
        channels.append(EvidenceChannelSpec(name, channel_dim, optional, float(weight)))
    if len({channel.name for channel in channels}) != len(channels):
        raise ValueError("gallery channel names must be unique")
    if sum(channel.dimension for channel in channels) != dimension:
        raise ValueError("gallery channel dimensions differ from total dimension")
    if sum(channel.weight for channel in channels) <= 0.0:
        raise ValueError("gallery fusion weights must have a positive sum")
    if sum(channel.weight for channel in channels if not channel.optional) <= 0.0:
        raise ValueError("required gallery channels must retain positive fusion weight")
    return tuple(channels)


def _validate_unit_vector(vector: np.ndarray, dimension: int, name: str) -> None:
    if (
        not isinstance(vector, np.ndarray)
        or vector.dtype != np.float32
        or vector.shape != (dimension,)
        or not np.isfinite(vector).all()
        or not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-5)
    ):
        raise RuntimeError(f"gallery channel {name!r} vector is invalid")


def _canonical_metadata(metadata: dict | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError("metadata keys must be strings")
    if set(metadata) & _RESERVED_METADATA_KEYS:
        raise ValueError(
            "metadata keys template_id, content_sha256, idempotency_key, and "
            "template_schema are reserved"
        )
    values: list[tuple[object, int]] = [
        (value, 1) for value in metadata.values()
    ]
    nodes = 0
    while values:
        value, depth = values.pop()
        nodes += 1
        if depth > 32 or nodes > 10_000:
            raise ValueError("metadata exceeds JSON structural limits")
        if isinstance(value, dict):
            values.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            values.extend((child, depth + 1) for child in value)
        elif isinstance(value, str) and not _is_bounded_utf8_text(
            value, maximum_bytes=_MAXIMUM_METADATA_BYTES, allow_empty=True
        ):
            raise ValueError("metadata string values must be bounded UTF-8 text")
    try:
        encoded = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("metadata must be finite JSON") from exc
    if len(encoded) > _MAXIMUM_METADATA_BYTES:
        raise ValueError("metadata exceeds its JSON size limit")
    canonical = json.loads(encoded.decode("utf-8"))
    if not isinstance(canonical, dict):
        raise TypeError("metadata must be an object")
    return canonical


def _is_bounded_utf8_text(
    value: object,
    *,
    maximum_bytes: int,
    allow_empty: bool,
) -> bool:
    if not isinstance(value, str) or (not allow_empty and not value):
        return False
    try:
        return len(value.encode("utf-8", errors="strict")) <= maximum_bytes
    except UnicodeEncodeError:
        return False


def _template_id(content_sha256: str) -> str:
    return hashlib.sha256(
        f"{_TEMPLATE_ID_DOMAIN}\0{content_sha256}".encode("ascii")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _is_canonical_uuid5(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 5 and str(parsed) == value


def _is_cardinality(value: object, *, maximum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and (maximum is None or value <= maximum)
    )
