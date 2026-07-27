from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import faiss
import numpy as np

from cvi.index.base import AbstractIdentityIndex


_MANIFEST_SCHEMA = "cvi.gallery_manifest.v4"
_TEMPLATE_SCHEMA = "cvi.gallery_template.v1"
_IDENTITY_AGGREGATION = "max"
_SCORER_ALGORITHM = "exact_available_intersection_weighted_cosine.v1"
_MAXIMUM_MANIFEST_BYTES = 2 * 1024 * 1024
_MAXIMUM_SIDECAR_JSON_BYTES = 64 * 1024 * 1024
_MAXIMUM_METADATA_BYTES = 64 * 1024
_MAXIMUM_IDEMPOTENCY_KEY_BYTES = _MAXIMUM_METADATA_BYTES
_MAXIMUM_GALLERY_TEMPLATES = 1_000_000
_MAXIMUM_BINARY_FILE_BYTES = 64 * 1024 * 1024 * 1024
_BINARY_FORMAT_OVERHEAD_BYTES = 1024 * 1024
_OPTIONAL_CHANNEL_OVERHEAD_BYTES = 64 * 1024
_RESERVED_METADATA_KEYS = {
    "template_id", "content_sha256", "idempotency_key", "template_schema"
}


@dataclass(frozen=True, slots=True)
class _Channel:
    name: str
    dimension: int
    optional: bool
    weight: float


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
    optional_channels: tuple[_Channel, ...],
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
    channel: _Channel,
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
    optional_channels: tuple[_Channel, ...],
) -> None:
    expected: dict[str, tuple[_Channel, str]] = {}
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
        raise RuntimeError(f"{label} must be a JSON object")
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


class SpeciesFilteredIndex(AbstractIdentityIndex):
    """Generation-published gallery with exact sparse-channel scoring."""

    def __init__(
        self,
        base_index_dir: Path,
        breed_mapping: dict[str, list[str]] | None = None,
        dim: int = 640,
        embedding_contract: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("gallery dimension must be a positive integer")
        self._dim = dim
        self._embedding_contract = _canonical_embedding_contract(
            embedding_contract, dim
        )
        self._channels = _contract_channels(self._embedding_contract, dim)
        self._required_channels = tuple(
            channel for channel in self._channels if not channel.optional
        )
        self._optional_channels = tuple(
            channel for channel in self._channels if channel.optional
        )
        if not self._required_channels:
            raise ValueError("gallery contract requires at least one required channel")
        self._required_dim = sum(channel.dimension for channel in self._required_channels)
        self._scorer_hash = _scorer_hash(self._channels)
        self._base_dir = base_index_dir
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
        self._breed_mapping = breed_mapping or {}
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
            legacy_paths = (
                self._base_dir / "master.idx",
                self._base_dir / "metadata.json",
                self._base_dir / "breed_index.json",
            )
            if _path_entry_exists(manifest_path):
                self._load_snapshot(manifest_path)
            elif any(_path_entry_exists(path) for path in legacy_paths):
                raise RuntimeError(
                    "unversioned gallery files are not accepted; rebuild the gallery"
                )
            else:
                if read_only:
                    raise RuntimeError("read-only gallery does not exist")
                self._index = faiss.IndexFlatIP(self._required_dim)
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
        required = {
            "schema_version", "dimension", "required_dimension",
            "embedding_contract", "count", "template_count", "identity_count",
            "identity_aggregation", "scorer", "files",
        }
        if set(manifest) != required or manifest["schema_version"] != _MANIFEST_SCHEMA:
            raise RuntimeError(
                "gallery is not manifest v4; migrate v3 into a new output directory"
            )
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
            "algorithm": _SCORER_ALGORITHM,
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
        expected_keys = {str(index) for index in range(template_count)}
        if (
            set(metadata) != expected_keys
            or set(breeds) != expected_keys
            or set(availability) != expected_keys
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
        self._metadata = {int(key): value for key, value in metadata.items()}
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
        expected_keys = set(range(count))
        if (
            set(self._metadata) != expected_keys
            or set(self._breed_index) != expected_keys
            or set(self._availability) != expected_keys
        ):
            raise RuntimeError("gallery index and sidecar cardinality are inconsistent")
        channel_names = {channel.name for channel in self._channels}
        required_names = {channel.name for channel in self._required_channels}
        ids: list[str] = []
        template_ids: list[str] = []
        content_hashes: list[str] = []
        idempotency_keys: list[str] = []
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
                raise RuntimeError("gallery identity metadata must be an object")
            try:
                _canonical_metadata(meta["metadata"])
            except ValueError as exc:
                raise RuntimeError("gallery identity metadata is invalid") from exc
            template_id = meta["template_id"]
            content_sha256 = meta["content_sha256"]
            idempotency_key = meta["idempotency_key"]
            if meta["template_schema"] != _TEMPLATE_SCHEMA:
                raise RuntimeError("gallery template metadata has an invalid schema")
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
                raise RuntimeError("gallery breed metadata must be a string")
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
            ids.append(registered_id)
            template_ids.append(template_id)
            content_hashes.append(content_sha256)
            idempotency_keys.append(idempotency_key)
        for channel in self._optional_channels:
            expected_rows = {
                idx for idx in expected_keys if self._availability[idx][channel.name]
            }
            vectors = self._optional_vectors[channel.name]
            if set(vectors) != expected_rows:
                raise RuntimeError(
                    f"gallery optional channel {channel.name!r} disagrees with availability"
                )
            for vector in vectors.values():
                _validate_unit_vector(vector, channel.dimension, channel.name)
        if len(template_ids) != len(set(template_ids)):
            raise RuntimeError("gallery contains duplicate template IDs")
        if len(content_hashes) != len(set(content_hashes)):
            raise RuntimeError("gallery contains duplicate template content")
        if len(idempotency_keys) != len(set(idempotency_keys)):
            raise RuntimeError("gallery contains duplicate idempotency keys")
        identity_count = len(set(ids))
        if expected_identity_count is not None and expected_identity_count != identity_count:
            raise RuntimeError("gallery manifest identity count is inconsistent")

    def enroll(
        self,
        embedding: np.ndarray | dict[str, np.ndarray],
        registered_dog_id: str,
        metadata: dict | None = None,
        idempotency_key: str | None = None,
        content_sha256: str | None = None,
    ) -> int:
        return self.enroll_with_breed(
            embedding, registered_dog_id, "unknown", metadata, idempotency_key,
            content_sha256,
        )

    def enroll_with_breed(
        self,
        embedding: np.ndarray | dict[str, np.ndarray],
        dog_id: str,
        breed: str,
        metadata: dict | None = None,
        idempotency_key: str | None = None,
        content_sha256: str | None = None,
    ) -> int:
        if not _is_canonical_uuid5(dog_id):
            raise ValueError(
                "registered_dog_id must be a canonical lowercase UUIDv5 string"
            )
        self._ensure_writer()
        if int(self._index.ntotal) >= _MAXIMUM_GALLERY_TEMPLATES:
            raise RuntimeError("gallery template cardinality limit has been reached")
        if not isinstance(breed, str):
            raise ValueError("breed must be a string")
        canonical_metadata = _canonical_metadata(metadata)
        vectors = self._canonical_vectors(embedding)
        if content_sha256 is None:
            digest = hashlib.sha256()
            for channel in self._channels:
                digest.update(channel.name.encode("utf-8"))
                digest.update(b"\0")
                if channel.name in vectors:
                    digest.update(vectors[channel.name].astype("<f4", copy=False).tobytes())
            content_sha256 = digest.hexdigest()
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

        availability = {
            channel.name: channel.name in vectors for channel in self._channels
        }
        for existing_idx, existing in self._metadata.items():
            same_template = (
                existing["template_id"] == template_id
                and existing["content_sha256"] == content_sha256
            )
            if same_template and existing["registered_dog_id"] != dog_id:
                raise ValueError(
                    f"template/content {template_id!r} is already bound to different "
                    f"registered identity {existing['registered_dog_id']!r}"
                )
            if existing["idempotency_key"] == idempotency_key:
                exact_retry = (
                    existing["registered_dog_id"] == dog_id
                    and same_template
                    and existing["metadata"] == canonical_metadata
                    and self._breed_index[existing_idx] == breed
                    and self._availability[existing_idx] == availability
                    and self._vectors_equal(existing_idx, vectors)
                )
                if exact_retry:
                    return int(existing_idx)
                raise ValueError(
                    f"idempotency key {idempotency_key!r} conflicts with an existing enrollment"
                )
            if same_template:
                raise ValueError(
                    f"template/content {template_id!r} is already enrolled with different "
                    "immutable evidence or metadata"
                )
            if existing["content_sha256"] == content_sha256:
                raise ValueError(
                    "template content is already bound to registered identity "
                    f"{existing['registered_dog_id']!r}"
                )

        idx = int(self._index.ntotal)
        required = np.concatenate(
            [vectors[channel.name] for channel in self._required_channels]
        ).astype(np.float32, copy=False)
        self._index.add(required.reshape(1, -1))
        for channel in self._optional_channels:
            if channel.name in vectors:
                self._optional_vectors[channel.name][idx] = vectors[channel.name]
        self._availability[idx] = availability
        self._metadata[idx] = {
            "registered_dog_id": dog_id,
            "template_id": template_id,
            "content_sha256": content_sha256,
            "idempotency_key": idempotency_key,
            "template_schema": _TEMPLATE_SCHEMA,
            "metadata": canonical_metadata,
        }
        self._breed_index[idx] = breed
        return idx

    def _canonical_vectors(
        self, embedding: np.ndarray | dict[str, np.ndarray]
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
            vectors[name] = np.asarray(vector / norm, dtype=np.float32)
        return vectors

    def _vectors_equal(self, index: int, vectors: dict[str, np.ndarray]) -> bool:
        existing = self._template_vectors(index)
        return set(existing) == set(vectors) and all(
            np.array_equal(existing[name], vectors[name]) for name in existing
        )

    def _template_vectors(self, index: int) -> dict[str, np.ndarray]:
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
        return vectors

    def search_filtered(
        self,
        query: np.ndarray | dict[str, np.ndarray],
        allowed_breeds: list[str] | None = None,
        top_k: int = 5,
    ) -> list[tuple[int, float, dict]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self._index.ntotal == 0:
            return []
        vectors = self._canonical_vectors(query)
        breed_set = set(allowed_breeds or [])
        best_by_identity: dict[str, tuple[int, float, dict]] = {}
        for idx in range(int(self._index.ntotal)):
            if breed_set and self._breed_index[idx] not in breed_set:
                continue
            candidate = self._score_template(vectors, idx)
            meta = self._metadata[idx]
            registered_id = meta["registered_dog_id"]
            row = (idx, candidate[0], candidate[1])
            current = best_by_identity.get(registered_id)
            if current is None or row[1] > current[1] or (
                row[1] == current[1]
                and row[2]["template_id"] < current[2]["template_id"]
            ):
                best_by_identity[registered_id] = row
        aggregated = sorted(
            best_by_identity.values(),
            key=lambda item: (-item[1], item[2]["registered_dog_id"]),
        )[:top_k]
        return [(idx, score, deepcopy(meta)) for idx, score, meta in aggregated]

    def search(
        self, query: np.ndarray | dict[str, np.ndarray], top_k: int = 5
    ) -> list[tuple[int, float, dict]]:
        return self.search_filtered(query, None, top_k)

    def explain_identity(
        self,
        query: np.ndarray | dict[str, np.ndarray],
        registered_dog_id: str,
    ) -> tuple[int, float, dict] | None:
        vectors = self._canonical_vectors(query)
        best: tuple[int, float, dict] | None = None
        for idx in range(int(self._index.ntotal)):
            if self._metadata[idx]["registered_dog_id"] != registered_dog_id:
                continue
            score, meta = self._score_template(vectors, idx)
            row = (idx, score, meta)
            if best is None or score > best[1] or (
                score == best[1]
                and meta["template_id"] < best[2]["template_id"]
            ):
                best = row
        return None if best is None else (best[0], best[1], deepcopy(best[2]))

    def _score_template(
        self, query: dict[str, np.ndarray], index: int
    ) -> tuple[float, dict[str, Any]]:
        template = self._template_vectors(index)
        intersection = [
            channel for channel in self._channels
            if channel.name in query and channel.name in template
        ]
        total_weight = sum(channel.weight for channel in intersection)
        if total_weight <= 0.0:
            raise RuntimeError("no positively weighted evidence intersection remains")
        evidence = {
            channel.name: float(np.dot(query[channel.name], template[channel.name]))
            for channel in intersection
        }
        score = sum(
            channel.weight * evidence[channel.name] for channel in intersection
        ) / total_weight
        meta = deepcopy(self._metadata[index])
        meta["_evidence"] = evidence
        meta["_evidence_availability"] = {
            channel.name: channel.name in query and channel.name in template
            for channel in self._channels
        }
        meta["_query_availability"] = {
            channel.name: channel.name in query for channel in self._channels
        }
        meta["_template_availability"] = dict(self._availability[index])
        meta["_scorer_hash"] = self._scorer_hash
        meta["_exact"] = True
        return float(score), meta

    def search_with_evidence(
        self,
        query: np.ndarray | dict[str, np.ndarray],
        top_k: int = 5,
        slices: list[tuple[int, int, str]] | None = None,
    ) -> list[dict[str, Any]]:
        if slices is not None:
            raise ValueError("v4 evidence slices come from the gallery contract")
        return [
            {
                "index": idx,
                "registered_dog_id": meta["registered_dog_id"],
                "template_id": meta["template_id"],
                "similarity": score,
                "evidence": dict(meta["_evidence"]),
                "evidence_availability": dict(meta["_evidence_availability"]),
                "scorer_hash": meta["_scorer_hash"],
                "exact": meta["_exact"],
                "metadata": _result_metadata(meta),
            }
            for idx, score, meta in self.search(query, top_k)
        ]

    def remove(self, index: int) -> None:
        raise NotImplementedError("remove not supported for SpeciesFilteredIndex")

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
        except Exception:
            pass

    def _ensure_writer(self) -> None:
        if self._read_only or self._lock_stream is None:
            raise RuntimeError("gallery was opened read-only")

    def save(self) -> None:
        self._ensure_writer()
        self._validate_state()
        template_count = int(self._index.ntotal)
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
                self._metadata,
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
                "schema_version": _MANIFEST_SCHEMA,
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
                    "algorithm": _SCORER_ALGORITHM,
                    "hash": self._scorer_hash,
                    "exact": True,
                },
                "files": {
                    kind: {"name": path.name, "sha256": hashes[kind]}
                    for kind, path in destinations.items()
                },
            }
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


def _contract_channels(contract: dict[str, Any], dimension: int) -> tuple[_Channel, ...]:
    raw_channels = contract.get("channels")
    fusion = contract.get("fusion")
    if not isinstance(raw_channels, list) or not raw_channels or not all(
        isinstance(value, dict) for value in raw_channels
    ):
        return (_Channel("__opaque__", dimension, False, 1.0),)
    if not isinstance(fusion, dict) or fusion.get("type") not in {
        "weighted_concatenated_cosine", _SCORER_ALGORITHM
    }:
        raise ValueError("gallery contract requires an exact fusion contract")
    weights = fusion.get("weights")
    if not isinstance(weights, list) or len(weights) != len(raw_channels):
        raise ValueError("gallery fusion weights must match channels")
    channels: list[_Channel] = []
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
        channels.append(_Channel(name, channel_dim, optional, float(weight)))
    if len({channel.name for channel in channels}) != len(channels):
        raise ValueError("gallery channel names must be unique")
    if sum(channel.dimension for channel in channels) != dimension:
        raise ValueError("gallery channel dimensions differ from total dimension")
    if sum(channel.weight for channel in channels) <= 0.0:
        raise ValueError("gallery fusion weights must have a positive sum")
    if sum(channel.weight for channel in channels if not channel.optional) <= 0.0:
        raise ValueError("required gallery channels must retain positive fusion weight")
    return tuple(channels)


def _scorer_hash(channels: tuple[_Channel, ...]) -> str:
    total = sum(channel.weight for channel in channels)
    payload = {
        "algorithm": _SCORER_ALGORITHM,
        "channels": [
            {
                "name": channel.name,
                "dimension": channel.dimension,
                "optional": channel.optional,
            }
            for channel in channels
        ],
        "weights": [channel.weight / total for channel in channels],
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


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
        raise ValueError("metadata must be an object")
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
        raise ValueError("metadata must be an object")
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
        f"{_TEMPLATE_SCHEMA}\0{content_sha256}".encode("ascii")
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


def _result_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta["metadata"],
        "template_id": meta["template_id"],
        "content_sha256": meta["content_sha256"],
        "idempotency_key": meta["idempotency_key"],
        "template_schema": meta["template_schema"],
    }
