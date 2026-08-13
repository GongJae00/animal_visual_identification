"""Migrate an immutable v3 gallery into a separate v4 directory."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import secrets
import shutil
import stat
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

import faiss
import numpy as np

from identity_retrieval.gallery import IdentityGallery

_MANIFEST_SCHEMA = "cvi.gallery_manifest.v3"
_TEMPLATE_SCHEMA = "cvi.gallery_template.v1"
_MAXIMUM_MANIFEST_BYTES = 2 * 1024 * 1024
_MAXIMUM_SIDECAR_JSON_BYTES = 64 * 1024 * 1024
_MAXIMUM_GALLERY_TEMPLATES = 1_000_000
_MAXIMUM_DIMENSION = 1_000_000
_MAXIMUM_VECTOR_BYTES = 4 * 1024 * 1024 * 1024
_MAXIMUM_INDEX_OVERHEAD_BYTES = 16 * 1024 * 1024
_RENAME_NOREPLACE = 1
_RESERVED_METADATA_KEYS = {
    "template_id", "content_sha256", "idempotency_key", "template_schema"
}


def migrate_gallery(source: Path, output: Path) -> None:
    source = Path(os.path.abspath(source))
    output = Path(os.path.abspath(output))
    if source == output:
        raise ValueError("migration output must differ from the source gallery")
    if not output.name:
        raise ValueError("migration output must name a new directory")

    rename_no_replace = _rename_no_replace_function()
    with ExitStack() as stack:
        source_fd = _open_directory(source, "source gallery root")
        stack.callback(os.close, source_fd)
        parent_fd = _open_directory(output.parent, "destination parent root")
        stack.callback(os.close, parent_fd)
        _validate_destination_parent(parent_fd)
        if _directory_is_within(source_fd, parent_fd):
            raise ValueError("migration output must not be inside the source gallery")
        if _entry_exists(parent_fd, output.name):
            raise ValueError("migration output must be a new, non-existing directory")

        manifest_stream = stack.enter_context(
            _open_regular_file_at(
                source_fd,
                "gallery_manifest.json",
                "source gallery manifest",
                maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
            )
        )
        manifest = _read_strict_json_object(
            manifest_stream,
            "source gallery manifest",
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
        dimension, count, contract, files = _validate_manifest(manifest)
        vector_bytes = count * dimension * np.dtype(np.float32).itemsize
        if vector_bytes > _MAXIMUM_VECTOR_BYTES:
            raise ValueError("source v3 aggregate vector bytes exceed the migration limit")

        streams: dict[str, BinaryIO] = {}
        maximum_sizes = {
            "index": vector_bytes + _MAXIMUM_INDEX_OVERHEAD_BYTES,
            "metadata": _MAXIMUM_SIDECAR_JSON_BYTES,
            "breeds": _MAXIMUM_SIDECAR_JSON_BYTES,
        }
        for kind in ("index", "metadata", "breeds"):
            entry = files[kind]
            streams[kind] = stack.enter_context(
                _open_regular_file_at(
                    source_fd,
                    entry["name"],
                    f"source v3 {kind} file",
                    maximum_bytes=maximum_sizes[kind],
                )
            )

        metadata = _read_strict_json_object(
            streams["metadata"],
            "source v3 metadata sidecar",
            maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            expected_sha256=files["metadata"]["sha256"],
        )
        breeds = _read_strict_json_object(
            streams["breeds"],
            "source v3 breeds sidecar",
            maximum_bytes=_MAXIMUM_SIDECAR_JSON_BYTES,
            expected_sha256=files["breeds"]["sha256"],
        )
        _validate_sidecars(metadata, breeds, manifest, count)

        index_stream = streams["index"]
        if _sha256_stream(index_stream) != files["index"]["sha256"]:
            raise ValueError("source v3 index file is missing or corrupted")
        descriptor_path = Path("/proc/self/fd") / str(index_stream.fileno())
        if not descriptor_path.exists():  # pragma: no cover - Linux release target
            raise RuntimeError("secure FAISS migration reads require /proc/self/fd")
        old_index = faiss.read_index(str(descriptor_path))
        if not isinstance(old_index, faiss.IndexFlatIP):
            raise ValueError("source v3 index must be an exact inner-product index")
        if int(old_index.ntotal) != count or old_index.d != dimension:
            raise ValueError("source v3 index dimensions or count are inconsistent")
        target_contract, channel_layout = _migration_contract(contract, dimension)

        staging_name = _create_private_staging(parent_fd)
        staging_path = Path("/proc/self/fd") / str(parent_fd) / staging_name
        published = False
        try:
            _build_gallery(
                staging_path,
                old_index,
                metadata,
                breeds,
                count,
                dimension,
                target_contract,
                channel_layout,
            )
            validated = IdentityGallery(
                staging_path,
                dim=dimension,
                embedding_contract=target_contract,
                read_only=True,
            )
            try:
                if validated.size != count:
                    raise RuntimeError("migrated v4 gallery count is inconsistent")
            finally:
                validated.close()
            os.fsync(parent_fd)
            _rename_no_replace(
                rename_no_replace, parent_fd, staging_name, output.name
            )
            published = True
        finally:
            if not published:
                try:
                    shutil.rmtree(staging_path)
                except FileNotFoundError:
                    pass


def _build_gallery(
    staging_path: Path,
    old_index: faiss.Index,
    metadata: dict[str, Any],
    breeds: dict[str, Any],
    count: int,
    dimension: int,
    target_contract: dict[str, Any],
    channel_layout: tuple[tuple[str, int], ...],
) -> None:
    target = IdentityGallery(
        staging_path, dim=dimension, embedding_contract=target_contract
    )
    try:
        for index in range(count):
            stored = old_index.reconstruct(index).astype(np.float32, copy=False)
            vectors = _existing_channel_vectors(stored, channel_layout)
            row = metadata[str(index)]
            target.enroll_with_breed(
                vectors,
                row["registered_dog_id"],
                breeds[str(index)],
                row["metadata"],
                row["idempotency_key"],
                row["content_sha256"],
            )
        target.save()
    finally:
        target.close()


def _validate_manifest(
    manifest: dict[str, Any],
) -> tuple[int, int, dict[str, Any], dict[str, dict[str, str]]]:
    required = {
        "schema_version", "dimension", "embedding_contract", "count",
        "template_count", "identity_count", "identity_aggregation", "files",
    }
    if set(manifest) != required or manifest["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError("source gallery must be an exact cvi.gallery_manifest.v3")
    dimension = manifest["dimension"]
    if not _is_cardinality(dimension, minimum=1, maximum=_MAXIMUM_DIMENSION):
        raise ValueError("source v3 dimension is invalid")
    count = manifest["count"]
    template_count = manifest["template_count"]
    identity_count = manifest["identity_count"]
    if not _is_cardinality(count, maximum=_MAXIMUM_GALLERY_TEMPLATES):
        raise ValueError("source v3 count is invalid")
    if template_count != count:
        raise ValueError("source v3 manifest count is inconsistent")
    if (
        not _is_cardinality(identity_count, maximum=_MAXIMUM_GALLERY_TEMPLATES)
        or identity_count > count
    ):
        raise ValueError("source v3 identity count is invalid")
    if manifest["identity_aggregation"] != "max":
        raise ValueError("source v3 identity aggregation is invalid")
    contract = manifest["embedding_contract"]
    if not isinstance(contract, dict):
        raise ValueError("source v3 embedding contract is invalid")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {"index", "metadata", "breeds"}:
        raise ValueError("source v3 file manifest is invalid")
    validated_files: dict[str, dict[str, str]] = {}
    names: set[str] = set()
    for kind, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
            raise ValueError(f"source v3 {kind} entry is invalid")
        name = entry["name"]
        digest = entry["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError(f"source v3 {kind} filename is invalid")
        if name in names:
            raise ValueError("source v3 files must have distinct names")
        if not _is_sha256(digest):
            raise ValueError(f"source v3 {kind} digest is invalid")
        names.add(name)
        validated_files[kind] = {"name": name, "sha256": digest}
    return dimension, count, contract, validated_files


def _validate_sidecars(
    metadata: dict[str, Any],
    breeds: dict[str, Any],
    manifest: dict[str, Any],
    count: int,
) -> None:
    if len(metadata) != count or len(breeds) != count:
        raise ValueError("source v3 sidecar cardinality is inconsistent")
    expected_keys = {str(index) for index in range(count)}
    if set(metadata) != expected_keys or set(breeds) != expected_keys:
        raise ValueError("source v3 sidecars do not match the manifest")
    identities: set[str] = set()
    template_ids: set[str] = set()
    content_hashes: set[str] = set()
    idempotency_keys: set[str] = set()
    required_row_keys = {
        "registered_dog_id", "template_id", "content_sha256", "idempotency_key",
        "template_schema", "metadata",
    }
    for index in range(count):
        key = str(index)
        row = metadata[key]
        if not isinstance(row, dict) or set(row) != required_row_keys:
            raise ValueError("source v3 metadata row has an invalid schema")
        registered_id = row["registered_dog_id"]
        content_hash = row["content_sha256"]
        template_id = row["template_id"]
        idempotency_key = row["idempotency_key"]
        user_metadata = row["metadata"]
        if not isinstance(registered_id, str) or not registered_id:
            raise ValueError("source v3 metadata has an invalid registered identity")
        if not _is_sha256(content_hash):
            raise ValueError("source v3 metadata has an invalid content hash")
        expected_template_id = hashlib.sha256(
            f"{_TEMPLATE_SCHEMA}\0{content_hash}".encode("ascii")
        ).hexdigest()
        if (
            template_id != expected_template_id
            or row["template_schema"] != _TEMPLATE_SCHEMA
        ):
            raise ValueError("source v3 metadata has an invalid template identity")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("source v3 metadata has an invalid idempotency key")
        if not isinstance(user_metadata, dict):
            raise ValueError("source v3 user metadata must be an object")
        if set(user_metadata) & _RESERVED_METADATA_KEYS:
            raise ValueError("source v3 user metadata contains reserved fields")
        if not isinstance(breeds[key], str):
            raise ValueError("source v3 breed metadata must be a string")
        identities.add(registered_id)
        if template_id in template_ids or content_hash in content_hashes:
            raise ValueError("source v3 contains duplicate template content")
        if idempotency_key in idempotency_keys:
            raise ValueError("source v3 contains duplicate idempotency keys")
        template_ids.add(template_id)
        content_hashes.add(content_hash)
        idempotency_keys.add(idempotency_key)
    if len(identities) != manifest["identity_count"]:
        raise ValueError("source v3 manifest identity count is inconsistent")


def _migration_contract(
    contract: dict[str, Any], dimension: int
) -> tuple[dict[str, Any], tuple[tuple[str, int], ...]]:
    if contract.get("schema_version") != "cvi.gallery_embedding_contract.v1":
        raise ValueError("source v3 embedding contract schema is invalid")
    if contract.get("dimension") != dimension:
        raise ValueError("source v3 embedding contract dimension is inconsistent")
    if "channels" not in contract:
        if set(contract) != {"schema_version", "kind", "dimension"}:
            raise ValueError("source v3 opaque embedding contract has invalid keys")
        if contract["kind"] != "opaque":
            raise ValueError("source v3 opaque embedding contract is invalid")
        return dict(contract), (("__opaque__", dimension),)
    if set(contract) != {"schema_version", "dimension", "channels", "fusion"}:
        raise ValueError("source v3 channel embedding contract has invalid keys")
    channels = contract["channels"]
    if not isinstance(channels, list) or not channels or not all(
        isinstance(channel, dict) for channel in channels
    ):
        raise ValueError("source v3 channel contract is invalid")
    layout: list[tuple[str, int]] = []
    target_channels: list[dict[str, Any]] = []
    for channel in channels:
        name = channel.get("name")
        channel_dimension = channel.get("dimension")
        if (
            not isinstance(name, str) or not name
            or not _is_cardinality(
                channel_dimension, minimum=1, maximum=_MAXIMUM_DIMENSION
            )
        ):
            raise ValueError("source v3 channel contract is invalid")
        optional = channel.get("optional", False)
        if not isinstance(optional, bool):
            raise ValueError("source v3 channel optional flag is invalid")
        if optional:
            raise ValueError(
                "source v3 claims optional evidence but has no exact availability "
                "sidecar; migration cannot invent availability"
            )
        target_channels.append({**channel, "optional": False})
        layout.append((name, channel_dimension))
    if len({name for name, _ in layout}) != len(layout):
        raise ValueError("source v3 channel names must be unique")
    if sum(value for _, value in layout) != dimension:
        raise ValueError("source v3 channel dimensions are inconsistent")
    fusion = contract["fusion"]
    if not isinstance(fusion, dict) or set(fusion) != {
        "type", "weights", "embedding_scales"
    }:
        raise ValueError("source v3 fusion contract is invalid")
    if fusion["type"] != "weighted_concatenated_cosine":
        raise ValueError("source v3 fusion contract type is invalid")
    weights = fusion["weights"]
    scales = fusion["embedding_scales"]
    if not _valid_finite_numbers(weights, len(layout), allow_zero=True):
        raise ValueError("source v3 fusion weights are invalid")
    weight_sum = sum(float(weight) for weight in weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("source v3 fusion weights must have a positive sum")
    if not _valid_finite_numbers(scales, len(layout), allow_zero=False):
        raise ValueError(
            "source v3 has a zero or invalid channel scale; migration would "
            "require inventing an embedding"
        )
    target = {**contract, "channels": target_channels}
    return target, tuple(layout)


def _existing_channel_vectors(
    stored: np.ndarray, layout: tuple[tuple[str, int], ...]
) -> np.ndarray | dict[str, np.ndarray]:
    if stored.shape != (sum(dimension for _, dimension in layout),):
        raise ValueError("source v3 index returned an invalid vector shape")
    if not np.isfinite(stored).all():
        raise ValueError("source v3 index contains a non-finite vector")
    if layout == (("__opaque__", len(stored)),):
        norm = float(np.linalg.norm(stored))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError("source v3 opaque embedding is not a unit vector")
        return stored.copy()
    vectors: dict[str, np.ndarray] = {}
    offset = 0
    for name, dimension in layout:
        value = stored[offset:offset + dimension]
        offset += dimension
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError(
                f"source v3 channel {name!r} has no recoverable stored embedding"
            )
        vectors[name] = np.asarray(value / norm, dtype=np.float32)
    return vectors


def _open_directory(path: Path, label: str) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:  # pragma: no cover - Linux target
        raise RuntimeError("secure gallery migration requires Linux no-follow I/O")
    flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a non-symlink directory") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):  # pragma: no cover - O_DIRECTORY
        os.close(descriptor)
        raise ValueError(f"{label} must be a non-symlink directory")
    return descriptor


def _open_regular_file_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    maximum_bytes: int,
) -> BinaryIO:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - Linux target
        raise RuntimeError("secure gallery migration requires O_NOFOLLOW")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(
            f"{label} must be a readable non-symlink regular file"
        ) from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if file_stat.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its byte limit")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _read_strict_json_object(
    stream: BinaryIO,
    label: str,
    *,
    maximum_bytes: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    stream.seek(0)
    payload = stream.read(maximum_bytes + 1)
    stream.seek(0)
    if len(payload) > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError(f"{label} is missing or corrupted")
    if not payload:
        raise ValueError(f"{label} must not be empty")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not accepted: {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is not accepted: {value}")
    return parsed


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _create_private_staging(parent_fd: int) -> str:
    for _ in range(100):
        name = f".cvi-migrate-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            return name
        except FileExistsError:
            continue
    raise RuntimeError("unable to allocate private migration staging")


def _validate_destination_parent(parent_fd: int) -> None:
    parent_stat = os.fstat(parent_fd)
    mode = stat.S_IMODE(parent_stat.st_mode)
    if mode & 0o022 and not mode & stat.S_ISVTX:
        raise ValueError(
            "destination parent root must not be writable by other principals "
            "unless sticky"
        )


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _directory_is_within(ancestor_fd: int, descendant_fd: int) -> bool:
    ancestor = os.fstat(ancestor_fd)
    current_fd = os.dup(descendant_fd)
    try:
        while True:
            current = os.fstat(current_fd)
            if (current.st_dev, current.st_ino) == (ancestor.st_dev, ancestor.st_ino):
                return True
            parent_fd = os.open(
                "..",
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current_fd,
            )
            parent = os.fstat(parent_fd)
            if (parent.st_dev, parent.st_ino) == (current.st_dev, current.st_ino):
                os.close(parent_fd)
                return False
            os.close(current_fd)
            current_fd = parent_fd
    finally:
        os.close(current_fd)


def _rename_no_replace_function():
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:  # pragma: no cover - supported release target is Linux/glibc
        raise RuntimeError("atomic no-replace publication requires renameat2")
    function.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    return function


def _rename_no_replace(function, parent_fd: int, source: str, output: str) -> None:
    result = function(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(output),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError("migration output must be a new, non-existing directory")
    raise OSError(error, os.strerror(error), output)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_cardinality(
    value: object,
    *,
    minimum: int = 0,
    maximum: int,
) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def _valid_finite_numbers(
    values: object, expected_length: int, *, allow_zero: bool
) -> bool:
    def valid(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            finite_value = float(value)
        except OverflowError:
            return False
        return math.isfinite(finite_value) and (
            finite_value >= 0.0 if allow_zero else finite_value > 0.0
        )

    return (
        isinstance(values, list)
        and len(values) == expected_length
        and all(valid(value) for value in values)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    migrate_gallery(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
