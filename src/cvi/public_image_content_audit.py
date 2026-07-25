"""Bounded, label-blind image-content audit for public canine manifests.

The audit reopens the source ZIP independently, rebinds it to the semantic
manifest, validates every image decode, and hashes a canonical RGB raster.
Identity, split, camera, cage, and sequence metadata are deliberately absent
from the pixel-hash input.  Exact-pixel duplicate evidence produced here is
not a perceptual duplicate decision, a split decision, or model admission.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import stat
import struct
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvi.provenance import content_sha256
from cvi.public_canine_manifest import PublicCanineManifest, PublicCanineRecord


_PIXEL_HASH_DOMAIN = b"CVI_PIXEL_CANONICAL_RGB_V1\0"
_ALLOWED_ZIP_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)


@dataclass(frozen=True, slots=True)
class ImageContentAuditPolicy:
    """Hard memory, decompression, geometry, and aggregate-work limits."""

    maximum_archive_bytes: int = 2_147_483_648
    maximum_records: int = 100_000
    maximum_member_encoded_bytes: int = 67_108_864
    maximum_member_compressed_bytes: int = 67_108_864
    maximum_total_encoded_bytes: int = 34_359_738_368
    maximum_container_bytes: int = 2_147_483_648
    minimum_temporary_free_bytes_after_stage: int = 1_073_741_824
    maximum_container_compression_ratio: float = 4.0
    maximum_member_compression_ratio: float = 200.0
    maximum_width: int = 32_768
    maximum_height: int = 32_768
    maximum_image_pixels: int = 33_554_432
    maximum_total_pixels: int = 250_000_000_000
    maximum_hash_chunk_bytes: int = 4_194_304
    zip_read_chunk_bytes: int = 1_048_576
    allowed_decoder_formats: tuple[str, ...] = ("JPEG", "PNG")
    allowed_source_modes: tuple[str, ...] = ("CMYK", "L", "RGB", "RGBA")
    allowed_zip_compression: tuple[int, ...] = _ALLOWED_ZIP_COMPRESSION
    canonical_mode: str = "RGB"
    apply_exif_orientation: bool = True
    schema_version: str = "cvi.image_content_audit_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.image_content_audit_policy.v1":
            raise ValueError("unsupported image-content audit policy")
        integer_fields = (
            "maximum_archive_bytes",
            "maximum_records",
            "maximum_member_encoded_bytes",
            "maximum_member_compressed_bytes",
            "maximum_total_encoded_bytes",
            "maximum_container_bytes",
            "minimum_temporary_free_bytes_after_stage",
            "maximum_width",
            "maximum_height",
            "maximum_image_pixels",
            "maximum_total_pixels",
            "maximum_hash_chunk_bytes",
            "zip_read_chunk_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "maximum_container_compression_ratio",
            "maximum_member_compression_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 1.0 <= float(value) <= 1_000_000.0:
                raise ValueError(f"{name} is outside the supported range")
        _require_sorted_unique_tokens(
            self.allowed_decoder_formats, "allowed_decoder_formats"
        )
        _require_sorted_unique_tokens(
            self.allowed_source_modes, "allowed_source_modes"
        )
        if any(value != value.upper() for value in self.allowed_decoder_formats):
            raise ValueError("allowed decoder formats must be uppercase")
        if (
            not isinstance(self.allowed_zip_compression, tuple)
            or not self.allowed_zip_compression
            or tuple(sorted(set(self.allowed_zip_compression)))
            != self.allowed_zip_compression
            or any(value not in _ALLOWED_ZIP_COMPRESSION for value in self.allowed_zip_compression)
        ):
            raise ValueError("allowed ZIP compression methods differ")
        if self.canonical_mode != "RGB":
            raise ValueError("canonical image mode is fixed to RGB")
        if not isinstance(self.apply_exif_orientation, bool):
            raise TypeError("apply_exif_orientation must be boolean")
        if self.maximum_hash_chunk_bytes < self.maximum_width * 3:
            raise ValueError("hash chunk cap cannot hold one maximum-width RGB row")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "maximum_archive_bytes": self.maximum_archive_bytes,
            "maximum_records": self.maximum_records,
            "maximum_member_encoded_bytes": self.maximum_member_encoded_bytes,
            "maximum_member_compressed_bytes": self.maximum_member_compressed_bytes,
            "maximum_total_encoded_bytes": self.maximum_total_encoded_bytes,
            "maximum_container_bytes": self.maximum_container_bytes,
            "minimum_temporary_free_bytes_after_stage": (
                self.minimum_temporary_free_bytes_after_stage
            ),
            "maximum_container_compression_ratio": float(
                self.maximum_container_compression_ratio
            ),
            "maximum_member_compression_ratio": float(
                self.maximum_member_compression_ratio
            ),
            "maximum_width": self.maximum_width,
            "maximum_height": self.maximum_height,
            "maximum_image_pixels": self.maximum_image_pixels,
            "maximum_total_pixels": self.maximum_total_pixels,
            "maximum_hash_chunk_bytes": self.maximum_hash_chunk_bytes,
            "zip_read_chunk_bytes": self.zip_read_chunk_bytes,
            "allowed_decoder_formats": list(self.allowed_decoder_formats),
            "allowed_source_modes": list(self.allowed_source_modes),
            "allowed_zip_compression": list(self.allowed_zip_compression),
            "canonical_mode": self.canonical_mode,
            "apply_exif_orientation": self.apply_exif_orientation,
        }


@dataclass(frozen=True, slots=True)
class ImageContentDigestRecord:
    """Content evidence associated with a sample, without an identity label."""

    dataset_name: str
    dataset_version: str
    source_variant: str
    source_sample_id: str
    paired_source_sample_id: str | None
    member_path: str
    container_member_path: str | None
    source_archive_sha256: str
    encoded_sha256: str
    pixel_sha256: str
    encoded_bytes: int
    decoded_format: str
    source_mode: str
    canonical_mode: str
    canonical_width: int
    canonical_height: int
    canonical_bytes: int
    exif_orientation_applied: bool
    schema_version: str = "cvi.image_content_digest_record.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.image_content_digest_record.v1":
            raise ValueError("unsupported image-content digest record")
        for name in (
            "dataset_name",
            "dataset_version",
            "source_variant",
            "source_sample_id",
            "member_path",
            "decoded_format",
            "source_mode",
        ):
            _require_token(getattr(self, name), name, maximum=4096)
        if self.paired_source_sample_id is not None:
            _require_token(
                self.paired_source_sample_id,
                "paired_source_sample_id",
                maximum=4096,
            )
        if self.container_member_path is not None:
            _require_token(
                self.container_member_path,
                "container_member_path",
                maximum=4096,
            )
        for name in ("source_archive_sha256", "encoded_sha256", "pixel_sha256"):
            _require_sha256(getattr(self, name), name)
        for name in (
            "encoded_bytes",
            "canonical_width",
            "canonical_height",
            "canonical_bytes",
        ):
            _require_positive_int(getattr(self, name), name)
        if self.canonical_mode != "RGB":
            raise ValueError("digest canonical mode differs")
        if self.canonical_bytes != self.canonical_width * self.canonical_height * 3:
            raise ValueError("canonical byte count differs from RGB geometry")
        if not isinstance(self.exif_orientation_applied, bool):
            raise TypeError("EXIF orientation flag must be boolean")

    @property
    def pixel_hash_input_fields(self) -> tuple[str, ...]:
        """Document the label-blind domain of ``pixel_sha256``."""

        return (
            "canonical_width",
            "canonical_height",
            "canonical_mode",
            "canonical_pixels",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class PixelExactDuplicateGroup:
    pixel_sha256: str
    source_sample_ids: tuple[str, ...]
    schema_version: str = "cvi.pixel_exact_duplicate_group.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pixel_exact_duplicate_group.v1":
            raise ValueError("unsupported exact-pixel duplicate group")
        _require_sha256(self.pixel_sha256, "pixel_sha256")
        if (
            not isinstance(self.source_sample_ids, tuple)
            or len(self.source_sample_ids) < 2
            or tuple(sorted(set(self.source_sample_ids))) != self.source_sample_ids
        ):
            raise ValueError("duplicate group sample IDs must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pixel_sha256": self.pixel_sha256,
            "source_sample_ids": list(self.source_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class ImageContentAuditReceipt:
    dataset_name: str
    dataset_version: str
    source_archive_sha256: str
    source_archive_receipt_sha256: str
    semantic_manifest_sha256: str
    policy_sha256: str
    decoder_name: str
    decoder_version: str
    records: tuple[ImageContentDigestRecord, ...]
    exact_duplicate_groups: tuple[PixelExactDuplicateGroup, ...]
    source_variant_counts: tuple[tuple[str, int], ...]
    paired_record_count: int
    total_encoded_bytes: int
    total_canonical_pixels: int
    unique_pixel_digest_count: int
    decision: str = "PASS_IMAGE_CONTENT_AUDIT"
    interpretation: str = (
        "DECODE_AND_PIXEL_EXACT_DUPLICATE_EVIDENCE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION"
    )
    schema_version: str = "cvi.image_content_audit_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.image_content_audit_receipt.v1":
            raise ValueError("unsupported image-content audit receipt")
        for name in (
            "dataset_name",
            "dataset_version",
            "decoder_name",
            "decoder_version",
        ):
            _require_token(getattr(self, name), name)
        for name in (
            "source_archive_sha256",
            "source_archive_receipt_sha256",
            "semantic_manifest_sha256",
            "policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if tuple(sorted(self.records, key=lambda item: item.source_sample_id)) != self.records:
            raise ValueError("digest records must be sorted by source sample ID")
        if len({item.source_sample_id for item in self.records}) != len(self.records):
            raise ValueError("digest records contain duplicate sample IDs")
        expected_groups = _build_duplicate_groups(self.records)
        if expected_groups != self.exact_duplicate_groups:
            raise ValueError("exact duplicate groups differ from digest records")
        expected_variants = tuple(sorted(Counter(
            item.source_variant for item in self.records
        ).items()))
        if self.source_variant_counts != expected_variants:
            raise ValueError("source variant counts differ")
        if self.paired_record_count != sum(
            item.paired_source_sample_id is not None for item in self.records
        ):
            raise ValueError("paired record count differs")
        if self.total_encoded_bytes != sum(item.encoded_bytes for item in self.records):
            raise ValueError("total encoded bytes differ")
        if self.total_canonical_pixels != sum(
            item.canonical_width * item.canonical_height for item in self.records
        ):
            raise ValueError("total canonical pixels differ")
        if self.unique_pixel_digest_count != len(
            {item.pixel_sha256 for item in self.records}
        ):
            raise ValueError("unique pixel digest count differs")
        if self.decision != "PASS_IMAGE_CONTENT_AUDIT":
            raise ValueError("image-content audit decision differs")
        if self.interpretation != (
            "DECODE_AND_PIXEL_EXACT_DUPLICATE_EVIDENCE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION"
        ):
            raise ValueError("image-content audit interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "source_archive_sha256": self.source_archive_sha256,
            "source_archive_receipt_sha256": self.source_archive_receipt_sha256,
            "semantic_manifest_sha256": self.semantic_manifest_sha256,
            "policy_sha256": self.policy_sha256,
            "decoder_name": self.decoder_name,
            "decoder_version": self.decoder_version,
            "records": [item.to_dict() for item in self.records],
            "exact_duplicate_groups": [
                item.to_dict() for item in self.exact_duplicate_groups
            ],
            "source_variant_counts": [list(item) for item in self.source_variant_counts],
            "paired_record_count": self.paired_record_count,
            "total_encoded_bytes": self.total_encoded_bytes,
            "total_canonical_pixels": self.total_canonical_pixels,
            "unique_pixel_digest_count": self.unique_pixel_digest_count,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }


def audit_public_canine_image_content(
    *,
    archive_path: Path,
    manifest: PublicCanineManifest,
    policy: ImageContentAuditPolicy | None = None,
) -> ImageContentAuditReceipt:
    """Decode and pixel-hash one source-bound public canine manifest."""

    if not isinstance(manifest, PublicCanineManifest):
        raise TypeError("manifest must be PublicCanineManifest")
    if policy is None:
        policy = ImageContentAuditPolicy()
    if not isinstance(policy, ImageContentAuditPolicy):
        raise TypeError("policy must be ImageContentAuditPolicy")
    ordered_records = tuple(sorted(manifest.records, key=lambda item: item.source_sample_id))
    if len(ordered_records) > policy.maximum_records:
        raise ValueError("manifest exceeds maximum record count")
    declared_bytes = sum(item.member_uncompressed_bytes for item in ordered_records)
    if declared_bytes > policy.maximum_total_encoded_bytes:
        raise ValueError("manifest exceeds maximum total encoded bytes")
    if any(
        item.member_uncompressed_bytes <= 0
        or item.member_uncompressed_bytes > policy.maximum_member_encoded_bytes
        for item in ordered_records
    ):
        raise ValueError("manifest member exceeds encoded-byte limit")

    PIL, Image, ImageFile, ImageOps, UnidentifiedImageError = _pillow()
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("Pillow truncated-image decoding must be disabled")
    pillow_pixel_limit = Image.MAX_IMAGE_PIXELS
    if (
        pillow_pixel_limit is not None
        and pillow_pixel_limit < policy.maximum_image_pixels
    ):
        raise RuntimeError(
            "Pillow pixel-bomb limit is lower than the explicit audit limit"
        )

    descriptor, initial_stat = _open_bound_archive(
        archive_path, policy.maximum_archive_bytes
    )
    digest_records: list[ImageContentDigestRecord] = []
    total_pixels = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as archive_stream:
            archive_sha256 = _sha256_stream(archive_stream, policy.zip_read_chunk_bytes)
            if archive_sha256 != manifest.source_archive_sha256:
                raise ValueError("archive bytes differ from semantic manifest binding")
            archive_stream.seek(0)
            try:
                with zipfile.ZipFile(archive_stream) as outer:
                    outer_index = _unique_info_index(outer)
                    direct = tuple(
                        item for item in ordered_records
                        if item.container_member_path is None
                    )
                    for record in direct:
                        result = _audit_member(
                            outer,
                            _bound_member_info(outer_index, record, policy),
                            record,
                            policy,
                            Image,
                            ImageOps,
                            UnidentifiedImageError,
                        )
                        total_pixels = _add_pixels(total_pixels, result, policy)
                        digest_records.append(result)

                    nested_groups: dict[str, list[PublicCanineRecord]] = defaultdict(list)
                    for record in ordered_records:
                        if record.container_member_path is not None:
                            nested_groups[record.container_member_path].append(record)
                    for container_path in sorted(nested_groups):
                        records = tuple(nested_groups[container_path])
                        container_info = _bound_container_info(
                            outer_index, records, policy
                        )
                        with _stage_nested_container(
                            outer, container_info, policy
                        ) as nested_stream:
                            with zipfile.ZipFile(nested_stream) as nested:
                                nested_index = _unique_info_index(nested)
                                for record in records:
                                    result = _audit_member(
                                        nested,
                                        _bound_member_info(
                                            nested_index, record, policy
                                        ),
                                        record,
                                        policy,
                                        Image,
                                        ImageOps,
                                        UnidentifiedImageError,
                                    )
                                    total_pixels = _add_pixels(
                                        total_pixels, result, policy
                                    )
                                    digest_records.append(result)
            except (zipfile.BadZipFile, RuntimeError) as error:
                raise ValueError("source ZIP decode failed") from error
            _verify_archive_stability(
                archive_path, archive_stream.fileno(), initial_stat
            )
    except Exception:
        # os.fdopen owns and closes the descriptor after entry.  If entry itself
        # failed, close the still-owned raw descriptor without masking the cause.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise

    records = tuple(sorted(digest_records, key=lambda item: item.source_sample_id))
    groups = _build_duplicate_groups(records)
    return ImageContentAuditReceipt(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        source_archive_sha256=manifest.source_archive_sha256,
        source_archive_receipt_sha256=manifest.source_archive_receipt_sha256,
        semantic_manifest_sha256=_semantic_manifest_sha256(manifest),
        policy_sha256=policy.policy_sha256,
        decoder_name="Pillow",
        decoder_version=PIL.__version__,
        records=records,
        exact_duplicate_groups=groups,
        source_variant_counts=tuple(sorted(Counter(
            item.source_variant for item in records
        ).items())),
        paired_record_count=sum(
            item.paired_source_sample_id is not None for item in records
        ),
        total_encoded_bytes=sum(item.encoded_bytes for item in records),
        total_canonical_pixels=total_pixels,
        unique_pixel_digest_count=len({item.pixel_sha256 for item in records}),
    )


def _audit_member(
    bundle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    record: PublicCanineRecord,
    policy: ImageContentAuditPolicy,
    Image: Any,
    ImageOps: Any,
    UnidentifiedImageError: type[Exception],
) -> ImageContentDigestRecord:
    encoded, encoded_sha256, encoded_bytes = _read_member_bounded(
        bundle, info, policy
    )
    with encoded:
        decoded = _decode_canonical_pixels(
            encoded,
            policy,
            Image,
            ImageOps,
            UnidentifiedImageError,
        )
    return ImageContentDigestRecord(
        dataset_name=record.dataset_name,
        dataset_version=record.dataset_version,
        source_variant=record.source_variant,
        source_sample_id=record.source_sample_id,
        paired_source_sample_id=record.paired_source_sample_id,
        member_path=record.member_path,
        container_member_path=record.container_member_path,
        source_archive_sha256=record.source_archive_sha256,
        encoded_sha256=encoded_sha256,
        pixel_sha256=decoded[0],
        encoded_bytes=encoded_bytes,
        decoded_format=decoded[1],
        source_mode=decoded[2],
        canonical_mode="RGB",
        canonical_width=decoded[3],
        canonical_height=decoded[4],
        canonical_bytes=decoded[3] * decoded[4] * 3,
        exif_orientation_applied=decoded[5],
    )


def _decode_canonical_pixels(
    encoded: io.BytesIO,
    policy: ImageContentAuditPolicy,
    Image: Any,
    ImageOps: Any,
    UnidentifiedImageError: type[Exception],
) -> tuple[str, str, str, int, int, bool]:
    encoded.seek(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(encoded)
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("image header decode failed") from error
    with image:
        decoded_format = image.format
        source_mode = image.mode
        if decoded_format not in policy.allowed_decoder_formats:
            raise ValueError("decoded image format is not allowed")
        if source_mode not in policy.allowed_source_modes:
            raise ValueError("decoded source mode is not allowed")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("decoded image must contain exactly one frame")
        _validate_geometry(image.size, policy)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image.load()
            orientation = image.getexif().get(274, 1)
            orientation_applied = policy.apply_exif_orientation and orientation in range(2, 9)
            canonical = (
                ImageOps.exif_transpose(image)
                if policy.apply_exif_orientation
                else image.copy()
            )
        except (OSError, SyntaxError, Image.DecompressionBombError) as error:
            raise ValueError("image payload decode failed") from error
        with canonical:
            _validate_geometry(canonical.size, policy)
            if canonical.mode == "RGBA":
                if canonical.getchannel("A").getextrema() != (255, 255):
                    raise ValueError("RGBA source contains non-opaque alpha")
            if canonical.mode == "RGB":
                digest, width, height = _hash_rgb_image(canonical, policy)
            else:
                with canonical.convert("RGB") as rgb:
                    digest, width, height = _hash_rgb_image(rgb, policy)
    return digest.hexdigest(), decoded_format, source_mode, width, height, orientation_applied


def _read_member_bounded(
    bundle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    policy: ImageContentAuditPolicy,
) -> tuple[io.BytesIO, str, int]:
    encoded = io.BytesIO()
    digest = hashlib.sha256()
    total = 0
    try:
        with bundle.open(info, "r") as stream:
            while chunk := stream.read(policy.zip_read_chunk_bytes):
                total += len(chunk)
                if total > policy.maximum_member_encoded_bytes:
                    raise ValueError("ZIP image exceeds encoded-byte limit while reading")
                encoded.write(chunk)
                digest.update(chunk)
    except ValueError:
        encoded.close()
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        encoded.close()
        raise ValueError("ZIP image payload or CRC is invalid") from error
    if total != info.file_size:
        encoded.close()
        raise ValueError("ZIP image byte count differs from central directory")
    encoded.seek(0)
    return encoded, digest.hexdigest(), total


def _hash_rgb_image(
    image: Any,
    policy: ImageContentAuditPolicy,
) -> tuple[Any, int, int]:
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(_PIXEL_HASH_DOMAIN)
    digest.update(struct.pack(">QQ", width, height))
    rows = max(1, policy.maximum_hash_chunk_bytes // (width * 3))
    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        with image.crop((0, top, width, bottom)) as strip:
            payload = strip.tobytes("raw", "RGB")
        if len(payload) != width * (bottom - top) * 3:
            raise ValueError("canonical RGB row byte count differs")
        digest.update(payload)
    return digest, width, height


def _bound_member_info(
    index: dict[str, zipfile.ZipInfo],
    record: PublicCanineRecord,
    policy: ImageContentAuditPolicy,
) -> zipfile.ZipInfo:
    try:
        info = index[record.member_path]
    except KeyError as error:
        raise ValueError("manifest image member is absent from ZIP") from error
    if info.CRC != record.member_crc32 or info.file_size != record.member_uncompressed_bytes:
        raise ValueError("ZIP image metadata differs from semantic manifest")
    _validate_regular_info(info, policy, container=False)
    return info


def _bound_container_info(
    index: dict[str, zipfile.ZipInfo],
    records: tuple[PublicCanineRecord, ...],
    policy: ImageContentAuditPolicy,
) -> zipfile.ZipInfo:
    paths = {item.container_member_path for item in records}
    crcs = {item.container_member_crc32 for item in records}
    sizes = {item.container_member_uncompressed_bytes for item in records}
    if len(paths) != 1 or len(crcs) != 1 or len(sizes) != 1 or None in paths:
        raise ValueError("nested-container manifest binding is inconsistent")
    path = next(iter(paths))
    try:
        info = index[path]  # type: ignore[index]
    except KeyError as error:
        raise ValueError("nested container is absent from outer ZIP") from error
    if info.CRC != next(iter(crcs)) or info.file_size != next(iter(sizes)):
        raise ValueError("nested container metadata differs from semantic manifest")
    _validate_regular_info(info, policy, container=True)
    return info


def _stage_nested_container(
    outer: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    policy: ImageContentAuditPolicy,
) -> Any:
    """Sequentially stage one nested ZIP to avoid quadratic ZipExtFile seeks."""

    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    available = shutil.disk_usage(temporary_root).free
    required = info.file_size + policy.minimum_temporary_free_bytes_after_stage
    if available < required:
        raise ValueError("insufficient temporary space for nested ZIP staging")
    staged = tempfile.TemporaryFile(
        mode="w+b",
        prefix="cvi-nested-public-zip-",
        dir=temporary_root,
    )
    total = 0
    try:
        with outer.open(info, "r") as source:
            while chunk := source.read(policy.zip_read_chunk_bytes):
                total += len(chunk)
                if total > policy.maximum_container_bytes:
                    raise ValueError("nested ZIP exceeds staging byte limit")
                staged.write(chunk)
        if total != info.file_size or staged.tell() != info.file_size:
            raise ValueError("nested ZIP staged byte count differs")
        staged.flush()
        staged.seek(0)
        return staged
    except BaseException:
        staged.close()
        raise


def _validate_regular_info(
    info: zipfile.ZipInfo,
    policy: ImageContentAuditPolicy,
    *,
    container: bool,
) -> None:
    mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISLNK(mode):
        raise ValueError("ZIP content member is not a regular file")
    if info.flag_bits & 0x1:
        raise ValueError("encrypted ZIP members are forbidden")
    if info.compress_type not in policy.allowed_zip_compression:
        raise ValueError("ZIP compression method is not allowed")
    if info.file_size <= 0 or info.compress_size <= 0:
        raise ValueError("ZIP content member must be non-empty")
    if container:
        if info.file_size > policy.maximum_container_bytes:
            raise ValueError("nested ZIP container exceeds byte limit")
        maximum_ratio = policy.maximum_container_compression_ratio
    else:
        if info.file_size > policy.maximum_member_encoded_bytes:
            raise ValueError("ZIP image exceeds encoded-byte limit")
        if info.compress_size > policy.maximum_member_compressed_bytes:
            raise ValueError("ZIP image exceeds compressed-byte limit")
        maximum_ratio = policy.maximum_member_compression_ratio
    if info.file_size / info.compress_size > maximum_ratio:
        raise ValueError("ZIP member exceeds compression-ratio limit")


def _unique_info_index(bundle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = bundle.infolist()
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate member paths")
    return {item.filename: item for item in infos}


def _add_pixels(
    total: int,
    record: ImageContentDigestRecord,
    policy: ImageContentAuditPolicy,
) -> int:
    result = total + record.canonical_width * record.canonical_height
    if result > policy.maximum_total_pixels:
        raise ValueError("decoded corpus exceeds maximum total pixels")
    return result


def _validate_geometry(size: tuple[int, int], policy: ImageContentAuditPolicy) -> None:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("decoded image geometry must be positive")
    if width > policy.maximum_width or height > policy.maximum_height:
        raise ValueError("decoded image exceeds dimension limit")
    if width * height > policy.maximum_image_pixels:
        raise ValueError("decoded image exceeds pixel limit")


def _build_duplicate_groups(
    records: tuple[ImageContentDigestRecord, ...],
) -> tuple[PixelExactDuplicateGroup, ...]:
    by_digest: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_digest[record.pixel_sha256].append(record.source_sample_id)
    return tuple(
        PixelExactDuplicateGroup(digest, tuple(sorted(sample_ids)))
        for digest, sample_ids in sorted(by_digest.items())
        if len(sample_ids) >= 2
    )


def _semantic_manifest_sha256(manifest: PublicCanineManifest) -> str:
    records = []
    for item in sorted(manifest.records, key=lambda value: value.source_sample_id):
        records.append({
            "schema_version": item.schema_version,
            "dataset_name": item.dataset_name,
            "dataset_version": item.dataset_version,
            "source_variant": item.source_variant,
            "source_sample_id": item.source_sample_id,
            "dataset_identity_id": item.dataset_identity_id,
            "identity_semantics": item.identity_semantics.value,
            "region": item.region.value,
            "original_split": item.original_split,
            "sequence_id": item.sequence_id,
            "camera_token": item.camera_token,
            "camera_token_verified": item.camera_token_verified,
            "filename_identity_token": item.filename_identity_token,
            "source_cluster_id": item.source_cluster_id,
            "in_no_mono_subset": item.in_no_mono_subset,
            "paired_source_sample_id": item.paired_source_sample_id,
            "member_path": item.member_path,
            "member_crc32": item.member_crc32,
            "member_uncompressed_bytes": item.member_uncompressed_bytes,
            "source_archive_sha256": item.source_archive_sha256,
            "source_archive_receipt_sha256": item.source_archive_receipt_sha256,
            "container_member_path": item.container_member_path,
            "container_member_crc32": item.container_member_crc32,
            "container_member_uncompressed_bytes": item.container_member_uncompressed_bytes,
        })
    return content_sha256({
        "schema_version": manifest.schema_version,
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "source_archive_sha256": manifest.source_archive_sha256,
        "source_archive_receipt_sha256": manifest.source_archive_receipt_sha256,
        "records": records,
    })


def _open_bound_archive(path: Path, maximum_bytes: int) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("source archive cannot be opened without symlink following") from error
    current = os.fstat(descriptor)
    if not stat.S_ISREG(current.st_mode):
        os.close(descriptor)
        raise ValueError("source archive must be a regular file")
    if current.st_size <= 0 or current.st_size > maximum_bytes:
        os.close(descriptor)
        raise ValueError("source archive exceeds byte limit")
    return descriptor, _stat_token(current)


def _verify_archive_stability(path: Path, descriptor: int, initial: tuple[int, ...]) -> None:
    current = os.fstat(descriptor)
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError("source archive path changed during audit") from error
    if _stat_token(current) != initial or _stat_token(named) != initial:
        raise ValueError("source archive changed during audit")


def _stat_token(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256_stream(stream: Any, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def _pillow() -> tuple[Any, Any, Any, Any, type[Exception]]:
    try:
        import PIL
        from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "image-content audit requires the optional Pillow dependency"
        ) from error
    return PIL, Image, ImageFile, ImageOps, UnidentifiedImageError


def _require_sorted_unique_tokens(values: object, name: str) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError(f"{name} must be sorted and unique")
    for value in values:
        _require_token(value, name)


def _require_token(value: object, name: str, *, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
