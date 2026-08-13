"""Protected label-blind pHash audit over the four public canine archives.

The runner rederives semantic manifests, authenticates the protected image-
content receipts, and then reopens each receipt-bound ZIP.  Every image is
decoded from the authenticated encoded bytes, EXIF-transposed, converted to a
canonical RGB raster, verified against the earlier pixel receipt, converted to
32 x 32 luma with Pillow Lanczos, and passed to :mod:`identity_methods.classical.phash_mih` under an
opaque identifier.  Identity and split metadata never enter fingerprinting or
candidate search.

The resulting candidate list is only bounded perceptual-similarity evidence.
It is not duplicate adjudication, a split, or model admission.
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

from data.public_canine_manifest import (
    DOGFACE_DATASET,
    MPDD_DATASET,
    SIBETAN_DATASET,
    YT_DATASET,
    ArchiveReceiptBinding,
    PublicCanineManifest,
    PublicCanineRecord,
)
from data.public_canine_semantic_intake import derive_public_canine_semantics
from data.public_dataset_receipt_io import read_public_archive_receipt_bundle
from foundation.protected_io import read_strict_json_object, write_private_json_bundle
from foundation.provenance import content_sha256
from identity_methods.classical.phash_mih import (
    MAXIMUM_EXACT_RADIUS,
    CandidateLimitExceeded,
    NearDuplicateCandidate,
    PHashFingerprint,
    find_near_duplicate_candidates,
    fingerprint_luma32,
    opaque_sample_id,
)

_DATASETS = (DOGFACE_DATASET, MPDD_DATASET, SIBETAN_DATASET, YT_DATASET)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PIXEL_HASH_DOMAIN = b"CVI_PIXEL_CANONICAL_RGB_V1\0"
_ALLOWED_ZIP_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)


@dataclass(frozen=True, slots=True)
class PublicCaninePHashSource:
    dataset_name: str
    archive_path: Path
    archive_receipt_path: Path
    semantic_receipt_path: Path
    image_content_receipt_path: Path
    dogface_classes_train_path: Path | None = None
    dogface_classes_test_path: Path | None = None
    schema_version: str = "cvi.public_canine_phash_source.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_canine_phash_source.v1":
            raise ValueError("unsupported pHash source schema")
        if self.dataset_name not in _DATASETS:
            raise ValueError("unsupported public canine dataset")
        for name in (
            "archive_path",
            "archive_receipt_path",
            "semantic_receipt_path",
            "image_content_receipt_path",
        ):
            _require_path(getattr(self, name), name)
        dogface_paths = (
            self.dogface_classes_train_path,
            self.dogface_classes_test_path,
        )
        if self.dataset_name == DOGFACE_DATASET:
            if any(path is None for path in dogface_paths):
                raise ValueError("DogFace pHash source requires both class files")
            for path in dogface_paths:
                _require_path(path, "DogFace class path")
        elif any(path is not None for path in dogface_paths):
            raise ValueError("non-DogFace source must not provide class files")


@dataclass(frozen=True, slots=True)
class PublicCaninePHashPolicy:
    radius: int = MAXIMUM_EXACT_RADIUS
    maximum_fingerprints: int = 50_000
    maximum_pair_inspections: int = 100_000_000
    maximum_expanded_candidates: int = 1_000_000
    maximum_source_spec_bytes: int = 65_536
    maximum_policy_bytes: int = 65_536
    maximum_receipt_bytes: int = 536_870_912
    maximum_archive_bytes: int = 2_147_483_648
    maximum_member_encoded_bytes: int = 67_108_864
    maximum_member_compressed_bytes: int = 67_108_864
    maximum_container_bytes: int = 2_147_483_648
    minimum_temporary_free_bytes_after_stage: int = 1_073_741_824
    maximum_member_compression_ratio: float = 200.0
    maximum_container_compression_ratio: float = 4.0
    maximum_image_pixels: int = 33_554_432
    maximum_total_pixels: int = 250_000_000_000
    read_chunk_bytes: int = 1_048_576
    resize_width: int = 32
    resize_height: int = 32
    canonical_mode: str = "RGB"
    luma_mode: str = "L"
    interpolation: str = "PILLOW_LANCZOS"
    apply_exif_orientation: bool = True
    allowed_zip_compression: tuple[int, ...] = _ALLOWED_ZIP_COMPRESSION
    schema_version: str = "cvi.public_canine_phash_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_canine_phash_policy.v1":
            raise ValueError("unsupported public canine pHash policy")
        integer_fields = (
            "maximum_fingerprints",
            "maximum_pair_inspections",
            "maximum_expanded_candidates",
            "maximum_source_spec_bytes",
            "maximum_policy_bytes",
            "maximum_receipt_bytes",
            "maximum_archive_bytes",
            "maximum_member_encoded_bytes",
            "maximum_member_compressed_bytes",
            "maximum_container_bytes",
            "minimum_temporary_free_bytes_after_stage",
            "maximum_image_pixels",
            "maximum_total_pixels",
            "read_chunk_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.radius, bool) or not isinstance(self.radius, int):
            raise TypeError("radius must be an integer")
        if not 0 <= self.radius <= MAXIMUM_EXACT_RADIUS:
            raise ValueError("radius must be inside exact MIH range 0..10")
        for name in (
            "maximum_member_compression_ratio",
            "maximum_container_compression_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 1.0 <= float(value) <= 1_000_000.0:
                raise ValueError(f"{name} is outside the supported range")
        if (
            self.resize_width,
            self.resize_height,
            self.canonical_mode,
            self.luma_mode,
            self.interpolation,
            self.apply_exif_orientation,
        ) != (32, 32, "RGB", "L", "PILLOW_LANCZOS", True):
            raise ValueError("pHash raster semantics are fixed")
        if self.allowed_zip_compression != _ALLOWED_ZIP_COMPRESSION:
            raise ValueError("ZIP compression policy differs")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "radius": self.radius,
            "maximum_fingerprints": self.maximum_fingerprints,
            "maximum_pair_inspections": self.maximum_pair_inspections,
            "maximum_expanded_candidates": self.maximum_expanded_candidates,
            "maximum_source_spec_bytes": self.maximum_source_spec_bytes,
            "maximum_policy_bytes": self.maximum_policy_bytes,
            "maximum_receipt_bytes": self.maximum_receipt_bytes,
            "maximum_archive_bytes": self.maximum_archive_bytes,
            "maximum_member_encoded_bytes": self.maximum_member_encoded_bytes,
            "maximum_member_compressed_bytes": self.maximum_member_compressed_bytes,
            "maximum_container_bytes": self.maximum_container_bytes,
            "minimum_temporary_free_bytes_after_stage": (
                self.minimum_temporary_free_bytes_after_stage
            ),
            "maximum_member_compression_ratio": float(
                self.maximum_member_compression_ratio
            ),
            "maximum_container_compression_ratio": float(
                self.maximum_container_compression_ratio
            ),
            "maximum_image_pixels": self.maximum_image_pixels,
            "maximum_total_pixels": self.maximum_total_pixels,
            "read_chunk_bytes": self.read_chunk_bytes,
            "resize_width": self.resize_width,
            "resize_height": self.resize_height,
            "canonical_mode": self.canonical_mode,
            "luma_mode": self.luma_mode,
            "interpolation": self.interpolation,
            "apply_exif_orientation": self.apply_exif_orientation,
            "allowed_zip_compression": list(self.allowed_zip_compression),
        }


@dataclass(frozen=True, slots=True)
class _AuthenticatedSource:
    source: PublicCaninePHashSource
    manifests: tuple[PublicCanineManifest, ...]
    archive_receipt_sha256: str
    semantic_receipt_sha256: str
    image_receipt_sha256: str
    image_policy_sha256: str
    image_decoder_name: str
    image_decoder_version: str
    image_records: dict[str, dict[str, Any]]


def read_public_canine_phash_sources(
    path: Path, *, maximum_bytes: int = 65_536
) -> tuple[PublicCaninePHashSource, ...]:
    _require_bounded_file(path, maximum_bytes, "source spec")
    payload = read_strict_json_object(path)
    if set(payload) != {"schema_version", "sources"} or payload[
        "schema_version"
    ] != "cvi.public_canine_phash_source_spec.v1":
        raise ValueError("public canine pHash source spec fields differ")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(_DATASETS):
        raise ValueError("pHash source spec must contain exactly four sources")
    sources = tuple(_source_from_dict(item) for item in raw_sources)
    names = tuple(source.dataset_name for source in sources)
    if tuple(sorted(names)) != tuple(sorted(_DATASETS)) or len(set(names)) != 4:
        raise ValueError("pHash source spec must contain each audited dataset once")
    return tuple(sorted(sources, key=lambda item: item.dataset_name))


def read_public_canine_phash_policy(path: Path) -> PublicCaninePHashPolicy:
    _require_bounded_file(path, 65_536, "pHash policy")
    payload = read_strict_json_object(path)
    expected = set(PublicCaninePHashPolicy().to_dict())
    if set(payload) != expected:
        raise ValueError("public canine pHash policy fields differ")
    values = dict(payload)
    compression = values.get("allowed_zip_compression")
    if not isinstance(compression, list):
        raise TypeError("allowed_zip_compression must be a list")
    values["allowed_zip_compression"] = tuple(compression)
    policy = PublicCaninePHashPolicy(**values)
    _require_bounded_file(path, policy.maximum_policy_bytes, "pHash policy")
    return policy


def run_public_canine_phash_audit(
    *,
    sources: tuple[PublicCaninePHashSource, ...],
    policy: PublicCaninePHashPolicy,
    evidence_output: Path,
    binding_output: Path,
    tool_provenance: dict[str, Any],
) -> tuple[str, str]:
    """Authenticate, fingerprint, search, and atomically publish two artifacts."""

    _validate_source_set(sources)
    if not isinstance(policy, PublicCaninePHashPolicy):
        raise TypeError("policy must be PublicCaninePHashPolicy")
    _validate_tool_provenance(tool_provenance)
    _preflight_outputs(evidence_output, binding_output)
    authenticated = tuple(_authenticate_source(source, policy) for source in sources)
    total_expected = sum(
        len(manifest.records)
        for item in authenticated
        for manifest in item.manifests
    )
    if total_expected > policy.maximum_fingerprints:
        raise ValueError("corpus exceeds pHash fingerprint cap")

    fingerprints: list[PHashFingerprint] = []
    bindings: list[dict[str, str]] = []
    exact_pixel_fingerprint_cache: dict[str, tuple[int, int]] = {}
    total_pixels = 0
    for item in authenticated:
        produced, source_bindings, source_pixels = _fingerprint_source(
            item, policy, exact_pixel_fingerprint_cache
        )
        fingerprints.extend(produced)
        bindings.extend(source_bindings)
        total_pixels += source_pixels
        if total_pixels > policy.maximum_total_pixels:
            raise ValueError("pHash decode exceeds aggregate pixel cap")
    if len(fingerprints) != total_expected or len(bindings) != total_expected:
        raise RuntimeError("pHash output cardinality differs from semantic manifests")
    if len({item.opaque_sample_id for item in fingerprints}) != len(fingerprints):
        raise ValueError("opaque sample ID collision")

    fingerprint_by_opaque = {item.opaque_sample_id: item for item in fingerprints}
    pixel_groups: dict[str, list[PHashFingerprint]] = defaultdict(list)
    for item in authenticated:
        for manifest in item.manifests:
            for record in manifest.records:
                opaque_id = opaque_sample_id(record.source_sample_id)
                pixel_groups[
                    item.image_records[record.source_sample_id]["pixel_sha256"]
                ].append(fingerprint_by_opaque[opaque_id])
    representatives: list[PHashFingerprint] = []
    components_by_representative: dict[str, tuple[str, ...]] = {}
    for component in pixel_groups.values():
        ordered_component = tuple(sorted(component, key=lambda value: value.opaque_sample_id))
        hashes = {
            (item.original_hash, item.horizontal_flip_hash) for item in ordered_component
        }
        if len(hashes) != 1:
            raise RuntimeError("exact-pixel component has inconsistent pHash")
        representative = ordered_component[0]
        representatives.append(representative)
        components_by_representative[representative.opaque_sample_id] = tuple(
            item.opaque_sample_id for item in ordered_component
        )
    representative_candidates = find_near_duplicate_candidates(
        representatives,
        radius=policy.radius,
        maximum_pair_inspections=policy.maximum_pair_inspections,
        maximum_accepted_candidates=policy.maximum_expanded_candidates,
        maximum_fingerprints=policy.maximum_fingerprints,
    )
    candidates = _expand_component_candidates(
        components_by_representative,
        representative_candidates,
        maximum_candidates=policy.maximum_expanded_candidates,
    )
    source_binding_rows = tuple(
        sorted(
            (
                {
                    "archive_receipt_sha256": item.archive_receipt_sha256,
                    "semantic_receipt_sha256": item.semantic_receipt_sha256,
                    "image_receipt_sha256": item.image_receipt_sha256,
                    "image_policy_sha256": item.image_policy_sha256,
                }
                for item in authenticated
            ),
            key=lambda row: tuple(row.values()),
        )
    )
    source_spec_sha256 = _source_spec_sha256(sources)
    pillow = _pillow()[0]
    provenance_sha256 = content_sha256(tool_provenance)
    evidence = {
        "schema_version": "cvi.public_canine_phash_evidence.v1",
        "policy": policy.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "source_spec_sha256": source_spec_sha256,
        "source_receipt_bindings": list(source_binding_rows),
        "source_receipt_bindings_sha256": content_sha256(source_binding_rows),
        "decoder": {
            "name": "Pillow",
            "version": pillow.__version__,
            "operation_order": "EXIF_TRANSPOSE_RGB_VERIFY_LUMA_LANCZOS_32X32",
            "interpolation": "PIL.Image.Resampling.LANCZOS",
        },
        "fingerprints": [
            {
                "schema_version": item.schema_version,
                "opaque_sample_id": item.opaque_sample_id,
                "original_hash_hex": f"{item.original_hash:016x}",
                "horizontal_flip_hash_hex": f"{item.horizontal_flip_hash:016x}",
            }
            for item in sorted(fingerprints, key=lambda value: value.opaque_sample_id)
        ],
        "candidates": [
            {
                "schema_version": item.schema_version,
                "left_opaque_sample_id": item.left_opaque_sample_id,
                "right_opaque_sample_id": item.right_opaque_sample_id,
                "hamming_distance": item.hamming_distance,
            }
            for item in candidates
        ],
        "fingerprint_count": len(fingerprints),
        "exact_pixel_component_count": len(representatives),
        "candidate_count": len(candidates),
        "total_decoded_source_pixels": total_pixels,
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": provenance_sha256,
        "decision": "PASS_LABEL_BLIND_PHASH_CANDIDATE_GENERATION",
        "interpretation": (
            "PERCEPTUAL_SIMILARITY_CANDIDATES_ONLY_NOT_DUPLICATE_"
            "ADJUDICATION_SPLIT_OR_MODEL_ADMISSION"
        ),
    }
    evidence_sha256 = content_sha256(evidence)
    evidence_bundle = {
        "schema_version": "cvi.public_canine_phash_evidence_bundle.v1",
        "evidence": evidence,
        "evidence_sha256": evidence_sha256,
    }
    binding = {
        "schema_version": "cvi.public_canine_phash_binding.v1",
        "evidence_sha256": evidence_sha256,
        "bindings": sorted(bindings, key=lambda row: row["opaque_sample_id"]),
        "binding_count": len(bindings),
        "source_spec_sha256": source_spec_sha256,
        "source_receipt_bindings_sha256": content_sha256(source_binding_rows),
        "interpretation": (
            "SENSITIVE_OPAQUE_TO_SOURCE_PROVENANCE_JOIN_ONLY_"
            "MUST_NOT_ENTER_CANDIDATE_GENERATION_OR_SCORING"
        ),
    }
    binding_sha256 = content_sha256(binding)
    binding_bundle = {
        "schema_version": "cvi.public_canine_phash_binding_bundle.v1",
        "binding": binding,
        "binding_sha256": binding_sha256,
    }
    write_private_json_bundle(
        ((evidence_output, evidence_bundle), (binding_output, binding_bundle))
    )
    return evidence_sha256, binding_sha256


def _authenticate_source(
    source: PublicCaninePHashSource, policy: PublicCaninePHashPolicy
) -> _AuthenticatedSource:
    for path in (
        source.archive_receipt_path,
        source.semantic_receipt_path,
        source.image_content_receipt_path,
    ):
        _require_bounded_file(path, policy.maximum_receipt_bytes, "protected receipt")
    archive_receipt = read_public_archive_receipt_bundle(source.archive_receipt_path)
    binding = ArchiveReceiptBinding(
        source.dataset_name,
        archive_receipt.archive_sha256,
        archive_receipt.receipt_sha256,
    )
    manifests, derived_semantic = derive_public_canine_semantics(
        dataset_name=source.dataset_name,
        archive_path=source.archive_path,
        binding=binding,
        dogface_classes_train=source.dogface_classes_train_path,
        dogface_classes_test=source.dogface_classes_test_path,
    )
    semantic_bundle = _read_semantic_bundle(source.semantic_receipt_path)
    if semantic_bundle["receipt"] != derived_semantic.to_dict() or semantic_bundle[
        "receipt_sha256"
    ] != derived_semantic.receipt_sha256:
        raise ValueError("protected semantic receipt differs from current derivation")
    image_bundle = _read_image_bundle(source.image_content_receipt_path)
    if image_bundle["semantic_receipt_sha256"] != derived_semantic.receipt_sha256:
        raise ValueError("image-content bundle semantic binding differs")
    receipt = image_bundle["receipt"]
    combined = PublicCanineManifest(
        source.dataset_name,
        manifests[0].dataset_version,
        binding.archive_sha256,
        binding.archive_receipt_sha256,
        tuple(record for manifest in manifests for record in manifest.records),
    )
    _verify_image_receipt(receipt, combined, image_bundle)
    records = receipt["records"]
    assert isinstance(records, list)
    return _AuthenticatedSource(
        source=source,
        manifests=manifests,
        archive_receipt_sha256=binding.archive_receipt_sha256,
        semantic_receipt_sha256=derived_semantic.receipt_sha256,
        image_receipt_sha256=image_bundle["receipt_sha256"],
        image_policy_sha256=image_bundle["policy_sha256"],
        image_decoder_name=receipt["decoder_name"],
        image_decoder_version=receipt["decoder_version"],
        image_records={item["source_sample_id"]: item for item in records},
    )


def _fingerprint_source(
    authenticated: _AuthenticatedSource,
    policy: PublicCaninePHashPolicy,
    exact_pixel_fingerprint_cache: dict[str, tuple[int, int]],
) -> tuple[list[PHashFingerprint], list[dict[str, str]], int]:
    source = authenticated.source
    records = tuple(
        sorted(
            (
                record
                for manifest in authenticated.manifests
                for record in manifest.records
            ),
            key=lambda item: item.source_sample_id,
        )
    )
    PIL, Image, ImageFile, ImageOps, UnidentifiedImageError = _pillow()
    if authenticated.image_decoder_name != "Pillow" or (
        authenticated.image_decoder_version != PIL.__version__
    ):
        raise ValueError("current Pillow differs from protected image receipt")
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("Pillow truncated-image decoding must be disabled")
    descriptor, initial_stat = _open_bound_archive(
        source.archive_path, policy.maximum_archive_bytes
    )
    produced: list[PHashFingerprint] = []
    bindings: list[dict[str, str]] = []
    total_pixels = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            digest = _sha256_stream(stream, policy.read_chunk_bytes)
            expected_sha = records[0].source_archive_sha256
            if digest != expected_sha:
                raise ValueError("archive bytes differ from authenticated source")
            stream.seek(0)
            with zipfile.ZipFile(stream) as outer:
                outer_index = _unique_info_index(outer)
                direct = tuple(
                    record for record in records if record.container_member_path is None
                )
                for record in direct:
                    fingerprint, pixel_count = _fingerprint_member(
                        outer,
                        _bound_member_info(outer_index, record, policy),
                        record,
                        authenticated.image_records[record.source_sample_id],
                        policy,
                        Image,
                        ImageOps,
                        UnidentifiedImageError,
                        exact_pixel_fingerprint_cache,
                    )
                    produced.append(fingerprint)
                    bindings.append(_binding_row(record, fingerprint))
                    total_pixels = _bounded_pixel_sum(total_pixels, pixel_count, policy)
                nested: dict[str, list[PublicCanineRecord]] = defaultdict(list)
                for record in records:
                    if record.container_member_path is not None:
                        nested[record.container_member_path].append(record)
                for container_path in sorted(nested):
                    group = tuple(nested[container_path])
                    info = _bound_container_info(outer_index, group, policy)
                    with _stage_nested_container(outer, info, policy) as nested_stream:
                        with zipfile.ZipFile(nested_stream) as inner:
                            inner_index = _unique_info_index(inner)
                            for record in group:
                                fingerprint, pixel_count = _fingerprint_member(
                                    inner,
                                    _bound_member_info(inner_index, record, policy),
                                    record,
                                    authenticated.image_records[record.source_sample_id],
                                    policy,
                                    Image,
                                    ImageOps,
                                    UnidentifiedImageError,
                                    exact_pixel_fingerprint_cache,
                                )
                                produced.append(fingerprint)
                                bindings.append(_binding_row(record, fingerprint))
                                total_pixels = _bounded_pixel_sum(
                                    total_pixels, pixel_count, policy
                                )
            _verify_archive_stability(source.archive_path, stream.fileno(), initial_stat)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return produced, bindings, total_pixels


def _fingerprint_member(
    bundle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    record: PublicCanineRecord,
    protected: dict[str, Any],
    policy: PublicCaninePHashPolicy,
    Image: Any,
    ImageOps: Any,
    UnidentifiedImageError: type[Exception],
    exact_pixel_fingerprint_cache: dict[str, tuple[int, int]] | None = None,
) -> tuple[PHashFingerprint, int]:
    opaque_id = opaque_sample_id(record.source_sample_id)
    rgb_payload, width, height, pixel_sha256 = _canonical_rgb_member(
        bundle,
        info,
        protected,
        policy,
        Image,
        ImageOps,
        UnidentifiedImageError,
    )
    with Image.frombytes("RGB", (width, height), rgb_payload) as rgb:
        with rgb.convert("L") as luma:
            with luma.resize(
                (32, 32), resample=Image.Resampling.LANCZOS
            ) as resized:
                luma_pixels = resized.tobytes("raw", "L")
    cached = (
        None
        if exact_pixel_fingerprint_cache is None
        else exact_pixel_fingerprint_cache.get(pixel_sha256)
    )
    if cached is None:
        fingerprint = fingerprint_luma32(
            opaque_id=opaque_id, luma_pixels=luma_pixels
        )
        if exact_pixel_fingerprint_cache is not None:
            exact_pixel_fingerprint_cache[pixel_sha256] = (
                fingerprint.original_hash,
                fingerprint.horizontal_flip_hash,
            )
    else:
        fingerprint = PHashFingerprint(
            opaque_sample_id=opaque_id,
            original_hash=cached[0],
            horizontal_flip_hash=cached[1],
        )
    return fingerprint, width * height


def _canonical_rgb_member(
    bundle: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    protected: dict[str, Any],
    policy: PublicCaninePHashPolicy,
    Image: Any,
    ImageOps: Any,
    UnidentifiedImageError: type[Exception],
) -> tuple[bytes, int, int, str]:
    """Return the receipt-authenticated canonical RGB raster for one member."""

    encoded, encoded_sha256 = _read_member(bundle, info, policy)
    if encoded_sha256 != protected["encoded_sha256"] or len(
        encoded.getbuffer()
    ) != protected["encoded_bytes"]:
        encoded.close()
        raise ValueError("encoded image differs from protected image receipt")
    try:
        encoded.seek(0)
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
        encoded.close()
        raise ValueError("pHash image header decode failed") from error
    try:
        with image:
            if image.format != protected["decoded_format"] or image.mode != protected[
                "source_mode"
            ]:
                raise ValueError("decoded image metadata differs from protected receipt")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("pHash image must contain exactly one frame")
            _validate_geometry(image.size, policy)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image.load()
            orientation = image.getexif().get(274, 1)
            orientation_applied = orientation in range(2, 9)
            canonical = ImageOps.exif_transpose(image)
            with canonical:
                _validate_geometry(canonical.size, policy)
                if canonical.mode == "RGBA" and canonical.getchannel("A").getextrema() != (
                    255,
                    255,
                ):
                    raise ValueError("RGBA source contains non-opaque alpha")
                rgb = canonical.copy() if canonical.mode == "RGB" else canonical.convert("RGB")
                with rgb:
                    width, height = rgb.size
                    pixel_sha256 = _hash_rgb(rgb, policy.read_chunk_bytes)
                    if (
                        pixel_sha256 != protected["pixel_sha256"]
                        or width != protected["canonical_width"]
                        or height != protected["canonical_height"]
                        or width * height * 3 != protected["canonical_bytes"]
                        or protected["canonical_mode"] != "RGB"
                        or orientation_applied
                        != protected["exif_orientation_applied"]
                    ):
                        raise ValueError("canonical pixels differ from protected receipt")
                    rgb_payload = rgb.tobytes("raw", "RGB")
                    if len(rgb_payload) != width * height * 3:
                        raise ValueError("canonical RGB byte count differs")
    finally:
        encoded.close()
    return rgb_payload, width, height, pixel_sha256


def _verify_image_receipt(
    receipt: dict[str, Any],
    manifest: PublicCanineManifest,
    bundle: dict[str, Any],
) -> None:
    expected_top = {
        "schema_version",
        "dataset_name",
        "dataset_version",
        "source_archive_sha256",
        "source_archive_receipt_sha256",
        "semantic_manifest_sha256",
        "policy_sha256",
        "decoder_name",
        "decoder_version",
        "records",
        "exact_duplicate_groups",
        "source_variant_counts",
        "paired_record_count",
        "total_encoded_bytes",
        "total_canonical_pixels",
        "unique_pixel_digest_count",
        "decision",
        "interpretation",
    }
    if set(receipt) != expected_top or receipt["schema_version"] != (
        "cvi.image_content_audit_receipt.v1"
    ):
        raise ValueError("image-content receipt fields differ")
    if (
        receipt["dataset_name"] != manifest.dataset_name
        or receipt["dataset_version"] != manifest.dataset_version
        or receipt["source_archive_sha256"] != manifest.source_archive_sha256
        or receipt["source_archive_receipt_sha256"]
        != manifest.source_archive_receipt_sha256
        or receipt["semantic_manifest_sha256"] != _semantic_manifest_sha256(manifest)
        or receipt["policy_sha256"] != bundle["policy_sha256"]
    ):
        raise ValueError("image-content receipt source binding differs")
    raw_records = receipt["records"]
    if not isinstance(raw_records, list) or len(raw_records) != len(manifest.records):
        raise ValueError("image-content record cardinality differs")
    expected_records = {item.source_sample_id: item for item in manifest.records}
    observed_ids: list[str] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise TypeError("image-content record must be an object")
        _verify_image_record(raw, expected_records)
        observed_ids.append(raw["source_sample_id"])
    if observed_ids != sorted(expected_records) or len(set(observed_ids)) != len(
        observed_ids
    ):
        raise ValueError("image-content records are not sorted unique coverage")
    if receipt["source_variant_counts"] != [
        list(item) for item in sorted(Counter(
            record.source_variant for record in manifest.records
        ).items())
    ]:
        raise ValueError("image-content source variant counts differ")
    if receipt["paired_record_count"] != sum(
        record.paired_source_sample_id is not None for record in manifest.records
    ):
        raise ValueError("image-content paired count differs")
    if receipt["total_encoded_bytes"] != sum(
        item["encoded_bytes"] for item in raw_records
    ) or receipt["total_canonical_pixels"] != sum(
        item["canonical_width"] * item["canonical_height"] for item in raw_records
    ):
        raise ValueError("image-content aggregate counts differ")
    if receipt["unique_pixel_digest_count"] != len(
        {item["pixel_sha256"] for item in raw_records}
    ):
        raise ValueError("image-content unique pixel count differs")
    expected_groups = []
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in raw_records:
        grouped[item["pixel_sha256"]].append(item["source_sample_id"])
    for digest, sample_ids in sorted(grouped.items()):
        if len(sample_ids) >= 2:
            expected_groups.append({
                "schema_version": "cvi.pixel_exact_duplicate_group.v1",
                "pixel_sha256": digest,
                "source_sample_ids": sorted(sample_ids),
            })
    if receipt["exact_duplicate_groups"] != expected_groups:
        raise ValueError("image-content exact duplicate groups differ")
    if receipt["decision"] != "PASS_IMAGE_CONTENT_AUDIT" or receipt[
        "interpretation"
    ] != "DECODE_AND_PIXEL_EXACT_DUPLICATE_EVIDENCE_ONLY_NOT_SPLIT_OR_MODEL_ADMISSION":
        raise ValueError("image-content receipt decision differs")


def _verify_image_record(
    raw: dict[str, Any], expected_records: dict[str, PublicCanineRecord]
) -> None:
    expected_keys = {
        "schema_version",
        "dataset_name",
        "dataset_version",
        "source_variant",
        "source_sample_id",
        "paired_source_sample_id",
        "member_path",
        "container_member_path",
        "source_archive_sha256",
        "encoded_sha256",
        "pixel_sha256",
        "encoded_bytes",
        "decoded_format",
        "source_mode",
        "canonical_mode",
        "canonical_width",
        "canonical_height",
        "canonical_bytes",
        "exif_orientation_applied",
    }
    if set(raw) != expected_keys or raw["schema_version"] != (
        "cvi.image_content_digest_record.v1"
    ):
        raise ValueError("image-content digest record fields differ")
    sample_id = raw["source_sample_id"]
    if sample_id not in expected_records:
        raise ValueError("image-content record is absent from semantic manifest")
    record = expected_records[sample_id]
    expected_binding = {
        "dataset_name": record.dataset_name,
        "dataset_version": record.dataset_version,
        "source_variant": record.source_variant,
        "source_sample_id": record.source_sample_id,
        "paired_source_sample_id": record.paired_source_sample_id,
        "member_path": record.member_path,
        "container_member_path": record.container_member_path,
        "source_archive_sha256": record.source_archive_sha256,
    }
    if any(raw[name] != value for name, value in expected_binding.items()):
        raise ValueError("image-content digest record source binding differs")
    for name in ("encoded_sha256", "pixel_sha256"):
        _require_sha256(raw[name], name)
    for name in (
        "encoded_bytes",
        "canonical_width",
        "canonical_height",
        "canonical_bytes",
    ):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("image-content digest count must be positive")
    if raw["canonical_bytes"] != raw["canonical_width"] * raw[
        "canonical_height"
    ] * 3 or raw["canonical_mode"] != "RGB":
        raise ValueError("image-content canonical geometry differs")
    if not isinstance(raw["exif_orientation_applied"], bool):
        raise TypeError("image-content EXIF flag must be boolean")


def _read_semantic_bundle(path: Path) -> dict[str, Any]:
    bundle = read_strict_json_object(path)
    expected = {
        "schema_version",
        "receipt_sha256",
        "receipt",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if set(bundle) != expected or bundle["schema_version"] != (
        "cvi.public_canine_semantic_bundle.v1"
    ):
        raise ValueError("public canine semantic bundle fields differ")
    if content_sha256(bundle["tool_provenance"]) != bundle[
        "tool_provenance_sha256"
    ] or content_sha256(bundle["receipt"]) != bundle["receipt_sha256"]:
        raise ValueError("public canine semantic bundle digest differs")
    return bundle


def _read_image_bundle(path: Path) -> dict[str, Any]:
    bundle = read_strict_json_object(path)
    expected = {
        "schema_version",
        "semantic_receipt_sha256",
        "policy",
        "policy_sha256",
        "receipt",
        "receipt_sha256",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if set(bundle) != expected or bundle["schema_version"] != (
        "cvi.image_content_audit_bundle.v1"
    ):
        raise ValueError("image-content bundle fields differ")
    for payload_name, digest_name in (
        ("policy", "policy_sha256"),
        ("receipt", "receipt_sha256"),
        ("tool_provenance", "tool_provenance_sha256"),
    ):
        if content_sha256(bundle[payload_name]) != bundle[digest_name]:
            raise ValueError(f"image-content {payload_name} digest differs")
    _validate_image_content_policy(bundle["policy"])
    if not isinstance(bundle["receipt"], dict):
        raise TypeError("image-content receipt must be an object")
    return bundle


def _validate_image_content_policy(value: object) -> None:
    if not isinstance(value, dict):
        raise TypeError("image-content policy must be an object")
    expected = {
        "schema_version",
        "maximum_archive_bytes",
        "maximum_records",
        "maximum_member_encoded_bytes",
        "maximum_member_compressed_bytes",
        "maximum_total_encoded_bytes",
        "maximum_container_bytes",
        "minimum_temporary_free_bytes_after_stage",
        "maximum_container_compression_ratio",
        "maximum_member_compression_ratio",
        "maximum_width",
        "maximum_height",
        "maximum_image_pixels",
        "maximum_total_pixels",
        "maximum_hash_chunk_bytes",
        "zip_read_chunk_bytes",
        "allowed_decoder_formats",
        "allowed_source_modes",
        "allowed_zip_compression",
        "canonical_mode",
        "apply_exif_orientation",
    }
    if set(value) != expected or value["schema_version"] != (
        "cvi.image_content_audit_policy.v1"
    ):
        raise ValueError("image-content policy fields differ")
    if value["canonical_mode"] != "RGB" or value["apply_exif_orientation"] is not True:
        raise ValueError("image-content canonical semantics differ")
    if value["allowed_zip_compression"] != list(_ALLOWED_ZIP_COMPRESSION):
        raise ValueError("image-content ZIP compression policy differs")


def _expand_component_candidates(
    components: dict[str, tuple[str, ...]],
    representative_candidates: tuple[NearDuplicateCandidate, ...],
    *,
    maximum_candidates: int,
) -> tuple[NearDuplicateCandidate, ...]:
    """Expand exact-pixel representatives under one explicit output cap."""

    output: list[NearDuplicateCandidate] = []

    def append(left: str, right: str, distance: int) -> None:
        if len(output) >= maximum_candidates:
            raise CandidateLimitExceeded("expanded pHash candidate cap exceeded")
        ordered = tuple(sorted((left, right)))
        output.append(NearDuplicateCandidate(ordered[0], ordered[1], distance))

    for component in sorted(components.values()):
        for left_index, left in enumerate(component):
            for right in component[left_index + 1:]:
                append(left, right, 0)
    for candidate in representative_candidates:
        left_component = components[candidate.left_opaque_sample_id]
        right_component = components[candidate.right_opaque_sample_id]
        for left in left_component:
            for right in right_component:
                append(left, right, candidate.hamming_distance)
    return tuple(sorted(
        output,
        key=lambda item: (
            item.left_opaque_sample_id,
            item.right_opaque_sample_id,
        ),
    ))


def _validate_tool_provenance(value: object) -> None:
    if not isinstance(value, dict):
        raise TypeError("tool provenance must be an object")
    expected = {
        "schema_version",
        "code_source_manifest_sha256",
        "code_source_files",
        "runtime",
        "runtime_sha256",
    }
    v2_expected = expected | {"logical_component", "entrypoints"}
    if set(value) not in (expected, v2_expected) or value["schema_version"] not in {
        "cvi.offline_tool_provenance.v1",
        "canine_identity.source_provenance.v2",
    }:
        raise ValueError("tool provenance fields differ")
    files = value["code_source_files"]
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
        raise TypeError("tool provenance source manifest differs")
    if content_sha256(files) != value["code_source_manifest_sha256"]:
        raise ValueError("tool provenance source manifest digest differs")
    if content_sha256(value["runtime"]) != value["runtime_sha256"]:
        raise ValueError("tool provenance runtime digest differs")


def _preflight_outputs(evidence: Path, binding: Path) -> None:
    if evidence == binding:
        raise ValueError("pHash evidence and binding outputs must differ")
    parents = {evidence.parent.resolve(strict=True), binding.parent.resolve(strict=True)}
    if len(parents) != 1:
        raise ValueError("pHash outputs must share one protected directory")
    for path in (evidence, binding):
        if path.is_symlink():
            raise ValueError("pHash output must not be a symlink")
        if path.exists():
            raise FileExistsError(path)


def _source_from_dict(value: object) -> PublicCaninePHashSource:
    if not isinstance(value, dict):
        raise TypeError("pHash source entry must be an object")
    expected = {
        "schema_version",
        "dataset_name",
        "archive_path",
        "archive_receipt_path",
        "semantic_receipt_path",
        "image_content_receipt_path",
        "dogface_classes_train_path",
        "dogface_classes_test_path",
    }
    if set(value) != expected:
        raise ValueError("pHash source entry fields differ")
    converted = dict(value)
    for name in (
        "archive_path",
        "archive_receipt_path",
        "semantic_receipt_path",
        "image_content_receipt_path",
        "dogface_classes_train_path",
        "dogface_classes_test_path",
    ):
        raw = converted[name]
        if raw is not None and not isinstance(raw, str):
            raise TypeError("pHash source paths must be strings or null")
        converted[name] = None if raw is None else Path(raw)
    return PublicCaninePHashSource(**converted)


def _validate_source_set(sources: tuple[PublicCaninePHashSource, ...]) -> None:
    if not isinstance(sources, tuple) or len(sources) != 4:
        raise ValueError("pHash audit requires exactly four sources")
    if any(not isinstance(item, PublicCaninePHashSource) for item in sources):
        raise TypeError("pHash sources must be PublicCaninePHashSource")
    names = tuple(item.dataset_name for item in sources)
    if tuple(sorted(names)) != tuple(sorted(_DATASETS)) or len(set(names)) != 4:
        raise ValueError("pHash audit requires each audited dataset exactly once")


def _source_spec_sha256(sources: tuple[PublicCaninePHashSource, ...]) -> str:
    rows = []
    for source in sorted(sources, key=lambda item: item.dataset_name):
        rows.append({
            "schema_version": source.schema_version,
            "dataset_name": source.dataset_name,
            "archive_path": str(source.archive_path),
            "archive_receipt_path": str(source.archive_receipt_path),
            "semantic_receipt_path": str(source.semantic_receipt_path),
            "image_content_receipt_path": str(source.image_content_receipt_path),
            "dogface_classes_train_path": (
                None
                if source.dogface_classes_train_path is None
                else str(source.dogface_classes_train_path)
            ),
            "dogface_classes_test_path": (
                None
                if source.dogface_classes_test_path is None
                else str(source.dogface_classes_test_path)
            ),
        })
    return content_sha256({
        "schema_version": "cvi.public_canine_phash_source_spec.v1",
        "sources": rows,
    })


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


def _binding_row(
    record: PublicCanineRecord, fingerprint: PHashFingerprint
) -> dict[str, str]:
    return {
        "opaque_sample_id": fingerprint.opaque_sample_id,
        "dataset_name": record.dataset_name,
        "source_sample_id": record.source_sample_id,
    }


def _open_bound_archive(path: Path, maximum_bytes: int) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("source archive cannot be opened without symlink following") from error
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0 or value.st_size > maximum_bytes:
        os.close(descriptor)
        raise ValueError("source archive is not a bounded regular file")
    return descriptor, _stat_token(value)


def _verify_archive_stability(path: Path, descriptor: int, initial: tuple[int, ...]) -> None:
    current = os.fstat(descriptor)
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError("source archive path changed during pHash audit") from error
    if _stat_token(current) != initial or _stat_token(named) != initial:
        raise ValueError("source archive changed during pHash audit")


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


def _read_member(
    bundle: zipfile.ZipFile, info: zipfile.ZipInfo, policy: PublicCaninePHashPolicy
) -> tuple[io.BytesIO, str]:
    output = io.BytesIO()
    digest = hashlib.sha256()
    total = 0
    try:
        with bundle.open(info, "r") as source:
            while chunk := source.read(policy.read_chunk_bytes):
                total += len(chunk)
                if total > policy.maximum_member_encoded_bytes:
                    raise ValueError("pHash image exceeds encoded-byte cap")
                output.write(chunk)
                digest.update(chunk)
    except BaseException:
        output.close()
        raise
    if total != info.file_size:
        output.close()
        raise ValueError("pHash ZIP member byte count differs")
    output.seek(0)
    return output, digest.hexdigest()


def _bound_member_info(
    index: dict[str, zipfile.ZipInfo],
    record: PublicCanineRecord,
    policy: PublicCaninePHashPolicy,
) -> zipfile.ZipInfo:
    try:
        info = index[record.member_path]
    except KeyError as error:
        raise ValueError("pHash image member is absent from ZIP") from error
    if info.CRC != record.member_crc32 or info.file_size != record.member_uncompressed_bytes:
        raise ValueError("pHash image metadata differs from semantic manifest")
    _validate_info(info, policy, container=False)
    return info


def _bound_container_info(
    index: dict[str, zipfile.ZipInfo],
    records: tuple[PublicCanineRecord, ...],
    policy: PublicCaninePHashPolicy,
) -> zipfile.ZipInfo:
    paths = {item.container_member_path for item in records}
    crcs = {item.container_member_crc32 for item in records}
    sizes = {item.container_member_uncompressed_bytes for item in records}
    if len(paths) != 1 or len(crcs) != 1 or len(sizes) != 1 or None in paths:
        raise ValueError("nested pHash container binding is inconsistent")
    path = next(iter(paths))
    try:
        info = index[path]  # type: ignore[index]
    except KeyError as error:
        raise ValueError("nested pHash container is absent") from error
    if info.CRC != next(iter(crcs)) or info.file_size != next(iter(sizes)):
        raise ValueError("nested pHash container metadata differs")
    _validate_info(info, policy, container=True)
    return info


def _validate_info(
    info: zipfile.ZipInfo,
    policy: PublicCaninePHashPolicy,
    *,
    container: bool,
) -> None:
    mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 0x1:
        raise ValueError("pHash ZIP member is not an unencrypted regular file")
    if info.compress_type not in policy.allowed_zip_compression:
        raise ValueError("pHash ZIP compression method differs")
    if info.file_size <= 0 or info.compress_size <= 0:
        raise ValueError("pHash ZIP member must be non-empty")
    maximum_bytes = (
        policy.maximum_container_bytes
        if container
        else policy.maximum_member_encoded_bytes
    )
    maximum_ratio = (
        policy.maximum_container_compression_ratio
        if container
        else policy.maximum_member_compression_ratio
    )
    if info.file_size > maximum_bytes or (
        not container and info.compress_size > policy.maximum_member_compressed_bytes
    ) or info.file_size / info.compress_size > maximum_ratio:
        raise ValueError("pHash ZIP member exceeds resource policy")


def _stage_nested_container(
    outer: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    policy: PublicCaninePHashPolicy,
) -> Any:
    temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
    if shutil.disk_usage(temporary_root).free < (
        info.file_size + policy.minimum_temporary_free_bytes_after_stage
    ):
        raise ValueError("insufficient temporary space for nested pHash ZIP")
    staged = tempfile.TemporaryFile(
        mode="w+b", prefix="cvi-phash-nested-", dir=temporary_root
    )
    total = 0
    try:
        with outer.open(info, "r") as source:
            while chunk := source.read(policy.read_chunk_bytes):
                total += len(chunk)
                if total > policy.maximum_container_bytes:
                    raise ValueError("nested pHash ZIP exceeds staging cap")
                staged.write(chunk)
        if total != info.file_size or staged.tell() != info.file_size:
            raise ValueError("nested pHash ZIP staged byte count differs")
        staged.flush()
        staged.seek(0)
        return staged
    except BaseException:
        staged.close()
        raise


def _unique_info_index(bundle: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = bundle.infolist()
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise ValueError("pHash ZIP contains duplicate member paths")
    return {item.filename: item for item in infos}


def _hash_rgb(image: Any, chunk_bytes: int) -> str:
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(_PIXEL_HASH_DOMAIN)
    digest.update(struct.pack(">QQ", width, height))
    rows = max(1, chunk_bytes // (width * 3))
    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        with image.crop((0, top, width, bottom)) as strip:
            payload = strip.tobytes("raw", "RGB")
        if len(payload) != width * (bottom - top) * 3:
            raise ValueError("pHash canonical RGB byte count differs")
        digest.update(payload)
    return digest.hexdigest()


def _validate_geometry(size: tuple[int, int], policy: PublicCaninePHashPolicy) -> None:
    width, height = size
    if width <= 0 or height <= 0 or width * height > policy.maximum_image_pixels:
        raise ValueError("pHash source geometry exceeds policy")


def _bounded_pixel_sum(
    total: int, pixels: int, policy: PublicCaninePHashPolicy
) -> int:
    result = total + pixels
    if result > policy.maximum_total_pixels:
        raise ValueError("pHash decode exceeds aggregate pixel cap")
    return result


def _pillow() -> tuple[Any, Any, Any, Any, type[Exception]]:
    try:
        import PIL
        from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
    except ModuleNotFoundError as error:
        raise RuntimeError("public canine pHash audit requires Pillow") from error
    return PIL, Image, ImageFile, ImageOps, UnidentifiedImageError


def _require_bounded_file(path: Path, maximum: int, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    value = resolved.stat()
    if not resolved.is_file() or value.st_size <= 0 or value.st_size > maximum:
        raise ValueError(f"{label} must be a bounded regular file")


def _require_path(value: object, name: str) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
