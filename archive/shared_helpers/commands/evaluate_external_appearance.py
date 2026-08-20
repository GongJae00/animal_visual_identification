"""Paired frozen/trained DINOv2 evaluation on protected external protocols."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.public_sources.public_canine_manifest import (
    DOGFACE_DATASET,
    MPDD_DATASET,
    SIBETAN_DATASET,
    ArchiveReceiptBinding,
    PublicCanineRecord,
)
from data.public_sources.public_canine_semantic_intake import derive_public_canine_semantics
from data.public_sources.public_dataset_receipt_io import read_public_archive_receipt_bundle
from evaluation.search_metrics.metrics import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
    identity_clustered_bootstrap_ci,
)
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from shared.foundation.provenance import content_sha256
from evaluation.splits.protected_public_split import PublicSplitSourceBundle
from evaluation.splits.split_registry_binding import (
    validate_assignment_and_evaluator_binding,
)
from evaluation.splits.training_admission import TrainingAdmissionReceipt
from identification.export.appearance import ReceiptBoundDinov2Small

if __package__:
    from archive.shared_helpers.commands.evaluate_roi_reid import _reconstruct_dinov2_model
else:
    from evaluate_roi_reid import _reconstruct_dinov2_model


_TARGET_PROTOCOL_DATASET = {
    "DOGFACE_CLOSED_SET": DOGFACE_DATASET,
    "MPDD_CLOSED_SET": MPDD_DATASET,
    "SIBETAN_CROSS_SEQUENCE": SIBETAN_DATASET,
}
_TARGET_DATASETS = frozenset(_TARGET_PROTOCOL_DATASET.values())
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAXIMUM_IMAGE_BYTES = 67_108_864
_MAXIMUM_IMAGE_PIXELS = 33_554_432
_MAXIMUM_COMPRESSION_RATIO = 200.0
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_PREPROCESSING = {
    "input_shape": ["batch", 3, 224, 224],
    "color_mode": "RGB",
    "resize": "bilinear_stretch_224x224",
    "scale": 1.0 / 255.0,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "dtype": "float32",
}
_CHECKPOINT_PREPROCESSING = {
    **_PREPROCESSING,
    "mean": np.asarray(_PREPROCESSING["mean"], dtype=np.float32).tolist(),
    "std": np.asarray(_PREPROCESSING["std"], dtype=np.float32).tolist(),
}


@dataclass(frozen=True, order=True, slots=True)
class PopulationKey:
    protocol: str
    episode: str
    gallery_size: int
    shot: int


@dataclass(frozen=True, slots=True)
class PopulationMember:
    sample_token: str
    identity_token: str
    event_token: str
    bootstrap_cluster_token: str | None


@dataclass(frozen=True, slots=True)
class Population:
    key: PopulationKey
    gallery: tuple[PopulationMember, ...]
    queries: tuple[PopulationMember, ...]


@dataclass(frozen=True, slots=True)
class ArchiveSource:
    dataset_name: str
    archive_path: Path
    archive_receipt_path: Path
    dogface_classes_train_path: Path | None
    dogface_classes_test_path: Path | None


@dataclass(frozen=True, slots=True)
class ImageLocation:
    archive_path: Path
    record: PublicCanineRecord


class _Dinov2Pooler(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=tensor)
        embeddings = getattr(output, "pooler_output", None)
        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("DINOv2 pooler output is unavailable")
        return F.normalize(embeddings.float(), p=2, dim=1)


def _sha256_file(path: Path) -> str:
    _require_regular_file(path, "hashed input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_sha256(path: Path, expected: str) -> str:
    _require_sha256(expected, "expected file SHA-256")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"file SHA-256 differs for {path}")
    return observed


def _read_sha256_pinned_bytes(path: Path, expected: str) -> bytes:
    """Read the exact regular-file bytes whose digest was externally pinned."""

    _require_regular_file(path, "pinned input")
    _require_sha256(expected, "expected file SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("pinned input must remain a regular file")
        while chunk := os.read(descriptor, 1_048_576):
            digest.update(chunk)
            chunks.append(chunk)
        final = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    initial_identity = (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
        initial.st_ctime_ns,
    )
    final_identity = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    )
    if initial_identity != final_identity or (named.st_dev, named.st_ino) != (
        initial.st_dev,
        initial.st_ino,
    ):
        raise RuntimeError("pinned input changed while reading")
    if digest.hexdigest() != expected:
        raise ValueError(f"file SHA-256 differs for {path}")
    return b"".join(chunks)


def _require_regular_file(path: Path, name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute: {path}")
    if path.is_symlink():
        raise ValueError(f"{name} path must not be a symlink: {path}")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} path does not exist: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} path must be a regular file: {path}")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _parse_sha256(value: str) -> str:
    return _require_sha256(value, "command-line SHA-256")


def _source_spec_from_payload(payload: object) -> tuple[ArchiveSource, ...]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "sources"}:
        raise ValueError("external appearance source spec fields differ")
    if payload["schema_version"] != "cvi.external_appearance_source_spec.v1":
        raise ValueError("unsupported external appearance source spec schema")
    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(_TARGET_DATASETS):
        raise ValueError("source spec must contain exactly three external datasets")
    expected_keys = {
        "schema_version",
        "dataset_name",
        "archive_path",
        "archive_receipt_path",
        "dogface_classes_train_path",
        "dogface_classes_test_path",
    }
    sources: list[ArchiveSource] = []
    for raw in raw_sources:
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("external appearance source fields differ")
        if raw["schema_version"] != "cvi.external_appearance_source.v1":
            raise ValueError("unsupported external appearance source schema")
        dataset_name = raw["dataset_name"]
        if dataset_name not in _TARGET_DATASETS:
            raise ValueError("unsupported external appearance dataset")
        train_value = raw["dogface_classes_train_path"]
        test_value = raw["dogface_classes_test_path"]
        if dataset_name == DOGFACE_DATASET:
            if not isinstance(train_value, str) or not isinstance(test_value, str):
                raise ValueError("DogFace source requires both publisher class files")
        elif train_value is not None or test_value is not None:
            raise ValueError("only DogFace may specify publisher class files")
        values = (raw["archive_path"], raw["archive_receipt_path"])
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("archive paths must be non-empty strings")
        source = ArchiveSource(
            dataset_name=dataset_name,
            archive_path=Path(raw["archive_path"]),
            archive_receipt_path=Path(raw["archive_receipt_path"]),
            dogface_classes_train_path=(Path(train_value) if train_value else None),
            dogface_classes_test_path=(Path(test_value) if test_value else None),
        )
        for name, path in (
            ("archive", source.archive_path),
            ("archive receipt", source.archive_receipt_path),
            ("DogFace train classes", source.dogface_classes_train_path),
            ("DogFace test classes", source.dogface_classes_test_path),
        ):
            if path is not None:
                _require_regular_file(path, name)
        sources.append(source)
    names = [source.dataset_name for source in sources]
    if set(names) != _TARGET_DATASETS or len(set(names)) != len(names):
        raise ValueError("source spec must contain each external dataset once")
    return tuple(sorted(sources, key=lambda source: source.dataset_name))


def _validate_split_documents(
    assignment: dict[str, Any],
    labels: dict[str, Any],
    receipt: dict[str, Any],
    source_payload: dict[str, Any],
    *,
    expected_receipt_sha256: str,
) -> tuple[PublicSplitSourceBundle, dict[str, Any]]:
    validate_assignment_and_evaluator_binding(
        assignment,
        receipt,
        labels,
        expected_receipt_sha256,
    )
    source = _validate_source_bundle_receipt(source_payload, receipt)

    source_by_token = {sample.sample_token: sample for sample in source.samples}
    assignment_by_token = {
        record["sample_token"]: record for record in assignment["records"]
    }
    for label in labels["records"]:
        token = label["sample_token"]
        sample = source_by_token.get(token)
        if sample is None:
            raise ValueError(
                "evaluator labels reference a sample outside source bundle"
            )
        assigned = assignment_by_token[token]
        expected = (
            sample.identity_token,
            sample.source_sample_id,
            sample.dataset_identity_id,
            sample.sequence_token,
            sample.raw_frame_index,
            sample.original_split,
            sample.region,
            sample.dataset_name,
            sample.source_variant,
        )
        observed = (
            label["identity_token"],
            label["source_sample_id"],
            label["dataset_identity_id"],
            label["sequence_token"],
            label["raw_frame_index"],
            label["original_split"],
            label["region"],
            assigned["dataset_name"],
            assigned["source_variant"],
        )
        if observed != expected:
            raise ValueError("assignment, evaluator labels, and source bundle differ")
        if token != _opaque_token("sample", sample.source_sample_id) or (
            sample.identity_token
            != _opaque_token("identity", sample.dataset_identity_id)
        ):
            raise ValueError("source bundle opaque token derivation differs")
    return source, source_by_token


def _validate_source_bundle_receipt(
    source_payload: dict[str, Any], receipt: Mapping[str, Any]
) -> PublicSplitSourceBundle:
    source = PublicSplitSourceBundle.from_dict(source_payload)
    source_sha256 = source.bundle_sha256
    if receipt.get("source_bundle_sha256") != source_sha256:
        raise ValueError("protected split receipt does not bind the source bundle")
    if receipt.get("evidence_bindings") != [
        list(item) for item in source.evidence_bindings
    ]:
        raise ValueError("protected split receipt evidence bindings differ")
    input_hashes = receipt.get("input_file_sha256s")
    if not isinstance(input_hashes, list):
        raise ValueError("protected split receipt input hashes differ")
    if any(
        not isinstance(item, list)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], str)
        for item in input_hashes
    ):
        raise ValueError("protected split receipt input hashes differ")
    input_hash_map = dict(input_hashes)
    if len(input_hash_map) != len(input_hashes):
        raise ValueError("protected split receipt input hashes repeat a name")
    if input_hash_map.get("source_bundle_payload_sha256") != source_sha256:
        raise ValueError("protected split receipt source input hash differs")
    return source


def _opaque_token(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\x00{value}".encode()).hexdigest()


def _build_populations(assignment: Mapping[str, Any]) -> tuple[Population, ...]:
    grouped: dict[PopulationKey, dict[str, list[PopulationMember]]] = defaultdict(
        lambda: {"GALLERY": [], "KNOWN_QUERY": []}
    )
    for record in assignment.get("records", []):
        for use in record.get("uses", []):
            protocol = use.get("protocol")
            if protocol not in _TARGET_PROTOCOL_DATASET:
                continue
            if record.get("dataset_name") != _TARGET_PROTOCOL_DATASET[protocol]:
                raise ValueError("protocol population crosses its dataset boundary")
            role = use.get("role")
            if role not in {"GALLERY", "KNOWN_QUERY"}:
                raise ValueError("closed-set population contains an unsupported role")
            key = PopulationKey(
                protocol=protocol,
                episode=use["episode"],
                gallery_size=use["gallery_size"],
                shot=use["shot"],
            )
            grouped[key][role].append(
                PopulationMember(
                    sample_token=record["sample_token"],
                    identity_token=record["identity_token"],
                    event_token=use["event_token"],
                    bootstrap_cluster_token=use["bootstrap_cluster_token"],
                )
            )

    populations: list[Population] = []
    for key, roles in sorted(grouped.items()):
        gallery = tuple(sorted(roles["GALLERY"], key=_population_member_key))
        queries = tuple(sorted(roles["KNOWN_QUERY"], key=_population_member_key))
        if not gallery or not queries:
            raise ValueError("external closed-set population is incomplete")
        gallery_tokens = [member.sample_token for member in gallery]
        query_tokens = [member.sample_token for member in queries]
        if len(set(gallery_tokens)) != len(gallery_tokens) or (
            len(set(query_tokens)) != len(query_tokens)
        ):
            raise ValueError("external population repeats a sample within one role")
        if set(gallery_tokens) & set(query_tokens):
            raise ValueError("external population overlaps gallery and query samples")
        gallery_counts = Counter(member.identity_token for member in gallery)
        query_identities = {member.identity_token for member in queries}
        if set(gallery_counts) != query_identities or any(
            count != key.shot for count in gallery_counts.values()
        ):
            raise ValueError(
                "external population violates its closed-set shot contract"
            )
        if any(member.bootstrap_cluster_token is None for member in queries):
            raise ValueError("external query lacks an identity bootstrap cluster")
        clusters_by_identity: dict[str, set[str | None]] = defaultdict(set)
        identities_by_cluster: dict[str | None, set[str]] = defaultdict(set)
        for member in queries:
            clusters_by_identity[member.identity_token].add(
                member.bootstrap_cluster_token
            )
            identities_by_cluster[member.bootstrap_cluster_token].add(
                member.identity_token
            )
        if any(len(values) != 1 for values in clusters_by_identity.values()) or any(
            len(values) != 1 for values in identities_by_cluster.values()
        ):
            raise ValueError("external query bootstrap clusters differ from identities")
        populations.append(Population(key=key, gallery=gallery, queries=queries))
    if {population.key.protocol for population in populations} != set(
        _TARGET_PROTOCOL_DATASET
    ):
        raise ValueError("one or more external appearance protocols are absent")
    return tuple(populations)


def _population_member_key(member: PopulationMember) -> tuple[str, str]:
    return member.event_token, member.sample_token


def _extract_raw_frame_index(record: PublicCanineRecord) -> int:
    match = re.search(r"frame:(\d+)", record.source_sample_id)
    if match:
        return int(match.group(1))
    match = re.search(r"image:(\d+)\.(\d+)", record.source_sample_id)
    if match:
        return int(match.group(2))
    match = re.search(r"_(\d+)\.jpg$", record.member_path)
    if match:
        return int(match.group(1))
    match = re.search(r"clip:(\d+):frame:(\d+)", record.source_sample_id)
    if match:
        return int(match.group(1)) * 1000 + int(match.group(2))
    return 0


def _derive_manifest_records(
    sources: tuple[ArchiveSource, ...],
) -> tuple[dict[str, PublicCanineRecord], list[dict[str, str | None]]]:
    records: dict[str, PublicCanineRecord] = {}
    provenance: list[dict[str, str | None]] = []
    for source in sources:
        receipt = read_public_archive_receipt_bundle(source.archive_receipt_path)
        manifests, _ = derive_public_canine_semantics(
            dataset_name=source.dataset_name,
            archive_path=source.archive_path,
            binding=ArchiveReceiptBinding(
                dataset_name=source.dataset_name,
                archive_sha256=receipt.archive_sha256,
                archive_receipt_sha256=receipt.receipt_sha256,
            ),
            dogface_classes_train=source.dogface_classes_train_path,
            dogface_classes_test=source.dogface_classes_test_path,
        )
        for manifest in manifests:
            for record in manifest.records:
                if record.source_sample_id in records:
                    raise ValueError("publisher manifests repeat a source sample ID")
                records[record.source_sample_id] = record
        provenance.append(
            {
                "dataset_name": source.dataset_name,
                "archive_sha256": receipt.archive_sha256,
                "archive_receipt_sha256": receipt.receipt_sha256,
                "archive_receipt_file_sha256": _sha256_file(
                    source.archive_receipt_path
                ),
                "dogface_classes_train_sha256": (
                    _sha256_file(source.dogface_classes_train_path)
                    if source.dogface_classes_train_path is not None
                    else None
                ),
                "dogface_classes_test_sha256": (
                    _sha256_file(source.dogface_classes_test_path)
                    if source.dogface_classes_test_path is not None
                    else None
                ),
            }
        )
    return records, provenance


def _bind_image_locations(
    populations: Sequence[Population],
    source_by_token: Mapping[str, Any],
    manifest_records: Mapping[str, PublicCanineRecord],
    sources: Sequence[ArchiveSource],
) -> dict[str, ImageLocation]:
    archive_by_dataset = {
        source.dataset_name: source.archive_path for source in sources
    }
    selected_tokens = {
        member.sample_token
        for population in populations
        for member in (*population.gallery, *population.queries)
    }
    locations: dict[str, ImageLocation] = {}
    for token in selected_tokens:
        sample = source_by_token[token]
        record = manifest_records.get(sample.source_sample_id)
        if record is None:
            raise ValueError(
                "selected source sample is absent from publisher manifests"
            )
        sequence_token = (
            sample.identity_token
            if sample.dataset_name in {DOGFACE_DATASET, MPDD_DATASET}
            else _opaque_token(
                "sequence",
                record.sequence_id or sample.dataset_identity_id,
            )
        )
        observed = (
            record.dataset_identity_id,
            record.dataset_name,
            record.source_variant,
            record.original_split,
            _extract_raw_frame_index(record),
            record.paired_source_sample_id,
            record.in_no_mono_subset,
            record.region.value,
            sequence_token,
        )
        expected = (
            sample.dataset_identity_id,
            sample.dataset_name,
            sample.source_variant,
            sample.original_split,
            sample.raw_frame_index,
            sample.paired_source_sample_id,
            sample.in_no_mono_subset,
            sample.region,
            sample.sequence_token,
        )
        if observed != expected:
            raise ValueError("publisher manifest differs from protected source bundle")
        locations[token] = ImageLocation(
            archive_path=archive_by_dataset[sample.dataset_name],
            record=record,
        )
    return locations


def _require_safe_member_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("ZIP member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ZIP member path is unsafe")


def _read_member_bytes(
    archive: zipfile.ZipFile,
    record: PublicCanineRecord,
) -> bytes:
    _require_safe_member_path(record.member_path)
    try:
        info = archive.getinfo(record.member_path)
    except KeyError as error:
        raise ValueError("publisher image member is absent") from error
    mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 0x1:
        raise ValueError("publisher image member type is unsafe")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise ValueError("publisher image compression is unsupported")
    if info.CRC != record.member_crc32 or info.file_size != (
        record.member_uncompressed_bytes
    ):
        raise ValueError("publisher image metadata differs from semantic manifest")
    if info.file_size > _MAXIMUM_IMAGE_BYTES or info.compress_size > (
        _MAXIMUM_IMAGE_BYTES
    ):
        raise ValueError("publisher image member exceeds byte limit")
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > _MAXIMUM_COMPRESSION_RATIO:
        raise ValueError("publisher image member exceeds compression-ratio limit")
    with archive.open(info, "r") as stream:
        payload = stream.read(_MAXIMUM_IMAGE_BYTES + 1)
    if len(payload) != info.file_size or len(payload) > _MAXIMUM_IMAGE_BYTES:
        raise ValueError("publisher image member byte count differs")
    return payload


def _open_verified_archive(
    stack: ExitStack, path: Path, expected_sha256: str
) -> zipfile.ZipFile:
    _require_regular_file(path, "publisher archive")
    _require_sha256(expected_sha256, "publisher archive SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    stream = stack.enter_context(os.fdopen(descriptor, "rb"))
    digest = hashlib.sha256()
    while chunk := stream.read(1_048_576):
        digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("publisher archive SHA-256 differs before inference")
    stream.seek(0)
    return stack.enter_context(zipfile.ZipFile(stream))


def _decode_image(archive: zipfile.ZipFile, record: PublicCanineRecord) -> Image.Image:
    payload = _read_member_bytes(archive, record)
    with Image.open(io.BytesIO(payload)) as encoded:
        width, height = encoded.size
        if width <= 0 or height <= 0 or width * height > _MAXIMUM_IMAGE_PIXELS:
            raise ValueError("publisher image dimensions exceed policy")
        image = encoded.convert("RGB")
        image.load()
    return image


def _preprocess_images(
    images: Sequence[Image.Image], device: torch.device
) -> torch.Tensor:
    if not images:
        raise ValueError("image batch must not be empty")
    arrays = [
        np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        for image in images
    ]
    batch = np.stack(arrays).transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(batch).to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    )
    tensor.div_(255.0)
    mean = torch.tensor(_PREPROCESSING["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(_PREPROCESSING["std"], device=device).view(1, 3, 1, 1)
    return tensor.sub_(mean).div_(std)


def _normalize_model_output(value: Any, batch_size: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (batch_size, 384):
        raise RuntimeError(f"{name} output must have shape [{batch_size}, 384]")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} output contains non-finite values")
    norms = torch.linalg.vector_norm(value.float(), dim=1)
    if torch.any(norms <= 1e-8):
        raise RuntimeError(f"{name} output contains a zero embedding")
    return F.normalize(value.float(), p=2, dim=1)


def _extract_paired_embeddings(
    token_order: Sequence[str],
    locations: Mapping[str, ImageLocation],
    archives: Mapping[Path, zipfile.ZipFile],
    frozen_model: torch.nn.Module,
    trained_model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frozen_cache: dict[str, np.ndarray] = {}
    trained_cache: dict[str, np.ndarray] = {}
    frozen_model.to(device).eval()
    trained_model.to(device).eval()
    for offset in range(0, len(token_order), batch_size):
        tokens = token_order[offset : offset + batch_size]
        images = [
            _decode_image(
                archives[locations[token].archive_path],
                locations[token].record,
            )
            for token in tokens
        ]
        tensor = _preprocess_images(images, device)
        with torch.inference_mode():
            frozen = _normalize_model_output(
                frozen_model(tensor), len(tokens), "frozen model"
            )
            trained = _normalize_model_output(
                trained_model(tensor), len(tokens), "trained model"
            )
        frozen_values = frozen.cpu().numpy().astype(np.float32, copy=False)
        trained_values = trained.cpu().numpy().astype(np.float32, copy=False)
        for index, token in enumerate(tokens):
            frozen_cache[token] = frozen_values[index]
            trained_cache[token] = trained_values[index]
    return frozen_cache, trained_cache


def _evaluate_population(
    population: Population,
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    query_embeddings = np.stack(
        [embeddings[member.sample_token] for member in population.queries]
    )
    gallery_embeddings = np.stack(
        [embeddings[member.sample_token] for member in population.gallery]
    )
    result = evaluate_multi_template_closed_set(
        compute_cosine_score_matrix(query_embeddings, gallery_embeddings),
        query_identity_ids=np.asarray(
            [member.identity_token for member in population.queries]
        ),
        gallery_template_identity_ids=np.asarray(
            [member.identity_token for member in population.gallery]
        ),
        self_match_policy="exclude",
        query_template_ids=np.asarray(
            [member.sample_token for member in population.queries]
        ),
        gallery_template_ids=np.asarray(
            [member.sample_token for member in population.gallery]
        ),
        rank_ks=(1, 5, 10),
    )
    for member, row in zip(population.queries, result["query_rows"], strict=True):
        row["bootstrap_cluster_id"] = member.bootstrap_cluster_token
        row["query_event_token"] = member.event_token
    return result


def _paired_identity_clustered_delta_ci(
    frozen_rows: Sequence[Mapping[str, Any]],
    trained_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if len(frozen_rows) != len(trained_rows) or not frozen_rows:
        raise ValueError("paired comparison requires equal non-empty query rows")
    delta_rows: list[dict[str, Any]] = []
    for index, (frozen, trained) in enumerate(
        zip(frozen_rows, trained_rows, strict=True)
    ):
        frozen_key = (
            frozen.get("query_event_token"),
            frozen.get("query_identity_id"),
            frozen.get("bootstrap_cluster_id"),
        )
        trained_key = (
            trained.get("query_event_token"),
            trained.get("query_identity_id"),
            trained.get("bootstrap_cluster_id"),
        )
        if frozen_key != trained_key:
            raise ValueError(f"paired query order differs at row {index}")
        try:
            delta = float(trained[metric]) - float(frozen[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"paired rows lack numeric metric {metric!r}") from error
        delta_rows.append(
            {
                "bootstrap_cluster_id": frozen["bootstrap_cluster_id"],
                "delta": delta,
            }
        )
    result = identity_clustered_bootstrap_ci(
        delta_rows,
        metric="delta",
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    result["metric"] = metric
    result["direction"] = "trained_minus_frozen"
    result["paired_query_order_verified"] = True
    result["interval_method"] = "paired_whole_identity_percentile_bootstrap"
    return result


def _aggregate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "num_queries",
            "num_gallery_templates",
            "num_gallery_identities",
            "closed_set",
            "ranking_unit",
            "aggregation",
            "tie_policy",
            "self_match_policy",
            "mAP",
            "mINP",
            "MRR",
            "Rank-1",
            "Rank-5",
            "Rank-10",
        )
    }


def _load_models(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    checkpoint_bytes = _read_sha256_pinned_bytes(
        args.checkpoint, args.checkpoint_sha256
    )
    checkpoint_sha256 = args.checkpoint_sha256
    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or payload.get("preprocessing") != (
        _CHECKPOINT_PREPROCESSING
    ):
        raise ValueError("training checkpoint preprocessing contract differs")
    trained = _reconstruct_dinov2_model(payload, args.model_dir)._backbone

    frozen_receipt = ReceiptBoundDinov2Small(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        device="cpu",
        max_batch_size=args.batch_size,
    )
    if frozen_receipt.model_sha256 != args.frozen_model_sha256:
        raise ValueError("frozen DINOv2 model SHA-256 differs from external pin")
    frozen_receipt._ensure_loaded()
    training_admission = payload.get("training_admission")
    if not isinstance(training_admission, dict) or set(training_admission) != {
        "receipt_sha256",
        "receipt",
    }:
        raise ValueError("training checkpoint lacks its admission receipt")
    admission_payload = training_admission["receipt"]
    if not isinstance(admission_payload, dict):
        raise ValueError("training checkpoint admission receipt must be an object")
    admission = TrainingAdmissionReceipt.from_dict(admission_payload)
    if training_admission["receipt_sha256"] != admission.receipt_sha256:
        raise ValueError("training checkpoint admission receipt hash differs")
    if admission.model_receipt_sha256 != frozen_receipt.weight_receipt_sha256:
        raise ValueError(
            "trained and frozen models do not share the admitted base weight"
        )
    frozen = _Dinov2Pooler(frozen_receipt._backbone)
    return (
        frozen,
        trained,
        {
            "frozen_model_sha256": frozen_receipt.model_sha256,
            "frozen_weight_intake_receipt_sha256": (
                frozen_receipt.weight_receipt_sha256
            ),
            "frozen_preprocessor_intake_receipt_sha256": (
                frozen_receipt.preprocessor_receipt_sha256
            ),
            "frozen_preprocessor_sha256": frozen_receipt.preprocessor_sha256,
            "trained_checkpoint_sha256": checkpoint_sha256,
            "trained_checkpoint_epoch": payload["epoch"],
            "trained_checkpoint_training_admission_receipt_sha256": (
                admission.receipt_sha256
            ),
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--split-receipt-sha256", required=True, type=_parse_sha256)
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--weight-intake-bundle", required=True, type=Path)
    parser.add_argument("--preprocessor-intake-bundle", required=True, type=Path)
    parser.add_argument("--frozen-model-sha256", required=True, type=_parse_sha256)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True, type=_parse_sha256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for name, path in (
        ("assignment", args.assignment),
        ("labels", args.labels),
        ("source bundle", args.source_bundle),
        ("split receipt", args.split_receipt),
        ("source spec", args.source_spec),
        ("weight intake bundle", args.weight_intake_bundle),
        ("preprocessor intake bundle", args.preprocessor_intake_bundle),
        ("checkpoint", args.checkpoint),
    ):
        _require_regular_file(path, name)
    if (
        not args.model_dir.is_absolute()
        or args.model_dir.is_symlink()
        or not (args.model_dir.is_dir())
    ):
        raise ValueError("model directory must be an absolute non-symlink directory")

    assignment = read_strict_json_object(args.assignment)
    labels = read_strict_json_object(args.labels)
    source_payload = read_strict_json_object(args.source_bundle)
    receipt = read_strict_json_object(args.split_receipt)
    source, source_by_token = _validate_split_documents(
        assignment,
        labels,
        receipt,
        source_payload,
        expected_receipt_sha256=args.split_receipt_sha256,
    )
    source_spec_payload = read_strict_json_object(args.source_spec)
    sources = _source_spec_from_payload(source_spec_payload)
    populations = _build_populations(assignment)
    manifest_records, archive_provenance = _derive_manifest_records(sources)
    locations = _bind_image_locations(
        populations, source_by_token, manifest_records, sources
    )
    token_order = tuple(sorted(locations))
    frozen_model, trained_model, model_provenance = _load_models(args)

    with ExitStack() as stack:
        archive_hashes = {
            item["dataset_name"]: item["archive_sha256"]
            for item in archive_provenance
        }
        archives = {
            source_item.archive_path: _open_verified_archive(
                stack,
                source_item.archive_path,
                archive_hashes[source_item.dataset_name],
            )
            for source_item in sources
        }
        frozen_cache, trained_cache = _extract_paired_embeddings(
            token_order,
            locations,
            archives,
            frozen_model,
            trained_model,
            device=torch.device(args.device),
            batch_size=args.batch_size,
        )

    results: list[dict[str, Any]] = []
    for index, population in enumerate(populations):
        frozen_result = _evaluate_population(population, frozen_cache)
        trained_result = _evaluate_population(population, trained_cache)
        delta = {
            metric: _paired_identity_clustered_delta_ci(
                frozen_result["query_rows"],
                trained_result["query_rows"],
                metric=metric,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + index,
            )
            for metric in ("Rank-1", "reciprocal_rank", "AP", "INP")
        }
        results.append(
            {
                "protocol": population.key.protocol,
                "episode": population.key.episode,
                "gallery_size": population.key.gallery_size,
                "shot": population.key.shot,
                "population_sha256": content_sha256(
                    {
                        "gallery": [
                            member.event_token for member in population.gallery
                        ],
                        "queries": [
                            member.event_token for member in population.queries
                        ],
                    }
                ),
                "gallery_order_sha256": content_sha256(
                    [member.sample_token for member in population.gallery]
                ),
                "query_order_sha256": content_sha256(
                    [member.sample_token for member in population.queries]
                ),
                "paired_query_gallery_order_verified": True,
                "frozen": _aggregate_metrics(frozen_result),
                "trained": _aggregate_metrics(trained_result),
                "paired_delta_ci": delta,
            }
        )

    report = {
        "schema_version": "cvi.external_appearance_paired_evaluation.v1",
        "status": "PASS_PAIRED_EXTERNAL_APPEARANCE_EVALUATION",
        "preprocessing": {
            **_PREPROCESSING,
            "same_tensor_object_per_model_pair": True,
        },
        "protocols": results,
        "provenance": {
            "split_receipt_sha256": args.split_receipt_sha256,
            "assignment_sha256": content_sha256(assignment),
            "labels_sha256": content_sha256(labels),
            "source_bundle_sha256": source.bundle_sha256,
            "source_spec_sha256": content_sha256(source_spec_payload),
            "split_policy_sha256": assignment["policy_sha256"],
            "split_evidence_root_sha256": assignment["evidence_root_sha256"],
            "split_seed_commitment": assignment["seed_commitment"],
            "publisher_archives": archive_provenance,
            **model_provenance,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "interpretation": (
            "PAIRED_EXTERNAL_CLOSED_SET_APPEARANCE_COMPARISON_NOT_BIOMETRIC_VALIDATION"
        ),
    }
    write_private_json_bundle(((args.output, report),))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": os.fspath(args.output),
                "protocol_populations": len(results),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
