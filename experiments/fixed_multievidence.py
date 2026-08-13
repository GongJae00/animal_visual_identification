"""Exposed publisher-test A0/F5/N3 fixed-panel diagnostics."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from evaluation.retrieval import (
    compute_cosine_score_matrix,
    identity_clustered_bootstrap_ci,
)
from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from foundation.provenance import content_sha256

PANEL_SCHEMA_VERSION = "cvi.fixed_multievidence_panel.v1"
PANEL_BUNDLE_SCHEMA_VERSION = "cvi.fixed_multievidence_panel_bundle.v1"
REPORT_SCHEMA_VERSION = "cvi.fixed_multievidence_evaluation.v1"
REPORT_BUNDLE_SCHEMA_VERSION = "cvi.fixed_multievidence_evaluation_bundle.v1"
F5_ARCHITECTURE = "cls_residual_v5"
F5_TRAINING_SEED = 42
SPLIT_COMMITMENT = "cvi.fixed_multievidence.publisher-test.v1:DEV"
DEV_FRACTION = 0.30
MINIMUM_DEV_IDENTITIES = 40
MINIMUM_EVAL_IDENTITIES = 100
FRAMES_PER_WINDOW = 5
METHODS = (
    "A0_frozen_dinov2_K5",
    "F5_cls_residual_K5",
    "N3_consistency_weak_nose_K5",
)
FUSIONS = {
    "A0_plus_F5": METHODS[:2],
    "A0_plus_N3": (METHODS[0], METHODS[2]),
    "F5_plus_N3": METHODS[1:],
    "A0_plus_F5_plus_N3": METHODS,
}
ALL_METHODS = (*METHODS, *FUSIONS)
METRICS = ("Rank-1", "Rank-5", "MRR")
LIMITATIONS = (
    "SAME_VIDEO_TRACK_GALLERY_AND_QUERY",
    "PUBLISHER_TEST_EXPOSED_DIAGNOSTIC",
    "PRIOR_PUBLISHER_TEST_EXPOSURE",
    "CLOSED_SET_ONLY_NO_UNKNOWN_REJECTION",
    "TRACK_IDENTITIES_NOT_LIFELONG_DOG_IDENTITIES",
    "WEAK_NOSE_ROI_INPUT_DIFFERS_FROM_NATIVE_NOSE_TRAINING_INPUT",
    "NOT_FINAL_EVALUATION",
    "NO_BIOMETRIC_OR_OPEN_SET_CLAIM",
)
_EXPECTED_N3_IDENTITY_LISTS = {
    "parent_seen_yt",
    "parent_seen_native_ssl_train",
    "ssl_train",
    "dev",
    "eval",
}
_EXPECTED_EXPOSURE_LISTS = {
    "f5_train",
    "f5_model_selection",
    *(f"n3_{name}" for name in _EXPECTED_N3_IDENTITY_LISTS),
}
_PUBLISHER_FRAME_RE = re.compile(
    r"^(?:YT-BB-dog/)?YT-BB-Dog/test/(?P<identity>[0-9]+)/"
    r"(?P=identity)_(?P<frame>[0-9]+)\.jpg$"
)
_ZSCORE_EPSILON = 1e-8
_PANEL_CODE_PATHS = (
    "experiments/fixed_multievidence.py",
    "embedding/methods/face/checkpoint.py",
    "parsing/roi_manifest.py",
    "embedding/methods/nose/training/embedding_consistency_training.py",
    "workflows/train_roi_face_reid.py",
    "workflows/build_fixed_multievidence_panel.py",
)
_PRE_TRAINING_OWNERSHIP_PANEL_CODE_PATHS = tuple(
    "parsing/nose_region/embedding_consistency_training.py"
    if path == "embedding/methods/nose/training/embedding_consistency_training.py"
    else path
    for path in _PANEL_CODE_PATHS
)
_PRE_EMBEDDING_PANEL_CODE_PATHS = tuple(
    path.replace("embedding/methods/", "identity_methods/", 1)
    if path.startswith("embedding/methods/")
    else path
    for path in _PRE_TRAINING_OWNERSHIP_PANEL_CODE_PATHS
)
_LEGACY_PANEL_CODE_PATHS = tuple(
    path.replace("parsing/", "localization/", 1)
    if path.startswith("parsing/")
    else path
    for path in _PRE_EMBEDDING_PANEL_CODE_PATHS
)
_EVALUATION_CODE_PATHS = (
    "contracts/artifact_manifest.py",
    "evaluation/retrieval.py",
    "experiments/fixed_multievidence.py",
    "embedding/methods/appearance/__init__.py",
    "embedding/methods/face/checkpoint.py",
    "embedding/methods/face/dataset.py",
    "embedding/methods/face/residual_model.py",
    "workflows/evaluate_fixed_multievidence.py",
)


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_uuid5(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical UUIDv5 string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UUIDv5 string") from error
    if parsed.version != 5 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv5 string")
    return value


def file_sha256(path: Path) -> str:
    """Hash one stable regular non-symlink file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {source}")
    before = source.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    after = source.stat(follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"input changed while hashing: {source}")
    return digest.hexdigest()


def _stable_file_bytes(path: Path, name: str) -> tuple[bytes, str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    before = source.stat(follow_symlinks=False)
    payload = source.read_bytes()
    after = source.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise RuntimeError(f"{name} changed while reading")
    return payload, hashlib.sha256(payload).hexdigest()


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or "\\" in value
        or not path.parts
        or path.parts[0].endswith(":")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical safe relative path")
    return path


def parse_publisher_frame_index(image_path: object) -> int:
    """Parse the frame index only from the canonical publisher-test path."""

    path = _safe_relative_path(image_path, "publisher image_path")
    match = _PUBLISHER_FRAME_RE.fullmatch(path.as_posix())
    if match is None:
        raise ValueError("publisher-test image path does not match the exact YT-BB-Dog layout")
    return int(match["frame"])


def _bound_file(root: Path, relative: object, expected_sha256: object) -> Path:
    pure = _safe_relative_path(relative, "artifact path")
    digest = _require_sha256(expected_sha256, "artifact SHA-256")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("bound artifact does not exist or is unsafe") from exc
    if (
        not resolved.is_relative_to(resolved_root)
        or resolved.relative_to(resolved_root).as_posix() != pure.as_posix()
        or resolved.is_symlink()
        or not resolved.is_file()
    ):
        raise ValueError("bound artifact must be a regular file under its root")
    if file_sha256(resolved) != digest:
        raise ValueError("bound artifact SHA-256 differs")
    return resolved


def read_bound_rgb(root: Path, relative: object, expected_sha256: object) -> Image.Image:
    """Read one exact bound image without trusting a second pathname read."""

    path = _bound_file(root, relative, expected_sha256)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("bound artifact changed between validation and image read")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
            image.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("bound artifact is not a valid image") from exc
    return image


def reconstruct_f5_training_split(
    manifest: Mapping[str, Any], *, seed: int = F5_TRAINING_SEED
) -> dict[str, Any]:
    """Reproduce the exact current ``train_roi_face_reid.py`` split logic."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("F5 training seed must be an integer")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise TypeError("F5 training ROI manifest records differ")
    selected: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("F5 training ROI record differs")
        if not record.get("face_crop_path") or not record.get("registered_identity_id"):
            continue
        sample_id = record.get("sample_id")
        quality = record.get("face_quality")
        if not isinstance(sample_id, str) or not isinstance(quality, Mapping):
            raise TypeError("F5 training face record binding differs")
        overall = quality.get("overall")
        if (
            isinstance(overall, bool)
            or not isinstance(overall, (int, float))
            or not math.isfinite(overall)
        ):
            raise ValueError("F5 training face quality differs")
        previous = selected.get(sample_id)
        if previous is None or overall > previous["face_quality"]["overall"]:
            selected[sample_id] = record
    values = tuple(selected.values())
    identities = sorted({record["registered_identity_id"] for record in values})
    np.random.RandomState(seed).shuffle(identities)
    split = int(0.8 * len(identities))
    train_candidates = set(identities[:split])
    counts = Counter(record["registered_identity_id"] for record in values)
    train_ids = {
        identity for identity, count in counts.items() if count >= 4
    } & train_candidates
    dev_ids = {
        identity for identity in identities[split:] if counts[identity] >= 2
    }
    train_records = tuple(
        record for record in values if record["registered_identity_id"] in train_ids
    )
    dev_records = tuple(
        record for record in values if record["registered_identity_id"] in dev_ids
    )
    payload = {
        "seed": seed,
        "train_identities": sorted(train_ids),
        "dev_identities": sorted(dev_ids),
        "train_samples": sorted(record["sample_id"] for record in train_records),
        "dev_samples": sorted(record["sample_id"] for record in dev_records),
    }
    return {**payload, "training_split_sha256": content_sha256(payload)}


def n3_identity_lists(lineage: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return the exact current N3 parent/SSL/DEV/EVAL identity lists."""

    try:
        identities = lineage["bindings"]["splits"]["identity_lists"]
    except (KeyError, TypeError) as exc:
        raise ValueError("N3 lineage identity lists are missing") from exc
    if not isinstance(identities, Mapping) or set(identities) != _EXPECTED_N3_IDENTITY_LISTS:
        raise ValueError("N3 lineage identity-list schema differs")
    result: dict[str, list[str]] = {}
    for name in sorted(_EXPECTED_N3_IDENTITY_LISTS):
        values = identities[name]
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"N3 lineage identity list {name} differs")
        result[name] = list(values)
    return result


def partition_identities(identities: Sequence[str]) -> tuple[list[str], list[str]]:
    """Apply the frozen identity-only hash commitment without model outputs."""

    values = list(identities)
    if values != sorted(set(values)):
        raise ValueError("panel identities must be sorted and unique")
    dev: list[str] = []
    evaluation: list[str] = []
    threshold = int(DEV_FRACTION * (1 << 256))
    for identity in values:
        if not isinstance(identity, str) or not identity:
            raise ValueError("panel identity must be non-empty text")
        digest = hashlib.sha256(f"{SPLIT_COMMITMENT}:{identity}".encode("ascii")).digest()
        (dev if int.from_bytes(digest, "big") < threshold else evaluation).append(identity)
    return dev, evaluation


def _complete_evidence(record: Mapping[str, Any]) -> bool:
    return all(
        record.get(name) is not None
        for name in (
            "image_path",
            "image_sha256",
            "quality",
            "face_crop_path",
            "face_crop_sha256",
            "face_quality",
            "weak_nose_crop_path",
            "weak_nose_crop_sha256",
        )
    )


def _panel_record(
    record: Mapping[str, Any], *, frame_index: int, partition: str, role: str
) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "instance_id": record["instance_id"],
        "registered_identity_id": record["registered_identity_id"],
        "partition": partition,
        "window_role": role,
        "publisher_frame_index": frame_index,
        "split_role": record["split_role"],
        "capture_group_id": record["capture_group_id"],
        "capture_group_kind": record["capture_group_kind"],
        "source": {
            "path": record["image_path"],
            "sha256": record["image_sha256"],
            "quality": record["quality"],
        },
        "face": {
            "path": record["face_crop_path"],
            "sha256": record["face_crop_sha256"],
            "quality": record["face_quality"],
        },
        "weak_nose": {
            "path": record["weak_nose_crop_path"],
            "sha256": record["weak_nose_crop_sha256"],
            "quality": record["quality"],
            "quality_semantics": "DOG_ROI_QUALITY_PROXY_NO_NOSE_SPECIFIC_SCORE",
        },
    }


def select_fixed_panel_population(
    records: Sequence[Mapping[str, Any]],
    *,
    f5_train_identities: Sequence[str],
    f5_model_selection_identities: Sequence[str],
    n3_lists: Mapping[str, Sequence[str]],
    minimum_dev_identities: int = MINIMUM_DEV_IDENTITIES,
    minimum_eval_identities: int = MINIMUM_EVAL_IDENTITIES,
) -> dict[str, Any]:
    """Select exact publisher-test K5 windows using metadata only."""

    if (
        isinstance(minimum_dev_identities, bool)
        or not isinstance(minimum_dev_identities, int)
        or minimum_dev_identities < 1
        or isinstance(minimum_eval_identities, bool)
        or not isinstance(minimum_eval_identities, int)
        or minimum_eval_identities < 1
    ):
        raise ValueError("minimum panel identity counts must be positive integers")
    if set(n3_lists) != _EXPECTED_N3_IDENTITY_LISTS:
        raise ValueError("N3 exposure lists differ")
    f5_train = set(f5_train_identities)
    f5_dev = set(f5_model_selection_identities)
    n3_exposed = set().union(*(set(n3_lists[name]) for name in n3_lists))
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    all_identity_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ignored_counts = {"non_test": 0, "unregistered": 0, "incomplete_evidence": 0}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("ROI panel candidate record differs")
        if record.get("split_role") != "test":
            ignored_counts["non_test"] += 1
            continue
        identity = record.get("registered_identity_id")
        if not isinstance(identity, str) or not identity:
            ignored_counts["unregistered"] += 1
            continue
        frame = parse_publisher_frame_index(record.get("image_path"))
        all_identity_rows[identity].append(record)
        if not _complete_evidence(record):
            ignored_counts["incomplete_evidence"] += 1
            continue
        grouped[identity].append((frame, record))

    reasons: dict[str, list[str]] = {}
    for identity in sorted(all_identity_rows):
        identity_reasons: list[str] = []
        rows = all_identity_rows[identity]
        groups = {
            (row.get("capture_group_kind"), row.get("capture_group_id")) for row in rows
        }
        if (
            len(groups) != 1
            or next(iter(groups))[0] != "VIDEO_TRACK"
            or not isinstance(next(iter(groups))[1], str)
            or not next(iter(groups))[1]
        ):
            identity_reasons.append("NOT_EXACTLY_ONE_VIDEO_TRACK_CAPTURE_GROUP")
        complete = grouped.get(identity, [])
        frames = [frame for frame, _ in complete]
        source_paths = [row["image_path"] for _, row in complete]
        sample_ids = [row["sample_id"] for _, row in complete]
        if (
            len(frames) != len(set(frames))
            or len(source_paths) != len(set(source_paths))
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise ValueError(f"publisher-test identity repeats a source frame: {identity}")
        if len(complete) < 2 * FRAMES_PER_WINDOW:
            identity_reasons.append("FEWER_THAN_TEN_COMPLETE_UNIQUE_SOURCE_FRAMES")
        if identity in f5_train:
            identity_reasons.append("F5_TRAIN_IDENTITY_EXPOSURE")
        if identity in f5_dev:
            identity_reasons.append("F5_MODEL_SELECTION_IDENTITY_EXPOSURE")
        if identity in n3_exposed:
            identity_reasons.append("N3_PARENT_SSL_DEV_OR_EVAL_IDENTITY_EXPOSURE")
        if identity_reasons:
            reasons[identity] = sorted(identity_reasons)

    eligible = sorted(set(all_identity_rows) - set(reasons))
    dev_ids, eval_ids = partition_identities(eligible)
    if len(dev_ids) < minimum_dev_identities or len(eval_ids) < minimum_eval_identities:
        raise ValueError(
            "fixed panel is too small after exposure exclusions: "
            f"DEV={len(dev_ids)} EVAL={len(eval_ids)}"
        )
    partition_by_identity = {
        **{identity: "DEV" for identity in dev_ids},
        **{identity: "EVAL" for identity in eval_ids},
    }
    selected_records: list[dict[str, Any]] = []
    for identity in eligible:
        ordered = sorted(
            grouped[identity], key=lambda item: (item[0], item[1]["sample_id"])
        )
        gallery = ordered[:FRAMES_PER_WINDOW]
        query = ordered[-FRAMES_PER_WINDOW:]
        gallery_frames = {frame for frame, _ in gallery}
        query_frames = {frame for frame, _ in query}
        gallery_samples = {row["sample_id"] for _, row in gallery}
        query_samples = {row["sample_id"] for _, row in query}
        if gallery_frames & query_frames or gallery_samples & query_samples:
            raise ValueError(f"earliest/latest K5 windows overlap: {identity}")
        for role, window in (("gallery", gallery), ("query", query)):
            selected_records.extend(
                _panel_record(
                    row,
                    frame_index=frame,
                    partition=partition_by_identity[identity],
                    role=role,
                )
                for frame, row in window
            )
    exclusion_counts = Counter(reason for values in reasons.values() for reason in values)
    return {
        "population": {
            "observed_publisher_test_identity_ids": sorted(all_identity_rows),
            "eligible_identity_ids": eligible,
            "dev_identity_ids": dev_ids,
            "eval_identity_ids": eval_ids,
            "dev_identity_ids_sha256": content_sha256({"identity_ids": dev_ids}),
            "eval_identity_ids_sha256": content_sha256({"identity_ids": eval_ids}),
            "minimum_dev_identities": minimum_dev_identities,
            "minimum_eval_identities": minimum_eval_identities,
        },
        "exclusions": {
            "ignored_record_counts": ignored_counts,
            "identity_reasons": reasons,
            "identity_reason_counts": dict(sorted(exclusion_counts.items())),
        },
        "records": selected_records,
    }


def _identity_lists_binding(
    f5_split: Mapping[str, Any], n3_lists_value: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    lists = {
        "f5_train": list(f5_split["train_identities"]),
        "f5_model_selection": list(f5_split["dev_identities"]),
        **{f"n3_{name}": list(n3_lists_value[name]) for name in sorted(n3_lists_value)},
    }
    return {
        "lists": lists,
        "sha256s": {
            name: content_sha256({"identity_ids": values})
            for name, values in lists.items()
        },
    }


def _document_binding(path: Path, document: Any) -> dict[str, Any]:
    return {
        "path": os.fspath(path),
        "raw_sha256": document.raw_sha256,
        "content_sha256": document.canonical_payload_sha256,
        "byte_size": document.byte_size,
    }


def _code_sha256s(repository: Path, paths: Sequence[str]) -> dict[str, str]:
    from contracts.source_provenance import build_source_provenance

    provenance = build_source_provenance(repository / relative for relative in paths)
    return {
        row["relative_path"]: row["content_sha256"]
        for row in provenance["code_source_files"]
    }


def _load_f5_checkpoint(
    checkpoint_path: Path,
    training_manifest: Mapping[str, Any],
    *,
    repository: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    import torch

    from embedding.methods.face.checkpoint import (
        expected_faceid_contract_for_checkpoint,
        validate_checkpoint_structure,
    )

    payload, observed_sha256 = _stable_file_bytes(checkpoint_path, "F5 checkpoint")
    if observed_sha256 != _require_sha256(expected_sha256, "F5 checkpoint SHA-256"):
        raise ValueError("F5 checkpoint SHA-256 differs from the external binding")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    expected_contract = expected_faceid_contract_for_checkpoint(
        checkpoint["faceid_contract"],
        repository,
        architecture=F5_ARCHITECTURE,
    )
    checkpoint_train_ids = validate_checkpoint_structure(
        checkpoint, expected_faceid_contract=expected_contract
    )
    split = reconstruct_f5_training_split(training_manifest)
    if list(checkpoint_train_ids) != split["train_identities"]:
        raise ValueError("F5 checkpoint identities differ from the reconstructed split")
    if checkpoint["training_split_sha256"] != split["training_split_sha256"]:
        raise ValueError("F5 checkpoint training split digest differs")
    if checkpoint["training_roi_manifest_sha256"] != content_sha256(training_manifest):
        raise ValueError("F5 checkpoint training ROI manifest binding differs")
    return checkpoint, split, observed_sha256


def _load_n3_lineage(path: Path, expected_content_sha256: str | None = None):
    from embedding.methods.nose.training.embedding_consistency_training import (
        validate_lineage_manifest,
    )

    document = read_strict_json_document(path)
    if (
        expected_content_sha256 is not None
        and document.canonical_payload_sha256
        != _require_sha256(expected_content_sha256, "N3 lineage content SHA-256")
    ):
        raise ValueError("N3 lineage content SHA-256 differs from the external pin")
    validate_lineage_manifest(document.payload, path.parent.resolve(strict=True))
    return document, n3_identity_lists(document.payload)


def _read_roi_bundle(path: Path, expected_content_sha256: str | None = None):
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if (
        expected_content_sha256 is not None
        and document.canonical_payload_sha256
        != _require_sha256(expected_content_sha256, "ROI bundle content SHA-256")
    ):
        raise ValueError("ROI bundle content SHA-256 differs from the external pin")
    from parsing.roi_manifest import validate_roi_manifest_bundle

    manifest = validate_roi_manifest_bundle(document.payload, root=path.parent)
    return document, manifest


def validate_panel_bundle(bundle: object) -> dict[str, Any]:
    """Validate the exact fixed-panel bundle and frozen K5 structure."""

    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        "panel_sha256",
        "panel",
    }:
        raise ValueError("fixed panel bundle fields differ")
    if bundle["schema_version"] != PANEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError("fixed panel bundle schema differs")
    _require_sha256(bundle["panel_sha256"], "panel_sha256")
    panel = bundle["panel"]
    expected = {
        "schema_version",
        "status",
        "interpretation",
        "protocol",
        "input_bindings",
        "exposure_identity_lists",
        "population",
        "exclusions",
        "records",
        "code_sha256s",
    }
    if not isinstance(panel, dict) or set(panel) != expected:
        raise ValueError("fixed panel fields differ")
    if content_sha256(panel) != bundle["panel_sha256"]:
        raise ValueError("fixed panel digest differs")
    if (
        panel["schema_version"] != PANEL_SCHEMA_VERSION
        or panel["status"] != "FROZEN_EXPOSED_PUBLISHER_TEST_DIAGNOSTIC_PANEL"
    ):
        raise ValueError("fixed panel schema or status differs")
    protocol = panel["protocol"]
    if protocol != {
        "population_source": "PUBLISHER_TEST_ROI_MANIFEST_ONLY",
        "selection_uses_model_outputs": False,
        "split_commitment": SPLIT_COMMITMENT,
        "dev_fraction": DEV_FRACTION,
        "gallery_selection": "EARLIEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
        "query_selection": "LATEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
        "frames_per_window": FRAMES_PER_WINDOW,
        "same_track_only": True,
        "limitations": list(LIMITATIONS),
    }:
        raise ValueError("fixed panel protocol differs")
    input_bindings = panel["input_bindings"]
    if not isinstance(input_bindings, dict) or set(input_bindings) != {
        "roi_manifest_bundle",
        "roi_manifest_sha256",
        "source_image_root",
        "f5_checkpoint",
        "f5_training_roi_manifest_bundle",
        "f5_training_roi_manifest_sha256",
        "n3_lineage",
    }:
        raise ValueError("fixed panel input binding fields differ")
    for name in ("roi_manifest_bundle", "f5_training_roi_manifest_bundle"):
        binding = input_bindings[name]
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "raw_sha256",
            "content_sha256",
            "byte_size",
        }:
            raise ValueError(f"fixed panel {name} binding differs")
        if not isinstance(binding["path"], str) or not binding["path"]:
            raise ValueError(f"fixed panel {name} path differs")
        _require_sha256(binding["raw_sha256"], f"fixed panel {name} raw SHA-256")
        _require_sha256(
            binding["content_sha256"], f"fixed panel {name} content SHA-256"
        )
        if (
            isinstance(binding["byte_size"], bool)
            or not isinstance(binding["byte_size"], int)
            or binding["byte_size"] < 1
        ):
            raise ValueError(f"fixed panel {name} byte size differs")
    for name in ("roi_manifest_sha256", "f5_training_roi_manifest_sha256"):
        _require_sha256(input_bindings[name], f"fixed panel {name}")
    if (
        not isinstance(input_bindings["source_image_root"], str)
        or not Path(input_bindings["source_image_root"]).is_absolute()
    ):
        raise ValueError("fixed panel source image root differs")
    f5_binding = input_bindings["f5_checkpoint"]
    if not isinstance(f5_binding, dict) or set(f5_binding) != {
        "path",
        "sha256",
        "training_split_sha256",
    }:
        raise ValueError("fixed panel F5 checkpoint binding differs")
    if not isinstance(f5_binding["path"], str) or not f5_binding["path"]:
        raise ValueError("fixed panel F5 checkpoint path differs")
    _require_sha256(f5_binding["sha256"], "fixed panel F5 checkpoint SHA-256")
    _require_sha256(
        f5_binding["training_split_sha256"], "fixed panel F5 training split SHA-256"
    )
    n3_binding = input_bindings["n3_lineage"]
    if not isinstance(n3_binding, dict) or set(n3_binding) != {
        "path",
        "raw_sha256",
        "content_sha256",
        "byte_size",
        "lineage_sha256",
    }:
        raise ValueError("fixed panel N3 lineage binding differs")
    if not isinstance(n3_binding["path"], str) or not n3_binding["path"]:
        raise ValueError("fixed panel N3 lineage path differs")
    for name in ("raw_sha256", "content_sha256", "lineage_sha256"):
        _require_sha256(n3_binding[name], f"fixed panel N3 {name}")
    if (
        isinstance(n3_binding["byte_size"], bool)
        or not isinstance(n3_binding["byte_size"], int)
        or n3_binding["byte_size"] < 1
    ):
        raise ValueError("fixed panel N3 lineage byte size differs")
    population = panel["population"]
    required_population = {
        "observed_publisher_test_identity_ids",
        "eligible_identity_ids",
        "dev_identity_ids",
        "eval_identity_ids",
        "dev_identity_ids_sha256",
        "eval_identity_ids_sha256",
        "minimum_dev_identities",
        "minimum_eval_identities",
    }
    if not isinstance(population, dict) or set(population) != required_population:
        raise ValueError("fixed panel population fields differ")
    dev_ids = population["dev_identity_ids"]
    eval_ids = population["eval_identity_ids"]
    eligible = population["eligible_identity_ids"]
    if (
        dev_ids != sorted(set(dev_ids))
        or eval_ids != sorted(set(eval_ids))
        or set(dev_ids) & set(eval_ids)
        or eligible != sorted([*dev_ids, *eval_ids])
        or population["minimum_dev_identities"] != MINIMUM_DEV_IDENTITIES
        or population["minimum_eval_identities"] != MINIMUM_EVAL_IDENTITIES
        or len(dev_ids) < MINIMUM_DEV_IDENTITIES
        or len(eval_ids) < MINIMUM_EVAL_IDENTITIES
        or partition_identities(eligible) != (dev_ids, eval_ids)
        or population["dev_identity_ids_sha256"]
        != content_sha256({"identity_ids": dev_ids})
        or population["eval_identity_ids_sha256"]
        != content_sha256({"identity_ids": eval_ids})
    ):
        raise ValueError("fixed panel identity partition differs")
    for identity in eligible:
        _require_uuid5(identity, "fixed panel registered identity")
    records = panel["records"]
    if not isinstance(records, list) or len(records) != 10 * len(eligible):
        raise ValueError("fixed panel record count differs")
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_samples: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("fixed panel record differs")
        identity = record.get("registered_identity_id")
        if identity not in set(eligible):
            raise ValueError("fixed panel record identity differs")
        if record.get("split_role") != "test" or record.get("capture_group_kind") != "VIDEO_TRACK":
            raise ValueError("fixed panel record exposure or capture group differs")
        if record.get("partition") != ("DEV" if identity in set(dev_ids) else "EVAL"):
            raise ValueError("fixed panel record partition differs")
        sample = record.get("sample_id")
        if not isinstance(sample, str) or not sample or sample in seen_samples:
            raise ValueError("fixed panel sample IDs must be unique")
        seen_samples.add(sample)
        if parse_publisher_frame_index(record.get("source", {}).get("path")) != record.get(
            "publisher_frame_index"
        ):
            raise ValueError("fixed panel publisher frame binding differs")
        for branch in ("source", "face", "weak_nose"):
            artifact = record.get(branch)
            if not isinstance(artifact, Mapping):
                raise TypeError("fixed panel artifact binding differs")
            _safe_relative_path(artifact.get("path"), f"{branch} path")
            _require_sha256(artifact.get("sha256"), f"{branch} SHA-256")
        by_identity[identity].append(record)
    for identity in eligible:
        rows = by_identity[identity]
        groups = {(row["capture_group_kind"], row["capture_group_id"]) for row in rows}
        gallery = [row for row in rows if row["window_role"] == "gallery"]
        query = [row for row in rows if row["window_role"] == "query"]
        if (
            len(groups) != 1
            or len(gallery) != FRAMES_PER_WINDOW
            or len(query) != FRAMES_PER_WINDOW
            or [row["publisher_frame_index"] for row in gallery]
            != sorted(row["publisher_frame_index"] for row in gallery)
            or [row["publisher_frame_index"] for row in query]
            != sorted(row["publisher_frame_index"] for row in query)
            or {row["sample_id"] for row in gallery} & {row["sample_id"] for row in query}
            or max(row["publisher_frame_index"] for row in gallery)
            >= min(row["publisher_frame_index"] for row in query)
        ):
            raise ValueError("fixed panel K5 windows differ")
    exposure = panel["exposure_identity_lists"]
    if not isinstance(exposure, dict) or set(exposure) != {"lists", "sha256s"}:
        raise ValueError("fixed panel exposure binding differs")
    if (
        set(exposure["lists"]) != _EXPECTED_EXPOSURE_LISTS
        or set(exposure["sha256s"]) != _EXPECTED_EXPOSURE_LISTS
    ):
        raise ValueError("fixed panel exposure identity-list hashes differ")
    exposed = set()
    for name, values in exposure["lists"].items():
        if values != sorted(set(values)):
            raise ValueError("fixed panel exposure identity list differs")
        for identity in values:
            _require_uuid5(identity, f"fixed panel exposure {name} identity")
        if exposure["sha256s"][name] != content_sha256({"identity_ids": values}):
            raise ValueError("fixed panel exposure identity-list digest differs")
        exposed.update(values)
    if exposed & set(eligible):
        raise ValueError("fixed panel contains an exposed checkpoint or lineage identity")
    code_hashes = panel["code_sha256s"]
    code_paths = set(code_hashes) if isinstance(code_hashes, dict) else set()
    path_families = tuple(
        set(paths)
        for paths in (
            _PANEL_CODE_PATHS,
            _PRE_TRAINING_OWNERSHIP_PANEL_CODE_PATHS,
            _PRE_EMBEDDING_PANEL_CODE_PATHS,
            _LEGACY_PANEL_CODE_PATHS,
        )
    )
    if not any(
        expected.issubset(code_paths)
        and code_paths.isdisjoint(set.union(*(other - expected for other in path_families)))
        for expected in path_families
    ):
        raise ValueError("fixed panel code hashes differ")
    for digest in code_hashes.values():
        _require_sha256(digest, "panel code SHA-256")
    return panel


def validate_panel_exposure(
    panel: Mapping[str, Any],
    *,
    f5_split: Mapping[str, Any],
    n3_lists_value: Mapping[str, Sequence[str]],
) -> None:
    expected = _identity_lists_binding(f5_split, n3_lists_value)
    if panel.get("exposure_identity_lists") != expected:
        raise ValueError("fixed panel exposure lists differ from current external inputs")
    selected = set(panel["population"]["eligible_identity_ids"])
    exposed = set().union(*(set(values) for values in expected["lists"].values()))
    if selected & exposed:
        raise ValueError("fixed panel identity overlaps F5 or N3 exposure")


def _validate_selected_artifacts(
    records: Sequence[Mapping[str, Any]], *, source_root: Path, roi_root: Path
) -> None:
    for record in records:
        read_bound_rgb(source_root, record["source"]["path"], record["source"]["sha256"])
        read_bound_rgb(roi_root, record["face"]["path"], record["face"]["sha256"])
        read_bound_rgb(
            roi_root, record["weak_nose"]["path"], record["weak_nose"]["sha256"]
        )


def build_fixed_panel(
    *,
    roi_manifest_path: Path,
    roi_manifest_sha256: str,
    source_image_root: Path,
    f5_checkpoint_path: Path,
    f5_checkpoint_sha256: str,
    f5_training_roi_manifest_path: Path,
    f5_training_roi_manifest_sha256: str,
    n3_lineage_path: Path,
    n3_lineage_sha256: str,
    output_path: Path,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Build and publish the fixed exposed panel without model inference."""

    repository = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    output = Path(os.path.abspath(os.fspath(output_path)))
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(repository):
        raise ValueError("fixed panel must be written outside the Git repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite fixed panel: {output}")
    roi_document, roi_manifest = _read_roi_bundle(
        roi_manifest_path, roi_manifest_sha256
    )
    training_document, training_manifest = _read_roi_bundle(
        f5_training_roi_manifest_path, f5_training_roi_manifest_sha256
    )
    _, f5_split, observed_checkpoint_sha256 = _load_f5_checkpoint(
        f5_checkpoint_path,
        training_manifest,
        repository=repository,
        expected_sha256=f5_checkpoint_sha256,
    )
    n3_document, n3_lists_value = _load_n3_lineage(
        n3_lineage_path, n3_lineage_sha256
    )
    selection = select_fixed_panel_population(
        roi_manifest["records"],
        f5_train_identities=f5_split["train_identities"],
        f5_model_selection_identities=f5_split["dev_identities"],
        n3_lists=n3_lists_value,
    )
    source_root = source_image_root.resolve(strict=True)
    roi_root = roi_manifest_path.parent.resolve(strict=True)
    _validate_selected_artifacts(
        selection["records"], source_root=source_root, roi_root=roi_root
    )
    panel = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "status": "FROZEN_EXPOSED_PUBLISHER_TEST_DIAGNOSTIC_PANEL",
        "interpretation": (
            "IDENTITY_DISJOINT_DEV_EVAL_CLOSED_SET_SAME_TRACK_EXPOSED_DIAGNOSTIC_"
            "NOT_FINAL_OR_BIOMETRIC_VALIDATION"
        ),
        "protocol": {
            "population_source": "PUBLISHER_TEST_ROI_MANIFEST_ONLY",
            "selection_uses_model_outputs": False,
            "split_commitment": SPLIT_COMMITMENT,
            "dev_fraction": DEV_FRACTION,
            "gallery_selection": "EARLIEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
            "query_selection": "LATEST_FIVE_BY_PARSED_PUBLISHER_FRAME_INDEX",
            "frames_per_window": FRAMES_PER_WINDOW,
            "same_track_only": True,
            "limitations": list(LIMITATIONS),
        },
        "input_bindings": {
            "roi_manifest_bundle": _document_binding(roi_manifest_path, roi_document),
            "roi_manifest_sha256": roi_document.payload["manifest_sha256"],
            "source_image_root": os.fspath(source_root),
            "f5_checkpoint": {
                "path": os.fspath(f5_checkpoint_path),
                "sha256": observed_checkpoint_sha256,
                "training_split_sha256": f5_split["training_split_sha256"],
            },
            "f5_training_roi_manifest_bundle": _document_binding(
                f5_training_roi_manifest_path, training_document
            ),
            "f5_training_roi_manifest_sha256": training_document.payload[
                "manifest_sha256"
            ],
            "n3_lineage": {
                **_document_binding(n3_lineage_path, n3_document),
                "lineage_sha256": n3_document.payload["lineage_sha256"],
            },
        },
        "exposure_identity_lists": _identity_lists_binding(f5_split, n3_lists_value),
        **selection,
        "code_sha256s": _code_sha256s(repository, _PANEL_CODE_PATHS),
    }
    panel = json.loads(json.dumps(panel, allow_nan=False))
    bundle = {
        "schema_version": PANEL_BUNDLE_SCHEMA_VERSION,
        "panel_sha256": content_sha256(panel),
        "panel": panel,
    }
    validate_panel_bundle(bundle)
    write_private_json_bundle(((output, bundle),))
    return bundle


def read_fixed_panel(path: Path, expected_content_sha256: str) -> tuple[Any, dict[str, Any]]:
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    if document.canonical_payload_sha256 != _require_sha256(
        expected_content_sha256, "fixed panel content SHA-256"
    ):
        raise ValueError("fixed panel content SHA-256 differs from the external pin")
    return document, validate_panel_bundle(document.payload)


def _normalize(vector: np.ndarray, context: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if (
        value.ndim != 1
        or not np.isfinite(value).all()
        or not math.isfinite(norm)
        or norm <= 1e-8
    ):
        raise ValueError(f"{context} produced a non-finite or zero-norm embedding")
    return np.asarray(value / norm, dtype=np.float32)


def _prototype(vectors: Sequence[np.ndarray], context: str) -> np.ndarray:
    if len(vectors) != FRAMES_PER_WINDOW:
        raise ValueError(f"{context} requires exactly five frame embeddings")
    mean = np.mean(np.stack([_normalize(vector, context) for vector in vectors]), axis=0)
    return _normalize(mean, context)


def rank_score_rows(scores: np.ndarray, identities: Sequence[str]) -> list[dict[str, Any]]:
    matrix = np.asarray(scores, dtype=np.float64)
    expected = (len(identities), len(identities))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"score matrix must be finite with shape {expected}")
    rows = []
    for index, identity in enumerate(identities):
        order = np.argsort(-matrix[index], kind="stable")
        rank = int(np.flatnonzero(order == index)[0]) + 1
        rows.append(
            {
                "registered_identity_id": identity,
                "rank": rank,
                "Rank-1": float(rank == 1),
                "Rank-5": float(rank <= 5),
                "MRR": 1.0 / rank,
            }
        )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]], gallery_count: int) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "gallery_count": gallery_count,
        **{
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in METRICS
        },
    }


def _row_zscores(scores: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(values) < 2:
        raise ValueError(f"{context} score matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError(f"{context} score matrix must be finite")
    standard_deviation = values.std(axis=1, ddof=0, keepdims=True)
    if np.any(standard_deviation <= _ZSCORE_EPSILON):
        raise ValueError(f"{context} has a near-zero row standard deviation")
    return (values - values.mean(axis=1, keepdims=True)) / standard_deviation


def _simplex_grid(channels: int, resolution: int):
    if channels == 2:
        for first in range(resolution + 1):
            yield np.asarray((first, resolution - first), dtype=np.float64) / resolution
        return
    if channels != 3:
        raise ValueError("simplex search supports two or three branches")
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            yield np.asarray(
                (first, second, resolution - first - second), dtype=np.float64
            ) / resolution


def calibrate_and_evaluate_fusions(
    dev_identities: Sequence[str],
    eval_identities: Sequence[str],
    dev_scores: Mapping[str, np.ndarray],
    eval_scores: Mapping[str, np.ndarray],
    *,
    resolution: int = 20,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Fit row-z-score simplex weights on DEV and apply them once to EVAL."""

    dev_ids = list(dev_identities)
    eval_ids = list(eval_identities)
    if dev_ids != sorted(set(dev_ids)) or eval_ids != sorted(set(eval_ids)):
        raise ValueError("DEV and EVAL identities must be sorted and unique")
    if set(dev_ids) & set(eval_ids):
        raise ValueError("DEV and EVAL identities must be disjoint")
    if min(len(dev_ids), len(eval_ids)) < 2:
        raise ValueError("DEV and EVAL each require at least two identities")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1:
        raise ValueError("fusion resolution must be a positive integer")
    if set(dev_scores) != set(METHODS) or set(eval_scores) != set(METHODS):
        raise ValueError("score matrices must contain exactly A0, F5, and N3")
    outcomes = {
        method: rank_score_rows(eval_scores[method], eval_ids) for method in METHODS
    }
    calibration: dict[str, Any] = {
        "labels_used": "DEV_ONLY",
        "evaluation_labels_used_for_weight_selection": False,
        "row_zscore": {
            "scope": "EACH_QUERY_ROW_WITHIN_PARTITION_GALLERY",
            "standard_deviation": "POPULATION_DDOF_0",
            "near_zero_threshold": _ZSCORE_EPSILON,
        },
        "fusions": {},
    }
    eval_normalized = {
        method: _row_zscores(eval_scores[method], f"EVAL {method}")
        for method in METHODS
    }
    for fusion, branches in FUSIONS.items():
        dev_normalized = {
            branch: _row_zscores(dev_scores[branch], f"DEV {branch}")
            for branch in branches
        }
        best_key: tuple[float, ...] | None = None
        best_weights: np.ndarray | None = None
        best_metrics: dict[str, Any] | None = None
        candidates = 0
        for weights in _simplex_grid(len(branches), resolution):
            fused = sum(
                weight * dev_normalized[branch]
                for branch, weight in zip(branches, weights, strict=True)
            )
            rows = rank_score_rows(fused, dev_ids)
            metrics = summarize_rows(rows, len(dev_ids))
            key = (
                metrics["Rank-1"],
                metrics["MRR"],
                metrics["Rank-5"],
                *weights.tolist(),
            )
            candidates += 1
            if best_key is None or key > best_key:
                best_key = key
                best_weights = weights
                best_metrics = metrics
        assert best_weights is not None and best_metrics is not None
        selected_weights = {
            branch: float(weight)
            for branch, weight in zip(branches, best_weights, strict=True)
        }
        calibration["fusions"][fusion] = {
            "branches": list(branches),
            "resolution": resolution,
            "candidate_count": candidates,
            "objective_lexicographic": ["Rank-1", "MRR", "Rank-5"],
            "tie_break": [f"HIGHER_{branch}_WEIGHT" for branch in branches],
            "selected_weights": selected_weights,
            "selected_dev_metrics": best_metrics,
        }
        fused_eval = sum(
            selected_weights[branch] * eval_normalized[branch] for branch in branches
        )
        outcomes[fusion] = rank_score_rows(fused_eval, eval_ids)
    return calibration, outcomes


def rescue_break_against_a0(
    outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Count paired Rank-1 rescues and breaks relative to A0."""

    if set(outcomes) != set(ALL_METHODS):
        raise ValueError("rescue/break outcomes must contain every fixed-panel method")
    baseline = list(outcomes[METHODS[0]])
    identities = [row["registered_identity_id"] for row in baseline]
    result = {}
    for method in ALL_METHODS[1:]:
        candidate = list(outcomes[method])
        if [row["registered_identity_id"] for row in candidate] != identities:
            raise ValueError("rescue/break rows are not exactly paired")
        rescue = sum(
            left["rank"] > 1 and right["rank"] == 1
            for left, right in zip(baseline, candidate, strict=True)
        )
        broken = sum(
            left["rank"] == 1 and right["rank"] > 1
            for left, right in zip(baseline, candidate, strict=True)
        )
        result[method] = {
            "paired_identity_count": len(identities),
            "rescue_count": rescue,
            "break_count": broken,
            "rescue_fraction": rescue / len(identities),
            "break_fraction": broken / len(identities),
        }
    return result


def build_topology_manifest(
    panel_records: Sequence[Mapping[str, Any]],
    frame_embeddings: Mapping[str, np.ndarray],
    *,
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact topology v1 manifest from normalized frame embeddings."""

    from experiments.identity_topology import (
        FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        validate_identity_topology_manifest,
    )

    bindings = dict(input_bindings)
    if not bindings:
        raise ValueError("topology input bindings must not be empty")

    if set(frame_embeddings) != set(METHODS):
        raise ValueError("topology embeddings must contain exactly A0, F5, and N3")
    rows = []
    for method in METHODS:
        embeddings = np.asarray(frame_embeddings[method])
        if embeddings.ndim != 2 or len(embeddings) != len(panel_records):
            raise ValueError("topology frame embedding shape differs")
        for record, vector in zip(panel_records, embeddings, strict=True):
            normalized = _normalize(vector, f"topology {method}")
            quality = (
                record["face"]["quality"]["overall"]
                if method == METHODS[1]
                else record["source"]["quality"]["overall"]
            )
            rows.append(
                {
                    "sample_token": record["sample_id"],
                    "identity_token": record["registered_identity_id"],
                    "session_token": record["capture_group_id"],
                    "branch": method,
                    "quality": float(quality),
                    "available": True,
                    "embedding": normalized.tolist(),
                }
            )
    manifest = {
        "schema_version": FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        "input_bindings": bindings,
        "input_bindings_sha256": content_sha256(bindings),
        "records": rows,
    }
    validate_identity_topology_manifest(manifest)
    return manifest


def validate_fixed_topology_bindings(
    panel_bundle: Mapping[str, Any],
    topology_manifest: Mapping[str, Any],
    *,
    n3_runtime_manifest_content_sha256: str | None = None,
    n3_onnx_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind fixed-panel topology vectors to the exact panel and N3 artifacts."""

    panel = validate_panel_bundle(panel_bundle)
    from experiments.identity_topology import (
        FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION,
        validate_identity_topology_manifest,
    )

    validate_identity_topology_manifest(topology_manifest)
    if (
        topology_manifest.get("schema_version")
        != FIXED_PANEL_TOPOLOGY_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("fixed-panel topology manifest schema differs")
    bindings = topology_manifest["input_bindings"]
    expected_fields = {
        "panel_bundle_content_sha256",
        "panel_sha256",
        "frozen_dinov2_sha256",
        "f5_checkpoint_sha256",
        "n3_lineage_content_sha256",
        "n3_runtime_manifest_content_sha256",
        "n3_runtime_manifest_raw_sha256",
        "n3_onnx_sha256",
        "execution",
    }
    if set(bindings) != expected_fields:
        raise ValueError("fixed-panel topology input binding fields differ")
    execution = bindings.get("execution")
    if (
        not isinstance(execution, dict)
        or set(execution) != {"device", "n3_device", "batch_size"}
        or execution["device"] not in {"cpu", "cuda"}
        or execution["n3_device"] not in {"cpu", "cuda"}
        or isinstance(execution["batch_size"], bool)
        or not isinstance(execution["batch_size"], int)
        or execution["batch_size"] < 1
    ):
        raise ValueError("fixed-panel topology execution binding differs")
    for name, digest in bindings.items():
        if name == "execution":
            continue
        _require_sha256(digest, f"topology {name}")
    if (
        bindings["panel_bundle_content_sha256"] != content_sha256(panel_bundle)
        or bindings["panel_sha256"] != panel_bundle["panel_sha256"]
        or bindings["f5_checkpoint_sha256"]
        != panel["input_bindings"]["f5_checkpoint"]["sha256"]
        or bindings["n3_lineage_content_sha256"]
        != panel["input_bindings"]["n3_lineage"]["content_sha256"]
    ):
        raise ValueError("fixed-panel topology differs from its panel bindings")
    if (
        n3_runtime_manifest_content_sha256 is not None
        and bindings["n3_runtime_manifest_content_sha256"]
        != _require_sha256(
            n3_runtime_manifest_content_sha256,
            "expected N3 runtime manifest content SHA-256",
        )
    ):
        raise ValueError("fixed-panel topology N3 preprocessing differs")
    if (
        n3_onnx_sha256 is not None
        and bindings["n3_onnx_sha256"]
        != _require_sha256(n3_onnx_sha256, "expected N3 ONNX SHA-256")
    ):
        raise ValueError("fixed-panel topology N3 ONNX differs")
    return bindings


def _score_partition(
    records: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, np.ndarray],
    identities: Sequence[str],
) -> dict[str, np.ndarray]:
    index_by_identity_role: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        index_by_identity_role[
            (record["registered_identity_id"], record["window_role"])
        ].append(index)
    result = {}
    for method in METHODS:
        gallery = []
        query = []
        for identity in identities:
            gallery.append(
                _prototype(
                    [embeddings[method][index] for index in index_by_identity_role[(identity, "gallery")]],
                    f"{method} gallery {identity}",
                )
            )
            query.append(
                _prototype(
                    [embeddings[method][index] for index in index_by_identity_role[(identity, "query")]],
                    f"{method} query {identity}",
                )
            )
        result[method] = compute_cosine_score_matrix(np.stack(query), np.stack(gallery))
    return result


def _metrics_and_intervals(
    outcomes: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = {}
    intervals = {}
    paired = {}
    for method_index, method in enumerate(ALL_METHODS):
        rows = list(outcomes[method])
        metrics[method] = summarize_rows(rows, len(rows))
        bootstrap_rows = [
            {"bootstrap_cluster_id": row["registered_identity_id"], **{name: row[name] for name in METRICS}}
            for row in rows
        ]
        intervals[method] = {
            metric: identity_clustered_bootstrap_ci(
                bootstrap_rows,
                metric=metric,
                resamples=resamples,
                seed=seed + method_index * len(METRICS) + metric_index,
                confidence_level=confidence_level,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    baseline = list(outcomes[METHODS[0]])
    for method_index, method in enumerate(ALL_METHODS[1:]):
        rows = [
            {
                "bootstrap_cluster_id": left["registered_identity_id"],
                **{metric: right[metric] - left[metric] for metric in METRICS},
            }
            for left, right in zip(baseline, outcomes[method], strict=True)
        ]
        paired[method] = {
            metric: identity_clustered_bootstrap_ci(
                rows,
                metric=metric,
                resamples=resamples,
                seed=seed + 10_000 + method_index * len(METRICS) + metric_index,
                confidence_level=confidence_level,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    return metrics, intervals, paired


def _compare_panel_input_binding(observed: Any, expected: Mapping[str, Any], name: str) -> None:
    if (
        observed.raw_sha256 != expected.get("raw_sha256")
        or observed.canonical_payload_sha256 != expected.get("content_sha256")
        or observed.byte_size != expected.get("byte_size")
    ):
        raise ValueError(f"{name} differs from the fixed panel binding")


def evaluate_fixed_panel(
    *,
    panel_path: Path,
    panel_sha256: str,
    roi_manifest_path: Path,
    source_image_root: Path,
    f5_checkpoint_path: Path,
    f5_training_roi_manifest_path: Path,
    n3_lineage_path: Path,
    n3_runtime_manifest_path: Path,
    n3_onnx_path: Path,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
    frozen_model_sha256: str,
    output_path: Path,
    topology_output_path: Path | None = None,
    device: str = "cpu",
    n3_use_cuda: bool = False,
    batch_size: int = 32,
    fusion_resolution: int = 20,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence_level: float = 0.95,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate frozen A0, trained F5, N3, and DEV-only fusions on EVAL."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    for value, name, minimum in (
        (batch_size, "batch_size", 1),
        (fusion_resolution, "fusion_resolution", 1),
        (bootstrap_resamples, "bootstrap_resamples", 1),
        (bootstrap_seed, "bootstrap_seed", 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} differs")
    if not 0.0 < float(bootstrap_confidence_level) < 1.0:
        raise ValueError("bootstrap_confidence_level must be in (0,1)")
    repository = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    output = Path(os.path.abspath(os.fspath(output_path)))
    output = output.parent.resolve(strict=True) / output.name
    if output.is_relative_to(repository):
        raise ValueError("fixed-panel report must be written outside the Git repository")
    outputs = [output]
    topology_output = None
    if topology_output_path is not None:
        raw_topology = Path(os.path.abspath(os.fspath(topology_output_path)))
        topology_output = raw_topology.parent.resolve(strict=True) / raw_topology.name
        if topology_output.is_relative_to(repository):
            raise ValueError("topology manifest must be written outside the Git repository")
        if topology_output.parent != output.parent:
            raise ValueError("report and optional topology manifest must share a directory")
        outputs.append(topology_output)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("refusing to overwrite fixed-panel evaluation output")

    panel_document, panel = read_fixed_panel(panel_path, panel_sha256)
    bindings = panel["input_bindings"]
    roi_document, roi_manifest = _read_roi_bundle(roi_manifest_path)
    training_document, training_manifest = _read_roi_bundle(f5_training_roi_manifest_path)
    _compare_panel_input_binding(
        roi_document, bindings["roi_manifest_bundle"], "ROI manifest bundle"
    )
    _compare_panel_input_binding(
        training_document,
        bindings["f5_training_roi_manifest_bundle"],
        "F5 training ROI manifest bundle",
    )
    checkpoint, f5_split, checkpoint_sha256 = _load_f5_checkpoint(
        f5_checkpoint_path,
        training_manifest,
        repository=repository,
        expected_sha256=bindings["f5_checkpoint"]["sha256"],
    )
    n3_document, n3_lists_value = _load_n3_lineage(n3_lineage_path)
    _compare_panel_input_binding(n3_document, bindings["n3_lineage"], "N3 lineage")
    validate_panel_exposure(panel, f5_split=f5_split, n3_lists_value=n3_lists_value)
    reconstructed = select_fixed_panel_population(
        roi_manifest["records"],
        f5_train_identities=f5_split["train_identities"],
        f5_model_selection_identities=f5_split["dev_identities"],
        n3_lists=n3_lists_value,
    )
    for name in ("population", "exclusions", "records"):
        if panel[name] != reconstructed[name]:
            raise ValueError(f"fixed panel {name} differs from current exact inputs")
    source_root = source_image_root.resolve(strict=True)
    roi_root = roi_manifest_path.parent.resolve(strict=True)

    artifacts = n3_document.payload["artifacts"]
    lineage_root = n3_lineage_path.parent.resolve(strict=True)
    expected_runtime = (lineage_root / artifacts["runtime_manifest"]["path"]).resolve(strict=True)
    expected_onnx = (lineage_root / artifacts["onnx"]["path"]).resolve(strict=True)
    if n3_runtime_manifest_path.resolve(strict=True) != expected_runtime:
        raise ValueError("N3 runtime manifest is not the lineage artifact")
    if n3_onnx_path.resolve(strict=True) != expected_onnx:
        raise ValueError("N3 ONNX is not the lineage artifact")

    import torch

    from contracts.artifact_manifest import (
        ExactOnnxRuntime,
        NoseEmbeddingManifest,
        UsageLane,
        preprocess_image,
    )
    from embedding.methods.appearance import ReceiptBoundDinov2Small
    from embedding.methods.face.checkpoint import (
        file_sha256 as face_file_sha256,
    )
    from embedding.methods.face.checkpoint import (
        normalize_dino_local_artifact_contract,
        validate_checkpoint_runtime_bindings,
    )
    from embedding.methods.face.dataset import prepare_roi_face_input
    from embedding.methods.face.trainer import (
        build_faceid_model,
        load_receipt_bound_frozen_dino,
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device)
    backbone, dino_contract = load_receipt_bound_frozen_dino(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )
    observed_dino_contract = normalize_dino_local_artifact_contract(
        {
            "model_sha256": dino_contract.model_sha256,
            "preprocessor_sha256": dino_contract.preprocessor_sha256,
            "weight_receipt_sha256": dino_contract.weight_receipt_sha256,
            "preprocessor_receipt_sha256": dino_contract.preprocessor_receipt_sha256,
            "config_sha256": dino_contract.config_sha256,
            "weight_source_contract_sha256": dino_contract.weight_source.contract_sha256,
            "preprocessor_source_contract_sha256": dino_contract.preprocessor_source.contract_sha256,
        }
    )
    validate_checkpoint_runtime_bindings(
        checkpoint,
        observed_dino_local_artifact_contract=observed_dino_contract,
        observed_weight_intake_bundle_sha256=face_file_sha256(weight_intake_bundle),
        observed_preprocessor_intake_bundle_sha256=face_file_sha256(
            preprocessor_intake_bundle
        ),
    )
    f5 = build_faceid_model(
        backbone, dino_contract, architecture=F5_ARCHITECTURE
    ).to(torch_device)
    f5.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    f5.quality_head.load_state_dict(checkpoint["quality_head_state_dict"], strict=True)
    f5.eval()
    a0 = ReceiptBoundDinov2Small(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
        device=device,
        max_batch_size=batch_size,
    )
    if a0.model_sha256 != _require_sha256(frozen_model_sha256, "frozen model SHA-256"):
        raise ValueError("frozen A0 model SHA-256 differs from the external pin")
    if a0.model_sha256 != dino_contract.model_sha256:
        raise ValueError("A0 and F5 frozen DINO bindings differ")
    n3_runtime_document = read_strict_json_document(n3_runtime_manifest_path)
    if n3_runtime_document.raw_sha256 != artifacts["runtime_manifest"]["sha256"]:
        raise ValueError("N3 runtime manifest differs from the lineage artifact hash")
    n3_manifest = NoseEmbeddingManifest.from_dict(n3_runtime_document.payload)
    if n3_manifest.license.usage_lane != UsageLane.RESEARCH_ONLY:
        raise ValueError("N3 runtime must remain research-only")
    n3_onnx_sha256 = file_sha256(n3_onnx_path)
    if n3_onnx_sha256 != artifacts["onnx"]["sha256"]:
        raise ValueError("N3 ONNX differs from the lineage artifact hash")
    n3_runtime = ExactOnnxRuntime(n3_onnx_path, n3_manifest, use_cuda=n3_use_cuda)

    panel_records = panel["records"]
    source_embeddings: list[np.ndarray] = []
    for offset in range(0, len(panel_records), batch_size):
        batch = panel_records[offset : offset + batch_size]
        images = [
            read_bound_rgb(source_root, row["source"]["path"], row["source"]["sha256"])
            for row in batch
        ]
        source_embeddings.extend(
            _normalize(vector, "A0") for vector in a0.extract_batch(images)
        )

    roi_by_instance = {record["instance_id"]: record for record in roi_manifest["records"]}
    face_records = tuple(roi_by_instance[row["instance_id"]] for row in panel_records)
    face_embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(panel_records), batch_size):
            prepared = [
                prepare_roi_face_input(
                    read_bound_rgb(
                        roi_root,
                        panel_record["face"]["path"],
                        panel_record["face"]["sha256"],
                    ),
                    face_record,
                    align=False,
                )
                for panel_record, face_record in zip(
                    panel_records[offset : offset + batch_size],
                    face_records[offset : offset + batch_size],
                    strict=True,
                )
            ]
            rgb = torch.stack([item["rgb"] for item in prepared]).to(
                device=torch_device, dtype=torch.float32
            )
            landmarks = torch.stack([item["landmarks"] for item in prepared]).to(
                device=torch_device, dtype=torch.float32
            )
            with torch.autocast(device_type=torch_device.type, enabled=torch_device.type == "cuda"):
                values = f5(rgb, landmarks)["embedding"]
            face_embeddings.extend(
                _normalize(vector, "F5") for vector in values.float().cpu().numpy()
            )

    nose_embeddings: list[np.ndarray] = []
    for offset in range(0, len(panel_records), batch_size):
        for row in panel_records[offset : offset + batch_size]:
            image = read_bound_rgb(
                roi_root, row["weak_nose"]["path"], row["weak_nose"]["sha256"]
            )
            nose_embeddings.append(
                _normalize(n3_runtime.run(preprocess_image(image, n3_manifest))[0], "N3")
            )
    frame_embeddings = {
        METHODS[0]: np.stack(source_embeddings),
        METHODS[1]: np.stack(face_embeddings),
        METHODS[2]: np.stack(nose_embeddings),
    }
    dev_ids = panel["population"]["dev_identity_ids"]
    eval_ids = panel["population"]["eval_identity_ids"]
    dev_scores = _score_partition(panel_records, frame_embeddings, dev_ids)
    eval_scores = _score_partition(panel_records, frame_embeddings, eval_ids)
    calibration, outcomes = calibrate_and_evaluate_fusions(
        dev_ids,
        eval_ids,
        dev_scores,
        eval_scores,
        resolution=fusion_resolution,
    )
    metrics, intervals, paired_intervals = _metrics_and_intervals(
        outcomes,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=float(bootstrap_confidence_level),
    )
    per_identity = [
        {
            "registered_identity_id": identity,
            "method_outcomes": {
                method: outcomes[method][index] for method in ALL_METHODS
            },
        }
        for index, identity in enumerate(eval_ids)
    ]
    topology_manifest = (
        None
        if topology_output is None
        else build_topology_manifest(
            panel_records,
            frame_embeddings,
            input_bindings={
                "panel_bundle_content_sha256": panel_document.canonical_payload_sha256,
                "panel_sha256": panel_document.payload["panel_sha256"],
                "frozen_dinov2_sha256": a0.model_sha256,
                "f5_checkpoint_sha256": checkpoint_sha256,
                "n3_lineage_content_sha256": n3_document.canonical_payload_sha256,
                "n3_runtime_manifest_content_sha256": (
                    n3_runtime_document.canonical_payload_sha256
                ),
                "n3_runtime_manifest_raw_sha256": n3_runtime_document.raw_sha256,
                "n3_onnx_sha256": n3_onnx_sha256,
                "execution": {
                    "device": device,
                    "n3_device": "cuda" if n3_use_cuda else "cpu",
                    "batch_size": batch_size,
                },
            },
        )
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS_EXPOSED_SAME_TRACK_CLOSED_SET_DIAGNOSTIC",
        "interpretation": (
            "EXPOSED_PUBLISHER_TEST_SAME_TRACK_CLOSED_SET_DIAGNOSTIC_"
            "NOT_FINAL_NOT_OPEN_SET_NOT_BIOMETRIC_VALIDATION"
        ),
        "methods": {
            METHODS[0]: "receipt-bound frozen DINOv2-small on publisher source crops",
            METHODS[1]: "trained raw-crop F5 face encoder on bound face crops",
            METHODS[2]: "research-only N3 ONNX on bound weak-nose crops",
            **{
                name: "DEV-only row-z-score simplex fusion of " + ", ".join(branches)
                for name, branches in FUSIONS.items()
            },
        },
        "execution": {
            "device": device,
            "n3_device": "cuda" if n3_use_cuda else "cpu",
            "batch_size": batch_size,
        },
        "protocol": {
            "gallery_query": "UNWEIGHTED_L2_NORMALIZED_K5_PROTOTYPES",
            "retrieval": "EXHAUSTIVE_COSINE_CLOSED_SET_ONE_PROTOTYPE_PER_IDENTITY",
            "fusion_selection": "DEV_ONLY",
            "evaluation_application": "FROZEN_WEIGHTS_APPLIED_ONCE_TO_IDENTITY_DISJOINT_EVAL",
            "bootstrap": {
                "cluster_unit": "registered_identity_id",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "confidence_level": float(bootstrap_confidence_level),
            },
            "same_track_only": True,
            "prior_publisher_test_exposure": True,
            "closed_set": True,
            "not_final": True,
            "open_set_claim": False,
            "biometric_claim": False,
            "limitations": list(LIMITATIONS),
        },
        "population": panel["population"],
        "input_bindings": {
            "panel": _document_binding(panel_path, panel_document),
            "panel_sha256": panel_document.payload["panel_sha256"],
            "roi_manifest_bundle": _document_binding(roi_manifest_path, roi_document),
            "f5_checkpoint_sha256": checkpoint_sha256,
            "f5_training_roi_manifest_bundle": _document_binding(
                f5_training_roi_manifest_path, training_document
            ),
            "n3_lineage": _document_binding(n3_lineage_path, n3_document),
            "n3_runtime_manifest": _document_binding(
                n3_runtime_manifest_path, n3_runtime_document
            ),
            "n3_onnx_sha256": n3_onnx_sha256,
            "a0_model_sha256": a0.model_sha256,
            "a0_preprocessor_sha256": a0.preprocessor_sha256,
            "a0_weight_receipt_sha256": a0.weight_receipt_sha256,
            "a0_preprocessor_receipt_sha256": a0.preprocessor_receipt_sha256,
            "frame_embedding_sha256s": {
                method: hashlib.sha256(
                    np.ascontiguousarray(values, dtype=np.float32).tobytes()
                ).hexdigest()
                for method, values in frame_embeddings.items()
            },
        },
        "code_sha256s": _code_sha256s(
            repository,
            (
                *_EVALUATION_CODE_PATHS,
                *(("experiments/identity_topology.py",) if topology_manifest is not None else ()),
            ),
        ),
        "calibration": calibration,
        "evaluation": {
            "metrics": metrics,
            "identity_bootstrap_cis": intervals,
            "paired_delta_vs_A0_bootstrap_cis": paired_intervals,
            "rescue_break_vs_A0": rescue_break_against_a0(outcomes),
            "per_identity": per_identity,
        },
        "topology_manifest": {
            "emitted": topology_manifest is not None,
            "path": None if topology_output is None else os.fspath(topology_output),
            "content_sha256": (
                None if topology_manifest is None else content_sha256(topology_manifest)
            ),
            "session_token_binding": "capture_group_id",
            "same_track_only": True,
            "cross_session_inference_allowed": False,
        },
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {
        "schema_version": REPORT_BUNDLE_SCHEMA_VERSION,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    publication = [(output, bundle)]
    if topology_output is not None and topology_manifest is not None:
        publication.append((topology_output, topology_manifest))
    write_private_json_bundle(tuple(publication))
    return bundle


__all__ = [
    "ALL_METHODS",
    "DEV_FRACTION",
    "F5_TRAINING_SEED",
    "FRAMES_PER_WINDOW",
    "METHODS",
    "MINIMUM_DEV_IDENTITIES",
    "MINIMUM_EVAL_IDENTITIES",
    "PANEL_BUNDLE_SCHEMA_VERSION",
    "PANEL_SCHEMA_VERSION",
    "REPORT_BUNDLE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SPLIT_COMMITMENT",
    "build_fixed_panel",
    "build_topology_manifest",
    "validate_fixed_topology_bindings",
    "calibrate_and_evaluate_fusions",
    "evaluate_fixed_panel",
    "file_sha256",
    "n3_identity_lists",
    "parse_publisher_frame_index",
    "partition_identities",
    "read_bound_rgb",
    "read_fixed_panel",
    "reconstruct_f5_training_split",
    "rescue_break_against_a0",
    "select_fixed_panel_population",
    "validate_panel_bundle",
    "validate_panel_exposure",
]
