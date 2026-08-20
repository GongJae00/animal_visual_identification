"""Extract lossless nose crops and pseudo-masks from original YT dog crops."""

from __future__ import annotations

from archive.root import repository_root as find_repo_root
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image

from shared.contracts.source_provenance import build_offline_tool_provenance
from data.public_sources.public_canine_manifest import (
    YT_DATASET,
    ArchiveReceiptBinding,
    parse_yt_bb_dog,
)
from data.public_sources.public_dataset_receipt_io import read_public_archive_receipt_bundle
from shared.foundation.protected_io import json_document_bytes, read_strict_json_document
from shared.foundation.provenance import content_sha256
from enrollment.registry.identity_registry import (
    compute_identity_token,
    compute_registered_dog_id,
    compute_sample_token,
)
from evaluation.splits.protected_public_split import PublicSplitSourceBundle
from evaluation.splits.split_registry_binding import (
    validate_assignment_receipt_binding,
)
from parsing.export.regions.native_yt import (
    TEACHER_SCHEMA,
    NativeYtSample,
    NestedYtArchive,
    build_manifest_bundle,
    decode_source_image,
    load_localizer_checkpoint,
    predict_localizer,
    process_native_sample,
    validate_manifest_bundle,
)
from parsing.training.regions.sam2_teacher import (
    SOURCE_IMAGE_MANIFEST_SCHEMA,
    validate_source_image_manifest,
    validate_teacher_manifest,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROI_BUNDLE_SCHEMA = "cvi.canid_roi_manifest_bundle.v2"
_ROI_MANIFEST_SCHEMA = "cvi.canid_roi_manifest.v2"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--split-receipt", required=True, type=Path)
    parser.add_argument("--expected-split-receipt-sha256", required=True)
    parser.add_argument("--yt-archive", required=True, type=Path)
    parser.add_argument("--yt-archive-receipt", required=True, type=Path)
    parser.add_argument("--yt-roi-manifest", required=True, type=Path)
    parser.add_argument("--localizer-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-mask-manifest", type=Path)
    parser.add_argument("--materialize-teacher-source-images", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--minimum-detector-confidence", type=float, default=0.5)
    parser.add_argument("--minimum-frontality", type=float, default=0.7)
    parser.add_argument("--minimum-native-short-side", type=int, default=32)
    parser.add_argument("--maximum-mask-uncertainty", type=float, default=0.65)
    return parser.parse_args(argv)


def _regular_bytes(path: Path, name: str) -> bytes:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} does not exist: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        raise RuntimeError(f"{name} changed while being read")
    payload = b"".join(chunks)
    if not payload or len(payload) != before.st_size:
        raise ValueError(f"{name} must be a non-empty stable file")
    return payload


def _require_regular_path(path: Path, name: str) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} does not exist: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be an absolute regular non-symlink file")


def _document(path: Path, name: str):
    _regular_bytes(path, name)
    return read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=20_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=2_000_000,
    )


def _validate_split_inputs(
    assignment: object,
    source_payload: object,
    receipt: object,
    expected_receipt_sha256: str,
) -> tuple[PublicSplitSourceBundle, tuple[dict[str, Any], ...]]:
    _require_sha256(expected_receipt_sha256, "expected split receipt SHA-256")
    if not isinstance(assignment, dict) or assignment.get("schema_version") != "cvi.protected_public_split_assignment.v1" or assignment.get("status") != "PASS_PROTECTED_SPLIT_CONSTRUCTION" or not isinstance(assignment.get("records"), list):
        raise ValueError("protected assignment schema or status differs")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "cvi.protected_public_split_receipt.v3" or receipt.get("status") != "PASS_PROTECTED_SPLIT_CONSTRUCTION":
        raise ValueError("protected split receipt schema or status differs")
    if receipt.get("receipt_sha256") != expected_receipt_sha256 or receipt["receipt_sha256"] != content_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise ValueError("protected split receipt digest differs")
    if receipt.get("assignment_sha256") != content_sha256(assignment):
        raise ValueError("protected split receipt does not bind the assignment")
    if not isinstance(source_payload, dict):
        raise TypeError("source bundle must be an object")
    source = PublicSplitSourceBundle.from_dict(source_payload)
    if receipt.get("source_bundle_sha256") != source.bundle_sha256:
        raise ValueError("protected split receipt does not bind the source bundle")
    source_by_token = {sample.sample_token: sample for sample in source.samples}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in assignment["records"]:
        if not isinstance(raw, dict):
            raise ValueError("protected assignment record must be an object")
        if raw.get("dataset_name") != YT_DATASET or raw.get("source_variant") != "original" or raw.get("identity_role") != "YT_FIT":
            continue
        if raw.get("model_access") != "MODEL_TRAINING" or raw.get("sample_disposition") != "PRIMARY_ORACLE_CROP":
            raise ValueError("YT_FIT source role carries an unsafe assignment disposition")
        token = raw.get("sample_token")
        source_row = source_by_token.get(token)
        if source_row is None or token in seen:
            raise ValueError("YT_FIT assignment/source-bundle sample coverage differs")
        seen.add(token)
        if (
            raw.get("identity_token") != source_row.identity_token
            or source_row.dataset_name != YT_DATASET
            or source_row.source_variant != "original"
            or source_row.region != "DOG_CROP"
            or source_row.original_split != "train"
            or token != compute_sample_token(source_row.source_sample_id)
            or source_row.identity_token != compute_identity_token(source_row.dataset_identity_id)
        ):
            raise ValueError("YT_FIT assignment identity/source binding differs")
        selected.append({"assignment": raw, "source": source_row})
    if not selected:
        raise ValueError("protected assignment contains no original YT_FIT samples")
    return source, tuple(sorted(selected, key=lambda row: row["source"].sample_token))


def _roi_source_hashes(bundle: object) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    if not isinstance(bundle, dict) or set(bundle) != {"schema_version", "manifest_sha256", "manifest"} or bundle["schema_version"] != _ROI_BUNDLE_SCHEMA:
        raise ValueError("YT ROI manifest bundle schema differs")
    _require_sha256(bundle["manifest_sha256"], "YT ROI manifest SHA-256")
    manifest = bundle["manifest"]
    if not isinstance(manifest, dict) or content_sha256(manifest) != bundle["manifest_sha256"] or manifest.get("schema_version") != _ROI_MANIFEST_SCHEMA or manifest.get("dataset_name") != YT_DATASET or not isinstance(manifest.get("records"), list):
        raise ValueError("YT ROI manifest content binding differs")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest["records"]:
        if not isinstance(row, dict):
            raise ValueError("YT ROI record must be an object")
        registered = row.get("registered_identity_id")
        image_path = row.get("image_path")
        if registered is None or not isinstance(image_path, str):
            continue
        name = PurePosixPath(image_path).name
        match = re.fullmatch(r"(\d+)_(\d+)\.(?:jpg|jpeg|png)", name, re.IGNORECASE)
        if match is None:
            continue
        frame = int(match.group(2))
        key = (registered, frame)
        if key in result:
            raise ValueError("YT ROI manifest repeats an identity/frame source")
        _require_sha256(row.get("image_sha256"), "YT ROI source image SHA-256")
        width, height = row.get("image_width"), row.get("image_height")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, height)):
            raise ValueError("YT ROI source dimensions differ")
        result[key] = {
            "source_sha256": row["image_sha256"],
            "width": width,
            "height": height,
            "basename": name,
        }
    return result, bundle["manifest_sha256"]


def _native_samples(
    selected: Sequence[Mapping[str, Any]],
    archive_records: Mapping[str, Any],
    roi_sources: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[NativeYtSample, ...]:
    samples: list[NativeYtSample] = []
    for row in selected:
        source = row["source"]
        record = archive_records.get(source.source_sample_id)
        if record is None:
            raise ValueError("YT_FIT source sample is absent from audited original archive")
        registered = compute_registered_dog_id(source.dataset_identity_id)
        if record.dataset_identity_id != source.dataset_identity_id or record.original_split != "train" or record.source_variant != "original" or record.sequence_id is None or record.container_member_path is None or record.container_member_crc32 is None or record.container_member_uncompressed_bytes is None:
            raise ValueError("audited YT source identity/role/container binding differs")
        metadata = roi_sources.get((registered, source.raw_frame_index))
        if metadata is not None:
            if metadata["basename"] != PurePosixPath(record.member_path).name:
                raise ValueError("YT ROI source path differs from audited archive member")
            expected_hash = str(metadata["source_sha256"])
        else:
            expected_hash = None
        samples.append(NativeYtSample(
            sample_token=source.sample_token,
            identity_token=source.identity_token,
            registered_dog_id=registered,
            source_sample_id=source.source_sample_id,
            sequence_token=source.sequence_token,
            track_token=source.sequence_token,
            frame_index=source.raw_frame_index,
            source_role="YT_FIT",
            member_path=record.member_path,
            member_crc32=record.member_crc32,
            member_uncompressed_bytes=record.member_uncompressed_bytes,
            container_member_path=record.container_member_path,
            container_member_crc32=record.container_member_crc32,
            container_member_uncompressed_bytes=record.container_member_uncompressed_bytes,
            expected_source_sha256=expected_hash,
            roi_metadata_available=metadata is not None,
        ))
    return tuple(samples)


def _teacher_records(path: Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    document = _document(path, "teacher mask manifest")
    payload = document.payload
    if not isinstance(payload, dict) or payload.get("schema_version") != TEACHER_SCHEMA or not isinstance(payload.get("records"), list):
        raise ValueError("teacher mask manifest schema differs")
    root = path.parent.resolve(strict=True)
    if set(payload) != {"schema_version", "records"}:
        validate_teacher_manifest(payload, root=root)
        result: dict[str, dict[str, Any]] = {}
        for row in payload["records"]:
            retained = {
                "sample_token": row["sample_token"],
                "source_sha256": row["source_sha256"],
                "mask_path": row["mask_path"],
                "mask_sha256": row["mask_sha256"],
                "coordinate_space": row["coordinate_space"],
                "status": row["status"],
                "selection": row["selection"],
            }
            if row["status"] == "ACCEPTED":
                target = root.joinpath(*PurePosixPath(row["mask_path"]).parts)
                retained["bytes"] = _regular_bytes(target, "teacher mask")
            result[row["sample_token"]] = retained
        return result, document.raw_sha256
    result: dict[str, dict[str, Any]] = {}
    expected_fields = {"sample_token", "source_sha256", "mask_path", "mask_sha256", "coordinate_space"}
    for row in payload["records"]:
        if not isinstance(row, dict) or set(row) != expected_fields or row["coordinate_space"] != "SOURCE_IMAGE_PIXELS":
            raise ValueError("teacher mask record schema differs")
        token = _require_sha256(row["sample_token"], "teacher sample_token")
        _require_sha256(row["source_sha256"], "teacher source_sha256")
        _require_sha256(row["mask_sha256"], "teacher mask_sha256")
        relative = _safe_relative_path(row["mask_path"], "teacher mask_path")
        target = root.joinpath(*relative.parts)
        if target.is_symlink():
            raise ValueError("teacher mask path is unsafe")
        try:
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("teacher mask path is unsafe") from exc
        if not resolved.is_relative_to(root) or resolved.relative_to(root).as_posix() != relative.as_posix() or not resolved.is_file():
            raise ValueError("teacher mask path is unsafe")
        mask_bytes = _regular_bytes(resolved, "teacher mask")
        if hashlib.sha256(mask_bytes).hexdigest() != row["mask_sha256"]:
            raise ValueError("teacher mask SHA-256 differs")
        if token in result:
            raise ValueError("teacher mask manifest repeats a sample token")
        result[token] = {**row, "bytes": mask_bytes}
    return result, document.raw_sha256


def _load_teacher(row: Mapping[str, Any], source_sha256: str, source_size: tuple[int, int]) -> Image.Image:
    if row["source_sha256"] != source_sha256:
        raise ValueError("teacher mask source SHA-256 differs")
    try:
        import io

        with Image.open(io.BytesIO(row["bytes"])) as opened:
            if opened.format != "PNG" or opened.mode != "L" or opened.size != source_size:
                raise ValueError("teacher mask must be a source-aligned L PNG")
            mask = opened.copy()
            mask.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("teacher mask is not a valid PNG") from exc
    return mask


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = find_repo_root(__file__)
    output = args.output_dir
    if not output.is_absolute():
        raise ValueError("output directory must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output_parent = output.parent.resolve(strict=True)
    if output_parent.is_relative_to(repository_root):
        raise ValueError("derived YT output must be outside the Git worktree")
    assignment_doc = _document(args.assignment, "assignment")
    source_doc = _document(args.source_bundle, "source bundle")
    receipt_doc = _document(args.split_receipt, "split receipt")
    roi_doc = _document(args.yt_roi_manifest, "YT ROI manifest")
    validate_assignment_receipt_binding(
        assignment_doc.payload,
        receipt_doc.payload,
        args.expected_split_receipt_sha256,
    )
    _, selected = _validate_split_inputs(
        assignment_doc.payload,
        source_doc.payload,
        receipt_doc.payload,
        args.expected_split_receipt_sha256,
    )
    roi_sources, roi_manifest_sha256 = _roi_source_hashes(roi_doc.payload)
    _require_regular_path(args.yt_archive, "YT archive")
    _require_regular_path(args.yt_archive_receipt, "YT archive receipt")
    archive_receipt_bytes = _regular_bytes(
        args.yt_archive_receipt, "YT archive receipt"
    )
    archive_receipt_file_sha256 = hashlib.sha256(archive_receipt_bytes).hexdigest()
    receipt = read_public_archive_receipt_bundle(args.yt_archive_receipt)
    yt_result = parse_yt_bb_dog(
        archive_path=args.yt_archive,
        binding=ArchiveReceiptBinding(YT_DATASET, receipt.archive_sha256, receipt.receipt_sha256),
    )
    archive_records = {record.source_sample_id: record for record in yt_result.original.records}
    samples = _native_samples(selected, archive_records, roi_sources)
    checkpoint_bytes = _regular_bytes(args.localizer_checkpoint, "localizer checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    model, device, localizer_bindings = load_localizer_checkpoint(checkpoint_bytes, args.device)
    teachers, teacher_manifest_file_sha256 = _teacher_records(args.teacher_mask_manifest)
    unknown_teachers = set(teachers) - {sample.sample_token for sample in samples}
    if unknown_teachers:
        raise ValueError("teacher mask manifest references samples outside admitted YT_FIT")
    policy = {
        "minimum_detector_confidence": args.minimum_detector_confidence,
        "minimum_frontality": args.minimum_frontality,
        "minimum_native_short_side": args.minimum_native_short_side,
        "maximum_mask_uncertainty": args.maximum_mask_uncertainty,
        "automatic_mask": "KEYPOINT_GEOMETRY_WITH_DETERMINISTIC_GRABCUT_WHERE_SUPPORTED",
        "teacher_mask_hook": "EXACT_SOURCE_ALIGNED_MASK_ONLY_NO_SAM_INFERENCE",
        "crop_encoding": "PNG_RGB_LOSSLESS",
        "mask_encoding": "PNG_L_LOSSLESS",
    }
    for name in ("minimum_detector_confidence", "minimum_frontality", "maximum_mask_uncertainty"):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    if isinstance(policy["minimum_native_short_side"], bool) or not isinstance(policy["minimum_native_short_side"], int) or policy["minimum_native_short_side"] <= 0:
        raise ValueError("minimum_native_short_side must be positive")
    provenance = build_offline_tool_provenance(
        Path(__file__),
        additional_paths=(
            repository_root / "parsing/export/regions/native_yt.py",
            repository_root / "parsing/export/regions/localizer.py",
            repository_root / "parsing/export/regions/manifest.py",
        ),
    )
    input_hashes = {
        "assignment_file_sha256": assignment_doc.raw_sha256,
        "assignment_payload_sha256": assignment_doc.canonical_payload_sha256,
        "source_bundle_file_sha256": source_doc.raw_sha256,
        "source_bundle_payload_sha256": source_doc.canonical_payload_sha256,
        "split_receipt_file_sha256": receipt_doc.raw_sha256,
        "split_receipt_payload_sha256": receipt_doc.canonical_payload_sha256,
        "yt_archive_sha256": receipt.archive_sha256,
        "yt_archive_receipt_sha256": receipt.receipt_sha256,
        "yt_roi_manifest_file_sha256": roi_doc.raw_sha256,
        "yt_roi_manifest_sha256": roi_manifest_sha256,
        "localizer_checkpoint_file_sha256": checkpoint_sha256,
        "localizer_bindings_sha256": localizer_bindings["content_sha256"],
    }
    if teacher_manifest_file_sha256 is not None:
        input_hashes["teacher_mask_manifest_file_sha256"] = teacher_manifest_file_sha256

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output_parent))
    try:
        for directory in ("crops", "soft_masks", "binary_masks"):
            (staging / directory).mkdir(mode=0o700)
        materialize_teacher_sources = bool(
            getattr(args, "materialize_teacher_source_images", False)
        )
        if materialize_teacher_sources:
            (staging / "teacher_source_images").mkdir(mode=0o700)
        teacher_source_records: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        started = time.monotonic()
        with NestedYtArchive(
            args.yt_archive,
            samples[0],
            expected_archive_sha256=receipt.archive_sha256,
        ) as archive:
            for index, sample in enumerate(samples, start=1):
                source_bytes = archive.read(sample)
                source_image = decode_source_image(source_bytes)
                prediction = predict_localizer(model, device, source_image)
                teacher = None
                teacher_uncertainty = None
                if sample.sample_token in teachers and teachers[sample.sample_token].get("status", "ACCEPTED") == "ACCEPTED":
                    teacher = _load_teacher(
                        teachers[sample.sample_token],
                        hashlib.sha256(source_bytes).hexdigest(),
                        source_image.size,
                    )
                    teacher_uncertainty = _teacher_uncertainty(
                        teachers[sample.sample_token]
                    )
                record, artifacts = process_native_sample(
                    sample,
                    source_bytes,
                    prediction,
                    policy=policy,
                    teacher_mask=teacher,
                    teacher_mask_uncertainty=teacher_uncertainty,
                )
                metadata = roi_sources.get((sample.registered_dog_id, sample.frame_index))
                if metadata is not None and (record["source_width"], record["source_height"]) != (metadata["width"], metadata["height"]):
                    raise ValueError("original YT dimensions differ from ROI source metadata")
                for relative, payload in artifacts.items():
                    path = _safe_relative_path(relative, "derived artifact path")
                    _write_exclusive(staging.joinpath(*path.parts), payload)
                if materialize_teacher_sources:
                    source_relative = f"teacher_source_images/{sample.sample_token}.jpg"
                    _write_exclusive(staging / source_relative, source_bytes)
                    teacher_source_records.append(
                        _teacher_source_record(record, source_relative)
                    )
                records.append(record)
                if index % 500 == 0 or index == len(samples):
                    print(
                        json.dumps(
                            {
                                "event": "native_yt_progress",
                                "processed": index,
                                "total": len(samples),
                                "elapsed_seconds": round(
                                    time.monotonic() - started, 1
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        records.sort(key=lambda row: row["sample_token"])
        bundle = build_manifest_bundle(
            records=records,
            input_sha256s=input_hashes,
            policy=policy,
            tool_provenance=provenance,
        )
        validate_manifest_bundle(bundle, root=staging)
        _write_exclusive(staging / "yt-native-nose-manifest.json", json_document_bytes(bundle))
        teacher_source_manifest = None
        if materialize_teacher_sources:
            teacher_source_records.sort(key=lambda row: row["sample_token"])
            teacher_source_manifest = {
                "schema_version": SOURCE_IMAGE_MANIFEST_SCHEMA,
                "source_receipt_file_sha256": archive_receipt_file_sha256,
                "records": teacher_source_records,
            }
            validate_source_image_manifest(
                teacher_source_manifest,
                root=staging,
                source_receipt_file_sha256=archive_receipt_file_sha256,
            )
            _write_exclusive(
                staging / "yt-native-nose-teacher-source-images.json",
                json_document_bytes(teacher_source_manifest),
            )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    counts = bundle["manifest"]["record_counts"]
    result = {
        "status": "CREATED",
        "output": str(output),
        "manifest": str(output / "yt-native-nose-manifest.json"),
        "manifest_sha256": bundle["manifest_sha256"],
        "record_counts": counts,
    }
    if teacher_source_manifest is not None:
        result["teacher_source_manifest"] = str(
            output / "yt-native-nose-teacher-source-images.json"
        )
        result["teacher_source_manifest_sha256"] = content_sha256(
            teacher_source_manifest
        )
    print(json.dumps(result, sort_keys=True))
    return result


def _teacher_source_record(
    record: Mapping[str, Any], source_image_path: str
) -> dict[str, Any]:
    return {
        "sample_token": record["sample_token"],
        "sequence_token": record["sequence_token"],
        "track_token": record["track_token"],
        "frame_index": record["frame_index"],
        "source_image_path": source_image_path,
        "source_sha256": record["source_sha256"],
        "source_width": record["source_width"],
        "source_height": record["source_height"],
        "nose_box_xyxy": record["nose_box_xyxy"],
        "keypoints": record["keypoints"],
    }


def _teacher_uncertainty(record: Mapping[str, Any]) -> float:
    selection = record.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("teacher selection schema differs")
    index = selection.get("selected_candidate_index")
    candidates = selection.get("candidates")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not isinstance(candidates, list)
        or not 0 <= index < len(candidates)
        or not isinstance(candidates[index], dict)
    ):
        raise ValueError("accepted teacher selection differs")
    score = candidates[index].get("combined_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("teacher combined score differs")
    return 1.0 - float(score)


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} is unsafe")
    return path


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
