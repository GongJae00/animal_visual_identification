"""Prepare or validate a contract-bound macro nose annotation batch."""

from __future__ import annotations

import argparse
import json
import math
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from contracts.artifact_manifest import NoseDetectorManifest
from foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
    write_private_json_directory_bundle,
)
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from identity_methods.nose.annotation import (
    ANNOTATION_TEMPLATE_SCHEMA,
    AcquisitionRecord,
    build_admission_receipt,
    canonical_jsonl_bytes,
    load_acquisition_jsonl,
    load_annotation_jsonl,
    secure_relative_file,
    validate_acquisition_records,
    validate_annotation_records,
)
from identity_methods.nose.extractor import YoloNoseDetector

_BATCH_SCHEMA = "cvi.noseid.annotation_review_batch.v1"
_BATCH_STATE = "NOT_ADMITTED_REQUIRES_HUMAN_ANNOTATION_AND_REVIEW"
_PREDICTION_SOURCE = "AUTOMATED_LOCALIZER_PROPOSAL_NOT_A_LABEL"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-completed", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acquisitions", type=Path)
    parser.add_argument("--detector-artifact", type=Path)
    parser.add_argument("--detector-manifest", type=Path)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument("--completed-annotations", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--use-cuda", action="store_true")
    return parser


def _required(args: argparse.Namespace, parser: argparse.ArgumentParser, names: tuple[str, ...]) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    if missing:
        parser.error("required arguments: " + ", ".join(missing))


def _new_output_parent(output_dir: Path) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("output directory must not exist")
    if output_dir.parent.is_symlink():
        raise ValueError("output directory parent must not be a symlink")
    parent = output_dir.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return parent


def _write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("protected batch write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_source_image(record: AcquisitionRecord, data_root: Path) -> Image.Image:
    payload = secure_relative_file(
        data_root,
        record.original_image.relative_path,
        expected_sha256=record.original_image.sha256,
        maximum_bytes=268_435_456,
        name="original image",
    )
    with Image.open(BytesIO(payload)) as opened:
        return opened.convert("RGB").copy()


def _prediction_template(
    record: AcquisitionRecord,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ANNOTATION_TEMPLATE_SCHEMA,
        "admission_status": _BATCH_STATE,
        "sample_id": record.sample_id,
        "acquisition_sha256": record.record_sha256,
        "original_image": record.original_image.to_dict(),
        "localizer_prediction": prediction,
        "completed_annotation": None,
    }


def create_review_batch(
    *,
    acquisitions_path: Path,
    data_root: Path,
    detector_artifact: Path,
    detector_manifest_path: Path,
    output_dir: Path,
    use_cuda: bool = False,
) -> dict[str, Any]:
    """Run the exact detector and publish only proposals plus blank templates."""

    parent = _new_output_parent(output_dir)
    records = load_acquisition_jsonl(acquisitions_path)
    validate_acquisition_records(records, data_root)
    manifest_document = read_strict_json_document(detector_manifest_path)
    detector_manifest = NoseDetectorManifest.from_dict(manifest_document.payload)
    if detector_artifact.is_symlink() or not detector_artifact.is_file():
        raise ValueError("detector artifact must be a non-symlink regular file")
    detector = YoloNoseDetector(
        detector_artifact, detector_manifest, use_cuda=use_cuda
    )
    acquisition_payload = canonical_jsonl_bytes(records)
    with TemporaryDirectory(prefix=".cvi-nose-annotation-", dir=parent) as temporary:
        staging = Path(temporary) / "batch"
        staging.mkdir(mode=0o700)
        crops = staging / "predicted-crops"
        crops.mkdir(mode=0o700)
        _write_bytes(staging / "acquisitions.jsonl", acquisition_payload)
        entries: list[dict[str, Any]] = []
        templates: list[dict[str, Any]] = []
        for record in records:
            image = _load_source_image(record, data_root)
            detected = detector.detect(image)
            prediction: dict[str, Any]
            if detected is None:
                prediction = {
                    "source": _PREDICTION_SOURCE,
                    "state": "NO_DETECTION",
                    "bbox_xyxy": None,
                    "confidence": None,
                    "native_short_side": None,
                    "crop": None,
                }
            else:
                x0, y0, x1, y1 = detected.box
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("detector produced an empty nose proposal")
                crop = image.crop(detected.box)
                buffer = BytesIO()
                crop.save(buffer, format="PNG", optimize=False)
                crop_payload = buffer.getvalue()
                crop_name = f"{sha256(record.sample_id.encode('utf-8')).hexdigest()}.png"
                crop_relative = PurePosixPath("predicted-crops") / crop_name
                _write_bytes(staging.joinpath(*crop_relative.parts), crop_payload)
                prediction = {
                    "source": _PREDICTION_SOURCE,
                    "state": "PREDICTED_ROI_NOT_A_LABEL",
                    "bbox_xyxy": list(detected.box),
                    "confidence": detected.confidence,
                    "native_short_side": min(x1 - x0, y1 - y0),
                    "crop": {
                        "relative_path": crop_relative.as_posix(),
                        "sha256": sha256(crop_payload).hexdigest(),
                        "width": x1 - x0,
                        "height": y1 - y0,
                    },
                }
            template = _prediction_template(record, prediction)
            templates.append(template)
            entries.append(
                {
                    "sample_id": record.sample_id,
                    "acquisition_sha256": record.record_sha256,
                    "prediction": prediction,
                }
            )
        template_payload = b"".join(
            json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
            for item in templates
        )
        _write_bytes(staging / "annotation-templates.jsonl", template_payload)
        batch = {
            "schema_version": _BATCH_SCHEMA,
            "admission_status": _BATCH_STATE,
            "localizer": {
                "artifact_id": detector_manifest.artifact_id,
                "artifact_sha256": detector_manifest.artifact_sha256,
                "manifest_sha256": manifest_document.canonical_payload_sha256,
            },
            "acquisitions": {
                "relative_path": "acquisitions.jsonl",
                "sha256": sha256(acquisition_payload).hexdigest(),
                "count": len(records),
            },
            "annotation_templates": {
                "relative_path": "annotation-templates.jsonl",
                "sha256": sha256(template_payload).hexdigest(),
                "count": len(templates),
                "contains_admitted_labels": False,
            },
            "entries": entries,
        }
        _write_bytes(staging / "batch.json", json_document_bytes(batch))
        fsync_directory(crops)
        fsync_directory(staging)
        rename_directory_noreplace(staging, parent / output_dir.name)
    fsync_directory(parent / output_dir.name)
    fsync_directory(parent)
    return batch


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} schema differs")
    return value


def _lowercase_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _validate_batch(batch_dir: Path) -> tuple[dict[str, Any], str, Path, bytes]:
    root = Path(os.path.abspath(os.fspath(batch_dir)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("batch directory must be a non-symlink directory")
    document = read_strict_json_document(root / "batch.json")
    batch = _exact(
        document.payload,
        {
            "schema_version",
            "admission_status",
            "localizer",
            "acquisitions",
            "annotation_templates",
            "entries",
        },
        "review batch",
    )
    if batch["schema_version"] != _BATCH_SCHEMA or batch["admission_status"] != _BATCH_STATE:
        raise ValueError("unsupported review batch")
    localizer = _exact(
        batch["localizer"],
        {"artifact_id", "artifact_sha256", "manifest_sha256"},
        "review batch localizer",
    )
    if not isinstance(localizer["artifact_id"], str) or not localizer["artifact_id"].strip():
        raise ValueError("review batch localizer artifact_id differs")
    _lowercase_sha256(localizer["artifact_sha256"], "localizer artifact_sha256")
    _lowercase_sha256(localizer["manifest_sha256"], "localizer manifest_sha256")
    acquisitions = _exact(
        batch["acquisitions"], {"relative_path", "sha256", "count"}, "batch acquisitions"
    )
    templates = _exact(
        batch["annotation_templates"],
        {"relative_path", "sha256", "count", "contains_admitted_labels"},
        "batch annotation templates",
    )
    if acquisitions["relative_path"] != "acquisitions.jsonl":
        raise ValueError("batch acquisition path differs")
    if templates["relative_path"] != "annotation-templates.jsonl" or templates[
        "contains_admitted_labels"
    ] is not False:
        raise ValueError("batch annotation templates are not explicitly blank")
    for value, name in (
        (acquisitions["sha256"], "batch acquisition sha256"),
        (templates["sha256"], "batch template sha256"),
    ):
        _lowercase_sha256(value, name)
    for value, name in (
        (acquisitions["count"], "batch acquisition count"),
        (templates["count"], "batch template count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    acquisition_path = root / "acquisitions.jsonl"
    secure_relative_file(
        root,
        PurePosixPath("acquisitions.jsonl"),
        expected_sha256=acquisitions["sha256"],
        maximum_bytes=268_435_456,
        name="batch acquisition JSONL",
    )
    template_payload = secure_relative_file(
        root,
        PurePosixPath("annotation-templates.jsonl"),
        expected_sha256=templates["sha256"],
        maximum_bytes=268_435_456,
        name="batch annotation template JSONL",
    )
    if not isinstance(batch["entries"], list) or len(batch["entries"]) != acquisitions["count"]:
        raise ValueError("batch entry count differs")
    if templates["count"] != acquisitions["count"]:
        raise ValueError("batch template count differs")
    for entry in batch["entries"]:
        _exact(entry, {"sample_id", "acquisition_sha256", "prediction"}, "batch entry")
        prediction = _exact(
            entry["prediction"],
            {"source", "state", "bbox_xyxy", "confidence", "native_short_side", "crop"},
            "batch prediction",
        )
        if prediction["source"] != _PREDICTION_SOURCE:
            raise ValueError("batch prediction source differs")
        crop = prediction["crop"]
        if prediction["state"] == "NO_DETECTION":
            if any(prediction[key] is not None for key in ("bbox_xyxy", "confidence", "native_short_side", "crop")):
                raise ValueError("NO_DETECTION prediction must not contain a proposal")
        elif prediction["state"] == "PREDICTED_ROI_NOT_A_LABEL":
            crop_value = _exact(
                crop, {"relative_path", "sha256", "width", "height"}, "predicted crop"
            )
            bbox = prediction["bbox_xyxy"]
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError("predicted bbox differs")
            confidence = prediction["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError("predicted confidence differs")
            width, height = crop_value["width"], crop_value["height"]
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in (width, height, prediction["native_short_side"])
                )
                or width != bbox[2] - bbox[0]
                or height != bbox[3] - bbox[1]
                or prediction["native_short_side"] != min(width, height)
            ):
                raise ValueError("predicted crop geometry differs")
            relative = PurePosixPath(crop_value["relative_path"])
            if relative.parent != PurePosixPath("predicted-crops"):
                raise ValueError("predicted crop path differs")
            crop_payload = secure_relative_file(
                root,
                relative,
                expected_sha256=crop_value["sha256"],
                maximum_bytes=67_108_864,
                name="predicted crop",
            )
            with Image.open(BytesIO(crop_payload)) as opened:
                if opened.format != "PNG" or opened.size != (
                    crop_value["width"],
                    crop_value["height"],
                ):
                    raise ValueError("predicted crop dimensions differ")
        else:
            raise ValueError("batch prediction state differs")
    return batch, content_sha256(batch), acquisition_path, template_payload


def validate_completed_batch(
    *,
    batch_dir: Path,
    completed_annotations: Path,
    data_root: Path,
    annotation_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate human-completed records and publish an admission receipt only."""

    _new_output_parent(output_dir)
    batch, batch_sha256, acquisition_path, template_payload = _validate_batch(batch_dir)
    acquisitions = load_acquisition_jsonl(acquisition_path)
    annotations = load_annotation_jsonl(completed_annotations)
    if canonical_jsonl_bytes(acquisitions) != secure_relative_file(
        Path(os.path.abspath(os.fspath(batch_dir))),
        PurePosixPath("acquisitions.jsonl"),
        expected_sha256=batch["acquisitions"]["sha256"],
        maximum_bytes=268_435_456,
        name="batch acquisition JSONL",
    ):
        raise ValueError("batch acquisitions are not canonical JSONL")
    expected_samples = [
        (entry["sample_id"], entry["acquisition_sha256"]) for entry in batch["entries"]
    ]
    observed_samples = [(item.sample_id, item.record_sha256) for item in acquisitions]
    if observed_samples != expected_samples:
        raise ValueError("batch entries differ from acquisition records")
    for acquisition, entry in zip(acquisitions, batch["entries"], strict=True):
        bbox = entry["prediction"]["bbox_xyxy"]
        if bbox is not None and (
            bbox[2] > acquisition.original_image.width
            or bbox[3] > acquisition.original_image.height
        ):
            raise ValueError("predicted bbox exceeds original image")
    expected_templates = b"".join(
        json.dumps(
            _prediction_template(acquisition, entry["prediction"]),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for acquisition, entry in zip(acquisitions, batch["entries"], strict=True)
    )
    if template_payload != expected_templates:
        raise ValueError("batch templates are not exact blank annotation templates")
    validate_annotation_records(
        acquisitions,
        annotations,
        data_root=data_root,
        annotation_root=annotation_root,
    )
    receipt = build_admission_receipt(
        acquisitions,
        annotations,
        acquisition_jsonl_sha256=sha256(canonical_jsonl_bytes(acquisitions)).hexdigest(),
        annotation_jsonl_sha256=sha256(canonical_jsonl_bytes(annotations)).hexdigest(),
        batch_manifest_sha256=batch_sha256,
    )
    write_private_json_directory_bundle(
        output_dir, (("admission-receipt.json", receipt),)
    )
    return receipt


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.validate_completed:
        _required(
            args,
            parser,
            ("batch_dir", "completed_annotations", "annotation_root"),
        )
        receipt = validate_completed_batch(
            batch_dir=args.batch_dir,
            completed_annotations=args.completed_annotations,
            data_root=args.data_root,
            annotation_root=args.annotation_root,
            output_dir=args.output_dir,
        )
        print(json.dumps({"admitted_count": receipt["admitted_count"]}, sort_keys=True))
    else:
        _required(
            args,
            parser,
            ("acquisitions", "detector_artifact", "detector_manifest"),
        )
        batch = create_review_batch(
            acquisitions_path=args.acquisitions,
            data_root=args.data_root,
            detector_artifact=args.detector_artifact,
            detector_manifest_path=args.detector_manifest,
            output_dir=args.output_dir,
            use_cuda=args.use_cuda,
        )
        print(json.dumps({"review_count": len(batch["entries"])}, sort_keys=True))


if __name__ == "__main__":
    main()
