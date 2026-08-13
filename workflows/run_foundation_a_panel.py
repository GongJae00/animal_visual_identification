"""Run a bounded C-RADIOv4 plus SAM2.1 Appearance-mask agreement panel."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data.adapters import adapt_ap10k_dog
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from parsing.regions.foundation_dense_runtime import FoundationDenseRuntime
from parsing.regions.foundation_region_candidate import derive_binary_foundation_candidate
from parsing.regions.region_teacher_consensus import (
    INTERPRETATION,
    RegionConsensusPolicy,
    RegionTeacherBinding,
    region_teacher_consensus,
)
from parsing.regions.sam2_prompt_runtime import Sam2PromptRuntime

SCHEMA = "cvi.foundation_appearance_agreement_panel.v1"


def run_panel(
    *,
    data_root: Path,
    cradio_model_directory: Path,
    cradio_manifest: Path,
    sam2_model_directory: Path,
    sam2_manifest: Path,
    output_directory: Path,
    maximum_samples: int,
    resolution: int = 1024,
) -> dict[str, Any]:
    if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples <= 0:
        raise ValueError("foundation A panel maximum samples must be positive")
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite foundation A panel: {output_directory}")
    root = data_root.resolve(strict=True)
    samples = tuple(sorted(adapt_ap10k_dog(root), key=lambda item: item.sample_id))
    selected = samples[:maximum_samples]
    if not selected:
        raise ValueError("foundation A panel selected no samples")
    cradio = FoundationDenseRuntime(
        model_directory=cradio_model_directory,
        manifest_bundle_path=cradio_manifest,
    )
    sam2 = Sam2PromptRuntime(
        model_directory=sam2_model_directory,
        manifest_bundle_path=sam2_manifest,
    )
    policy = RegionConsensusPolicy()
    teachers = (
        RegionTeacherBinding(
            "cradio-v4-dense-geometry",
            "CRADIO_V4",
            cradio.artifact.manifest.manifest_sha256,
        ),
        RegionTeacherBinding(
            "sam2.1-prompt",
            "SAM2_1",
            sam2.artifact.manifest.manifest_sha256,
        ),
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging-", dir=output_directory.parent
        )
    )
    try:
        arrays_directory = staging / "arrays"
        arrays_directory.mkdir()
        records = [
            _process_sample(
                sample,
                root=root,
                arrays_directory=arrays_directory,
                cradio=cradio,
                sam2=sam2,
                teachers=teachers,
                policy=policy,
                resolution=resolution,
            )
            for sample in selected
        ]
        body = {
            "schema_version": SCHEMA,
            "interpretation": INTERPRETATION,
            "scope": "BOUNDED_AP10K_APPEARANCE_MASK_TEACHER_AGREEMENT_NOT_FULL_DATASET",
            "dataset_name": "ap10k-dog",
            "dataset_version": selected[0].dataset_version,
            "adapter_record_count": len(samples),
            "panel_record_count": len(records),
            "complete_adapter_dataset": len(records) == len(samples),
            "resolution": resolution,
            "models": {"cradio": cradio.binding, "sam2": sam2.binding},
            "consensus_policy": policy.to_dict(),
            "sam2_selection_policy": {
                "minimum_predicted_iou": 0.80,
                "minimum_top_two_margin": 0.05,
            },
            "records": records,
            "counts": {
                state: sum(record["state"] == state for record in records)
                for state in ("HARD_CANDIDATE", "SOFT_CANDIDATE", "ABSTAIN")
            },
        }
        report = {**body, "report_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "report.json", report),))
        fsync_directory(arrays_directory)
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_directory)
        fsync_directory(output_directory.parent)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_foundation_a_panel(path: Path) -> dict[str, Any]:
    document = read_strict_json_document(
        path,
        maximum_bytes=16_777_216,
        maximum_nodes=1_000_000,
        maximum_keys=500_000,
        maximum_array_length=100_000,
    )
    report = document.payload
    expected = {
        "schema_version",
        "interpretation",
        "scope",
        "dataset_name",
        "dataset_version",
        "adapter_record_count",
        "panel_record_count",
        "complete_adapter_dataset",
        "resolution",
        "models",
        "consensus_policy",
        "sam2_selection_policy",
        "records",
        "counts",
        "report_sha256",
    }
    body = {name: value for name, value in report.items() if name != "report_sha256"}
    if (
        set(report) != expected
        or report["schema_version"] != SCHEMA
        or report["interpretation"] != INTERPRETATION
        or report["scope"]
        != "BOUNDED_AP10K_APPEARANCE_MASK_TEACHER_AGREEMENT_NOT_FULL_DATASET"
        or content_sha256(body) != report["report_sha256"]
    ):
        raise ValueError("foundation A panel report differs")
    records = report["records"]
    if (
        not isinstance(records, list)
        or len(records) != report["panel_record_count"]
        or records != sorted(records, key=lambda item: item.get("sample_id", ""))
    ):
        raise ValueError("foundation A panel records differ")
    counts = {
        state: sum(record.get("state") == state for record in records)
        for state in ("HARD_CANDIDATE", "SOFT_CANDIDATE", "ABSTAIN")
    }
    if counts != report["counts"]:
        raise ValueError("foundation A panel counts differ")
    root = path.parent.resolve(strict=True)
    for record in records:
        _validate_panel_record(record, root=root)
    return report


def _validate_panel_record(record: object, *, root: Path) -> None:
    expected = {
        "sample_id",
        "split_role",
        "image_path",
        "image_sha256",
        "width",
        "height",
        "state",
        "metrics",
        "sam2_selected_candidate",
        "sam2_predicted_iou",
        "sam2_top_two_margin",
        "cradio_confidence",
        "artifact",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError("foundation A panel record schema differs")
    if record["state"] not in {"HARD_CANDIDATE", "SOFT_CANDIDATE", "ABSTAIN"}:
        raise ValueError("foundation A panel state differs")
    if not isinstance(record["width"], int) or not isinstance(record["height"], int):
        raise TypeError("foundation A panel dimensions differ")
    if (
        isinstance(record["width"], bool)
        or isinstance(record["height"], bool)
        or record["width"] <= 0
        or record["height"] <= 0
    ):
        raise ValueError("foundation A panel dimensions differ")
    artifact = record["artifact"]
    if artifact is None:
        if record["state"] != "ABSTAIN":
            raise ValueError("foundation A candidate lacks its array artifact")
        return
    if not isinstance(artifact, dict) or set(artifact) != {
        "relative_path",
        "sha256",
        "byte_size",
    }:
        raise ValueError("foundation A panel array binding differs")
    relative = Path(artifact["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("foundation A panel array path is unsafe")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError("foundation A panel array path is unsafe")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("foundation A panel array path is unsafe")
    read_retained_regular_file(
        resolved,
        expected_bytes=artifact["byte_size"],
        expected_sha256=artifact["sha256"],
        maximum_bytes=268_435_456,
        subject="foundation A panel arrays",
    )
    with np.load(resolved, allow_pickle=False) as loaded:
        expected_arrays = {
            "cradio_probability",
            "sam2_probabilities",
            "consensus_soft_probabilities",
            "consensus_uncertainty",
            "consensus_hard_mask",
            "source_geometry_support",
        }
        if set(loaded.files) != expected_arrays:
            raise ValueError("foundation A panel array names differ")
        height, width = record["height"], record["width"]
        cradio = loaded["cradio_probability"]
        sam2 = loaded["sam2_probabilities"]
        geometry = loaded["source_geometry_support"]
        if cradio.shape != (height, width) or sam2.shape != (3, height, width):
            raise ValueError("foundation A teacher array shapes differ")
        if geometry.shape != (height, width) or geometry.dtype != np.bool_:
            raise ValueError("foundation A geometry array differs")
        if (
            not np.isfinite(cradio).all()
            or not np.isfinite(sam2).all()
            or np.any((cradio < 0.0) | (cradio > 1.0))
            or np.any((sam2 < 0.0) | (sam2 > 1.0))
        ):
            raise ValueError("foundation A teacher probabilities differ")
        soft = loaded["consensus_soft_probabilities"]
        uncertainty = loaded["consensus_uncertainty"]
        hard = loaded["consensus_hard_mask"]
        if record["state"] == "ABSTAIN":
            if soft.size or uncertainty.size or hard.size:
                raise ValueError("abstained foundation A record carries consensus arrays")
        elif (
            soft.shape != (2, height, width)
            or uncertainty.shape != (height, width)
            or hard.shape != (height, width)
        ):
            raise ValueError("foundation A consensus array shapes differ")


def _process_sample(
    sample,
    *,
    root: Path,
    arrays_directory: Path,
    cradio: FoundationDenseRuntime,
    sam2: Sam2PromptRuntime,
    teachers: tuple[RegionTeacherBinding, ...],
    policy: RegionConsensusPolicy,
    resolution: int,
) -> dict[str, Any]:
    image = _read_image(root, sample)
    points = tuple(
        (float(x), float(y))
        for x, y, confidence in (sample.body_keypoints or {}).values()
        if confidence > 0.0
    )[:12]
    if sample.dog_boxes_xyxy is None or not points:
        return _abstained_record(sample, "PUBLISHER_GEOMETRY_INCOMPLETE")
    box = tuple(float(item) for item in sample.dog_boxes_xyxy)
    dense = cradio.extract((image,), resolution=resolution)
    radio = derive_binary_foundation_candidate(
        dense.features[0],
        transform=dense.transforms[0],
        source_validity=dense.source_validity[0],
        box_xyxy=box,
        positive_points_xy=points,
    )
    sam = sam2.predict(image, box_xyxy=box, positive_points_xy=points)
    order = np.argsort(sam.predicted_ious)[::-1]
    selected = int(order[0])
    top_score = float(sam.predicted_ious[selected])
    margin = top_score - float(sam.predicted_ious[int(order[1])])
    if top_score < 0.80:
        state = "ABSTAIN"
        metrics = {"decision_reasons": ["LOW_SAM2_PREDICTED_IOU"]}
        consensus_soft = np.empty((0,), dtype=np.float16)
        uncertainty = np.empty((0,), dtype=np.float16)
        hard_mask = np.empty((0,), dtype=np.uint8)
    elif margin < 0.05:
        state = "ABSTAIN"
        metrics = {"decision_reasons": ["AMBIGUOUS_SAM2_MULTIMASK"]}
        consensus_soft = np.empty((0,), dtype=np.float16)
        uncertainty = np.empty((0,), dtype=np.float16)
        hard_mask = np.empty((0,), dtype=np.uint8)
    else:
        radio_probability = radio.source_probability
        sam_probability = sam.mask_probabilities[selected]
        probabilities = np.stack(
            (
                np.stack((1.0 - radio_probability, radio_probability)),
                np.stack((1.0 - sam_probability, sam_probability)),
            )
        )
        consensus = region_teacher_consensus(
            "A",
            probabilities,
            teachers=teachers,
            source_validity=np.ones((sample.height, sample.width), dtype=bool),
            geometry_support=radio.source_geometry_support,
            policy=policy,
        )
        state = consensus.state.value
        metrics = consensus.metrics
        consensus_soft = (
            np.empty((0,), dtype=np.float16)
            if consensus.soft_probabilities is None
            else consensus.soft_probabilities.astype(np.float16)
        )
        uncertainty = (
            np.empty((0,), dtype=np.float16)
            if consensus.uncertainty is None
            else consensus.uncertainty.astype(np.float16)
        )
        hard_mask = (
            np.empty((0,), dtype=np.uint8)
            if consensus.hard_mask is None
            else consensus.hard_mask.astype(np.uint8)
        )
    artifact_path = arrays_directory / f"{sample.sample_id}.npz"
    np.savez_compressed(
        artifact_path,
        cradio_probability=radio.source_probability.astype(np.float16),
        sam2_probabilities=sam.mask_probabilities.astype(np.float16),
        consensus_soft_probabilities=consensus_soft,
        consensus_uncertainty=uncertainty,
        consensus_hard_mask=hard_mask,
        source_geometry_support=radio.source_geometry_support,
    )
    binding = _file_binding(artifact_path)
    return {
        "sample_id": sample.sample_id,
        "split_role": sample.split_role,
        "image_path": sample.image_path,
        "image_sha256": sample.image_sha256,
        "width": sample.width,
        "height": sample.height,
        "state": state,
        "metrics": metrics,
        "sam2_selected_candidate": selected,
        "sam2_predicted_iou": top_score,
        "sam2_top_two_margin": margin,
        "cradio_confidence": radio.confidence,
        "artifact": {"relative_path": f"arrays/{artifact_path.name}", **binding},
    }


def _abstained_record(sample, reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "split_role": sample.split_role,
        "image_path": sample.image_path,
        "image_sha256": sample.image_sha256,
        "width": sample.width,
        "height": sample.height,
        "state": "ABSTAIN",
        "metrics": {"decision_reasons": [reason]},
        "sam2_selected_candidate": None,
        "sam2_predicted_iou": None,
        "sam2_top_two_margin": None,
        "cradio_confidence": None,
        "artifact": None,
    }


def _read_image(root: Path, sample) -> Image.Image:
    candidate = root / sample.image_path
    if candidate.is_symlink():
        raise ValueError("foundation A panel source image path is unsafe")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("foundation A panel source image path is unsafe")
    retained = read_retained_regular_file(
        path,
        expected_sha256=sample.image_sha256,
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="foundation A panel source image",
    )
    assert retained.payload is not None
    with Image.open(io.BytesIO(retained.payload)) as opened:
        if opened.size != (sample.width, sample.height):
            raise ValueError("foundation A panel source image dimensions differ")
        image = opened.convert("RGB")
        image.load()
    return image


def _file_binding(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "byte_size": size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cradio-model-directory", type=Path, required=True)
    parser.add_argument("--cradio-manifest", type=Path, required=True)
    parser.add_argument("--sam2-model-directory", type=Path, required=True)
    parser.add_argument("--sam2-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-samples", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_panel(**vars(args))
    print(
        json.dumps(
            {
                "status": "CREATED_FOUNDATION_APPEARANCE_PANEL",
                "report_sha256": report["report_sha256"],
                "counts": report["counts"],
                "output": str(args.output_directory),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
