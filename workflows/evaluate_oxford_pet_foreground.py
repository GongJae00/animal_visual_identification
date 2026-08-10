"""Evaluate whole-pet foreground masks on the Oxford-IIIT Pet test trimaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from artifact_contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
)
from artifact_contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)
from data_pipeline.acquisition import sha256_file
from foundation.protected_io import json_document_bytes
from foundation.protected_publication import (
    fsync_directory,
    rename_directory_noreplace,
)
from foundation.provenance import content_sha256
from localization.animal_instance_segmentation import (
    AnimalInstanceSegmentationRuntime,
)
from localization.animal_parsing import (
    AnimalParsingPolicy,
    AnimalParsingRuntime,
    ParsedAnimalInstance,
)
from localization.foreground_segmentation import ForegroundSegmentationRuntime

REPORT_SCHEMA = "cvi.oxford_pet_foreground_evaluation.v3"
INTERPRETATION = (
    "UNASSISTED_MULTI_INSTANCE_INFERENCE_POSTHOC_TRIMAP_MATCHING_"
    "EMPTY_GROUND_TRUTH_EXCLUDED_NOT_PRODUCTION_VALIDATION"
)
_MODEL_NAMES = ("birefnet_box_refinement", "rf_detr", "refined")
_SPECIES_BY_ID = {1: "cat", 2: "dog"}
_SAMPLE_NAME = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class OxfordPetSample:
    name: str
    class_id: int
    species: str
    breed_id: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--images-archive", type=Path, required=True)
    parser.add_argument("--annotations-archive", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--instance-model-directory", type=Path, required=True)
    parser.add_argument("--instance-model-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "trainval"), default="test")
    parser.add_argument("--species", choices=("all", "cat", "dog"), default="all")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_evaluation(
        dataset_root=args.dataset_root,
        images_archive=args.images_archive,
        annotations_archive=args.annotations_archive,
        model_directory=args.model_directory,
        model_manifest=args.model_manifest,
        instance_model_directory=args.instance_model_directory,
        instance_model_manifest=args.instance_model_manifest,
        output_directory=args.output_directory,
        split=args.split,
        species=args.species,
        sample_count=args.sample_count,
        threshold=args.threshold,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_OXFORD_PET_FOREGROUND_EVALUATION",
                "output": str(args.output_directory),
                "record_count": len(report["records"]),
                "report_sha256": content_sha256(report),
                "overall": report["aggregates"]["all"],
            },
            sort_keys=True,
        )
    )
    return 0


def run_evaluation(
    *,
    dataset_root: Path,
    images_archive: Path,
    annotations_archive: Path,
    model_directory: Path,
    model_manifest: Path,
    instance_model_directory: Path,
    instance_model_manifest: Path,
    output_directory: Path,
    split: str,
    species: str,
    sample_count: int | None,
    threshold: float,
    device: str,
) -> dict[str, Any]:
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    output_parent = output_directory.parent.resolve(strict=True)
    root = dataset_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Oxford Pet dataset root must be a regular directory")
    selected_samples = _load_split_samples(
        root, split=split, species=species, sample_count=sample_count
    )
    samples, exclusions = _preflight_samples(selected_samples, dataset_root=root)
    if not samples:
        raise ValueError("Oxford Pet selection has no evaluable trimaps")
    foreground_artifact = ForegroundSegmentationArtifact.load(
        model_directory=model_directory,
        manifest_bundle_path=model_manifest,
    )
    instance_artifact = InstanceSegmentationArtifact.load(
        model_directory=instance_model_directory,
        manifest_bundle_path=instance_model_manifest,
    )
    foreground_runtime = ForegroundSegmentationRuntime(
        artifact=foreground_artifact,
        device=device,
        threshold=threshold,
    )
    instance_runtime = AnimalInstanceSegmentationRuntime(
        artifact=instance_artifact,
        device=device,
        mask_threshold=threshold,
    )
    parsing_policy = AnimalParsingPolicy(foreground_threshold=threshold)
    parsing_runtime = AnimalParsingRuntime(
        instance_runtime=instance_runtime,
        foreground_runtime=foreground_runtime,
        policy=parsing_policy,
    )
    records: list[dict[str, Any]] = []
    for sample in samples:
        image_path = root / "images" / f"{sample.name}.jpg"
        trimap_path = root / "annotations" / "trimaps" / f"{sample.name}.png"
        _require_retained_dataset_file(image_path, root=root, subject="image")
        _require_retained_dataset_file(trimap_path, root=root, subject="trimap")
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        trimap = _load_trimap(trimap_path, expected_size=source.size)
        started = time.perf_counter()
        parsing = parsing_runtime.predict(source)
        parsing_seconds = time.perf_counter() - started
        matched = _match_parsed_instance(
            parsing.instances, trimap=trimap, species=sample.species
        )
        empty_mask = np.zeros(trimap.shape, dtype=np.uint8)
        if matched is None:
            birefnet_mask = empty_mask
            instance_mask = empty_mask
            refined_mask = empty_mask
            matched_index = None
            matched_query = None
            matched_score = None
            refined_state = "MISSED"
            refined_reasons = ["NO_MATCHING_SPECIES_INSTANCE"]
            refined_flags: list[str] = []
            refined_quality = None
        else:
            birefnet_mask = np.ascontiguousarray(
                matched.foreground_probability >= threshold, dtype=np.uint8
            )
            instance_mask = np.ascontiguousarray(
                matched.instance_probability >= threshold, dtype=np.uint8
            )
            refined_mask = matched.hard_mask
            matched_index = matched.instance_index
            matched_query = matched.query_index
            matched_score = matched.class_score
            refined_state = matched.quality.state
            refined_reasons = list(matched.quality.reasons)
            refined_flags = list(matched.quality.flags)
            refined_quality = {
                "semantic_shape_iou": matched.quality.semantic_shape_iou,
                "ownership_retention": matched.quality.ownership_retention,
                "foreground_pixels": matched.quality.foreground_pixels,
                "component_count": matched.quality.component_count,
                "touches_source_border": matched.quality.touches_source_border,
            }
        records.append(
            {
                "sample_name": sample.name,
                "class_id": sample.class_id,
                "species": sample.species,
                "breed_id": sample.breed_id,
                "source_image": _file_binding(image_path, root),
                "ground_truth_trimap": _file_binding(trimap_path, root),
                "source_width": source.width,
                "source_height": source.height,
                "inference": {
                    "annotation_assistance": False,
                    "predicted_instance_count": len(parsing.instances),
                    "posthoc_matching": "MAX_CLASSIFIED_PIXEL_IOU_SAME_SPECIES",
                    "matched_instance_index": matched_index,
                    "matched_query_index": matched_query,
                    "elapsed_seconds": parsing_seconds,
                },
                "predictions": {
                    "birefnet_box_refinement": {
                        "evaluation": _evaluate_mask(birefnet_mask, trimap),
                        "state": "CANDIDATE" if matched is not None else "MISSED",
                        "reasons": (
                            [] if matched is not None else ["NO_MATCHING_SPECIES_INSTANCE"]
                        ),
                    },
                    "rf_detr": {
                        "evaluation": _evaluate_mask(instance_mask, trimap),
                        "state": "CANDIDATE" if matched is not None else "MISSED",
                        "reasons": (
                            [] if matched is not None else ["NO_MATCHING_SPECIES_INSTANCE"]
                        ),
                        "class_score": matched_score,
                        "query_index": matched_query,
                    },
                    "refined": {
                        "evaluation": _evaluate_mask(refined_mask, trimap),
                        "state": refined_state,
                        "reasons": refined_reasons,
                        "flags": refined_flags,
                        "quality": refined_quality,
                    },
                },
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA,
        "interpretation": INTERPRETATION,
        "dataset": {
            "name": "Oxford-IIIT Pet",
            "split": split,
            "species_filter": species,
            "trimap_semantics": {
                "1": "foreground",
                "2": "background",
                "3": "not_classified_excluded_from_metrics",
            },
            "images_archive": _external_file_binding(images_archive),
            "annotations_archive": _external_file_binding(annotations_archive),
            "research_use_only": True,
        },
        "selection": {
            "algorithm": (
                "ALL_ELIGIBLE_IN_CANONICAL_NAME_ORDER"
                if sample_count is None
                else "SHA256_DOMAIN_SEPARATED_V1"
            ),
            "requested_count": sample_count,
            "selected_count": len(selected_samples),
            "evaluated_count": len(samples),
            "excluded_count": len(exclusions),
        },
        "exclusions": exclusions,
        "threshold": threshold,
        "parsing_policy": parsing_policy.to_dict(),
        "parsing_policy_sha256": parsing_policy.policy_sha256,
        "model": _artifact_binding(foreground_artifact),
        "semantic_gate_model": {
            **_artifact_binding(instance_artifact),
            "target_class_policy": "PREDICT_ALL_DOG_AND_CAT",
            "query_selection_policy": "NO_ANNOTATION_ASSISTANCE",
            "evaluation_matching_policy": "POSTHOC_TRIMAP_IOU_SAME_SPECIES",
        },
        "metric_policy": {
            "classified_pixel_iou": "TP/(TP+FP+FN)",
            "classified_pixel_dice": "2TP/(2TP+FP+FN)",
            "foreground_recall": "TP/(TP+FN)",
            "background_leakage_rate": "FP/(FP+TN)",
            "correction_rate": "(FP+FN)/(TP+FP+FN+TN)",
            "macro_average": "UNWEIGHTED_PER_IMAGE_MEAN",
            "micro_average": "PIXEL_COUNTS_POOLED_ACROSS_IMAGES",
        },
        "records": records,
        "aggregates": _aggregate_records(records),
    }
    with TemporaryDirectory(prefix=".oxford-foreground-", dir=output_parent) as temporary:
        staging = Path(temporary) / "evaluation"
        staging.mkdir(mode=0o700)
        _write_regular_file(staging / "report.json", json_document_bytes(report))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_parent / output_directory.name)
    fsync_directory(output_parent / output_directory.name)
    fsync_directory(output_parent)
    return report


def _load_split_samples(
    dataset_root: Path,
    *,
    split: str,
    species: str,
    sample_count: int | None,
) -> tuple[OxfordPetSample, ...]:
    if split not in {"test", "trainval"}:
        raise ValueError("Oxford Pet split differs")
    if species not in {"all", "cat", "dog"}:
        raise ValueError("Oxford Pet species filter differs")
    if sample_count is not None and (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("Oxford Pet sample count must be positive")
    split_path = dataset_root / "annotations" / f"{split}.txt"
    rows: list[OxfordPetSample] = []
    names: set[str] = set()
    for line_number, line in enumerate(
        split_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 4 or not _SAMPLE_NAME.fullmatch(fields[0]):
            raise ValueError(f"Oxford Pet split row {line_number} differs")
        try:
            class_id, species_id, breed_id = map(int, fields[1:])
            species_name = _SPECIES_BY_ID[species_id]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Oxford Pet split row {line_number} labels differ"
            ) from exc
        name = fields[0]
        if name in names:
            raise ValueError("Oxford Pet split sample names must be unique")
        names.add(name)
        if species == "all" or species == species_name:
            rows.append(OxfordPetSample(name, class_id, species_name, breed_id))
    rows.sort(key=lambda item: item.name)
    if sample_count is not None:
        rows.sort(
            key=lambda item: hashlib.sha256(
                b"cvi.oxford_pet_foreground.v1\0" + item.name.encode("ascii")
            ).digest()
        )
        if len(rows) < sample_count:
            raise ValueError("insufficient Oxford Pet split samples")
        rows = rows[:sample_count]
        rows.sort(key=lambda item: item.name)
    if not rows:
        raise ValueError("Oxford Pet selection is empty")
    return tuple(rows)


def _load_trimap(path: Path, *, expected_size: tuple[int, int]) -> np.ndarray:
    trimap = _read_trimap(path, expected_size=expected_size)
    if not np.any(trimap == 1) or not np.any(trimap == 2):
        raise ValueError("Oxford Pet trimap lacks classified foreground or background")
    return trimap


def _read_trimap(path: Path, *, expected_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.size != expected_size:
            raise ValueError("Oxford Pet image and trimap dimensions differ")
        trimap = np.asarray(opened, dtype=np.uint8).copy()
    if trimap.ndim != 2 or not set(np.unique(trimap)).issubset({1, 2, 3}):
        raise ValueError("Oxford Pet trimap labels differ")
    return trimap


def _preflight_samples(
    samples: tuple[OxfordPetSample, ...], *, dataset_root: Path
) -> tuple[tuple[OxfordPetSample, ...], list[dict[str, Any]]]:
    eligible: list[OxfordPetSample] = []
    exclusions: list[dict[str, Any]] = []
    for sample in samples:
        image_path = dataset_root / "images" / f"{sample.name}.jpg"
        trimap_path = (
            dataset_root / "annotations" / "trimaps" / f"{sample.name}.png"
        )
        _require_retained_dataset_file(
            image_path, root=dataset_root, subject="image"
        )
        _require_retained_dataset_file(
            trimap_path, root=dataset_root, subject="trimap"
        )
        with Image.open(image_path) as opened:
            source_size = opened.size
        trimap = _read_trimap(trimap_path, expected_size=source_size)
        if not np.any(trimap == 1):
            exclusions.append(
                {
                    "sample_name": sample.name,
                    "species": sample.species,
                    "reason": "GROUND_TRUTH_TRIMAP_HAS_NO_FOREGROUND",
                    "source_image": _file_binding(image_path, dataset_root),
                    "ground_truth_trimap": _file_binding(
                        trimap_path, dataset_root
                    ),
                    "trimap_labels": [int(value) for value in np.unique(trimap)],
                }
            )
            continue
        if not np.any(trimap == 2):
            raise ValueError("Oxford Pet trimap lacks classified background")
        eligible.append(sample)
    return tuple(eligible), exclusions


def _match_parsed_instance(
    instances: tuple[ParsedAnimalInstance, ...],
    *,
    trimap: np.ndarray,
    species: str,
) -> ParsedAnimalInstance | None:
    candidates = [item for item in instances if item.class_name == species]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            _evaluate_mask(item.hard_mask, trimap)["metrics"][
                "classified_pixel_iou"
            ],
            item.class_score,
            -item.query_index,
        ),
    )


def _evaluate_mask(mask: np.ndarray, trimap: np.ndarray) -> dict[str, Any]:
    if (
        mask.shape != trimap.shape
        or mask.dtype != np.uint8
        or not set(np.unique(mask)).issubset({0, 1})
    ):
        raise ValueError("Oxford Pet predicted mask differs")
    predicted = mask.astype(bool)
    foreground = trimap == 1
    background = trimap == 2
    uncertain = trimap == 3
    counts = {
        "true_positive_pixels": int(np.count_nonzero(predicted & foreground)),
        "false_positive_pixels": int(np.count_nonzero(predicted & background)),
        "false_negative_pixels": int(np.count_nonzero(~predicted & foreground)),
        "true_negative_pixels": int(np.count_nonzero(~predicted & background)),
        "not_classified_pixels": int(np.count_nonzero(uncertain)),
        "predicted_foreground_not_classified_pixels": int(
            np.count_nonzero(predicted & uncertain)
        ),
    }
    return {"counts": counts, "metrics": _metrics_from_counts(counts)}


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["true_positive_pixels"]
    fp = counts["false_positive_pixels"]
    fn = counts["false_negative_pixels"]
    tn = counts["true_negative_pixels"]
    classified = tp + fp + fn + tn
    return {
        "classified_pixel_iou": tp / (tp + fp + fn),
        "classified_pixel_dice": 2 * tp / (2 * tp + fp + fn),
        "foreground_recall": tp / (tp + fn),
        "background_leakage_rate": fp / (fp + tn),
        "correction_rate": (fp + fn) / classified,
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in ("all", "cat", "dog"):
        selected = [
            record
            for record in records
            if stratum == "all" or record["species"] == stratum
        ]
        if not selected:
            continue
        result[stratum] = {}
        for model_name in _MODEL_NAMES:
            aggregate = _aggregate_evaluations(
                [record["predictions"][model_name]["evaluation"] for record in selected]
            )
            states = [record["predictions"][model_name]["state"] for record in selected]
            aggregate["state_counts"] = {
                state: states.count(state) for state in sorted(set(states))
            }
            result[stratum][model_name] = aggregate
    return result


def _aggregate_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("cannot aggregate empty Oxford Pet evaluations")
    count_names = tuple(evaluations[0]["counts"])
    metric_names = tuple(evaluations[0]["metrics"])
    counts = {
        name: sum(item["counts"][name] for item in evaluations)
        for name in count_names
    }
    return {
        "record_count": len(evaluations),
        "counts": counts,
        "micro_average": _metrics_from_counts(counts),
        "macro_average": {
            name: float(
                np.mean([item["metrics"][name] for item in evaluations])
            )
            for name in metric_names
        },
    }


def _artifact_binding(artifact: Any) -> dict[str, Any]:
    return {
        "model_id": artifact.manifest.model_id,
        "source_revision": artifact.manifest.source_revision,
        "manifest_sha256": artifact.manifest.manifest_sha256,
        "bundle_raw_sha256": artifact.bundle_sha256,
    }


def _require_retained_dataset_file(path: Path, *, root: Path, subject: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Oxford Pet {subject} must be a regular file")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"Oxford Pet {subject} escapes dataset root")


def _file_binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _external_file_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("Oxford Pet archive must be a regular file")
    return {
        "file_name": resolved.name,
        "sha256": sha256_file(resolved),
        "byte_size": resolved.stat().st_size,
    }


def _write_regular_file(path: Path, payload: bytes) -> None:
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
                raise OSError("Oxford Pet evaluation write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
