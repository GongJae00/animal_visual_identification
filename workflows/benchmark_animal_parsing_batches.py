"""Benchmark fixed dog-only parser batches on 512 Oxford dog images."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

from contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
)
from contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)
from data.acquisition import sha256_file
from foundation.protected_io import json_document_bytes
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from embedding.methods.full_segment.preparation.materialization import _parser_decisions
from parsing.full_segment.animal_instance_segmentation import AnimalInstanceSegmentationRuntime
from parsing.full_segment.animal_parsing import AnimalParsingPolicy, AnimalParsingRuntime
from parsing.full_segment.foreground_segmentation import ForegroundSegmentationRuntime
from workflows.evaluate_oxford_pet_foreground import (
    OxfordPetSample,
    _load_split_samples,
    _preflight_samples,
)

REPORT_SCHEMA = "cvi.animal_parsing_batch_benchmark.v1"
INTERPRETATION = "FIXED_OXFORD_DOG_BATCH_EQUIVALENCE_AND_THROUGHPUT_NOT_BIOMETRIC_VALIDATION"
_SAMPLE_COUNT = 512
_REPEAT_COUNT = 3
_BATCH_SIZES = (4, 8, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--instance-model-directory", type=Path, required=True)
    parser.add_argument("--instance-model-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report = run_benchmark(
        dataset_root=args.dataset_root,
        model_directory=args.model_directory,
        model_manifest=args.model_manifest,
        instance_model_directory=args.instance_model_directory,
        instance_model_manifest=args.instance_model_manifest,
        output_directory=args.output_directory,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_ANIMAL_PARSING_BATCH_BENCHMARK",
                "output": str(args.output_directory),
                "selected_batch_size": report["selected_batch_size"],
                "report_sha256": content_sha256(report),
            },
            sort_keys=True,
        )
    )
    return 0


def run_benchmark(
    *,
    dataset_root: Path,
    model_directory: Path,
    model_manifest: Path,
    instance_model_directory: Path,
    instance_model_manifest: Path,
    output_directory: Path,
    device: str,
) -> dict[str, Any]:
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(output_directory)
    output_parent = output_directory.parent.resolve(strict=True)
    root = dataset_root.resolve(strict=True)
    if dataset_root.is_symlink() or not root.is_dir():
        raise ValueError("Oxford benchmark dataset root must be a regular directory")
    samples = _select_samples(root)
    sample_bindings = tuple(_sample_binding(root, sample) for sample in samples)
    foreground = ForegroundSegmentationArtifact.load(
        model_directory=model_directory,
        manifest_bundle_path=model_manifest,
    )
    instance = InstanceSegmentationArtifact.load(
        model_directory=instance_model_directory,
        manifest_bundle_path=instance_model_manifest,
    )
    policy = AnimalParsingPolicy()
    results = []
    for batch_size in _BATCH_SIZES:
        runtime = AnimalParsingRuntime(
            instance_runtime=AnimalInstanceSegmentationRuntime(
                artifact=instance,
                device=device,
                mask_threshold=policy.foreground_threshold,
            ),
            foreground_runtime=ForegroundSegmentationRuntime(
                artifact=foreground,
                device=device,
                threshold=policy.foreground_threshold,
            ),
            policy=policy,
        )
        _run_pass(runtime, root=root, samples=samples, batch_size=batch_size, measure=False)
        repeats = tuple(
            _run_pass(
                runtime,
                root=root,
                samples=samples,
                batch_size=batch_size,
                measure=True,
            )
            for _ in range(_REPEAT_COUNT)
        )
        results.append(_batch_result(batch_size, repeats))
        del runtime
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()

    reference = results[0]
    for item in results:
        item["comparison_to_batch_4"] = _compare_batch_results(reference, item)
    eligible = tuple(
        item
        for item in results
        if item["comparison_to_batch_4"]["semantic_prediction_mismatch_count"] == 0
        and item["comparison_to_batch_4"]["terminal_decision_mismatch_count"] == 0
    )
    if not eligible:
        raise AssertionError("batch 4 must remain equivalent to itself")
    selected = min(eligible, key=lambda item: item["parser_seconds_median"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "interpretation": INTERPRETATION,
        "selection": {
            "dataset": "Oxford-IIIT Pet",
            "split": "test",
            "species": "dog",
            "algorithm": "SHA256_DOMAIN_SEPARATED_ELIGIBLE_PREFIX_V1",
            "sample_count": _SAMPLE_COUNT,
            "sample_names_sha256": content_sha256([item.name for item in samples]),
            "sample_bindings_sha256": content_sha256(sample_bindings),
        },
        "protocol": {
            "batch_sizes": list(_BATCH_SIZES),
            "warmup_pass_count": 1,
            "measured_repeat_count": _REPEAT_COUNT,
            "warmup_sample_count_per_batch": _SAMPLE_COUNT,
            "prediction_equivalence": "EXACT_ARRAY_AND_METADATA_SHA256",
            "terminal_decision_equivalence": "EXACT_CONTENT_SHA256",
            "parser_timing": "PRELOADED_IMAGES_PREDICT_BATCH_ONLY",
            "end_to_end_timing": "IMAGE_LOAD_PLUS_PARSER",
        },
        "policy": policy.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "models": {
            "foreground_manifest_sha256": foreground.manifest.manifest_sha256,
            "foreground_bundle_sha256": foreground.bundle_sha256,
            "instance_manifest_sha256": instance.manifest.manifest_sha256,
            "instance_bundle_sha256": instance.bundle_sha256,
        },
        "environment": _environment(device),
        "results": results,
        "semantic_predictions_equivalent_across_batch_sizes": all(
            item["comparison_to_batch_4"]["semantic_prediction_mismatch_count"] == 0
            for item in results
        ),
        "exact_numerical_predictions_equivalent_across_batch_sizes": len(
            {item["exact_prediction_fingerprint_sha256"] for item in results}
        )
        == 1,
        "terminal_decisions_equivalent_across_batch_sizes": all(
            item["comparison_to_batch_4"]["terminal_decision_mismatch_count"] == 0
            for item in results
        ),
        "selected_batch_size": selected["batch_size"],
        "selection_rule": "LOWEST_MEDIAN_PARSER_SECONDS_AMONG_EXACT_EQUIVALENT_BATCHES",
    }
    with TemporaryDirectory(prefix=".parser-benchmark-", dir=output_parent) as temporary:
        staging = Path(temporary) / "benchmark"
        staging.mkdir(mode=0o700)
        _write_new(staging / "report.json", json_document_bytes(report))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_parent / output_directory.name)
    fsync_directory(output_parent / output_directory.name)
    fsync_directory(output_parent)
    return report


def _select_samples(root: Path) -> tuple[OxfordPetSample, ...]:
    candidates = _load_split_samples(root, split="test", species="dog", sample_count=None)
    eligible, _ = _preflight_samples(candidates, dataset_root=root)
    ordered = sorted(
        eligible,
        key=lambda item: hashlib.sha256(
            b"cvi.animal_parsing_batch_benchmark.v1\0" + item.name.encode("ascii")
        ).digest(),
    )
    if len(ordered) < _SAMPLE_COUNT:
        raise ValueError("Oxford benchmark has insufficient eligible dog images")
    return tuple(sorted(ordered[:_SAMPLE_COUNT], key=lambda item: item.name))


def _sample_binding(root: Path, sample: OxfordPetSample) -> dict[str, Any]:
    path = root / "images" / f"{sample.name}.jpg"
    return {
        "sample_name": sample.name,
        "relative_path": path.relative_to(root).as_posix(),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _run_pass(
    runtime: AnimalParsingRuntime,
    *,
    root: Path,
    samples: Sequence[OxfordPetSample],
    batch_size: int,
    measure: bool,
) -> dict[str, Any] | None:
    import torch

    if runtime.instance_runtime._device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    end_to_end_started = time.perf_counter()
    parser_seconds = 0.0
    exact_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = []
        for sample in chunk:
            with Image.open(root / "images" / f"{sample.name}.jpg") as opened:
                images.append(opened.convert("RGB"))
        if runtime.instance_runtime._device.type == "cuda":
            torch.cuda.synchronize()
        parser_started = time.perf_counter()
        predictions = runtime.predict_batch(
            tuple(images),
            instance_batch_size=batch_size,
            foreground_batch_size=batch_size,
        )
        if runtime.instance_runtime._device.type == "cuda":
            torch.cuda.synchronize()
        parser_seconds += time.perf_counter() - parser_started
        if measure:
            for sample, prediction in zip(chunk, predictions, strict=True):
                exact_rows.append(_exact_prediction_fingerprint(prediction))
                semantic_rows.append(_semantic_prediction_fingerprint(prediction))
                decision_rows.append(
                    _decision_fingerprint(
                        sample.name,
                        prediction,
                        policy_sha256=runtime.policy.policy_sha256,
                    )
                )
        del predictions, images
    if runtime.instance_runtime._device.type == "cuda":
        torch.cuda.synchronize()
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    if not measure:
        return None
    exact_prediction_fingerprint = content_sha256(exact_rows)
    semantic_prediction_fingerprint = content_sha256(semantic_rows)
    terminal_fingerprint = content_sha256(decision_rows)
    return {
        "parser_seconds": parser_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "parser_images_per_second": len(samples) / parser_seconds,
        "end_to_end_images_per_second": len(samples) / end_to_end_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
        "exact_prediction_fingerprint_sha256": exact_prediction_fingerprint,
        "semantic_prediction_fingerprint_sha256": semantic_prediction_fingerprint,
        "terminal_decision_fingerprint_sha256": terminal_fingerprint,
        "sample_semantic_prediction_sha256s": [
            content_sha256(item) for item in semantic_rows
        ],
        "sample_terminal_decision_sha256s": [
            content_sha256(item) for item in decision_rows
        ],
    }


def _exact_prediction_fingerprint(prediction: Any) -> dict[str, Any]:
    return {
        "source_width": prediction.source_width,
        "source_height": prediction.source_height,
        "policy_sha256": prediction.policy_sha256,
        "instances": [
            {
                "instance_index": item.instance_index,
                "query_index": item.query_index,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "class_score": item.class_score,
                "detector_box_xyxy": list(item.detector_box_xyxy),
                "refinement_box_xyxy": list(item.refinement_box_xyxy),
                "mask_box_xyxy": (
                    None if item.mask_box_xyxy is None else list(item.mask_box_xyxy)
                ),
                "quality": {
                    "state": item.quality.state,
                    "reasons": list(item.quality.reasons),
                    "flags": list(item.quality.flags),
                    "semantic_shape_iou": item.quality.semantic_shape_iou,
                    "ownership_retention": item.quality.ownership_retention,
                    "foreground_pixels": item.quality.foreground_pixels,
                    "component_count": item.quality.component_count,
                    "touches_source_border": item.quality.touches_source_border,
                },
                "arrays": {
                    name: hashlib.sha256(
                        np.ascontiguousarray(getattr(item, name)).tobytes()
                    ).hexdigest()
                    for name in (
                        "instance_probability",
                        "foreground_probability",
                        "ownership_probability",
                        "hard_mask",
                    )
                },
            }
            for item in prediction.instances
        ],
    }


def _semantic_prediction_fingerprint(prediction: Any) -> dict[str, Any]:
    return {
        "source_width": prediction.source_width,
        "source_height": prediction.source_height,
        "policy_sha256": prediction.policy_sha256,
        "instances": [
            {
                "instance_index": item.instance_index,
                "query_index": item.query_index,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "detector_box_xyxy": list(item.detector_box_xyxy),
                "refinement_box_xyxy": list(item.refinement_box_xyxy),
                "mask_box_xyxy": (
                    None if item.mask_box_xyxy is None else list(item.mask_box_xyxy)
                ),
                "quality_state": item.quality.state,
                "quality_reasons": list(item.quality.reasons),
                "quality_flags": list(item.quality.flags),
                "foreground_pixels": item.quality.foreground_pixels,
                "component_count": item.quality.component_count,
                "touches_source_border": item.quality.touches_source_border,
                "hard_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(item.hard_mask).tobytes()
                ).hexdigest(),
            }
            for item in prediction.instances
        ],
    }


def _decision_fingerprint(
    sample_name: str, prediction: Any, *, policy_sha256: str
) -> dict[str, Any]:
    token = hashlib.sha256(f"benchmark:{sample_name}".encode("ascii")).hexdigest()
    row = {
        "schema_version": "cvi.full128_route_plan_record.v3",
        "sample_token": token,
        "dataset_name": "oxford-pets-dog",
        "record_sha256": hashlib.sha256(
            f"benchmark-record:{sample_name}".encode("ascii")
        ).hexdigest(),
    }
    receipt = {
        "parser_cache_key": hashlib.sha256(
            f"benchmark-cache:{sample_name}".encode("ascii")
        ).hexdigest(),
        "receipt_sha256": hashlib.sha256(
            f"benchmark-receipt:{sample_name}".encode("ascii")
        ).hexdigest(),
        "prediction_sha256": content_sha256(_exact_prediction_fingerprint(prediction)),
        "runtime": {"parser_policy_sha256": policy_sha256},
    }
    decision = _parser_decisions((row,), prediction=prediction, cache_receipt=receipt)[token]
    return {
        "actual_route": decision.actual_route.value,
        "source_view_scope": decision.source_view_scope.value,
        "selected_instance_index": (
            None if decision.association is None else decision.association.instance_index
        ),
        "selection": decision.parser_lineage["selection"],
        "terminal_reason": decision.terminal_reason,
    }


def _batch_result(batch_size: int, repeats: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
    measured = tuple(item for item in repeats if item is not None)
    exact_prediction_hashes = {
        item["exact_prediction_fingerprint_sha256"] for item in measured
    }
    semantic_prediction_hashes = {
        item["semantic_prediction_fingerprint_sha256"] for item in measured
    }
    decision_hashes = {item["terminal_decision_fingerprint_sha256"] for item in measured}
    semantic_sample_hashes = {
        tuple(item["sample_semantic_prediction_sha256s"]) for item in measured
    }
    decision_sample_hashes = {
        tuple(item["sample_terminal_decision_sha256s"]) for item in measured
    }
    if (
        len(measured) != _REPEAT_COUNT
        or len(semantic_prediction_hashes) != 1
        or len(decision_hashes) != 1
        or len(semantic_sample_hashes) != 1
        or len(decision_sample_hashes) != 1
    ):
        raise RuntimeError("animal parsing benchmark repeats differ")
    parser_values = [item["parser_seconds"] for item in measured]
    end_to_end_values = [item["end_to_end_seconds"] for item in measured]
    return {
        "batch_size": batch_size,
        "repeats": list(measured),
        "parser_seconds_median": statistics.median(parser_values),
        "parser_seconds_mean": statistics.fmean(parser_values),
        "end_to_end_seconds_median": statistics.median(end_to_end_values),
        "end_to_end_seconds_mean": statistics.fmean(end_to_end_values),
        "parser_images_per_second_median": _SAMPLE_COUNT
        / statistics.median(parser_values),
        "end_to_end_images_per_second_median": _SAMPLE_COUNT
        / statistics.median(end_to_end_values),
        "peak_cuda_allocated_bytes_max": _optional_max(
            item["peak_cuda_allocated_bytes"] for item in measured
        ),
        "peak_cuda_reserved_bytes_max": _optional_max(
            item["peak_cuda_reserved_bytes"] for item in measured
        ),
        "exact_prediction_fingerprint_sha256s": sorted(exact_prediction_hashes),
        "exact_numerical_predictions_repeatable": len(exact_prediction_hashes) == 1,
        "exact_prediction_fingerprint_sha256": (
            next(iter(exact_prediction_hashes))
            if len(exact_prediction_hashes) == 1
            else None
        ),
        "semantic_prediction_fingerprint_sha256": next(
            iter(semantic_prediction_hashes)
        ),
        "terminal_decision_fingerprint_sha256": next(iter(decision_hashes)),
        "sample_semantic_prediction_sha256s": list(
            next(iter(semantic_sample_hashes))
        ),
        "sample_terminal_decision_sha256s": list(next(iter(decision_sample_hashes))),
    }


def _compare_batch_results(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    semantic_mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(
                reference["sample_semantic_prediction_sha256s"],
                candidate["sample_semantic_prediction_sha256s"],
                strict=True,
            )
        )
        if left != right
    ]
    decision_mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(
                reference["sample_terminal_decision_sha256s"],
                candidate["sample_terminal_decision_sha256s"],
                strict=True,
            )
        )
        if left != right
    ]
    return {
        "semantic_prediction_mismatch_count": len(semantic_mismatches),
        "semantic_prediction_mismatch_indices": semantic_mismatches,
        "terminal_decision_mismatch_count": len(decision_mismatches),
        "terminal_decision_mismatch_indices": decision_mismatches,
    }


def _optional_max(values: Any) -> int | None:
    retained = [value for value in values if value is not None]
    return max(retained) if retained else None


def _environment(device: str) -> dict[str, Any]:
    import torch

    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "python": platform.python_version(),
        "libraries": {
            name: importlib.metadata.version(distribution)
            for name, distribution in {
                "numpy": "numpy",
                "pillow": "Pillow",
                "torch": "torch",
                "torchvision": "torchvision",
                "transformers": "transformers",
            }.items()
        },
    }


def _write_new(path: Path, payload: bytes) -> None:
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
                raise OSError("benchmark report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
