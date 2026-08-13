"""DINOv2 patch-token A/F/N candidate segmentation and embeddings."""

from __future__ import annotations

import io
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from artifact_contracts.dinov2_contract import Dinov2LocalArtifactContract
from data_pipeline.types import UnifiedCanidSample
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file

MANIFEST_SCHEMA = "cvi.dinov2_region_candidates.v1"
BUNDLE_SCHEMA = "cvi.dinov2_region_candidates_bundle.v1"
INTERPRETATION = (
    "RESEARCH_ONLY_DINOV2_PATCH_TOKEN_MODEL_GENERATED_CANDIDATES_"
    "NOT_VERIFIED_SEMANTIC_SEGMENTATION_OR_BIOMETRIC_VALIDATION"
)
PATCH_GRID = 16
EMBEDDING_DIMENSION = 384
UNAVAILABLE_MASK_VALUE = 255
_BODY_SOURCE_DATASETS = frozenset({"ap10k-dog", "sibetan", "yt-bb-dog"})
_FACE_SOURCE_DATASETS = frozenset({"dogfacenet224", "dogflw", "mpdd"})
_REGIONS = ("A", "F", "N")


class Dinov2RegionRuntime:
    """Receipt-bound DINOv2-small dense patch feature runtime."""

    def __init__(
        self,
        *,
        model_directory: Path,
        weight_intake_bundle: Path,
        preprocessor_intake_bundle: Path,
        device: str,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("DINOv2 region runtime device must be cpu or cuda")
        self.contract = Dinov2LocalArtifactContract.load(
            model_directory=model_directory,
            weight_intake_bundle=weight_intake_bundle,
            preprocessor_intake_bundle=preprocessor_intake_bundle,
        )
        import torch
        from transformers import Dinov2Model

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.contract.revalidate_local_files()
        self.device = torch.device(device)
        self.model = Dinov2Model.from_pretrained(
            str(self.contract.model_directory),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.contract.revalidate_local_files()

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "model_id": self.contract.weight_source.source_model_id,
            "source_revision": self.contract.weight_source.source_revision,
            "model_sha256": self.contract.model_sha256,
            "weight_intake_receipt_sha256": self.contract.weight_receipt_sha256,
            "preprocessor_sha256": self.contract.preprocessor_sha256,
            "preprocessor_intake_receipt_sha256": (
                self.contract.preprocessor_receipt_sha256
            ),
            "config_sha256": self.contract.config_sha256,
            "device": self.device.type,
            "output": "FINAL_LAYER_PATCH_TOKENS_16X16X384",
        }

    def patch_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("DINOv2 region runtime requires at least one image")
        import torch

        arrays = np.stack(
            [
                np.asarray(
                    image.convert("RGB").resize(
                        (224, 224), Image.Resampling.BICUBIC
                    ),
                    dtype=np.uint8,
                ).copy()
                for image in images
            ]
        )
        tensor = torch.from_numpy(arrays).permute(0, 3, 1, 2).to(
            self.device, dtype=torch.float32
        )
        tensor.div_(255.0)
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(pixel_values=(tensor - mean) / std)
        tokens = output.last_hidden_state[:, 1:]
        if tokens.shape[1:] != (PATCH_GRID * PATCH_GRID, EMBEDDING_DIMENSION):
            raise RuntimeError("DINOv2 patch-token shape differs")
        tokens = torch.nn.functional.normalize(tokens.float(), dim=2)
        return (
            tokens.reshape(-1, PATCH_GRID, PATCH_GRID, EMBEDDING_DIMENSION)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def derive_patch_region_candidates(
    features: np.ndarray,
    *,
    dataset_name: str,
    image_width: int,
    image_height: int,
    dog_box: Sequence[float] | None,
    body_keypoints: Mapping[str, Sequence[float]] | None,
    face_box: Sequence[float] | None,
    face_landmarks: Mapping[str, Sequence[float]] | None,
    geometry_source: str,
) -> dict[str, Any]:
    """Derive auditable patch masks; outputs remain model-generated candidates."""

    values = np.asarray(features, dtype=np.float32)
    if values.shape != (PATCH_GRID, PATCH_GRID, EMBEDDING_DIMENSION):
        raise ValueError("region features must be [16,16,384]")
    if not np.isfinite(values).all():
        raise ValueError("region features must be finite")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("source image dimensions must be positive")

    body_roi = _normalized_box(dog_box, image_width, image_height)
    body_points = _normalized_points(body_keypoints, image_width, image_height)
    if dataset_name in _BODY_SOURCE_DATASETS:
        a_mask, a_confidence = _foreground_mask(
            values,
            roi=body_roi or (0.0, 0.0, 1.0, 1.0),
            seeds=tuple(body_points.values()) or ((0.5, 0.5),),
            retained_fraction=0.68,
        )
        a = _available_region(
            a_mask.astype(np.uint8),
            confidence=a_confidence,
            geometry_source=geometry_source,
            semantic_target="FULL_BODY_DOG",
            classes={"0": "background", "1": "dog"},
            features=values,
        )
    else:
        a = _unavailable_region("SOURCE_REGION_DOES_NOT_CONTAIN_FULL_BODY")

    normalized_face_box = _normalized_box(face_box, image_width, image_height)
    if normalized_face_box is None:
        normalized_face_box = _face_box_from_points(body_points)
    if normalized_face_box is None and dataset_name in _FACE_SOURCE_DATASETS:
        normalized_face_box = (0.0, 0.0, 1.0, 1.0)
    face_points = _normalized_points(face_landmarks, image_width, image_height)
    if normalized_face_box is None:
        f = _unavailable_region("FACE_GEOMETRY_UNAVAILABLE")
        n = _unavailable_region("NOSE_GEOMETRY_UNAVAILABLE")
    else:
        face_seed = _face_seed(body_points, face_points, normalized_face_box)
        head_support, head_confidence = _foreground_mask(
            values,
            roi=normalized_face_box,
            seeds=(face_seed,),
            retained_fraction=0.74,
        )
        f_mask = _partition_face_mask(head_support, normalized_face_box)
        if set(np.unique(f_mask)) >= {0, 1, 2, 3}:
            f = _available_region(
                f_mask,
                confidence=head_confidence * 0.75,
                geometry_source=geometry_source,
                semantic_target="EARS_FACE_NECK",
                classes={
                    "0": "background",
                    "1": "ears",
                    "2": "face",
                    "3": "neck",
                },
                features=values,
            )
        else:
            f = _unavailable_region("F_CLASS_SUPPORT_INCOMPLETE")
        nose_seed = _nose_seed(body_points, face_points, normalized_face_box)
        n_mask, n_confidence = _nose_mask(values, normalized_face_box, nose_seed)
        if set(np.unique(n_mask)) >= {0, 1, 2}:
            n = _available_region(
                n_mask,
                confidence=n_confidence,
                geometry_source=geometry_source,
                semantic_target="NOSE",
                classes={"0": "context", "1": "nasal_surface", "2": "nostril"},
                features=values,
            )
        else:
            n = _unavailable_region("N_CLASS_SUPPORT_INCOMPLETE")
    return {"A": a, "F": f, "N": n}


def produce_dataset_region_candidates(
    samples: Sequence[UnifiedCanidSample],
    *,
    data_root: Path,
    output_dir: Path,
    runtime: Dinov2RegionRuntime,
    pose_adapter: Any | None = None,
    batch_size: int = 32,
    maximum_samples: int | None = None,
) -> dict[str, Any]:
    """Run DINO and optional pose geometry over one complete adapter dataset."""

    if not samples:
        raise ValueError("region candidate production requires samples")
    datasets = {sample.dataset_name for sample in samples}
    versions = {sample.dataset_version for sample in samples}
    if len(datasets) != 1 or len(versions) != 1:
        raise ValueError("region candidate production requires one dataset/version")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("region candidate batch_size must be positive")
    if maximum_samples is not None and (
        isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples <= 0
    ):
        raise ValueError("maximum_samples must be positive or null")
    root = data_root.resolve(strict=True)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite region candidates: {output_dir}")
    selected = tuple(sorted(samples, key=lambda item: item.sample_id))
    if maximum_samples is not None:
        selected = selected[:maximum_samples]
    count = len(selected)
    masks = {
        region: np.full(
            (count, PATCH_GRID, PATCH_GRID),
            UNAVAILABLE_MASK_VALUE,
            dtype=np.uint8,
        )
        for region in _REGIONS
    }
    embeddings = {
        region: np.full((count, EMBEDDING_DIMENSION), np.nan, dtype=np.float16)
        for region in _REGIONS
    }
    records: list[dict[str, Any]] = []
    for offset in range(0, count, batch_size):
        rows = selected[offset : offset + batch_size]
        images = [_read_bound_image(root, sample) for sample in rows]
        dense = runtime.patch_features(images)
        for local_index, (sample, image, features) in enumerate(
            zip(rows, images, dense, strict=True)
        ):
            geometry = _sample_geometry(sample)
            if pose_adapter is not None and (
                geometry["dog_box"] is None or geometry["body_keypoints"] is None
            ):
                prediction = pose_adapter.detect(image, image_id=sample.sample_id)
                if prediction.dog_boxes:
                    best = max(
                        range(len(prediction.dog_boxes)),
                        key=lambda index: prediction.dog_boxes[index].confidence,
                    )
                    box = prediction.dog_boxes[best]
                    geometry["dog_box"] = (box.x1, box.y1, box.x2, box.y2)
                    if best < len(prediction.body_keypoints):
                        geometry["body_keypoints"] = {
                            name: (point.x, point.y, point.confidence)
                            for name, point in prediction.body_keypoints[
                                best
                            ].keypoints.items()
                        }
                    geometry["source"] = "AP10K_POSE_STUDENT"
            candidates = derive_patch_region_candidates(
                features,
                dataset_name=sample.dataset_name,
                image_width=sample.width,
                image_height=sample.height,
                dog_box=geometry["dog_box"],
                body_keypoints=geometry["body_keypoints"],
                face_box=geometry["face_box"],
                face_landmarks=geometry["face_landmarks"],
                geometry_source=geometry["source"],
            )
            index = offset + local_index
            region_rows: dict[str, Any] = {}
            for region in _REGIONS:
                candidate = candidates[region]
                if candidate["state"] == "AVAILABLE":
                    masks[region][index] = candidate.pop("mask")
                    embeddings[region][index] = candidate.pop("embedding").astype(
                        np.float16
                    )
                region_rows[region] = candidate
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset_name": sample.dataset_name,
                    "dataset_version": sample.dataset_version,
                    "image_path": sample.image_path,
                    "image_sha256": sample.image_sha256,
                    "image_width": sample.width,
                    "image_height": sample.height,
                    "registered_identity_id": sample.registered_identity_id,
                    "raw_identity_id": sample.raw_identity_id,
                    "capture_group_id": sample.capture_group_id,
                    "split_role": sample.split_role,
                    "array_index": index,
                    "regions": region_rows,
                }
            )
    arrays = {
        **{f"{region}_masks": masks[region] for region in _REGIONS},
        **{f"{region}_embeddings": embeddings[region] for region in _REGIONS},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        arrays_path = staging / "region_candidates.npz"
        np.savez_compressed(arrays_path, **arrays)
        arrays_binding = _file_binding(arrays_path)
        body = {
            "schema_version": MANIFEST_SCHEMA,
            "dataset_name": next(iter(datasets)),
            "dataset_version": next(iter(versions)),
            "record_count": len(records),
            "complete_adapter_dataset": maximum_samples is None,
            "model_binding": runtime.binding,
            "pose_model_binding": (
                None if pose_adapter is None else pose_adapter.to_dict()
            ),
            "algorithm": _algorithm_contract(),
            "arrays": {
                "relative_path": arrays_path.name,
                **arrays_binding,
                "mask_shape": [len(records), PATCH_GRID, PATCH_GRID],
                "embedding_shape": [len(records), EMBEDDING_DIMENSION],
                "unavailable_mask_value": UNAVAILABLE_MASK_VALUE,
            },
            "records": records,
            "interpretation": INTERPRETATION,
        }
        bundle = {
            "schema_version": BUNDLE_SCHEMA,
            "manifest_sha256": content_sha256(body),
            "manifest": body,
        }
        write_private_json_bundle(((staging / "region_candidates.json", bundle),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return bundle


def read_region_candidates(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    document = read_strict_json_document(
        path,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    bundle = document.payload
    if (
        set(bundle) != {"schema_version", "manifest_sha256", "manifest"}
        or bundle["schema_version"] != BUNDLE_SCHEMA
        or not isinstance(bundle["manifest"], dict)
        or content_sha256(bundle["manifest"]) != bundle["manifest_sha256"]
    ):
        raise ValueError("DINOv2 region candidate bundle differs")
    manifest = bundle["manifest"]
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("interpretation") != INTERPRETATION
        or manifest.get("algorithm") != _algorithm_contract()
        or manifest.get("record_count") != len(manifest.get("records", ()))
    ):
        raise ValueError("DINOv2 region candidate manifest differs")
    arrays_meta = manifest["arrays"]
    arrays_path = path.parent / arrays_meta["relative_path"]
    retained = read_retained_regular_file(
        arrays_path,
        expected_bytes=arrays_meta["byte_size"],
        expected_sha256=arrays_meta["sha256"],
        maximum_bytes=8_589_934_592,
        capture_payload=False,
        subject="DINOv2 region candidate arrays",
    )
    if retained.sha256 != arrays_meta["sha256"]:
        raise ValueError("region candidate array digest differs")
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    expected_names = {
        *(f"{region}_masks" for region in _REGIONS),
        *(f"{region}_embeddings" for region in _REGIONS),
    }
    if set(arrays) != expected_names:
        raise ValueError("region candidate array names differ")
    count = manifest["record_count"]
    for region in _REGIONS:
        if arrays[f"{region}_masks"].shape != (count, PATCH_GRID, PATCH_GRID):
            raise ValueError("region candidate mask array shape differs")
        if arrays[f"{region}_embeddings"].shape != (
            count,
            EMBEDDING_DIMENSION,
        ):
            raise ValueError("region candidate embedding array shape differs")
    return manifest, arrays


def _sample_geometry(sample: UnifiedCanidSample) -> dict[str, Any]:
    return {
        "dog_box": sample.dog_boxes_xyxy,
        "body_keypoints": sample.body_keypoints,
        "face_box": sample.face_box_xyxy,
        "face_landmarks": sample.face_landmarks,
        "source": "PUBLISHER_ANNOTATION" if (
            sample.dog_boxes_xyxy is not None
            or sample.body_keypoints is not None
            or sample.face_box_xyxy is not None
            or sample.face_landmarks is not None
        ) else "SOURCE_REGION_PRIOR",
    }


def _read_bound_image(root: Path, sample: UnifiedCanidSample) -> Image.Image:
    relative = Path(sample.image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("region source image path is unsafe")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("region source image is unsafe")
    retained = read_retained_regular_file(
        path,
        expected_sha256=sample.image_sha256,
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="region source image",
    )
    assert retained.payload is not None
    with Image.open(io.BytesIO(retained.payload)) as opened:
        if opened.size != (sample.width, sample.height):
            raise ValueError("region source image dimensions differ")
        image = opened.convert("RGB")
        image.load()
    return image


def _available_region(
    mask: np.ndarray,
    *,
    confidence: float,
    geometry_source: str,
    semantic_target: str,
    classes: Mapping[str, str],
    features: np.ndarray,
) -> dict[str, Any]:
    support = mask > 0
    embedding = features[support].mean(axis=0)
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 1e-8:
        return _unavailable_region("ZERO_NORM_PATCH_EMBEDDING")
    return {
        "state": "AVAILABLE",
        "qualification": "MODEL_GENERATED_CANDIDATE",
        "semantic_target": semantic_target,
        "class_map": dict(classes),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "support_fraction": float(support.mean()),
        "geometry_source": geometry_source,
        "coordinate_space": "DINOV2_PATCH_GRID_16X16",
        "mask": np.asarray(mask, dtype=np.uint8),
        "embedding": np.asarray(embedding / norm, dtype=np.float32),
    }


def _unavailable_region(reason: str) -> dict[str, Any]:
    return {"state": "UNAVAILABLE", "reason": reason}


def _foreground_mask(
    features: np.ndarray,
    *,
    roi: tuple[float, float, float, float],
    seeds: Sequence[tuple[float, float]],
    retained_fraction: float,
) -> tuple[np.ndarray, float]:
    yy, xx = np.mgrid[0:PATCH_GRID, 0:PATCH_GRID]
    x = (xx + 0.5) / PATCH_GRID
    y = (yy + 0.5) / PATCH_GRID
    x1, y1, x2, y2 = roi
    inside = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    if not np.any(inside):
        return np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool), 0.0
    seed_mask = np.zeros_like(inside)
    for seed_x, seed_y in seeds:
        distance = (x - seed_x) ** 2 + (y - seed_y) ** 2
        seed_mask |= distance <= (1.5 / PATCH_GRID) ** 2
    seed_mask &= inside
    if not np.any(seed_mask):
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        nearest = np.argmin((x - center_x) ** 2 + (y - center_y) ** 2)
        seed_mask.flat[int(nearest)] = True
    border = (xx == 0) | (yy == 0) | (xx == PATCH_GRID - 1) | (yy == PATCH_GRID - 1)
    background = (~inside) | (border & ~seed_mask)
    if not np.any(background):
        background = border & ~seed_mask
    foreground_prototype = features[seed_mask].mean(axis=0)
    background_prototype = features[background].mean(axis=0)
    foreground_prototype /= max(float(np.linalg.norm(foreground_prototype)), 1e-8)
    background_prototype /= max(float(np.linalg.norm(background_prototype)), 1e-8)
    score = features @ foreground_prototype - features @ background_prototype
    spatial = np.maximum.reduce(
        [
            np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / 0.08)
            for sx, sy in seeds
        ]
    )
    score = score + 0.20 * spatial
    threshold = float(np.quantile(score[inside], 1.0 - retained_fraction))
    mask = (score >= threshold) & inside
    mask |= seed_mask
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    mask = _largest_component(mask, seed_mask)
    separation = float(score[mask].mean() - score[~mask].mean()) if np.any(~mask) else 0.0
    confidence = 1.0 / (1.0 + math.exp(-4.0 * separation))
    return mask, confidence


def _largest_component(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask
    seed_labels = labels[seeds]
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size:
        selected = int(Counter(seed_labels.tolist()).most_common(1)[0][0])
    else:
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == selected


def _partition_face_mask(
    support: np.ndarray, roi: tuple[float, float, float, float]
) -> np.ndarray:
    yy, xx = np.mgrid[0:PATCH_GRID, 0:PATCH_GRID]
    x = (xx + 0.5) / PATCH_GRID
    y = (yy + 0.5) / PATCH_GRID
    x1, y1, x2, y2 = roi
    relative_x = (x - x1) / max(x2 - x1, 1e-6)
    relative_y = (y - y1) / max(y2 - y1, 1e-6)
    ears = support & (relative_y < 0.45) & (
        (relative_x < 0.30) | (relative_x > 0.70)
    )
    neck = support & (relative_y >= 0.72)
    face = support & ~ears & ~neck
    result = np.zeros_like(support, dtype=np.uint8)
    result[ears] = 1
    result[face] = 2
    result[neck] = 3
    return result


def _nose_mask(
    features: np.ndarray,
    roi: tuple[float, float, float, float],
    seed: tuple[float, float],
) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = roi
    width, height = x2 - x1, y2 - y1
    nose_roi = (
        max(x1, seed[0] - 0.20 * width),
        max(y1, seed[1] - 0.16 * height),
        min(x2, seed[0] + 0.20 * width),
        min(y2, seed[1] + 0.16 * height),
    )
    support, confidence = _foreground_mask(
        features, roi=nose_roi, seeds=(seed,), retained_fraction=0.78
    )
    yy, xx = np.mgrid[0:PATCH_GRID, 0:PATCH_GRID]
    x = (xx + 0.5) / PATCH_GRID
    y = (yy + 0.5) / PATCH_GRID
    nostril = support & (
        (
            ((x - (seed[0] - 0.07 * width)) / max(0.08 * width, 1e-6)) ** 2
            + ((y - (seed[1] + 0.03 * height)) / max(0.07 * height, 1e-6)) ** 2
            <= 1.0
        )
        | (
            ((x - (seed[0] + 0.07 * width)) / max(0.08 * width, 1e-6)) ** 2
            + ((y - (seed[1] + 0.03 * height)) / max(0.07 * height, 1e-6)) ** 2
            <= 1.0
        )
    )
    result = np.zeros_like(support, dtype=np.uint8)
    result[support] = 1
    result[nostril] = 2
    return result, confidence * 0.65


def _normalized_box(
    box: Sequence[float] | None, width: int, height: int
) -> tuple[float, float, float, float] | None:
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return (
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    )


def _normalized_points(
    points: Mapping[str, Sequence[float]] | None, width: int, height: int
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for name, value in (points or {}).items():
        if len(value) < 2:
            continue
        x, y = float(value[0]) / width, float(value[1]) / height
        confidence = float(value[2]) if len(value) > 2 else 1.0
        if math.isfinite(x) and math.isfinite(y) and confidence > 0.0:
            result[name] = (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
    return result


def _face_box_from_points(
    points: Mapping[str, tuple[float, float]]
) -> tuple[float, float, float, float] | None:
    anchors = [points[name] for name in ("left_eye", "right_eye", "nose_center", "neck") if name in points]
    if len(anchors) < 2:
        return None
    xs = [point[0] for point in anchors]
    ys = [point[1] for point in anchors]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.12)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys[:-1] if len(ys) > 2 else ys))
    return (
        max(0.0, center_x - 1.4 * span),
        max(0.0, center_y - 1.0 * span),
        min(1.0, center_x + 1.4 * span),
        min(1.0, center_y + 1.6 * span),
    )


def _face_seed(
    body: Mapping[str, tuple[float, float]],
    face: Mapping[str, tuple[float, float]],
    roi: tuple[float, float, float, float],
) -> tuple[float, float]:
    nose = body.get("nose_center")
    if nose is None:
        nose_points = [point for name, point in face.items() if name in {"face46.25", "face46.26", "face46.27"}]
        nose = tuple(np.mean(nose_points, axis=0)) if nose_points else None
    if nose is not None:
        return float(nose[0]), float(nose[1])
    x1, y1, x2, y2 = roi
    return (x1 + x2) / 2.0, y1 + 0.55 * (y2 - y1)


def _nose_seed(
    body: Mapping[str, tuple[float, float]],
    face: Mapping[str, tuple[float, float]],
    roi: tuple[float, float, float, float],
) -> tuple[float, float]:
    if "nose_center" in body:
        return body["nose_center"]
    nose_points = [
        point
        for name, point in face.items()
        if name in {"face46.25", "face46.26", "face46.27", "face46.32", "face46.33", "face46.34", "face46.35"}
    ]
    if nose_points:
        value = np.mean(nose_points, axis=0)
        return float(value[0]), float(value[1])
    x1, y1, x2, y2 = roi
    return (x1 + x2) / 2.0, y1 + 0.62 * (y2 - y1)


def _algorithm_contract() -> dict[str, Any]:
    return {
        "schema_version": "cvi.dinov2_region_candidate_algorithm.v1",
        "input_resize": "PIL_RGB_BICUBIC_224X224",
        "normalization": "IMAGENET_MEAN_STD",
        "patch_grid": [PATCH_GRID, PATCH_GRID],
        "patch_embedding_dimension": EMBEDDING_DIMENSION,
        "foreground": "COSINE_FOREGROUND_MINUS_BORDER_BACKGROUND_WITH_SPATIAL_SEEDS",
        "connectivity": "LARGEST_8_CONNECTED_COMPONENT_CONTAINING_SEED",
        "face_partition": "POSE_OR_FACE_ROI_RELATIVE_EARS_FACE_NECK_PRIOR",
        "nose_partition": "POSE_OR_LANDMARK_SEEDED_PATCH_SUPPORT_WITH_NOSTRIL_PRIOR",
        "embedding": "L2_NORMALIZED_MEAN_FOREGROUND_PATCH_TOKEN",
        "qualification": "MODEL_GENERATED_CANDIDATE",
    }


def _file_binding(path: Path) -> dict[str, Any]:
    retained = read_retained_regular_file(
        path, capture_payload=False, subject="region candidate array"
    )
    return {
        "sha256": retained.sha256,
        "byte_size": retained.byte_count,
    }


__all__ = [
    "Dinov2RegionRuntime",
    "derive_patch_region_candidates",
    "produce_dataset_region_candidates",
    "read_region_candidates",
]
