"""Supervised animal-instance selection for foreground parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from artifact_contracts.instance_segmentation_model import (
    InstanceSegmentationArtifact,
)


@dataclass(frozen=True, slots=True)
class AnimalInstanceCandidate:
    probability: np.ndarray
    hard_mask: np.ndarray
    source_box_xyxy: tuple[int, int, int, int]
    query_index: int
    class_id: int
    class_name: str
    class_score: float

    def __post_init__(self) -> None:
        if (
            self.probability.ndim != 2
            or self.hard_mask.shape != self.probability.shape
            or self.probability.dtype != np.float32
            or self.hard_mask.dtype != np.uint8
            or not np.isfinite(self.probability).all()
            or float(self.probability.min()) < 0.0
            or float(self.probability.max()) > 1.0
            or not set(np.unique(self.hard_mask)).issubset({0, 1})
            or not self.hard_mask.any()
        ):
            raise ValueError("animal instance candidate arrays differ")
        height, width = self.hard_mask.shape
        x1, y1, x2, y2 = self.source_box_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("animal instance candidate box differs")
        if (
            isinstance(self.query_index, bool)
            or not isinstance(self.query_index, int)
            or self.query_index < 0
            or isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
            or not isinstance(self.class_name, str)
            or not self.class_name
            or not math.isfinite(self.class_score)
            or not 0.0 <= self.class_score <= 1.0
        ):
            raise ValueError("animal instance candidate metadata differs")


class AnimalInstanceSegmentationRuntime:
    """Execute an exact RF-DETR COCO instance model without network access."""

    def __init__(
        self,
        *,
        artifact: InstanceSegmentationArtifact,
        device: str = "cpu",
        mask_threshold: float = 0.5,
        minimum_class_score: float = 0.25,
    ) -> None:
        if artifact.manifest.model_family != "RF_DETR_SEGMENTATION_COCO":
            raise ValueError("unsupported animal instance model family")
        for value, name in (
            (mask_threshold, "mask threshold"),
            (minimum_class_score, "minimum class score"),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"animal instance {name} differs")
        import torch
        from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA animal instance runtime requested without CUDA")
        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        artifact.revalidate_local_files()
        root = str(artifact.model_directory)
        self._processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        self._model = RfDetrForInstanceSegmentation.from_pretrained(
            root, local_files_only=True, dtype=self._dtype
        ).to(self._device)
        self._model.eval()
        self.artifact = artifact
        self.mask_threshold = float(mask_threshold)
        self.minimum_class_score = float(minimum_class_score)

    def predict_all(
        self,
        image: Image.Image,
        *,
        class_names: tuple[str, ...],
        duplicate_mask_iou: float = 0.8,
        maximum_instances: int = 32,
    ) -> tuple[AnimalInstanceCandidate, ...]:
        if (
            not isinstance(class_names, tuple)
            or not class_names
            or any(not isinstance(name, str) or not name for name in class_names)
            or len(class_names) != len(set(class_names))
        ):
            raise ValueError("animal instance classes must be unique canonical names")
        if (
            not math.isfinite(duplicate_mask_iou)
            or not 0.0 < duplicate_mask_iou < 1.0
        ):
            raise ValueError("animal instance duplicate IoU threshold differs")
        if (
            isinstance(maximum_instances, bool)
            or not isinstance(maximum_instances, int)
            or maximum_instances <= 0
        ):
            raise ValueError("animal instance maximum count must be positive")
        class_ids: dict[int, str] = {}
        for name in class_names:
            try:
                class_id = int(self._model.config.label2id[name])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "animal instance target class is absent from label space"
                ) from exc
            if class_id in class_ids:
                raise ValueError("animal instance target classes share a class ID")
            class_ids[class_id] = name

        rgb = image.convert("RGB")
        width, height = rgb.size
        class_probabilities, mask_probabilities, query_indices = self._infer(
            rgb,
            class_ids=tuple(sorted(class_ids)),
        )
        candidates = _all_target_candidates(
            class_probabilities=class_probabilities,
            mask_probabilities=mask_probabilities,
            class_names_by_id=class_ids,
            mask_threshold=self.mask_threshold,
            minimum_class_score=self.minimum_class_score,
            query_indices=query_indices,
        )
        retained = _suppress_duplicate_candidates(
            candidates,
            duplicate_mask_iou=duplicate_mask_iou,
            maximum_instances=maximum_instances,
        )
        if any(item.hard_mask.shape != (height, width) for item in retained):
            raise RuntimeError("animal instance source dimensions differ")
        return tuple(
            sorted(
                retained,
                key=lambda item: (
                    item.source_box_xyxy[1],
                    item.source_box_xyxy[0],
                    item.source_box_xyxy[3],
                    item.source_box_xyxy[2],
                    item.class_id,
                    -item.class_score,
                    item.query_index,
                ),
            )
        )

    def _infer(
        self, rgb: Image.Image, *, class_ids: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width, height = rgb.size
        inputs = {
            name: value.to(self._device)
            for name, value in self._processor(images=rgb, return_tensors="pt").items()
        }
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            class_probabilities = outputs.logits[0].softmax(dim=-1).float()
            target_ids = self._torch.asarray(class_ids, device=class_probabilities.device)
            selected = class_probabilities[:, target_ids].amax(dim=1) >= (
                self.minimum_class_score
            )
            query_indices = selected.nonzero(as_tuple=False)[:, 0]
            class_probabilities = class_probabilities[selected].cpu().numpy()
            if not query_indices.numel():
                return (
                    class_probabilities,
                    np.empty((0, height, width), dtype=np.float32),
                    query_indices.cpu().numpy(),
                )
            mask_probabilities = outputs.pred_masks[0].sigmoid().float()[selected]
            mask_probabilities = self._torch.nn.functional.interpolate(
                mask_probabilities.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0].cpu().numpy()
            query_indices = query_indices.cpu().numpy()
        return class_probabilities, mask_probabilities, query_indices


def _all_target_candidates(
    *,
    class_probabilities: np.ndarray,
    mask_probabilities: np.ndarray,
    class_names_by_id: dict[int, str],
    mask_threshold: float,
    minimum_class_score: float,
    query_indices: np.ndarray | None = None,
) -> list[AnimalInstanceCandidate]:
    if (
        class_probabilities.ndim != 2
        or mask_probabilities.ndim != 3
        or class_probabilities.shape[0] != mask_probabilities.shape[0]
        or class_probabilities.shape[1] < 2
        or mask_probabilities.shape[1] <= 0
        or mask_probabilities.shape[2] <= 0
        or not np.isfinite(class_probabilities).all()
        or not np.isfinite(mask_probabilities).all()
        or any(
            isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or not 0 <= class_id < class_probabilities.shape[1] - 1
            or not isinstance(name, str)
            or not name
            for class_id, name in class_names_by_id.items()
        )
    ):
        raise ValueError("animal instance query outputs differ")
    if query_indices is None:
        query_indices = np.arange(class_probabilities.shape[0], dtype=np.int64)
    if (
        query_indices.shape != (class_probabilities.shape[0],)
        or not np.issubdtype(query_indices.dtype, np.integer)
        or np.any(query_indices < 0)
        or len(set(query_indices.tolist())) != len(query_indices)
    ):
        raise ValueError("animal instance source query indices differ")
    target_class_ids = np.asarray(sorted(class_names_by_id), dtype=np.int64)
    target_probabilities = class_probabilities[:, target_class_ids]
    predicted_target_offsets = target_probabilities.argmax(axis=1)
    candidates: list[AnimalInstanceCandidate] = []
    for query_index, target_offset in enumerate(predicted_target_offsets.tolist()):
        class_id = int(target_class_ids[target_offset])
        class_name = class_names_by_id[class_id]
        score = float(class_probabilities[query_index, class_id])
        if score < minimum_class_score:
            continue
        probability = np.ascontiguousarray(
            np.clip(mask_probabilities[query_index], 0.0, 1.0), dtype=np.float32
        )
        hard_mask = np.ascontiguousarray(
            probability >= mask_threshold, dtype=np.uint8
        )
        if not hard_mask.any():
            continue
        candidates.append(
            AnimalInstanceCandidate(
                probability=probability,
                hard_mask=hard_mask,
                source_box_xyxy=_mask_box(hard_mask),
                query_index=int(query_indices[query_index]),
                class_id=class_id,
                class_name=class_name,
                class_score=score,
            )
        )
    return candidates


def _suppress_duplicate_candidates(
    candidates: list[AnimalInstanceCandidate],
    *,
    duplicate_mask_iou: float,
    maximum_instances: int,
) -> list[AnimalInstanceCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (-item.class_score, item.class_id, item.query_index),
    )
    retained: list[tuple[AnimalInstanceCandidate, int]] = []
    for candidate in ordered:
        area = int(candidate.hard_mask.sum())
        duplicate = False
        for existing, existing_area in retained:
            intersection = int(
                np.count_nonzero(candidate.hard_mask & existing.hard_mask)
            )
            union = area + existing_area - intersection
            if intersection / union >= duplicate_mask_iou or intersection / min(
                area, existing_area
            ) >= 0.9:
                duplicate = True
                break
        if duplicate:
            continue
        retained.append((candidate, area))
        if len(retained) == maximum_instances:
            break
    return [candidate for candidate, _ in retained]


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    if mask.ndim != 2 or mask.dtype != np.uint8 or not mask.any():
        raise ValueError("animal instance mask cannot produce a source box")
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return (
        columns[0].item(),
        rows[0].item(),
        columns[-1].item() + 1,
        rows[-1].item() + 1,
    )


__all__ = [
    "AnimalInstanceCandidate",
    "AnimalInstanceSegmentationRuntime",
]
