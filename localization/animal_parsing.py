"""Production-style visible animal-instance parsing in source coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from foundation.provenance import content_sha256
from localization.animal_instance_segmentation import (
    AnimalInstanceCandidate,
    AnimalInstanceSegmentationRuntime,
    _mask_box,
)
from localization.foreground_segmentation import ForegroundSegmentationRuntime

PARSING_SCHEMA = "cvi.visible_animal_instance_parsing.v1"
PARSING_ONTOLOGY = "VISIBLE_ANIMAL_INSTANCE_APPEARANCE_V1"
PARSING_ONTOLOGY_DESCRIPTION = (
    "Visible pixels assigned to one detected animal instance. Occluded pixels are not "
    "inferred. Other detected animal instances and disconnected scene foreground are "
    "excluded. Tightly attached clothing, collars, and harnesses may remain because "
    "they are not independently labeled by the bound models."
)


@dataclass(frozen=True, slots=True)
class AnimalParsingPolicy:
    class_names: tuple[str, ...] = ("dog", "cat")
    duplicate_mask_iou: float = 0.6
    maximum_instances: int = 32
    refinement_context_fraction: float = 0.1
    semantic_support_threshold: float = 0.2
    semantic_core_threshold: float = 0.7
    foreground_threshold: float = 0.5
    support_dilation_fraction: float = 0.03
    minimum_support_dilation_pixels: int = 3
    maximum_support_dilation_pixels: int = 31
    minimum_mask_pixels: int = 256
    minimum_semantic_shape_iou: float = 0.05
    review_semantic_shape_iou: float = 0.25
    review_ownership_retention: float = 0.9
    minimum_ownership_retention: float = 0.5
    review_component_count: int = 4
    maximum_component_count: int = 16
    schema_version: str = "cvi.animal_parsing_policy.v4"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.animal_parsing_policy.v4":
            raise ValueError("animal parsing policy schema differs")
        if (
            not isinstance(self.class_names, tuple)
            or not self.class_names
            or any(not isinstance(name, str) or not name for name in self.class_names)
            or len(self.class_names) != len(set(self.class_names))
        ):
            raise ValueError("animal parsing classes must be unique canonical names")
        for name in (
            "duplicate_mask_iou",
            "semantic_support_threshold",
            "semantic_core_threshold",
            "foreground_threshold",
            "minimum_semantic_shape_iou",
            "review_semantic_shape_iou",
            "review_ownership_retention",
            "minimum_ownership_retention",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 < value < 1.0
            ):
                raise ValueError(f"animal parsing {name} must lie strictly in (0, 1)")
        for name in ("refinement_context_fraction", "support_dilation_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"animal parsing {name} must lie in [0, 1]")
        for name in (
            "maximum_instances",
            "minimum_support_dilation_pixels",
            "maximum_support_dilation_pixels",
            "minimum_mask_pixels",
            "review_component_count",
            "maximum_component_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"animal parsing {name} must be positive")
        if (
            self.semantic_support_threshold >= self.semantic_core_threshold
            or self.minimum_semantic_shape_iou
            > self.review_semantic_shape_iou
            or self.minimum_support_dilation_pixels
            > self.maximum_support_dilation_pixels
            or self.minimum_ownership_retention > self.review_ownership_retention
            or self.review_component_count > self.maximum_component_count
        ):
            raise ValueError("animal parsing policy thresholds conflict")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "class_names": list(self.class_names),
            "duplicate_mask_iou": self.duplicate_mask_iou,
            "maximum_instances": self.maximum_instances,
            "refinement_context_fraction": self.refinement_context_fraction,
            "semantic_support_threshold": self.semantic_support_threshold,
            "semantic_core_threshold": self.semantic_core_threshold,
            "foreground_threshold": self.foreground_threshold,
            "support_dilation_fraction": self.support_dilation_fraction,
            "minimum_support_dilation_pixels": (
                self.minimum_support_dilation_pixels
            ),
            "maximum_support_dilation_pixels": (
                self.maximum_support_dilation_pixels
            ),
            "minimum_mask_pixels": self.minimum_mask_pixels,
            "minimum_semantic_shape_iou": self.minimum_semantic_shape_iou,
            "review_semantic_shape_iou": self.review_semantic_shape_iou,
            "review_ownership_retention": self.review_ownership_retention,
            "minimum_ownership_retention": self.minimum_ownership_retention,
            "review_component_count": self.review_component_count,
            "maximum_component_count": self.maximum_component_count,
        }


@dataclass(frozen=True, slots=True)
class ParsedAnimalQuality:
    state: str
    reasons: tuple[str, ...]
    flags: tuple[str, ...]
    semantic_shape_iou: float
    ownership_retention: float
    foreground_pixels: int
    component_count: int
    touches_source_border: bool

    def __post_init__(self) -> None:
        if self.state not in {"USABLE", "REVIEW", "UNUSABLE"}:
            raise ValueError("parsed animal quality state differs")
        if (self.state == "UNUSABLE") != bool(self.reasons):
            raise ValueError("parsed animal quality reasons differ")
        if self.state == "USABLE" and self.flags:
            raise ValueError("usable parsed animal cannot retain review flags")


@dataclass(frozen=True, slots=True)
class ParsedAnimalInstance:
    instance_index: int
    query_index: int
    class_id: int
    class_name: str
    class_score: float
    detector_box_xyxy: tuple[int, int, int, int]
    refinement_box_xyxy: tuple[int, int, int, int]
    mask_box_xyxy: tuple[int, int, int, int] | None
    instance_probability: np.ndarray
    foreground_probability: np.ndarray
    ownership_probability: np.ndarray
    hard_mask: np.ndarray
    quality: ParsedAnimalQuality

    def __post_init__(self) -> None:
        arrays = (
            self.instance_probability,
            self.foreground_probability,
            self.ownership_probability,
        )
        if (
            isinstance(self.instance_index, bool)
            or not isinstance(self.instance_index, int)
            or self.instance_index < 0
            or self.hard_mask.ndim != 2
            or self.hard_mask.dtype != np.uint8
            or not set(np.unique(self.hard_mask)).issubset({0, 1})
            or any(
                value.shape != self.hard_mask.shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
                or float(value.min()) < 0.0
                or float(value.max()) > 1.0
                for value in arrays
            )
        ):
            raise ValueError("parsed animal instance arrays differ")
        if self.hard_mask.any() != (self.mask_box_xyxy is not None):
            raise ValueError("parsed animal instance mask box differs")


@dataclass(frozen=True, slots=True)
class AnimalParsingPrediction:
    source_width: int
    source_height: int
    instances: tuple[ParsedAnimalInstance, ...]
    policy_sha256: str
    ontology: str = PARSING_ONTOLOGY
    ontology_description: str = PARSING_ONTOLOGY_DESCRIPTION
    schema_version: str = PARSING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PARSING_SCHEMA or self.ontology != PARSING_ONTOLOGY:
            raise ValueError("animal parsing prediction contract differs")
        if (
            isinstance(self.source_width, bool)
            or not isinstance(self.source_width, int)
            or self.source_width <= 0
            or isinstance(self.source_height, bool)
            or not isinstance(self.source_height, int)
            or self.source_height <= 0
        ):
            raise ValueError("animal parsing source dimensions differ")
        if tuple(item.instance_index for item in self.instances) != tuple(
            range(len(self.instances))
        ):
            raise ValueError("animal parsing instance indices differ")
        occupied = np.zeros((self.source_height, self.source_width), dtype=np.uint8)
        for item in self.instances:
            if item.hard_mask.shape != occupied.shape or np.any(
                occupied & item.hard_mask
            ):
                raise ValueError("parsed animal masks overlap or differ in size")
            occupied |= item.hard_mask


@dataclass(frozen=True, slots=True)
class AnimalIdentityCrop:
    box_rgb: Image.Image
    masked_rgb: Image.Image
    mask: Image.Image
    source_box_xyxy: tuple[int, int, int, int]
    instance_index: int
    class_name: str
    parsing_quality_state: str


@dataclass(slots=True)
class _DraftInstance:
    candidate: AnimalInstanceCandidate
    refinement_box_xyxy: tuple[int, int, int, int]
    foreground_probability: np.ndarray
    ownership_probability: np.ndarray
    preownership_mask: np.ndarray
    semantic_shape_iou: float
    refinement_empty: bool


class AnimalParsingRuntime:
    """Parse every supported visible animal instance without annotation prompts."""

    def __init__(
        self,
        *,
        instance_runtime: AnimalInstanceSegmentationRuntime,
        foreground_runtime: ForegroundSegmentationRuntime,
        policy: AnimalParsingPolicy | None = None,
    ) -> None:
        self.instance_runtime = instance_runtime
        self.foreground_runtime = foreground_runtime
        self.policy = policy or AnimalParsingPolicy()

    def predict(self, image: Image.Image) -> AnimalParsingPrediction:
        rgb = image.convert("RGB")
        width, height = rgb.size
        candidates = self.instance_runtime.predict_all(
            rgb,
            class_names=self.policy.class_names,
            duplicate_mask_iou=self.policy.duplicate_mask_iou,
            maximum_instances=self.policy.maximum_instances,
        )
        drafts: list[_DraftInstance] = []
        for candidate in candidates:
            refinement_box = _expand_box(
                candidate.source_box_xyxy,
                width=width,
                height=height,
                fraction=self.policy.refinement_context_fraction,
            )
            foreground = self.foreground_runtime.predict(
                rgb, target_box_xyxy=refinement_box
            )
            mask, ownership, agreement, refinement_empty = _seeded_refinement(
                foreground_probability=foreground.probability,
                instance_probability=candidate.probability,
                policy=self.policy,
            )
            drafts.append(
                _DraftInstance(
                    candidate=candidate,
                    refinement_box_xyxy=refinement_box,
                    foreground_probability=foreground.probability,
                    ownership_probability=ownership,
                    preownership_mask=mask,
                    semantic_shape_iou=agreement,
                    refinement_empty=refinement_empty,
                )
            )
        owned_masks = _exclusive_ownership(drafts, shape=(height, width))
        instances = tuple(
            _finalize_instance(index, draft, owned_masks[index], self.policy)
            for index, draft in enumerate(drafts)
        )
        return AnimalParsingPrediction(
            source_width=width,
            source_height=height,
            instances=instances,
            policy_sha256=self.policy.policy_sha256,
        )


def materialize_identity_crop(
    image: Image.Image,
    instance: ParsedAnimalInstance,
    *,
    context_fraction: float = 0.05,
    background_rgb: tuple[int, int, int] = (127, 127, 127),
    require_usable: bool = True,
) -> AnimalIdentityCrop:
    if require_usable and instance.quality.state != "USABLE":
        raise ValueError("identity crop requires a usable parsed animal instance")
    if instance.mask_box_xyxy is None:
        raise ValueError("identity crop requires nonempty parsed animal support")
    if (
        not math.isfinite(context_fraction)
        or not 0.0 <= context_fraction <= 1.0
        or len(background_rgb) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
            for value in background_rgb
        )
    ):
        raise ValueError("identity crop materialization policy differs")
    rgb = image.convert("RGB")
    if instance.hard_mask.shape != (rgb.height, rgb.width):
        raise ValueError("identity crop source dimensions differ")
    box = _expand_box(
        instance.mask_box_xyxy,
        width=rgb.width,
        height=rgb.height,
        fraction=context_fraction,
    )
    source = np.asarray(rgb, dtype=np.uint8)
    x1, y1, x2, y2 = box
    box_values = np.ascontiguousarray(source[y1:y2, x1:x2])
    mask_values = np.ascontiguousarray(instance.hard_mask[y1:y2, x1:x2])
    foreground = mask_values.astype(bool)
    masked = np.empty_like(box_values)
    masked[...] = background_rgb
    masked[foreground] = box_values[foreground]
    return AnimalIdentityCrop(
        box_rgb=Image.fromarray(box_values, mode="RGB"),
        masked_rgb=Image.fromarray(masked, mode="RGB"),
        mask=Image.fromarray(mask_values * 255, mode="L"),
        source_box_xyxy=box,
        instance_index=instance.instance_index,
        class_name=instance.class_name,
        parsing_quality_state=instance.quality.state,
    )


def _seeded_refinement(
    *,
    foreground_probability: np.ndarray,
    instance_probability: np.ndarray,
    policy: AnimalParsingPolicy,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if (
        foreground_probability.shape != instance_probability.shape
        or foreground_probability.ndim != 2
        or foreground_probability.dtype != np.float32
        or instance_probability.dtype != np.float32
        or not np.isfinite(foreground_probability).all()
        or not np.isfinite(instance_probability).all()
    ):
        raise ValueError("animal parsing refinement probabilities differ")
    semantic_support = np.ascontiguousarray(
        instance_probability >= policy.semantic_support_threshold, dtype=np.uint8
    )
    semantic_core = np.ascontiguousarray(
        instance_probability >= policy.semantic_core_threshold, dtype=np.uint8
    )
    semantic_hard_raw = np.ascontiguousarray(
        instance_probability >= 0.5, dtype=np.uint8
    )
    if not semantic_support.any():
        raise ValueError("animal parsing semantic support is empty")
    minimum_component_pixels = max(16, round(float(semantic_support.sum()) * 0.001))
    semantic_support = _remove_small_components(
        semantic_support, minimum_pixels=minimum_component_pixels
    )
    semantic_core = _remove_small_components(
        semantic_core, minimum_pixels=minimum_component_pixels
    )
    semantic_hard = _remove_small_components(
        semantic_hard_raw, minimum_pixels=minimum_component_pixels
    )
    if not semantic_core.any():
        semantic_core = semantic_hard.copy()
    if not semantic_core.any():
        semantic_core = semantic_support.copy()
    x1, y1, x2, y2 = _mask_box(semantic_support)
    radius = round(min(x2 - x1, y2 - y1) * policy.support_dilation_fraction)
    radius = min(
        policy.maximum_support_dilation_pixels,
        max(policy.minimum_support_dilation_pixels, radius),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    allowed = cv2.dilate(semantic_support, kernel, iterations=1).astype(bool)
    shape = foreground_probability >= policy.foreground_threshold
    shape_allowed = shape & allowed
    candidate = np.ascontiguousarray(shape_allowed, dtype=np.uint8)
    count, labels = cv2.connectedComponents(candidate, connectivity=8)
    touched = np.zeros(count, dtype=bool)
    touched[np.unique(labels[semantic_core.astype(bool)])] = True
    touched[0] = False
    retained = np.ascontiguousarray(touched[labels], dtype=np.uint8)
    refinement_empty = not retained.any()
    final = np.ascontiguousarray(
        retained.astype(bool) | semantic_hard_raw.astype(bool), dtype=np.uint8
    )
    semantic_hard_bool = semantic_hard.astype(bool)
    union = int(np.count_nonzero(semantic_hard_bool | shape_allowed))
    agreement = (
        float(np.count_nonzero(semantic_hard_bool & shape_allowed) / union)
        if union
        else 0.0
    )
    ownership = np.ascontiguousarray(
        instance_probability * (0.5 + 0.5 * foreground_probability),
        dtype=np.float32,
    )
    return final, ownership, agreement, refinement_empty


def _exclusive_ownership(
    drafts: list[_DraftInstance], *, shape: tuple[int, int]
) -> tuple[np.ndarray, ...]:
    if not drafts:
        return ()
    if any(
        draft.preownership_mask.shape != shape
        or draft.ownership_probability.shape != shape
        for draft in drafts
    ):
        raise ValueError("animal parsing ownership dimensions differ")
    best_score = np.full(shape, -1.0, dtype=np.float32)
    owners = np.full(shape, -1, dtype=np.int16)
    for index, draft in enumerate(drafts):
        support = draft.preownership_mask.astype(bool)
        better = support & (draft.ownership_probability > best_score)
        best_score[better] = draft.ownership_probability[better]
        owners[better] = index
    return tuple(
        np.ascontiguousarray(owners == index, dtype=np.uint8)
        for index in range(len(drafts))
    )


def _finalize_instance(
    index: int,
    draft: _DraftInstance,
    hard_mask: np.ndarray,
    policy: AnimalParsingPolicy,
) -> ParsedAnimalInstance:
    preownership_pixels = int(draft.preownership_mask.sum())
    foreground_pixels = int(hard_mask.sum())
    retention = (
        foreground_pixels / preownership_pixels if preownership_pixels else 0.0
    )
    reasons: list[str] = []
    flags: list[str] = []
    if foreground_pixels < policy.minimum_mask_pixels:
        reasons.append("MASK_SUPPORT_BELOW_POLICY")
    if retention < policy.minimum_ownership_retention:
        reasons.append("OWNERSHIP_RETENTION_BELOW_POLICY")
    elif retention < policy.review_ownership_retention:
        flags.append("INSTANCE_OVERLAP_RESOLVED")
    if draft.refinement_empty:
        reasons.append("FOREGROUND_REFINEMENT_EMPTY")
    if draft.semantic_shape_iou < policy.minimum_semantic_shape_iou:
        reasons.append("SEMANTIC_SHAPE_AGREEMENT_BELOW_POLICY")
    elif draft.semantic_shape_iou < policy.review_semantic_shape_iou:
        flags.append("LOW_SEMANTIC_SHAPE_AGREEMENT")
    touches_border = bool(
        foreground_pixels
        and (
            hard_mask[0].any()
            or hard_mask[-1].any()
            or hard_mask[:, 0].any()
            or hard_mask[:, -1].any()
        )
    )
    if touches_border:
        flags.append("SOURCE_BORDER_TRUNCATED")
    component_count = (
        cv2.connectedComponents(hard_mask, connectivity=8)[0] - 1
        if foreground_pixels
        else 0
    )
    if component_count > policy.maximum_component_count:
        reasons.append("VISIBLE_COMPONENT_COUNT_ABOVE_POLICY")
    elif component_count > policy.review_component_count:
        flags.append("MULTIPLE_VISIBLE_COMPONENTS")
    reasons = list(dict.fromkeys(reasons))
    flags = list(dict.fromkeys(flags))
    state = "UNUSABLE" if reasons else "REVIEW" if flags else "USABLE"
    candidate = draft.candidate
    return ParsedAnimalInstance(
        instance_index=index,
        query_index=candidate.query_index,
        class_id=candidate.class_id,
        class_name=candidate.class_name,
        class_score=candidate.class_score,
        detector_box_xyxy=candidate.source_box_xyxy,
        refinement_box_xyxy=draft.refinement_box_xyxy,
        mask_box_xyxy=_mask_box(hard_mask) if foreground_pixels else None,
        instance_probability=candidate.probability,
        foreground_probability=draft.foreground_probability,
        ownership_probability=draft.ownership_probability,
        hard_mask=np.ascontiguousarray(hard_mask, dtype=np.uint8),
        quality=ParsedAnimalQuality(
            state=state,
            reasons=tuple(reasons),
            flags=tuple(flags),
            semantic_shape_iou=draft.semantic_shape_iou,
            ownership_retention=retention,
            foreground_pixels=foreground_pixels,
            component_count=component_count,
            touches_source_border=touches_border,
        ),
    )


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    fraction: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if (
        not 0 <= x1 < x2 <= width
        or not 0 <= y1 < y2 <= height
        or not math.isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
    ):
        raise ValueError("animal parsing expansion box differs")
    pad_x = math.ceil((x2 - x1) * fraction)
    pad_y = math.ceil((y2 - y1) * fraction)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(
        width, x2 + pad_x
    ), min(height, y2 + pad_y)


def _remove_small_components(mask: np.ndarray, *, minimum_pixels: int) -> np.ndarray:
    if (
        mask.ndim != 2
        or mask.dtype != np.uint8
        or isinstance(minimum_pixels, bool)
        or not isinstance(minimum_pixels, int)
        or minimum_pixels <= 0
    ):
        raise ValueError("animal parsing component cleanup policy differs")
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    areas = np.bincount(labels.ravel(), minlength=count)
    keep = areas >= minimum_pixels
    keep[0] = False
    retained = np.ascontiguousarray(keep[labels], dtype=np.uint8)
    if retained.any():
        return retained
    largest = int(np.argmax(areas[1:])) + 1
    return np.ascontiguousarray(labels == largest, dtype=np.uint8)


__all__ = [
    "PARSING_ONTOLOGY",
    "PARSING_ONTOLOGY_DESCRIPTION",
    "PARSING_SCHEMA",
    "AnimalIdentityCrop",
    "AnimalParsingPolicy",
    "AnimalParsingPrediction",
    "AnimalParsingRuntime",
    "ParsedAnimalInstance",
    "ParsedAnimalQuality",
    "materialize_identity_crop",
]
