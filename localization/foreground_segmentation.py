"""Receipt-bound high-resolution foreground segmentation runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from artifact_contracts.foreground_segmentation_model import (
    ForegroundSegmentationArtifact,
)

_RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ForegroundSegmentationPrediction:
    probability: np.ndarray
    hard_mask: np.ndarray
    source_box_xyxy: tuple[int, int, int, int]
    inference_width: int
    inference_height: int
    threshold: float
    foreground_fraction: float
    border_foreground_fraction: float
    state: str
    reasons: tuple[str, ...]

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
        ):
            raise ValueError("foreground prediction arrays differ")
        if self.state not in {"CANDIDATE", "ABSTAIN"}:
            raise ValueError("foreground prediction state differs")
        if (self.state == "ABSTAIN") != bool(self.reasons):
            raise ValueError("foreground prediction reasons differ")


class ForegroundSegmentationRuntime:
    """Execute an exact local BiRefNet snapshot without network access."""

    def __init__(
        self,
        *,
        artifact: ForegroundSegmentationArtifact,
        device: str = "cpu",
        threshold: float = 0.5,
    ) -> None:
        if artifact.manifest.model_family != "BIREFNET_DYNAMIC_SWIN_V1_LARGE":
            raise ValueError("unsupported foreground model family")
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("foreground threshold must lie strictly between zero and one")
        import torch
        from transformers import AutoModelForImageSegmentation

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA foreground runtime requested without CUDA")
        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        artifact.revalidate_local_files()
        self._model = AutoModelForImageSegmentation.from_pretrained(
            str(artifact.model_directory),
            trust_remote_code=True,
            local_files_only=True,
            dtype=self._dtype,
        ).to(self._device)
        self._model.eval()
        self.artifact = artifact
        self.threshold = float(threshold)

    def predict(
        self,
        image: Image.Image,
        *,
        target_box_xyxy: tuple[float, float, float, float] | None = None,
    ) -> ForegroundSegmentationPrediction:
        rgb = image.convert("RGB")
        source_width, source_height = rgb.size
        box = _validate_target_box(
            target_box_xyxy, width=source_width, height=source_height
        )
        crop = rgb.crop(box)
        inference_width, inference_height = _compute_inference_size(
            crop.width,
            crop.height,
            multiple=self.artifact.manifest.input_multiple,
            minimum_side=self.artifact.manifest.minimum_inference_side,
            maximum_side=self.artifact.manifest.maximum_inference_side,
        )
        resized = crop.resize(
            (inference_width, inference_height), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels -= _RGB_MEAN
        pixels /= _RGB_STD
        tensor = self._torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._dtype)
        with self._torch.inference_mode():
            output = self._model(tensor)
            logits = output[-1]
            if logits.ndim != 4 or logits.shape[:2] != (1, 1):
                raise RuntimeError("foreground model output shape differs")
            probability = logits.sigmoid().to(dtype=self._torch.float32)
            probability = self._torch.nn.functional.interpolate(
                probability,
                size=(box[3] - box[1], box[2] - box[0]),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise RuntimeError("foreground model output is non-finite")
        source_probability = np.zeros((source_height, source_width), dtype=np.float32)
        source_probability[box[1] : box[3], box[0] : box[2]] = probability
        np.clip(source_probability, 0.0, 1.0, out=source_probability)
        source_probability = np.ascontiguousarray(source_probability, dtype=np.float32)
        hard_mask = np.ascontiguousarray(
            source_probability >= self.threshold, dtype=np.uint8
        )
        foreground_fraction = float(hard_mask.mean())
        border = np.concatenate(
            (hard_mask[0], hard_mask[-1], hard_mask[:, 0], hard_mask[:, -1])
        )
        reasons: list[str] = []
        if not hard_mask.any():
            reasons.append("EMPTY_FOREGROUND")
        if hard_mask.all():
            reasons.append("FULL_FRAME_FOREGROUND")
        return ForegroundSegmentationPrediction(
            probability=source_probability,
            hard_mask=hard_mask,
            source_box_xyxy=box,
            inference_width=inference_width,
            inference_height=inference_height,
            threshold=self.threshold,
            foreground_fraction=foreground_fraction,
            border_foreground_fraction=float(border.mean()),
            state="ABSTAIN" if reasons else "CANDIDATE",
            reasons=tuple(reasons),
        )


def _compute_inference_size(
    width: int,
    height: int,
    *,
    multiple: int,
    minimum_side: int,
    maximum_side: int,
) -> tuple[int, int]:
    values = (width, height, multiple, minimum_side, maximum_side)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("foreground inference dimensions must be positive integers")
    if minimum_side > maximum_side or minimum_side % multiple or maximum_side % multiple:
        raise ValueError("foreground inference bounds differ")
    scale = max(1.0, minimum_side / min(width, height))
    scale = min(scale, maximum_side / max(width, height))

    def aligned(value: int) -> int:
        result = max(multiple, round(value * scale / multiple) * multiple)
        return min(maximum_side, result)

    inference_width = aligned(width)
    inference_height = aligned(height)
    if max(inference_width, inference_height) > maximum_side:
        raise AssertionError("aligned foreground input exceeds maximum")
    return inference_width, inference_height


def _validate_target_box(
    value: tuple[float, float, float, float] | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if value is None:
        return (0, 0, width, height)
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("foreground target box must contain four coordinates")
    coordinates = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinates):
        raise ValueError("foreground target box must be finite")
    x1, y1, x2, y2 = coordinates
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("foreground target box lies outside source image")
    result = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("foreground target box is empty after source alignment")
    return result


__all__ = [
    "ForegroundSegmentationPrediction",
    "ForegroundSegmentationRuntime",
]
