"""Exact local SAM2.1 image-prompt inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from contracts.prompt_segmentation_model import PromptSegmentationArtifact


@dataclass(frozen=True, slots=True)
class Sam2PromptOutput:
    mask_probabilities: np.ndarray
    predicted_ious: np.ndarray
    original_size: tuple[int, int]
    model_binding: dict[str, Any]


class Sam2PromptRuntime:
    """BF16 image-only SAM2.1 runtime with zero-tolerance weight conversion."""

    def __init__(
        self,
        *,
        model_directory: Path,
        manifest_bundle_path: Path,
        device: str = "cuda",
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("SAM2 prompt runtime device must be cpu or cuda")
        self.artifact = PromptSegmentationArtifact.load(
            model_directory=model_directory,
            manifest_bundle_path=manifest_bundle_path,
        )
        if self.artifact.manifest.model_family != "SAM2_1_HIERA_LARGE":
            raise ValueError("SAM2 prompt runtime model family differs")
        import torch
        from transformers import Sam2Model, Sam2VideoProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        root = str(self.artifact.model_directory)
        self.processor = Sam2VideoProcessor.from_pretrained(root, local_files_only=True)
        model, loading = Sam2Model.from_pretrained(
            root,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            if loading.get(name):
                raise RuntimeError(f"SAM2 image runtime {name} are non-empty")
        self.artifact.revalidate_local_files()
        self.model = model.to(self.device, dtype=self.dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.artifact.revalidate_local_files()

    @property
    def binding(self) -> dict[str, Any]:
        manifest = self.artifact.manifest
        return {
            "model_id": manifest.model_id,
            "source_revision": manifest.source_revision,
            "model_family": manifest.model_family,
            "manifest_sha256": manifest.manifest_sha256,
            "bundle_sha256": self.artifact.bundle_sha256,
            "runtime_conversion": manifest.runtime_conversion,
            "device": self.device.type,
            "dtype": str(self.dtype).removeprefix("torch."),
        }

    def predict(
        self,
        image: Image.Image,
        *,
        box_xyxy: tuple[float, float, float, float],
        positive_points_xy: tuple[tuple[float, float], ...],
        negative_points_xy: tuple[tuple[float, float], ...] = (),
    ) -> Sam2PromptOutput:
        rgb = image.convert("RGB")
        width, height = rgb.size
        _validate_box(box_xyxy, width=width, height=height)
        if not positive_points_xy:
            raise ValueError("SAM2 prompt requires positive points")
        points = (*positive_points_xy, *negative_points_xy)
        for point in points:
            _validate_point(point, width=width, height=height)
        labels = [1] * len(positive_points_xy) + [0] * len(negative_points_xy)
        inputs = self.processor(
            images=[rgb],
            input_points=[[[list(point) for point in points]]],
            input_labels=[[labels]],
            input_boxes=[[[float(item) for item in box_xyxy]]],
            return_tensors="pt",
        )
        original_sizes = inputs.pop("original_sizes")
        model_inputs = {name: value.to(self.device) for name, value in inputs.items()}
        import torch

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(**model_inputs, multimask_output=True)
        masks = self.processor.post_process_masks(
            output.pred_masks.float().cpu(),
            original_sizes,
            binarize=False,
        )[0]
        values = torch.sigmoid(masks[0]).numpy().astype(np.float32, copy=False)
        scores = output.iou_scores[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
        if values.ndim != 3 or values.shape[1:] != (height, width):
            raise RuntimeError("SAM2 post-processed mask shape differs")
        if scores.shape != (values.shape[0],):
            raise RuntimeError("SAM2 predicted IoU shape differs")
        if not np.isfinite(values).all() or not np.isfinite(scores).all():
            raise RuntimeError("SAM2 output contains non-finite values")
        return Sam2PromptOutput(values, scores, (width, height), self.binding)


def _validate_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> None:
    if len(box) != 4 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in box
    ):
        raise ValueError("SAM2 prompt box differs")
    x1, y1, x2, y2 = box
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("SAM2 prompt box lies outside the image")


def _validate_point(point: tuple[float, float], *, width: int, height: int) -> None:
    if len(point) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in point
    ):
        raise ValueError("SAM2 prompt point differs")
    if not 0.0 <= point[0] <= width or not 0.0 <= point[1] <= height:
        raise ValueError("SAM2 prompt point lies outside the image")


__all__ = ["Sam2PromptOutput", "Sam2PromptRuntime"]
