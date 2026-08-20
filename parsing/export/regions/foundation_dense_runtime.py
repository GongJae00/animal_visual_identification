"""High-resolution dense features from content-bound foundation backbones."""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shared.contracts.foundation_vision_model import (
    FoundationModelFamily,
    FoundationVisionArtifact,
)


@dataclass(frozen=True, slots=True)
class FoundationImageTransform:
    source_width: int
    source_height: int
    canvas_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @property
    def scale(self) -> float:
        return min(
            self.resized_width / self.source_width,
            self.resized_height / self.source_height,
        )

    def source_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.pad_left + x * self.scale, self.pad_top + y * self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "canvas_size": self.canvas_size,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "scale": self.scale,
            "policy": "ASPECT_PRESERVING_BICUBIC_CENTER_PAD_ZERO",
        }


@dataclass(frozen=True, slots=True)
class DenseFeatureBatch:
    features: np.ndarray
    summaries: np.ndarray
    source_validity: np.ndarray
    transforms: tuple[FoundationImageTransform, ...]
    model_binding: dict[str, Any]


class FoundationDenseRuntime:
    """Local-only BF16 runtime with exact executable-source revalidation."""

    def __init__(
        self,
        *,
        model_directory: Path,
        manifest_bundle_path: Path,
        device: str = "cuda",
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("foundation runtime device must be cpu or cuda")
        self.artifact = FoundationVisionArtifact.load(
            model_directory=model_directory,
            manifest_bundle_path=manifest_bundle_path,
        )
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        manifest = self.artifact.manifest
        if manifest.family is FoundationModelFamily.CRADIO_V4:
            self.model = _load_bound_cradio(self.artifact)
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                str(self.artifact.model_directory),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            ).eval()
        self.artifact.revalidate_local_files()
        self.model.to(self.device, dtype=self.dtype)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.artifact.revalidate_local_files()

    @property
    def binding(self) -> dict[str, Any]:
        manifest = self.artifact.manifest
        return {
            "model_id": manifest.model_id,
            "source_revision": manifest.source_revision,
            "family": manifest.family.value,
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_document_sha256": self.artifact.manifest_document_sha256,
            "patch_size": manifest.patch_size,
            "dense_feature_dimension": manifest.dense_feature_dimension,
            "summary_dimension": manifest.summary_dimension,
            "device": self.device.type,
            "dtype": str(self.dtype).removeprefix("torch."),
            "attention": "MODEL_NATIVE_MEMORY_EFFICIENT_ATTENTION",
        }

    def extract(
        self,
        images: Sequence[Image.Image],
        *,
        resolution: int,
        maximum_tokens_per_batch: int = 4096,
    ) -> DenseFeatureBatch:
        if not images:
            raise ValueError("foundation dense extraction requires images")
        if (
            isinstance(maximum_tokens_per_batch, bool)
            or not isinstance(maximum_tokens_per_batch, int)
            or maximum_tokens_per_batch <= 0
        ):
            raise ValueError("maximum tokens per batch must be positive")
        manifest = self.artifact.manifest
        if (
            isinstance(resolution, bool)
            or not isinstance(resolution, int)
            or resolution <= 0
            or resolution > manifest.maximum_resolution
            or resolution % manifest.patch_size != 0
        ):
            raise ValueError("foundation resolution is unsupported")
        grid = resolution // manifest.patch_size
        tokens_per_image = grid * grid
        batch_size = max(1, maximum_tokens_per_batch // tokens_per_image)
        features: list[np.ndarray] = []
        summaries: list[np.ndarray] = []
        validity: list[np.ndarray] = []
        transforms: list[FoundationImageTransform] = []
        for offset in range(0, len(images), batch_size):
            rows = images[offset : offset + batch_size]
            tensor, row_validity, row_transforms = _prepare_images(
                rows,
                resolution=resolution,
                patch_size=manifest.patch_size,
                family=manifest.family,
            )
            dense, summary = self._forward(tensor, grid=grid)
            features.extend(dense)
            summaries.extend(summary)
            validity.extend(row_validity)
            transforms.extend(row_transforms)
        return DenseFeatureBatch(
            features=np.stack(features).astype(np.float32, copy=False),
            summaries=np.stack(summaries).astype(np.float32, copy=False),
            source_validity=np.stack(validity),
            transforms=tuple(transforms),
            model_binding=self.binding,
        )

    def _forward(self, values: np.ndarray, *, grid: int) -> tuple[np.ndarray, np.ndarray]:
        import torch

        tensor = torch.from_numpy(values).to(self.device, dtype=self.dtype)
        enabled = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        ):
            output = self.model(tensor)
        manifest = self.artifact.manifest
        if manifest.family is FoundationModelFamily.CRADIO_V4:
            if hasattr(output, "features") and hasattr(output, "summary"):
                dense, summary = output.features, output.summary
            elif isinstance(output, tuple) and len(output) == 2:
                summary, dense = output
            else:
                raise RuntimeError("C-RADIO output contract differs")
        else:
            hidden = output.last_hidden_state
            dense = hidden[:, -grid * grid :]
            summary = output.pooler_output
        if dense.shape[1:] != (grid * grid, manifest.dense_feature_dimension):
            raise RuntimeError("foundation dense feature shape differs")
        if summary.shape[1:] != (manifest.summary_dimension,):
            raise RuntimeError("foundation summary feature shape differs")
        dense = torch.nn.functional.normalize(dense.float(), dim=2)
        summary = torch.nn.functional.normalize(summary.float(), dim=1)
        return (
            dense.reshape(-1, grid, grid, manifest.dense_feature_dimension)
            .cpu()
            .numpy(),
            summary.cpu().numpy(),
        )


def _prepare_images(
    images: Sequence[Image.Image],
    *,
    resolution: int,
    patch_size: int,
    family: FoundationModelFamily,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    list[FoundationImageTransform],
]:
    arrays: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    transforms: list[FoundationImageTransform] = []
    grid = resolution // patch_size
    for image in images:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = resolution / max(width, height)
        resized_width = max(1, min(resolution, round(width * scale)))
        resized_height = max(1, min(resolution, round(height * scale)))
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        left = (resolution - resized_width) // 2
        top = (resolution - resized_height) // 2
        canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = np.asarray(
            resized, dtype=np.uint8
        )
        tensor = canvas.astype(np.float32) / 255.0
        if family is FoundationModelFamily.DINOV3_VIT:
            tensor = (
                tensor - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        arrays.append(tensor.transpose(2, 0, 1))
        patch_validity = np.zeros((grid, grid), dtype=bool)
        x1 = math.floor(left / patch_size)
        y1 = math.floor(top / patch_size)
        x2 = math.ceil((left + resized_width) / patch_size)
        y2 = math.ceil((top + resized_height) / patch_size)
        patch_validity[y1:y2, x1:x2] = True
        validity.append(patch_validity)
        transforms.append(
            FoundationImageTransform(
                source_width=width,
                source_height=height,
                canvas_size=resolution,
                resized_width=resized_width,
                resized_height=resized_height,
                pad_left=left,
                pad_top=top,
            )
        )
    return np.stack(arrays), validity, transforms


def _load_bound_cradio(artifact: FoundationVisionArtifact) -> Any:
    """Import C-RADIO from the exact validated directory, not an HF code cache."""

    root = artifact.model_directory
    namespace = f"_cvi_cradio_{artifact.manifest.manifest_sha256}"
    package = sys.modules.get(namespace)
    if package is None:
        package = types.ModuleType(namespace)
        package.__path__ = [str(root)]
        package.__package__ = namespace
        sys.modules[namespace] = package
    module_name = f"{namespace}.hf_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = spec_from_file_location(module_name, root / "hf_model.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("bound C-RADIO module could not be loaded")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    artifact.revalidate_local_files()
    config = module.RADIOConfig.from_pretrained(str(root), local_files_only=True)
    model = module.RADIOModel.from_pretrained(
        str(root),
        config=config,
        local_files_only=True,
        use_safetensors=True,
    ).eval()
    artifact.revalidate_local_files()
    return model


__all__ = [
    "DenseFeatureBatch",
    "FoundationDenseRuntime",
    "FoundationImageTransform",
]
