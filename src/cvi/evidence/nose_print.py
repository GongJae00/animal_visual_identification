from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from cvi.evidence.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    NoseDetectorManifest,
    NoseEmbeddingManifest,
    NoseMaskManifest,
    preprocess_image,
)
from cvi.evidence.base import EvidenceObservation, EvidenceUnavailableReason
from cvi.provenance import content_sha256


class TinyViTBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        raise RuntimeError(
            "TinyViTBackbone is disabled: the former implementation was an "
            "untrained CNN, not TinyViT. Supply an exact nose embedding artifact."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("TinyViTBackbone is disabled")


class MagFaceNoseHead(nn.Module):
    """Disabled random training head retained only for an explicit failure."""

    def __init__(self, *args: object, **kwargs: object):
        raise RuntimeError(
            "MagFaceNoseHead is disabled until a trained, exact checkpoint contract exists"
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("MagFaceNoseHead is disabled")


class NoseEnhancer:
    """Disabled ambiguous alias; CLAHE is an optional preprocessing transform."""

    def __init__(self, *args: object, **kwargs: object):
        raise RuntimeError(
            "NoseEnhancer is disabled; declare optional ClaheTransform in the artifact manifest"
        )


class MiewIDNoseExtractor:
    """Disabled former alias: MiewID is not a nose-print embedding model."""

    def __init__(self, onnx_path: Path, input_size: int = 440):
        warnings.warn(
            "MiewIDNoseExtractor is deprecated and disabled because MiewID is "
            "a whole-crop wildlife ReID model, not a nose biometric.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "MiewIDNoseExtractor is deprecated and disabled; use an exact "
            "detector and nose-embedding artifact bundle"
        )


@dataclass(frozen=True, slots=True)
class NoseDetection:
    box: tuple[int, int, int, int]
    confidence: float


class YoloNoseDetector:
    """Exact normalized-xyxy detector adapter; no model is created implicitly."""

    def __init__(
        self,
        model_path: Path,
        manifest: NoseDetectorManifest,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, NoseDetectorManifest):
            raise ArtifactContractError(
                "YoloNoseDetector requires a NoseDetectorManifest"
            )
        self._manifest = manifest
        self._runtime = ExactOnnxRuntime(model_path, manifest, use_cuda=use_cuda)

    def detect(self, image: Image.Image) -> NoseDetection | None:
        detections = self._runtime.run(preprocess_image(image, self._manifest))[0]
        confidence = detections[:, 4]
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ArtifactContractError("detector confidence must be between 0 and 1")
        eligible = np.flatnonzero(confidence >= self._manifest.confidence_threshold)
        if eligible.size == 0:
            return None
        best_index = int(eligible[np.argmax(confidence[eligible])])
        x0, y0, x1, y1 = np.clip(detections[best_index, :4], 0.0, 1.0)
        width, height = image.size
        clipped = (
            int(math.floor(float(x0) * width)),
            int(math.floor(float(y0) * height)),
            int(math.ceil(float(x1) * width)),
            int(math.ceil(float(y1) * height)),
        )
        return NoseDetection(clipped, float(confidence[best_index]))


class DNPMask:
    """Optional exact mask adapter; the former random U-Net is not available."""

    def __init__(
        self,
        model_path: Path,
        manifest: NoseMaskManifest,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, NoseMaskManifest):
            raise ArtifactContractError("DNPMask requires a NoseMaskManifest")
        self._manifest = manifest
        self._runtime = ExactOnnxRuntime(model_path, manifest, use_cuda=use_cuda)

    def apply(self, nose_crop: np.ndarray) -> np.ndarray:
        if (
            not isinstance(nose_crop, np.ndarray)
            or nose_crop.dtype != np.uint8
            or nose_crop.ndim != 3
            or nose_crop.shape[2] != 3
            or nose_crop.shape[0] == 0
            or nose_crop.shape[1] == 0
        ):
            raise ArtifactContractError("nose crop must be a non-empty uint8 RGB array")
        image = Image.fromarray(nose_crop, mode="RGB")
        probabilities = self._runtime.run(
            preprocess_image(image, self._manifest)
        )[0, 0]
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ArtifactContractError("mask probabilities must be between 0 and 1")
        height, width = nose_crop.shape[:2]
        mask = cv2.resize(probabilities, (width, height), interpolation=cv2.INTER_LINEAR)
        foreground = mask >= self._manifest.threshold
        return np.where(foreground[:, :, None], nose_crop, 128).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class NoseRoiPolicy:
    min_box_width: int
    min_box_height: int
    min_resolution_width: int
    min_resolution_height: int

    def __post_init__(self) -> None:
        values = (
            self.min_box_width,
            self.min_box_height,
            self.min_resolution_width,
            self.min_resolution_height,
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in values
        ):
            raise ArtifactContractError("nose ROI policy values must be positive integers")
        if (
            self.min_resolution_width < self.min_box_width
            or self.min_resolution_height < self.min_box_height
        ):
            raise ArtifactContractError(
                "minimum ROI resolution must not be smaller than minimum box size"
            )


NoseAbstainReason = EvidenceUnavailableReason
NoseEvidenceResult = EvidenceObservation


class NosePrintExtractor:
    """Composed detector-to-ROI-to-embedding channel with typed abstention."""

    name = "nose_print"

    def __init__(
        self,
        detector_path: Path,
        detector_manifest: NoseDetectorManifest,
        embedding_path: Path,
        embedding_manifest: NoseEmbeddingManifest,
        roi_policy: NoseRoiPolicy,
        *,
        mask_path: Path | None = None,
        mask_manifest: NoseMaskManifest | None = None,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(detector_manifest, NoseDetectorManifest):
            raise ArtifactContractError(
                "NosePrintExtractor requires a NoseDetectorManifest"
            )
        if not isinstance(embedding_manifest, NoseEmbeddingManifest):
            raise ArtifactContractError(
                "NosePrintExtractor requires a NoseEmbeddingManifest"
            )
        if not isinstance(roi_policy, NoseRoiPolicy):
            raise ArtifactContractError("roi_policy must be a NoseRoiPolicy")
        if (mask_path is None) != (mask_manifest is None):
            raise ArtifactContractError(
                "mask_path and mask_manifest must be supplied together"
            )
        self._detector = YoloNoseDetector(
            detector_path, detector_manifest, use_cuda=use_cuda
        )
        self._embedding_manifest = embedding_manifest
        self._embedding_runtime = ExactOnnxRuntime(
            embedding_path, embedding_manifest, use_cuda=use_cuda
        )
        self._mask = (
            DNPMask(mask_path, mask_manifest, use_cuda=use_cuda)
            if mask_path is not None and mask_manifest is not None
            else None
        )
        self._roi_policy = roi_policy
        contract = {
            "detector": detector_manifest.to_dict(),
            "embedding": embedding_manifest.to_dict(),
            "mask": None if mask_manifest is None else mask_manifest.to_dict(),
            "roi_policy": {
                "min_box_width": roi_policy.min_box_width,
                "min_box_height": roi_policy.min_box_height,
                "min_resolution_width": roi_policy.min_resolution_width,
                "min_resolution_height": roi_policy.min_resolution_height,
            },
        }
        self.gallery_contract_fields = {
            "model_sha256": content_sha256(contract),
            "detector_artifact_sha256": detector_manifest.artifact_sha256,
            "embedding_artifact_sha256": embedding_manifest.artifact_sha256,
            "mask_artifact_sha256": (
                None if mask_manifest is None else mask_manifest.artifact_sha256
            ),
            "manifest_contract_sha256": content_sha256(contract),
        }

    @property
    def output_dim(self) -> int:
        return self._embedding_manifest.output_shape[1]

    def extract(self, image: Image.Image) -> NoseEvidenceResult:
        detection = self._detector.detect(image)
        if detection is None:
            return EvidenceObservation.unavailable(
                self.name, EvidenceUnavailableReason.NO_ROI
            )
        x0, y0, x1, y1 = detection.box
        width, height = x1 - x0, y1 - y0
        if width < self._roi_policy.min_box_width or height < self._roi_policy.min_box_height:
            return EvidenceObservation.unavailable(
                self.name,
                EvidenceUnavailableReason.ROI_TOO_SMALL,
                details={
                    "roi_box": list(detection.box),
                    "detection_confidence": detection.confidence,
                },
            )
        if (
            width < self._roi_policy.min_resolution_width
            or height < self._roi_policy.min_resolution_height
        ):
            return EvidenceObservation.unavailable(
                self.name,
                EvidenceUnavailableReason.ROI_LOW_RESOLUTION,
                details={
                    "roi_box": list(detection.box),
                    "detection_confidence": detection.confidence,
                },
            )
        crop = np.asarray(image.convert("RGB").crop(detection.box), dtype=np.uint8)
        if self._mask is not None:
            crop = self._mask.apply(crop)
        output = self._embedding_runtime.run(
            preprocess_image(Image.fromarray(crop, mode="RGB"), self._embedding_manifest)
        )[0]
        norm = float(np.linalg.norm(output))
        if not math.isfinite(norm) or norm <= 0:
            raise ArtifactContractError(
                "nose embedding artifact produced a non-finite or zero-norm vector"
            )
        embedding = np.asarray(output / norm, dtype=np.float32)
        return EvidenceObservation.available(
            self.name,
            embedding,
            details={
                "roi_box": list(detection.box),
                "detection_confidence": detection.confidence,
            },
        )

    def extract_batch(self, images: list[Image.Image]) -> list[NoseEvidenceResult]:
        return [self.extract(image) for image in images]


__all__ = [
    "DNPMask",
    "MiewIDNoseExtractor",
    "MagFaceNoseHead",
    "NoseAbstainReason",
    "NoseDetection",
    "NoseEnhancer",
    "NoseEvidenceResult",
    "NosePrintExtractor",
    "NoseRoiPolicy",
    "TinyViTBackbone",
    "YoloNoseDetector",
]
