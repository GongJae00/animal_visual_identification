from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from contracts.artifact_manifest import (
    ArtifactContractError,
    ExactOnnxRuntime,
    LandmarkGraphManifest,
    LandmarkKeypointManifest,
    preprocess_image,
)
from evidence_fusion.base import (
    AbstractEvidencer,
    EvidenceInsufficiency,
    EvidenceObservation,
    EvidenceUnavailableReason,
)
from foundation.provenance import content_sha256

DOGFLW_LANDMARKS: tuple[str, ...] = (
    "left_eye",
    "right_eye",
    "nose_tip",
    "left_ear_base",
    "right_ear_base",
    "left_ear_tip",
    "right_ear_tip",
    "muzzle_left",
    "muzzle_right",
    "mouth_center",
    "chin",
    "left_cheek",
    "right_cheek",
    "forehead_center",
    "crown",
    "left_eye_corner",
    "right_eye_corner",
)


@dataclass(frozen=True, slots=True)
class LandmarkDecodeResult:
    pixel_points: np.ndarray
    normalized_points: np.ndarray
    confidence: np.ndarray
    visible: np.ndarray
    crop_size: tuple[int, int]
    keypoint_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pixel_points, np.ndarray):
            raise ArtifactContractError("pixel_points must be an ndarray")
        count = self.pixel_points.shape[0]
        if self.pixel_points.dtype != np.float32 or self.pixel_points.shape != (count, 2):
            raise ArtifactContractError("pixel_points must be float32 [K,2]")
        if (
            self.normalized_points.dtype != np.float32
            or self.normalized_points.shape != (count, 2)
            or np.any(self.normalized_points < 0.0)
            or np.any(self.normalized_points > 1.0)
        ):
            raise ArtifactContractError("normalized_points must be float32 [K,2] in [0,1]")
        if (
            self.confidence.dtype != np.float32
            or self.confidence.shape != (count,)
            or np.any(self.confidence < 0.0)
            or np.any(self.confidence > 1.0)
        ):
            raise ArtifactContractError("confidence must be float32 [K] in [0,1]")
        if self.visible.dtype != np.bool_ or self.visible.shape != (count,):
            raise ArtifactContractError("visible must be bool [K]")
        if len(self.keypoint_order) != count:
            raise ArtifactContractError("keypoint_order must match the decoded point count")


def decode_landmark_heatmaps(
    heatmaps: np.ndarray,
    manifest: LandmarkKeypointManifest,
    crop_size: tuple[int, int],
) -> LandmarkDecodeResult:
    """Decode exact [1,K,H,W] probability maps in the source crop geometry."""

    if not isinstance(manifest, LandmarkKeypointManifest):
        raise ArtifactContractError("decoder requires a LandmarkKeypointManifest")
    if (
        not isinstance(heatmaps, np.ndarray)
        or heatmaps.dtype != np.float32
        or heatmaps.shape != manifest.output_shape
    ):
        dtype = getattr(heatmaps, "dtype", None)
        shape = getattr(heatmaps, "shape", None)
        raise ArtifactContractError(
            f"landmark heatmaps must be float32 {manifest.output_shape}, got "
            f"{dtype} {shape}"
        )
    if not np.isfinite(heatmaps).all():
        raise ArtifactContractError("landmark heatmaps contain non-finite values")
    if np.any((heatmaps < 0.0) | (heatmaps > 1.0)):
        raise ArtifactContractError("landmark heatmaps must contain probabilities in [0,1]")
    if (
        not isinstance(crop_size, tuple)
        or len(crop_size) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 2
            for value in crop_size
        )
    ):
        raise ArtifactContractError("crop_size must contain width and height of at least 2")

    _, keypoints, heatmap_height, heatmap_width = heatmaps.shape
    flattened = heatmaps[0].reshape(keypoints, -1)
    maximum_indices = flattened.argmax(axis=1)
    confidence = flattened[np.arange(keypoints), maximum_indices].astype(
        np.float32, copy=False
    )
    visible = np.asarray(
        confidence >= manifest.visibility_threshold, dtype=np.bool_
    )
    if int(visible.sum()) < manifest.min_visible_keypoints:
        raise EvidenceInsufficiency(
            EvidenceUnavailableReason.INSUFFICIENT_LANDMARKS,
            {
                "visible_keypoints": int(visible.sum()),
                "required_keypoints": manifest.min_visible_keypoints,
            },
        )

    heatmap_x = (maximum_indices % heatmap_width).astype(np.float32)
    heatmap_y = (maximum_indices // heatmap_width).astype(np.float32)
    crop_width, crop_height = crop_size
    pixel_x = heatmap_x * np.float32((crop_width - 1) / (heatmap_width - 1))
    pixel_y = heatmap_y * np.float32((crop_height - 1) / (heatmap_height - 1))
    pixel_points = np.stack([pixel_x, pixel_y], axis=1).astype(np.float32)
    normalized_points = np.stack(
        [pixel_x / (crop_width - 1), pixel_y / (crop_height - 1)], axis=1
    ).astype(np.float32)
    return LandmarkDecodeResult(
        pixel_points=pixel_points,
        normalized_points=normalized_points,
        confidence=confidence,
        visible=visible,
        crop_size=crop_size,
        keypoint_order=manifest.keypoint_order,
    )


class HRNetHeatmap:
    """Exact keypoint artifact adapter; no random HRNet-like model is built."""

    def __init__(
        self,
        artifact_path: Path,
        manifest: LandmarkKeypointManifest,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, LandmarkKeypointManifest):
            raise ArtifactContractError("HRNetHeatmap requires a LandmarkKeypointManifest")
        self.manifest = manifest
        self._runtime = ExactOnnxRuntime(artifact_path, manifest, use_cuda=use_cuda)

    def infer(self, image: Image.Image) -> np.ndarray:
        return self._runtime.run(preprocess_image(image, self.manifest))


class LandmarkGraphEmbedder:
    """Checkpoint-backed graph adapter consuming normalized landmark records."""

    def __init__(
        self,
        artifact_path: Path,
        manifest: LandmarkGraphManifest,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, LandmarkGraphManifest):
            raise ArtifactContractError(
                "LandmarkGraphEmbedder requires a LandmarkGraphManifest"
            )
        self.manifest = manifest
        self._runtime = ExactOnnxRuntime(artifact_path, manifest, use_cuda=use_cuda)

    def embed(self, decoded: LandmarkDecodeResult) -> np.ndarray:
        if not isinstance(decoded, LandmarkDecodeResult):
            raise ArtifactContractError("graph input must be a LandmarkDecodeResult")
        point_count = len(self.manifest.keypoint_order)
        if decoded.keypoint_order != self.manifest.keypoint_order:
            raise ArtifactContractError(
                "decoded landmarks and graph manifest use different schema order"
            )
        if decoded.normalized_points.shape != (point_count, 2):
            raise ArtifactContractError("decoded landmark count differs from graph schema")
        records = np.concatenate(
            [
                decoded.normalized_points,
                decoded.confidence[:, None],
                decoded.visible.astype(np.float32)[:, None],
            ],
            axis=1,
        )[None].astype(np.float32)
        output = self._runtime.run(records)[0]
        norm = float(np.linalg.norm(output))
        if not np.isfinite(norm) or norm <= 0:
            raise ArtifactContractError(
                "landmark graph artifact produced a non-finite or zero-norm embedding"
            )
        return np.asarray(output / norm, dtype=np.float32)


class LandmarkEvidencer(AbstractEvidencer):
    name = "landmark"

    def __init__(
        self,
        keypoint_path: Path,
        keypoint_manifest: LandmarkKeypointManifest,
        graph_path: Path,
        graph_manifest: LandmarkGraphManifest,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(keypoint_manifest, LandmarkKeypointManifest):
            raise ArtifactContractError(
                "LandmarkEvidencer requires a LandmarkKeypointManifest"
            )
        if not isinstance(graph_manifest, LandmarkGraphManifest):
            raise ArtifactContractError(
                "LandmarkEvidencer requires a LandmarkGraphManifest"
            )
        if keypoint_manifest.keypoint_order != graph_manifest.keypoint_order:
            raise ArtifactContractError(
                "keypoint and graph manifests must bind the same schema and order"
            )
        self._heatmap = HRNetHeatmap(
            keypoint_path, keypoint_manifest, use_cuda=use_cuda
        )
        self._graph = LandmarkGraphEmbedder(
            graph_path, graph_manifest, use_cuda=use_cuda
        )
        self.output_dim = graph_manifest.output_shape[1]
        self.gallery_contract_fields = {
            "model_sha256": content_sha256({
                "keypoint_artifact_sha256": keypoint_manifest.artifact_sha256,
                "graph_artifact_sha256": graph_manifest.artifact_sha256,
            }),
            "keypoint_artifact_sha256": keypoint_manifest.artifact_sha256,
            "graph_artifact_sha256": graph_manifest.artifact_sha256,
            "manifest_contract_sha256": content_sha256({
                "keypoint": keypoint_manifest.to_dict(),
                "graph": graph_manifest.to_dict(),
            }),
        }

    def extract(self, image: Image.Image) -> EvidenceObservation:
        heatmaps = self._heatmap.infer(image)
        try:
            decoded = decode_landmark_heatmaps(
                heatmaps, self._heatmap.manifest, image.size
            )
        except EvidenceInsufficiency as exc:
            return EvidenceObservation.unavailable(
                self.name, exc.reason, details=exc.details
            )
        return EvidenceObservation.available(self.name, self._graph.embed(decoded))

    def extract_batch(self, images: list[Image.Image]) -> list[EvidenceObservation]:
        return [self.extract(image) for image in images]


__all__ = [
    "DOGFLW_LANDMARKS",
    "HRNetHeatmap",
    "LandmarkDecodeResult",
    "LandmarkEvidencer",
    "LandmarkGraphEmbedder",
    "decode_landmark_heatmaps",
]
