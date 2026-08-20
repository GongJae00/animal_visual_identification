"""Model adapter interface for localization models."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from parsing.export.types import (
    AP10K_BODY_17_KEYPOINT_NAMES,
    AP10K_BODY_17_SCHEMA,
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationResult,
)

def _stage_verified_artifact(
    path: Path, expected_sha256: str
) -> tuple[tempfile.TemporaryDirectory[str], Path, int]:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("model SHA256 must be a lowercase digest")
    source_path = Path(os.path.abspath(os.fspath(path)))
    staging = tempfile.TemporaryDirectory(prefix="cvi-localizer-model-")
    staged_path = Path(staging.name) / f"model{source_path.suffix}"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(source_path, flags)
        with (
            os.fdopen(descriptor, "rb") as source,
            staged_path.open("xb") as target,
        ):
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise ValueError("localization model must be a non-empty regular file")
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
            after = os.fstat(source.fileno())
        named = os.stat(source_path, follow_symlinks=False)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            before_identity != after_identity
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
            or staged_path.stat().st_size != before.st_size
        ):
            raise RuntimeError("localization model changed while being verified")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("localization model SHA256 differs")
        return staging, staged_path, before.st_size
    except BaseException:
        staging.cleanup()
        raise


class AbstractLocalizationAdapter(ABC):
    model_family: str
    model_name: str
    requires_gpu: bool = False

    @abstractmethod
    def detect(
        self, image: Image.Image, *, image_id: str = ""
    ) -> LocalizationResult: ...

    @abstractmethod
    def detect_batch(
        self, images: list[Image.Image], *, image_ids: list[str] | None = None
    ) -> list[LocalizationResult]: ...

    @property
    @abstractmethod
    def artifact_size_bytes(self) -> int: ...

    @property
    @abstractmethod
    def license_id(self) -> str: ...

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def device(self) -> str:
        return str(self._device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.model_family,
            "name": self.model_name,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "license_id": self.license_id,
            "device": self.device,
        }


class UltralyticsDogAdapter(AbstractLocalizationAdapter):
    """Pinned Ultralytics detector exposed through the localization contract."""

    model_family = "ultralytics-coco-detection"
    requires_gpu = True

    def __init__(self, model_path: Path, model_sha256: str, *, device: str) -> None:
        from parsing.export.detection.detection import DogDetector, DogDetectorConfig

        self._model_path = Path(model_path)
        self._artifact_sha256 = model_sha256
        self._device = device
        self._model_staging, staged_path, self._artifact_size_bytes = (
            _stage_verified_artifact(self._model_path, model_sha256)
        )
        try:
            self._detector = DogDetector(
                DogDetectorConfig(
                    model_path=str(staged_path),
                    model_sha256=model_sha256,
                    device=device,
                )
            )
        except BaseException:
            self._model_staging.cleanup()
            raise
        self.model_name = self._model_path.stem

    def detect(self, image: Image.Image, *, image_id: str = "") -> LocalizationResult:
        boxes = tuple(
            DetectionBox(
                float(box.x1),
                float(box.y1),
                float(box.x2),
                float(box.y2),
                box.confidence,
                box.class_id,
                box.class_name,
            )
            for box in self._detector.detect_dogs(image)
        )
        return LocalizationResult(
            image_id=image_id,
            dog_boxes=boxes,
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=(),
            face_landmarks=(),
            model_name=self.model_name,
            model_family=self.model_family,
            inference_ms=0.0,
            metadata={"artifact_sha256": self._artifact_sha256},
        )

    def detect_batch(
        self, images: list[Image.Image], *, image_ids: list[str] | None = None
    ) -> list[LocalizationResult]:
        ids = image_ids or [str(index) for index in range(len(images))]
        if len(ids) != len(images):
            raise ValueError("image_ids must match images")
        return [
            self.detect(image, image_id=image_id)
            for image, image_id in zip(images, ids, strict=True)
        ]

    @property
    def artifact_size_bytes(self) -> int:
        return self._artifact_size_bytes

    @property
    def license_id(self) -> str:
        return "AGPL-3.0-only"

    def close(self) -> None:
        self._detector.close()
        self._model_staging.cleanup()


class TorchvisionFasterRCNNDogAdapter(AbstractLocalizationAdapter):
    """Locally pinned Faster R-CNN COCO dog detector."""

    model_family = "torchvision-fasterrcnn"
    model_name = "fasterrcnn-resnet50-fpn-v2-coco"
    requires_gpu = True

    def __init__(self, model_path: Path, model_sha256: str, *, device: str) -> None:
        self._model_path = Path(model_path)
        self._artifact_sha256 = model_sha256
        self._model_staging, staged_path, self._artifact_size_bytes = (
            _stage_verified_artifact(self._model_path, model_sha256)
        )
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

            self._device = torch.device(device)
            self._model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                box_score_thresh=0.25,
                box_nms_thresh=0.45,
                box_detections_per_img=5,
            )
            state = torch.load(staged_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state, strict=True)
            self._model.eval().to(self._device)
        except BaseException:
            self._model_staging.cleanup()
            raise

    def detect(self, image: Image.Image, *, image_id: str = "") -> LocalizationResult:
        return self.detect_batch([image], image_ids=[image_id])[0]

    def detect_batch(
        self, images: list[Image.Image], *, image_ids: list[str] | None = None
    ) -> list[LocalizationResult]:
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        ids = image_ids or [str(index) for index in range(len(images))]
        if len(ids) != len(images):
            raise ValueError("image_ids must match images")
        tensors = [
            pil_to_tensor(image).to(self._device, dtype=torch.float32) / 255.0
            for image in images
        ]
        with torch.inference_mode():
            outputs = self._model(tensors)
        results: list[LocalizationResult] = []
        for image_id, image, output in zip(ids, images, outputs, strict=True):
            boxes: list[DetectionBox] = []
            for raw_box, score, label in zip(
                output["boxes"].cpu().tolist(),
                output["scores"].cpu().tolist(),
                output["labels"].cpu().tolist(),
                strict=True,
            ):
                if int(label) != 18:
                    continue
                x1, y1, x2, y2 = raw_box
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(image.width), x2), min(float(image.height), y2)
                if x2 > x1 and y2 > y1:
                    boxes.append(DetectionBox(x1, y1, x2, y2, float(score), 18, "dog"))
            results.append(
                LocalizationResult(
                    image_id=image_id,
                    dog_boxes=tuple(boxes),
                    face_boxes=(),
                    nose_boxes=(),
                    body_keypoints=(),
                    face_landmarks=(),
                    model_name=self.model_name,
                    model_family=self.model_family,
                    inference_ms=0.0,
                    metadata={"artifact_sha256": self._artifact_sha256},
                )
            )
        return results

    @property
    def artifact_size_bytes(self) -> int:
        return self._artifact_size_bytes

    @property
    def license_id(self) -> str:
        return "Unknown"

    def close(self) -> None:
        import torch

        del self._model
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._model_staging.cleanup()


class UltralyticsDogPoseAdapter(AbstractLocalizationAdapter):
    """Fine-tuned AP-10K dog detector and 17-keypoint pose student."""

    model_family = "ultralytics-ap10k-dog-pose"
    requires_gpu = True

    def __init__(self, model_path: Path, model_sha256: str, *, device: str) -> None:
        self._model_path = Path(model_path)
        self._artifact_sha256 = model_sha256
        self._device = device
        self._model_staging, staged_path, self._artifact_size_bytes = (
            _stage_verified_artifact(self._model_path, model_sha256)
        )
        try:
            from parsing.export.detection.detection import YOLO

            self._model = YOLO(str(staged_path))
            self._model.to(device)
        except BaseException:
            self._model_staging.cleanup()
            raise
        self.model_name = self._model_path.stem

    def detect(self, image: Image.Image, *, image_id: str = "") -> LocalizationResult:
        output = self._model(
            image, conf=0.25, iou=0.45, max_det=5, imgsz=640, verbose=False
        )[0]
        boxes: list[DetectionBox] = []
        point_sets: list[KeypointSet] = []
        if output.boxes is not None and output.keypoints is not None:
            xy = output.keypoints.xy.cpu().numpy()
            confidence = output.keypoints.conf.cpu().numpy()
            for index, (raw_box, score, label) in enumerate(
                zip(
                    output.boxes.xyxy.cpu().numpy(),
                    output.boxes.conf.cpu().numpy(),
                    output.boxes.cls.cpu().numpy().astype(int),
                    strict=True,
                )
            ):
                if int(label) != 0:
                    continue
                x1, y1, x2, y2 = raw_box
                x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                x2, y2 = (
                    min(float(image.width), float(x2)),
                    min(float(image.height), float(y2)),
                )
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(DetectionBox(x1, y1, x2, y2, float(score), 0, "dog"))
                points = {
                    name: Keypoint(
                        float(x), float(y), float(confidence[index, point_index])
                    )
                    for point_index, (name, (x, y)) in enumerate(
                        zip(AP10K_BODY_17_KEYPOINT_NAMES, xy[index], strict=True)
                    )
                    if float(confidence[index, point_index]) > 0.0
                }
                point_sets.append(KeypointSet(points, AP10K_BODY_17_SCHEMA))
        return LocalizationResult(
            image_id=image_id,
            dog_boxes=tuple(boxes),
            face_boxes=(),
            nose_boxes=(),
            body_keypoints=tuple(point_sets),
            face_landmarks=(),
            model_name=self.model_name,
            model_family=self.model_family,
            inference_ms=0.0,
            metadata={"artifact_sha256": self._artifact_sha256},
        )

    def detect_batch(
        self, images: list[Image.Image], *, image_ids: list[str] | None = None
    ) -> list[LocalizationResult]:
        ids = image_ids or [str(index) for index in range(len(images))]
        if len(ids) != len(images):
            raise ValueError("image_ids must match images")
        return [
            self.detect(image, image_id=image_id)
            for image, image_id in zip(images, ids, strict=True)
        ]

    @property
    def artifact_size_bytes(self) -> int:
        return self._artifact_size_bytes

    @property
    def license_id(self) -> str:
        return "AGPL-3.0-only"

    def close(self) -> None:
        del self._model
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        self._model_staging.cleanup()


__all__ = [
    "AbstractLocalizationAdapter",
    "TorchvisionFasterRCNNDogAdapter",
    "UltralyticsDogAdapter",
    "UltralyticsDogPoseAdapter",
]
