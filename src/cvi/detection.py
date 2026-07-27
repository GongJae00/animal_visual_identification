from __future__ import annotations

import hashlib
import itertools
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image



def YOLO(*args: Any, **kwargs: Any) -> Any:
    """Load the separately licensed optional detector only when requested."""

    try:
        from ultralytics import YOLO as ultralytics_yolo
    except ImportError as exc:
        raise RuntimeError(
            "dog detection requires a separately installed and licensed "
            "Ultralytics package"
        ) from exc
    return ultralytics_yolo(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def face_region(self, face_ratio: float = 0.45) -> Detection:
        face_h = int(self.height * face_ratio)
        return Detection(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y1 + face_h,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "confidence": round(self.confidence, 4),
            "class": self.class_name,
        }


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    sharpness: float
    face_coverage: float
    brightness: float
    is_blurry: bool
    is_dark: bool

    def acceptable(self, min_sharpness: float = 50.0,
                   min_coverage: float = 0.1,
                   min_brightness: float = 0.05) -> bool:
        if self.is_blurry and self.sharpness < min_sharpness:
            return False
        if self.face_coverage < min_coverage:
            return False
        if self.is_dark and self.brightness < min_brightness:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpness": round(self.sharpness, 2),
            "face_coverage": round(self.face_coverage, 4),
            "brightness": round(self.brightness, 4),
            "is_blurry": self.is_blurry,
            "is_dark": self.is_dark,
        }


@dataclass
class DogDetectorConfig:
    model_path: str | None = None
    model_sha256: str | None = None
    model_size: str = "n"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    max_detections: int = 5
    input_size: int = 640
    target_size: int = 224
    face_ratio: float = 0.45
    min_sharpness: float = 50.0
    min_face_coverage: float = 0.1
    landmark_model_path: str | None = None
    use_alignment: bool = False


class DogDetector:
    def __init__(self, config: DogDetectorConfig | None = None) -> None:
        self._cfg = config or DogDetectorConfig()
        device = self._cfg.device if self._cfg.device else "cpu"
        if not self._cfg.model_path or not self._cfg.model_sha256:
            raise RuntimeError(
                "dog detection requires an explicit local model_path and SHA256"
            )
        model_path = Path(self._cfg.model_path)
        self._model_staging = tempfile.TemporaryDirectory(
            prefix="cvi-detector-model-"
        )
        staged_path = Path(self._model_staging.name) / "model.pt"
        try:
            descriptor = os.open(
                model_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            with (
                os.fdopen(descriptor, "rb") as source,
                staged_path.open("xb") as target,
            ):
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("dog detector model must be a regular file")
                if before.st_size <= 0 or before.st_size > 2_147_483_648:
                    raise ValueError("dog detector model size is invalid")
                digest = hashlib.sha256()
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
                after = os.fstat(source.fileno())
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or staged_path.stat().st_size != before.st_size
            ):
                raise RuntimeError("dog detector model changed while being verified")
            if digest.hexdigest() != self._cfg.model_sha256:
                raise RuntimeError("dog detector model SHA256 differs")
            self._model = YOLO(str(staged_path))
        except BaseException:
            self._model_staging.cleanup()
            raise
        try:
            self._model.to(device)
        except BaseException:
            self._model_staging.cleanup()
            raise

    @property
    def config(self) -> DogDetectorConfig:
        return self._cfg

    def detect(self, image: Image.Image) -> list[Detection]:
        orig_w, orig_h = image.size
        results = self._model(
            image,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.iou_threshold,
            max_det=self._cfg.max_detections,
            imgsz=self._cfg.input_size,
            verbose=False,
        )[0]
        dets: list[Detection] = []
        if results.boxes is None:
            return dets
        for box, conf, cls_id in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = box
            x1_i = max(0, int(x1))
            y1_i = max(0, int(y1))
            x2_i = min(orig_w, int(x2))
            y2_i = min(orig_h, int(y2))
            class_name = results.names.get(cls_id, "unknown")
            dets.append(Detection(
                x1=x1_i, y1=y1_i, x2=x2_i, y2=y2_i,
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=class_name,
            ))
        return dets

    def detect_dogs(self, image: Image.Image) -> list[Detection]:
        all_dets = self.detect(image)
        return [d for d in all_dets if d.class_name == "dog"]

    def crop_face(self, image: Image.Image, det: Detection,
                  size: int | None = None) -> Image.Image:
        face = det.face_region(self._cfg.face_ratio)
        crop = image.crop((face.x1, face.y1, face.x2, face.y2))
        target = size or self._cfg.target_size
        return crop.resize((target, target), Image.BILINEAR)

    @staticmethod
    def compute_quality(image: Image.Image,
                        det: Detection | None = None
                        ) -> QualityMetrics:
        arr = np.array(image.convert("L"), dtype=np.float32)
        lap = np.gradient(arr)[0]
        sharpness = float(np.var(lap))

        if det is not None:
            face_h = int(det.height * 0.45)
            face_area = det.width * face_h
            total_area = det.width * det.height
            face_coverage = face_area / max(total_area, 1)
        else:
            face_coverage = 1.0

        mean_brightness = float(np.mean(arr) / 255.0)

        return QualityMetrics(
            sharpness=sharpness,
            face_coverage=face_coverage,
            brightness=mean_brightness,
            is_blurry=sharpness < 100.0,
            is_dark=mean_brightness < 0.1,
        )

    def detect_and_crop(self, image: Image.Image,
                        quality_filter: bool = True
                        ) -> list[tuple[Detection, Image.Image, QualityMetrics]]:
        dogs = self.detect_dogs(image)
        results: list[tuple[Detection, Image.Image, QualityMetrics]] = []
        for d in dogs:
            q = self.compute_quality(image, d)
            if quality_filter and not q.acceptable(
                min_sharpness=self._cfg.min_sharpness,
                min_coverage=self._cfg.min_face_coverage,
            ):
                continue
            crop = self.crop_face(image, d)
            results.append((d, crop, q))
        return results

    def close(self) -> None:
        del self._model
        self._model_staging.cleanup()
        import torch
        torch.cuda.empty_cache()


class FrameSelector:
    def __init__(self, top_k_frames: int = 3,
                 min_sharpness: float = 50.0) -> None:
        self._top_k = top_k_frames
        self._min_sharpness = min_sharpness

    def select(self, frames: list[tuple[int, Image.Image, Detection]]
               ) -> list[tuple[int, Image.Image, Detection]]:
        scored: list[tuple[float, int, Image.Image, Detection]] = []
        for idx, img, det in frames:
            q = DogDetector.compute_quality(img, det)
            score = q.sharpness * (det.area ** 0.5)
            scored.append((score, idx, img, det))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(idx, img, det) for score, idx, img, det
                in scored[:self._top_k]]
