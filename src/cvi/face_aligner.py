from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class FaceLandmarks:
    keypoints: np.ndarray  # (N, 2) array of (x, y) in image coordinates
    confidence: float

    @property
    def num_keypoints(self) -> int:
        return self.keypoints.shape[0]


@dataclass(frozen=True, slots=True)
class AlignedFace:
    image: Image.Image
    landmarks: FaceLandmarks
    transform_matrix: np.ndarray | None = None


class LandmarkDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image) -> FaceLandmarks | None:
        ...

    @abstractmethod
    def detect_batch(self, images: list[Image.Image]) -> list[FaceLandmarks | None]:
        ...


class DogFLWLandmarkDetector(LandmarkDetector):
    def __init__(self, model_path: Path, use_alignment: bool = True) -> None:
        self._model_path = model_path
        self._use_alignment = use_alignment

    def _load_interpreter(self):
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=str(self._model_path))

    def detect(self, image: Image.Image) -> FaceLandmarks | None:
        if not self._model_path.exists():
            return None
        interpreter = self._load_interpreter()
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_size = input_details[0]["shape"][1]
        img = image.convert("RGB").resize((input_size, input_size))
        inp = np.array(img, dtype=np.uint8)[np.newaxis, :]
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])[0]
        h, w = image.size[1], image.size[0]
        scale_x = w / input_size
        scale_y = h / input_size
        out[::2] *= scale_x
        out[1::2] *= scale_y
        kpts = out.reshape(-1, 2).astype(np.float32)
        return FaceLandmarks(keypoints=kpts, confidence=1.0)

    def detect_batch(self, images: list[Image.Image]
                     ) -> list[FaceLandmarks | None]:
        return [self.detect(img) for img in images]


class HeuristicLandmarkDetector(LandmarkDetector):
    def detect(self, image: Image.Image) -> FaceLandmarks | None:
        w, h = image.size
        nose = np.array([[w * 0.5, h * 0.6]], dtype=np.float32)
        leye = np.array([[w * 0.3, h * 0.35]], dtype=np.float32)
        reye = np.array([[w * 0.7, h * 0.35]], dtype=np.float32)
        kpts = np.concatenate([nose, leye, reye], axis=0)
        return FaceLandmarks(keypoints=kpts, confidence=0.5)

    def detect_batch(self, images: list[Image.Image]
                     ) -> list[FaceLandmarks | None]:
        return [self.detect(img) for img in images]


class FaceAligner:
    def __init__(self, landmark_detector: LandmarkDetector | None = None,
                 target_size: int = 224,
                 left_eye_idx: int = 1,
                 right_eye_idx: int = 2) -> None:
        self._detector = landmark_detector or HeuristicLandmarkDetector()
        self._target_size = target_size
        self._left_eye_idx = left_eye_idx
        self._right_eye_idx = right_eye_idx

    def align(self, image: Image.Image) -> AlignedFace | None:
        landmarks = self._detector.detect(image)
        if landmarks is None or landmarks.num_keypoints < 3:
            return None
        kpts = landmarks.keypoints
        aligned = self._apply_affine(image, kpts)
        return AlignedFace(
            image=aligned,
            landmarks=landmarks,
        )

    def _apply_affine(self, image: Image.Image,
                      kpts: np.ndarray) -> Image.Image:
        if kpts.shape[0] >= 3:
            le = kpts[self._left_eye_idx]
            re = kpts[self._right_eye_idx]
            nose = kpts[0]
            target_le = np.array([0.35, 0.35], dtype=np.float32)
            target_re = np.array([0.65, 0.35], dtype=np.float32)
            target_nose = np.array([0.5, 0.55], dtype=np.float32)
            src = np.array([le, re, nose], dtype=np.float32)
            dst = np.array([
                [target_le[0] * self._target_size, target_le[1] * self._target_size],
                [target_re[0] * self._target_size, target_re[1] * self._target_size],
                [target_nose[0] * self._target_size, target_nose[1] * self._target_size],
            ], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is None:
                M = cv2.getAffineTransform(src[:2], dst[:2])
            aligned = cv2.warpAffine(
                np.array(image), M, (self._target_size, self._target_size),
                flags=cv2.INTER_LINEAR,
            )
            return Image.fromarray(aligned)
        return image.resize((self._target_size, self._target_size), Image.BILINEAR)

    def align_batch(self, images: list[Image.Image]) -> list[AlignedFace | None]:
        landmarks_batch = self._detector.detect_batch(images)
        results: list[AlignedFace | None] = []
        for img, lm in zip(images, landmarks_batch):
            if lm is None:
                results.append(None)
                continue
            aligned = self._apply_affine(img, lm.keypoints)
            results.append(AlignedFace(image=aligned, landmarks=lm))
        return results
