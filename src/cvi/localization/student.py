"""Student distillation interface for runtime localization.

The runtime student is a single model distilled from the teacher ensemble.
It must not depend on the ensemble at inference time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvi.localization.types import DetectionBox, KeypointSet


@dataclass(frozen=True, slots=True)
class TeacherLabel:
    image_id: str
    dog_boxes: tuple[DetectionBox, ...]
    body_keypoints: tuple[KeypointSet, ...]
    face_landmarks: tuple[KeypointSet, ...]
    admission: str


class AbstractStudentTrainer(ABC):
    model_family: str
    model_name: str

    @abstractmethod
    def train(
        self,
        labels: tuple[TeacherLabel, ...],
        *,
        output_dir: Path,
        device: str = "cuda",
        epochs: int = 50,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def export_onnx(self, checkpoint_path: Path, output_path: Path) -> None:
        ...

    @property
    @abstractmethod
    def expected_checkpoint_bytes(self) -> int:
        ...


__all__ = ["AbstractStudentTrainer", "TeacherLabel"]
