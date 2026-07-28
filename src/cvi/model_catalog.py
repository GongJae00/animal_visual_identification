"""Content-bound model inventory with logical role aliases."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cvi.model_paths import CHECKPOINTS_DIR

_ARTIFACT_ID = re.compile(r"[a-z0-9][a-z0-9.-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ModelAdmission(str, Enum):
    RESEARCH_ONLY = "RESEARCH_ONLY"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    artifact_id: str
    relative_path: str
    sha256: str
    source_model_id: str
    source_revision: str
    license_id: str
    admission: ModelAdmission

    def __post_init__(self) -> None:
        if _ARTIFACT_ID.fullmatch(self.artifact_id) is None:
            raise ValueError("model artifact_id is not canonical")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
            raise ValueError(
                "model relative_path must identify a file under its artifact directory"
            )
        if path.parts[0] != self.artifact_id:
            raise ValueError("model relative_path must start with artifact_id")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("model sha256 must be a lowercase SHA256 digest")
        for name in ("source_model_id", "source_revision", "license_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"model {name} must be non-empty")
        if not isinstance(self.admission, ModelAdmission):
            raise TypeError("model admission must be a ModelAdmission")

    def path(self, checkpoints_dir: Path = CHECKPOINTS_DIR) -> Path:
        return checkpoints_dir / self.relative_path


MODEL_CATALOG: tuple[ModelArtifact, ...] = (
    ModelArtifact(
        artifact_id="dinov2-small-onnx-cpu-parity-v1-20260726",
        relative_path=(
            "dinov2-small-onnx-cpu-parity-v1-20260726/"
            "dinov2-small-ed25f3a3-20260726-cpu-parity-v1.onnx"
        ),
        sha256="980a6565f7fc7c3fec07ac33d1e7e9b31ddb00cc79fc7fe73541d5d0bbacce92",
        source_model_id="facebook/dinov2-small",
        source_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        license_id="Apache-2.0",
        admission=ModelAdmission.DEPLOYMENT_CANDIDATE,
    ),
    ModelArtifact(
        artifact_id="hf-facebook-dinov2-small-ed25f3a31f01632728cabb09d1542f84ab7b0056",
        relative_path=(
            "hf-facebook-dinov2-small-ed25f3a31f01632728cabb09d1542f84ab7b0056/"
            "model.safetensors"
        ),
        sha256="ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        source_model_id="facebook/dinov2-small",
        source_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        license_id="Apache-2.0",
        admission=ModelAdmission.DEPLOYMENT_CANDIDATE,
    ),
    ModelArtifact(
        artifact_id="hf-timm-mobilenetv4-conv-small-c9f31ac64483d7f0590db9edccb4418392a96eea",
        relative_path=(
            "hf-timm-mobilenetv4-conv-small-c9f31ac64483d7f0590db9edccb4418392a96eea/"
            "model.safetensors"
        ),
        sha256="5a2ef04d419ce6d1bf27bfa735bb200d3f8d8997c3ac36320f5bf30382f6b43c",
        source_model_id="timm/mobilenetv4_conv_small.e1200_r224_in1k",
        source_revision="c9f31ac64483d7f0590db9edccb4418392a96eea",
        license_id="Apache-2.0",
        admission=ModelAdmission.DEPLOYMENT_CANDIDATE,
    ),
    ModelArtifact(
        artifact_id="mobilenetv4-conv-small-onnx-export",
        relative_path="mobilenetv4-conv-small-onnx-export/mobilenetv4-conv-small.onnx",
        sha256="1e5a02df8052a8be5c4680a6548d466fa47fee98cf354d21a471739e3211cba9",
        source_model_id="timm/mobilenetv4_conv_small.e1200_r224_in1k",
        source_revision="c9f31ac64483d7f0590db9edccb4418392a96eea",
        license_id="Apache-2.0",
        admission=ModelAdmission.RESEARCH_ONLY,
    ),
    ModelArtifact(
        artifact_id="yolo11n-coco-20260728",
        relative_path="yolo11n-coco-20260728/yolo11n.pt",
        sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        source_model_id="ultralytics/yolo11n",
        source_revision="unverified-local-acquisition",
        license_id="AGPL-3.0-only",
        admission=ModelAdmission.RESEARCH_ONLY,
    ),
    ModelArtifact(
        artifact_id="fasterrcnn-resnet50-fpn-v2-coco-20260728",
        relative_path=(
            "fasterrcnn-resnet50-fpn-v2-coco-20260728/"
            "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        ),
        sha256="dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf",
        source_model_id="torchvision/fasterrcnn_resnet50_fpn_v2",
        source_revision="COCO_V1-dd69338a",
        license_id="Unknown",
        admission=ModelAdmission.RESEARCH_ONLY,
    ),
    ModelArtifact(
        artifact_id="hrnet-w32-ap10k-20260728",
        relative_path=(
            "hrnet-w32-ap10k-20260728/hrnet_w32_ap10k_256x256-18aac840_20211029.pth"
        ),
        sha256="18aac840eee49f190e1443d93c6c23784920d2fd90813a42d35c918ec23a36e9",
        source_model_id="openmmlab/hrnet-w32-ap10k",
        source_revision="2021-10-29-18aac840",
        license_id="Unknown",
        admission=ModelAdmission.RESEARCH_ONLY,
    ),
    ModelArtifact(
        artifact_id="yolo11n-pose-coco-20260728",
        relative_path="yolo11n-pose-coco-20260728/yolo11n-pose.pt",
        sha256="869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
        source_model_id="ultralytics/yolo11n-pose",
        source_revision="unverified-local-acquisition",
        license_id="AGPL-3.0-only",
        admission=ModelAdmission.RESEARCH_ONLY,
    ),
    ModelArtifact(
        artifact_id="yolo11n-pose-ap10k-dog-v2-20260728",
        relative_path="yolo11n-pose-ap10k-dog-v2-20260728/best.pt",
        sha256="7edc2d96c2ca06942d172527b097d98b12bcf50c42b1c232b3b42ed0f1858760",
        source_model_id="cvi/yolo11n-pose-ap10k-dog-v2",
        source_revision="derived-local-training-receipt-unavailable",
        license_id="AGPL-3.0-only",
        admission=ModelAdmission.RESEARCH_ONLY,
    ),
)


MODEL_ROLE_ALIASES: dict[str, str] = {
    "appearance-backbone": "hf-facebook-dinov2-small-ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "appearance-onnx": "dinov2-small-onnx-cpu-parity-v1-20260726",
    "dog-detector": "yolo11n-pose-ap10k-dog-v2-20260728",
    "dog-detector-coco": "yolo11n-coco-20260728",
    "dog-detector-teacher": "fasterrcnn-resnet50-fpn-v2-coco-20260728",
    "dog-pose": "yolo11n-pose-ap10k-dog-v2-20260728",
    "dog-pose-initializer": "yolo11n-pose-coco-20260728",
    "dog-pose-reference": "hrnet-w32-ap10k-20260728",
    "mobile-appearance-backbone": "hf-timm-mobilenetv4-conv-small-c9f31ac64483d7f0590db9edccb4418392a96eea",
    "mobile-appearance-onnx": "mobilenetv4-conv-small-onnx-export",
}


_CATALOG_BY_ID = {artifact.artifact_id: artifact for artifact in MODEL_CATALOG}
if len(_CATALOG_BY_ID) != len(MODEL_CATALOG):
    raise RuntimeError("duplicate model artifact_id")
if set(MODEL_ROLE_ALIASES) & set(_CATALOG_BY_ID):
    raise RuntimeError("model role aliases must not collide with artifact IDs")
if unknown := set(MODEL_ROLE_ALIASES.values()) - set(_CATALOG_BY_ID):
    raise RuntimeError(
        f"model role aliases reference unknown artifacts: {sorted(unknown)}"
    )


def get_model_artifact(artifact_or_role: str) -> ModelArtifact:
    artifact_id = MODEL_ROLE_ALIASES.get(artifact_or_role, artifact_or_role)
    try:
        return _CATALOG_BY_ID[artifact_id]
    except KeyError as exc:
        raise KeyError(f"unknown model artifact or role: {artifact_or_role!r}") from exc


def verify_model_artifact(
    artifact_or_role: str | ModelArtifact,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
) -> Path:
    artifact = (
        get_model_artifact(artifact_or_role)
        if isinstance(artifact_or_role, str)
        else artifact_or_role
    )
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("artifact_or_role must be a model name, role, or ModelArtifact")
    path = artifact.path(checkpoints_dir)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"model artifact is not a regular non-symlink file: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise ValueError(f"model artifact SHA256 mismatch: {artifact.artifact_id}")
    return path


__all__ = [
    "MODEL_CATALOG",
    "MODEL_ROLE_ALIASES",
    "ModelAdmission",
    "ModelArtifact",
    "get_model_artifact",
    "verify_model_artifact",
]
