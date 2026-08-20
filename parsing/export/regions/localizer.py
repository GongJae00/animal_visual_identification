"""Research-only partial-keypoint dog nose parsing.

The model consumes caller-provided dog or face crops. Training coordinates use
the same deterministic stretch resize declared by the ONNX runtime manifest,
so exported normalized boxes map directly back to the caller crop. AP-10K's
nose-center annotation supervises the shared ``nasal_inferior`` (nose-tip)
channel; DogFLW supervises all channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
from torch import nn

MOBILENETV4_MODEL_NAME = "mobilenetv4_conv_small.e1200_r224_in1k"
MOBILENETV4_WEIGHTS_SHA256 = (
    "5a2ef04d419ce6d1bf27bfa735bb200d3f8d8997c3ac36320f5bf30382f6b43c"
)
INPUT_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
KEYPOINT_ORDER = (
    "left_eye_center",
    "right_eye_center",
    "nasal_root",
    "nasal_inferior",
    "left_nostril_center",
    "right_nostril_center",
    "left_alar_boundary",
    "right_alar_boundary",
)
NOSE_POINT_INDICES = tuple(range(2, 8))
AP10K_SUPPORTED_INDICES = (0, 1, 3)
DOGFLW_DERIVATION = {
    "left_eye_center": (16, 18, 20, 22),
    "right_eye_center": (17, 19, 21, 23),
    "nasal_root": (25,),
    "nasal_inferior": (35,),
    "left_nostril_center": (32, 33),
    "right_nostril_center": (32, 34),
    "left_alar_boundary": (26,),
    "right_alar_boundary": (27,),
}

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Integer resize and padding geometry for a deterministic square canvas."""

    source_width: int
    source_height: int
    target_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @classmethod
    def create(
        cls, source_width: int, source_height: int, target_size: int = INPUT_SIZE
    ) -> LetterboxTransform:
        if min(source_width, source_height, target_size) <= 0:
            raise ValueError("letterbox dimensions must be positive")
        scale = min(target_size / source_width, target_size / source_height)
        resized_width = min(target_size, max(1, int(math.floor(source_width * scale + 0.5))))
        resized_height = min(
            target_size, max(1, int(math.floor(source_height * scale + 0.5)))
        )
        return cls(
            source_width=source_width,
            source_height=source_height,
            target_size=target_size,
            resized_width=resized_width,
            resized_height=resized_height,
            pad_left=(target_size - resized_width) // 2,
            pad_top=(target_size - resized_height) // 2,
        )

    def normalized_point(self, x: float, y: float) -> tuple[float, float]:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("keypoint coordinates must be finite")
        transformed_x = self.pad_left + x * self.resized_width / self.source_width
        transformed_y = self.pad_top + y * self.resized_height / self.source_height
        return transformed_x / self.target_size, transformed_y / self.target_size

    @property
    def normalized_content_diagonal(self) -> float:
        return math.hypot(self.resized_width, self.resized_height) / self.target_size
def letterbox_image(
    image: Image.Image,
    target_size: int = INPUT_SIZE,
    *,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, LetterboxTransform]:
    """Return a bilinear RGB letterbox and its exact integer geometry."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    transform = LetterboxTransform.create(*image.size, target_size)
    resized = image.convert("RGB").resize(
        (transform.resized_width, transform.resized_height),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (target_size, target_size), fill)
    canvas.paste(resized, (transform.pad_left, transform.pad_top))
    return canvas, transform
def image_to_tensor(image: Image.Image) -> torch.Tensor:
    values = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(IMAGE_STD, dtype=np.float32)
    values = (values - mean) / std
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))
class MobileNetV4NoseLocalizer(nn.Module):
    """MobileNetV4 Conv Small with an eight-keypoint xy-confidence head."""

    def __init__(self, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, len(KEYPOINT_ORDER) * 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if hasattr(self.backbone, "forward_head"):
            features = self.backbone.forward_head(features, pre_logits=True)
        elif features.ndim == 4:
            features = features.mean(dim=(2, 3))
        raw = self.head(features).reshape(images.shape[0], len(KEYPOINT_ORDER), 3)
        return torch.sigmoid(raw)
def mobilenetv4_feature_dim(backbone: nn.Module) -> int:
    """Return the actual pre-logits width, not timm's map-channel metadata."""

    classifier = getattr(backbone, "classifier", None)
    feature_dim = getattr(classifier, "in_features", None)
    if isinstance(feature_dim, int) and feature_dim > 0:
        return feature_dim
    feature_dim = getattr(backbone, "num_features", None)
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("MobileNetV4 pre-logits feature dimension is unavailable")
    return feature_dim
def load_mobilenetv4_localizer(weights_path: Path) -> MobileNetV4NoseLocalizer:
    """Load the exact caller-provided timm safetensors with strict key matching."""

    path = Path(weights_path)
    if path.suffix != ".safetensors" or path.is_symlink() or not path.is_file():
        raise ValueError("weights_path must be a regular .safetensors file")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != MOBILENETV4_WEIGHTS_SHA256:
        raise ValueError(
            "MobileNetV4 safetensors SHA256 differs: "
            f"expected {MOBILENETV4_WEIGHTS_SHA256}, got {actual_sha256}"
        )
    from safetensors.torch import load_file
    import timm

    backbone = timm.create_model(MOBILENETV4_MODEL_NAME, pretrained=False)
    state_dict = load_file(str(path), device="cpu")
    backbone.load_state_dict(state_dict, strict=True)
    return MobileNetV4NoseLocalizer(backbone, mobilenetv4_feature_dim(backbone))
class NoseDetectorWrapper(nn.Module):
    """Differentiably convert six NoseID keypoints to one xyxy-confidence box."""

    def __init__(self, localizer: nn.Module, margin: float = 0.08) -> None:
        super().__init__()
        if not 0.0 < margin < 0.5:
            raise ValueError("detector margin must be in (0, 0.5)")
        self.localizer = localizer
        self.margin = margin

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        predictions = self.localizer(images)
        nose = predictions[:, 2:, :]
        minimum = nose[..., :2].amin(dim=1) - self.margin
        maximum = nose[..., :2].amax(dim=1) + self.margin
        confidence = nose[..., 2].mean(dim=1, keepdim=True)
        detection = torch.cat((minimum.clamp(0.0, 1.0), maximum.clamp(0.0, 1.0), confidence), dim=1)
        return detection.unsqueeze(1)
def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

__all__ = [
    "AP10K_SUPPORTED_INDICES",
    "DOGFLW_DERIVATION",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "INPUT_SIZE",
    "KEYPOINT_ORDER",
    "NOSE_POINT_INDICES",
    "LetterboxTransform",
    "MOBILENETV4_MODEL_NAME",
    "MOBILENETV4_WEIGHTS_SHA256",
    "MobileNetV4NoseLocalizer",
    "NoseDetectorWrapper",
    "file_sha256",
    "image_to_tensor",
    "letterbox_image",
    "load_mobilenetv4_localizer",
    "mobilenetv4_feature_dim",
]
