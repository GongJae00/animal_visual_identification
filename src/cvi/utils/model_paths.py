"""Central model paths and download configuration."""

import os
from pathlib import Path

_MODELS_DIR_ENV = "CVI_MODELS_DIR"
_MODELS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "models"

MODELS_DIR = Path(os.environ.get(_MODELS_DIR_ENV, str(_MODELS_DIR_DEFAULT)))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-trained model paths
DOGFLW_LANDMARK_PATH = MODELS_DIR / "pretrained" / "dogflw_landmark.tflite"
SUPERANIMAL_QUADRUPED_PATH = MODELS_DIR / "pretrained" / "superanimal_quadruped.pth"
MIEWID_NOSE_ONNX_PATH = MODELS_DIR / "onnx" / "miewid.onnx"

# Training outputs
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_EXPORT_DIR = MODELS_DIR / "onnx"
BACKBONE_WEIGHTS_DIR = MODELS_DIR / "backbones"
