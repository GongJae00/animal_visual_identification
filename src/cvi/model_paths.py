from __future__ import annotations

import os
from pathlib import Path

# ── 모델 저장 경로 ──
MODELS_DIR = Path(
    os.environ.get(
        "CVI_MODELS_DIR",
        Path.home() / ".cache" / "cvi" / "models",
    )
)

# ── 데이터 저장 경로 ──
# CVI_DATA_DIR=/mnt/ssd/canine_data  or  ln -s /mnt/ssd/canine_data ~/cvi_data
DATA_DIR = Path(
    os.environ.get(
        "CVI_DATA_DIR",
        Path.home() / "cvi_data",
    )
)

# 하위 디렉토리 (심볼릭 링크 사용 가능)
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_REGISTRY_DIR = DATA_DIR / "registry"

# ── 지원 데이터셋 ──
SUPPORTED_DATASETS: dict[str, dict] = {
    "yt-bb-dog": {
        "name": "YouTube-BoundingBoxes Dog",
        "dir": "yt_bb_dog",
        "desc": "유튜브 영상에서 추출한 개 크롭 (12,078장, 25품종)",
    },
    "dogfacenet": {
        "name": "DogFaceNet",
        "dir": "dogfacenet",
        "desc": "개 얼굴 인식 데이터셋 (품종별 정면 얼굴)",
    },
    "mpdd": {
        "name": "MPetDoorDataset",
        "dir": "mpdd",
        "desc": "펫도어 카메라로 촬영한 반려견 데이터셋",
    },
    "sibetan": {
        "name": "SiBeTan",
        "dir": "sibetan",
        "desc": "시베리안 허스키 + 벨지안 말리노이즈 품종 식별",
    },
}


def dataset_path(name: str) -> Path:
    if name not in SUPPORTED_DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Supported: {list(SUPPORTED_DATASETS)}")
    return DATA_RAW_DIR / SUPPORTED_DATASETS[name]["dir"]


def processed_path(name: str) -> Path:
    return DATA_PROCESSED_DIR / SUPPORTED_DATASETS.get(name, {"dir": name})["dir"]

# ── 모델 다운로드 URL ──
DOGFLW_LANDMARK_URL = (
    "https://huggingface.co/datasets/canine-video-identity/dogflw-models/resolve/main/"
    "dogflw_landmark_full.tflite"
)
DOGFLW_LANDMARK_PATH = MODELS_DIR / "dogflw_landmark_full.tflite"
DOGFLW_LANDMARK_MD5 = "e4e0a6b0f8a9b1c2d3e4f5a6b7c8d9e0"

SUPERANIMAL_QUADRUPED_URL = (
    "https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/resolve/main/"
    "superanimal_quadruped_hrnet_w32.pt"
)
SUPERANIMAL_QUADRUPED_PATH = MODELS_DIR / "superanimal_quadruped_hrnet_w32.pt"
SUPERANIMAL_ONNX_PATH = MODELS_DIR / "superanimal_quadruped.onnx"

MIEWID_NOSE_ONNX_URL = (
    "https://huggingface.co/james-burgess/miewid/resolve/main/miewid.onnx"
)
MIEWID_NOSE_ONNX_PATH = MODELS_DIR / "miewid_nose.onnx"
