from __future__ import annotations

import os
from pathlib import Path

# ── 데이터 저장 경로 ──
# CVI_DATA_DIR=/mnt/r/...  or  ln -s /your/data/path ~/cvi_data
DATA_DIR = Path(
    os.environ.get(
        "CVI_DATA_DIR",
        Path.home() / "cvi_data",
    )
)

# 데이터 하위 디렉토리 (실제 SSD/canine_video_identity_secure 구조 기준)
DATASETS_DIR = DATA_DIR / "datasets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = DATA_DIR / "cache"
RECEIPTS_DIR = DATA_DIR / "receipts"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
DOWNLOADS_DIR = DATA_DIR / "downloads"
MANIFESTS_DIR = DATA_DIR / "manifests"

# ── 레거시 하위 디렉토리 (optional, raw/processed/registry 구조 사용 시) ──
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_REGISTRY_DIR = DATA_DIR / "registry"

# ── 모델 저장 경로 ──
MODELS_DIR = Path(
    os.environ.get(
        "CVI_MODELS_DIR",
        Path.home() / ".cache" / "cvi" / "models",
    )
)

# ── 지원 데이터셋 ──
# url이 있으면 download_datasets.py가 자동 다운로드.
# url이 없으면 수동 준비 (라이선스 동의 필요 등).
SUPPORTED_DATASETS: dict[str, dict] = {
    "yt-bb-dog": {
        "name": "YouTube-BoundingBoxes Dog",
        "dir": "yt-bb-dog-outer-official-2026-07-22",
        "url": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "desc": "유튜브 영상에서 추출한 개 크롭 (27,036장, 2,723마리)",
    },
    "dogfacenet": {
        "name": "DogFaceNet",
        "dir": "dogfacenet-224-zenodo-12578449-v1",
        "desc": "개 얼굴 인식 데이터셋 (8,363장, 1,393마리)",
    },
    "mpdd": {
        "name": "MPetDoorDataset",
        "dir": "mpdd-mendeley-v5j6m8dzhv-v1",
        "url": "https://github.com/hacilab/MPDD",
        "desc": "펫도어 카메라 촬영 반려견 데이터셋 (1,657장) — 라이선스 동의 필요",
    },
    "sibetan": {
        "name": "SiBeTan",
        "dir": "sibetan-official-2026-07-22",
        "url": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "desc": "롱텀 교차카메라 개 재식별 데이터셋 (1,755장, 59마리)",
    },
}


def dataset_path(name: str) -> Path:
    if name not in SUPPORTED_DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Supported: {list(SUPPORTED_DATASETS)}")
    return DATASETS_DIR / SUPPORTED_DATASETS[name]["dir"]


def processed_path(name: str) -> Path:
    return DATA_PROCESSED_DIR / SUPPORTED_DATASETS.get(name, {"dir": name})["dir"]


# ── 사전훈련 ONNX 모델 경로 (checkpoints/deployment-eligible/onnx-models/) ──
DINOV2_SMALL_ONNX = CHECKPOINTS_DIR / "deployment-eligible" / "onnx-models" / "dinov2-small.onnx"
MOBILENETV4_CONV_SMALL_ONNX = CHECKPOINTS_DIR / "deployment-eligible" / "onnx-models" / "mobilenetv4-conv-small.onnx"

# ── 캐시 레지스트리 ──
IDENTITY_REGISTRY_DB = CACHE_DIR / "registries" / "identity_registry.db"
BINDING_JSON = CACHE_DIR / "registries" / "binding.json"

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

# ── MiewID 모델 (비문/코주름, 공개 MIT) ──
MIEWID_NOSE_ONNX_PATH = MODELS_DIR / "miewid_nose.onnx"
MIEWID_MSV3_HF_REPO = "conservationxlabs/miewid-msv3"
