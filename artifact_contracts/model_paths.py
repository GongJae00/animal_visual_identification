from __future__ import annotations

import os
from pathlib import Path

# Data root
# Set CANINE_IDENTITY_DATA_DIR to an absolute local data root.
DATA_DIR = Path(
    os.environ.get(
        "CANINE_IDENTITY_DATA_DIR",
        Path.home() / "canine_identity_data",
    )
)

# Dataset and artifact subdirectories.
DATASETS_DIR = DATA_DIR / "datasets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = DATA_DIR / "cache"
RECEIPTS_DIR = DATA_DIR / "receipts"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
DOWNLOADS_DIR = DATA_DIR / "downloads"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Optional legacy data-layout directories
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_REGISTRY_DIR = DATA_DIR / "registry"

# Model cache root
MODELS_DIR = Path(
    os.environ.get(
        "CANINE_IDENTITY_MODELS_DIR",
        Path.home() / ".cache" / "canine_identity" / "models",
    )
)

# Dataset path metadata. Automatic acquisition is not admitted.
SUPPORTED_DATASETS: dict[str, dict] = {
    "ap10k-dog": {
        "name": "AP-10K domestic dog subset",
        "dir": "ap10k",
        "url": "https://github.com/AlexTheBad/AP-10K",
        "desc": "External dataset; automatic acquisition is disabled",
    },
    "dogflw": {
        "name": "Dog Facial Landmarks in the Wild",
        "dir": "dogflw",
        "url": "https://www.kaggle.com/datasets/georgemartvel/dogflw",
        "desc": "External dataset; automatic acquisition is disabled",
    },
    "yt-bb-dog": {
        "name": "YouTube-BoundingBoxes Dog",
        "dir": "yt-bb-dog",
        "url": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "desc": "External dataset; automatic acquisition is disabled",
    },
    "dogfacenet": {
        "name": "DogFaceNet",
        "dir": "dogfacenet224",
        "desc": "External dataset; automatic acquisition is disabled",
    },
    "mpdd": {
        "name": "Multi-pose dog dataset",
        "dir": "mpdd",
        "url": "https://data.mendeley.com/datasets/v5j6m8dzhv/1",
        "desc": "External dataset; automatic acquisition is disabled",
    },
    "sibetan": {
        "name": "SiBeTan",
        "dir": "sibetan",
        "url": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "desc": "External dataset; automatic acquisition is disabled",
    },
}


def dataset_path(name: str) -> Path:
    if name not in SUPPORTED_DATASETS:
        raise KeyError(
            f"Unknown dataset: {name}. Supported: {list(SUPPORTED_DATASETS)}"
        )
    return DATASETS_DIR / SUPPORTED_DATASETS[name]["dir"]


def processed_path(name: str) -> Path:
    return DATA_PROCESSED_DIR / SUPPORTED_DATASETS.get(name, {"dir": name})["dir"]


# Registry cache paths
IDENTITY_REGISTRY_DB = CACHE_DIR / "registries" / "identity_registry.db"
BINDING_JSON = CACHE_DIR / "registries" / "binding.json"

# Disabled DogFLW candidate
DOGFLW_LANDMARK_PATH = MODELS_DIR / "dogflw_landmark_full.tflite"
# No publisher-authoritative checksum or redistributable model source is verified.
DOGFLW_LANDMARK_MD5: str | None = None

SUPERANIMAL_HF_REPO = "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped"
SUPERANIMAL_REVISION = "1ad30fb80cd666f1e5c91578d1cf63bccfa84f80"
SUPERANIMAL_WEIGHTS_SHA256 = (
    "a42e58d56bb32f8b4f4fe17ea9ed9511cebc7b7949ac56b97fa6f3a49587c31e"
)
SUPERANIMAL_QUADRUPED_URL = (
    f"https://huggingface.co/{SUPERANIMAL_HF_REPO}/resolve/"
    f"{SUPERANIMAL_REVISION}/superanimal_quadruped_hrnet_w32.pt"
)
SUPERANIMAL_QUADRUPED_PATH = MODELS_DIR / "superanimal_quadruped_hrnet_w32.pt"
SUPERANIMAL_ONNX_PATH = MODELS_DIR / "superanimal_quadruped.onnx"

# MiewID-msv3 wildlife ReID candidate (not a nose-print model)
MIEWID_REID_ONNX_PATH = MODELS_DIR / "miewid_msv3_reid.onnx"
# Legacy artifact path retained for discovery only; it is not canonical.
MIEWID_MSV3_HF_REPO = "conservationxlabs/miewid-msv3"
MIEWID_MSV3_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"
MIEWID_MSV3_WEIGHTS_SHA256 = (
    "adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348"
)
