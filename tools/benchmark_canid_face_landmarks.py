"""Acquisition instructions for Phase 3 model zoo candidates.

This file records exact URLs, commit hashes, command lines, and expected
checksums for every model that the CVI benchmark should evaluate.  It is
not an automated downloader — it is the authoritative specification that
any person or script must follow to reproduce the model zoo.
"""

from __future__ import annotations

import json
from pathlib import Path


ACQUISITION_REGISTRY = {
    # ── Dog Detection & Pose ──────────────────────────────────────────
    "ultralytics-yolo11-pose": {
        "family": "YOLO11-Pose",
        "role": "dog_detector_pose",
        "variants": ("yolo11n-pose", "yolo11s-pose"),
        "source": "pip install ultralytics; models download automatically",
        "license": "AGPL-3.0",
        "notes": "No dog-specific fine-tuning exists. Requires dog annotation data "
                 "(Dog-Pose or StanfordExtra) for validation.",
        "blocker": "DOG_POSE_ANNOTATIONS_REQUIRED",
        "acquisition_command": (
            "uv run python -c 'from ultralytics import YOLO; "
            "YOLO(\"yolo11n-pose.pt\"); YOLO(\"yolo11s-pose.pt\")'"
        ),
    },
    "rtmdet-rtmpose": {
        "family": "RTMDet + RTMPose",
        "role": "dog_detector_pose",
        "variants": ("rtmdet-m", "rtmpose-m"),
        "source": "https://github.com/open-mmlab/mmpose (mmpose v1.x)",
        "license": "Apache-2.0",
        "notes": "OpenMMLab mmpose. Animal checkpoints exist (AP-10K). "
                 "Requires mmpose + mmdet installation.",
        "blocker": "DOG_POSE_ANNOTATIONS_REQUIRED",
        "acquisition_command": (
            "pip install mmpose mmdet mmengine; "
            "mim download mmpose --config rtmpose-m_8xb256-420e_ap10k-256x256 "
            "--dest checkpoints/rtmpose/"
        ),
    },
    "superanimal-quadruped": {
        "family": "SuperAnimal",
        "role": "dog_pose",
        "variants": ("superanimal_quadruped_hrnetw32",),
        "source": "https://github.com/DeepLabCut/SuperAnimal",
        "license": "Academic non-commercial",
        "notes": "Official quadruped pose model. Zero-shot evaluation possible. "
                 "Fine-tuning requires DLC project setup and dog annotations. "
                 "License is ACADEMIC_ONLY — cannot deploy.",
        "blocker": "ACADEMIC_LICENSE_ONLY",
        "acquisition_command": "dlc.SuperAnimal.convert2deeplabcut('superanimal_quadruped_hrnetw32')",
    },
    # ── Open-Vocabulary Teacher (smoke only) ─────────────────────────
    "grounding-dino": {
        "family": "Grounding DINO",
        "role": "teacher_proposal",
        "variants": ("GroundingDINO-SwinT",),
        "source": "https://github.com/IDEA-Research/GroundingDINO",
        "license": "Apache-2.0",
        "notes": "Text-prompted open-vocabulary detector. Use prompt 'dog' or "
                 "'domestic dog' for proposal generation. Not a runtime candidate.",
        "blocker": "TEACHER_ONLY_NOT_RUNTIME",
        "acquisition_command": (
            "git clone https://github.com/IDEA-Research/GroundingDINO; "
            "wget https://github.com/IDEA-Research/GroundingDINO/releases/download/..."
        ),
    },
    # ── Face Landmarks ───────────────────────────────────────────────
    "rtmpose-m-face46": {
        "family": "RTMPose-M",
        "role": "face_landmark",
        "variants": ("rtmpose-m_face46",),
        "source": "https://github.com/open-mmlab/mmpose (DogFLW face46 config)",
        "license": "Apache-2.0",
        "notes": "Requires DogFLW dataset for training. 46-point face configuration. "
                 "DogFLW download is currently disabled (no authoritative hash).",
        "blocker": "DOGFLW_DATASET_REQUIRED",
        "acquisition_command": "mim download mmpose --config rtmpose-m_dogflw-face46",
    },
    "hrnet-w32-face46": {
        "family": "HRNet-W32",
        "role": "face_landmark",
        "variants": ("hrnet_w32_dogflw",),
        "source": "https://github.com/open-mmlab/mmpose",
        "license": "Apache-2.0",
        "notes": "CVI already has HRNet ONNX infrastructure (LandmarkEvidencer). "
                 "Extend to DogFLW 46-point schema.",
        "blocker": "DOGFLW_DATASET_REQUIRED",
        "acquisition_command": None,
    },
    "vitpose-s-face46": {
        "family": "ViTPose-S",
        "role": "face_landmark",
        "variants": ("vitpose-s_dogflw",),
        "source": "https://github.com/ViTAE-Transformer/ViTPose",
        "license": "Apache-2.0",
        "notes": "Current SOTA for animal pose. Requires adaptation to 46-point schema.",
        "blocker": "DOGFLW_DATASET_REQUIRED",
        "acquisition_command": None,
    },
    # ── Segmentation ─────────────────────────────────────────────────
    "segformer-b2-dog": {
        "family": "SegFormer-B2",
        "role": "foreground_segmentation",
        "variants": ("segformer-b2-dog",),
        "source": "pip install transformers; HuggingFace nvidia/mit-b2",
        "license": "Apache-2.0",
        "notes": "Fine-tune on StanfordExtra dog masks or Oxford-IIIT Pet trimap. "
                 "DeepLabV3-ResNet50-COCO already acquired for foreground diagnosis.",
        "blocker": "SEGMENTATION_ANNOTATIONS_REQUIRED",
        "acquisition_command": None,
    },
}

EXISTING_CHECKPOINTS: dict[str, Path] = {
    "mobilenetv4-conv-small": Path(
        "/mnt/r/research-data/canine_video_identity_secure/checkpoints/"
        "deployment-eligible/hf-timm-mobilenetv4-conv-small-"
        "c9f31ac64483d7f0590db9edccb4418392a96eea"
    ),
    "dinov2-small": Path(
        "/mnt/r/research-data/canine_video_identity_secure/checkpoints/"
        "deployment-eligible/hf-facebook-dinov2-small-"
        "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    ),
    "deeplabv3-resnet50-coco": Path(
        "/mnt/r/research-data/canine_video_identity_secure/checkpoints/"
        "research-only/torchvision-deeplabv3-resnet50-coco-voc-v1/"
    ),
}


def main() -> None:
    print(json.dumps(
        {
            name: {
                "family": entry["family"],
                "role": entry["role"],
                "blocker": entry.get("blocker", "NONE"),
                "license": entry["license"],
            }
            for name, entry in sorted(ACQUISITION_REGISTRY.items())
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
