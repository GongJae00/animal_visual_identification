"""Immutable source registry for all canid datasets.

Admission decisions are versioned and must not be silently changed.
"""

from __future__ import annotations

from artifact_contracts.model_paths import DATASETS_DIR
from data_pipeline.types import (
    CanidDatasetRecord,
    CaptureGroupKind,
    DatasetAdmission,
)


def _dataset_root(directory: str) -> str:
    return str(DATASETS_DIR / directory)


SOURCE_REGISTRY: tuple[CanidDatasetRecord, ...] = (
    CanidDatasetRecord(
        canonical_name="ap10k-dog",
        official_name="AP-10K domestic dog subset",
        version="official-split1-2021-11-01",
        license_id="CC-BY-4.0",
        url="https://github.com/AlexTheBad/AP-10K",
        data_root=_dataset_root("ap10k"),
        sha256_checksums={
            "ap-10k.zip": "420980abb135d6f66bcc8e29f289a46081214016192ae197ad24bc1525c8e62c",
        },
        total_images=1000,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.UNKNOWN,
        has_dog_bbox=True,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=True,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_TRAIN,
        notes="Domestic-dog category only: 1,129 annotated instances across "
        "1,000 unique images using official split 1.",
    ),
    CanidDatasetRecord(
        canonical_name="dogflw",
        official_name="Dog Facial Landmarks in the Wild",
        version="kaggle-2025-07-02",
        license_id="CC-BY-NC-4.0",
        url="https://www.kaggle.com/datasets/georgemartvel/dogflw",
        data_root=_dataset_root("dogflw"),
        sha256_checksums={
            "dogflw.zip": "20ad33deae314dd529385a7065f19e71c38c847922413ee2120589cdc30da659",
        },
        total_images=4335,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        has_dog_bbox=False,
        has_face_bbox=True,
        has_face_landmarks=True,
        has_body_keypoints=False,
        has_breed=True,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_TEACHER_ONLY,
        notes="Publisher train/test face crops with one face bbox and 46 landmarks; "
        "non-commercial research use only and no dog identity labels.",
    ),
    CanidDatasetRecord(
        canonical_name="dogfacenet224",
        official_name="DogFaceNet 224 (resized)",
        version="zenodo-12578449-v1",
        license_id="CC-BY-4.0",
        url="https://zenodo.org/records/12578449",
        data_root=_dataset_root("dogfacenet224"),
        sha256_checksums={
            "DogFaceNet_224resized.zip": (
                "b3b335180bfd8d18b17e13601c9b0fa9c7c92bf9c18d64fe2999597f2e71f871"
            ),
        },
        total_images=8363,
        total_identities=1393,
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_TRAIN,
        notes="Per-identity web-album crops; no bbox, keypoint, or breed labels. "
        "Useful for appearance/face ReID identity training.",
    ),
    CanidDatasetRecord(
        canonical_name="mpdd",
        official_name="Multi-pose dog dataset",
        version="mendeley-v5j6m8dzhv-v1",
        license_id="CC-BY-4.0",
        url="https://data.mendeley.com/datasets/v5j6m8dzhv/1",
        data_root=_dataset_root("mpdd"),
        sha256_checksums={},
        total_images=1657,
        total_identities=191,
        capture_group_kind=CaptureGroupKind.POSE_VIEW_CLUSTER,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_VALIDATION_ONLY,
        notes="Audited archive has 1,657 images of 191 dogs. Filename cNsN/cN_sN "
        "camera and sequence fields remain unverified; retained for validation only.",
    ),
    CanidDatasetRecord(
        canonical_name="sibetan",
        official_name="Sibetan",
        version="publisher-v1-2025-10-27",
        license_id="CC-BY-4.0",
        url="https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        data_root=_dataset_root("sibetan"),
        sha256_checksums={},
        total_images=1755,
        total_identities=59,
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_VALIDATION_ONLY,
        notes="Publisher reports 1,755 images of 59 dogs from one week of "
        "cross-camera trap recordings; use as a validation cohort.",
    ),
    CanidDatasetRecord(
        canonical_name="yt-bb-dog",
        official_name="YT-BB-Dog",
        version="publisher-v1-2025-10-27",
        license_id="CC-BY-4.0",
        url="https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        data_root=_dataset_root("yt-bb-dog"),
        sha256_checksums={
            "YT-BB-Dog.zip": (
                "36b368d9d945137ece5e4d5f4d8208362f7a18bb8fa77bf28c5ea2857034b526"
            ),
        },
        total_images=27036,
        total_identities=2723,
        capture_group_kind=CaptureGroupKind.VIDEO_TRACK,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_TRAIN,
        notes="Largest dataset. Video-track identity labels (not verified "
        "lifelong dog identity). train=19,932, test=7,104 images. "
        "Identities are video-track, not cross-video verified.",
    ),
    CanidDatasetRecord(
        canonical_name="oxford-pets-dog",
        official_name="Oxford-IIIT Pet (dog subset)",
        version="publisher-splits-v1",
        license_id="Research-only; original-source terms apply",
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/",
        data_root=_dataset_root("oxford-iiit-pet"),
        sha256_checksums={
            "annotations/README": (
                "e31ae5da0d657c614e055a08c2045c4d49f770f41361321f017ddf15df7ebcd6"
            ),
            "annotations/list.txt": (
                "6a54ab256e22f7a33c6f17a7669e58ea5f6f9c7a080ec2622c205aefd4b354da"
            ),
            "annotations/trainval.txt": (
                "408f3f609481b939c94634169e6413414b733a3faeba440cbdcc5c02142eebdc"
            ),
            "annotations/test.txt": (
                "a5454003774ffe01f4f322756d3ba5495bae21cb30bb217ab285dbfa2bef245c"
            ),
        },
        total_images=4978,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.UNKNOWN,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=True,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_TEACHER_ONLY,
        notes="Publisher trainval/test dog subset with trimaps and optional XML head "
        "ROIs. No per-animal identity or capture grouping; research purposes only.",
    ),
    CanidDatasetRecord(
        canonical_name="petface-dog",
        official_name="PetFace dog subset",
        version="eccv-2024-local-archive-intake-v1",
        license_id="PetFace research-only; no redistribution",
        url="https://dahlian00.github.io/PetFacePage/",
        data_root=_dataset_root("petface"),
        sha256_checksums={},
        total_images=0,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.BLOCKED_ACCESS,
        notes="Local archives contain publisher split metadata and a research-only, "
        "no-redistribution README. Source receipt and publisher-bound archive "
        "checksums are absent, so this is intake-only and not admitted.",
    ),
    # --- NOT YET ACQUIRED ---
    CanidDatasetRecord(
        canonical_name="dog-pose",
        official_name="Dog-Pose (Stanley et al.)",
        version="unacquired",
        license_id="Unknown",
        url="https://github.com/runa91/dog_pose_dataset_v1",
        data_root="",
        sha256_checksums={},
        total_images=0,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.UNKNOWN,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=True,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.BLOCKED_ACCESS,
        notes="Not yet acquired. Has 22 keypoints per dog. License unclear.",
    ),
    CanidDatasetRecord(
        canonical_name="stanford-extra",
        official_name="StanfordExtra (Biggs et al.)",
        version="unacquired",
        license_id="Unknown",
        url="https://github.com/benjiebob/StanfordExtra",
        data_root="",
        sha256_checksums={},
        total_images=0,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.UNKNOWN,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=True,
        has_breed=True,
        has_nose_mask=False,
        admission=DatasetAdmission.BLOCKED_ACCESS,
        notes="Not yet acquired. 12,000 images with 2D keypoints and breed "
        "labels from Stanford Dogs dataset. Commercial use restricted.",
    ),
)


def get_record(canonical_name: str) -> CanidDatasetRecord:
    for record in SOURCE_REGISTRY:
        if record.canonical_name == canonical_name:
            return record
    raise KeyError(f"unknown canid dataset: {canonical_name!r}")


def admitted_records() -> tuple[CanidDatasetRecord, ...]:
    return tuple(
        record
        for record in SOURCE_REGISTRY
        if record.admission
        in (
            DatasetAdmission.ADMIT_TRAIN,
            DatasetAdmission.ADMIT_VALIDATION_ONLY,
            DatasetAdmission.ADMIT_TEACHER_ONLY,
        )
    )


__all__ = ["SOURCE_REGISTRY", "admitted_records", "get_record"]
