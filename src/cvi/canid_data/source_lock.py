"""Immutable source registry for all canid datasets.

Admission decisions are versioned and must not be silently changed.
"""

from __future__ import annotations

from cvi.canid_data.types import (
    CanidDatasetRecord,
    CaptureGroupKind,
    DatasetAdmission,
)

_DATA_ROOT = "/mnt/r/research-data/canine_video_identity_secure/datasets"

SOURCE_REGISTRY: tuple[CanidDatasetRecord, ...] = (
    CanidDatasetRecord(
        canonical_name="dogfacenet224",
        official_name="DogFaceNet 224 (resized)",
        version="zenodo-12578449-v1",
        license_id="CC-BY-4.0",
        url="https://zenodo.org/records/12578449",
        data_root=f"{_DATA_ROOT}/dogfacenet224",
        sha256_checksums={
            "DogFaceNet_224resized.zip": (
                "ee37ec14c661a9dba49f4c45dd5d77618bcb0b8678af7a96a7ca7b75b79ba510"
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
        official_name="MPDD (Mendeley Pet Dog Dataset)",
        version="mendeley-v5j6m8dzhv-v1",
        license_id="CC-BY-4.0",
        url="https://data.mendeley.com/datasets/v5j6m8dzhv/1",
        data_root=f"{_DATA_ROOT}/mpdd",
        sha256_checksums={},
        total_images=1657,
        total_identities=191,
        capture_group_kind=CaptureGroupKind.REAL_CAMERA_SESSION,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.ADMIT_VALIDATION_ONLY,
        notes="Small multi-session dataset with explicit camera/sequence IDs. "
              "45 identities, suitable for open-set calibration or DEV only.",
    ),
    CanidDatasetRecord(
        canonical_name="sibetan",
        official_name="Sibetan",
        version="official-2026-07-22",
        license_id="CC-BY-NC-4.0",
        url=None,
        data_root=f"{_DATA_ROOT}/sibetan",
        sha256_checksums={},
        total_images=1755,
        total_identities=223,
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=False,
        has_nose_mask=False,
        admission=DatasetAdmission.BLOCKED_LICENSE,
        notes="Non-commercial license (CC-BY-NC-4.0). Cluster-based identity "
              "from field camera trapping. Cannot be used for deployment. "
              "May be ADMIT_TEACHER_ONLY after legal review.",
    ),
    CanidDatasetRecord(
        canonical_name="yt-bb-dog",
        official_name="YT-BB-Dog",
        version="outer-official-2026-07-22",
        license_id="CC-BY-4.0",
        url=None,
        data_root=f"{_DATA_ROOT}/yt-bb-dog",
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
    CanidDatasetRecord(
        canonical_name="oxford-pets-dog",
        official_name="Oxford-IIIT Pet (dog subset)",
        version="unacquired",
        license_id="CC-BY-SA-4.0",
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/",
        data_root="",
        sha256_checksums={},
        total_images=0,
        total_identities=0,
        capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        has_dog_bbox=False,
        has_face_bbox=False,
        has_face_landmarks=False,
        has_body_keypoints=False,
        has_breed=True,
        has_nose_mask=False,
        admission=DatasetAdmission.BLOCKED_ACCESS,
        notes="Not yet acquired. 25 dog breeds with trimap segmentation. "
              "No per-instance identity labels. Useful for breed classification.",
    ),
)


def get_record(canonical_name: str) -> CanidDatasetRecord:
    for record in SOURCE_REGISTRY:
        if record.canonical_name == canonical_name:
            return record
    raise KeyError(f"unknown canid dataset: {canonical_name!r}")


def admitted_records() -> tuple[CanidDatasetRecord, ...]:
    return tuple(
        record for record in SOURCE_REGISTRY
        if record.admission in (
            DatasetAdmission.ADMIT_TRAIN,
            DatasetAdmission.ADMIT_VALIDATION_ONLY,
            DatasetAdmission.ADMIT_TEACHER_ONLY,
        )
    )


__all__ = ["SOURCE_REGISTRY", "get_record", "admitted_records"]
