"""Label mapping utilities — preserve original keys, expose alias layer."""

from __future__ import annotations

CANID_KEYPOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "nose_center": ("nose", "nose_tip", "nose_center"),
    "chin": ("chin",),
    "left_eye": ("left_eye", "leye", "eye_left"),
    "right_eye": ("right_eye", "reye", "eye_right"),
    "left_ear_base": ("left_ear_base", "lear_base", "ear_base_left"),
    "right_ear_base": ("right_ear_base", "rear_base", "ear_base_right"),
    "left_ear_tip": ("left_ear_tip", "lear_tip", "ear_tip_left"),
    "right_ear_tip": ("right_ear_tip", "rear_tip", "ear_tip_right"),
    "throat": ("throat",),
    "neck": ("neck", "withers"),
    "tail_base": ("tail_base",),
    "tail_tip": ("tail_tip",),
    "left_shoulder": ("left_shoulder", "shoulder_left"),
    "right_shoulder": ("right_shoulder", "shoulder_right"),
    "left_elbow": ("left_elbow", "elbow_left"),
    "right_elbow": ("right_elbow", "elbow_right"),
    "left_wrist": ("left_wrist", "wrist_left", "carpus_left"),
    "right_wrist": ("right_wrist", "wrist_right", "carpus_right"),
    "left_hip": ("left_hip", "hip_left"),
    "right_hip": ("right_hip", "hip_right"),
    "left_knee": ("left_knee", "knee_left", "stifle_left"),
    "right_knee": ("right_knee", "knee_right", "stifle_right"),
    "left_ankle": ("left_ankle", "ankle_left", "hock_left"),
    "right_ankle": ("right_ankle", "ankle_right", "hock_right"),
}

CANID_BREED_ALIASES: dict[str, tuple[str, ...]] = {}

CANID_SPECIES: tuple[str, ...] = (
    "Canis lupus familiaris",
    "Canis lupus",
    "Canis latrans",
    "Canis aureus",
    "Vulpes vulpes",
    "Vulpes lagopus",
)


def resolve_keypoint_name(name: str) -> str | None:
    """Map any known alias to the canonical canine keypoint name."""
    lowered = name.lower().replace(" ", "_").replace("-", "_")
    for canonical, aliases in CANID_KEYPOINT_ALIASES.items():
        if lowered == canonical.lower() or lowered in {a.lower() for a in aliases}:
            return canonical
    return None


def is_known_canid_species(species: str) -> bool:
    return species in CANID_SPECIES


__all__ = [
    "CANID_BREED_ALIASES",
    "CANID_KEYPOINT_ALIASES",
    "CANID_SPECIES",
    "is_known_canid_species",
    "resolve_keypoint_name",
]
