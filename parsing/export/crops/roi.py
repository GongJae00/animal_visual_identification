"""ROI extraction utilities: crop generation from bbox predictions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from parsing.export.types import DetectionBox, KeypointSet


def compute_iou(first: DetectionBox, second: DetectionBox) -> float:
    """Compute intersection over union for two boxes."""

    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / max(union, 1e-6)


def expand_bbox(
    bbox: DetectionBox,
    scale: float = 1.15,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    half_size = max(bbox.width, bbox.height) * scale / 2.0
    x1 = max(0.0, center_x - half_size)
    y1 = max(0.0, center_y - half_size)
    x2 = min(float(image_width), center_x + half_size)
    y2 = min(float(image_height), center_y + half_size)
    return (int(x1), int(y1), int(x2), int(y2))


def square_padded_crop(
    image: Image.Image,
    bbox: DetectionBox,
    *,
    margin: float = 1.15,
    target_size: int = 224,
) -> Image.Image:
    padded, _, _ = square_padded_crop_with_mask(
        image, bbox, margin=margin, target_size=target_size
    )
    return padded


def square_padded_crop_with_mask(
    image: Image.Image,
    bbox: DetectionBox,
    *,
    margin: float = 1.15,
    target_size: int = 224,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = expand_bbox(
        bbox, scale=margin, image_width=image.width, image_height=image.height
    )
    cropped = image.crop((x1, y1, x2, y2))
    side = max(cropped.width, cropped.height)
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    mask = Image.new("L", (side, side), 0)
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    padded.paste(cropped, offset)
    mask.paste(255, (*offset, offset[0] + cropped.width, offset[1] + cropped.height))
    return (
        padded.resize((target_size, target_size), Image.Resampling.BILINEAR),
        mask.resize((target_size, target_size), Image.Resampling.NEAREST),
        (x1, y1, x2, y2),
    )


def normalize_source_point_to_square_crop(
    x: float,
    y: float,
    crop_rect_xyxy: tuple[int, int, int, int] | list[int],
) -> tuple[float, float]:
    """Map a source-image point into the final square crop's normalized space."""

    x1, y1, x2, y2 = crop_rect_xyxy
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("crop rectangle must be non-empty")
    side = max(width, height)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    normalized_x = (x - x1 + offset_x) / side
    normalized_y = (y - y1 + offset_y) / side
    return (
        min(1.0, max(0.0, normalized_x)),
        min(1.0, max(0.0, normalized_y)),
    )


def face_roi_from_dog(
    dog_bbox: DetectionBox,
    *,
    ratio_height: float = 0.55,
    ratio_width: float = 0.40,
    vertical_shift: float = -0.05,
) -> DetectionBox:
    width = dog_bbox.width * ratio_width
    height = dog_bbox.height * ratio_height
    center_x = (dog_bbox.x1 + dog_bbox.x2) / 2.0
    center_y = dog_bbox.y1 + dog_bbox.height * (0.35 + vertical_shift)
    return DetectionBox(
        x1=center_x - width / 2.0,
        y1=center_y - height / 2.0,
        x2=center_x + width / 2.0,
        y2=center_y + height / 2.0,
        confidence=dog_bbox.confidence,
        class_name="face",
    )


def face_and_weak_nose_rois_from_pose(
    landmarks: KeypointSet,
    dog_bbox: DetectionBox,
    *,
    image_width: int,
    image_height: int,
) -> tuple[DetectionBox | None, DetectionBox | None]:
    """Derive weak face/nose ROIs from real AP-10K eye and nose anchors."""

    nose = landmarks.named("nose_center")
    left_eye = landmarks.named("left_eye")
    right_eye = landmarks.named("right_eye")
    if nose is None or nose.confidence < 0.1:
        return None, None
    visible_eyes = [
        eye
        for eye in (left_eye, right_eye)
        if eye is not None and eye.confidence >= 0.1
    ]
    if not visible_eyes:
        return None, None
    if len(visible_eyes) == 2:
        eye_mid_x = (visible_eyes[0].x + visible_eyes[1].x) / 2.0
        eye_mid_y = (visible_eyes[0].y + visible_eyes[1].y) / 2.0
        eye_span = float(
            np.hypot(
                visible_eyes[0].x - visible_eyes[1].x,
                visible_eyes[0].y - visible_eyes[1].y,
            )
        )
    else:
        eye_mid_x, eye_mid_y = visible_eyes[0].x, visible_eyes[0].y
        eye_span = 2.0 * float(np.hypot(nose.x - eye_mid_x, nose.y - eye_mid_y))
    scale = max(eye_span, 0.12 * max(dog_bbox.width, dog_bbox.height), 4.0)

    def clipped_box(
        center_x: float, center_y: float, width: float, height: float, class_name: str
    ) -> DetectionBox | None:
        x1 = max(0.0, center_x - width / 2.0)
        y1 = max(0.0, center_y - height / 2.0)
        x2 = min(float(image_width), center_x + width / 2.0)
        y2 = min(float(image_height), center_y + height / 2.0)
        if x2 <= x1 or y2 <= y1:
            return None
        confidence = float(
            np.mean([nose.confidence, *(eye.confidence for eye in visible_eyes)])
        )
        return DetectionBox(x1, y1, x2, y2, confidence, class_name=class_name)

    face_center_x = (eye_mid_x + nose.x) / 2.0
    face_center_y = eye_mid_y + 0.65 * (nose.y - eye_mid_y)
    face = clipped_box(face_center_x, face_center_y, 3.2 * scale, 3.0 * scale, "face")
    weak_nose = clipped_box(nose.x, nose.y, 1.5 * scale, 1.1 * scale, "nose")
    return face, weak_nose


def is_truncated(
    bbox: DetectionBox,
    *,
    image_width: int,
    image_height: int,
    threshold: float = 0.20,
) -> bool:
    if not bbox.width or not bbox.height:
        return True
    visible_x1 = max(0.0, bbox.x1)
    visible_y1 = max(0.0, bbox.y1)
    visible_x2 = min(float(image_width), bbox.x2)
    visible_y2 = min(float(image_height), bbox.y2)
    visible_area = max(0.0, visible_x2 - visible_x1) * max(0.0, visible_y2 - visible_y1)
    return 1.0 - visible_area / bbox.area > threshold


__all__ = [
    "compute_iou",
    "expand_bbox",
    "face_roi_from_dog",
    "face_and_weak_nose_rois_from_pose",
    "is_truncated",
    "normalize_source_point_to_square_crop",
    "square_padded_crop",
    "square_padded_crop_with_mask",
]
