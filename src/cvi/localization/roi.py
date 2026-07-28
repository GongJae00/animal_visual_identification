"""ROI extraction utilities: crop generation from bbox predictions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from cvi.localization.types import DetectionBox


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
    x1, y1, x2, y2 = expand_bbox(
        bbox, scale=margin, image_width=image.width, image_height=image.height
    )
    cropped = image.crop((x1, y1, x2, y2))
    side = max(cropped.width, cropped.height)
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    padded.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    return padded.resize((target_size, target_size), Image.Resampling.BILINEAR)


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
    "expand_bbox",
    "face_roi_from_dog",
    "is_truncated",
    "square_padded_crop",
]
