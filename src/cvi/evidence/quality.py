from __future__ import annotations

import numpy as np
from PIL import Image


def estimate_blur(image: Image.Image) -> float:
    from scipy.ndimage import laplace
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size < 9:
        return 0.0
    lap_response = laplace(arr)
    variance = float(lap_response.var())
    return float(np.clip(variance / 100.0, 0.0, 1.0))


def estimate_brightness(image: Image.Image) -> float:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return float(np.mean(arr) / 255.0)


def estimate_contrast(image: Image.Image) -> float:
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    return float(np.std(arr) / 128.0)


def estimate_occlusion(image: Image.Image, face_box: tuple[int, int, int, int] | None = None) -> float:
    if face_box is None:
        return 0.0
    x0, y0, x1, y1 = face_box
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    face = arr[y0:y1, x0:x1]
    dark_ratio = float(np.mean(face < 30))
    return min(dark_ratio * 5.0, 1.0)


def overall_quality(image: Image.Image) -> float:
    b = estimate_blur(image)
    br = estimate_brightness(image)
    c = estimate_contrast(image)
    score = 0.5 * b + 0.25 * (1.0 - abs(br - 0.5) * 2) + 0.25 * min(c, 1.0)
    return float(np.clip(score, 0.0, 1.0))
