"""MiewID-msv3 wildlife re-identification inference contract.

MiewID is a general wildlife ReID feature extractor, not a canine nose-print
biometric.  This module enforces the published model-card preprocessing and
the exported ONNX tensor contract before any image is processed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cvi.evidence.base import AbstractEvidencer

MIEWID_IMAGE_SIZE = 440
MIEWID_OUTPUT_DIM = 2152
MIEWID_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
MIEWID_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class MiewIDModelContractError(RuntimeError):
    """Raised when an artifact does not match the pinned MiewID contract."""


class MiewIDReIDExtractor(AbstractEvidencer):
    """Pinned MiewID-msv3 whole-crop wildlife ReID extractor."""

    name = "wildlife_reid"
    output_dim = MIEWID_OUTPUT_DIM

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"MiewID ONNX model not found: {onnx_path}\n"
                "  Download the pinned export with: "
                "python tools/download_models.py --model miewid"
            )
        self._sess = ort.InferenceSession(str(onnx_path))
        model_input = self._sess.get_inputs()[0]
        model_output = self._sess.get_outputs()[0]
        self._input_name = model_input.name
        input_shape = model_input.shape
        output_shape = model_output.shape
        if (
            len(input_shape) != 4
            or input_shape[1] != 3
            or input_shape[2] != MIEWID_IMAGE_SIZE
            or input_shape[3] != MIEWID_IMAGE_SIZE
        ):
            raise MiewIDModelContractError(
                "MiewID input must be [batch, 3, 440, 440], "
                f"got {input_shape!r}"
            )
        if len(output_shape) != 2 or output_shape[1] != MIEWID_OUTPUT_DIM:
            raise MiewIDModelContractError(
                "MiewID output must be [batch, 2152], "
                f"got {output_shape!r}"
            )

    @staticmethod
    def _preprocess(image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB").resize(
            (MIEWID_IMAGE_SIZE, MIEWID_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        arr = (arr - MIEWID_MEAN) / MIEWID_STD
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])

    def extract(self, image: Image.Image) -> np.ndarray:
        batch = self._preprocess(image)
        output = np.asarray(
            self._sess.run(None, {self._input_name: batch})[0],
            dtype=np.float32,
        )
        if output.shape != (1, MIEWID_OUTPUT_DIM):
            raise MiewIDModelContractError(
                f"MiewID runtime output must be (1, 2152), got {output.shape}"
            )
        embedding = output[0]
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise MiewIDModelContractError(
                "MiewID produced a non-finite or zero-norm embedding"
            )
        return embedding / norm

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([self.extract(image) for image in images])
