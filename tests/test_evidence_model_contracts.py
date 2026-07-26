from __future__ import annotations

import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from cvi.backbones import get_backbone
from cvi.evidence.landmark_graph import LandmarkEvidencer
from cvi.evidence.miewid import (
    MIEWID_OUTPUT_DIM,
    MiewIDModelContractError,
    MiewIDReIDExtractor,
)
from cvi.evidence.nose_print import (
    DNPMask,
    MiewIDNoseExtractor,
    YoloNoseDetector,
)


class _TensorInfo:
    def __init__(self, name: str, shape: list[object]):
        self.name = name
        self.shape = shape


class _FakeSession:
    input_shape: list[object] = ["batch", 3, 440, 440]
    last_batch: np.ndarray | None = None

    def __init__(self, path: str):
        self.path = path

    def get_inputs(self):
        return [_TensorInfo("pixel_values", self.input_shape)]

    def get_outputs(self):
        return [_TensorInfo("embedding", ["batch", MIEWID_OUTPUT_DIM])]

    def run(self, output_names, feeds):
        type(self).last_batch = feeds["pixel_values"]
        return [np.ones((1, MIEWID_OUTPUT_DIM), dtype=np.float32)]


class EvidenceModelContractTests(unittest.TestCase):
    def _artifact(self) -> tempfile.NamedTemporaryFile:
        return tempfile.NamedTemporaryFile(suffix=".onnx")

    def test_miewid_enforces_official_preprocessing_and_dimension(self) -> None:
        fake_ort = types.SimpleNamespace(InferenceSession=_FakeSession)
        with self._artifact() as artifact, patch.dict(
            sys.modules, {"onnxruntime": fake_ort}
        ):
            extractor = MiewIDReIDExtractor(Path(artifact.name))
            embedding = extractor.extract(
                Image.fromarray(np.full((20, 30, 3), 255, dtype=np.uint8))
            )
        self.assertEqual(_FakeSession.last_batch.shape, (1, 3, 440, 440))
        self.assertAlmostEqual(
            float(_FakeSession.last_batch[0, 0, 0, 0]),
            (1.0 - 0.485) / 0.229,
            places=5,
        )
        self.assertEqual(embedding.shape, (MIEWID_OUTPUT_DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_miewid_rejects_wrong_spatial_contract(self) -> None:
        class WrongShapeSession(_FakeSession):
            input_shape = ["batch", 3, 160, 160]

        fake_ort = types.SimpleNamespace(InferenceSession=WrongShapeSession)
        with self._artifact() as artifact, patch.dict(
            sys.modules, {"onnxruntime": fake_ort}
        ):
            with self.assertRaises(MiewIDModelContractError):
                MiewIDReIDExtractor(Path(artifact.name))

    def test_deprecated_nose_alias_rejects_160(self) -> None:
        with self._artifact() as artifact, warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(ValueError):
                MiewIDNoseExtractor(Path(artifact.name), input_size=160)

    def test_untrained_channels_are_not_runtime_defaults(self) -> None:
        with self.assertRaises(KeyError):
            get_backbone("tinyvit")
        with self.assertRaises(RuntimeError):
            LandmarkEvidencer()

    def test_missing_nose_models_do_not_fabricate_evidence(self) -> None:
        image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        self.assertTrue(np.array_equal(DNPMask().apply(image), image))
        self.assertIsNone(
            YoloNoseDetector().detect(Image.fromarray(image))
        )


if __name__ == "__main__":
    unittest.main()
