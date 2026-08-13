from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from contracts.artifact_manifest import (
    ArtifactContractError,
    ArtifactLicense,
    ClaheTransform,
    ImagePreprocessing,
    LandmarkGraphManifest,
    LandmarkGraphPreprocessing,
    LandmarkKeypointManifest,
    NoseDetectorManifest,
    NoseEmbeddingManifest,
    NoseMaskManifest,
    UsageLane,
)
from embedding.evidence.base import EvidenceInsufficiency, EvidenceUnavailableReason
from embedding.methods.nose.extractor import (
    DNPMask,
    NosePrintExtractor,
    NoseRoiPolicy,
    YoloNoseDetector,
)
from embedding.methods.landmark import (
    LandmarkEvidencer,
    LandmarkGraphEmbedder,
    decode_landmark_heatmaps,
)


class ExactArtifactContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.license = ArtifactLicense("LicenseRef-Test-Fixture", UsageLane.TEST_FIXTURE)
        self.preprocessing = ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bilinear",
            scale=1.0 / 255.0,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            clahe=None,
        )

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_manifests_bind_sha_static_shapes_license_and_keypoint_order(self) -> None:
        path = self.root / "keypoints.onnx"
        heatmaps = self._visible_heatmaps()
        self._write_constant_model(path, (1, 3, 8, 8), heatmaps)
        manifest = self._keypoint_manifest(path)

        self.assertEqual(manifest.output_shape, (1, 2, 2, 3))
        self.assertEqual(manifest.keypoint_order, ("left_eye", "right_eye"))
        self.assertIs(manifest.license.usage_lane, UsageLane.TEST_FIXTURE)
        with self.assertRaisesRegex(ArtifactContractError, "lowercase SHA256"):
            LandmarkKeypointManifest(
                artifact_id="bad",
                artifact_sha256="0" * 63,
                input_name="input",
                input_shape=(1, 3, 8, 8),
                output_name="output",
                output_shape=(1, 2, 2, 3),
                license=self.license,
                preprocessing=self.preprocessing,
                keypoint_order=("left_eye", "right_eye"),
                visibility_threshold=0.5,
                min_visible_keypoints=2,
            )
        with self.assertRaisesRegex(ArtifactContractError, "channel count"):
            LandmarkKeypointManifest(
                artifact_id="bad-schema",
                artifact_sha256="0" * 64,
                input_name="input",
                input_shape=(1, 3, 8, 8),
                output_name="output",
                output_shape=(1, 2, 2, 3),
                license=self.license,
                preprocessing=self.preprocessing,
                keypoint_order=("nose",),
                visibility_threshold=0.5,
                min_visible_keypoints=1,
            )

    def test_all_artifact_manifests_round_trip_with_exact_json_keys(self) -> None:
        keypoint_path = self.root / "keypoints.onnx"
        graph_path = self.root / "graph.onnx"
        detector_path = self.root / "detector.onnx"
        embedding_path = self.root / "embedding.onnx"
        mask_path = self.root / "mask.onnx"
        self._write_constant_model(
            keypoint_path, (1, 3, 8, 8), self._visible_heatmaps()
        )
        self._write_flatten_model(graph_path, (1, 2, 4))
        self._write_constant_model(
            detector_path,
            (1, 3, 8, 8),
            np.asarray([[[0.1, 0.1, 0.9, 0.9, 0.8]]], dtype=np.float32),
        )
        self._write_global_average_model(embedding_path, (1, 3, 8, 8))
        self._write_constant_model(
            mask_path,
            (1, 3, 8, 8),
            np.ones((1, 1, 8, 8), dtype=np.float32),
        )
        manifests = (
            self._keypoint_manifest(keypoint_path),
            self._graph_manifest(graph_path),
            self._detector_manifest(detector_path),
            self._embedding_manifest(embedding_path, clahe=True),
            self._mask_manifest(mask_path),
        )
        for manifest in manifests:
            with self.subTest(manifest=type(manifest).__name__):
                payload = manifest.to_dict()
                self.assertEqual(type(manifest).from_dict(payload), manifest)
                with self.assertRaisesRegex(ArtifactContractError, "keys mismatch"):
                    type(manifest).from_dict({**payload, "unknown": True})

    def test_exact_sha_is_checked_before_artifact_activation(self) -> None:
        expected = self.root / "expected.onnx"
        substituted = self.root / "substituted.onnx"
        self._write_constant_model(
            expected, (1, 3, 8, 8), np.asarray([[[0.1, 0.1, 0.9, 0.9, 0.8]]], dtype=np.float32)
        )
        self._write_constant_model(
            substituted, (1, 3, 8, 8), np.asarray([[[0.1, 0.1, 0.9, 0.9, 0.9]]], dtype=np.float32)
        )
        with self.assertRaisesRegex(ArtifactContractError, "SHA256"):
            YoloNoseDetector(substituted, self._detector_manifest(expected))

    def test_artifact_components_require_exact_contracts(self) -> None:
        with self.assertRaises(TypeError):
            DNPMask()
        with self.assertRaises(ArtifactContractError):
            LandmarkEvidencer(object(), object(), object(), object())

    def test_landmark_decoder_validates_shape_visibility_and_crop_normalization(self) -> None:
        path = self.root / "keypoints.onnx"
        heatmaps = self._visible_heatmaps()
        self._write_constant_model(path, (1, 3, 8, 8), heatmaps)
        manifest = self._keypoint_manifest(path)
        decoded = decode_landmark_heatmaps(heatmaps, manifest, (11, 21))

        np.testing.assert_allclose(
            decoded.pixel_points,
            np.asarray([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            decoded.normalized_points,
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(decoded.visible, np.asarray([True, True]))
        with self.assertRaisesRegex(ArtifactContractError, "must be float32"):
            decode_landmark_heatmaps(heatmaps.astype(np.float64), manifest, (11, 21))
        with self.assertRaises(EvidenceInsufficiency):
            low_confidence = heatmaps.copy()
            low_confidence[0, 1, 1, 2] = 0.4
            decode_landmark_heatmaps(low_confidence, manifest, (11, 21))

    def test_landmark_evidencer_requires_matching_schema_and_checkpoint_graph(self) -> None:
        keypoint_path = self.root / "keypoints.onnx"
        graph_path = self.root / "graph.onnx"
        heatmaps = self._visible_heatmaps()
        self._write_constant_model(keypoint_path, (1, 3, 8, 8), heatmaps)
        self._write_flatten_model(graph_path, (1, 2, 4))
        keypoint_manifest = self._keypoint_manifest(keypoint_path)
        graph_manifest = self._graph_manifest(graph_path)

        evidencer = LandmarkEvidencer(
            keypoint_path, keypoint_manifest, graph_path, graph_manifest
        )
        observation = evidencer.extract(Image.new("RGB", (11, 21), "white"))
        self.assertTrue(observation.is_available)
        embedding = observation.embedding
        expected = np.asarray(
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0], dtype=np.float32
        )
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(embedding, expected, atol=1e-6)

        mismatched = LandmarkGraphManifest(
            artifact_id="graph-reordered",
            artifact_sha256=sha256(graph_path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 2, 4),
            output_name="output",
            output_shape=(1, 8),
            license=self.license,
            keypoint_order=("right_eye", "left_eye"),
            preprocessing=self._graph_preprocessing(),
        )
        with self.assertRaisesRegex(ArtifactContractError, "same schema and order"):
            LandmarkEvidencer(
                keypoint_path, keypoint_manifest, graph_path, mismatched
            )
        decoded = decode_landmark_heatmaps(
            heatmaps, keypoint_manifest, (11, 21)
        )
        with self.assertRaisesRegex(ArtifactContractError, "schema order"):
            LandmarkGraphEmbedder(graph_path, mismatched).embed(decoded)

    def test_nose_detector_clips_boxes_and_composed_extractor_embeds_valid_roi(self) -> None:
        detector_path = self.root / "detector.onnx"
        embedding_path = self.root / "embedding.onnx"
        self._write_constant_model(
            detector_path,
            (1, 3, 8, 8),
            np.asarray([[[-0.1, 0.1, 1.2, 0.9, 0.9]]], dtype=np.float32),
        )
        self._write_global_average_model(embedding_path, (1, 3, 8, 8))
        detector_manifest = self._detector_manifest(detector_path)
        embedding_manifest = self._embedding_manifest(embedding_path)
        image = Image.new("RGB", (100, 100), (255, 128, 64))

        detection = YoloNoseDetector(detector_path, detector_manifest).detect(image)
        self.assertIsNotNone(detection)
        self.assertEqual(detection.box, (0, 10, 100, 90))
        extractor = NosePrintExtractor(
            detector_path,
            detector_manifest,
            embedding_path,
            embedding_manifest,
            NoseRoiPolicy(2, 2, 16, 16),
        )
        result = extractor.extract(image)
        self.assertFalse(result.abstained)
        self.assertEqual(result.roi_box, (0, 10, 100, 90))
        self.assertEqual(result.embedding.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=6)

    def test_nose_extractor_abstains_for_missing_tiny_and_low_resolution_roi(self) -> None:
        embedding_path = self.root / "embedding.onnx"
        self._write_global_average_model(embedding_path, (1, 3, 8, 8))
        embedding_manifest = self._embedding_manifest(embedding_path, clahe=True)
        cases = (
            (
                "missing",
                [0.1, 0.1, 0.9, 0.9, 0.1],
                EvidenceUnavailableReason.ROI_MISSING,
            ),
            (
                "tiny",
                [0.1, 0.1, 0.105, 0.105, 0.9],
                EvidenceUnavailableReason.ROI_TOO_SMALL,
            ),
            (
                "low-resolution",
                [0.1, 0.1, 0.3, 0.3, 0.9],
                EvidenceUnavailableReason.ROI_LOW_RESOLUTION,
            ),
        )
        image = Image.new("RGB", (100, 100), "gray")
        for name, detection, expected_reason in cases:
            with self.subTest(case=name):
                detector_path = self.root / f"{name}.onnx"
                self._write_constant_model(
                    detector_path,
                    (1, 3, 8, 8),
                    np.asarray([[detection]], dtype=np.float32),
                )
                extractor = NosePrintExtractor(
                    detector_path,
                    self._detector_manifest(detector_path),
                    embedding_path,
                    embedding_manifest,
                    NoseRoiPolicy(2, 2, 32, 32),
                )
                result = extractor.extract(image)
                self.assertTrue(result.abstained)
                self.assertIs(result.abstain_reason, expected_reason)

    def test_optional_mask_requires_and_runs_an_exact_artifact(self) -> None:
        mask_path = self.root / "mask.onnx"
        self._write_constant_model(
            mask_path,
            (1, 3, 8, 8),
            np.ones((1, 1, 2, 2), dtype=np.float32),
        )
        manifest = NoseMaskManifest(
            artifact_id="fixture-mask",
            artifact_sha256=sha256(mask_path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 1, 2, 2),
            license=self.license,
            preprocessing=self.preprocessing,
            threshold=0.5,
        )
        crop = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
        np.testing.assert_array_equal(DNPMask(mask_path, manifest).apply(crop), crop)

    @staticmethod
    def _visible_heatmaps() -> np.ndarray:
        heatmaps = np.zeros((1, 2, 2, 3), dtype=np.float32)
        heatmaps[0, 0, 0, 0] = 1.0
        heatmaps[0, 1, 1, 2] = 0.8
        return heatmaps

    def _keypoint_manifest(self, path: Path) -> LandmarkKeypointManifest:
        return LandmarkKeypointManifest(
            artifact_id="fixture-keypoints",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 2, 2, 3),
            license=self.license,
            preprocessing=self.preprocessing,
            keypoint_order=("left_eye", "right_eye"),
            visibility_threshold=0.5,
            min_visible_keypoints=2,
        )

    def _graph_manifest(self, path: Path) -> LandmarkGraphManifest:
        return LandmarkGraphManifest(
            artifact_id="fixture-graph",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 2, 4),
            output_name="output",
            output_shape=(1, 8),
            license=self.license,
            keypoint_order=("left_eye", "right_eye"),
            preprocessing=self._graph_preprocessing(),
        )

    @staticmethod
    def _graph_preprocessing() -> LandmarkGraphPreprocessing:
        return LandmarkGraphPreprocessing(
            coordinate_space="crop_normalized_xy",
            confidence_range=(0.0, 1.0),
            visibility_encoding="binary_0_1",
        )

    def _detector_manifest(self, path: Path) -> NoseDetectorManifest:
        return NoseDetectorManifest(
            artifact_id="fixture-detector",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 1, 5),
            license=self.license,
            preprocessing=self.preprocessing,
            confidence_threshold=0.5,
        )

    def _embedding_manifest(self, path: Path, *, clahe: bool = False) -> NoseEmbeddingManifest:
        preprocessing = self.preprocessing
        if clahe:
            preprocessing = ImagePreprocessing(
                color_mode="RGB",
                layout="NCHW",
                dtype="float32",
                resize="bilinear",
                scale=1.0 / 255.0,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                clahe=ClaheTransform(2.0, (4, 4)),
            )
        return NoseEmbeddingManifest(
            artifact_id="fixture-embedding",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 3),
            license=self.license,
            preprocessing=preprocessing,
        )

    def _mask_manifest(self, path: Path) -> NoseMaskManifest:
        return NoseMaskManifest(
            artifact_id="fixture-mask",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 1, 8, 8),
            license=self.license,
            preprocessing=self.preprocessing,
            threshold=0.5,
        )

    @staticmethod
    def _write_constant_model(
        path: Path,
        input_shape: tuple[int, ...],
        output: np.ndarray,
    ) -> None:
        value = numpy_helper.from_array(output, "constant_value")
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["output"], value=value)],
            "constant-fixture",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, output.shape)],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
        )

    @staticmethod
    def _write_flatten_model(path: Path, input_shape: tuple[int, ...]) -> None:
        graph = helper.make_graph(
            [helper.make_node("Flatten", ["input"], ["output"], axis=1)],
            "flatten-fixture",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
            [
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, (input_shape[0], int(np.prod(input_shape[1:])))
                )
            ],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
        )

    @staticmethod
    def _write_global_average_model(path: Path, input_shape: tuple[int, ...]) -> None:
        graph = helper.make_graph(
            [
                helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
                helper.make_node("Flatten", ["pooled"], ["output"], axis=1),
            ],
            "embedding-fixture",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, (1, 3))],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
        )


if __name__ == "__main__":
    unittest.main()
