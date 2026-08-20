from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from PIL import Image

from prototype.runtime.engine import IdentityEngine
from shared.contracts.artifact_manifest import (
    ArtifactContractError,
    ArtifactLicense,
    ExactOnnxRuntime,
    ImagePreprocessing,
    NoseEmbeddingManifest,
    UsageLane,
)

from tests.repo_root import REPO_ROOT
from shared.contracts.model_parity import (
    ModelParityReceipt,
    ModelUsageLane,
    ParityFixtureKind,
    ParityFixtureResult,
    ParityThresholds,
)
from shared.contracts.model_paths import (
    MIEWID_MSV3_HF_REPO,
    MIEWID_MSV3_REVISION,
    MIEWID_MSV3_WEIGHTS_SHA256,
)
from shared.foundation.provenance import content_sha256
from identification.export.backbones.extractors import (
    EvidenceExtractorRegistry,
    OnnxExtractor,
    SuperAnimalExtractor,
)
from identification.export.backbones.miewid import (
    MIEWID_OUTPUT_DIM,
    MiewIDArtifactManifest,
    MiewIDModelContractError,
    MiewIDReIDExtractor,
)
from identification.export.nose.extractor import (
    DNPMask,
    YoloNoseDetector,
)
from identification.export.landmark import LandmarkEvidencer
from data.download_models import _convert_superanimal_to_onnx, download_model

class _TensorInfo:
    def __init__(self, name: str, shape: list[object]):
        self.name = name
        self.shape = shape

class _FakeSessionOptions:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value

    def get_session_config_entry(self, key: str) -> str:
        return self.entries[key]

class _FakeSession:
    input_shape: list[object] = ["batch", 3, 440, 440]
    actual_providers = ["CPUExecutionProvider"]
    last_batch: np.ndarray | None = None
    last_kwargs: dict | None = None
    fallback_disabled = False

    def __init__(self, path: str, **kwargs):
        self.path = path
        type(self).last_kwargs = kwargs
        type(self).fallback_disabled = False

    def disable_fallback(self) -> None:
        type(self).fallback_disabled = True

    def get_providers(self) -> list[str]:
        return list(self.actual_providers)

    def get_inputs(self):
        return [_TensorInfo("pixel_values", self.input_shape)]

    def get_outputs(self):
        return [_TensorInfo("embedding", ["batch", MIEWID_OUTPUT_DIM])]

    def run(self, output_names, feeds):
        type(self).last_batch = feeds["pixel_values"]
        return [np.ones((1, MIEWID_OUTPUT_DIM), dtype=np.float32)]

class _ExactFakeSession(_FakeSession):
    input_shape = [1, 3, 8, 8]

    def get_inputs(self):
        return [_TensorInfo("input", self.input_shape)]

    def get_outputs(self):
        return [_TensorInfo("output", [1, 3])]

class _FakeSuperAnimalSession(_FakeSession):
    input_shape = ["batch", 3, 384, 384]

    def get_outputs(self):
        return [_TensorInfo("keypoints", ["batch", 39])]

def _fake_ort(
    *,
    session_type: type[_FakeSession] = _FakeSession,
    available: tuple[str, ...] = ("CPUExecutionProvider",),
):
    return types.SimpleNamespace(
        InferenceSession=session_type,
        SessionOptions=_FakeSessionOptions,
        get_available_providers=lambda: list(available),
    )

class EvidenceModelContractTests(unittest.TestCase):
    def _artifact(self) -> tempfile.NamedTemporaryFile:
        return tempfile.NamedTemporaryFile(suffix=".onnx")

    @staticmethod
    def _write_superanimal_contract(path: Path) -> None:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        weights = numpy_helper.from_array(
            np.ones((39, 3), dtype=np.float32), "weights"
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "GlobalAveragePool", inputs=["images"], outputs=["pooled"]
                ),
                helper.make_node(
                    "Flatten", inputs=["pooled"], outputs=["flat"], axis=1
                ),
                helper.make_node(
                    "Gemm",
                    inputs=["flat", "weights"],
                    outputs=["keypoints"],
                    transB=1,
                ),
            ],
            "rejected-contract",
            [helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, ["batch", 3, 384, 384]
            )],
            [helper.make_tensor_value_info(
                "keypoints", TensorProto.FLOAT, ["batch", 39]
            )],
            [weights],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)]
        )
        onnx.save(model, path)

    @staticmethod
    def _write_miewid_contract(path: Path, image_size: int = 440) -> None:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        weights = numpy_helper.from_array(
            np.ones((MIEWID_OUTPUT_DIM, 3), dtype=np.float32), "weights"
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "GlobalAveragePool", inputs=["pixel_values"], outputs=["pooled"]
                ),
                helper.make_node(
                    "Flatten", inputs=["pooled"], outputs=["flat"], axis=1
                ),
                helper.make_node(
                    "Gemm",
                    inputs=["flat", "weights"],
                    outputs=["embedding"],
                    transB=1,
                ),
            ],
            "miewid-contract",
            [helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT,
                ["batch", 3, image_size, image_size],
            )],
            [helper.make_tensor_value_info(
                "embedding", TensorProto.FLOAT,
                ["batch", MIEWID_OUTPUT_DIM],
            )],
            [weights],
        )
        onnx.save(
            helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 18)]
            ),
            path,
        )

    @staticmethod
    def _write_exact_contract(path: Path) -> None:
        from onnx import TensorProto, helper, numpy_helper

        weights = numpy_helper.from_array(
            np.ones((3, 3), dtype=np.float32), "weights"
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "GlobalAveragePool", inputs=["input"], outputs=["pooled"]
                ),
                helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"]),
                helper.make_node(
                    "Gemm",
                    inputs=["flat", "weights"],
                    outputs=["output"],
                    transB=1,
                ),
            ],
            "exact-contract",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
            [weights],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
            path,
        )

    @staticmethod
    def _exact_manifest(path: Path) -> NoseEmbeddingManifest:
        return NoseEmbeddingManifest(
            artifact_id="exact-test",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 3),
            license=ArtifactLicense("LicenseRef-Test", UsageLane.TEST_FIXTURE),
            preprocessing=ImagePreprocessing(
                "RGB",
                "NCHW",
                "float32",
                "bilinear",
                1.0 / 255.0,
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                None,
            ),
        )

    @staticmethod
    def _write_shape_only_miewid(path: Path) -> None:
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["pixel_values"], outputs=["flat"])],
            "shape-only-miewid",
            [helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, ["batch", 3, 440, 440]
            )],
            [helper.make_tensor_value_info(
                "embedding", TensorProto.FLOAT, ["batch", MIEWID_OUTPUT_DIM]
            )],
        )
        # Keep the declared output shape while making the graph intentionally
        # devoid of learned parameters; graph validation must reject it first.
        graph.node[0].output[0] = "embedding"
        onnx.save(
            helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 18)]
            ),
            path,
        )

    @staticmethod
    def _write_miewid_parity(model_path: Path, receipt_path: Path) -> None:
        preprocessing = {
            "color_mode": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "resize": "bilinear",
            "image_size": 440,
            "scale": "1/255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        receipt = ModelParityReceipt(
            model_id=MIEWID_MSV3_HF_REPO,
            artifact_sha256=sha256(model_path.read_bytes()).hexdigest(),
            source_weights_sha256=MIEWID_MSV3_WEIGHTS_SHA256,
            weight_intake_receipt_sha256=None,
            preprocessing_sha256=content_sha256(preprocessing),
            preprocessor_intake_receipt_sha256=None,
            usage_lane=ModelUsageLane.RESEARCH_ONLY,
            reference_backend="synthetic-test-reference",
            candidate_backend="synthetic-test-candidate",
            thresholds=ParityThresholds(0.0, 0.0, 1e-8, 1.0),
            fixture_panel_receipt_sha256=None,
            fixtures=(
                ParityFixtureResult(
                    fixture_id="synthetic-test",
                    fixture_kind=ParityFixtureKind.SYNTHETIC,
                    input_sha256="1" * 64,
                    reference_output_sha256="2" * 64,
                    candidate_output_sha256="2" * 64,
                    maximum_absolute_error=0.0,
                    maximum_relative_error=0.0,
                    cosine_similarity=1.0,
                    decision="PASS",
                ),
            ),
            decision="PASS",
        )
        receipt_path.write_text(
            json.dumps(receipt.to_dict(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _miewid_manifest(model_path: Path, receipt_path: Path) -> MiewIDArtifactManifest:
        return MiewIDArtifactManifest.from_dict({
            "schema_version": "cvi.miewid_artifact_bundle.v1",
            "model_id": MIEWID_MSV3_HF_REPO,
            "onnx_sha256": sha256(model_path.read_bytes()).hexdigest(),
            "source_revision": MIEWID_MSV3_REVISION,
            "source_weights_sha256": MIEWID_MSV3_WEIGHTS_SHA256,
            "input_name": "pixel_values",
            "input_shape": ["batch", 3, 440, 440],
            "output_name": "embedding",
            "output_shape": ["batch", MIEWID_OUTPUT_DIM],
            "preprocessing": {
                "color_mode": "RGB",
                "layout": "NCHW",
                "dtype": "float32",
                "resize": "bilinear",
                "image_size": 440,
                "scale": "1/255",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "embedding_normalization": "L2",
            "external_data": False,
            "usage_state": "RESEARCH_ONLY",
            "license_state": "UNVERIFIED",
            "parity_receipt_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
        })

    def test_miewid_enforces_official_preprocessing_and_dimension(self) -> None:
        fake_ort = _fake_ort()
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules, {"onnxruntime": fake_ort}
        ):
            artifact = Path(tmpdir) / "miewid.onnx"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_miewid_contract(artifact)
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            extractor = MiewIDReIDExtractor(artifact, manifest, receipt)
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
        self.assertEqual(
            _FakeSession.last_kwargs["providers"], ["CPUExecutionProvider"]
        )
        self.assertEqual(_FakeSession.last_kwargs["enable_fallback"], 0)
        self.assertEqual(_FakeSession.last_kwargs["sess_options"].entries, {})
        self.assertTrue(_FakeSession.fallback_disabled)

    def test_miewid_rejects_wrong_spatial_contract(self) -> None:
        class FailIfLoaded:
            def __init__(self, *args, **kwargs):
                raise AssertionError("rejected MiewID graph reached ONNX Runtime")

        fake_ort = types.SimpleNamespace(InferenceSession=FailIfLoaded)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules, {"onnxruntime": fake_ort}
        ):
            artifact = Path(tmpdir) / "miewid.onnx"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_miewid_contract(artifact, image_size=160)
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            with self.assertRaisesRegex(MiewIDModelContractError, "440"):
                MiewIDReIDExtractor(artifact, manifest, receipt)

    def test_miewid_rejects_shape_only_graph_and_unbound_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "shape-only.onnx"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_shape_only_miewid(artifact)
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            with self.assertRaisesRegex(MiewIDModelContractError, "shape-only"):
                MiewIDReIDExtractor(artifact, manifest, receipt)
            with self.assertRaises(TypeError):
                MiewIDReIDExtractor(artifact)  # type: ignore[call-arg]

    def test_miewid_rejects_stale_split_external_data_precisely(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "stale-split.onnx"
            sidecar = Path(tmpdir) / "stale-split.weights"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_miewid_contract(artifact)
            model = onnx.load_model(artifact)
            onnx.save_model(
                model,
                artifact,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=sidecar.name,
                size_threshold=0,
            )
            self.assertTrue(sidecar.exists())
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            with self.assertRaisesRegex(
                MiewIDModelContractError, "split external-data artifacts"
            ):
                MiewIDReIDExtractor(artifact, manifest, receipt)

    def test_miewid_binds_parity_usage_license_and_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "miewid.onnx"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_miewid_contract(artifact)
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)

            receipt.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(MiewIDModelContractError, "valid passing"):
                MiewIDReIDExtractor(artifact, manifest, receipt)

            payload = manifest.to_dict()
            payload["usage_state"] = "DEPLOYMENT"
            with self.assertRaisesRegex(MiewIDModelContractError, "RESEARCH_ONLY"):
                MiewIDArtifactManifest.from_dict(payload)
            payload = manifest.to_dict()
            payload["license_state"] = "VERIFIED"
            with self.assertRaisesRegex(MiewIDModelContractError, "UNVERIFIED"):
                MiewIDArtifactManifest.from_dict(payload)

            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            fake_ort = _fake_ort()
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                with self.assertRaisesRegex(
                    MiewIDModelContractError, "requested.*unavailable"
                ):
                    MiewIDReIDExtractor(
                        artifact, manifest, receipt, use_cuda=True
                    )

    def test_miewid_rejects_cuda_provider_substitution_and_disables_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "miewid.onnx"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_miewid_contract(artifact)
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            fake_ort = _fake_ort(
                available=("CUDAExecutionProvider", "CPUExecutionProvider")
            )
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                with self.assertRaisesRegex(
                    MiewIDModelContractError, "strict MiewID CUDA session"
                ):
                    MiewIDReIDExtractor(
                        artifact, manifest, receipt, use_cuda=True
                    )
        self.assertEqual(
            _FakeSession.last_kwargs["providers"], ["CUDAExecutionProvider"]
        )
        self.assertEqual(_FakeSession.last_kwargs["enable_fallback"], 0)
        self.assertEqual(
            _FakeSession.last_kwargs["sess_options"].get_session_config_entry(
                "session.disable_cpu_ep_fallback"
            ),
            "1",
        )
        self.assertTrue(_FakeSession.fallback_disabled)

    def test_exact_runtime_cpu_only_session_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "exact.onnx"
            self._write_exact_contract(artifact)
            with patch.dict(
                sys.modules,
                {"onnxruntime": _fake_ort(session_type=_ExactFakeSession)},
            ):
                ExactOnnxRuntime(artifact, self._exact_manifest(artifact))
        self.assertEqual(
            _ExactFakeSession.last_kwargs["providers"], ["CPUExecutionProvider"]
        )
        self.assertEqual(_ExactFakeSession.last_kwargs["enable_fallback"], 0)
        self.assertEqual(_ExactFakeSession.last_kwargs["sess_options"].entries, {})
        self.assertTrue(_ExactFakeSession.fallback_disabled)

    def test_exact_runtime_rejects_unavailable_and_substituted_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "exact.onnx"
            self._write_exact_contract(artifact)
            manifest = self._exact_manifest(artifact)
            with patch.dict(
                sys.modules,
                {"onnxruntime": _fake_ort(session_type=_ExactFakeSession)},
            ):
                with self.assertRaisesRegex(
                    ArtifactContractError, "requested but is not available"
                ):
                    ExactOnnxRuntime(artifact, manifest, use_cuda=True)

            with patch.dict(
                sys.modules,
                {"onnxruntime": _fake_ort(
                    session_type=_ExactFakeSession,
                    available=("CUDAExecutionProvider", "CPUExecutionProvider"),
                )},
            ):
                with self.assertRaisesRegex(
                    ArtifactContractError, "strict CUDA session"
                ):
                    ExactOnnxRuntime(artifact, manifest, use_cuda=True)
        self.assertEqual(
            _ExactFakeSession.last_kwargs["providers"], ["CUDAExecutionProvider"]
        )
        self.assertEqual(_ExactFakeSession.last_kwargs["enable_fallback"], 0)
        self.assertEqual(
            _ExactFakeSession.last_kwargs["sess_options"].get_session_config_entry(
                "session.disable_cpu_ep_fallback"
            ),
            "1",
        )
        self.assertTrue(_ExactFakeSession.fallback_disabled)

    def test_miewid_rejects_hash_bound_but_untyped_parity_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "miewid.onnx"
            receipt = Path(tmpdir) / "parity-receipt.json"
            self._write_miewid_contract(artifact)
            receipt.write_text('{"decision":"PASS"}\n', encoding="utf-8")
            manifest = self._miewid_manifest(artifact, receipt)
            with self.assertRaisesRegex(
                MiewIDModelContractError, "valid passing parity evidence"
            ):
                MiewIDReIDExtractor(artifact, manifest, receipt)

    def test_miewid_api_requires_exact_bundle_paths_and_defaults_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "miewid.onnx"
            receipt = root / "parity-receipt.json"
            manifest_path = root / "manifest.json"
            self._write_miewid_contract(artifact)
            self._write_miewid_parity(artifact, receipt)
            manifest = self._miewid_manifest(artifact, receipt)
            manifest_path.write_text(json.dumps(manifest.to_dict()))
            runtime = IdentityEngine.__new__(IdentityEngine)
            runtime._config = {  # type: ignore[attr-defined]
                "channels": {
                    "wildlife": {
                        "type": "miewid_reid",
                        "model_path": str(artifact),
                        "manifest_path": str(manifest_path),
                        "parity_receipt_path": str(receipt),
                    }
                }
            }
            sentinel = object()
            with patch(
                "identification.export.backbones.miewid.MiewIDReIDExtractor",
                return_value=sentinel,
            ) as extractor:
                evidence = runtime._build_evidence()
            self.assertIs(evidence["wildlife"], sentinel)
            extractor.assert_called_once_with(
                artifact, manifest, receipt, use_cuda=False
            )

            runtime._config = {  # type: ignore[attr-defined]
                "channels": {
                    "wildlife": {
                        "type": "miewid_reid",
                        "path": str(artifact),
                        "allow_research_only_model": True,
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "exact MiewID bundle schema"):
                runtime._build_evidence()

    def test_untrained_channels_are_not_runtime_defaults(self) -> None:
        with self.assertRaises(TypeError):
            LandmarkEvidencer()

    def test_missing_nose_models_do_not_fabricate_evidence(self) -> None:
        with self.assertRaises(TypeError):
            DNPMask()
        with self.assertRaises(TypeError):
            YoloNoseDetector()

    def test_superanimal_runtime_rejects_dummy_onnx_before_loading(self) -> None:
        class FailIfLoaded:
            def __init__(self, *args, **kwargs):
                raise AssertionError("disabled artifact reached ONNX Runtime")

        fake_ort = types.SimpleNamespace(InferenceSession=FailIfLoaded)
        with self._artifact() as artifact, patch.dict(
            sys.modules, {"onnxruntime": fake_ort}
        ):
            with self.assertRaisesRegex(RuntimeError, "SuperAnimal evidence is disabled"):
                SuperAnimalExtractor(Path(artifact.name))
        disabled = SuperAnimalExtractor.__new__(SuperAnimalExtractor)
        with self.assertRaisesRegex(RuntimeError, "SuperAnimal evidence is disabled"):
            disabled.extract(Image.new("RGB", (8, 8)))

    def test_legacy_registry_rejects_structural_onnx(self) -> None:
        with self._artifact() as artifact:
            with self.assertRaisesRegex(RuntimeError, "ONNX evidence is disabled"):
                EvidenceExtractorRegistry.from_onnx_dict(
                    {"structural": Path(artifact.name)}
                )

    def test_generic_onnx_alias_rejects_superanimal_contract(self) -> None:
        class FailIfLoaded:
            def __init__(self, *args, **kwargs):
                raise AssertionError("rejected graph reached ONNX Runtime")

        fake_ort = types.SimpleNamespace(InferenceSession=FailIfLoaded)
        with self._artifact() as artifact, patch.dict(
            sys.modules, {"onnxruntime": fake_ort}
        ):
            self._write_superanimal_contract(Path(artifact.name))
            with self.assertRaisesRegex(RuntimeError, "replacement ONNX contract"):
                EvidenceExtractorRegistry.from_onnx_dict(
                    {"visual": Path(artifact.name)}, input_sizes={"visual": 384}
                )
        with tempfile.NamedTemporaryFile(prefix="superanimal-", suffix=".onnx") as artifact:
            with self.assertRaisesRegex(RuntimeError, "ONNX artifacts are disabled"):
                OnnxExtractor(Path(artifact.name))

    def test_superanimal_export_and_download_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "checkpoint.pt"
            output = root / "model.onnx"
            source.write_bytes(b"not a checkpoint")
            output.write_bytes(b"existing unverified artifact")
            before = output.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "export is disabled"):
                _convert_superanimal_to_onnx(source, output)
            self.assertEqual(output.read_bytes(), before)

        with patch("data.download_models._download_hf") as download:
            with self.assertRaisesRegex(RuntimeError, "SuperAnimal is disabled"):
                download_model("superanimal")
            download.assert_not_called()

    def test_superanimal_cli_and_environment_check_report_disabled(self) -> None:
        completed = subprocess.run(
            [sys.executable, "data/download_models.py", "--model", "superanimal"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SuperAnimal is disabled", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

        script = (
            REPO_ROOT / "setup" / "check_env.sh"
        ).read_text()
        self.assertIn("SuperAnimal runtime: DISABLED", script)
        self.assertNotIn("--model superanimal", script)
        self.assertIn('"runtime",', script)
        self.assertNotIn('"cvi",', script)
        downloader = (
            REPO_ROOT / "data" / "download_models.py"
        ).read_text()
        self.assertIn("external_data=False", downloader)

if __name__ == "__main__":
    unittest.main()
