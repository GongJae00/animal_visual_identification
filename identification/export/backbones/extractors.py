from __future__ import annotations

from abc import abstractmethod
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from shared.contracts.model_contract import (
    ConvNeXtModelManifest,
    DogFaceNetModelManifest,
    OnnxEvidenceContractError,
    OnnxEvidenceModelManifest,
    PetReIDModelManifest,
)
from shared.contracts.model_contracts import (
    reject_unverified_superanimal_onnx,
    validated_onnx_bytes,
)
from shared.foundation.provenance import content_sha256
from representation.evidence.base import AbstractEvidencer


class EvidenceExtractor(AbstractEvidencer):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray: ...

    @abstractmethod
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray: ...

    def close(self) -> None:
        pass


class OnnxExtractor(EvidenceExtractor):
    def __init__(
        self,
        model_path: Path,
        manifest: OnnxEvidenceModelManifest | None = None,
        *,
        use_cuda: bool = False,
        **legacy_options: object,
    ) -> None:
        reject_unverified_superanimal_onnx(model_path)
        model_bytes = validated_onnx_bytes(model_path)
        if manifest is None:
            raise OnnxEvidenceContractError(
                "a model manifest is required; legacy input_size, "
                "output_dim, mean, std, and provider arguments are not trusted"
            )
        if not isinstance(manifest, OnnxEvidenceModelManifest):
            raise TypeError("manifest must be an OnnxEvidenceModelManifest")
        if legacy_options:
            names = ", ".join(sorted(legacy_options))
            raise OnnxEvidenceContractError(
                f"legacy ONNX options are not supported ({names}); declare them in manifest"
            )
        actual_sha256 = sha256(model_bytes).hexdigest()
        if actual_sha256 != manifest.model_sha256:
            raise OnnxEvidenceContractError(
                "ONNX artifact SHA256 does not match the model manifest"
            )

        import onnx

        model = onnx.load_model_from_string(model_bytes)
        self._validate_graph(model.graph, manifest, onnx.TensorProto.FLOAT)
        del model

        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        if provider not in available:
            raise OnnxEvidenceContractError(
                f"{provider} was requested but is not available"
            )
        session_options = ort.SessionOptions()
        if use_cuda:
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
        self._sess = ort.InferenceSession(
            model_bytes,
            sess_options=session_options,
            providers=[provider],
            enable_fallback=0,
        )
        self._sess.disable_fallback()
        actual_providers = tuple(self._sess.get_providers())
        expected_providers = (
            ("CUDAExecutionProvider", "CPUExecutionProvider")
            if use_cuda
            else ("CPUExecutionProvider",)
        )
        if actual_providers != expected_providers:
            raise OnnxEvidenceContractError(
                "actual ONNX providers differ from the requested strict "
                f"{'CUDA' if use_cuda else 'CPU'} session"
            )
        inputs = self._sess.get_inputs()
        outputs = self._sess.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise OnnxEvidenceContractError(
                "ONNX Runtime must expose exactly one input and one output"
            )
        self._inp = inputs[0]
        self._out = outputs[0]
        self._validate_runtime_metadata(self._inp, self._out, manifest)
        self._manifest = manifest
        self._model_sha256 = actual_sha256
        self._mean = np.asarray(manifest.preprocessing.mean, dtype=np.float32)
        self._std = np.asarray(manifest.preprocessing.std, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return self._manifest.output_dim

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def gallery_contract_fields(self) -> dict[str, str]:
        return {
            "model_sha256": self._model_sha256,
            "model_manifest_sha256": content_sha256(self._manifest.to_dict()),
            "preprocessing_sha256": content_sha256(
                self._manifest.preprocessing.to_dict()
            ),
        }

    def preprocess(self, image: Image.Image) -> np.ndarray:
        _, _, height, width = self._manifest.input_shape
        resampling = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }[self._manifest.preprocessing.resize]
        img = image.convert(self._manifest.preprocessing.color_mode).resize(
            (width, height), resampling
        )
        arr = np.asarray(img, dtype=np.float32) * self._manifest.preprocessing.scale
        arr = (arr - self._mean) / self._std
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[np.newaxis, :])

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._normalized(self._run(self.preprocess(image)))[0]

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.output_dim), dtype=np.float32)
        if self._manifest.input_shape[0] == 1:
            return np.stack([self.extract(image) for image in images])
        tensors = np.concatenate([self.preprocess(img) for img in images], axis=0)
        return self._normalized(self._run(tensors))

    def run_raw(self, image: Image.Image) -> np.ndarray:
        return self._run(self.preprocess(image))[0]

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        result = self._sess.run([self._out.name], {self._inp.name: tensor})
        if len(result) != 1 or not isinstance(result[0], np.ndarray):
            raise OnnxEvidenceContractError(
                "ONNX runtime must return exactly one ndarray output"
            )
        output = result[0]
        expected_shape = (tensor.shape[0], self.output_dim)
        if output.shape != expected_shape:
            raise OnnxEvidenceContractError(
                f"runtime output must have shape {expected_shape}, got {output.shape}"
            )
        if output.dtype != np.float32:
            raise OnnxEvidenceContractError(
                f"runtime output must be float32, got {output.dtype}"
            )
        if not np.isfinite(output).all():
            raise OnnxEvidenceContractError("runtime output contains non-finite values")
        norms = np.linalg.norm(output, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise OnnxEvidenceContractError(
                "runtime output must have finite nonzero norm"
            )
        return output

    @staticmethod
    def _normalized(output: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        return np.asarray(output / norms, dtype=np.float32)

    @staticmethod
    def _validate_graph(
        graph: object,
        manifest: OnnxEvidenceModelManifest,
        float_tensor_type: int,
    ) -> None:
        inputs = graph.input  # type: ignore[attr-defined]
        outputs = graph.output  # type: ignore[attr-defined]
        if len(inputs) != 1 or len(outputs) != 1:
            raise OnnxEvidenceContractError(
                "ONNX graph must declare exactly one input and one output"
            )
        model_input = inputs[0]
        model_output = outputs[0]
        if model_input.name != manifest.input_name:
            raise OnnxEvidenceContractError(
                f"ONNX input name must be {manifest.input_name!r}"
            )
        if model_output.name != manifest.output_name:
            raise OnnxEvidenceContractError(
                f"ONNX output name must be {manifest.output_name!r}"
            )
        if model_input.type.tensor_type.elem_type != float_tensor_type:
            raise OnnxEvidenceContractError("ONNX input tensor must be float32")
        if model_output.type.tensor_type.elem_type != float_tensor_type:
            raise OnnxEvidenceContractError("ONNX output tensor must be float32")
        input_shape = _onnx_shape(model_input)
        output_shape = _onnx_shape(model_output)
        _validate_shapes(input_shape, output_shape, manifest)

    @staticmethod
    def _validate_runtime_metadata(
        model_input: object,
        model_output: object,
        manifest: OnnxEvidenceModelManifest,
    ) -> None:
        if model_input.name != manifest.input_name:  # type: ignore[attr-defined]
            raise OnnxEvidenceContractError("runtime input name differs from manifest")
        if model_output.name != manifest.output_name:  # type: ignore[attr-defined]
            raise OnnxEvidenceContractError("runtime output name differs from manifest")
        input_type = getattr(model_input, "type", "tensor(float)")
        output_type = getattr(model_output, "type", "tensor(float)")
        if input_type != "tensor(float)" or output_type != "tensor(float)":
            raise OnnxEvidenceContractError("runtime tensors must be float32")
        _validate_shapes(
            tuple(model_input.shape),  # type: ignore[attr-defined]
            tuple(model_output.shape),  # type: ignore[attr-defined]
            manifest,
        )


class DogFaceNetExtractor(EvidenceExtractor):
    def __init__(
        self,
        model_path: Path,
        manifest: DogFaceNetModelManifest | None = None,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, DogFaceNetModelManifest):
            raise OnnxEvidenceContractError(
                "DogFaceNetExtractor requires a DogFaceNetModelManifest"
            )
        self._onnx = OnnxExtractor(model_path, manifest, use_cuda=use_cuda)

    @property
    def output_dim(self) -> int:
        return self._onnx.output_dim

    @property
    def gallery_contract_fields(self) -> dict[str, str]:
        return self._onnx.gallery_contract_fields

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._onnx.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._onnx.extract_batch(images)


class ConvNeXtExtractor(EvidenceExtractor):
    def __init__(
        self,
        model_path: Path,
        manifest: ConvNeXtModelManifest | None = None,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, ConvNeXtModelManifest):
            raise OnnxEvidenceContractError(
                "ConvNeXtExtractor requires a ConvNeXtModelManifest"
            )
        self._onnx = OnnxExtractor(model_path, manifest, use_cuda=use_cuda)

    @property
    def output_dim(self) -> int:
        return self._onnx.output_dim

    @property
    def gallery_contract_fields(self) -> dict[str, str]:
        return self._onnx.gallery_contract_fields

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._onnx.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._onnx.extract_batch(images)


class SuperAnimalExtractor(EvidenceExtractor):
    _DISABLED_REASON = (
        "SuperAnimal evidence is disabled. IdentityEngine has no verified HRNet-W32 "
        "ONNX export/decoder contract, and the official model weights are "
        "licensed for academic non-commercial use only."
    )

    def __init__(
        self,
        model_path: Path,
        num_keypoints: int = 39,
        output_dim: int = 256,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        raise RuntimeError(self._DISABLED_REASON)

    @property
    def output_dim(self) -> int:
        raise RuntimeError(self._DISABLED_REASON)

    def extract(self, image: Image.Image) -> np.ndarray:
        raise RuntimeError(self._DISABLED_REASON)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        raise RuntimeError(self._DISABLED_REASON)


class PetReIDExtractor(EvidenceExtractor):
    def __init__(
        self,
        model_path: Path,
        manifest: PetReIDModelManifest | None = None,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, PetReIDModelManifest):
            raise OnnxEvidenceContractError(
                "PetReIDExtractor requires a PetReIDModelManifest"
            )
        self._onnx = OnnxExtractor(model_path, manifest, use_cuda=use_cuda)

    @property
    def output_dim(self) -> int:
        return self._onnx.output_dim

    @property
    def gallery_contract_fields(self) -> dict[str, str]:
        return self._onnx.gallery_contract_fields

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._onnx.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._onnx.extract_batch(images)


class EvidenceExtractorRegistry:
    def __init__(self) -> None:
        self._extractors: dict[str, EvidenceExtractor] = {}

    def register(self, name: str, extractor: EvidenceExtractor) -> None:
        self._extractors[name] = extractor

    def get(self, name: str) -> EvidenceExtractor:
        if name not in self._extractors:
            raise KeyError(
                f"no extractor registered for '{name}'; "
                f"available: {list(self._extractors)}"
            )
        return self._extractors[name]

    @property
    def names(self) -> list[str]:
        return list(self._extractors)

    @property
    def visual(self) -> EvidenceExtractor | None:
        return self._extractors.get("visual")

    @property
    def texture(self) -> EvidenceExtractor | None:
        return self._extractors.get("texture")

    @property
    def structural(self) -> EvidenceExtractor | None:
        return self._extractors.get("structural")

    @property
    def nose(self) -> EvidenceExtractor | None:
        return self._extractors.get("nose")

    def close(self) -> None:
        for ext in self._extractors.values():
            ext.close()

    @staticmethod
    def from_onnx_dict(
        paths: dict[str, Path],
        input_sizes: dict[str, int] | None = None,
        output_dims: dict[str, int] | None = None,
        manifests: dict[str, OnnxEvidenceModelManifest] | None = None,
    ) -> EvidenceExtractorRegistry:
        registry = EvidenceExtractorRegistry()
        for name, p in paths.items():
            if not p.exists():
                continue
            if name.lower() in {"superanimal", "structural", "landmark"}:
                raise RuntimeError(
                    f"'{name}' ONNX evidence is disabled until its model, "
                    "preprocessing, output decoder, and license are verified"
                )
            # Inspect the graph before legacy argument rejection so a disguised
            # SuperAnimal replacement cannot pass through the generic alias.
            validated_onnx_bytes(p)
            if input_sizes or output_dims:
                raise OnnxEvidenceContractError(
                    "input_sizes/output_dims are not trusted; provide manifests"
                )
            manifest = (manifests or {}).get(name)
            if manifest is None:
                raise OnnxEvidenceContractError(
                    f"a model manifest is required for ONNX evidence {name!r}"
                )
            registry.register(name, OnnxExtractor(p, manifest))
        return registry


def _onnx_shape(value_info: object) -> tuple[int | str | None, ...]:
    dimensions: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:  # type: ignore[attr-defined]
        if dimension.dim_value > 0:
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _validate_shapes(
    input_shape: tuple[object, ...],
    output_shape: tuple[object, ...],
    manifest: OnnxEvidenceModelManifest,
) -> None:
    if input_shape != manifest.input_shape:
        raise OnnxEvidenceContractError(
            f"ONNX input shape must be {manifest.input_shape}, got {input_shape}"
        )
    expected_output_shape = (manifest.input_shape[0], manifest.output_dim)
    if output_shape != expected_output_shape:
        raise OnnxEvidenceContractError(
            f"ONNX output shape must be {expected_output_shape}, got {output_shape}"
        )
