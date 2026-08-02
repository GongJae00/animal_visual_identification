"""Exact artifact-bundle contract for MiewID-msv3 wildlife ReID."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from evidence_fusion.base import AbstractEvidencer
from artifact_contracts.model_parity import (
    ModelParityError,
    ModelUsageLane,
    load_model_parity_receipt,
    validate_parity_binding,
)
from artifact_contracts.model_paths import (
    MIEWID_MSV3_HF_REPO,
    MIEWID_MSV3_REVISION,
    MIEWID_MSV3_WEIGHTS_SHA256,
)
from foundation.provenance import content_sha256

MIEWID_IMAGE_SIZE = 440
MIEWID_OUTPUT_DIM = 2152
MIEWID_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
MIEWID_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

_MANIFEST_SCHEMA = "cvi.miewid_artifact_bundle.v1"
_MAXIMUM_ONNX_BYTES = 2_147_483_648
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PREPROCESSING = {
    "color_mode": "RGB",
    "layout": "NCHW",
    "dtype": "float32",
    "resize": "bilinear",
    "image_size": MIEWID_IMAGE_SIZE,
    "scale": "1/255",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class MiewIDModelContractError(RuntimeError):
    """Raised when a MiewID bundle does not match its exact contract."""


@dataclass(frozen=True, slots=True)
class MiewIDPreprocessingManifest:
    color_mode: str
    layout: str
    dtype: str
    resize: str
    image_size: int
    scale: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.to_dict() != _PREPROCESSING:
            raise MiewIDModelContractError(
                "MiewID preprocessing must be the exact 440px RGB contract"
            )

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any]
    ) -> MiewIDPreprocessingManifest:
        if not isinstance(payload, dict) or set(payload) != set(_PREPROCESSING):
            raise MiewIDModelContractError(
                "MiewID preprocessing must use its exact-key schema"
            )
        mean = payload["mean"]
        std = payload["std"]
        if not isinstance(mean, list) or not isinstance(std, list):
            raise MiewIDModelContractError(
                "MiewID preprocessing mean and std must be JSON arrays"
            )
        return cls(
            color_mode=payload["color_mode"],
            layout=payload["layout"],
            dtype=payload["dtype"],
            resize=payload["resize"],
            image_size=payload["image_size"],
            scale=payload["scale"],
            mean=tuple(mean),
            std=tuple(std),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "color_mode": self.color_mode,
            "layout": self.layout,
            "dtype": self.dtype,
            "resize": self.resize,
            "image_size": self.image_size,
            "scale": self.scale,
            "mean": list(self.mean),
            "std": list(self.std),
        }


@dataclass(frozen=True, slots=True)
class MiewIDArtifactManifest:
    """Identity and evidence binding for one self-contained MiewID export."""

    onnx_sha256: str
    source_revision: str
    source_weights_sha256: str
    parity_receipt_sha256: str
    external_data: bool
    usage_state: str
    license_state: str
    preprocessing: MiewIDPreprocessingManifest
    schema_version: str = _MANIFEST_SCHEMA
    model_id: str = MIEWID_MSV3_HF_REPO
    input_name: str = "pixel_values"
    input_shape: tuple[object, ...] = ("batch", 3, 440, 440)
    output_name: str = "embedding"
    output_shape: tuple[object, ...] = ("batch", 2152)
    embedding_normalization: str = "L2"

    def __post_init__(self) -> None:
        if self.schema_version != _MANIFEST_SCHEMA:
            raise MiewIDModelContractError("unsupported MiewID manifest schema")
        if self.model_id != MIEWID_MSV3_HF_REPO:
            raise MiewIDModelContractError("MiewID model_id is not the pinned source")
        if self.source_revision != MIEWID_MSV3_REVISION:
            raise MiewIDModelContractError(
                "MiewID source_revision is not the pinned revision"
            )
        if self.source_weights_sha256 != MIEWID_MSV3_WEIGHTS_SHA256:
            raise MiewIDModelContractError(
                "MiewID source_weights_sha256 is not the pinned weight digest"
            )
        _require_sha256(self.onnx_sha256, "onnx_sha256")
        _require_sha256(self.parity_receipt_sha256, "parity_receipt_sha256")
        if self.external_data is not False:
            raise MiewIDModelContractError(
                "MiewID external_data must be exactly false"
            )
        if self.input_name != "pixel_values" or self.output_name != "embedding":
            raise MiewIDModelContractError("MiewID tensor names are not pinned")
        if self.input_shape != ("batch", 3, 440, 440):
            raise MiewIDModelContractError(
                "MiewID input_shape must be ['batch', 3, 440, 440]"
            )
        if self.output_shape != ("batch", 2152):
            raise MiewIDModelContractError(
                "MiewID output_shape must be ['batch', 2152]"
            )
        if not isinstance(self.preprocessing, MiewIDPreprocessingManifest):
            raise MiewIDModelContractError(
                "MiewID preprocessing must be a MiewIDPreprocessingManifest"
            )
        if self.embedding_normalization != "L2":
            raise MiewIDModelContractError(
                "MiewID embedding_normalization must be 'L2'"
            )
        if self.usage_state != "RESEARCH_ONLY":
            raise MiewIDModelContractError(
                "MiewID usage_state must remain RESEARCH_ONLY"
            )
        if self.license_state != "UNVERIFIED":
            raise MiewIDModelContractError(
                "MiewID license_state must remain UNVERIFIED"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MiewIDArtifactManifest:
        expected = {
            "schema_version",
            "model_id",
            "onnx_sha256",
            "source_revision",
            "source_weights_sha256",
            "input_name",
            "input_shape",
            "output_name",
            "output_shape",
            "preprocessing",
            "embedding_normalization",
            "external_data",
            "usage_state",
            "license_state",
            "parity_receipt_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise MiewIDModelContractError(
                "MiewID manifest must use its exact-key schema"
            )
        input_shape = payload["input_shape"]
        output_shape = payload["output_shape"]
        preprocessing = payload["preprocessing"]
        if not isinstance(input_shape, list) or not isinstance(output_shape, list):
            raise MiewIDModelContractError(
                "MiewID manifest shapes must be JSON arrays"
            )
        if not isinstance(preprocessing, dict):
            raise MiewIDModelContractError(
                "MiewID manifest preprocessing must be an object"
            )
        return cls(
            schema_version=payload["schema_version"],
            model_id=payload["model_id"],
            onnx_sha256=payload["onnx_sha256"],
            source_revision=payload["source_revision"],
            source_weights_sha256=payload["source_weights_sha256"],
            input_name=payload["input_name"],
            input_shape=tuple(input_shape),
            output_name=payload["output_name"],
            output_shape=tuple(output_shape),
            preprocessing=MiewIDPreprocessingManifest.from_dict(preprocessing),
            embedding_normalization=payload["embedding_normalization"],
            external_data=payload["external_data"],
            usage_state=payload["usage_state"],
            license_state=payload["license_state"],
            parity_receipt_sha256=payload["parity_receipt_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "onnx_sha256": self.onnx_sha256,
            "source_revision": self.source_revision,
            "source_weights_sha256": self.source_weights_sha256,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "output_name": self.output_name,
            "output_shape": list(self.output_shape),
            "preprocessing": self.preprocessing.to_dict(),
            "embedding_normalization": self.embedding_normalization,
            "external_data": self.external_data,
            "usage_state": self.usage_state,
            "license_state": self.license_state,
            "parity_receipt_sha256": self.parity_receipt_sha256,
        }


class MiewIDReIDExtractor(AbstractEvidencer):
    """Pinned MiewID-msv3 whole-crop wildlife ReID extractor."""

    name = "wildlife_reid"
    output_dim = MIEWID_OUTPUT_DIM

    def __init__(
        self,
        onnx_path: Path,
        manifest: MiewIDArtifactManifest,
        parity_receipt_path: Path,
        *,
        use_cuda: bool = False,
    ) -> None:
        if not isinstance(manifest, MiewIDArtifactManifest):
            raise TypeError("manifest must be a MiewIDArtifactManifest")
        if not isinstance(use_cuda, bool):
            raise TypeError("use_cuda must be a bool")

        model_bytes = _read_stable_file(
            onnx_path, maximum_bytes=_MAXIMUM_ONNX_BYTES, label="MiewID ONNX"
        )
        import onnx

        try:
            model = onnx.load_model_from_string(model_bytes)
        except Exception as exc:
            raise MiewIDModelContractError("MiewID ONNX protobuf is invalid") from exc
        _reject_external_data(model.graph, onnx)
        if sha256(model_bytes).hexdigest() != manifest.onnx_sha256:
            raise MiewIDModelContractError(
                "MiewID ONNX SHA256 does not match its manifest"
            )
        try:
            parity_receipt = load_model_parity_receipt(
                parity_receipt_path,
                expected_sha256=manifest.parity_receipt_sha256,
            )
            validate_parity_binding(
                parity_receipt,
                model_id=manifest.model_id,
                artifact_sha256=manifest.onnx_sha256,
                source_weights_sha256=manifest.source_weights_sha256,
                preprocessing_sha256=content_sha256(
                    manifest.preprocessing.to_dict()
                ),
                usage_lane=ModelUsageLane.RESEARCH_ONLY,
            )
        except (ModelParityError, OSError, TypeError, ValueError) as exc:
            raise MiewIDModelContractError(
                "MiewID parity receipt is not valid passing parity evidence"
            ) from exc
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            raise MiewIDModelContractError(
                "MiewID ONNX failed graph validation"
            ) from exc
        _validate_graph(model.graph, manifest, onnx)

        import onnxruntime as ort

        available = tuple(ort.get_available_providers())
        provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        if provider not in available:
            unavailable = "unavailable" if use_cuda else "not available"
            raise MiewIDModelContractError(
                f"{provider} was requested for MiewID but is {unavailable}"
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
            raise MiewIDModelContractError(
                "actual ONNX providers differ from the requested strict MiewID "
                f"{'CUDA' if use_cuda else 'CPU'} session"
            )
        runtime_inputs = self._sess.get_inputs()
        runtime_outputs = self._sess.get_outputs()
        if len(runtime_inputs) != 1 or len(runtime_outputs) != 1:
            raise MiewIDModelContractError(
                "MiewID Runtime must expose exactly one input and one output"
            )
        model_input = runtime_inputs[0]
        model_output = runtime_outputs[0]
        if (
            model_input.name != manifest.input_name
            or model_output.name != manifest.output_name
        ):
            raise MiewIDModelContractError(
                "MiewID Runtime tensor names do not match the manifest"
            )
        _validate_shapes(model_input.shape, model_output.shape)
        if (
            getattr(model_input, "type", "tensor(float)") != "tensor(float)"
            or getattr(model_output, "type", "tensor(float)") != "tensor(float)"
        ):
            raise MiewIDModelContractError(
                "MiewID Runtime tensors must be float32"
            )
        self._input_name = model_input.name
        self._model_sha256 = manifest.onnx_sha256
        self._manifest = manifest

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def gallery_contract_fields(self) -> dict[str, object]:
        return {
            "model_sha256": self._manifest.onnx_sha256,
            "source_revision": self._manifest.source_revision,
            "source_weights_sha256": self._manifest.source_weights_sha256,
            "parity_receipt_sha256": self._manifest.parity_receipt_sha256,
            "usage_state": self._manifest.usage_state,
            "license_state": self._manifest.license_state,
            "artifact_bundle_schema": self._manifest.schema_version,
        }

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
        outputs = self._sess.run(["embedding"], {self._input_name: batch})
        if len(outputs) != 1:
            raise MiewIDModelContractError(
                "MiewID Runtime must return exactly one output"
            )
        output = outputs[0]
        if (
            not isinstance(output, np.ndarray)
            or output.dtype != np.float32
            or output.shape != (1, MIEWID_OUTPUT_DIM)
        ):
            raise MiewIDModelContractError(
                "MiewID runtime output must be a float32 ndarray with shape "
                f"(1, 2152), got {getattr(output, 'dtype', None)} "
                f"{getattr(output, 'shape', None)}"
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


def _validate_graph(
    graph: Any,
    manifest: MiewIDArtifactManifest,
    onnx: Any,
) -> None:
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise MiewIDModelContractError(
            "MiewID graph must have exactly one input and one output"
        )
    model_input, model_output = graph.input[0], graph.output[0]
    if model_input.name != manifest.input_name or model_output.name != manifest.output_name:
        raise MiewIDModelContractError(
            "MiewID graph tensor names do not match the manifest"
        )
    if (
        model_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or model_output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
    ):
        raise MiewIDModelContractError("MiewID graph tensors must be float32")
    _validate_shapes(_shape(model_input), _shape(model_output))
    if not graph.node or not graph.initializer:
        raise MiewIDModelContractError(
            "MiewID shape-only graphs without learned parameters are rejected"
        )
    if not any(_has_inline_data(tensor) for tensor in graph.initializer):
        raise MiewIDModelContractError(
            "MiewID graph has no inline learned parameter data"
        )


def _reject_external_data(graph: Any, onnx: Any) -> None:
    def reject_tensor(tensor: Any) -> None:
        if (
            tensor.data_location == onnx.TensorProto.EXTERNAL
            or len(tensor.external_data) != 0
        ):
            raise MiewIDModelContractError(
                "MiewID split external-data artifacts are rejected; "
                "external_data must be false and the ONNX must be self-contained"
            )

    def reject_sparse(sparse: Any) -> None:
        reject_tensor(sparse.values)
        reject_tensor(sparse.indices)

    for tensor in graph.initializer:
        reject_tensor(tensor)
    for sparse in graph.sparse_initializer:
        reject_sparse(sparse)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                reject_tensor(attribute.t)
            for tensor in attribute.tensors:
                reject_tensor(tensor)
            if attribute.HasField("sparse_tensor"):
                reject_sparse(attribute.sparse_tensor)
            for sparse in attribute.sparse_tensors:
                reject_sparse(sparse)
            if attribute.HasField("g"):
                _reject_external_data(attribute.g, onnx)
            for nested_graph in attribute.graphs:
                _reject_external_data(nested_graph, onnx)


def _read_stable_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        before = path.stat()
        if not path.is_file() or before.st_size <= 0:
            raise MiewIDModelContractError(f"{label} must be a non-empty regular file")
        if before.st_size > maximum_bytes:
            raise MiewIDModelContractError(f"{label} exceeds its byte limit")
        with path.open("rb") as stream:
            payload = stream.read(maximum_bytes + 1)
        after = path.stat()
    except OSError as exc:
        raise MiewIDModelContractError(f"unable to read {label}") from exc
    if len(payload) > maximum_bytes:
        raise MiewIDModelContractError(f"{label} exceeds its byte limit")
    if (
        len(payload) != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise MiewIDModelContractError(f"{label} changed while being inspected")
    return payload


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MiewIDModelContractError(
            f"MiewID {label} must be an exact lowercase SHA256 digest"
        )


def _has_inline_data(tensor: Any) -> bool:
    return bool(
        tensor.raw_data
        or tensor.float_data
        or tensor.double_data
        or tensor.int32_data
        or tensor.int64_data
        or tensor.uint64_data
        or tensor.string_data
    )


def _shape(value_info: object) -> tuple[object, ...]:
    dimensions: list[object] = []
    for dimension in value_info.type.tensor_type.shape.dim:  # type: ignore[attr-defined]
        if dimension.dim_value > 0:
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _validate_shapes(input_shape: object, output_shape: object) -> None:
    if (
        not isinstance(input_shape, (list, tuple))
        or tuple(input_shape)
        != ("batch", 3, MIEWID_IMAGE_SIZE, MIEWID_IMAGE_SIZE)
    ):
        raise MiewIDModelContractError(
            "MiewID input must be [batch, 3, 440, 440], "
            f"got {input_shape!r}"
        )
    if (
        not isinstance(output_shape, (list, tuple))
        or tuple(output_shape) != ("batch", MIEWID_OUTPUT_DIM)
    ):
        raise MiewIDModelContractError(
            "MiewID output must be [batch, 2152], "
            f"got {output_shape!r}"
        )


__all__ = [
    "MIEWID_IMAGE_SIZE",
    "MIEWID_OUTPUT_DIM",
    "MiewIDArtifactManifest",
    "MiewIDModelContractError",
    "MiewIDPreprocessingManifest",
    "MiewIDReIDExtractor",
]
