"""Strict optional ONNX Runtime CPU embedding backend."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from shared.contracts.model_contracts import reject_unverified_superanimal_onnx
from shared.foundation.provenance import content_sha256
from prototype.export.embedding_producer import (
    EmbeddingBackendIdentity,
    EmbeddingRuntimeResources,
)


class ImageTensorLayout(StrEnum):
    NCHW = "NCHW"
    NHWC = "NHWC"


class ImageResizePolicy(StrEnum):
    EXACT = "EXACT"
    STRETCH = "STRETCH"
    SHORTEST_EDGE_CENTER_CROP = "SHORTEST_EDGE_CENTER_CROP"


class ImageInterpolation(StrEnum):
    NEAREST = "NEAREST"
    BILINEAR = "BILINEAR"
    BICUBIC = "BICUBIC"


class ImageChannelOrder(StrEnum):
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"


def dinov2_image_preprocessing_config(
    contract: Any,
    *,
    decoder_version: str,
    maximum_source_width: int = 16_384,
    maximum_source_height: int = 16_384,
    maximum_source_pixels: int = 67_108_864,
) -> ImagePreprocessingConfig:
    processor = contract.preprocessor
    return ImagePreprocessingConfig(
        schema_version="cvi.image_preprocessing.v2",
        width=processor["crop_size"]["width"],
        height=processor["crop_size"]["height"],
        color_mode="RGB",
        channel_order=ImageChannelOrder.RGB,
        layout=ImageTensorLayout.NCHW,
        resize_policy=ImageResizePolicy.SHORTEST_EDGE_CENTER_CROP,
        interpolation=ImageInterpolation.BICUBIC,
        value_scale=processor["rescale_factor"],
        mean=tuple(processor["image_mean"]),
        std=tuple(processor["image_std"]),
        maximum_source_width=maximum_source_width,
        maximum_source_height=maximum_source_height,
        maximum_source_pixels=maximum_source_pixels,
        allowed_source_modes=("L", "RGB"),
        decoder_version=decoder_version,
        allowed_formats=("PNG",),
        operation_order="CONVERT_THEN_RESIZE_THEN_CENTER_CROP",
        resize_shortest_edge=processor["size"]["shortest_edge"],
    )


class OnnxGraphOptimization(StrEnum):
    DISABLE_ALL = "ORT_DISABLE_ALL"
    ENABLE_BASIC = "ORT_ENABLE_BASIC"
    ENABLE_EXTENDED = "ORT_ENABLE_EXTENDED"
    ENABLE_ALL = "ORT_ENABLE_ALL"


_CUDA_PROVIDER_NAME = "CUDAExecutionProvider"
_CPU_PROVIDER_NAME = "CPUExecutionProvider"
_CUDA_SAFE_OPTION_KEYS = frozenset(
    {
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "cudnn_conv_use_max_workspace",
        "device_id",
        "do_copy_in_default_stream",
        "enable_cuda_graph",
        "gpu_mem_limit",
        "prefer_nhwc",
        "use_ep_level_unified_stream",
        "use_tf32",
    }
)
_CUDA_ORT_127_ACTUAL_DEFAULTS = {
    "cudnn_conv1d_pad_to_nc1d": "0",
    "enable_skip_layer_norm_strict_mode": "0",
    "fuse_conv_bias": "0",
    "gpu_external_alloc": "0",
    "gpu_external_empty_cache": "0",
    "gpu_external_free": "0",
    "has_user_compute_stream": "0",
    "sdpa_kernel": "0",
    "tunable_op_enable": "0",
    "tunable_op_max_tuning_duration_ms": "0",
    "tunable_op_tuning_enable": "0",
    "user_compute_stream": "0",
}


@dataclass(frozen=True, slots=True)
class ImagePreprocessingConfig:
    width: int
    height: int
    color_mode: str
    channel_order: ImageChannelOrder
    layout: ImageTensorLayout
    resize_policy: ImageResizePolicy
    interpolation: ImageInterpolation
    value_scale: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    maximum_source_width: int
    maximum_source_height: int
    maximum_source_pixels: int
    allowed_source_modes: tuple[str, ...]
    decoder_version: str
    allowed_formats: tuple[str, ...] = ("PNG",)
    tensor_dtype: str = "float32"
    decoder: str = "pillow"
    exif_orientation: str = "IGNORE"
    alpha_policy: str = "REJECT"
    icc_profile_policy: str = "REJECT"
    gamma_policy: str = "IGNORE_METADATA"
    operation_order: str = "CONVERT_THEN_RESIZE"
    schema_version: str = "cvi.image_preprocessing.v1"
    resize_shortest_edge: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "cvi.image_preprocessing.v1",
            "cvi.image_preprocessing.v2",
        }:
            raise ValueError("unsupported image preprocessing schema")
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        _require_positive_int(
            self.maximum_source_width,
            "maximum_source_width",
        )
        _require_positive_int(
            self.maximum_source_height,
            "maximum_source_height",
        )
        _require_positive_int(
            self.maximum_source_pixels,
            "maximum_source_pixels",
        )
        if self.color_mode not in {"RGB", "L"}:
            raise ValueError("color_mode must be RGB or L")
        expected_order = (
            {ImageChannelOrder.RGB, ImageChannelOrder.BGR}
            if self.color_mode == "RGB"
            else {ImageChannelOrder.GRAY}
        )
        if self.channel_order not in expected_order:
            raise ValueError("channel order is incompatible with color mode")
        channels = self.channels
        if len(self.mean) != channels or len(self.std) != channels:
            raise ValueError("mean/std length differs from channel count")
        for value in self.mean:
            _require_finite(value, "mean")
        for value in self.std:
            _require_finite_positive(value, "std")
        _require_finite_positive(self.value_scale, "value_scale")
        if not self.allowed_formats:
            raise ValueError("at least one image format is required")
        _validate_sorted_uppercase_strings(
            self.allowed_formats,
            "allowed_formats",
        )
        if not self.allowed_source_modes:
            raise ValueError("at least one source image mode is required")
        _validate_sorted_strings(
            self.allowed_source_modes,
            "allowed_source_modes",
        )
        if any(
            mode not in {"L", "RGB"}
            for mode in self.allowed_source_modes
        ):
            raise ValueError("initial source modes are restricted to L/RGB")
        if self.tensor_dtype != "float32":
            raise ValueError("initial ONNX tensor dtype is fixed to float32")
        if self.decoder != "pillow":
            raise ValueError("initial image decoder is fixed to pillow")
        _require_nonempty(self.decoder_version, "decoder_version")
        if self.exif_orientation != "IGNORE":
            raise ValueError("EXIF orientation behavior is fixed to IGNORE")
        if self.alpha_policy != "REJECT":
            raise ValueError("initial alpha policy is fixed to REJECT")
        if self.icc_profile_policy != "REJECT":
            raise ValueError("initial ICC profile policy is fixed to REJECT")
        if self.gamma_policy != "IGNORE_METADATA":
            raise ValueError(
                "initial gamma policy is fixed to IGNORE_METADATA"
            )
        if self.resize_policy is ImageResizePolicy.SHORTEST_EDGE_CENTER_CROP:
            if self.schema_version != "cvi.image_preprocessing.v2":
                raise ValueError("shortest-edge preprocessing requires schema v2")
            _require_positive_int(
                self.resize_shortest_edge,
                "resize_shortest_edge",
            )
            if self.resize_shortest_edge < max(self.width, self.height):
                raise ValueError("resize_shortest_edge must cover the center crop")
            if self.operation_order != "CONVERT_THEN_RESIZE_THEN_CENTER_CROP":
                raise ValueError(
                    "shortest-edge preprocessing operation order differs"
                )
        elif self.resize_shortest_edge is not None:
            raise ValueError(
                "resize_shortest_edge is only valid for shortest-edge preprocessing"
            )
        elif self.operation_order != "CONVERT_THEN_RESIZE":
            raise ValueError("initial operation order is CONVERT_THEN_RESIZE")

    @property
    def channels(self) -> int:
        return 3 if self.color_mode == "RGB" else 1

    @property
    def config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode,
            "channel_order": self.channel_order.value,
            "layout": self.layout.value,
            "resize_policy": self.resize_policy.value,
            "interpolation": self.interpolation.value,
            "value_scale": self.value_scale,
            "mean": list(self.mean),
            "std": list(self.std),
            "maximum_source_width": self.maximum_source_width,
            "maximum_source_height": self.maximum_source_height,
            "maximum_source_pixels": self.maximum_source_pixels,
            "allowed_source_modes": list(self.allowed_source_modes),
            "decoder_version": self.decoder_version,
            "allowed_formats": list(self.allowed_formats),
            "tensor_dtype": self.tensor_dtype,
            "decoder": self.decoder,
            "exif_orientation": self.exif_orientation,
            "alpha_policy": self.alpha_policy,
            "icc_profile_policy": self.icc_profile_policy,
            "gamma_policy": self.gamma_policy,
            "operation_order": self.operation_order,
        }
        if self.schema_version == "cvi.image_preprocessing.v2":
            payload["resize_shortest_edge"] = self.resize_shortest_edge
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> ImagePreprocessingConfig:
        schema_version = payload.get("schema_version")
        expected_keys = {
            "schema_version",
            "width",
            "height",
            "color_mode",
            "channel_order",
            "layout",
            "resize_policy",
            "interpolation",
            "value_scale",
            "mean",
            "std",
            "maximum_source_width",
            "maximum_source_height",
            "maximum_source_pixels",
            "allowed_source_modes",
            "decoder_version",
            "allowed_formats",
            "tensor_dtype",
            "decoder",
            "exif_orientation",
            "alpha_policy",
            "icc_profile_policy",
            "gamma_policy",
            "operation_order",
        }
        if schema_version == "cvi.image_preprocessing.v2":
            expected_keys.add("resize_shortest_edge")
        _require_exact_keys(
            payload,
            expected_keys,
            "image preprocessing config",
        )
        mean = payload["mean"]
        std = payload["std"]
        formats = payload["allowed_formats"]
        source_modes = payload["allowed_source_modes"]
        if (
            not isinstance(mean, list)
            or not isinstance(std, list)
            or not isinstance(formats, list)
            or not isinstance(source_modes, list)
        ):
            raise TypeError(
                "mean, std, formats, and source modes must be lists"
            )
        return cls(
            schema_version=payload["schema_version"],
            width=payload["width"],
            height=payload["height"],
            color_mode=payload["color_mode"],
            channel_order=ImageChannelOrder(payload["channel_order"]),
            layout=ImageTensorLayout(payload["layout"]),
            resize_policy=ImageResizePolicy(payload["resize_policy"]),
            interpolation=ImageInterpolation(payload["interpolation"]),
            value_scale=payload["value_scale"],
            mean=tuple(mean),
            std=tuple(std),
            maximum_source_width=payload["maximum_source_width"],
            maximum_source_height=payload["maximum_source_height"],
            maximum_source_pixels=payload["maximum_source_pixels"],
            allowed_source_modes=tuple(source_modes),
            decoder_version=payload["decoder_version"],
            allowed_formats=tuple(formats),
            tensor_dtype=payload["tensor_dtype"],
            decoder=payload["decoder"],
            exif_orientation=payload["exif_orientation"],
            alpha_policy=payload["alpha_policy"],
            icc_profile_policy=payload["icc_profile_policy"],
            gamma_policy=payload["gamma_policy"],
            operation_order=payload["operation_order"],
            resize_shortest_edge=payload.get("resize_shortest_edge"),
        )


@dataclass(frozen=True, slots=True)
class OnnxProviderOption:
    key: str
    value: str

    def __post_init__(self) -> None:
        _require_nonempty(self.key, "provider option key")
        _require_nonempty(self.value, "provider option value")

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnnxProviderOption:
        _require_exact_keys(
            payload,
            {"key", "value"},
            "ONNX provider option",
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class OnnxProviderSpec:
    name: str
    options: tuple[OnnxProviderOption, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "provider name")
        keys = tuple(item.key for item in self.options)
        if len(keys) != len(set(keys)):
            raise ValueError("ONNX provider option keys must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError("ONNX provider options must be key-sorted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "options": [item.to_dict() for item in self.options],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnnxProviderSpec:
        _require_exact_keys(
            payload,
            {"name", "options"},
            "ONNX provider specification",
        )
        options = payload["options"]
        if not isinstance(options, list):
            raise TypeError("ONNX provider options must be a list")
        return cls(
            name=payload["name"],
            options=tuple(
                OnnxProviderOption.from_dict(item) for item in options
            ),
        )


@dataclass(frozen=True, slots=True)
class OnnxRuntimeBackendConfig:
    preprocessing_config_sha256: str
    input_name: str
    output_name: str
    input_tensor_type: str
    output_tensor_type: str
    input_layout: ImageTensorLayout
    vector_dimension: int
    maximum_batch_size: int
    graph_optimization: OnnxGraphOptimization
    execution_mode: str
    intra_op_num_threads: int
    inter_op_num_threads: int
    allow_intra_op_spinning: bool
    allow_inter_op_spinning: bool
    enable_mem_pattern: bool
    enable_cpu_mem_arena: bool
    use_deterministic_compute: bool
    providers: tuple[OnnxProviderSpec, ...]
    maximum_model_bytes: int
    custom_op_libraries: tuple[str, ...] = ()
    disable_fallback: bool = True
    ort_load_config_from_model: str = "DISABLED"
    inference_api: str = "session_run_named_output"
    session_log_severity_level: int = 3
    schema_version: str = "cvi.onnx_runtime_backend_config.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.onnx_runtime_backend_config.v1":
            raise ValueError("unsupported ONNX backend config schema")
        _validate_sha256(
            self.preprocessing_config_sha256,
            "preprocessing_config_sha256",
        )
        _require_nonempty(self.input_name, "input_name")
        _require_nonempty(self.output_name, "output_name")
        if self.input_name == self.output_name:
            raise ValueError("ONNX input and output names must differ")
        if self.input_tensor_type != "tensor(float)":
            raise ValueError("initial ONNX input type is tensor(float)")
        if self.output_tensor_type != "tensor(float)":
            raise ValueError("initial ONNX output type is tensor(float)")
        _require_positive_int(self.vector_dimension, "vector_dimension")
        _require_positive_int(self.maximum_batch_size, "maximum_batch_size")
        _require_positive_int(self.maximum_model_bytes, "maximum_model_bytes")
        if self.execution_mode != "ORT_SEQUENTIAL":
            raise ValueError("initial ONNX execution mode is ORT_SEQUENTIAL")
        _require_positive_int(
            self.intra_op_num_threads,
            "intra_op_num_threads",
        )
        _require_positive_int(
            self.inter_op_num_threads,
            "inter_op_num_threads",
        )
        if not self.providers:
            raise ValueError("at least one ONNX provider is required")
        names = tuple(provider.name for provider in self.providers)
        if len(names) != len(set(names)):
            raise ValueError("ONNX providers must be unique")
        if self.disable_fallback is not True:
            raise ValueError("ONNX provider fallback must be disabled")
        if self.custom_op_libraries:
            raise ValueError(
                "initial ONNX backend rejects custom-op libraries"
            )
        if self.ort_load_config_from_model != "DISABLED":
            raise ValueError(
                "ORT model-embedded session config must be disabled"
            )
        if self.inference_api != "session_run_named_output":
            raise ValueError("initial ONNX inference API is session.run")
        if (
            isinstance(self.session_log_severity_level, bool)
            or not isinstance(self.session_log_severity_level, int)
            or not 0 <= self.session_log_severity_level <= 4
        ):
            raise ValueError("session log severity must be in [0, 4]")

    @property
    def config_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "preprocessing_config_sha256": (
                self.preprocessing_config_sha256
            ),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_tensor_type": self.input_tensor_type,
            "output_tensor_type": self.output_tensor_type,
            "input_layout": self.input_layout.value,
            "vector_dimension": self.vector_dimension,
            "maximum_batch_size": self.maximum_batch_size,
            "graph_optimization": self.graph_optimization.value,
            "execution_mode": self.execution_mode,
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "allow_intra_op_spinning": self.allow_intra_op_spinning,
            "allow_inter_op_spinning": self.allow_inter_op_spinning,
            "enable_mem_pattern": self.enable_mem_pattern,
            "enable_cpu_mem_arena": self.enable_cpu_mem_arena,
            "use_deterministic_compute": self.use_deterministic_compute,
            "providers": [provider.to_dict() for provider in self.providers],
            "maximum_model_bytes": self.maximum_model_bytes,
            "custom_op_libraries": list(self.custom_op_libraries),
            "disable_fallback": self.disable_fallback,
            "ort_load_config_from_model": (
                self.ort_load_config_from_model
            ),
            "inference_api": self.inference_api,
            "session_log_severity_level": (
                self.session_log_severity_level
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> OnnxRuntimeBackendConfig:
        _require_exact_keys(
            payload,
            {
                "schema_version",
                "preprocessing_config_sha256",
                "input_name",
                "output_name",
                "input_tensor_type",
                "output_tensor_type",
                "input_layout",
                "vector_dimension",
                "maximum_batch_size",
                "graph_optimization",
                "execution_mode",
                "intra_op_num_threads",
                "inter_op_num_threads",
                "allow_intra_op_spinning",
                "allow_inter_op_spinning",
                "enable_mem_pattern",
                "enable_cpu_mem_arena",
                "use_deterministic_compute",
                "providers",
                "maximum_model_bytes",
                "custom_op_libraries",
                "disable_fallback",
                "ort_load_config_from_model",
                "inference_api",
                "session_log_severity_level",
            },
            "ONNX runtime backend config",
        )
        providers = payload["providers"]
        custom_ops = payload["custom_op_libraries"]
        if not isinstance(providers, list) or not isinstance(
            custom_ops,
            list,
        ):
            raise TypeError(
                "ONNX providers and custom-op libraries must be lists"
            )
        return cls(
            schema_version=payload["schema_version"],
            preprocessing_config_sha256=payload[
                "preprocessing_config_sha256"
            ],
            input_name=payload["input_name"],
            output_name=payload["output_name"],
            input_tensor_type=payload["input_tensor_type"],
            output_tensor_type=payload["output_tensor_type"],
            input_layout=ImageTensorLayout(payload["input_layout"]),
            vector_dimension=payload["vector_dimension"],
            maximum_batch_size=payload["maximum_batch_size"],
            graph_optimization=OnnxGraphOptimization(
                payload["graph_optimization"]
            ),
            execution_mode=payload["execution_mode"],
            intra_op_num_threads=payload["intra_op_num_threads"],
            inter_op_num_threads=payload["inter_op_num_threads"],
            allow_intra_op_spinning=payload[
                "allow_intra_op_spinning"
            ],
            allow_inter_op_spinning=payload[
                "allow_inter_op_spinning"
            ],
            enable_mem_pattern=payload["enable_mem_pattern"],
            enable_cpu_mem_arena=payload["enable_cpu_mem_arena"],
            use_deterministic_compute=payload[
                "use_deterministic_compute"
            ],
            providers=tuple(
                OnnxProviderSpec.from_dict(item) for item in providers
            ),
            maximum_model_bytes=payload["maximum_model_bytes"],
            custom_op_libraries=tuple(custom_ops),
            disable_fallback=payload["disable_fallback"],
            ort_load_config_from_model=payload[
                "ort_load_config_from_model"
            ],
            inference_api=payload["inference_api"],
            session_log_severity_level=payload[
                "session_log_severity_level"
            ],
        )


class OnnxRuntimeCpuBackend:
    """CPU-only ORT reference with explicit provider and metadata checks."""

    def __init__(
        self,
        *,
        model_path: Path,
        config: OnnxRuntimeBackendConfig,
        preprocessing: ImagePreprocessingConfig,
    ) -> None:
        if config.preprocessing_config_sha256 != (
            preprocessing.config_sha256
        ):
            raise ValueError(
                "ONNX config and preprocessing semantics differ"
            )
        if config.input_layout is not preprocessing.layout:
            raise ValueError("ONNX input layout and preprocessing differ")
        expected_provider = (
            OnnxProviderSpec("CPUExecutionProvider"),
        )
        if config.providers != expected_provider:
            raise ValueError(
                "CPU reference requires CPUExecutionProvider only"
            )
        resolved_model = _regular_file(model_path, "ONNX model")
        reject_unverified_superanimal_onnx(resolved_model)
        if os.environ.get("ORT_LOAD_CONFIG_FROM_MODEL") not in {None, "0"}:
            raise RuntimeError(
                "ORT_LOAD_CONFIG_FROM_MODEL must be unset or 0"
            )
        np, _, ort, onnx, pillow_version = _optional_dependencies()
        if pillow_version != preprocessing.decoder_version:
            raise RuntimeError(
                "installed Pillow version differs from preprocessing config"
            )
        model_bytes, model_sha256 = _read_model_bytes(
            resolved_model,
            maximum_bytes=config.maximum_model_bytes,
        )
        reject_unverified_superanimal_onnx(
            resolved_model,
            model_sha256=model_sha256,
        )
        parsed_model = onnx.load_model_from_string(model_bytes)
        _reject_superanimal_graph_contract(
            resolved_model,
            parsed_model.graph,
            model_sha256,
        )
        _reject_external_data(parsed_model.graph, onnx)
        onnx.checker.check_model(parsed_model)
        del parsed_model
        available = tuple(ort.get_available_providers())
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is unavailable")

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel,
            config.graph_optimization.value,
        )
        options.intra_op_num_threads = config.intra_op_num_threads
        options.inter_op_num_threads = config.inter_op_num_threads
        options.enable_mem_pattern = config.enable_mem_pattern
        options.enable_cpu_mem_arena = config.enable_cpu_mem_arena
        options.use_deterministic_compute = (
            config.use_deterministic_compute
        )
        options.log_severity_level = config.session_log_severity_level
        options.add_session_config_entry(
            "session.intra_op.allow_spinning",
            "1" if config.allow_intra_op_spinning else "0",
        )
        options.add_session_config_entry(
            "session.inter_op.allow_spinning",
            "1" if config.allow_inter_op_spinning else "0",
        )
        providers = [provider.name for provider in config.providers]
        provider_options = [
            {item.key: item.value for item in provider.options}
            for provider in config.providers
        ]
        self._session = ort.InferenceSession(
            model_bytes,
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
            enable_fallback=0,
        )
        self._session.disable_fallback()
        self._actual_providers = tuple(self._session.get_providers())
        if self._actual_providers != tuple(providers):
            raise RuntimeError(
                "actual ONNX providers differ from frozen provider order"
            )
        self._actual_provider_options = self._session.get_provider_options()
        expected_provider_options = {
            provider.name: {
                item.key: item.value for item in provider.options
            }
            for provider in config.providers
        }
        if self._actual_provider_options != expected_provider_options:
            raise RuntimeError(
                "actual ONNX provider options differ from frozen options"
            )
        _validate_model_metadata(
            self._session,
            config,
            preprocessing,
        )
        self._np = np
        self._config = config
        self._preprocessing = preprocessing
        self._model_sha256 = model_sha256
        self._identity = EmbeddingBackendIdentity(
            backend_name="onnxruntime.cpu",
            backend_version="cvi.onnx_cpu_backend.v1",
            runtime_version=(
                f"onnxruntime={ort.__version__};onnx={onnx.__version__};"
                f"numpy={np.__version__};pillow={pillow_version}"
            ),
            execution_provider="CPUExecutionProvider",
            device="cpu",
            precision="fp32",
            determinism_mode=(
                "REQUESTED_NOT_PROVEN"
                if config.use_deterministic_compute
                else "NOT_REQUESTED"
            ),
            backend_config_sha256=config.config_sha256,
        )

    @property
    def identity(self) -> EmbeddingBackendIdentity:
        return self._identity

    @property
    def preprocessing_semantics_sha256(self) -> str:
        return self._preprocessing.config_sha256

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def actual_providers(self) -> tuple[str, ...]:
        return self._actual_providers

    @property
    def actual_provider_options(self) -> dict[str, dict[str, str]]:
        return {
            name: dict(options)
            for name, options in self._actual_provider_options.items()
        }

    def infer_batch(
        self,
        artifact_paths: tuple[Path, ...],
    ) -> tuple[memoryview, ...]:
        if not artifact_paths:
            raise ValueError("ONNX inference batch must not be empty")
        if len(artifact_paths) > self._config.maximum_batch_size:
            raise ValueError("ONNX inference batch exceeds frozen maximum")
        tensor = preprocess_image_batch(
            artifact_paths,
            self._preprocessing,
        )
        return self.infer_preprocessed_batch(tensor)

    def infer_preprocessed_batch(
        self,
        tensor: Any,
    ) -> tuple[memoryview, ...]:
        return _infer_preprocessed_batch(
            session=self._session,
            tensor=tensor,
            config=self._config,
            preprocessing=self._preprocessing,
            np=self._np,
        )

    def synchronize(self) -> None:
        return None

    def runtime_resources(self) -> EmbeddingRuntimeResources:
        return EmbeddingRuntimeResources.unavailable()


class OnnxRuntimeCudaBackend:
    """Full-graph CUDA reference with both ORT fallback layers disabled."""

    def __init__(
        self,
        *,
        model_path: Path,
        config: OnnxRuntimeBackendConfig,
        preprocessing: ImagePreprocessingConfig,
    ) -> None:
        if config.preprocessing_config_sha256 != (
            preprocessing.config_sha256
        ):
            raise ValueError(
                "ONNX config and preprocessing semantics differ"
            )
        if config.input_layout is not preprocessing.layout:
            raise ValueError("ONNX input layout and preprocessing differ")
        requested_options = _validate_cuda_provider_spec(config.providers)
        _reject_cuda_environment_overrides()
        distribution_name, distribution_version = (
            onnxruntime_distribution_identity(require_gpu=True)
        )
        resolved_model = _regular_file(model_path, "ONNX model")
        reject_unverified_superanimal_onnx(resolved_model)
        np, _, ort, onnx, pillow_version = _optional_dependencies()
        if pillow_version != preprocessing.decoder_version:
            raise RuntimeError(
                "installed Pillow version differs from preprocessing config"
            )
        if not hasattr(ort, "preload_dlls"):
            raise RuntimeError(
                "CUDA backend requires ONNX Runtime preload_dlls support"
            )
        ort.preload_dlls(cuda=True, cudnn=True, directory="")
        model_bytes, model_sha256 = _read_model_bytes(
            resolved_model,
            maximum_bytes=config.maximum_model_bytes,
        )
        reject_unverified_superanimal_onnx(
            resolved_model,
            model_sha256=model_sha256,
        )
        parsed_model = onnx.load_model_from_string(model_bytes)
        _reject_superanimal_graph_contract(
            resolved_model,
            parsed_model.graph,
            model_sha256,
        )
        _reject_external_data(parsed_model.graph, onnx)
        onnx.checker.check_model(parsed_model)
        del parsed_model
        available = tuple(ort.get_available_providers())
        if _CUDA_PROVIDER_NAME not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable")

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel,
            config.graph_optimization.value,
        )
        options.intra_op_num_threads = config.intra_op_num_threads
        options.inter_op_num_threads = config.inter_op_num_threads
        options.enable_mem_pattern = config.enable_mem_pattern
        options.enable_cpu_mem_arena = config.enable_cpu_mem_arena
        options.use_deterministic_compute = (
            config.use_deterministic_compute
        )
        options.log_severity_level = config.session_log_severity_level
        options.add_session_config_entry(
            "session.intra_op.allow_spinning",
            "1" if config.allow_intra_op_spinning else "0",
        )
        options.add_session_config_entry(
            "session.inter_op.allow_spinning",
            "1" if config.allow_inter_op_spinning else "0",
        )
        options.add_session_config_entry(
            "session.disable_cpu_ep_fallback",
            "1",
        )
        self._session = ort.InferenceSession(
            model_bytes,
            sess_options=options,
            providers=[_CUDA_PROVIDER_NAME],
            provider_options=[requested_options],
            enable_fallback=0,
        )
        self._session.disable_fallback()
        self._actual_providers = tuple(self._session.get_providers())
        if self._actual_providers != (
            _CUDA_PROVIDER_NAME,
            _CPU_PROVIDER_NAME,
        ):
            raise RuntimeError(
                "actual ONNX providers differ from strict CUDA session"
            )
        self._actual_provider_options = (
            self._session.get_provider_options()
        )
        expected_actual = {
            _CPU_PROVIDER_NAME: {},
            _CUDA_PROVIDER_NAME: {
                **_CUDA_ORT_127_ACTUAL_DEFAULTS,
                **requested_options,
            },
        }
        if self._actual_provider_options != expected_actual:
            raise RuntimeError(
                "actual CUDA provider options differ from frozen semantics"
            )
        _validate_model_metadata(
            self._session,
            config,
            preprocessing,
        )
        self._np = np
        self._config = config
        self._preprocessing = preprocessing
        self._model_sha256 = model_sha256
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
        self._identity = EmbeddingBackendIdentity(
            backend_name="onnxruntime.cuda.full_graph",
            backend_version="cvi.onnx_cuda_backend.v1",
            runtime_version=(
                f"distribution={distribution_name}=={distribution_version};"
                f"onnxruntime={ort.__version__};onnx={onnx.__version__};"
                f"numpy={np.__version__};pillow={pillow_version}"
            ),
            execution_provider=(
                "CUDAExecutionProvider_FULL_GRAPH_CPU_EP_DISABLED"
            ),
            device=(
                f"cuda:{requested_options['device_id']};"
                f"CUDA_VISIBLE_DEVICES={visible_devices}"
            ),
            precision="fp32_tf32_disabled",
            determinism_mode="REQUESTED_NOT_PROVEN",
            backend_config_sha256=config.config_sha256,
        )

    @property
    def identity(self) -> EmbeddingBackendIdentity:
        return self._identity

    @property
    def preprocessing_semantics_sha256(self) -> str:
        return self._preprocessing.config_sha256

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def actual_providers(self) -> tuple[str, ...]:
        return self._actual_providers

    @property
    def actual_provider_options(self) -> dict[str, dict[str, str]]:
        return {
            name: dict(options)
            for name, options in self._actual_provider_options.items()
        }

    def infer_batch(
        self,
        artifact_paths: tuple[Path, ...],
    ) -> tuple[memoryview, ...]:
        if not artifact_paths:
            raise ValueError("ONNX inference batch must not be empty")
        if len(artifact_paths) > self._config.maximum_batch_size:
            raise ValueError("ONNX inference batch exceeds frozen maximum")
        tensor = preprocess_image_batch(
            artifact_paths,
            self._preprocessing,
        )
        return self.infer_preprocessed_batch(tensor)

    def infer_preprocessed_batch(
        self,
        tensor: Any,
    ) -> tuple[memoryview, ...]:
        return _infer_preprocessed_batch(
            session=self._session,
            tensor=tensor,
            config=self._config,
            preprocessing=self._preprocessing,
            np=self._np,
        )

    def synchronize(self) -> None:
        # session.run returns only after its requested CPU output is available.
        return None

    def runtime_resources(self) -> EmbeddingRuntimeResources:
        return EmbeddingRuntimeResources.unavailable()


def preprocess_image_batch(
    artifact_paths: tuple[Path, ...],
    config: ImagePreprocessingConfig,
) -> Any:
    """Decode and deterministically construct one contiguous float32 tensor."""

    if not artifact_paths:
        raise ValueError("preprocessing batch must not be empty")
    np, Image, _, _, pillow_version = _optional_dependencies()
    if pillow_version != config.decoder_version:
        raise RuntimeError(
            "installed Pillow version differs from preprocessing config"
        )
    shape = (
        (len(artifact_paths), config.channels, config.height, config.width)
        if config.layout is ImageTensorLayout.NCHW
        else (len(artifact_paths), config.height, config.width, config.channels)
    )
    batch = np.empty(shape, dtype=np.float32)
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    interpolation = {
        ImageInterpolation.NEAREST: Image.Resampling.NEAREST,
        ImageInterpolation.BILINEAR: Image.Resampling.BILINEAR,
        ImageInterpolation.BICUBIC: Image.Resampling.BICUBIC,
    }[config.interpolation]
    for index, path in enumerate(artifact_paths):
        resolved = _regular_file(path, "embedding image")
        with Image.open(resolved) as image:
            if image.format not in config.allowed_formats:
                raise ValueError("embedding image format is not allowed")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("embedding image must contain one frame")
            if image.mode not in config.allowed_source_modes:
                raise ValueError("embedding image source mode is not allowed")
            if image.info.get("icc_profile"):
                raise ValueError("embedding image ICC profile is not allowed")
            if "transparency" in image.info:
                raise ValueError("embedding image alpha is not allowed")
            source_width, source_height = image.size
            if (
                source_width > config.maximum_source_width
                or source_height > config.maximum_source_height
            ):
                raise ValueError(
                    "embedding image exceeds source dimension cap"
                )
            if source_width * source_height > config.maximum_source_pixels:
                raise ValueError("embedding image exceeds source pixel cap")
            expected_size = (config.width, config.height)
            image = image.convert(config.color_mode)
            if config.resize_policy is ImageResizePolicy.EXACT:
                if image.size != expected_size:
                    raise ValueError("embedding image size differs from EXACT")
            elif config.resize_policy is ImageResizePolicy.STRETCH and (
                image.size != expected_size
            ):
                image = image.resize(
                    expected_size,
                    resample=interpolation,
                )
            elif config.resize_policy is ImageResizePolicy.SHORTEST_EDGE_CENTER_CROP:
                shortest_edge = config.resize_shortest_edge
                if shortest_edge is None:  # pragma: no cover - config validation
                    raise RuntimeError("shortest-edge preprocessing is incomplete")
                if source_width <= source_height:
                    resized_width = shortest_edge
                    resized_height = int(shortest_edge * source_height / source_width)
                else:
                    resized_height = shortest_edge
                    resized_width = int(shortest_edge * source_width / source_height)
                image = image.resize(
                    (resized_width, resized_height),
                    resample=interpolation,
                )
                left = (resized_width - config.width) // 2
                top = (resized_height - config.height) // 2
                image = image.crop(
                    (left, top, left + config.width, top + config.height)
                )
            array = np.asarray(image, dtype=np.float32)
        if config.color_mode == "L":
            array = array[..., np.newaxis]
        elif config.channel_order is ImageChannelOrder.BGR:
            array = array[..., ::-1]
        normalized = (
            array * np.float32(config.value_scale) - mean
        ) / std
        if config.layout is ImageTensorLayout.NCHW:
            normalized = np.transpose(normalized, (2, 0, 1))
        batch[index] = normalized
    return np.ascontiguousarray(batch)


def _expected_input_shape(
    preprocessing: ImagePreprocessingConfig,
) -> tuple[int, int, int]:
    if preprocessing.layout is ImageTensorLayout.NCHW:
        return (
            preprocessing.channels,
            preprocessing.height,
            preprocessing.width,
        )
    return (
        preprocessing.height,
        preprocessing.width,
        preprocessing.channels,
    )


def _validate_output_array(
    output: Any,
    *,
    np: Any,
    expected_shape: tuple[int, int],
) -> None:
    if not isinstance(output, np.ndarray):
        raise TypeError("ONNX embedding output must be a NumPy array")
    if output.shape != expected_shape:
        raise ValueError("ONNX embedding output shape mismatch")
    if (
        output.dtype != np.dtype(np.float32)
        or not output.dtype.isnative
    ):
        raise ValueError("ONNX embedding output dtype mismatch")
    if not output.flags.c_contiguous:
        raise ValueError("ONNX embedding output must be C-contiguous")


def _infer_preprocessed_batch(
    *,
    session: Any,
    tensor: Any,
    config: OnnxRuntimeBackendConfig,
    preprocessing: ImagePreprocessingConfig,
    np: Any,
) -> tuple[memoryview, ...]:
    if not isinstance(tensor, np.ndarray):
        raise TypeError("ONNX input tensor must be a NumPy array")
    expected_tail = _expected_input_shape(preprocessing)
    if tensor.ndim != len(expected_tail) + 1:
        raise ValueError("ONNX input tensor rank mismatch")
    batch_size = tensor.shape[0]
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or batch_size > config.maximum_batch_size
    ):
        raise ValueError("ONNX input tensor batch size is outside the cap")
    if tuple(tensor.shape[1:]) != expected_tail:
        raise ValueError("ONNX input tensor shape mismatch")
    if tensor.dtype != np.dtype(np.float32) or not tensor.dtype.isnative:
        raise ValueError("ONNX input tensor dtype mismatch")
    if not tensor.flags.c_contiguous:
        raise ValueError("ONNX input tensor must be C-contiguous")
    if not bool(np.isfinite(tensor).all()):
        raise ValueError("ONNX input tensor must contain only finite values")
    outputs = session.run(
        [config.output_name],
        {config.input_name: tensor},
    )
    if len(outputs) != 1:
        raise RuntimeError("ONNX session returned an unexpected output set")
    output = outputs[0]
    _validate_output_array(
        output,
        np=np,
        expected_shape=(batch_size, config.vector_dimension),
    )
    return tuple(memoryview(output[index]) for index in range(len(output)))


def _validate_model_metadata(
    session: Any,
    config: OnnxRuntimeBackendConfig,
    preprocessing: ImagePreprocessingConfig,
) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            "initial ONNX backend requires one input and one output"
        )
    model_input = inputs[0]
    model_output = outputs[0]
    if model_input.name != config.input_name:
        raise ValueError("ONNX model input name mismatch")
    if model_output.name != config.output_name:
        raise ValueError("ONNX model output name mismatch")
    if model_input.type != config.input_tensor_type:
        raise ValueError("ONNX model input dtype mismatch")
    if model_output.type != config.output_tensor_type:
        raise ValueError("ONNX model output dtype mismatch")
    _validate_dynamic_batch_shape(
        model_input.shape,
        _expected_input_shape(preprocessing),
        "input",
    )
    _validate_dynamic_batch_shape(
        model_output.shape,
        (config.vector_dimension,),
        "output",
    )


def _validate_cuda_provider_spec(
    providers: tuple[OnnxProviderSpec, ...],
) -> dict[str, str]:
    if len(providers) != 1 or providers[0].name != _CUDA_PROVIDER_NAME:
        raise ValueError(
            "CUDA reference requires CUDAExecutionProvider only"
        )
    options = {
        option.key: option.value for option in providers[0].options
    }
    if set(options) != _CUDA_SAFE_OPTION_KEYS:
        raise ValueError("CUDA provider option set is not the frozen safe set")
    fixed = {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "cudnn_conv_use_max_workspace": "0",
        "do_copy_in_default_stream": "1",
        "enable_cuda_graph": "0",
        "prefer_nhwc": "0",
        "use_ep_level_unified_stream": "0",
        "use_tf32": "0",
    }
    for key, expected in fixed.items():
        if options[key] != expected:
            raise ValueError(f"unsafe CUDA provider option: {key}")
    device_id = options["device_id"]
    memory_limit = options["gpu_mem_limit"]
    if not device_id.isascii() or not device_id.isdecimal():
        raise ValueError("CUDA device_id must be a non-negative integer")
    if not memory_limit.isascii() or not memory_limit.isdecimal():
        raise ValueError("CUDA gpu_mem_limit must be a positive integer")
    if int(memory_limit) <= 0:
        raise ValueError("CUDA gpu_mem_limit must be a positive integer")
    return options


def _reject_cuda_environment_overrides() -> None:
    for name in (
        "ORT_CUDA_TUNABLE_OP_ENABLE",
        "ORT_CUDA_TUNABLE_OP_TUNING_ENABLE",
    ):
        if os.environ.get(name) not in {None, "0"}:
            raise RuntimeError(f"{name} must be unset or 0")
    if os.environ.get("ORT_LOAD_CONFIG_FROM_MODEL") not in {None, "0"}:
        raise RuntimeError(
            "ORT_LOAD_CONFIG_FROM_MODEL must be unset or 0"
        )


def onnxruntime_distribution_identity(
    *,
    require_gpu: bool,
) -> tuple[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    installed: list[tuple[str, str]] = []
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            installed.append((name, version(name)))
        except PackageNotFoundError:
            continue
    if len(installed) != 1:
        raise RuntimeError(
            "exactly one ONNX Runtime distribution must be installed"
        )
    if require_gpu and installed[0][0] != "onnxruntime-gpu":
        raise RuntimeError(
            "CUDA backend requires the onnxruntime-gpu distribution"
        )
    if not require_gpu and installed[0][0] != "onnxruntime":
        raise RuntimeError(
            "CPU worker requires the onnxruntime distribution"
        )
    return installed[0]


def _validate_dynamic_batch_shape(
    observed: list[Any],
    expected_tail: tuple[int, ...],
    name: str,
) -> None:
    if len(observed) != len(expected_tail) + 1:
        raise ValueError(f"ONNX model {name} rank mismatch")
    if isinstance(observed[0], int):
        raise ValueError(f"ONNX model {name} batch axis must be dynamic")
    for actual, expected in zip(observed[1:], expected_tail, strict=True):
        if actual != expected:
            raise ValueError(f"ONNX model {name} shape mismatch")


def _read_model_bytes(
    path: Path,
    *,
    maximum_bytes: int,
) -> tuple[bytes, str]:
    before = path.stat()
    if before.st_size > maximum_bytes:
        raise ValueError("ONNX model exceeds frozen byte cap")
    with path.open("rb") as stream:
        payload = stream.read(maximum_bytes + 1)
    after = path.stat()
    if len(payload) > maximum_bytes:
        raise ValueError("ONNX model exceeds frozen byte cap")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
        or len(payload) != before.st_size
    ):
        raise ValueError("ONNX model changed while being loaded")
    return payload, sha256(payload).hexdigest()


def _reject_superanimal_graph_contract(
    model_path: Path,
    graph: Any,
    model_sha256: str,
) -> None:
    if len(graph.input) != 1 or len(graph.output) != 1:
        return

    def shape(value_info: Any) -> tuple[object, ...]:
        dimensions: list[object] = []
        for dimension in value_info.type.tensor_type.shape.dim:
            if dimension.dim_value > 0:
                dimensions.append(int(dimension.dim_value))
            elif dimension.dim_param:
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
        return tuple(dimensions)

    reject_unverified_superanimal_onnx(
        model_path,
        model_sha256=model_sha256,
        input_shape=shape(graph.input[0]),
        output_shape=shape(graph.output[0]),
    )


def _reject_external_data(graph: Any, onnx: Any) -> None:
    def reject_tensor(tensor: Any) -> None:
        if (
            tensor.data_location == onnx.TensorProto.EXTERNAL
            or len(tensor.external_data) != 0
        ):
            raise ValueError(
                "ONNX external-data tensors are not supported"
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


def _optional_dependencies() -> tuple[Any, Any, Any, Any, str]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import PIL
        from PIL import Image
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX CPU backend requires `uv sync --extra cpu`"
        ) from error
    return np, Image, ort, onnx, PIL.__version__


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_sorted_strings(
    values: tuple[str, ...],
    name: str,
) -> None:
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be sorted and unique")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")


def _validate_sorted_uppercase_strings(
    values: tuple[str, ...],
    name: str,
) -> None:
    _validate_sorted_strings(values, name)
    if any(item != item.upper() for item in values):
        raise ValueError(f"{name} must contain uppercase strings")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")


def _require_finite_positive(value: float, name: str) -> None:
    _require_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
