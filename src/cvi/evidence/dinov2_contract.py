"""Receipt and local-artifact validation for DINOv2-small inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from cvi.pretrained_supporting_asset_intake import (
    PretrainedSupportingAssetIntakeReceipt,
    PretrainedSupportingAssetKind,
    PretrainedSupportingAssetSourceContract,
    parse_bounded_strict_json_object,
    validate_pretrained_supporting_asset_receipt_binding,
)
from cvi.pretrained_weight_intake import (
    PretrainedWeightFileFormat,
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
    PretrainedWeightUsageLane,
    validate_pretrained_weight_receipt_binding,
)
from cvi.protected_io import read_strict_json_object
from cvi.provenance import content_sha256
from cvi.retained_file import read_retained_regular_file


_WEIGHT_BUNDLE_KEYS = {
    "schema_version",
    "source_contract_sha256",
    "source_contract",
    "receipt_sha256",
    "receipt",
    "tool_provenance",
    "tool_provenance_sha256",
}
_ASSET_BUNDLE_KEYS = _WEIGHT_BUNDLE_KEYS
_PREPROCESSOR_KEYS = {
    "crop_size",
    "do_center_crop",
    "do_convert_rgb",
    "do_normalize",
    "do_rescale",
    "do_resize",
    "image_mean",
    "image_processor_type",
    "image_std",
    "resample",
    "rescale_factor",
    "size",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ONNX_MANIFEST_KEYS = {
    "schema_version",
    "model_id",
    "source_revision",
    "source_weights_sha256",
    "weight_intake_receipt_sha256",
    "preprocessor_sha256",
    "preprocessor_intake_receipt_sha256",
    "config_sha256",
    "preprocessing_config_sha256",
    "onnx_sha256",
    "onnx_bytes",
    "input_name",
    "input_shape",
    "output_name",
    "output_shape",
    "output_semantics",
    "external_data",
    "usage_lane",
    "license_id",
    "parity_receipt_sha256",
}


@dataclass(frozen=True, slots=True)
class Dinov2LocalArtifactContract:
    """Validated paths and receipt bindings for one local DINOv2-small copy."""

    model_directory: Path
    weight_source: PretrainedWeightSourceContract
    weight_receipt: PretrainedWeightIntakeReceipt
    preprocessor_source: PretrainedSupportingAssetSourceContract
    preprocessor_receipt: PretrainedSupportingAssetIntakeReceipt
    preprocessor: dict[str, Any]
    config: dict[str, Any]
    config_sha256: str

    @classmethod
    def load(
        cls,
        *,
        model_directory: Path,
        weight_intake_bundle: Path,
        preprocessor_intake_bundle: Path,
    ) -> Dinov2LocalArtifactContract:
        root = _require_local_model_directory(model_directory)
        weight_source, weight_receipt = _read_weight_bundle(
            weight_intake_bundle
        )
        preprocessor_source, preprocessor_receipt = _read_preprocessor_bundle(
            preprocessor_intake_bundle
        )
        _validate_source_pair(
            weight_source,
            weight_receipt,
            preprocessor_source,
            preprocessor_receipt,
        )
        preprocessor, config, config_sha256 = _validate_local_files(
            root,
            weight_source,
            weight_receipt,
            preprocessor_source,
            preprocessor_receipt,
        )
        return cls(
            model_directory=root,
            weight_source=weight_source,
            weight_receipt=weight_receipt,
            preprocessor_source=preprocessor_source,
            preprocessor_receipt=preprocessor_receipt,
            preprocessor=preprocessor,
            config=config,
            config_sha256=config_sha256,
        )

    @property
    def model_sha256(self) -> str:
        return self.weight_receipt.weight_sha256

    @property
    def preprocessor_sha256(self) -> str:
        return self.preprocessor_receipt.asset_sha256

    @property
    def weight_receipt_sha256(self) -> str:
        return self.weight_receipt.receipt_sha256

    @property
    def preprocessor_receipt_sha256(self) -> str:
        return self.preprocessor_receipt.receipt_sha256

    def revalidate_local_files(self) -> None:
        """Recheck admitted bytes immediately before framework loading."""

        preprocessor, config, config_sha256 = _validate_local_files(
            self.model_directory,
            self.weight_source,
            self.weight_receipt,
            self.preprocessor_source,
            self.preprocessor_receipt,
        )
        if (
            preprocessor != self.preprocessor
            or config != self.config
            or config_sha256 != self.config_sha256
        ):
            raise RuntimeError("DINOv2 local JSON changed after contract loading")


@dataclass(frozen=True, slots=True)
class Dinov2OnnxArtifactManifest:
    """Exact deployment-candidate binding for one self-contained export."""

    source_revision: str
    source_weights_sha256: str
    weight_intake_receipt_sha256: str
    preprocessor_sha256: str
    preprocessor_intake_receipt_sha256: str
    config_sha256: str
    preprocessing_config_sha256: str
    onnx_sha256: str
    onnx_bytes: int
    usage_lane: str
    license_id: str
    parity_receipt_sha256: str
    schema_version: str = "cvi.dinov2_onnx_artifact_manifest.v1"
    model_id: str = "facebook/dinov2-small"
    input_name: str = "images"
    input_shape: tuple[object, ...] = ("batch", 3, 224, 224)
    output_name: str = "embedding"
    output_shape: tuple[object, ...] = ("batch", 384)
    output_semantics: str = "POOLER_OUTPUT_L2_NORMALIZED"
    external_data: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.dinov2_onnx_artifact_manifest.v1":
            raise ValueError("unsupported DINOv2 ONNX manifest schema")
        if self.model_id != "facebook/dinov2-small":
            raise ValueError("DINOv2 ONNX model_id differs")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("DINOv2 source_revision must be non-empty")
        for name in (
            "source_weights_sha256",
            "weight_intake_receipt_sha256",
            "preprocessor_sha256",
            "preprocessor_intake_receipt_sha256",
            "config_sha256",
            "preprocessing_config_sha256",
            "onnx_sha256",
            "parity_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            isinstance(self.onnx_bytes, bool)
            or not isinstance(self.onnx_bytes, int)
            or self.onnx_bytes <= 0
        ):
            raise ValueError("DINOv2 onnx_bytes must be a positive integer")
        if self.input_name != "images" or self.output_name != "embedding":
            raise ValueError("DINOv2 ONNX tensor names differ")
        if self.input_shape != ("batch", 3, 224, 224):
            raise ValueError("DINOv2 ONNX input shape differs")
        if self.output_shape != ("batch", 384):
            raise ValueError("DINOv2 ONNX output shape differs")
        if self.output_semantics != "POOLER_OUTPUT_L2_NORMALIZED":
            raise ValueError("DINOv2 ONNX output semantics differ")
        if self.external_data is not False:
            raise ValueError("DINOv2 ONNX external_data must be false")
        if self.usage_lane not in {"RESEARCH_ONLY", "DEPLOYMENT_CANDIDATE"}:
            raise ValueError("DINOv2 ONNX usage lane differs")
        if not isinstance(self.license_id, str) or not self.license_id.strip():
            raise ValueError("DINOv2 ONNX license_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "source_weights_sha256": self.source_weights_sha256,
            "weight_intake_receipt_sha256": self.weight_intake_receipt_sha256,
            "preprocessor_sha256": self.preprocessor_sha256,
            "preprocessor_intake_receipt_sha256": (
                self.preprocessor_intake_receipt_sha256
            ),
            "config_sha256": self.config_sha256,
            "preprocessing_config_sha256": self.preprocessing_config_sha256,
            "onnx_sha256": self.onnx_sha256,
            "onnx_bytes": self.onnx_bytes,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "output_name": self.output_name,
            "output_shape": list(self.output_shape),
            "output_semantics": self.output_semantics,
            "external_data": self.external_data,
            "usage_lane": self.usage_lane,
            "license_id": self.license_id,
            "parity_receipt_sha256": self.parity_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Dinov2OnnxArtifactManifest:
        if not isinstance(payload, dict) or set(payload) != _ONNX_MANIFEST_KEYS:
            raise ValueError("DINOv2 ONNX manifest keys differ")
        input_shape = payload["input_shape"]
        output_shape = payload["output_shape"]
        if not isinstance(input_shape, list) or not isinstance(output_shape, list):
            raise TypeError("DINOv2 ONNX manifest shapes must be arrays")
        values = dict(payload)
        values["input_shape"] = tuple(input_shape)
        values["output_shape"] = tuple(output_shape)
        return cls(**values)


def dinov2_image_preprocessing_config(
    contract: Dinov2LocalArtifactContract,
    *,
    decoder_version: str,
    maximum_source_width: int = 16_384,
    maximum_source_height: int = 16_384,
    maximum_source_pixels: int = 67_108_864,
) -> Any:
    """Translate the admitted Hugging Face processor into the ONNX contract."""

    from cvi.onnx_backend import (
        ImageChannelOrder,
        ImageInterpolation,
        ImagePreprocessingConfig,
        ImageResizePolicy,
        ImageTensorLayout,
    )

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


def _read_weight_bundle(
    path: Path,
) -> tuple[PretrainedWeightSourceContract, PretrainedWeightIntakeReceipt]:
    bundle = read_strict_json_object(path)
    if (
        set(bundle) != _WEIGHT_BUNDLE_KEYS
        or bundle["schema_version"]
        != "cvi.pretrained_weight_intake_bundle.v1"
    ):
        raise ValueError("DINOv2 weight intake bundle schema differs")
    source = PretrainedWeightSourceContract.from_dict(bundle["source_contract"])
    receipt = PretrainedWeightIntakeReceipt.from_dict(bundle["receipt"])
    _validate_bundle_hashes(bundle, source.contract_sha256, receipt.receipt_sha256)
    validate_pretrained_weight_receipt_binding(receipt, source)
    return source, receipt


def _read_preprocessor_bundle(
    path: Path,
) -> tuple[
    PretrainedSupportingAssetSourceContract,
    PretrainedSupportingAssetIntakeReceipt,
]:
    bundle = read_strict_json_object(path)
    if (
        set(bundle) != _ASSET_BUNDLE_KEYS
        or bundle["schema_version"]
        != "cvi.pretrained_supporting_asset_intake_bundle.v1"
    ):
        raise ValueError("DINOv2 preprocessor intake bundle schema differs")
    source = PretrainedSupportingAssetSourceContract.from_dict(
        bundle["source_contract"]
    )
    receipt = PretrainedSupportingAssetIntakeReceipt.from_dict(bundle["receipt"])
    _validate_bundle_hashes(bundle, source.contract_sha256, receipt.receipt_sha256)
    validate_pretrained_supporting_asset_receipt_binding(receipt, source)
    return source, receipt


def _validate_bundle_hashes(
    bundle: dict[str, Any],
    source_sha256: str,
    receipt_sha256: str,
) -> None:
    if bundle["source_contract_sha256"] != source_sha256:
        raise ValueError("DINOv2 intake bundle source contract hash differs")
    if bundle["receipt_sha256"] != receipt_sha256:
        raise ValueError("DINOv2 intake bundle receipt hash differs")
    provenance = bundle["tool_provenance"]
    if not isinstance(provenance, dict) or (
        content_sha256(provenance) != bundle["tool_provenance_sha256"]
    ):
        raise ValueError("DINOv2 intake bundle tool provenance hash differs")


def _validate_source_pair(
    weight_source: PretrainedWeightSourceContract,
    weight_receipt: PretrainedWeightIntakeReceipt,
    preprocessor_source: PretrainedSupportingAssetSourceContract,
    preprocessor_receipt: PretrainedSupportingAssetIntakeReceipt,
) -> None:
    if weight_source.source_model_id != "facebook/dinov2-small":
        raise ValueError("DINOv2 weight source model ID differs")
    if weight_source.weight_filename != "model.safetensors" or (
        weight_source.file_format is not PretrainedWeightFileFormat.SAFETENSORS
    ):
        raise ValueError("DINOv2-small requires model.safetensors")
    if preprocessor_source.source_model_id != weight_source.source_model_id:
        raise ValueError("DINOv2 preprocessor source model ID differs")
    if preprocessor_source.source_revision != weight_source.source_revision:
        raise ValueError("DINOv2 preprocessor source revision differs")
    if (
        preprocessor_source.asset_filename != "preprocessor_config.json"
        or preprocessor_source.asset_kind
        is not PretrainedSupportingAssetKind.PREPROCESSOR_CONFIG
    ):
        raise ValueError("DINOv2 preprocessor source kind or filename differs")
    expected_weight_receipt = weight_receipt.receipt_sha256
    if (
        preprocessor_source.associated_pretrained_weight_receipt_sha256
        != expected_weight_receipt
        or preprocessor_receipt.associated_pretrained_weight_receipt_sha256
        != expected_weight_receipt
    ):
        raise ValueError("DINOv2 preprocessor weight receipt binding differs")
    if (
        weight_receipt.admitted_lane is PretrainedWeightUsageLane.RESEARCH_ONLY
        and preprocessor_receipt.admitted_lane
        is PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
    ):
        raise ValueError(
            "research-only DINOv2 weight cannot bind a deployment preprocessor"
        )


def _validate_local_files(
    root: Path,
    weight_source: PretrainedWeightSourceContract,
    weight_receipt: PretrainedWeightIntakeReceipt,
    preprocessor_source: PretrainedSupportingAssetSourceContract,
    preprocessor_receipt: PretrainedSupportingAssetIntakeReceipt,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    weight_result = read_retained_regular_file(
        root / "model.safetensors",
        expected_bytes=weight_source.expected_file_bytes,
        expected_sha256=weight_source.expected_sha256,
        capture_payload=False,
        subject="DINOv2 model.safetensors",
    )
    if (
        weight_result.sha256 != weight_receipt.weight_sha256
        or weight_result.byte_count != weight_receipt.weight_bytes
    ):
        raise ValueError("DINOv2 weight bytes differ from intake receipt")

    preprocessor_result = read_retained_regular_file(
        root / "preprocessor_config.json",
        expected_bytes=preprocessor_source.expected_file_bytes,
        expected_sha256=preprocessor_source.expected_sha256,
        maximum_bytes=4_194_304,
        capture_payload=True,
        subject="DINOv2 preprocessor_config.json",
    )
    if (
        preprocessor_result.sha256 != preprocessor_receipt.asset_sha256
        or preprocessor_result.byte_count != preprocessor_receipt.asset_bytes
    ):
        raise ValueError("DINOv2 preprocessor bytes differ from intake receipt")
    if preprocessor_result.payload is None:  # pragma: no cover - helper contract
        raise RuntimeError("DINOv2 preprocessor payload was not retained")
    preprocessor = parse_bounded_strict_json_object(preprocessor_result.payload)
    if content_sha256(preprocessor) != preprocessor_receipt.json_structure_sha256:
        raise ValueError("DINOv2 preprocessor JSON structure hash differs")
    _validate_preprocessor(preprocessor)

    config_result = read_retained_regular_file(
        root / "config.json",
        maximum_bytes=4_194_304,
        capture_payload=True,
        subject="DINOv2 config.json",
    )
    if config_result.payload is None:  # pragma: no cover - helper contract
        raise RuntimeError("DINOv2 config payload was not retained")
    config = parse_bounded_strict_json_object(config_result.payload)
    _validate_model_config(config)
    return preprocessor, config, config_result.sha256


def _validate_preprocessor(value: dict[str, Any]) -> None:
    expected = {
        "crop_size": {"height": 224, "width": 224},
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_processor_type": "BitImageProcessor",
        "image_std": [0.229, 0.224, 0.225],
        "resample": 3,
        "rescale_factor": 1 / 255,
        "size": {"shortest_edge": 256},
    }
    if (
        set(value) != _PREPROCESSOR_KEYS
        or content_sha256(value) != content_sha256(expected)
    ):
        raise ValueError("DINOv2 admitted preprocessor structure differs")


def _validate_model_config(value: dict[str, Any]) -> None:
    expected = {
        "model_type": "dinov2",
        "architectures": ["Dinov2Model"],
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 384,
        "image_size": 518,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "patch_size": 14,
        "num_channels": 3,
        "num_attention_heads": 6,
        "num_hidden_layers": 12,
        "qkv_bias": True,
        "use_swiglu_ffn": False,
    }
    for field, required in expected.items():
        observed = value.get(field)
        if content_sha256(observed) != content_sha256(required):
            raise ValueError(f"DINOv2-small config field {field} differs")


def _require_local_model_directory(path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("DINOv2 model_directory must be a local directory")
    return root


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA256 digest")


__all__ = [
    "Dinov2LocalArtifactContract",
    "Dinov2OnnxArtifactManifest",
    "dinov2_image_preprocessing_config",
]
