"""Produce or independently verify receipt-bound DINOv2 ONNX parity."""

from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from contracts.dinov2_contract import (
    Dinov2LocalArtifactContract,
    Dinov2OnnxArtifactManifest,
)
from contracts.model_parity import (
    ModelUsageLane,
    ParityThresholds,
    load_model_parity_receipt,
    validate_parity_binding,
)
from contracts.intake.pretrained_supporting_asset_intake import (
    parse_bounded_strict_json_object,
)
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from systems.inference.onnx_backend import ImagePreprocessingConfig
from workflows.export_pretrained_to_onnx import export_dinov2_small


def verify_dinov2_bundle(
    *,
    artifact_path: Path,
    preprocessing_path: Path,
    parity_receipt_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
    public_production: bool,
) -> Dinov2OnnxArtifactManifest:
    manifest_result = read_retained_regular_file(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        maximum_bytes=1_048_576,
        capture_payload=True,
        subject="DINOv2 ONNX manifest",
    )
    if manifest_result.payload is None:  # pragma: no cover - helper contract
        raise RuntimeError("DINOv2 manifest payload was not retained")
    manifest = Dinov2OnnxArtifactManifest.from_dict(
        parse_bounded_strict_json_object(manifest_result.payload)
    )
    artifact = read_retained_regular_file(
        artifact_path,
        expected_bytes=manifest.onnx_bytes,
        expected_sha256=manifest.onnx_sha256,
        maximum_bytes=manifest.onnx_bytes,
        capture_payload=True,
        subject="DINOv2 ONNX artifact",
    )
    if artifact.payload is None:  # pragma: no cover - helper contract
        raise RuntimeError("DINOv2 ONNX payload was not retained")
    _validate_self_contained_onnx(artifact.payload)

    preprocessing_result = read_retained_regular_file(
        preprocessing_path,
        maximum_bytes=1_048_576,
        capture_payload=True,
        subject="DINOv2 preprocessing config",
    )
    if preprocessing_result.payload is None:  # pragma: no cover
        raise RuntimeError("DINOv2 preprocessing payload was not retained")
    preprocessing_payload = parse_bounded_strict_json_object(
        preprocessing_result.payload
    )
    preprocessing = ImagePreprocessingConfig.from_dict(preprocessing_payload)
    if (
        content_sha256(preprocessing_payload)
        != manifest.preprocessing_config_sha256
        or preprocessing.config_sha256 != manifest.preprocessing_config_sha256
    ):
        raise RuntimeError("DINOv2 preprocessing config binding differs")

    source = Dinov2LocalArtifactContract.load(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )
    expected_source = {
        "source_revision": source.weight_source.source_revision,
        "source_weights_sha256": source.model_sha256,
        "weight_intake_receipt_sha256": source.weight_receipt_sha256,
        "preprocessor_sha256": source.preprocessor_sha256,
        "preprocessor_intake_receipt_sha256": (
            source.preprocessor_receipt_sha256
        ),
        "config_sha256": source.config_sha256,
        "usage_lane": source.weight_receipt.admitted_lane.value,
        "license_id": source.weight_source.license_id,
    }
    for name, expected in expected_source.items():
        if getattr(manifest, name) != expected:
            raise RuntimeError(f"DINOv2 manifest {name} differs from local intake")

    parity = load_model_parity_receipt(
        parity_receipt_path,
        expected_sha256=manifest.parity_receipt_sha256,
    )
    validate_parity_binding(
        parity,
        model_id=manifest.model_id,
        artifact_sha256=manifest.onnx_sha256,
        source_weights_sha256=manifest.source_weights_sha256,
        preprocessing_sha256=manifest.preprocessing_config_sha256,
        usage_lane=ModelUsageLane(manifest.usage_lane),
        weight_intake_receipt_sha256=manifest.weight_intake_receipt_sha256,
        preprocessor_intake_receipt_sha256=(
            manifest.preprocessor_intake_receipt_sha256
        ),
        public_production=public_production,
    )
    _cpu_smoke(artifact.payload)
    return manifest


def _validate_self_contained_onnx(payload: bytes) -> None:
    import onnx

    model = onnx.load_model_from_string(payload)
    onnx.checker.check_model(model)

    def check_graph(graph: object) -> None:
        def check_tensor(tensor: object) -> None:
            if (
                tensor.data_location == onnx.TensorProto.EXTERNAL
                or tensor.external_data
            ):
                raise RuntimeError("DINOv2 ONNX uses external tensor data")

        for tensor in graph.initializer:
            check_tensor(tensor)
        for sparse in graph.sparse_initializer:
            check_tensor(sparse.values)
            check_tensor(sparse.indices)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    check_tensor(attribute.t)
                for tensor in attribute.tensors:
                    check_tensor(tensor)
                if attribute.HasField("sparse_tensor"):
                    check_tensor(attribute.sparse_tensor.values)
                    check_tensor(attribute.sparse_tensor.indices)
                for sparse in attribute.sparse_tensors:
                    check_tensor(sparse.values)
                    check_tensor(sparse.indices)
                if attribute.HasField("g"):
                    check_graph(attribute.g)
                for nested in attribute.graphs:
                    check_graph(nested)

    check_graph(model.graph)


def _cpu_smoke(payload: bytes) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].name != "images"
        or outputs[0].name != "embedding"
    ):
        raise RuntimeError("DINOv2 ONNX runtime metadata differs")
    result = session.run(
        ["embedding"],
        {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)},
    )[0]
    if result.shape != (1, 384) or result.dtype != np.float32:
        raise RuntimeError("DINOv2 ONNX CPU smoke output differs")
    if not bool(np.isfinite(result).all()):
        raise RuntimeError("DINOv2 ONNX CPU smoke output is non-finite")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--weight-intake-bundle", required=True, type=Path)
    parser.add_argument("--preprocessor-intake-bundle", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce-dinov2")
    _add_source_arguments(produce)
    produce.add_argument("--output-dir", required=True, type=Path)
    produce.add_argument("--artifact-stem", required=True)
    produce.add_argument("--maximum-absolute-error", type=float, default=1e-4)
    produce.add_argument("--maximum-relative-error", type=float, default=1e-2)
    produce.add_argument("--relative-error-floor", type=float, default=1e-4)
    produce.add_argument(
        "--minimum-cosine-similarity", type=float, default=0.999999
    )
    produce.add_argument("--crop-export-receipt", type=Path)
    produce.add_argument("--expected-crop-export-receipt-sha256")
    produce.add_argument("--crop-root", type=Path)
    produce.add_argument("--crop-token", action="append", default=[])

    verify = subparsers.add_parser("verify-dinov2")
    _add_source_arguments(verify)
    verify.add_argument("--artifact", required=True, type=Path)
    verify.add_argument("--preprocessing", required=True, type=Path)
    verify.add_argument("--parity-receipt", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.add_argument("--public-production", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "produce-dinov2":
        outputs = export_dinov2_small(
            model_directory=args.model_directory,
            weight_intake_bundle=args.weight_intake_bundle,
            preprocessor_intake_bundle=args.preprocessor_intake_bundle,
            output_directory=args.output_dir,
            artifact_stem=args.artifact_stem,
            thresholds=ParityThresholds(
                args.maximum_absolute_error,
                args.maximum_relative_error,
                args.relative_error_floor,
                args.minimum_cosine_similarity,
            ),
            crop_export_receipt=args.crop_export_receipt,
            expected_crop_export_receipt_sha256=(
                args.expected_crop_export_receipt_sha256
            ),
            crop_root=args.crop_root,
            crop_tokens=tuple(args.crop_token),
        )
        for path in outputs:
            print(
                f"{path} sha256={sha256(path.read_bytes()).hexdigest()} "
                f"bytes={path.stat().st_size}"
            )
        return
    manifest = verify_dinov2_bundle(
        artifact_path=args.artifact,
        preprocessing_path=args.preprocessing,
        parity_receipt_path=args.parity_receipt,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        model_directory=args.model_directory,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        public_production=args.public_production,
    )
    print(
        f"PASS model={manifest.model_id} artifact_sha256={manifest.onnx_sha256} "
        f"lane={manifest.usage_lane}"
    )


if __name__ == "__main__":
    main()
