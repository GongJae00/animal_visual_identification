"""Offline, receipt-bound DINOv2-small ONNX export and parity production."""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import PIL
import torch
from PIL import Image

from contracts.dinov2_contract import (
    Dinov2LocalArtifactContract,
    Dinov2OnnxArtifactManifest,
)
from contracts.model_parity import (
    ModelParityReceipt,
    ModelUsageLane,
    ParityFixtureKind,
    ParityFixtureResult,
    ParityThresholds,
)
from contracts.intake.pretrained_supporting_asset_intake import (
    parse_bounded_strict_json_object,
)
from data.crop_export import CropExportReceipt
from foundation.protected_io import read_strict_json_object
from foundation.provenance import content_sha256
from foundation.retained_file import read_retained_regular_file
from systems.inference.onnx_backend import (
    dinov2_image_preprocessing_config,
    preprocess_image_batch,
)


class _Dinov2ExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=images).pooler_output
        return torch.nn.functional.normalize(output.float(), p=2, dim=1)


def export_dinov2_small(
    *,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
    output_directory: Path,
    artifact_stem: str,
    thresholds: ParityThresholds,
    crop_export_receipt: Path | None = None,
    expected_crop_export_receipt_sha256: str | None = None,
    crop_root: Path | None = None,
    crop_tokens: tuple[str, ...] = (),
) -> tuple[Path, Path, Path, Path]:
    """Export and atomically publish a no-overwrite DINO artifact bundle."""

    output_root = output_directory.resolve(strict=True)
    if not output_root.is_dir() or output_directory.is_symlink():
        raise ValueError("output_directory must be an existing local directory")
    if (
        not artifact_stem
        or artifact_stem in {".", ".."}
        or Path(artifact_stem).name != artifact_stem
    ):
        raise ValueError("artifact_stem must be one safe path component")
    targets = (
        output_root / f"{artifact_stem}.onnx",
        output_root / f"{artifact_stem}.preprocessing.json",
        output_root / f"{artifact_stem}.parity.json",
        output_root / f"{artifact_stem}.manifest.json",
    )
    existing = tuple(path for path in targets if path.exists() or path.is_symlink())
    if existing:
        raise FileExistsError(
            "refusing to overwrite artifact bundle: "
            + ", ".join(str(path) for path in existing)
        )

    contract = Dinov2LocalArtifactContract.load(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )
    preprocessing = dinov2_image_preprocessing_config(
        contract,
        decoder_version=PIL.__version__,
    )
    panel_receipt_sha256, panel_paths = _load_crop_panel(
        crop_export_receipt=crop_export_receipt,
        expected_receipt_sha256=expected_crop_export_receipt_sha256,
        crop_root=crop_root,
        crop_tokens=crop_tokens,
    )

    contract.revalidate_local_files()
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(
        str(contract.model_directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("local DINOv2 loader did not return a torch module")
    contract.revalidate_local_files()
    model.to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    wrapper = _Dinov2ExportWrapper(model).eval()

    with TemporaryDirectory(prefix=".cvi-dinov2-export-", dir=output_root) as temp:
        stage = Path(temp)
        staged_model = stage / targets[0].name
        staged_preprocessing = stage / targets[1].name
        staged_parity = stage / targets[2].name
        staged_manifest = stage / targets[3].name
        dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy,),
                staged_model,
                input_names=["images"],
                output_names=["embedding"],
                dynamic_axes={
                    "images": {0: "batch"},
                    "embedding": {0: "batch"},
                },
                opset_version=18,
                external_data=False,
                dynamo=False,
            )
        model_sha256, model_bytes = _validate_onnx(staged_model)
        _write_json(staged_preprocessing, preprocessing.to_dict())

        parity = _produce_parity(
            wrapper=wrapper,
            onnx_path=staged_model,
            artifact_sha256=model_sha256,
            contract=contract,
            preprocessing=preprocessing,
            thresholds=thresholds,
            panel_receipt_sha256=panel_receipt_sha256,
            panel_paths=panel_paths,
            temporary_root=stage,
        )
        _write_json(staged_parity, parity.to_dict())
        parity_file_sha256 = _sha256_file(staged_parity)
        manifest = Dinov2OnnxArtifactManifest(
            source_revision=contract.weight_source.source_revision,
            source_weights_sha256=contract.model_sha256,
            weight_intake_receipt_sha256=contract.weight_receipt_sha256,
            preprocessor_sha256=contract.preprocessor_sha256,
            preprocessor_intake_receipt_sha256=(
                contract.preprocessor_receipt_sha256
            ),
            config_sha256=contract.config_sha256,
            preprocessing_config_sha256=preprocessing.config_sha256,
            onnx_sha256=model_sha256,
            onnx_bytes=model_bytes,
            usage_lane=contract.weight_receipt.admitted_lane.value,
            license_id=contract.weight_source.license_id,
            parity_receipt_sha256=parity_file_sha256,
        )
        _write_json(staged_manifest, manifest.to_dict())
        _validate_staged_bundle(
            model_path=staged_model,
            preprocessing_path=staged_preprocessing,
            parity_path=staged_parity,
            manifest_path=staged_manifest,
        )
        _publish_no_replace(
            tuple(zip(
                (
                    staged_model,
                    staged_preprocessing,
                    staged_parity,
                    staged_manifest,
                ),
                targets,
                strict=True,
            ))
        )
    return targets


def _produce_parity(
    *,
    wrapper: torch.nn.Module,
    onnx_path: Path,
    artifact_sha256: str,
    contract: Dinov2LocalArtifactContract,
    preprocessing: Any,
    thresholds: ParityThresholds,
    panel_receipt_sha256: str | None,
    panel_paths: tuple[tuple[str, Path], ...],
    temporary_root: Path,
) -> ModelParityReceipt:
    fixture_paths = list(_write_synthetic_fixtures(temporary_root))
    fixture_paths.extend(
        (f"crop-{token}", ParityFixtureKind.RECEIPT_BOUND_CROP, path)
        for token, path in panel_paths
    )
    fixture_paths.sort(key=lambda item: item[0])
    session = ort.InferenceSession(
        onnx_path.read_bytes(), providers=["CPUExecutionProvider"]
    )
    results: list[ParityFixtureResult] = []
    for fixture_id, fixture_kind, path in fixture_paths:
        tensor = preprocess_image_batch((path,), preprocessing)
        with torch.inference_mode():
            reference = wrapper(torch.from_numpy(tensor)).cpu().numpy()
        candidate = session.run(["embedding"], {"images": tensor})[0]
        result = _compare_fixture(
            fixture_id=fixture_id,
            fixture_kind=fixture_kind,
            input_sha256=_sha256_file(path),
            reference=reference,
            candidate=candidate,
            thresholds=thresholds,
        )
        results.append(result)
    return ModelParityReceipt(
        model_id="facebook/dinov2-small",
        artifact_sha256=artifact_sha256,
        source_weights_sha256=contract.model_sha256,
        weight_intake_receipt_sha256=contract.weight_receipt_sha256,
        preprocessing_sha256=preprocessing.config_sha256,
        preprocessor_intake_receipt_sha256=(
            contract.preprocessor_receipt_sha256
        ),
        usage_lane=ModelUsageLane(contract.weight_receipt.admitted_lane.value),
        reference_backend=f"torch={torch.__version__};transformers-local",
        candidate_backend=f"onnxruntime-cpu={ort.__version__}",
        thresholds=thresholds,
        fixture_panel_receipt_sha256=panel_receipt_sha256,
        fixtures=tuple(results),
        decision="PASS",
    )


def _compare_fixture(
    *,
    fixture_id: str,
    fixture_kind: ParityFixtureKind,
    input_sha256: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    thresholds: ParityThresholds,
) -> ParityFixtureResult:
    if (
        reference.dtype != np.float32
        or candidate.dtype != np.float32
        or reference.shape != (1, 384)
        or candidate.shape != (1, 384)
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise RuntimeError("DINOv2 parity output contract differs")
    difference = np.abs(reference - candidate)
    maximum_absolute_error = float(np.max(difference))
    denominator = np.maximum(
        np.abs(reference), np.float32(thresholds.relative_error_floor)
    )
    maximum_relative_error = float(np.max(difference / denominator))
    cosine = float(
        np.dot(reference[0], candidate[0])
        / (np.linalg.norm(reference[0]) * np.linalg.norm(candidate[0]))
    )
    if (
        maximum_absolute_error > thresholds.maximum_absolute_error
        or maximum_relative_error > thresholds.maximum_relative_error
        or cosine < thresholds.minimum_cosine_similarity
    ):
        raise RuntimeError(
            f"DINOv2 parity failed for {fixture_id}: "
            f"abs={maximum_absolute_error}, rel={maximum_relative_error}, "
            f"cosine={cosine}"
        )
    return ParityFixtureResult(
        fixture_id=fixture_id,
        fixture_kind=fixture_kind,
        input_sha256=input_sha256,
        reference_output_sha256=sha256(
            np.ascontiguousarray(reference).tobytes()
        ).hexdigest(),
        candidate_output_sha256=sha256(
            np.ascontiguousarray(candidate).tobytes()
        ).hexdigest(),
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        cosine_similarity=min(1.0, cosine),
        decision="PASS",
    )


def _write_synthetic_fixtures(
    root: Path,
) -> tuple[tuple[str, ParityFixtureKind, Path], ...]:
    fixture_root = root / "synthetic-fixtures"
    fixture_root.mkdir()
    specifications = (
        ("synthetic-gradient-landscape", 319, 173, 11),
        ("synthetic-gradient-portrait", 181, 307, 37),
        ("synthetic-checker-square", 257, 257, 73),
    )
    fixtures = []
    for fixture_id, width, height, offset in specifications:
        y, x = np.indices((height, width), dtype=np.uint32)
        array = np.stack(
            (
                (x + offset) % 256,
                (3 * y + offset) % 256,
                ((x // 7 + y // 5 + offset) % 2) * 255,
            ),
            axis=2,
        ).astype(np.uint8)
        path = fixture_root / f"{fixture_id}.png"
        Image.fromarray(array, mode="RGB").save(path, format="PNG")
        fixtures.append((fixture_id, ParityFixtureKind.SYNTHETIC, path))
    return tuple(fixtures)


def _load_crop_panel(
    *,
    crop_export_receipt: Path | None,
    expected_receipt_sha256: str | None,
    crop_root: Path | None,
    crop_tokens: tuple[str, ...],
) -> tuple[str | None, tuple[tuple[str, Path], ...]]:
    supplied = (
        crop_export_receipt is not None,
        expected_receipt_sha256 is not None,
        crop_root is not None,
        bool(crop_tokens),
    )
    if not any(supplied):
        return None, ()
    if not all(supplied):
        raise ValueError(
            "crop parity requires receipt, expected hash, root, and crop tokens"
        )
    assert crop_export_receipt is not None
    assert expected_receipt_sha256 is not None
    assert crop_root is not None
    receipt_result = read_retained_regular_file(
        crop_export_receipt,
        expected_sha256=expected_receipt_sha256,
        maximum_bytes=64 * 1024 * 1024,
        capture_payload=True,
        subject="crop export receipt",
    )
    if receipt_result.payload is None:  # pragma: no cover - helper contract
        raise RuntimeError("crop export receipt payload was not retained")
    receipt = CropExportReceipt.from_dict(
        parse_bounded_strict_json_object(receipt_result.payload)
    )
    entries = {
        entry.artifact_token: entry for entry in receipt.artifact_manifest.entries
    }
    if len(set(crop_tokens)) != len(crop_tokens):
        raise ValueError("crop parity tokens must be unique")
    root = crop_root.resolve(strict=True)
    panel = []
    for token in sorted(crop_tokens):
        if token not in entries:
            raise ValueError(f"crop token is absent from receipt: {token}")
        entry = entries[token]
        path = root / entry.relative_path
        result = read_retained_regular_file(
            path,
            expected_bytes=entry.byte_size,
            expected_sha256=entry.content_sha256,
            capture_payload=False,
            subject="receipt-bound parity crop",
        )
        if result.sha256 != entry.content_sha256:
            raise RuntimeError("receipt-bound parity crop changed")
        panel.append((token, path))
    return receipt_result.sha256, tuple(panel)


def _validate_onnx(path: Path) -> tuple[str, int]:
    model_bytes = path.read_bytes()
    model = onnx.load_model_from_string(model_bytes)
    onnx.checker.check_model(model)
    _reject_external_data(model.graph)
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].name != "images"
        or outputs[0].name != "embedding"
        or tuple(inputs[0].shape) != ("batch", 3, 224, 224)
        or tuple(outputs[0].shape) != ("batch", 384)
        or inputs[0].type != "tensor(float)"
        or outputs[0].type != "tensor(float)"
    ):
        raise RuntimeError("exported DINOv2 ONNX tensor contract differs")
    probe = session.run(
        ["embedding"],
        {"images": np.zeros((1, 3, 224, 224), dtype=np.float32)},
    )[0]
    if probe.shape != (1, 384) or probe.dtype != np.float32 or not np.isfinite(probe).all():
        raise RuntimeError("exported DINOv2 ONNX CPU smoke failed")
    return sha256(model_bytes).hexdigest(), len(model_bytes)


def _reject_external_data(graph: Any) -> None:
    def reject_tensor(tensor: Any) -> None:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise RuntimeError("exported ONNX contains external tensor data")

    for tensor in graph.initializer:
        reject_tensor(tensor)
    for sparse in graph.sparse_initializer:
        reject_tensor(sparse.values)
        reject_tensor(sparse.indices)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                reject_tensor(attribute.t)
            for tensor in attribute.tensors:
                reject_tensor(tensor)
            if attribute.HasField("sparse_tensor"):
                reject_tensor(attribute.sparse_tensor.values)
                reject_tensor(attribute.sparse_tensor.indices)
            for sparse in attribute.sparse_tensors:
                reject_tensor(sparse.values)
                reject_tensor(sparse.indices)
            if attribute.HasField("g"):
                _reject_external_data(attribute.g)
            for nested in attribute.graphs:
                _reject_external_data(nested)


def _validate_staged_bundle(
    *,
    model_path: Path,
    preprocessing_path: Path,
    parity_path: Path,
    manifest_path: Path,
) -> None:
    manifest = Dinov2OnnxArtifactManifest.from_dict(
        read_strict_json_object(manifest_path)
    )
    if _sha256_file(model_path) != manifest.onnx_sha256:
        raise RuntimeError("staged DINOv2 model hash differs")
    preprocessing = read_strict_json_object(preprocessing_path)
    if content_sha256(preprocessing) != manifest.preprocessing_config_sha256:
        raise RuntimeError("staged DINOv2 preprocessing hash differs")
    if _sha256_file(parity_path) != manifest.parity_receipt_sha256:
        raise RuntimeError("staged DINOv2 parity receipt hash differs")
    parity = ModelParityReceipt.from_dict(read_strict_json_object(parity_path))
    if parity.artifact_sha256 != manifest.onnx_sha256 or parity.decision != "PASS":
        raise RuntimeError("staged DINOv2 parity binding differs")


def _publish_no_replace(items: tuple[tuple[Path, Path], ...]) -> None:
    created: list[Path] = []
    try:
        for source, target in items:
            os.link(source, target)
            created.append(target)
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline receipt-bound DINOv2-small ONNX export"
    )
    parser.add_argument("--model", choices=("dinov2-small",), default="dinov2-small")
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--weight-intake-bundle", required=True, type=Path)
    parser.add_argument("--preprocessor-intake-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--maximum-absolute-error", type=float, default=1e-4)
    parser.add_argument("--maximum-relative-error", type=float, default=1e-2)
    parser.add_argument("--relative-error-floor", type=float, default=1e-4)
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.999999)
    parser.add_argument("--crop-export-receipt", type=Path)
    parser.add_argument("--expected-crop-export-receipt-sha256")
    parser.add_argument("--crop-root", type=Path)
    parser.add_argument("--crop-token", action="append", default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    targets = export_dinov2_small(
        model_directory=args.model_directory,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        output_directory=args.output_dir,
        artifact_stem=args.artifact_stem,
        thresholds=ParityThresholds(
            maximum_absolute_error=args.maximum_absolute_error,
            maximum_relative_error=args.maximum_relative_error,
            relative_error_floor=args.relative_error_floor,
            minimum_cosine_similarity=args.minimum_cosine_similarity,
        ),
        crop_export_receipt=args.crop_export_receipt,
        expected_crop_export_receipt_sha256=(
            args.expected_crop_export_receipt_sha256
        ),
        crop_root=args.crop_root,
        crop_tokens=tuple(args.crop_token),
    )
    for path in targets:
        print(f"{path} sha256={_sha256_file(path)} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
