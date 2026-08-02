from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from artifact_contracts.artifact_manifest import ExactOnnxRuntime, NoseMaskManifest, UsageLane
from identity_governance.identity_registry import compute_registered_dog_id
from localization.nose_region.native_yt import (
    NativeYtSample,
    build_manifest_bundle,
    process_native_sample,
)
from localization.nose_region.sam2_teacher import (
    SOURCE_IMAGE_MANIFEST_SCHEMA,
    produce_teacher_manifest,
    validate_source_image_manifest,
)
from localization.nose_region.segmentation_training import (
    INTERPRETATION,
    LICENSE_ID,
    NoseSegmentationDataset,
    NoseSegmentationStudent,
    SegmentationRecord,
    build_runtime_manifest,
    build_segmentation_checkpoint,
    build_training_config,
    export_static_onnx,
    load_segmentation_checkpoint,
    load_training_records,
    produce_cpu_ort_parity_receipt,
    save_segmentation_checkpoint,
    segmentation_loss,
)
from foundation.protected_io import (
    json_document_bytes,
    read_strict_json_document,
)
from foundation.provenance import content_sha256


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _source_bytes(index: int) -> bytes:
    y, x = np.indices((48, 64), dtype=np.uint16)
    array = np.stack(
        (
            (x + 20 * index) % 256,
            (2 * y + 30 * index) % 256,
            (x + y + 40 * index) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _prediction() -> list[list[float]]:
    return [
        [0.35, 0.20, 0.9],
        [0.65, 0.20, 0.9],
        [0.50, 0.35, 0.9],
        [0.50, 0.65, 0.9],
        [0.43, 0.58, 0.9],
        [0.57, 0.58, 0.9],
        [0.38, 0.52, 0.9],
        [0.62, 0.52, 0.9],
    ]


def _producer() -> dict[str, object]:
    tool = {"schema_version": "fixture.tool.v1"}
    return {
        "model_name": "sam2.1",
        "sam2_checkout_commit": "1" * 40,
        "sam2_python_sources_sha256": _sha("sam2-sources"),
        "sam2_config_relative_path": "configs/sam2.1/fixture.yaml",
        "sam2_config_sha256": _sha("sam2-config"),
        "sam2_checkpoint_filename": "sam2.1_fixture.pt",
        "sam2_checkpoint_sha256": _sha("sam2-checkpoint"),
        "license_id": "Apache-2.0",
        "license_snapshot_sha256": _sha("sam2-license"),
        "device": "cpu",
        "prompt_contract": "NOSE_BOX_AND_POSITIVE_NOSE_KEYPOINTS",
        "output_encoding": "SOURCE_RESOLUTION_BINARY_L_PNG",
        "tool_provenance": tool,
        "tool_provenance_sha256": content_sha256(tool),
    }


class _MaskPredictor:
    def __init__(self, rejected_index: int) -> None:
        self.index = 0
        self.rejected_index = rejected_index
        self.shape = (0, 0)

    def set_image(self, image: np.ndarray) -> None:
        self.shape = image.shape[:2]

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
        height, width = self.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.index != self.rejected_index:
            left, top, right, bottom = np.asarray(kwargs["box"], dtype=np.int32)
            center = ((left + right) // 2, (top + bottom) // 2)
            axes = (max(2, (right - left) // 3), max(2, (bottom - top) // 3))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        self.index += 1
        return mask[None], np.asarray([0.95], dtype=np.float32), None


def _fixture_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    native_root = tmp_path / "native"
    teacher_root = tmp_path / "teacher"
    for directory in (
        native_root / "crops",
        native_root / "soft_masks",
        native_root / "binary_masks",
        native_root / "teacher_source_images",
        teacher_root / "masks",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    policy = {
        "minimum_detector_confidence": 0.1,
        "minimum_frontality": 0.0,
        "minimum_native_short_side": 4,
        "maximum_mask_uncertainty": 1.0,
    }
    records = []
    source_rows = []
    for index in range(5):
        source_bytes = _source_bytes(index)
        sample_token = _sha(f"sample-{index}")
        sample = NativeYtSample(
            sample_token=sample_token,
            identity_token=_sha(f"identity-{index}"),
            registered_dog_id=compute_registered_dog_id(
                f"yt-bb-dog:v1:video-track:{index}"
            ),
            source_sample_id=f"yt-bb-dog:v1:original:video-track:{index}:frame:0",
            sequence_token=_sha(f"sequence-{index}"),
            track_token=_sha(f"track-{index}"),
            frame_index=0,
            source_role="YT_FIT",
            member_path=f"YT-BB-Dog/train/{index}/0.png",
            member_crc32=index,
            member_uncompressed_bytes=len(source_bytes),
            container_member_path="YT-BB-dog/YT-BB-Dog.zip",
            container_member_crc32=1,
            container_member_uncompressed_bytes=123,
            expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            roi_metadata_available=True,
        )
        record, artifacts = process_native_sample(
            sample, source_bytes, _prediction(), policy=policy
        )
        for relative, payload in artifacts.items():
            (native_root / relative).write_bytes(payload)
        source_relative = f"teacher_source_images/{sample_token}.png"
        (native_root / source_relative).write_bytes(source_bytes)
        source_rows.append(
            {
                "sample_token": sample_token,
                "sequence_token": record["sequence_token"],
                "track_token": record["track_token"],
                "frame_index": record["frame_index"],
                "source_image_path": source_relative,
                "source_sha256": record["source_sha256"],
                "source_width": record["source_width"],
                "source_height": record["source_height"],
                "nose_box_xyxy": record["nose_box_xyxy"],
                "keypoints": record["keypoints"],
            }
        )
        records.append(record)
    records.sort(key=lambda row: row["sample_token"])
    source_rows.sort(key=lambda row: row["sample_token"])
    native = build_manifest_bundle(
        records=records,
        input_sha256s={"source_receipt": _sha("source-receipt")},
        policy=policy,
        tool_provenance={"schema_version": "fixture.native.v1"},
    )
    native_path = native_root / "yt-native-nose-manifest.json"
    native_path.write_bytes(json_document_bytes(native))
    source_payload = {
        "schema_version": SOURCE_IMAGE_MANIFEST_SCHEMA,
        "source_receipt_file_sha256": _sha("source-receipt"),
        "records": source_rows,
    }
    source_path = native_root / "yt-native-nose-teacher-source-images.json"
    source_path.write_bytes(json_document_bytes(source_payload))
    source_document = read_strict_json_document(source_path)
    sources = validate_source_image_manifest(
        source_payload,
        root=native_root,
        source_receipt_file_sha256=_sha("source-receipt"),
    )
    teacher, masks = produce_teacher_manifest(
        sources,
        _MaskPredictor(rejected_index=2),
        source_binding={
            "source_manifest_schema": SOURCE_IMAGE_MANIFEST_SCHEMA,
            "source_manifest_file_sha256": source_document.raw_sha256,
            "source_manifest_payload_sha256": source_document.canonical_payload_sha256,
            "source_receipt_filename": "source-receipt.json",
            "source_receipt_file_sha256": _sha("source-receipt"),
        },
        producer=_producer(),
    )
    for relative, payload in masks.items():
        (teacher_root / relative).write_bytes(payload)
    teacher_path = teacher_root / "yt-native-nose-teacher-masks.json"
    teacher_path.write_bytes(json_document_bytes(teacher))
    return teacher_path, source_path, native_path


def _backbone_binding(bindings: dict) -> dict:
    return {
        **bindings,
        "backbone": {
            "model_name": "mobilenetv4_conv_small.e1200_r224_in1k",
            "source_revision": "c9f31ac64483d7f0590db9edccb4418392a96eea",
            "safetensors_sha256": "5a2ef04d419ce6d1bf27bfa735bb200d3f8d8997c3ac36320f5bf30382f6b43c",
        },
    }


def _config() -> dict:
    return build_training_config(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        decoder_channels=4,
        num_workers=0,
        seed=7,
        dev_fraction=0.4,
        threshold=0.5,
        device_name="cpu",
        parity_max_absolute_error=1e-4,
    )


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.register_buffer("batch_counter", torch.tensor(0, dtype=torch.int64))

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(self.conv(images), kernel_size=4)


def test_spatial_student_loss_reaches_backbone_stem_and_decoder() -> None:
    model = NoseSegmentationStudent(_TinyBackbone(), 4, decoder_channels=4)
    images = torch.randn(2, 3, 224, 224)
    target = torch.zeros(2, 1, 224, 224)
    target[:, :, 64:160, 72:152] = 1.0

    logits = model(images)
    losses = segmentation_loss(logits, target)
    losses["total"].backward()

    assert logits.shape == target.shape
    assert torch.isfinite(losses["total"])
    assert model.backbone.conv.weight.grad is not None
    assert model.rgb_stem[0].weight.grad is not None
    assert model.decoder[0].weight.grad is not None


def test_manifest_cross_binding_accepted_only_identity_split_and_exact_crop(
    tmp_path: Path,
) -> None:
    teacher_path, source_path, native_path = _fixture_manifests(tmp_path)
    first = load_training_records(
        teacher_manifest_path=teacher_path,
        source_manifest_path=source_path,
        native_manifest_path=native_path,
        dev_fraction=0.4,
        seed=7,
    )
    second = load_training_records(
        teacher_manifest_path=teacher_path,
        source_manifest_path=source_path,
        native_manifest_path=native_path,
        dev_fraction=0.4,
        seed=7,
    )
    train, dev, bindings = first

    assert [record.sample_token for record in train] == [
        record.sample_token for record in second[0]
    ]
    assert [record.sample_token for record in dev] == [
        record.sample_token for record in second[1]
    ]
    assert len(train) + len(dev) == 4
    assert bindings["rejected_mask_count"] == 1
    assert {record.registered_dog_id for record in train}.isdisjoint(
        record.registered_dog_id for record in dev
    )
    image, mask, index = NoseSegmentationDataset(
        train, training=False, seed=7
    )[0]
    assert image.shape == (3, 224, 224)
    assert mask.shape == (1, 224, 224)
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    assert index == 0

    changed = json.loads(source_path.read_text(encoding="utf-8"))
    changed["records"][0]["frame_index"] += 1
    source_path.write_bytes(json_document_bytes(changed))
    with pytest.raises(ValueError, match="teacher/source manifest binding differs"):
        load_training_records(
            teacher_manifest_path=teacher_path,
            source_manifest_path=source_path,
            native_manifest_path=native_path,
            dev_fraction=0.4,
            seed=7,
        )


def test_checkpoint_hashes_reject_input_and_tensor_tamper(tmp_path: Path) -> None:
    teacher_path, source_path, native_path = _fixture_manifests(tmp_path)
    _, _, bindings = load_training_records(
        teacher_manifest_path=teacher_path,
        source_manifest_path=source_path,
        native_manifest_path=native_path,
        dev_fraction=0.4,
        seed=7,
    )
    bindings = _backbone_binding(bindings)
    model = NoseSegmentationStudent(_TinyBackbone(), 4, decoder_channels=4)
    payload = build_segmentation_checkpoint(
        model=model,
        epoch=1,
        selection={
            "metric_order": ["DEV_teacher_Dice", "DEV_teacher_IoU"],
            "Dice": 0.75,
            "IoU": 0.6,
            "interpretation": INTERPRETATION,
        },
        input_bindings=bindings,
        training_config=_config(),
    )
    checkpoint_path = tmp_path / "selected.pt"
    save_segmentation_checkpoint(checkpoint_path, payload)
    loaded = load_segmentation_checkpoint(
        checkpoint_path,
        expected_input_bindings=bindings,
        expected_training_config=_config(),
    )
    assert loaded["input_bindings"]["license"] == {
        "license_id": LICENSE_ID,
        "usage_lane": "RESEARCH_ONLY",
        "reason": "Derived from the research-only native nose localization lane.",
    }

    metadata_tamper = torch.load(checkpoint_path, weights_only=True)
    metadata_tamper["input_bindings"]["manifests"]["teacher"][
        "file_sha256"
    ] = "0" * 64
    metadata_path = tmp_path / "metadata-tamper.pt"
    torch.save(metadata_tamper, metadata_path)
    with pytest.raises(ValueError, match="input bindings digest"):
        load_segmentation_checkpoint(metadata_path)

    tensor_tamper = torch.load(checkpoint_path, weights_only=True)
    tensor_name = next(
        name
        for name, tensor in tensor_tamper["model_state_dict"].items()
        if tensor.is_floating_point()
    )
    tensor_tamper["model_state_dict"][tensor_name].view(-1)[0] += 1.0
    tensor_path = tmp_path / "tensor-tamper.pt"
    torch.save(tensor_tamper, tensor_path)
    with pytest.raises(ValueError, match="model state digest"):
        load_segmentation_checkpoint(tensor_path)


def test_actual_static_onnx_runtime_manifest_and_cpu_ort_parity(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = NoseSegmentationStudent(_TinyBackbone(), 4, decoder_channels=4).eval()
    onnx_path = tmp_path / "nose-mask.onnx"
    digest, _ = export_static_onnx(model, onnx_path)
    manifest = build_runtime_manifest(onnx_path, threshold=0.5)
    loaded_manifest = NoseMaskManifest.from_dict(manifest.to_dict())
    runtime = ExactOnnxRuntime(onnx_path, loaded_manifest)
    output = runtime.run(np.zeros((1, 3, 224, 224), dtype=np.float32))

    assert loaded_manifest.artifact_sha256 == digest
    assert loaded_manifest.output_name == "mask"
    assert loaded_manifest.output_shape == (1, 1, 224, 224)
    assert loaded_manifest.license.usage_lane is UsageLane.RESEARCH_ONLY
    assert np.isfinite(output).all()
    assert np.all((output >= 0.0) & (output <= 1.0))

    source = Image.new("RGB", (20, 18), color=(80, 120, 160))
    source_path = tmp_path / "source.png"
    source.save(source_path, format="PNG")
    mask_array = np.zeros((18, 20), dtype=np.uint8)
    mask_array[5:14, 6:15] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask_array, mode="L").save(mask_path, format="PNG")
    record = SegmentationRecord(
        sample_token="1" * 64,
        registered_dog_id=compute_registered_dog_id("fixture:parity"),
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        mask_path=mask_path,
        mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        source_size=(20, 18),
        nose_box_xyxy=(3, 3, 18, 17),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"bound selected checkpoint fixture")
    receipt = produce_cpu_ort_parity_receipt(
        model=model,
        onnx_path=onnx_path,
        checkpoint_path=checkpoint_path,
        record=record,
        maximum_absolute_error=1e-4,
    )
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    assert receipt["decision"] == "PASS"
    assert receipt["finite"] is True
    assert receipt["range"] == [0.0, 1.0]
    assert receipt["receipt_sha256"] == content_sha256(receipt_body)

    onnx_path.write_bytes(onnx_path.read_bytes() + b"tamper")
    with pytest.raises(Exception, match="pre-runtime validation"):
        ExactOnnxRuntime(onnx_path, loaded_manifest)


def test_cli_help_avoids_heavy_imports() -> None:
    tool = Path(__file__).parents[1] / "workflows" / "train_nose_segmentation_student.py"
    command = (
        "import runpy,sys; tool=sys.argv[1]; sys.argv=[tool,'--help']; "
        "\ntry: runpy.run_path(tool,run_name='__main__')"
        "\nexcept SystemExit as exc: assert exc.code == 0"
        "\nassert 'torch' not in sys.modules and 'onnx' not in sys.modules "
        "and 'onnxruntime' not in sys.modules and 'timm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command, str(tool)], check=True)
