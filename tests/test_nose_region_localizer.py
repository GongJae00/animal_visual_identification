from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
from PIL import Image
from torch import nn

import parsing.nose_region.localizer as localizer_module
from contracts.artifact_manifest import (
    ArtifactLicense,
    ExactOnnxRuntime,
    ImagePreprocessing,
    NoseDetectorManifest,
    UsageLane,
)
from parsing.nose_region.localizer import (
    AP10K_SUPPORTED_INDICES,
    KEYPOINT_ORDER,
    LetterboxTransform,
    MobileNetV4NoseLocalizer,
    NoseDetectorWrapper,
    ZipNoseKeypointDataset,
    dogflw_points,
    keypoint_metrics,
    load_mobilenetv4_localizer,
    mobilenetv4_feature_dim,
    parse_ap10k_zip,
    parse_dogflw_zip,
    partial_keypoint_loss,
)


def _png_bytes(size: tuple[int, int] = (100, 50)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, (20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


def _ap10k_zip(path: Path) -> None:
    keypoints = []
    for index in range(17):
        keypoints.extend((30.0 + index, 15.0 + index, 2))
    with ZipFile(path, "w") as archive:
        archive.writestr("ap-10k/data/dog.png", _png_bytes())
        for split_index, split in enumerate(("train", "val", "test")):
            payload = {
                "images": [{"id": 1, "file_name": "dog.png"}],
                "annotations": [
                    {
                        "id": split_index + 10,
                        "image_id": 1,
                        "category_id": 8,
                        "bbox": [10, 5, 80, 40],
                        "keypoints": keypoints,
                    },
                    {
                        "id": split_index + 20,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "keypoints": keypoints,
                    },
                ],
            }
            archive.writestr(
                f"ap-10k/annotations/ap10k-{split}-split1.json",
                json.dumps(payload),
            )


def _dogflw_zip(path: Path) -> None:
    landmarks = [[float(index), float(index * 2)] for index in range(46)]
    label = {"landmarks": landmarks, "bounding_boxes": [0, 0, 100, 50]}
    with ZipFile(path, "w") as archive:
        for split in ("train", "test"):
            archive.writestr(
                f"bundle/DogFLW/{split}/images/breed_1.png", _png_bytes()
            )
            archive.writestr(
                f"bundle/DogFLW/{split}/labels/breed_1.json", json.dumps(label)
            )


def test_letterbox_geometry_is_deterministic_and_targets_are_normalized() -> None:
    transform = LetterboxTransform.create(100, 50, 224)
    assert (
        transform.resized_width,
        transform.resized_height,
        transform.pad_left,
        transform.pad_top,
    ) == (224, 112, 0, 56)
    assert transform.normalized_point(50, 25) == pytest.approx((0.5, 0.5))
    assert transform.normalized_content_diagonal == pytest.approx(5**0.5 / 2)


def test_dogflw_documented_keypoint_derivation() -> None:
    landmarks = [[float(index), float(index * 10)] for index in range(46)]
    points = dogflw_points(landmarks)
    assert points[0] == pytest.approx((19.0, 190.0))
    assert points[1] == pytest.approx((20.0, 200.0))
    assert points[2] == (25.0, 250.0)
    assert points[3] == (35.0, 350.0)
    assert points[4] == pytest.approx((32.5, 325.0))
    assert points[5] == pytest.approx((33.0, 330.0))
    assert points[6:] == ((26.0, 260.0), (27.0, 270.0))

    landmarks[18] = ["NaN", 1]
    assert dogflw_points(landmarks)[0] is None


def test_ap10k_zip_parser_and_dataset_use_partial_supervision(tmp_path: Path) -> None:
    archive_path = tmp_path / "ap10k.zip"
    _ap10k_zip(archive_path)
    splits = parse_ap10k_zip(archive_path)
    assert {name: len(records) for name, records in splits.items()} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    record = splits["train"][0]
    assert tuple(index for index, value in enumerate(record.supported) if value) == (
        AP10K_SUPPORTED_INDICES
    )
    assert record.points[2] is None
    assert record.points[3] == (32.0, 17.0)

    sample = ZipNoseKeypointDataset(splits["train"])[0]
    assert sample["image"].shape == (3, 224, 224)
    assert torch.equal(
        sample["support"],
        torch.tensor([True, True, False, True, False, False, False, False]),
    )
    assert sample["visibility"].tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert sample["target"][0].tolist() == pytest.approx([0.25, 0.25])


def test_dogflw_zip_parser_preserves_publisher_splits(tmp_path: Path) -> None:
    archive_path = tmp_path / "dogflw.zip"
    _dogflw_zip(archive_path)
    splits = parse_dogflw_zip(archive_path)
    assert set(splits) == {"train", "test"}
    assert len(splits["train"]) == len(splits["test"]) == 1
    assert splits["train"][0].supported == (True,) * len(KEYPOINT_ORDER)
    assert splits["train"][0].points[0] == pytest.approx((19.0, 38.0))


def test_dogflw_zip_parser_uses_full_image_for_blank_publisher_box(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "dogflw.zip"
    landmarks = [[float(index), float(index)] for index in range(46)]
    label = {"landmarks": landmarks, "bounding_boxes": ["", "", "", ""]}
    with ZipFile(archive_path, "w") as archive:
        for split in ("train", "test"):
            archive.writestr(
                f"DogFLW/{split}/images/dog.png", _png_bytes((100, 50))
            )
            archive.writestr(
                f"DogFLW/{split}/labels/dog.json", json.dumps(label)
            )
    splits = parse_dogflw_zip(archive_path)
    assert splits["train"][0].crop_xyxy == (0.0, 0.0, 100.0, 50.0)


def test_dogflw_zip_parser_treats_publisher_nan_as_missing_point(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "dogflw.zip"
    landmarks = [[float(index), float(index)] for index in range(46)]
    landmarks[18] = [float("nan"), 1.0]
    label = {"landmarks": landmarks, "bounding_boxes": [0, 0, 100, 50]}
    with ZipFile(archive_path, "w") as archive:
        for split in ("train", "test"):
            archive.writestr(f"DogFLW/{split}/images/dog.png", _png_bytes())
            archive.writestr(
                f"DogFLW/{split}/labels/dog.json",
                json.dumps(label, allow_nan=True),
            )
    assert parse_dogflw_zip(archive_path)["train"][0].points[0] is None


def test_dogflw_dataset_uses_full_image_for_out_of_bounds_publisher_box(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "dogflw.zip"
    landmarks = [[float(index), float(index)] for index in range(46)]
    label = {"landmarks": landmarks, "bounding_boxes": [500, 500, 600, 600]}
    with ZipFile(archive_path, "w") as archive:
        for split in ("train", "test"):
            archive.writestr(f"DogFLW/{split}/images/dog.png", _png_bytes())
            archive.writestr(
                f"DogFLW/{split}/labels/dog.json", json.dumps(label)
            )
    record = parse_dogflw_zip(archive_path)["train"][0]
    sample = ZipNoseKeypointDataset((record,))[0]
    assert sample["visibility"][0]


def test_zip_parser_rejects_unsafe_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../ap-10k/annotations/ap10k-train-split1.json", "{}")
    with pytest.raises(ValueError, match="unsafe ZIP-relative path"):
        parse_ap10k_zip(archive_path)


class _TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)

    def forward_head(
        self, features: torch.Tensor, *, pre_logits: bool = False
    ) -> torch.Tensor:
        assert pre_logits
        return features.mean(dim=(2, 3))


def test_model_loss_and_detector_wrapper_are_differentiable() -> None:
    model = MobileNetV4NoseLocalizer(_TinyBackbone(), feature_dim=4)
    images = torch.randn(1, 3, 16, 16)
    prediction = model(images)
    assert prediction.shape == (1, 8, 3)
    assert torch.all((prediction >= 0) & (prediction <= 1))

    target = torch.zeros(1, 8, 2)
    visibility = torch.tensor(
        [[True, True, False, True, False, False, False, False]]
    )
    support = torch.tensor(
        [[True, True, False, True, False, False, False, False]]
    )
    losses = partial_keypoint_loss(prediction, target, visibility, support)
    losses["total"].backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()

    detector_output = NoseDetectorWrapper(model)(images)
    assert detector_output.shape == (1, 1, 5)
    assert torch.all((detector_output >= 0) & (detector_output <= 1))


def test_feature_dimension_prefers_actual_classifier_input() -> None:
    backbone = _TinyBackbone()
    backbone.classifier = nn.Linear(7, 2)
    assert mobilenetv4_feature_dim(backbone) == 7


def test_detector_exports_static_manifest_compatible_tensor(tmp_path: Path) -> None:
    model = MobileNetV4NoseLocalizer(_TinyBackbone(), feature_dim=4).eval()
    detector = NoseDetectorWrapper(model).eval()
    artifact = tmp_path / "detector.onnx"
    dummy = torch.zeros(1, 3, 16, 16)
    torch.onnx.export(
        detector,
        (dummy,),
        artifact,
        input_names=["images"],
        output_names=["detections"],
        opset_version=18,
        external_data=False,
        dynamo=False,
    )
    manifest = NoseDetectorManifest(
        artifact_id="synthetic-nose-localizer",
        artifact_sha256=sha256(artifact.read_bytes()).hexdigest(),
        input_name="images",
        input_shape=(1, 3, 16, 16),
        output_name="detections",
        output_shape=(1, 1, 5),
        license=ArtifactLicense("CC-BY-NC-4.0-derived", UsageLane.RESEARCH_ONLY),
        preprocessing=ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bilinear",
            scale=1.0 / 255.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            clahe=None,
        ),
        confidence_threshold=0.5,
    )
    runtime = ExactOnnxRuntime(artifact, manifest)
    output = runtime.run(torch.zeros(1, 3, 16, 16).numpy())
    assert output.shape == (1, 1, 5)


def test_partial_loss_does_not_backpropagate_unsupported_channels() -> None:
    prediction = torch.full((1, 8, 3), 0.5, requires_grad=True)
    target = torch.zeros(1, 8, 2)
    visibility = torch.tensor(
        [[True, True, False, True, False, False, False, False]]
    )
    support = visibility.clone()
    partial_keypoint_loss(prediction, target, visibility, support)["total"].backward()
    assert torch.count_nonzero(prediction.grad[:, 2]).item() == 0
    assert torch.count_nonzero(prediction.grad[:, 4:]).item() == 0
    assert torch.count_nonzero(prediction.grad[:, (0, 1, 3)]).item() > 0


def test_strict_safetensors_loading_uses_caller_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import timm
    from safetensors.torch import save_file

    expected = _TinyBackbone()
    weights = tmp_path / "model.safetensors"
    save_file(expected.state_dict(), str(weights))
    with pytest.raises(ValueError, match="safetensors SHA256 differs"):
        load_mobilenetv4_localizer(weights)
    monkeypatch.setattr(
        localizer_module,
        "MOBILENETV4_WEIGHTS_SHA256",
        sha256(weights.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(timm, "create_model", lambda *args, **kwargs: _TinyBackbone())
    loaded = load_mobilenetv4_localizer(weights)
    assert isinstance(loaded.backbone, _TinyBackbone)

    bad_weights = tmp_path / "bad.safetensors"
    state = dict(expected.state_dict())
    state["unexpected"] = torch.zeros(1)
    save_file(state, str(bad_weights))
    monkeypatch.setattr(
        localizer_module,
        "MOBILENETV4_WEIGHTS_SHA256",
        sha256(bad_weights.read_bytes()).hexdigest(),
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_mobilenetv4_localizer(bad_weights)


def test_metrics_report_nme_and_coverage() -> None:
    prediction = torch.zeros(1, 8, 3)
    target = torch.zeros(1, 8, 2)
    visibility = torch.zeros(1, 8, dtype=torch.bool)
    visibility[0, :2] = True
    prediction[0, 0] = torch.tensor((0.3, 0.4, 0.9))
    prediction[0, 1] = torch.tensor((0.0, 0.0, 0.1))
    metrics = keypoint_metrics(
        prediction, target, visibility, torch.tensor([1.0]), confidence_threshold=0.5
    )
    assert metrics["coverage"] == 0.5
    assert metrics["NME"] == pytest.approx(0.25)
    assert metrics["covered_NME"] == pytest.approx(0.5)


def test_training_cli_help_does_not_require_training_execution() -> None:
    tool = Path(__file__).resolve().parents[1] / "workflows" / "train_nose_localizer.py"
    completed = subprocess.run(
        [sys.executable, str(tool), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--backbone-weights" in completed.stdout
    assert "--device {cpu,cuda}" in completed.stdout
