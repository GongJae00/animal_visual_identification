"""Receipt-bound RGB nose-region embedding training and static ONNX export."""

from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from contracts.artifact_manifest import (
    ArtifactLicense,
    ImagePreprocessing,
    NoseEmbeddingManifest,
    UsageLane,
    preprocess_image,
)
from contracts.dinov2_contract import Dinov2LocalArtifactContract
from contracts.model_parity import (
    ModelParityReceipt,
    ModelUsageLane,
    ParityFixtureKind,
    ParityFixtureResult,
    ParityThresholds,
)
from foundation.protected_io import (
    read_strict_json_document,
    read_strict_json_object,
    write_private_json_bundle,
)
from foundation.provenance import content_sha256
from localization.nose_region.manifest import read_nose_region_manifest

CHECKPOINT_SCHEMA = "cvi.nose_region_rgb_embedding_checkpoint.v1"
LINEAGE_SCHEMA = "cvi.nose_region_rgb_embedding_artifact_bundle.v1"
DEV_REPORT_SCHEMA = "cvi.nose_region_rgb_embedding_dev_selection.v1"
MODEL_ID = "cvi.nose_region_rgb_embedding.dinov2-small-cls.v1"
LICENSE_ID = "CC-BY-NC-4.0-derived"
DEV_INTERPRETATION = (
    "IDENTITY_DISJOINT_SAME_SESSION_DEV_DIAGNOSTIC_NOT_CROSS_SESSION_"
    "GENERALIZATION_OR_FINAL_PERFORMANCE"
)
SELECTION_METRICS = ("DEV_mAP", "DEV_Rank-1")
EMBEDDING_DIM = 384
IMAGE_SIZE = 224

_CHECKPOINT_KEYS = {
    "schema_version",
    "epoch",
    "global_step",
    "selection",
    "bindings",
    "bindings_sha256",
    "training_config",
    "training_config_sha256",
    "model_state_dict",
    "model_state_sha256",
    "arcface_state_dict",
    "arcface_state_sha256",
    "checkpoint_payload_sha256",
}


def build_dev_protocol(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the exact image-level leave-one-out DEV population."""

    dev_records = tuple(record for record in records if record["split_role"] == "DEV")
    if not dev_records:
        raise ValueError("nose-region embedding training requires DEV records")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in dev_records:
        grouped[record["registered_dog_id"]].append(record)
    multi_session = sorted(
        identity
        for identity, items in grouped.items()
        if len({item["capture_session_token"] for item in items}) != 1
    )
    if multi_session:
        raise ValueError(
            "same-session DEV diagnostic requires one capture_session_token per identity"
        )
    eligible = sorted(identity for identity, items in grouped.items() if len(items) >= 2)
    excluded = sorted(set(grouped) - set(eligible))
    if len(eligible) < 2:
        raise ValueError(
            "DEV requires at least two identities eligible for leave-one-out retrieval"
        )
    eligible_set = set(eligible)
    eligible_samples = sorted(
        record["sample_token"]
        for record in dev_records
        if record["registered_dog_id"] in eligible_set
    )
    return {
        "interpretation": DEV_INTERPRETATION,
        "identity_disjoint_from_train": True,
        "same_session_within_each_dev_identity": True,
        "selection_metric_order": list(SELECTION_METRICS),
        "dev_registered_dog_ids": sorted(grouped),
        "eligible_registered_dog_ids": eligible,
        "excluded_singleton_registered_dog_ids": excluded,
        "eligible_sample_tokens": eligible_samples,
        "eligible_image_count": len(eligible_samples),
    }


def evaluate_dev_leave_one_out(
    embeddings: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic image-level leave-one-out closed-set retrieval."""

    if not isinstance(embeddings, np.ndarray) or embeddings.dtype != np.float32:
        raise TypeError("DEV embeddings must be a float32 ndarray")
    if embeddings.ndim != 2 or embeddings.shape != (len(records), EMBEDDING_DIM):
        raise ValueError("DEV embeddings must have shape [records,384]")
    if not np.isfinite(embeddings).all():
        raise ValueError("DEV embeddings must be finite")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("DEV embeddings must be L2-normalized")
    expected_protocol = build_dev_protocol(records)
    if protocol is not None and dict(protocol) != expected_protocol:
        raise ValueError("DEV protocol differs from the record population")

    eligible = set(expected_protocol["eligible_registered_dog_ids"])
    indices = [
        index
        for index, record in enumerate(records)
        if record["registered_dog_id"] in eligible
    ]
    average_precisions: list[float] = []
    rank_one_hits = 0
    for query_index in indices:
        query_identity = records[query_index]["registered_dog_id"]
        candidates = [index for index in indices if index != query_index]
        scores = embeddings[candidates] @ embeddings[query_index]
        ordered = sorted(
            range(len(candidates)),
            key=lambda position: (
                -float(scores[position]),
                records[candidates[position]]["sample_token"],
            ),
        )
        relevance = [
            records[candidates[position]]["registered_dog_id"] == query_identity
            for position in ordered
        ]
        positive_count = sum(relevance)
        if positive_count <= 0:  # pragma: no cover - prevented by protocol construction
            raise RuntimeError("eligible DEV query has no leave-one-out positive")
        hits = 0
        precision_sum = 0.0
        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(precision_sum / positive_count)
        rank_one_hits += int(relevance[0])
    return {
        "mAP": float(np.mean(average_precisions)),
        "Rank-1": rank_one_hits / len(indices),
        "query_count": len(indices),
        "eligible_identity_count": len(eligible),
        "eligible_image_count": len(indices),
        "excluded_singleton_registered_dog_ids": expected_protocol[
            "excluded_singleton_registered_dog_ids"
        ],
        "interpretation": DEV_INTERPRETATION,
    }


class IdentityBalancedBatchSampler(Sampler[list[int]]):
    """Select the same number of samples from every identity each epoch."""

    def __init__(
        self,
        labels: Sequence[int],
        *,
        batch_size: int,
        samples_per_identity: int = 1,
        seed: int = 0,
    ) -> None:
        if not labels:
            raise ValueError("identity-balanced sampler requires labels")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            isinstance(samples_per_identity, bool)
            or not isinstance(samples_per_identity, int)
            or samples_per_identity <= 0
        ):
            raise ValueError("samples_per_identity must be positive")
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise ValueError("identity labels must be nonnegative integers")
            grouped[label].append(index)
        if sorted(grouped) != list(range(len(grouped))):
            raise ValueError("identity labels must be contiguous from zero")
        self._grouped = dict(grouped)
        self._batch_size = batch_size
        self._samples_per_identity = samples_per_identity
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be nonnegative")
        self._epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self._seed + self._epoch)
        identities = tuple(sorted(self._grouped))
        for sample_round in range(self._samples_per_identity):
            order = torch.randperm(len(identities), generator=generator).tolist()
            selected: list[int] = []
            for position in order:
                identity = identities[position]
                candidates = self._grouped[identity]
                offset = self._epoch * self._samples_per_identity + sample_round
                selected.append(candidates[offset % len(candidates)])
            for start in range(0, len(selected), self._batch_size):
                yield selected[start : start + self._batch_size]

    def __len__(self) -> int:
        return self._samples_per_identity * math.ceil(
            len(self._grouped) / self._batch_size
        )


class NoseRegionCropDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Manifest-bound crop dataset with deterministic mild augmentation."""

    def __init__(
        self,
        root: Path,
        records: Sequence[Mapping[str, Any]],
        labels: Sequence[int],
        *,
        mean: Sequence[float],
        std: Sequence[float],
        training: bool,
        seed: int,
    ) -> None:
        if len(records) != len(labels) or not records:
            raise ValueError("crop records and labels must be non-empty and aligned")
        if tuple(mean) != (0.485, 0.456, 0.406) or tuple(std) != (
            0.229,
            0.224,
            0.225,
        ):
            raise ValueError("nose-region preprocessing must use exact ImageNet mean/std")
        self._root = root.resolve(strict=True)
        self._records = tuple(records)
        self._labels = tuple(labels)
        self._mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self._training = training
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self._records[index]
        path = self._root.joinpath(*Path(record["crop_path"]).parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(self._root)
        ):
            raise ValueError("nose-region crop path is unsafe")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["crop_sha256"]:
            raise ValueError("nose-region crop bytes changed after manifest validation")
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        if self._training:
            image = self._augment(image, index)
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().mul_(1.0 / 255.0)
        tensor = (tensor - self._mean) / self._std
        return tensor, self._labels[index], index

    def _augment(self, image: Image.Image, index: int) -> Image.Image:
        rng = random.Random(self._seed + self._epoch * 1_000_003 + index)
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        angle = rng.uniform(-5.0, 5.0)
        translation = (
            round(rng.uniform(-0.03, 0.03) * image.width),
            round(rng.uniform(-0.03, 0.03) * image.height),
        )
        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            translate=translation,
            fillcolor=(123, 116, 103),
        )
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
        return ImageEnhance.Color(image).enhance(rng.uniform(0.95, 1.05))


class NoseEmbeddingModel(torch.nn.Module):
    """DINOv2-small CLS token projected only by L2 normalization."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=images)
        if isinstance(output, torch.Tensor):
            embedding = output
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("DINOv2 backbone must expose last_hidden_state")
            embedding = hidden[:, 0, :]
        if not torch.jit.is_tracing() and (
            embedding.ndim != 2 or embedding.shape[1] != EMBEDDING_DIM
        ):
            raise ValueError("DINOv2-small CLS embedding must have shape [B,384]")
        return F.normalize(embedding.float(), p=2, dim=1)


class ArcFaceClassificationHead(torch.nn.Module):
    """Training-only additive angular-margin classification head."""

    def __init__(
        self,
        num_classes: int,
        *,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or scale <= 0 or not 0.0 < margin < math.pi / 2:
            raise ValueError("ArcFace class count, scale, or margin is invalid")
        self.weight = torch.nn.Parameter(torch.empty(num_classes, EMBEDDING_DIM))
        torch.nn.init.xavier_normal_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(embeddings, F.normalize(self.weight, dim=1)).clamp(-1, 1)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-12))
        target = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        return (one_hot * target + (1.0 - one_hot) * cosine) * self.scale


def load_receipt_bound_dinov2(
    *,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
    gradient_checkpointing: bool = False,
) -> tuple[NoseEmbeddingModel, Dinov2LocalArtifactContract]:
    """Load DINOv2-small from exact local safetensors without remote code."""

    contract = Dinov2LocalArtifactContract.load(
        model_directory=model_directory,
        weight_intake_bundle=weight_intake_bundle,
        preprocessor_intake_bundle=preprocessor_intake_bundle,
    )
    contract.revalidate_local_files()
    from transformers import Dinov2Model

    backbone = Dinov2Model.from_pretrained(
        str(contract.model_directory),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    if not isinstance(backbone, torch.nn.Module):
        raise TypeError("local DINOv2 loader must return torch.nn.Module")
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
    contract.revalidate_local_files()
    return NoseEmbeddingModel(backbone), contract


def _state_dict_sha256(state: Mapping[str, Any], name: str) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a non-empty state dictionary")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not key or not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must contain named tensors only")
        tensor = value.detach().cpu().contiguous()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
            tensor
        ).all():
            raise ValueError(f"{name} contains non-finite tensors")
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(item) for item in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_embedding_checkpoint(
    *,
    model: torch.nn.Module,
    arcface: torch.nn.Module,
    epoch: int,
    global_step: int,
    selection: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = dict(model.state_dict())
    arcface_state = dict(arcface.state_dict())
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "global_step": global_step,
        "selection": dict(selection),
        "bindings": dict(bindings),
        "bindings_sha256": content_sha256(dict(bindings)),
        "training_config": dict(training_config),
        "training_config_sha256": content_sha256(dict(training_config)),
        "model_state_dict": model_state,
        "model_state_sha256": _state_dict_sha256(model_state, "model_state_dict"),
        "arcface_state_dict": arcface_state,
        "arcface_state_sha256": _state_dict_sha256(
            arcface_state, "arcface_state_dict"
        ),
    }
    payload["checkpoint_payload_sha256"] = _checkpoint_metadata_sha256(payload)
    validate_embedding_checkpoint(payload)
    return payload


def _checkpoint_metadata_sha256(payload: Mapping[str, Any]) -> str:
    return content_sha256(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "model_state_dict",
                "arcface_state_dict",
                "checkpoint_payload_sha256",
            }
        }
    )


def validate_embedding_checkpoint(payload: object) -> None:
    """Strictly validate metadata and every serialized model tensor."""

    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("nose-region embedding checkpoint keys differ")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported nose-region embedding checkpoint schema")
    for name in ("epoch", "global_step"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint {name} must be nonnegative")
    selection = payload["selection"]
    expected_selection_keys = {"metric_order", "mAP", "Rank-1", "interpretation"}
    if not isinstance(selection, dict) or set(selection) != expected_selection_keys:
        raise ValueError("checkpoint selection keys differ")
    if selection["metric_order"] != list(SELECTION_METRICS) or selection[
        "interpretation"
    ] != DEV_INTERPRETATION:
        raise ValueError("checkpoint selection contract differs")
    for name in ("mAP", "Rank-1"):
        value = selection[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"checkpoint {name} must be finite and in [0,1]")
    for name in ("bindings", "training_config"):
        if not isinstance(payload[name], dict) or not payload[name]:
            raise ValueError(f"checkpoint {name} must be a non-empty object")
    _validate_checkpoint_bindings(payload["bindings"])
    _validate_checkpoint_training_config(payload["training_config"])
    if content_sha256(payload["bindings"]) != payload["bindings_sha256"]:
        raise ValueError("checkpoint bindings digest differs")
    if content_sha256(payload["training_config"]) != payload[
        "training_config_sha256"
    ]:
        raise ValueError("checkpoint training config digest differs")
    if payload["model_state_sha256"] != _state_dict_sha256(
        payload["model_state_dict"], "model_state_dict"
    ):
        raise ValueError("checkpoint model state digest differs")
    if payload["arcface_state_sha256"] != _state_dict_sha256(
        payload["arcface_state_dict"], "arcface_state_dict"
    ):
        raise ValueError("checkpoint ArcFace state digest differs")
    if payload["checkpoint_payload_sha256"] != _checkpoint_metadata_sha256(payload):
        raise ValueError("checkpoint payload digest differs")


def replace_embedding_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a mutable best/last checkpoint alias."""

    validate_embedding_checkpoint(dict(payload))
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("checkpoint target must be a regular file or absent")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        validate_embedding_checkpoint(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_embedding_checkpoint(
    path: Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    expected_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    validate_embedding_checkpoint(payload)
    if expected_bindings is not None and payload["bindings"] != dict(expected_bindings):
        raise ValueError("checkpoint bindings differ from expected bindings")
    if expected_training_config is not None and payload["training_config"] != dict(
        expected_training_config
    ):
        raise ValueError("checkpoint training config differs from expected config")
    return payload


def build_runtime_manifest(onnx_path: Path) -> NoseEmbeddingManifest:
    """Create the exact static-batch-one runtime embedding manifest."""

    return NoseEmbeddingManifest(
        artifact_id=MODEL_ID,
        artifact_sha256=_sha256_file(onnx_path),
        input_name="images",
        input_shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        output_name="embedding",
        output_shape=(1, EMBEDDING_DIM),
        license=ArtifactLicense(LICENSE_ID, UsageLane.RESEARCH_ONLY),
        preprocessing=ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bicubic",
            scale=1.0 / 255.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            clahe=None,
        ),
    )


def export_static_onnx(model: torch.nn.Module, path: Path) -> tuple[str, int]:
    """Export and validate a self-contained static [1,3,224,224] ONNX."""

    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite ONNX artifact: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model = _StaticBatchOneExport(model.to(torch.device("cpu")).eval()).eval()
        dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (dummy,),
                temporary,
                input_names=["images"],
                output_names=["embedding"],
                opset_version=18,
                external_data=False,
                dynamo=False,
            )
        os.chmod(temporary, 0o600)
        digest, byte_count = validate_static_onnx(temporary)
        os.link(temporary, target)
        _fsync_directory(target.parent)
        return digest, byte_count
    finally:
        temporary.unlink(missing_ok=True)


class _StaticBatchOneExport(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).reshape(1, EMBEDDING_DIM)


def validate_static_onnx(path: Path) -> tuple[str, int]:
    import onnx
    import onnxruntime as ort

    payload = path.read_bytes()
    model = onnx.load_model_from_string(payload)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("nose embedding ONNX must have one input and one output")
    model_input, model_output = model.graph.input[0], model.graph.output[0]
    if model_input.name != "images" or model_output.name != "embedding":
        raise ValueError("nose embedding ONNX tensor names differ")
    if _onnx_shape(model_input) != (1, 3, IMAGE_SIZE, IMAGE_SIZE) or _onnx_shape(
        model_output
    ) != (1, EMBEDDING_DIM):
        raise ValueError("nose embedding ONNX tensor shapes differ")
    if any(initializer.external_data for initializer in model.graph.initializer):
        raise ValueError("nose embedding ONNX must not use external data")
    session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    if tuple(session.get_inputs()[0].shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE) or tuple(
        session.get_outputs()[0].shape
    ) != (1, EMBEDDING_DIM):
        raise ValueError("CPU ONNX Runtime tensor contract differs")
    output = session.run(
        ["embedding"],
        {"images": np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)},
    )[0]
    _validate_embedding_output(output, "CPU ONNX Runtime smoke")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def produce_parity_receipt(
    *,
    model: torch.nn.Module,
    onnx_path: Path,
    runtime_manifest: NoseEmbeddingManifest,
    crop_root: Path,
    crop_records: Sequence[Mapping[str, Any]],
    crop_manifest_file_sha256: str,
    source_weights_sha256: str,
    weight_intake_receipt_sha256: str,
    preprocessor_intake_receipt_sha256: str,
    thresholds: ParityThresholds,
) -> ModelParityReceipt:
    """Compare selected PyTorch and CPU ORT on real crops and synthetic probes."""

    import onnxruntime as ort

    if not crop_records:
        raise ValueError("parity requires receipt-bound nose crops")
    model = model.to(torch.device("cpu")).eval()
    session = ort.InferenceSession(
        onnx_path.read_bytes(), providers=["CPUExecutionProvider"]
    )
    fixtures: list[tuple[str, ParityFixtureKind, str, Image.Image]] = []
    for record in sorted(crop_records, key=lambda item: item["sample_token"]):
        path = crop_root.joinpath(*Path(record["crop_path"]).parts)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["crop_sha256"]:
            raise ValueError("parity crop bytes differ from the bound manifest")
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        fixtures.append(
            (
                f"crop-{record['sample_token']}",
                ParityFixtureKind.RECEIPT_BOUND_CROP,
                record["crop_sha256"],
                image,
            )
        )
    fixtures.extend(_synthetic_parity_fixtures())
    results: list[ParityFixtureResult] = []
    for fixture_id, fixture_kind, input_sha256, image in sorted(fixtures):
        tensor = preprocess_image(image, runtime_manifest)
        with torch.inference_mode():
            reference = model(torch.from_numpy(tensor)).cpu().numpy()
        candidate = session.run(["embedding"], {"images": tensor})[0]
        _validate_embedding_output(reference, f"PyTorch parity {fixture_id}")
        _validate_embedding_output(candidate, f"ORT parity {fixture_id}")
        difference = np.abs(reference - candidate)
        absolute_error = float(np.max(difference))
        relative_error = float(
            np.max(
                difference
                / np.maximum(
                    np.abs(reference), np.float32(thresholds.relative_error_floor)
                )
            )
        )
        cosine = float(reference[0] @ candidate[0])
        if (
            absolute_error > thresholds.maximum_absolute_error
            or relative_error > thresholds.maximum_relative_error
            or cosine < thresholds.minimum_cosine_similarity
        ):
            raise RuntimeError(
                f"nose embedding parity failed for {fixture_id}: "
                f"abs={absolute_error}, rel={relative_error}, cosine={cosine}"
            )
        results.append(
            ParityFixtureResult(
                fixture_id=fixture_id,
                fixture_kind=fixture_kind,
                input_sha256=input_sha256,
                reference_output_sha256=hashlib.sha256(
                    np.ascontiguousarray(reference).tobytes()
                ).hexdigest(),
                candidate_output_sha256=hashlib.sha256(
                    np.ascontiguousarray(candidate).tobytes()
                ).hexdigest(),
                maximum_absolute_error=absolute_error,
                maximum_relative_error=relative_error,
                cosine_similarity=min(1.0, cosine),
                decision="PASS",
            )
        )
    return ModelParityReceipt(
        model_id=MODEL_ID,
        artifact_sha256=runtime_manifest.artifact_sha256,
        source_weights_sha256=source_weights_sha256,
        weight_intake_receipt_sha256=weight_intake_receipt_sha256,
        preprocessing_sha256=content_sha256(runtime_manifest.preprocessing.to_dict()),
        preprocessor_intake_receipt_sha256=preprocessor_intake_receipt_sha256,
        usage_lane=ModelUsageLane.RESEARCH_ONLY,
        reference_backend=f"torch={torch.__version__};selected-checkpoint",
        candidate_backend=f"onnxruntime-cpu={ort.__version__}",
        thresholds=thresholds,
        fixture_panel_receipt_sha256=crop_manifest_file_sha256,
        fixtures=tuple(results),
        decision="PASS",
    )


def _synthetic_parity_fixtures(
) -> tuple[tuple[str, ParityFixtureKind, str, Image.Image], ...]:
    results = []
    for name, width, height, offset in (
        ("synthetic-gradient", 91, 67, 17),
        ("synthetic-checker", 73, 109, 41),
    ):
        y, x = np.indices((height, width), dtype=np.uint32)
        array = np.stack(
            (
                (x + offset) % 256,
                (3 * y + offset) % 256,
                ((x // 5 + y // 7 + offset) % 2) * 255,
            ),
            axis=2,
        ).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False, compress_level=9)
        results.append(
            (
                name,
                ParityFixtureKind.SYNTHETIC,
                hashlib.sha256(output.getvalue()).hexdigest(),
                image,
            )
        )
    return tuple(results)


def _validate_embedding_output(output: np.ndarray, context: str) -> None:
    if (
        not isinstance(output, np.ndarray)
        or output.dtype != np.float32
        or output.shape != (1, EMBEDDING_DIM)
        or not np.isfinite(output).all()
    ):
        raise ValueError(f"{context} output must be finite float32 [1,384]")
    norm = float(np.linalg.norm(output[0]))
    if not math.isclose(norm, 1.0, abs_tol=1e-4, rel_tol=0.0):
        raise ValueError(f"{context} output must have unit L2 norm")


def train_and_export(
    *,
    manifest_path: Path,
    model_directory: Path,
    weight_intake_bundle: Path,
    preprocessor_intake_bundle: Path,
    output_dir: Path,
    device_name: str,
    epochs: int,
    batch_size: int,
    samples_per_identity: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    arcface_scale: float,
    arcface_margin: float,
    embedding_consistency_weight: float,
    label_smoothing: float,
    freeze_backbone_epochs: int,
    num_workers: int,
    seed: int,
    gradient_checkpointing: bool,
    mixed_precision: bool,
    parity_crop_count: int,
    parity_thresholds: ParityThresholds,
) -> dict[str, Any]:
    """Train all TRAIN identities, select on DEV, and export the selected model."""

    _validate_training_arguments(
        epochs=epochs,
        batch_size=batch_size,
        samples_per_identity=samples_per_identity,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        embedding_consistency_weight=embedding_consistency_weight,
        label_smoothing=label_smoothing,
        freeze_backbone_epochs=freeze_backbone_epochs,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        parity_crop_count=parity_crop_count,
    )
    output_root = _create_private_output_directory(output_dir)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(mode=0o700)
    try:
        manifest_document = read_strict_json_document(
            manifest_path,
            maximum_bytes=536_870_912,
            maximum_nodes=10_000_000,
            maximum_keys=5_000_000,
            maximum_array_length=1_000_000,
        )
        manifest = read_nose_region_manifest(manifest_path)
        if content_sha256(manifest) != manifest_document.payload["manifest_sha256"]:
            raise RuntimeError("nose-region manifest changed between strict reads")
        records = tuple(manifest["records"])
        train_records = tuple(item for item in records if item["split_role"] == "TRAIN")
        dev_records = tuple(item for item in records if item["split_role"] == "DEV")
        if not train_records or not dev_records:
            raise ValueError("nose-region embedding training requires TRAIN and DEV")
        protocol = build_dev_protocol(dev_records)
        train_identities = sorted(
            {item["registered_dog_id"] for item in train_records}
        )
        dev_identities = sorted({item["registered_dog_id"] for item in dev_records})
        if set(train_identities) & set(dev_identities):
            raise ValueError("TRAIN and DEV identity populations overlap")
        identity_to_index = {
            identity: index for index, identity in enumerate(train_identities)
        }

        model, contract = load_receipt_bound_dinov2(
            model_directory=model_directory,
            weight_intake_bundle=weight_intake_bundle,
            preprocessor_intake_bundle=preprocessor_intake_bundle,
            gradient_checkpointing=gradient_checkpointing,
        )
        processor = contract.preprocessor
        training_config = {
            "schema_version": "cvi.nose_region_rgb_embedding_training_config.v1",
            "model": "DINOv2-small CLS L2-normalized 384D",
            "input": {
                "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
                "resize": "DIRECT_BICUBIC_STRETCH",
                "scale": processor["rescale_factor"],
                "mean": processor["image_mean"],
                "std": processor["image_std"],
            },
            "augmentation": {
                "horizontal_flip_probability": 0.5,
                "rotation_degrees": 5.0,
                "translation_fraction": 0.03,
                "brightness": [0.9, 1.1],
                "contrast": [0.9, 1.1],
                "saturation": [0.95, 1.05],
                "interpretation": "MILD_REGION_SAFE_NO_GENERIC_RANDAUGMENT",
            },
            "optimizer": {
                "type": "AdamW",
                "backbone_lr": backbone_lr,
                "arcface_head_lr": head_lr,
                "weight_decay": weight_decay,
            },
            "arcface": {"scale": arcface_scale, "margin": arcface_margin},
            "embedding_consistency_weight": embedding_consistency_weight,
            "label_smoothing": label_smoothing,
            "freeze_backbone_epochs": freeze_backbone_epochs,
            "epochs": epochs,
            "batch_size": batch_size,
            "samples_per_identity_per_epoch": samples_per_identity,
            "num_workers": num_workers,
            "seed": seed,
            "gradient_checkpointing": gradient_checkpointing,
            "mixed_precision": mixed_precision,
            "device": device_name,
            "parity": {
                "receipt_bound_crop_count": parity_crop_count,
                "thresholds": parity_thresholds.to_dict(),
            },
            "selection_metric_order": list(SELECTION_METRICS),
            "selection_interpretation": DEV_INTERPRETATION,
        }
        bindings = {
            "training_code": {
                "embedding_training_sha256": _sha256_file(Path(__file__)),
                "tool_sha256": _sha256_file(
                    Path(__file__).parents[2] / "workflows/train_nose_region_embedding.py"
                ),
            },
            "crop_manifest": {
                "payload_sha256": manifest_document.payload["manifest_sha256"],
                "file_sha256": manifest_document.raw_sha256,
                "protocol_plan_sha256": manifest["summary"][
                    "protocol_plan_sha256"
                ],
                "summary_sha256": manifest["summary"]["summary_sha256"],
            },
            "identity_populations": {
                "train_registered_dog_ids": train_identities,
                "dev_registered_dog_ids": dev_identities,
                "evaluation_eligible_registered_dog_ids": protocol[
                    "eligible_registered_dog_ids"
                ],
                "excluded_singleton_registered_dog_ids": protocol[
                    "excluded_singleton_registered_dog_ids"
                ],
            },
            "development_protocol": protocol,
            "dinov2": {
                "source_model_id": contract.weight_source.source_model_id,
                "source_revision": contract.weight_source.source_revision,
                "weight_sha256": contract.model_sha256,
                "preprocessor_sha256": contract.preprocessor_sha256,
                "config_sha256": contract.config_sha256,
                "weight_intake_receipt_sha256": contract.weight_receipt_sha256,
                "preprocessor_intake_receipt_sha256": (
                    contract.preprocessor_receipt_sha256
                ),
                "weight_intake_bundle_file_sha256": _sha256_file(
                    weight_intake_bundle
                ),
                "preprocessor_intake_bundle_file_sha256": _sha256_file(
                    preprocessor_intake_bundle
                ),
            },
            "license": {
                "license_id": LICENSE_ID,
                "usage_lane": UsageLane.RESEARCH_ONLY.value,
                "inherited_crop_licensing_lanes": sorted(
                    {item["licensing_lane"] for item in records}
                ),
            },
        }

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device_name not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device_name == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        device = torch.device(device_name)
        model.to(device)
        reference_model = None
        if embedding_consistency_weight > 0.0:
            reference_model = copy.deepcopy(model).to(device).eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
        arcface = ArcFaceClassificationHead(
            len(identity_to_index), scale=arcface_scale, margin=arcface_margin
        ).to(device)
        train_labels = [
            identity_to_index[item["registered_dog_id"]] for item in train_records
        ]
        train_dataset = NoseRegionCropDataset(
            manifest_path.parent,
            train_records,
            train_labels,
            mean=processor["image_mean"],
            std=processor["image_std"],
            training=True,
            seed=seed,
        )
        dev_dataset = NoseRegionCropDataset(
            manifest_path.parent,
            dev_records,
            [0] * len(dev_records),
            mean=processor["image_mean"],
            std=processor["image_std"],
            training=False,
            seed=seed,
        )
        sampler = IdentityBalancedBatchSampler(
            train_labels,
            batch_size=batch_size,
            samples_per_identity=samples_per_identity,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": backbone_lr},
                {"params": arcface.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        scaler = (
            torch.amp.GradScaler("cuda")
            if device.type == "cuda" and mixed_precision
            else None
        )
        history: list[dict[str, Any]] = []
        global_step = 0
        best_key = (-1.0, -1.0)
        best_epoch = -1
        for epoch in range(epochs + 1):
            if epoch > 0:
                train_dataset.set_epoch(epoch - 1)
                sampler.set_epoch(epoch - 1)
                backbone_trainable = epoch - 1 >= freeze_backbone_epochs
                for parameter in model.parameters():
                    parameter.requires_grad_(backbone_trainable)
                train_metrics, steps = _train_epoch(
                    model,
                    arcface,
                    train_loader,
                    optimizer,
                    device,
                    scaler,
                    reference_model=reference_model,
                    embedding_consistency_weight=embedding_consistency_weight,
                    label_smoothing=label_smoothing,
                )
                global_step += steps
            else:
                train_metrics = None
                backbone_trainable = False
            dev_embeddings = _extract_embeddings(
                model,
                dev_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            metrics = evaluate_dev_leave_one_out(
                dev_embeddings, dev_records, protocol
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "backbone_trainable": backbone_trainable,
                    "mAP": metrics["mAP"],
                    "Rank-1": metrics["Rank-1"],
                    "query_count": metrics["query_count"],
                }
            )
            selection = {
                "metric_order": list(SELECTION_METRICS),
                "mAP": metrics["mAP"],
                "Rank-1": metrics["Rank-1"],
                "interpretation": DEV_INTERPRETATION,
            }
            checkpoint = build_embedding_checkpoint(
                model=model,
                arcface=arcface,
                epoch=epoch,
                global_step=global_step,
                selection=selection,
                bindings=bindings,
                training_config=training_config,
            )
            selection_key = (metrics["mAP"], metrics["Rank-1"])
            if selection_key > best_key:
                best_key = selection_key
                best_epoch = epoch
                replace_embedding_checkpoint(checkpoint_dir / "best.pt", checkpoint)
            if epoch > 0:
                replace_embedding_checkpoint(checkpoint_dir / "last.pt", checkpoint)

        best_path = checkpoint_dir / "best.pt"
        last_path = checkpoint_dir / "last.pt"
        selected = load_embedding_checkpoint(
            best_path,
            expected_bindings=bindings,
            expected_training_config=training_config,
        )
        model.load_state_dict(selected["model_state_dict"], strict=True)
        model.to(torch.device("cpu")).eval()
        onnx_path = output_root / "nose_embedding.onnx"
        onnx_sha256, onnx_bytes = export_static_onnx(model, onnx_path)
        runtime_manifest = build_runtime_manifest(onnx_path)
        if runtime_manifest.artifact_sha256 != onnx_sha256:
            raise RuntimeError("runtime manifest ONNX digest differs")
        runtime_path = output_root / "nose_embedding.runtime.json"
        write_private_json_bundle(((runtime_path, runtime_manifest.to_dict()),))

        parity_records = tuple(
            sorted(dev_records, key=lambda item: item["sample_token"])[
                :parity_crop_count
            ]
        )
        parity = produce_parity_receipt(
            model=model,
            onnx_path=onnx_path,
            runtime_manifest=runtime_manifest,
            crop_root=manifest_path.parent,
            crop_records=parity_records,
            crop_manifest_file_sha256=manifest_document.raw_sha256,
            source_weights_sha256=contract.model_sha256,
            weight_intake_receipt_sha256=contract.weight_receipt_sha256,
            preprocessor_intake_receipt_sha256=(
                contract.preprocessor_receipt_sha256
            ),
            thresholds=parity_thresholds,
        )
        parity_path = output_root / "nose_embedding.parity.json"
        write_private_json_bundle(((parity_path, parity.to_dict()),))
        dev_report = {
            "schema_version": DEV_REPORT_SCHEMA,
            "interpretation": DEV_INTERPRETATION,
            "selection_metric_order": list(SELECTION_METRICS),
            "best_epoch": best_epoch,
            "best_mAP": best_key[0],
            "best_Rank-1": best_key[1],
            "eligible_registered_dog_ids": protocol[
                "eligible_registered_dog_ids"
            ],
            "excluded_singleton_registered_dog_ids": protocol[
                "excluded_singleton_registered_dog_ids"
            ],
            "eligible_image_count": protocol["eligible_image_count"],
            "history": history,
        }
        dev_path = output_root / "dev_selection.json"
        write_private_json_bundle(((dev_path, dev_report),))

        lineage = _build_lineage(
            output_root=output_root,
            onnx_path=onnx_path,
            onnx_bytes=onnx_bytes,
            runtime_path=runtime_path,
            parity_path=parity_path,
            parity=parity,
            best_path=best_path,
            last_path=last_path,
            dev_path=dev_path,
            dev_report=dev_report,
            bindings=bindings,
            training_config=training_config,
        )
        lineage_path = output_root / "artifact_lineage.json"
        write_private_json_bundle(((lineage_path, lineage),))
        validate_lineage_manifest(lineage, output_root)
        return lineage
    except BaseException:
        # Keep the private, non-overwriting output for forensic inspection.
        raise


def _train_epoch(
    model: NoseEmbeddingModel,
    arcface: ArcFaceClassificationHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    *,
    reference_model: NoseEmbeddingModel | None,
    embedding_consistency_weight: float,
    label_smoothing: float,
) -> tuple[dict[str, float], int]:
    model.train()
    arcface.train()
    total_loss = 0.0
    total_classification = 0.0
    total_consistency = 0.0
    total_samples = 0
    steps = 0
    for images, labels, _ in loader:
        images = images.to(device=device, non_blocking=device.type == "cuda")
        labels = labels.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            embeddings = model(images)
            classification = F.cross_entropy(
                arcface(embeddings, labels),
                labels,
                label_smoothing=label_smoothing,
            )
            consistency = classification.new_zeros(())
            if reference_model is not None:
                with torch.no_grad():
                    reference = reference_model(images)
                consistency = 1.0 - F.cosine_similarity(
                    embeddings.float(), reference.float(), dim=1
                ).mean()
            loss = classification + embedding_consistency_weight * consistency
        if not torch.isfinite(loss):
            raise RuntimeError("ArcFace training loss became non-finite")
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *arcface.parameters()], max_norm=5.0
            )
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *arcface.parameters()], max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()
        batch_count = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_count
        total_classification += float(classification.detach()) * batch_count
        total_consistency += float(consistency.detach()) * batch_count
        total_samples += batch_count
        steps += 1
    if total_samples <= 0:
        raise RuntimeError("identity-balanced training loader was empty")
    return {
        "total": total_loss / total_samples,
        "classification": total_classification / total_samples,
        "embedding_consistency": total_consistency / total_samples,
    }, steps


def _extract_embeddings(
    model: NoseEmbeddingModel,
    dataset: NoseRegionCropDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model.eval()
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _, _ in loader:
            outputs = model(images.to(device=device, non_blocking=device.type == "cuda"))
            results.append(outputs.cpu().numpy().astype(np.float32, copy=False))
    return np.ascontiguousarray(np.concatenate(results, axis=0), dtype=np.float32)


def _build_lineage(
    *,
    output_root: Path,
    onnx_path: Path,
    onnx_bytes: int,
    runtime_path: Path,
    parity_path: Path,
    parity: ModelParityReceipt,
    best_path: Path,
    last_path: Path,
    dev_path: Path,
    dev_report: Mapping[str, Any],
    bindings: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> dict[str, Any]:
    def artifact(path: Path) -> dict[str, Any]:
        if path.parent != output_root and path.parent != output_root / "checkpoints":
            raise ValueError("lineage artifact is outside the output directory")
        relative = path.relative_to(output_root).as_posix()
        return {
            "path": relative,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }

    lineage = {
        "schema_version": LINEAGE_SCHEMA,
        "artifacts": {
            "onnx": artifact(onnx_path),
            "runtime_manifest": artifact(runtime_path),
            "parity_receipt": artifact(parity_path),
            "selected_checkpoint": artifact(best_path),
            "last_checkpoint": artifact(last_path),
            "dev_selection": artifact(dev_path),
        },
        "onnx_contract": {
            "input_name": "images",
            "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "output_name": "embedding",
            "output_shape": [1, EMBEDDING_DIM],
            "opset": 18,
            "external_data": False,
            "onnx_bytes": onnx_bytes,
        },
        "bindings": dict(bindings),
        "bindings_sha256": content_sha256(dict(bindings)),
        "training_config": dict(training_config),
        "training_config_sha256": content_sha256(dict(training_config)),
        "dev_selection_payload_sha256": content_sha256(dict(dev_report)),
        "parity_payload_sha256": parity.receipt_sha256,
        "license": {
            "license_id": LICENSE_ID,
            "usage_lane": UsageLane.RESEARCH_ONLY.value,
        },
        "interpretation": DEV_INTERPRETATION,
    }
    lineage["lineage_sha256"] = content_sha256(lineage)
    return lineage


def validate_lineage_manifest(payload: object, root: Path) -> None:
    expected = {
        "schema_version",
        "artifacts",
        "onnx_contract",
        "bindings",
        "bindings_sha256",
        "training_config",
        "training_config_sha256",
        "dev_selection_payload_sha256",
        "parity_payload_sha256",
        "license",
        "interpretation",
        "lineage_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("nose embedding lineage keys differ")
    if payload["schema_version"] != LINEAGE_SCHEMA:
        raise ValueError("unsupported nose embedding lineage schema")
    expected_digest = content_sha256(
        {key: value for key, value in payload.items() if key != "lineage_sha256"}
    )
    if payload["lineage_sha256"] != expected_digest:
        raise ValueError("nose embedding lineage digest differs")
    if content_sha256(payload["bindings"]) != payload["bindings_sha256"]:
        raise ValueError("nose embedding lineage bindings digest differs")
    _validate_checkpoint_bindings(payload["bindings"])
    if content_sha256(payload["training_config"]) != payload[
        "training_config_sha256"
    ]:
        raise ValueError("nose embedding lineage config digest differs")
    _validate_checkpoint_training_config(payload["training_config"])
    if payload["license"] != {
        "license_id": LICENSE_ID,
        "usage_lane": UsageLane.RESEARCH_ONLY.value,
    } or payload["interpretation"] != DEV_INTERPRETATION:
        raise ValueError("nose embedding lineage license or interpretation differs")
    artifacts = payload["artifacts"]
    expected_artifacts = {
        "onnx",
        "runtime_manifest",
        "parity_receipt",
        "selected_checkpoint",
        "last_checkpoint",
        "dev_selection",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("nose embedding lineage artifacts differ")
    expected_paths = {
        "onnx": "nose_embedding.onnx",
        "runtime_manifest": "nose_embedding.runtime.json",
        "parity_receipt": "nose_embedding.parity.json",
        "selected_checkpoint": "checkpoints/best.pt",
        "last_checkpoint": "checkpoints/last.pt",
        "dev_selection": "dev_selection.json",
    }
    resolved_root = root.resolve(strict=True)
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "bytes"}:
            raise ValueError("nose embedding lineage artifact keys differ")
        if artifact["path"] != expected_paths[name]:
            raise ValueError("nose embedding lineage artifact name differs")
        relative = Path(artifact["path"])
        path = resolved_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve(strict=True).is_relative_to(resolved_root)
        ):
            raise ValueError("nose embedding lineage artifact path is unsafe")
        if path.stat().st_size != artifact["bytes"] or _sha256_file(path) != artifact[
            "sha256"
        ]:
            raise ValueError("nose embedding lineage artifact bytes differ")
    runtime_payload = read_strict_json_object(
        resolved_root / artifacts["runtime_manifest"]["path"]
    )
    runtime_manifest = NoseEmbeddingManifest.from_dict(runtime_payload)
    if (
        runtime_manifest.artifact_id != MODEL_ID
        or runtime_manifest.artifact_sha256 != artifacts["onnx"]["sha256"]
        or runtime_manifest.license
        != ArtifactLicense(LICENSE_ID, UsageLane.RESEARCH_ONLY)
    ):
        raise ValueError("nose embedding runtime manifest lineage differs")
    expected_onnx_contract = {
        "input_name": "images",
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "output_name": "embedding",
        "output_shape": [1, EMBEDDING_DIM],
        "opset": 18,
        "external_data": False,
        "onnx_bytes": artifacts["onnx"]["bytes"],
    }
    if payload["onnx_contract"] != expected_onnx_contract:
        raise ValueError("nose embedding lineage ONNX contract differs")
    parity_payload = read_strict_json_object(
        resolved_root / artifacts["parity_receipt"]["path"]
    )
    parity = ModelParityReceipt.from_dict(parity_payload)
    if parity.receipt_sha256 != payload["parity_payload_sha256"]:
        raise ValueError("nose embedding lineage parity payload differs")
    dino = payload["bindings"]["dinov2"]
    if (
        parity.model_id != MODEL_ID
        or parity.artifact_sha256 != artifacts["onnx"]["sha256"]
        or parity.source_weights_sha256 != dino["weight_sha256"]
        or parity.weight_intake_receipt_sha256
        != dino["weight_intake_receipt_sha256"]
        or parity.preprocessor_intake_receipt_sha256
        != dino["preprocessor_intake_receipt_sha256"]
        or parity.usage_lane is not ModelUsageLane.RESEARCH_ONLY
    ):
        raise ValueError("nose embedding lineage parity binding differs")
    dev_payload = read_strict_json_object(
        resolved_root / artifacts["dev_selection"]["path"]
    )
    if content_sha256(dev_payload) != payload["dev_selection_payload_sha256"]:
        raise ValueError("nose embedding lineage DEV selection differs")
    if (
        dev_payload.get("schema_version") != DEV_REPORT_SCHEMA
        or dev_payload.get("interpretation") != DEV_INTERPRETATION
        or dev_payload.get("selection_metric_order") != list(SELECTION_METRICS)
    ):
        raise ValueError("nose embedding lineage DEV report contract differs")
    selected = load_embedding_checkpoint(
        resolved_root / artifacts["selected_checkpoint"]["path"],
        expected_bindings=payload["bindings"],
        expected_training_config=payload["training_config"],
    )
    last = load_embedding_checkpoint(
        resolved_root / artifacts["last_checkpoint"]["path"],
        expected_bindings=payload["bindings"],
        expected_training_config=payload["training_config"],
    )
    if (
        selected["epoch"] != dev_payload["best_epoch"]
        or selected["selection"]["mAP"] != dev_payload["best_mAP"]
        or selected["selection"]["Rank-1"] != dev_payload["best_Rank-1"]
        or last["epoch"] != payload["training_config"]["epochs"]
    ):
        raise ValueError("nose embedding lineage checkpoint selection differs")


def _validate_checkpoint_bindings(bindings: object) -> None:
    expected = {
        "crop_manifest",
        "training_code",
        "identity_populations",
        "development_protocol",
        "dinov2",
        "license",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected:
        raise ValueError("checkpoint binding keys differ")
    training_code = bindings["training_code"]
    if not isinstance(training_code, dict) or set(training_code) != {
        "embedding_training_sha256",
        "tool_sha256",
    }:
        raise ValueError("checkpoint training-code binding keys differ")
    for name, value in training_code.items():
        _require_sha256(value, f"training_code.{name}")
    crop = bindings["crop_manifest"]
    if not isinstance(crop, dict) or set(crop) != {
        "payload_sha256",
        "file_sha256",
        "protocol_plan_sha256",
        "summary_sha256",
    }:
        raise ValueError("checkpoint crop-manifest binding keys differ")
    for name, value in crop.items():
        _require_sha256(value, f"crop_manifest.{name}")
    populations = bindings["identity_populations"]
    population_keys = {
        "train_registered_dog_ids",
        "dev_registered_dog_ids",
        "evaluation_eligible_registered_dog_ids",
        "excluded_singleton_registered_dog_ids",
    }
    if not isinstance(populations, dict) or set(populations) != population_keys:
        raise ValueError("checkpoint identity population keys differ")
    for name, values in populations.items():
        if (
            not isinstance(values, list)
            or values != sorted(values)
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"checkpoint identity population {name} differs")
    train = set(populations["train_registered_dog_ids"])
    dev = set(populations["dev_registered_dog_ids"])
    eligible = set(populations["evaluation_eligible_registered_dog_ids"])
    excluded = set(populations["excluded_singleton_registered_dog_ids"])
    if not train or not dev or train & dev or eligible | excluded != dev or eligible & excluded:
        raise ValueError("checkpoint identity populations are inconsistent")
    protocol = bindings["development_protocol"]
    protocol_keys = {
        "interpretation",
        "identity_disjoint_from_train",
        "same_session_within_each_dev_identity",
        "selection_metric_order",
        "dev_registered_dog_ids",
        "eligible_registered_dog_ids",
        "excluded_singleton_registered_dog_ids",
        "eligible_sample_tokens",
        "eligible_image_count",
    }
    if not isinstance(protocol, dict) or set(protocol) != protocol_keys:
        raise ValueError("checkpoint development protocol keys differ")
    if (
        protocol["interpretation"] != DEV_INTERPRETATION
        or protocol["identity_disjoint_from_train"] is not True
        or protocol["same_session_within_each_dev_identity"] is not True
        or protocol["selection_metric_order"] != list(SELECTION_METRICS)
        or protocol["dev_registered_dog_ids"]
        != populations["dev_registered_dog_ids"]
        or protocol["eligible_registered_dog_ids"]
        != populations["evaluation_eligible_registered_dog_ids"]
        or protocol["excluded_singleton_registered_dog_ids"]
        != populations["excluded_singleton_registered_dog_ids"]
        or not isinstance(protocol["eligible_sample_tokens"], list)
        or protocol["eligible_sample_tokens"]
        != sorted(protocol["eligible_sample_tokens"])
        or protocol["eligible_image_count"]
        != len(protocol["eligible_sample_tokens"])
    ):
        raise ValueError("checkpoint development protocol differs")
    for token in protocol["eligible_sample_tokens"]:
        _require_sha256(token, "eligible_sample_token")
    dino = bindings["dinov2"]
    dino_keys = {
        "source_model_id",
        "source_revision",
        "weight_sha256",
        "preprocessor_sha256",
        "config_sha256",
        "weight_intake_receipt_sha256",
        "preprocessor_intake_receipt_sha256",
        "weight_intake_bundle_file_sha256",
        "preprocessor_intake_bundle_file_sha256",
    }
    if (
        not isinstance(dino, dict)
        or set(dino) != dino_keys
        or dino["source_model_id"] != "facebook/dinov2-small"
        or not isinstance(dino["source_revision"], str)
        or not dino["source_revision"]
    ):
        raise ValueError("checkpoint DINOv2 binding differs")
    for name in dino_keys - {"source_model_id", "source_revision"}:
        _require_sha256(dino[name], f"dinov2.{name}")
    license_binding = bindings["license"]
    if (
        not isinstance(license_binding, dict)
        or set(license_binding)
        != {"license_id", "usage_lane", "inherited_crop_licensing_lanes"}
        or license_binding["license_id"] != LICENSE_ID
        or license_binding["usage_lane"] != UsageLane.RESEARCH_ONLY.value
        or not isinstance(license_binding["inherited_crop_licensing_lanes"], list)
        or not license_binding["inherited_crop_licensing_lanes"]
        or license_binding["inherited_crop_licensing_lanes"]
        != sorted(set(license_binding["inherited_crop_licensing_lanes"]))
    ):
        raise ValueError("checkpoint research license binding differs")


def _validate_checkpoint_training_config(config: object) -> None:
    expected = {
        "schema_version",
        "model",
        "input",
        "augmentation",
        "optimizer",
        "arcface",
        "embedding_consistency_weight",
        "label_smoothing",
        "freeze_backbone_epochs",
        "epochs",
        "batch_size",
        "samples_per_identity_per_epoch",
        "num_workers",
        "seed",
        "gradient_checkpointing",
        "mixed_precision",
        "device",
        "parity",
        "selection_metric_order",
        "selection_interpretation",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("checkpoint training config keys differ")
    if (
        config["schema_version"]
        != "cvi.nose_region_rgb_embedding_training_config.v1"
        or config["model"] != "DINOv2-small CLS L2-normalized 384D"
        or config["selection_metric_order"] != list(SELECTION_METRICS)
        or config["selection_interpretation"] != DEV_INTERPRETATION
        or config["device"] not in {"cpu", "cuda"}
    ):
        raise ValueError("checkpoint training config contract differs")
    input_config = config["input"]
    if not isinstance(input_config, dict) or input_config != {
        "shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "resize": "DIRECT_BICUBIC_STRETCH",
        "scale": 1.0 / 255.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }:
        raise ValueError("checkpoint input preprocessing differs")
    augmentation = config["augmentation"]
    if (
        not isinstance(augmentation, dict)
        or augmentation.get("interpretation")
        != "MILD_REGION_SAFE_NO_GENERIC_RANDAUGMENT"
    ):
        raise ValueError("checkpoint augmentation contract differs")
    if not isinstance(config["optimizer"], dict) or not isinstance(
        config["arcface"], dict
    ):
        raise ValueError("checkpoint optimizer or ArcFace config differs")
    if (
        isinstance(config["embedding_consistency_weight"], bool)
        or not isinstance(config["embedding_consistency_weight"], (int, float))
        or not math.isfinite(config["embedding_consistency_weight"])
        or config["embedding_consistency_weight"] < 0.0
        or isinstance(config["label_smoothing"], bool)
        or not isinstance(config["label_smoothing"], (int, float))
        or not math.isfinite(config["label_smoothing"])
        or not 0.0 <= config["label_smoothing"] < 1.0
        or isinstance(config["freeze_backbone_epochs"], bool)
        or not isinstance(config["freeze_backbone_epochs"], int)
        or not 0 <= config["freeze_backbone_epochs"] <= config["epochs"]
    ):
        raise ValueError("checkpoint regularization config differs")
    for name in ("epochs", "batch_size", "samples_per_identity_per_epoch"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"checkpoint {name} must be positive")
    if (
        isinstance(config["num_workers"], bool)
        or not isinstance(config["num_workers"], int)
        or config["num_workers"] < 0
        or isinstance(config["seed"], bool)
        or not isinstance(config["seed"], int)
        or not isinstance(config["gradient_checkpointing"], bool)
        or not isinstance(config["mixed_precision"], bool)
    ):
        raise ValueError("checkpoint worker, seed, or gradient config differs")
    parity = config["parity"]
    if (
        not isinstance(parity, dict)
        or set(parity) != {"receipt_bound_crop_count", "thresholds"}
        or isinstance(parity["receipt_bound_crop_count"], bool)
        or not isinstance(parity["receipt_bound_crop_count"], int)
        or parity["receipt_bound_crop_count"] <= 0
        or not isinstance(parity["thresholds"], dict)
    ):
        raise ValueError("checkpoint parity config differs")
    ParityThresholds.from_dict(parity["thresholds"])


def _validate_training_arguments(**values: Any) -> None:
    for name in (
        "epochs",
        "batch_size",
        "samples_per_identity",
        "parity_crop_count",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        isinstance(values["num_workers"], bool)
        or not isinstance(values["num_workers"], int)
        or values["num_workers"] < 0
    ):
        raise ValueError("num_workers must be nonnegative")
    if not isinstance(values["mixed_precision"], bool):
        raise ValueError("mixed_precision must be boolean")
    for name in ("backbone_lr", "head_lr", "weight_decay"):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if name != "weight_decay" else value < 0)
        ):
            raise ValueError(f"{name} is invalid")
    consistency = values["embedding_consistency_weight"]
    smoothing = values["label_smoothing"]
    freeze_epochs = values["freeze_backbone_epochs"]
    if (
        isinstance(consistency, bool)
        or not isinstance(consistency, (int, float))
        or not math.isfinite(consistency)
        or consistency < 0.0
        or isinstance(smoothing, bool)
        or not isinstance(smoothing, (int, float))
        or not math.isfinite(smoothing)
        or not 0.0 <= smoothing < 1.0
        or isinstance(freeze_epochs, bool)
        or not isinstance(freeze_epochs, int)
        or not 0 <= freeze_epochs <= values["epochs"]
    ):
        raise ValueError("training regularization arguments are invalid")


def _create_private_output_directory(path: Path) -> Path:
    if os.path.lexists(path):
        raise FileExistsError("output directory must be new and non-existing")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    target = parent / path.name
    target.mkdir(mode=0o700, exist_ok=False)
    return target


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"hash input must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _onnx_shape(value_info: object) -> tuple[int, ...]:
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:  # type: ignore[attr-defined]
        if dimension.dim_value <= 0:
            raise ValueError("nose embedding ONNX dimensions must be static")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEV_INTERPRETATION",
    "EMBEDDING_DIM",
    "IdentityBalancedBatchSampler",
    "LICENSE_ID",
    "MODEL_ID",
    "NoseEmbeddingModel",
    "ArcFaceClassificationHead",
    "build_dev_protocol",
    "build_embedding_checkpoint",
    "build_runtime_manifest",
    "evaluate_dev_leave_one_out",
    "export_static_onnx",
    "load_embedding_checkpoint",
    "load_receipt_bound_dinov2",
    "produce_parity_receipt",
    "replace_embedding_checkpoint",
    "train_and_export",
    "validate_embedding_checkpoint",
    "validate_lineage_manifest",
    "validate_static_onnx",
]
