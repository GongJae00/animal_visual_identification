"""Score and run the frozen YT-BB-Dog → Sibetan comparable-transfer protocol.

CLI: ``uv run python -m evaluation.commands.evaluate comparable-transfer --help``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

from data.adapters import load as load_dataset
from data.types import CaptureGroupKind, UnifiedCanidSample
from evaluation.search_metrics.metrics import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
)
from evaluation.splits.comparable_transfer import (
    COMPARISON_VARIABLE,
    INTERPRETATION,
    PARSER_POLICY_SCHEMA,
    PROTOCOL_SCHEMA,
    SPLIT_SEED,
    ComparableTransferRow,
    ComparableTransferSplit,
    bind_crops,
    freeze_comparable_transfer,
)
from shared.contracts.identity_ids import compute_registered_dog_id, compute_sample_token
from shared.foundation.protected_io import (
    read_strict_json_object,
    write_private_json_bundle,
)
from shared.foundation.provenance import content_sha256

REPORT_SCHEMA = "evaluation.comparable_transfer_report.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "shared"
    / "contracts"
    / "schemas"
    / "evaluation.comparable_transfer.v1.schema.json"
)
_SMOKE_DIM = 8
_VIS_STAGES = (
    "parsing",
    "identification",
    "representation",
    "enrollment",
    "gallery",
    "search",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_split_document(payload: Mapping[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: list(item.path),
    )
    if errors:
        raise ValueError("comparable-transfer schema failed: " + errors[0].message)


def load_split(path: Path) -> ComparableTransferSplit:
    payload = read_strict_json_object(path)
    _validate_split_document(payload)
    split = ComparableTransferSplit.from_dict(payload)
    if payload["split_sha256"] != split.split_sha256:
        raise ValueError("split_sha256 disagrees with payload")
    if payload["gallery_list_sha256"] != split.gallery_list_sha256:
        raise ValueError("gallery_list_sha256 disagrees with payload")
    if payload["query_list_sha256"] != split.query_list_sha256:
        raise ValueError("query_list_sha256 disagrees with payload")
    return split


def write_split(path: Path, split: ComparableTransferSplit) -> dict[str, Any]:
    payload = split.to_dict()
    _validate_split_document(payload)
    write_private_json_bundle(((path, payload),))
    return payload


def _matrix_for_rows(
    rows: Sequence[ComparableTransferRow],
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    missing = [row.sample_id for row in rows if row.sample_id not in embeddings]
    if missing:
        raise ValueError(
            f"embeddings missing {len(missing)} frozen sample(s); first={missing[0]}"
        )
    stacked = np.stack([np.asarray(embeddings[row.sample_id]) for row in rows])
    if stacked.ndim != 2 or stacked.shape[0] != len(rows):
        raise ValueError("embedding matrix does not match frozen rows")
    return stacked.astype(np.float64, copy=False)


def score_comparable_transfer(
    split: ComparableTransferSplit,
    embeddings: Mapping[str, np.ndarray],
    *,
    backbone_id: str,
) -> dict[str, Any]:
    if not backbone_id or backbone_id != backbone_id.strip():
        raise ValueError("backbone_id must be non-empty")
    if split.crop_binding_status != "bound":
        raise ValueError("comparable-transfer scoring requires parser v6 crop binding")
    gallery = _matrix_for_rows(split.gallery, embeddings)
    query = _matrix_for_rows(split.query, embeddings)
    scores = compute_cosine_score_matrix(query, gallery)
    metrics = evaluate_multi_template_closed_set(
        scores,
        query_identity_ids=np.asarray([row.identity_id for row in split.query]),
        gallery_template_identity_ids=np.asarray(
            [row.identity_id for row in split.gallery]
        ),
        self_match_policy="exclude",
        query_template_ids=np.asarray([row.sample_id for row in split.query]),
        gallery_template_ids=np.asarray([row.sample_id for row in split.gallery]),
        rank_ks=(1, 5),
        aggregation="max",
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "interpretation": INTERPRETATION,
        "comparable": split.comparable,
        "split_sha256": split.split_sha256,
        "gallery_list_sha256": split.gallery_list_sha256,
        "query_list_sha256": split.query_list_sha256,
        "train_identity_sha256": split.train_identity_sha256,
        "parser_policy_schema": PARSER_POLICY_SCHEMA,
        "comparison_variable": COMPARISON_VARIABLE,
        "backbone_id": backbone_id,
        "counts": {
            "train_identities": len(split.train_identities),
            "train_samples": len(split.train_samples),
            "gallery_templates": len(split.gallery),
            "queries": len(split.query),
            "eval_identities": len({row.identity_id for row in split.gallery}),
        },
        "metrics": {
            "Rank-1": metrics["Rank-1"],
            "Rank-5": metrics["Rank-5"],
            "mAP": metrics["mAP"],
        },
    }
    report["report_sha256"] = content_sha256(report)
    return report


def assert_backbone_only_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    shared = (
        "split_sha256",
        "gallery_list_sha256",
        "query_list_sha256",
        "train_identity_sha256",
        "parser_policy_schema",
    )
    for key in shared:
        if left.get(key) != right.get(key):
            raise ValueError(f"comparison changed {key}; only backbone may change")
    if left.get("backbone_id") == right.get("backbone_id"):
        raise ValueError("backbone_id must differ when comparing backbones")


def _sample(
    *,
    dataset: str,
    identity: str,
    sequence: str,
    frame: int,
    split_role: str,
    kind: CaptureGroupKind,
) -> UnifiedCanidSample:
    dataset_identity = f"{dataset}:v1:fixture:{identity}"
    sample_key = f"{dataset}:{identity}:{sequence}:{frame}"
    digest = _sha256_bytes(sample_key.encode("utf-8"))
    return UnifiedCanidSample(
        sample_id=compute_sample_token(sample_key),
        dataset_name=dataset,
        dataset_version="publisher-v1-2025-10-27",
        source_group_id=sequence,
        image_path=f"{dataset}/{identity}/{sequence}/{frame}.png",
        image_sha256=digest,
        width=16,
        height=16,
        registered_identity_id=compute_registered_dog_id(dataset_identity),
        raw_identity_id=identity,
        capture_group_id=sequence,
        capture_group_kind=kind,
        split_role=split_role,
    )


def smoke_samples() -> tuple[tuple[UnifiedCanidSample, ...], tuple[UnifiedCanidSample, ...]]:
    train = tuple(
        _sample(
            dataset="yt-bb-dog",
            identity=f"t{index}",
            sequence=f"t{index}",
            frame=frame,
            split_role="train",
            kind=CaptureGroupKind.VIDEO_TRACK,
        )
        for index in range(4)
        for frame in (0, 1)
    )
    held_out_test = _sample(
        dataset="yt-bb-dog",
        identity="test-hold",
        sequence="test-hold",
        frame=0,
        split_role="test",
        kind=CaptureGroupKind.VIDEO_TRACK,
    )
    eval_samples = tuple(
        _sample(
            dataset="sibetan",
            identity=f"s{index}",
            sequence=f"s{index}-seq{sequence}",
            frame=frame,
            split_role="UNASSIGNED",
            kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
        )
        for index in range(3)
        for sequence in (0, 1)
        for frame in (0, 1)
    )
    return train + (held_out_test,), eval_samples


def _digest_vector(token: str, dim: int) -> np.ndarray:
    raw = np.frombuffer(hashlib.sha256(token.encode("utf-8")).digest(), dtype=np.uint8)
    return raw[:dim].astype(np.float64)


def _unit_from_tokens(identity_id: str, sample_id: str, *, dim: int = _SMOKE_DIM) -> np.ndarray:
    acc = 32.0 * _digest_vector(identity_id, dim) + 0.05 * _digest_vector(sample_id, dim)
    norm = float(np.linalg.norm(acc))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("smoke embedding is degenerate")
    return acc / norm


def smoke_embeddings(split: ComparableTransferSplit) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for row in (*split.train_samples, *split.gallery, *split.query):
        vectors[row.sample_id] = _unit_from_tokens(
            row.identity_id, row.sample_id, dim=_SMOKE_DIM
        )
    return vectors


def _write_smoke_crops(
    split: ComparableTransferSplit, crop_dir: Path
) -> dict[str, str]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    bound: dict[str, str] = {}
    for row in (*split.train_samples, *split.gallery, *split.query):
        color = (
            int(row.raw_identity_id.encode("utf-8")[0]) % 200 + 20,
            80,
            140,
        )
        image = Image.new("RGB", (16, 16), color)
        path = crop_dir / f"{row.sample_id}.png"
        image.save(path, format="PNG")
        bound[row.sample_id] = _sha256_bytes(path.read_bytes())
    return bound


def _ordered_embeddings(
    rows: Sequence[ComparableTransferRow],
    embeddings: Mapping[str, np.ndarray],
) -> list[list[float]]:
    return [embeddings[row.sample_id].astype(np.float64).tolist() for row in rows]


def visualization_traces(
    split: ComparableTransferSplit,
    embeddings: Mapping[str, np.ndarray],
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    eval_rows = tuple(split.gallery) + tuple(split.query)
    eval_matrix = [embeddings[row.sample_id].tolist() for row in eval_rows]
    identities = [row.identity_id for row in eval_rows]
    datasets = [row.dataset_name for row in eval_rows]
    views = [row.sequence_id for row in eval_rows]
    vector_payload = {
        "embeddings": eval_matrix,
        "identity": identities,
        "dataset": datasets,
        "view": views,
        "channels": {"appearance": eval_matrix},
    }
    metrics = report["metrics"]
    return {
        "parsing": {
            "stage": "parsing",
            "substages": {
                "00_detection": {
                    "summary": (
                        f"parser policy v6; "
                        f"{len(split.train_samples)} train + "
                        f"{len(eval_rows)} eval source images"
                    )
                },
                "01_segmentation": {
                    "summary": "dog-only policy v6 visible-instance masks"
                },
                "02_regions": {"summary": "body crop from single usable dog"},
                "03_quality": {"summary": "USABLE dog instance required"},
                "04_crops": {
                    "summary": (
                        f"crop binding {split.crop_binding_status}; "
                        f"gallery {len(split.gallery)}; query {len(split.query)}"
                    )
                },
            },
        },
        "identification": {
            "stage": "identification",
            "substages": {
                "00_appearance": dict(vector_payload),
                "01_face": {},
                "02_nose": {},
            },
        },
        "representation": {
            "stage": "representation",
            "substages": {
                "00_evidence": {
                    "summary": "appearance required; face/nose not in this protocol"
                },
                "01_channels": dict(vector_payload),
                "02_quality": {
                    "summary": "appearance required; face/nose not in this protocol"
                },
            },
        },
        "enrollment": {
            "stage": "enrollment",
            "substages": {
                "00_registry": {
                    "summary": (
                        f"{len({row.identity_id for row in split.gallery})} "
                        "Sibetan identities enrolled"
                    )
                },
                "01_write": {
                    "summary": f"{len(split.gallery)} gallery templates written"
                },
            },
        },
        "gallery": {
            "stage": "gallery",
            "substages": {
                "00_store": {
                    "summary": (
                        f"frozen gallery {split.gallery_list_sha256[:12]}; "
                        f"{len(split.gallery)} keys"
                    )
                }
            },
        },
        "search": {
            "stage": "search",
            "substages": {
                "00_scoring": {
                    "summary": (
                        f"Rank-1={metrics['Rank-1']:.4f} "
                        f"Rank-5={metrics['Rank-5']:.4f} "
                        f"mAP={metrics['mAP']:.4f}"
                    )
                },
                "01_matching": {
                    "summary": "identity-level max template; self-match excluded"
                },
            },
        },
    }


def write_traces(
    output_dir: Path,
    traces: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    bundle: list[tuple[Path, dict[str, Any]]] = []
    for stage in _VIS_STAGES:
        path = trace_dir / f"{stage}.json"
        bundle.append((path, dict(traces[stage])))
        written[stage] = path
    write_private_json_bundle(tuple(bundle))
    return written


def render_traces(trace_paths: Mapping[str, Path], vis_root: Path) -> None:
    vis_root.mkdir(parents=True, exist_ok=True)
    for stage, path in trace_paths.items():
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "visualization.commands.render",
                "--stage",
                stage,
                "--trace",
                str(path),
                "--output",
                str(vis_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if "No module named 'matplotlib'" in detail:
                raise RuntimeError(
                    "visualization extra is required: "
                    "uv run --extra visualization python -m "
                    "evaluation.commands.evaluate comparable-transfer ..."
                )
            raise RuntimeError(f"visualization render {stage} failed: {detail}")


def _load_embedding_file(path: Path) -> dict[str, np.ndarray]:
    payload = read_strict_json_object(path)
    sample_ids = payload.get("sample_ids")
    matrix = payload.get("embeddings")
    if not isinstance(sample_ids, list) or not isinstance(matrix, list):
        raise ValueError("embedding file must contain sample_ids and embeddings lists")
    if len(sample_ids) != len(matrix):
        raise ValueError("embedding rows do not match sample_ids")
    result: dict[str, np.ndarray] = {}
    for sample_id, vector in zip(sample_ids, matrix, strict=True):
        if not isinstance(sample_id, str) or sample_id in result:
            raise ValueError("embedding sample_ids must be unique non-empty strings")
        result[sample_id] = np.asarray(vector, dtype=np.float64)
    return result


def _select_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return "cuda"


def _file_binding(path: Path) -> Any:
    from shared.contracts.model_file_binding import ModelFileBinding

    payload = path.read_bytes()
    return ModelFileBinding(
        relative_path=path.name,
        byte_size=len(payload),
        sha256=_sha256_bytes(payload),
    )


def _load_parser_runtime(args: argparse.Namespace) -> Any:
    from shared.contracts.foreground_segmentation_model import (
        ForegroundSegmentationArtifact,
    )
    from shared.contracts.instance_segmentation_model import (
        InstanceSegmentationArtifact,
    )
    from parsing.export.segmentation.animal_instance_segmentation import (
        AnimalInstanceSegmentationRuntime,
    )
    from parsing.export.segmentation.animal_parsing import (
        AnimalParsingPolicy,
        AnimalParsingRuntime,
    )
    from parsing.export.segmentation.foreground_segmentation import (
        ForegroundSegmentationRuntime,
    )

    foreground = ForegroundSegmentationArtifact.load(
        model_directory=args.foreground_model_dir,
        manifest_bundle_path=args.foreground_model_manifest,
    )
    instance = InstanceSegmentationArtifact.load(
        model_directory=args.instance_model_dir,
        manifest_bundle_path=args.instance_model_manifest,
    )
    policy = AnimalParsingPolicy()
    if policy.schema_version != PARSER_POLICY_SCHEMA:
        raise ValueError("comparable-transfer requires parsing.policy.v6")
    device = _select_device(args.device)
    return AnimalParsingRuntime(
        instance_runtime=AnimalInstanceSegmentationRuntime(
            artifact=instance,
            device=device,
            mask_threshold=policy.foreground_threshold,
        ),
        foreground_runtime=ForegroundSegmentationRuntime(
            artifact=foreground,
            device=device,
            threshold=policy.foreground_threshold,
        ),
        policy=policy,
    )


def _usable_dog_instance(prediction: Any) -> Any:
    dogs = tuple(
        instance for instance in prediction.instances if instance.class_name == "dog"
    )
    if len(dogs) != 1:
        return None
    if dogs[0].quality.state not in {"USABLE", "REVIEW"}:
        return None
    return dogs[0]


def materialize_parser_v6_crops(
    rows: Sequence[ComparableTransferRow],
    *,
    dataset_root: Path,
    runtime: Any,
    crop_dir: Path,
    parser_batch_size: int = 4,
) -> dict[str, str]:
    from parsing.export.segmentation.animal_parsing import materialize_identity_crop

    if parser_batch_size <= 0:
        raise ValueError("parser_batch_size must be positive")
    crop_dir.mkdir(parents=True, exist_ok=True)
    bound: dict[str, str] = {}
    root = dataset_root.resolve(strict=True)
    row_list = tuple(rows)
    for offset in range(0, len(row_list), parser_batch_size):
        chunk = row_list[offset : offset + parser_batch_size]
        images: list[Image.Image] = []
        for row in chunk:
            source_path = root.joinpath(*Path(row.image_path).parts)
            payload = source_path.read_bytes()
            if _sha256_bytes(payload) != row.image_sha256:
                raise ValueError(f"source bytes differ for {row.sample_id}")
            with Image.open(source_path) as opened:
                image = opened.convert("RGB")
                image.load()
            images.append(image)
        predictions = runtime.predict_batch(
            tuple(images),
            instance_batch_size=parser_batch_size,
            foreground_batch_size=parser_batch_size,
        )
        for row, image, prediction in zip(chunk, images, predictions, strict=True):
            instance = _usable_dog_instance(prediction)
            target = crop_dir / f"{row.sample_id}.png"
            if instance is None:
                image.save(target, format="PNG")
            else:
                crop = materialize_identity_crop(
                    image, instance, require_usable=False
                )
                crop.masked_rgb.save(target, format="PNG")
            bound[row.sample_id] = _sha256_bytes(target.read_bytes())
    return bound


def embed_crops(
    rows: Sequence[ComparableTransferRow],
    *,
    crop_dir: Path,
    evidencer: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    images: list[Image.Image] = []
    sample_ids: list[str] = []
    for row in rows:
        path = crop_dir / f"{row.sample_id}.png"
        if _sha256_bytes(path.read_bytes()) != row.crop_sha256:
            raise ValueError(f"bound crop bytes differ for {row.sample_id}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.load()
        images.append(image)
        sample_ids.append(row.sample_id)
    vectors: dict[str, np.ndarray] = {}
    for offset in range(0, len(images), batch_size):
        batch = images[offset : offset + batch_size]
        encoded = evidencer.extract_batch(batch)
        for sample_id, vector in zip(
            sample_ids[offset : offset + len(batch)], encoded, strict=True
        ):
            vectors[sample_id] = np.asarray(vector, dtype=np.float64)
    return vectors


def admit_local_models(args: argparse.Namespace) -> dict[str, str]:
    from shared.contracts.foreground_segmentation_model import (
        ForegroundSegmentationModelManifest,
        foreground_segmentation_model_bundle,
    )
    from shared.contracts.instance_segmentation_model import (
        InstanceSegmentationModelManifest,
        instance_segmentation_model_bundle,
    )
    from shared.contracts.intake.pretrained_supporting_asset_intake import (
        PretrainedSupportingAssetKind,
        PretrainedSupportingAssetSourceContract,
        audit_pretrained_supporting_asset,
    )
    from shared.contracts.intake.pretrained_weight_intake import (
        PretrainedWeightChecksumAuthority,
        PretrainedWeightFileFormat,
        PretrainedWeightSourceContract,
        PretrainedWeightUsageLane,
        audit_pretrained_weight_file,
    )
    from shared.contracts.source_provenance import build_offline_tool_provenance

    args.output_dir.mkdir(parents=True, exist_ok=True)
    foreground_files = tuple(
        sorted(
            (
                _file_binding(args.foreground_model_dir / name)
                for name in (
                    "BiRefNet_config.py",
                    "birefnet.py",
                    "config.json",
                    "model.safetensors",
                )
            ),
            key=lambda item: item.relative_path,
        )
    )
    instance_files = tuple(
        sorted(
            (
                _file_binding(args.instance_model_dir / name)
                for name in (
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                )
            ),
            key=lambda item: item.relative_path,
        )
    )
    foreground = ForegroundSegmentationModelManifest(
        model_id="ZhengPeng7/BiRefNet_dynamic",
        source_revision="280306042f57b7a33854319da62fd86aaa89ec4c",
        model_family="BIREFNET_DYNAMIC_SWIN_V1_LARGE",
        task="HIGH_RESOLUTION_DICHOTOMOUS_IMAGE_SEGMENTATION",
        license_id="MIT",
        license_url="https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        input_multiple=32,
        minimum_inference_side=256,
        maximum_inference_side=2304,
        files=foreground_files,
    )
    instance = InstanceSegmentationModelManifest(
        model_id="Roboflow/rf-detr-segmentation",
        source_revision="4c7844eeab07e90df88ca8f6425a8a7759acdc74",
        model_family="RF_DETR_SEGMENTATION_COCO",
        training_label_space="COCO_2017_INSTANCE_91_CATEGORY_IDS",
        license_id="Apache-2.0",
        license_url="https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        files=instance_files,
    )
    foreground_path = args.output_dir / "foreground.bundle.json"
    instance_path = args.output_dir / "instance.bundle.json"
    write_private_json_bundle(
        (
            (foreground_path, foreground_segmentation_model_bundle(foreground)),
            (instance_path, instance_segmentation_model_bundle(instance)),
        )
    )
    weight = args.dinov2_model_dir / "model.safetensors"
    preprocessor = args.dinov2_model_dir / "preprocessor_config.json"
    license_sha = _sha256_bytes(args.license_snapshot.read_bytes())
    training_sha = _sha256_bytes(args.training_snapshot.read_bytes())
    weight_sha = _sha256_bytes(weight.read_bytes())
    source = PretrainedWeightSourceContract(
        source_model_id="facebook/dinov2-small",
        source_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        source_model_page_url="https://huggingface.co/facebook/dinov2-small",
        source_file_url=(
            "https://huggingface.co/facebook/dinov2-small/resolve/"
            "ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors"
        ),
        weight_filename="model.safetensors",
        license_id="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
        license_snapshot_sha256=license_sha,
        license_usage_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
        training_description=(
            "DINOv2 ViT-S/14 feature encoder. Upstream model card declares "
            "LVD-142M training; absence of overlap with public canine images "
            "is not asserted."
        ),
        training_description_url=(
            "https://raw.githubusercontent.com/facebookresearch/dinov2/"
            "7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md"
        ),
        training_description_snapshot_sha256=training_sha,
        expected_file_bytes=weight.stat().st_size,
        expected_sha256=weight_sha,
        checksum_authority=PretrainedWeightChecksumAuthority.PUBLISHED_SHA256,
        target_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
        file_format=PretrainedWeightFileFormat.SAFETENSORS,
    )
    weight_receipt = audit_pretrained_weight_file(
        weight_path=weight,
        license_snapshot_path=args.license_snapshot,
        training_description_snapshot_path=args.training_snapshot,
        source=source,
    )
    preprocessor_sha = _sha256_bytes(preprocessor.read_bytes())
    preprocessor_source = PretrainedSupportingAssetSourceContract(
        source_model_id="facebook/dinov2-small",
        source_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        source_model_page_url="https://huggingface.co/facebook/dinov2-small",
        source_file_url=(
            "https://huggingface.co/facebook/dinov2-small/resolve/"
            "ed25f3a31f01632728cabb09d1542f84ab7b0056/preprocessor_config.json"
        ),
        asset_filename="preprocessor_config.json",
        asset_kind=PretrainedSupportingAssetKind.PREPROCESSOR_CONFIG,
        expected_file_bytes=preprocessor.stat().st_size,
        expected_sha256=preprocessor_sha,
        license_id="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
        license_snapshot_sha256=license_sha,
        license_usage_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
        associated_pretrained_weight_receipt_sha256=weight_receipt.receipt_sha256,
        target_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
    )
    preprocessor_receipt = audit_pretrained_supporting_asset(
        asset_path=preprocessor,
        license_snapshot_path=args.license_snapshot,
        source=preprocessor_source,
        associated_weight_source=source,
        associated_weight_receipt=weight_receipt,
    )
    provenance = build_offline_tool_provenance(Path(__file__))
    weight_bundle = {
        "schema_version": "shared.pretrained_weight_intake_bundle.v1",
        "source_contract_sha256": source.contract_sha256,
        "source_contract": source.to_dict(),
        "receipt_sha256": weight_receipt.receipt_sha256,
        "receipt": weight_receipt.to_dict(),
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
    }
    preprocessor_bundle = {
        "schema_version": "shared.pretrained_supporting_asset_intake_bundle.v1",
        "source_contract_sha256": preprocessor_source.contract_sha256,
        "source_contract": preprocessor_source.to_dict(),
        "receipt_sha256": preprocessor_receipt.receipt_sha256,
        "receipt": preprocessor_receipt.to_dict(),
        "tool_provenance": provenance,
        "tool_provenance_sha256": content_sha256(provenance),
    }
    weight_path = args.output_dir / "dinov2-weight-intake.json"
    preprocessor_path = args.output_dir / "dinov2-preprocessor-intake.json"
    write_private_json_bundle(
        (
            (weight_path, weight_bundle),
            (preprocessor_path, preprocessor_bundle),
        )
    )
    return {
        "foreground_bundle": str(foreground_path),
        "instance_bundle": str(instance_path),
        "dinov2_weight_intake": str(weight_path),
        "dinov2_preprocessor_intake": str(preprocessor_path),
    }


def _cmd_admit(args: argparse.Namespace) -> int:
    paths = admit_local_models(args)
    print(
        json.dumps(
            {"event": "comparable_transfer_models_admitted", **paths},
            sort_keys=True,
        )
    )
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    train = load_dataset("yt-bb-dog", args.yt_bb_dog)
    eval_samples = load_dataset("sibetan", args.sibetan)
    split = freeze_comparable_transfer(train, eval_samples, split_seed=args.seed)
    payload = write_split(args.output, split)
    print(
        json.dumps(
            {
                "event": "comparable_transfer_frozen",
                "schema_version": PROTOCOL_SCHEMA,
                "output": str(args.output),
                "comparable": split.comparable,
                "split_sha256": payload["split_sha256"],
                "train_identities": len(split.train_identities),
                "gallery": len(split.gallery),
                "query": len(split.query),
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    split = load_split(args.split)
    gallery = _load_embedding_file(args.gallery_embeddings)
    query = _load_embedding_file(args.query_embeddings)
    overlap = set(gallery) & set(query)
    if overlap:
        raise ValueError("gallery and query embedding files share sample_ids")
    embeddings = {**gallery, **query}
    report = score_comparable_transfer(
        split, embeddings, backbone_id=args.backbone_id
    )
    write_private_json_bundle(((args.output, report),))
    print(json.dumps({"event": "comparable_transfer_scored", **report["metrics"]}, sort_keys=True))
    return 0


def run_smoke(output_dir: Path, *, vis_root: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, eval_samples = smoke_samples()
    split = freeze_comparable_transfer(train, eval_samples)
    crops = _write_smoke_crops(split, output_dir / "crops")
    split = bind_crops(split, crops, include_train=True)
    write_split(output_dir / "split.json", split)
    embeddings = smoke_embeddings(split)
    report = score_comparable_transfer(
        split, embeddings, backbone_id="smoke.hash-identity-unit"
    )
    write_private_json_bundle(((output_dir / "report.json", report),))
    traces = visualization_traces(split, embeddings, report)
    trace_paths = write_traces(output_dir, traces)
    if vis_root is not None:
        render_traces(trace_paths, vis_root)
    return {
        "event": "comparable_transfer_smoke_done",
        "output_dir": str(output_dir),
        "split_sha256": split.split_sha256,
        "comparable": split.comparable,
        "metrics": report["metrics"],
        "vis": vis_root is not None,
    }


def _cmd_smoke(args: argparse.Namespace) -> int:
    result = run_smoke(args.output_dir, vis_root=args.vis_root)
    print(json.dumps(result, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.split is not None:
        split = load_split(args.split)
    else:
        split = freeze_comparable_transfer(
            load_dataset("yt-bb-dog", args.yt_bb_dog),
            load_dataset("sibetan", args.sibetan),
            split_seed=args.seed,
        )
        write_split(args.output_dir / "split.unbound.json", split)
    if args.yt_bb_dog is None or args.sibetan is None:
        raise ValueError("run requires --yt-bb-dog and --sibetan dataset roots")
    runtime = _load_parser_runtime(args)
    eval_rows = tuple(split.gallery) + tuple(split.query)
    train_bound = materialize_parser_v6_crops(
        split.train_samples,
        dataset_root=args.yt_bb_dog,
        runtime=runtime,
        crop_dir=args.output_dir / "crops" / "train",
        parser_batch_size=args.parser_batch_size,
    )
    eval_bound = materialize_parser_v6_crops(
        eval_rows,
        dataset_root=args.sibetan,
        runtime=runtime,
        crop_dir=args.output_dir / "crops" / "eval",
        parser_batch_size=args.parser_batch_size,
    )
    del runtime
    if args.device == "cuda":
        import torch

        torch.cuda.empty_cache()
    split = bind_crops(split, {**train_bound, **eval_bound}, include_train=True)
    write_split(args.output_dir / "split.json", split)
    from identification.export.appearance import ReceiptBoundDinov2Small

    device = _select_device(args.device)
    evidencer = ReceiptBoundDinov2Small(
        model_directory=args.dinov2_model_dir,
        weight_intake_bundle=args.dinov2_weight_intake,
        preprocessor_intake_bundle=args.dinov2_preprocessor_intake,
        device=device,
        max_batch_size=args.batch_size,
    )
    embeddings = embed_crops(
        (*split.gallery, *split.query),
        crop_dir=args.output_dir / "crops" / "eval",
        evidencer=evidencer,
        batch_size=args.batch_size,
    )
    report = score_comparable_transfer(
        split,
        embeddings,
        backbone_id=f"dinov2-small:{evidencer.weight_receipt_sha256}",
    )
    write_private_json_bundle(((args.output_dir / "report.json", report),))
    traces = visualization_traces(split, embeddings, report)
    trace_paths = write_traces(args.output_dir, traces)
    if args.vis_root is not None:
        render_traces(trace_paths, args.vis_root)
    print(
        json.dumps(
            {
                "event": "comparable_transfer_run_done",
                "output_dir": str(args.output_dir),
                "split_sha256": split.split_sha256,
                "backbone_id": report["backbone_id"],
                "metrics": report["metrics"],
                "device": device,
            },
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.commands.evaluate comparable-transfer",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    admit = sub.add_parser(
        "admit-models",
        help="Write capability-owned parser and DINOv2 intake bundles from local checkpoints",
    )
    admit.add_argument("--foreground-model-dir", type=Path, required=True)
    admit.add_argument("--instance-model-dir", type=Path, required=True)
    admit.add_argument("--dinov2-model-dir", type=Path, required=True)
    admit.add_argument("--license-snapshot", type=Path, required=True)
    admit.add_argument("--training-snapshot", type=Path, required=True)
    admit.add_argument("--output-dir", type=Path, required=True)
    admit.set_defaults(func=_cmd_admit)

    freeze = sub.add_parser("freeze", help="Freeze train IDs and gallery/query lists")
    freeze.add_argument("--yt-bb-dog", type=Path, required=True)
    freeze.add_argument("--sibetan", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=SPLIT_SEED)
    freeze.set_defaults(func=_cmd_freeze)

    score = sub.add_parser("score", help="Score frozen gallery/query embeddings")
    score.add_argument("--split", type=Path, required=True)
    score.add_argument("--gallery-embeddings", type=Path, required=True)
    score.add_argument("--query-embeddings", type=Path, required=True)
    score.add_argument("--backbone-id", required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(func=_cmd_score)

    smoke = sub.add_parser("smoke", help="Synthetic freeze, bind, score, traces")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--vis-root", type=Path)
    smoke.set_defaults(func=_cmd_smoke)

    run = sub.add_parser(
        "run",
        help="Parser v6 crops, DINOv2 embed, frozen gallery/query, Rank-1/5/mAP, vis",
    )
    run.add_argument("--yt-bb-dog", type=Path, required=True)
    run.add_argument("--sibetan", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--split", type=Path)
    run.add_argument("--seed", type=int, default=SPLIT_SEED)
    run.add_argument("--foreground-model-dir", type=Path, required=True)
    run.add_argument("--foreground-model-manifest", type=Path, required=True)
    run.add_argument("--instance-model-dir", type=Path, required=True)
    run.add_argument("--instance-model-manifest", type=Path, required=True)
    run.add_argument("--dinov2-model-dir", type=Path, required=True)
    run.add_argument("--dinov2-weight-intake", type=Path, required=True)
    run.add_argument("--dinov2-preprocessor-intake", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--parser-batch-size", type=int, default=4)
    run.add_argument("--vis-root", type=Path)
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
