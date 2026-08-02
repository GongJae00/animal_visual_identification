"""Evaluate DINO appearance, face, or weak-nose crops from an ROI manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_pipeline.adapters import ADAPTERS
from data_pipeline.source_lock import get_record
from evaluation.retrieval import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
)
from identity_methods.appearance import ReceiptBoundDinov2Small
from localization.roi_manifest import read_roi_manifest
from foundation.protected_io import write_private_json_bundle
from foundation.provenance import content_sha256
from representation_learning.trainer import ArcFaceModel, Dinov2Embedding, TrainConfig
from identity_governance.training_admission import TrainingAdmissionReceipt

_CHANNEL_PATH = {
    "source": "dog_crop_path",
    "dog": "dog_crop_path",
    "face": "face_crop_path",
    "weak_nose": "weak_nose_crop_path",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reconstruct_dinov2_model(payload: object, model_directory: Path) -> ArcFaceModel:
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "cvi.training_checkpoint.v1"
    ):
        raise RuntimeError("unsupported or legacy training checkpoint")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise RuntimeError("checkpoint is missing its training configuration")
    config = TrainConfig.from_dict(config_payload)
    if config.model_name != "dinov2-small" or config.embedding_dim != 384:
        raise RuntimeError("ROI ReID checkpoint must use DINOv2-small")
    model = ArcFaceModel(
        config,
        backbone_factory=partial(
            Dinov2Embedding, model_directory=model_directory
        ),
    )
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--channel", choices=sorted(_CHANNEL_PATH), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.checkpoint is None) != (args.checkpoint_sha256 is None):
        parser.error("--checkpoint and --checkpoint-sha256 must be provided together")
    if args.checkpoint is not None and args.channel == "source":
        parser.error("trained checkpoint evaluation requires a materialized ROI channel")
    return args


def main() -> None:
    args = parse_args()
    manifest = read_roi_manifest(args.roi_manifest)
    data_root = Path(get_record(args.dataset).data_root)
    source_samples = ADAPTERS[args.dataset](data_root)
    source_by_id = {sample.sample_id: sample for sample in source_samples}
    selected_by_sample: dict[str, dict] = {}
    path_field = _CHANNEL_PATH[args.channel]
    quality_field = "face_quality" if args.channel == "face" else "quality"
    for record in manifest["records"]:
        if not record[path_field]:
            continue
        previous = selected_by_sample.get(record["sample_id"])
        if (
            previous is None
            or record[quality_field]["overall"] > previous[quality_field]["overall"]
        ):
            selected_by_sample[record["sample_id"]] = record

    by_identity: dict[str, list[dict]] = {}
    for sample_id, record in selected_by_sample.items():
        identity = source_by_id[sample_id].registered_identity_id
        if identity is not None:
            by_identity.setdefault(identity, []).append(record)
    eligible = {
        identity: sorted(records, key=lambda record: record["sample_id"])
        for identity, records in by_identity.items()
        if len(records) >= 2
    }
    gallery = [records[0] for _, records in sorted(eligible.items())]
    queries = [
        record for _, records in sorted(eligible.items()) for record in records[1:]
    ]
    if not gallery or not queries:
        raise RuntimeError(
            "ROI manifest does not contain a closed-set evaluation cohort"
        )

    if args.checkpoint is None:
        evidencer = ReceiptBoundDinov2Small(
            model_directory=str(args.model_dir),
            weight_intake_bundle=str(args.weight_intake_bundle),
            preprocessor_intake_bundle=str(args.preprocessor_intake_bundle),
            device=args.device,
            max_batch_size=args.batch_size,
        )
        embedding_provenance = {
            "weight_intake_bundle_sha256": _sha256(args.weight_intake_bundle),
            "preprocessor_intake_bundle_sha256": _sha256(
                args.preprocessor_intake_bundle
            ),
        }
    else:
        checkpoint_sha256 = _sha256(args.checkpoint)
        if checkpoint_sha256 != args.checkpoint_sha256:
            raise ValueError("training checkpoint SHA-256 differs")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = _reconstruct_dinov2_model(payload, args.model_dir)
        intake = payload.get("training_admission")
        if not isinstance(intake, dict) or set(intake) != {"receipt_sha256", "receipt"}:
            raise ValueError("training checkpoint lacks its admission receipt")
        admission = TrainingAdmissionReceipt.from_dict(intake["receipt"])
        if intake["receipt_sha256"] != admission.receipt_sha256:
            raise ValueError("training checkpoint admission receipt hash differs")
        evidencer = ReceiptBoundDinov2Small(
            model_directory=str(args.model_dir),
            weight_intake_bundle=str(args.weight_intake_bundle),
            preprocessor_intake_bundle=str(args.preprocessor_intake_bundle),
            device=args.device,
            max_batch_size=args.batch_size,
        )
        if admission.model_receipt_sha256 != evidencer.weight_receipt_sha256:
            raise ValueError("training checkpoint base model receipt differs")
        model._backbone._backbone.to(torch.device(args.device)).eval()
        evidencer._backbone = model._backbone._backbone
        embedding_provenance = {
            "training_checkpoint_sha256": checkpoint_sha256,
            "selection_metric": payload["selection_metric"],
            "selected_epoch": payload["epoch"],
        }
    crop_root = args.roi_manifest.parent

    def extract(records: list[dict]) -> tuple[np.ndarray, float]:
        embeddings = np.empty((len(records), 384), dtype=np.float32)
        started = time.perf_counter()
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset : offset + args.batch_size]
            images = [
                Image.open(
                    data_root / source_by_id[record["sample_id"]].image_path
                    if args.channel == "source"
                    else crop_root / record[path_field]
                ).convert("RGB")
                for record in batch
            ]
            embeddings[offset : offset + len(batch)] = evidencer.extract_batch(images)
        return embeddings, time.perf_counter() - started

    gallery_embeddings, gallery_seconds = extract(gallery)
    query_embeddings, query_seconds = extract(queries)
    scores = compute_cosine_score_matrix(query_embeddings, gallery_embeddings)
    metrics = evaluate_multi_template_closed_set(
        scores,
        query_identity_ids=np.asarray(
            [
                source_by_id[record["sample_id"]].registered_identity_id
                for record in queries
            ]
        ),
        gallery_template_identity_ids=np.asarray(
            [
                source_by_id[record["sample_id"]].registered_identity_id
                for record in gallery
            ]
        ),
        self_match_policy="exclude",
        query_template_ids=np.asarray([record["sample_id"] for record in queries]),
        gallery_template_ids=np.asarray([record["sample_id"] for record in gallery]),
        rank_ks=(1, 5, 10),
    )
    query_rows = metrics["query_rows"]
    quality_subsets = {}
    for name, predicate in {
        "quality_ge_0.65": lambda quality: quality >= 0.65,
        "quality_lt_0.65": lambda quality: quality < 0.65,
    }.items():
        indices = [
            index
            for index, record in enumerate(queries)
            if predicate(float(record[quality_field]["overall"]))
        ]
        quality_subsets[name] = {
            "queries": len(indices),
            "Rank-1": float(np.mean([query_rows[index]["Rank-1"] for index in indices]))
            if indices
            else None,
            "MRR": float(
                np.mean([query_rows[index]["reciprocal_rank"] for index in indices])
            )
            if indices
            else None,
        }
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()
    report = {
        "schema_version": "cvi.roi_reid_evaluation.v1",
        "dataset": args.dataset,
        "channel": args.channel,
        "protocol": "one-gallery-per-video-track-remaining-query.v1",
        "interpretation": "within-video-track closed-set diagnostic",
        "source_samples": len(manifest["source_sample_ids"]),
        "channel_samples": len(selected_by_sample),
        "sample_coverage": len(selected_by_sample) / len(manifest["source_sample_ids"]),
        "gallery_identities": len(gallery),
        "queries": len(queries),
        "embedding_dimension": 384,
        "Rank-1": metrics["Rank-1"],
        "Rank-5": metrics["Rank-5"],
        "Rank-10": metrics["Rank-10"],
        "MRR": metrics["MRR"],
        "metric_note": "One relevant gallery identity per query; AP and INP equal reciprocal rank and are not reported.",
        "quality_subsets": quality_subsets,
        "throughput": {
            "gallery_images_per_second": len(gallery) / gallery_seconds,
            "query_images_per_second": len(queries) / query_seconds,
        },
        "provenance": {
            "code_commit": commit,
            "roi_manifest_sha256": content_sha256(manifest),
            **embedding_provenance,
            "dependency_lock_sha256": _sha256(
                Path(__file__).resolve().parents[1] / "uv.lock"
            ),
            "device": args.device,
            "batch_size": args.batch_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_private_json_bundle(((args.output, report),))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
