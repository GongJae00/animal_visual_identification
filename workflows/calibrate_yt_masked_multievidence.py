"""Fit and evaluate availability-aware A/F/N fusion on fixed YT source windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from contracts.artifact_manifest import (
    NoseEmbeddingManifest,
    UsageLane,
    preprocess_image,
)
from experiments.sibetan_multievidence import (
    BRANCHES,
    evaluate_effective_k_panel,
    face_reliability,
    fit_effective_k_weights,
    nose_reliability,
)
from experiments.unified_multievidence import (
    _extract_dino_embeddings,
    _load_bound_lineage,
    _native_source_key,
    _read_bound_rgb,
    _roi_source_key,
)
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256
from identity_methods.appearance import ReceiptBoundDinov2Small
from localization.nose_region.embedding_consistency_training import (
    load_consistency_checkpoint,
)
from localization.nose_region.embedding_training import load_receipt_bound_dinov2
from localization.nose_region.native_yt import validate_manifest_bundle
from localization.roi_manifest import read_roi_manifest

SCHEMA = "cvi.yt_masked_multievidence_policy.v2"
BUNDLE_SCHEMA = "cvi.yt_masked_multievidence_policy_bundle.v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value):
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 1e-8:
        raise ValueError("YT masked embedding differs")
    return vector / norm


def _prototype(values, weights=None):
    return _normalize(np.average(np.stack([_normalize(value) for value in values]), axis=0, weights=weights))


def _fixed_population(records, identities):
    by_identity = defaultdict(list)
    for row in records:
        if row["registered_dog_id"] in set(identities):
            by_identity[row["registered_dog_id"]].append(row)
    population = []
    for identity in sorted(identities):
        rows = sorted(by_identity[identity], key=lambda row: (row["frame_index"], row["sample_token"]))
        if len(rows) < 10:
            raise ValueError("YT fixed identity lacks ten source records")
        gallery, query = rows[:5], rows[-5:]
        if {row["sample_token"] for row in gallery} & {row["sample_token"] for row in query}:
            raise ValueError("YT fixed source windows overlap")
        population.append({"identity": identity, "gallery": gallery, "query": query})
    return population


def _face_map(roi_records):
    result = defaultdict(list)
    for row in roi_records:
        if row["registered_identity_id"] is not None and row["face_crop_path"] is not None:
            result[(row["registered_identity_id"], _roi_source_key(row["image_path"]))].append(row)
    return result


def _extract_population(
    population, *, source_root, native_root, roi_root, face_map, dino,
    nose_model, nose_device, nose_manifest, batch_size,
):
    all_rows = [row for item in population for role in ("gallery", "query") for row in item[role]]
    appearance_images = [
        _read_bound_rgb(source_root, _native_source_key(row), row["source_sha256"])
        for row in all_rows
    ]
    appearance_vectors = _extract_dino_embeddings(appearance_images, dino, batch_size=batch_size)
    appearance_by_token = dict(zip((row["sample_token"] for row in all_rows), appearance_vectors, strict=True))
    appearance_quality_by_token = {token: 1.0 for token in appearance_by_token}
    face_rows = {}
    for row in all_rows:
        candidates = face_map.get((row["registered_dog_id"], _native_source_key(row)), [])
        if len(candidates) > 1:
            raise ValueError("YT fixed source repeats an identity-bound Face")
        if candidates:
            face = candidates[0]
            if face["image_sha256"] != row["source_sha256"]:
                raise ValueError("YT fixed Face source hash differs")
            face_rows[row["sample_token"]] = face
    face_tokens = sorted(face_rows)
    face_vectors = _extract_dino_embeddings(
        [_read_bound_rgb(roi_root, face_rows[token]["face_crop_path"], face_rows[token]["face_crop_sha256"]) for token in face_tokens],
        dino, batch_size=batch_size,
    )
    face_by_token = dict(zip(face_tokens, face_vectors, strict=True))
    face_quality_by_token = {}
    for token in face_tokens:
        face = face_rows[token]
        rect = face["face_crop_rect_xyxy"]
        face_quality_by_token[token] = face_reliability(
            upstream_overall=float(face["face_quality"]["overall"]),
            native_short_side=min(rect[2] - rect[0], rect[3] - rect[1]),
        )
    nose_rows = [row for row in all_rows if row["crop_path"] is not None]
    nose_by_token = {}
    nose_quality_by_token = {}
    for offset in range(0, len(nose_rows), batch_size):
        batch_rows = nose_rows[offset : offset + batch_size]
        arrays = [
            preprocess_image(
                _read_bound_rgb(native_root, row["crop_path"], row["crop_sha256"]),
                nose_manifest,
            )[0]
            for row in batch_rows
        ]
        tensor = torch.from_numpy(np.stack(arrays)).to(nose_device)
        with torch.inference_mode():
            values = nose_model(tensor).cpu().numpy().astype(np.float32, copy=False)
        for row, value in zip(batch_rows, values, strict=True):
            nose_by_token[row["sample_token"]] = _normalize(value)
            quality = row["quality"]
            nose_quality_by_token[row["sample_token"]] = nose_reliability(
                detector_confidence=float(quality["detector_confidence"]),
                frontality=float(quality["frontality"]),
                native_short_side=int(quality["native_short_side"]),
                blur_score=float(quality["blur_score"]),
                contrast_score=float(quality["contrast_score"]),
            )

    gallery, queries = [], []
    embeddings = {branch: {} for branch in BRANCHES}
    quality_maps = {branch: {} for branch in BRANCHES}
    pairings = []
    for item in population:
        identity = item["identity"]
        for row in item["gallery"]:
            token = row["sample_token"]
            gallery.append({"sample_token": token, "identity_token": identity})
            embeddings[BRANCHES[0]][token] = appearance_by_token[token]
            quality_maps[BRANCHES[0]][token] = 1.0
            if token in face_by_token:
                embeddings[BRANCHES[1]][token] = face_by_token[token]
                quality_maps[BRANCHES[1]][token] = face_quality_by_token[token]
            if token in nose_by_token:
                embeddings[BRANCHES[2]][token] = nose_by_token[token]
                quality_maps[BRANCHES[2]][token] = nose_quality_by_token[token]
        query_token = content_sha256({"registered_dog_id": identity, "role": "YT_MASKED_QUERY_PROTOTYPE_V1"})
        queries.append({"sample_token": query_token, "identity_token": identity})
        for branch, source, source_quality in (
            (BRANCHES[0], appearance_by_token, appearance_quality_by_token),
            (BRANCHES[1], face_by_token, face_quality_by_token),
            (BRANCHES[2], nose_by_token, nose_quality_by_token),
        ):
            tokens = [row["sample_token"] for row in item["query"] if row["sample_token"] in source]
            values = [source[token] for token in tokens]
            if values:
                weights = [source_quality[token] for token in tokens]
                embeddings[branch][query_token] = _prototype(values, weights)
                quality_maps[branch][query_token] = float(np.mean(weights))
        pairings.append({
            "registered_dog_id": identity,
            "gallery_sample_tokens": [row["sample_token"] for row in item["gallery"]],
            "query_sample_tokens": [row["sample_token"] for row in item["query"]],
            "query_prototype_token": query_token,
        })
    return gallery, queries, embeddings, quality_maps, pairings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--roi-manifest-sha256", required=True)
    parser.add_argument("--nose-lineage", type=Path, required=True)
    parser.add_argument("--nose-lineage-sha256", required=True)
    parser.add_argument("--nose-manifest", type=Path, required=True)
    parser.add_argument("--nose-manifest-sha256", required=True)
    parser.add_argument("--nose-onnx", type=Path, required=True)
    parser.add_argument("--nose-checkpoint", type=Path, required=True)
    parser.add_argument("--nose-checkpoint-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--frozen-model-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nose-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fusion-resolution", type=int, default=20)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    native_document = read_strict_json_document(args.native_bundle, maximum_bytes=536_870_912, maximum_nodes=10_000_000, maximum_keys=5_000_000, maximum_array_length=1_000_000)
    if native_document.canonical_payload_sha256 != args.native_bundle_sha256:
        raise ValueError("YT native bundle pin differs")
    native_root = args.native_root.resolve(strict=True)
    native = validate_manifest_bundle(native_document.payload, root=native_root)
    print(json.dumps({"stage": "native_validated"}), flush=True)
    roi_document = read_strict_json_document(args.roi_manifest, maximum_bytes=536_870_912, maximum_nodes=10_000_000, maximum_keys=5_000_000, maximum_array_length=1_000_000)
    if roi_document.canonical_payload_sha256 != args.roi_manifest_sha256:
        raise ValueError("YT ROI manifest pin differs")
    roi = read_roi_manifest(args.roi_manifest)
    print(json.dumps({"stage": "roi_validated"}), flush=True)
    lineage_document, bindings = _load_bound_lineage(args.nose_lineage, args.nose_lineage_sha256, args.nose_manifest, args.nose_onnx)
    nose_document = read_strict_json_document(args.nose_manifest)
    if nose_document.canonical_payload_sha256 != args.nose_manifest_sha256:
        raise ValueError("YT Nose manifest pin differs")
    nose_manifest = NoseEmbeddingManifest.from_dict(nose_document.payload)
    if nose_manifest.license.usage_lane != UsageLane.RESEARCH_ONLY or _sha(args.nose_onnx) != nose_manifest.artifact_sha256:
        raise ValueError("YT Nose runtime differs")
    dino = ReceiptBoundDinov2Small(
        model_directory=args.model_dir, weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        device=args.device, max_batch_size=args.batch_size,
    )
    if dino.model_sha256 != args.frozen_model_sha256:
        raise ValueError("YT frozen DINO differs")
    selected_artifact = lineage_document.payload["artifacts"]["selected_checkpoint"]
    if (
        args.nose_checkpoint.resolve(strict=True)
        != (args.nose_lineage.parent / selected_artifact["path"]).resolve(strict=True)
        or _sha(args.nose_checkpoint) != args.nose_checkpoint_sha256
        or args.nose_checkpoint_sha256 != selected_artifact["sha256"]
    ):
        raise ValueError("YT selected Nose checkpoint differs from lineage")
    nose_model, _ = load_receipt_bound_dinov2(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
    )
    checkpoint = load_consistency_checkpoint(args.nose_checkpoint)
    nose_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    nose_device = torch.device(args.nose_device)
    nose_model.to(nose_device).eval()
    print(json.dumps({"stage": "models_loaded"}), flush=True)
    splits = bindings["splits"]["identity_lists"]
    face_by_source = _face_map(roi["records"])
    extracted = {}
    roles = ("dev",) if args.dev_only else ("dev", "eval")
    for role in roles:
        population = _fixed_population(native["records"], splits[role])
        extracted[role] = _extract_population(
            population, source_root=args.source_image_root.resolve(strict=True),
            native_root=native_root, roi_root=args.roi_manifest.parent.resolve(strict=True),
            face_map=face_by_source, dino=dino, nose_model=nose_model,
            nose_device=nose_device,
            nose_manifest=nose_manifest, batch_size=args.batch_size,
        )
        print(json.dumps({"stage": f"{role}_embeddings_extracted"}), flush=True)
    dev_gallery, dev_queries, dev_embeddings, dev_quality, dev_pairings = extracted["dev"]
    calibration = fit_effective_k_weights(
        gallery=dev_gallery, queries=dev_queries, embeddings=dev_embeddings,
        quality=dev_quality,
        resolution=args.fusion_resolution,
    )
    print(json.dumps({"stage": "dev_weights_fitted"}), flush=True)
    weights = {method: value["selected_weights"] for method, value in calibration["fusions"].items()}
    dev_result = evaluate_effective_k_panel(
        gallery=dev_gallery, queries=dev_queries, embeddings=dev_embeddings,
        quality=dev_quality,
        transfer_weights=weights,
    )
    if args.dev_only:
        eval_queries, eval_pairings, eval_result = [], [], None
    else:
        eval_gallery, eval_queries, eval_embeddings, eval_quality, eval_pairings = extracted["eval"]
        eval_result = evaluate_effective_k_panel(
            gallery=eval_gallery, queries=eval_queries, embeddings=eval_embeddings,
            transfer_weights=weights, quality=eval_quality,
        )
    report = {
        "schema_version": SCHEMA,
        "status": "PASS_YT_DEV_MASKED_FUSION_POLICY",
        "interpretation": "YT_TRACK_PROXY_MASKED_FUSION_RESEARCH_POLICY_NOT_LIFELONG_IDENTITY_VALIDATION",
        "policy_semantics": {
            "source_windows": "FIXED_EARLIEST5_LATEST5_BEFORE_BRANCH_AVAILABILITY",
            "aggregation": "L2_NORMALIZED_MEAN_AVAILABLE_FIXED_OBSERVATIONS",
            "normalization": "QUERY_ROW_ZSCORE_AVAILABLE_CANDIDATES_ONLY",
            "fusion": "CANDIDATE_WISE_POSITIVE_WEIGHT_RENORMALIZATION",
            "missing_score_sentinel": None,
            "reliability": "CONTINUOUS_FACE_AND_NOSE_QUALITY_WEIGHTED_PROTOTYPE_AND_FUSION",
        },
        "calibration": calibration,
        "dev": dev_result,
        "evaluation": eval_result,
        "population": {"dev_identity_count": len(dev_queries), "eval_identity_count": len(eval_queries)},
        "input_bindings": {
            "native_bundle_content_sha256": native_document.canonical_payload_sha256,
            "roi_manifest_content_sha256": roi_document.canonical_payload_sha256,
            "nose_lineage_content_sha256": lineage_document.canonical_payload_sha256,
            "nose_runtime_manifest_content_sha256": nose_document.canonical_payload_sha256,
            "nose_onnx_sha256": nose_manifest.artifact_sha256,
            "frozen_dinov2_sha256": dino.model_sha256,
            "dev_pairing_sha256": content_sha256(dev_pairings),
            "eval_pairing_sha256": content_sha256(eval_pairings) if eval_pairings else None,
            "code_sha256s": {
                relative: _sha(Path(__file__).resolve().parents[1] / relative)
                for relative in (
                    "experiments/sibetan_multievidence.py",
                    "experiments/unified_multievidence.py",
                    "evaluation/retrieval.py",
                    "workflows/calibrate_yt_masked_multievidence.py",
                )
            },
        },
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {"schema_version": BUNDLE_SCHEMA, "report_sha256": content_sha256(report), "report": report}
    write_private_json_bundle(((args.output, bundle),))
    print(json.dumps({
        "status": report["status"], "output": os.fspath(args.output),
        "report_sha256": bundle["report_sha256"], "weights": weights,
        "dev_identity_count": len(dev_queries), "eval_identity_count": len(eval_queries),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
