"""Evaluate a trained FaceID checkpoint after strict ROI partition checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.provenance import git_worktree_provenance as _git_provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("regional_v4", "cls_residual_v5", "aligned_cls_residual_v5"),
        default="regional_v4",
    )
    parser.add_argument(
        "--expected-split-role",
        default="test",
        help="Exact split_role required on every ROI record (default: test).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Execution device; CUDA is never selected as an implicit CPU fallback.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from embedding.methods.face.checkpoint import (
        expected_faceid_contract_for_checkpoint,
        file_sha256,
        normalize_dino_local_artifact_contract,
        validate_checkpoint_runtime_bindings,
        validate_checkpoint_structure,
        validate_evaluation_partition,
    )
    from embedding.methods.face.dataset import RoiFaceReIDDataset
    from experiments.face_evaluation import (
        evaluate_face_retrieval,
        extract_face_embeddings,
        paired_face_retrieval_comparison,
    )
    from embedding.methods.face.trainer import (
        build_faceid_model,
        load_receipt_bound_frozen_dino,
    )
    from parsing.roi_manifest import read_roi_manifest
    from foundation.protected_io import write_private_json_bundle

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu explicitly"
        )
    device = torch.device(args.device)
    manifest = read_roi_manifest(args.roi_manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    repository = Path(__file__).resolve().parents[1]
    faceid_contract = expected_faceid_contract_for_checkpoint(
        checkpoint["faceid_contract"],
        repository,
        architecture=args.architecture,
    )
    training_identity_ids = validate_checkpoint_structure(
        checkpoint, expected_faceid_contract=faceid_contract
    )
    independence = validate_evaluation_partition(
        manifest,
        training_roi_manifest_sha256=checkpoint["training_roi_manifest_sha256"],
        training_identity_ids=training_identity_ids,
        expected_split_role=args.expected_split_role,
    )
    selected: dict[str, dict] = {}
    for record in manifest["records"]:
        if not record["face_crop_path"] or not record["registered_identity_id"]:
            continue
        previous = selected.get(record["sample_id"])
        if (
            previous is None
            or record["face_quality"]["overall"] > previous["face_quality"]["overall"]
        ):
            selected[record["sample_id"]] = record
    by_identity: dict[str, list[dict]] = {}
    for record in selected.values():
        by_identity.setdefault(record["registered_identity_id"], []).append(record)
    eligible = {
        identity: sorted(records, key=lambda record: record["sample_id"])
        for identity, records in by_identity.items()
        if len(records) >= 2
    }
    gallery = tuple(records[0] for _, records in sorted(eligible.items()))
    queries = tuple(
        record for _, records in sorted(eligible.items()) for record in records[1:]
    )
    identity_index = {
        identity: index for index, identity in enumerate(sorted(eligible))
    }
    if not gallery or not queries:
        raise ValueError("evaluation ROI manifest has no closed-set FaceID cohort")
    gallery_dataset = RoiFaceReIDDataset(
        args.roi_manifest.parent,
        gallery,
        identity_index,
        align=args.architecture == "aligned_cls_residual_v5",
    )
    query_dataset = RoiFaceReIDDataset(
        args.roi_manifest.parent,
        queries,
        identity_index,
        align=args.architecture == "aligned_cls_residual_v5",
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    backbone, contract = load_receipt_bound_frozen_dino(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
    )
    observed_dino_contract = normalize_dino_local_artifact_contract(
        {
            "model_sha256": contract.model_sha256,
            "preprocessor_sha256": contract.preprocessor_sha256,
            "weight_receipt_sha256": contract.weight_receipt_sha256,
            "preprocessor_receipt_sha256": contract.preprocessor_receipt_sha256,
            "config_sha256": contract.config_sha256,
            "weight_source_contract_sha256": contract.weight_source.contract_sha256,
            "preprocessor_source_contract_sha256": (
                contract.preprocessor_source.contract_sha256
            ),
        }
    )
    validate_checkpoint_runtime_bindings(
        checkpoint,
        observed_dino_local_artifact_contract=observed_dino_contract,
        observed_weight_intake_bundle_sha256=file_sha256(args.weight_intake_bundle),
        observed_preprocessor_intake_bundle_sha256=file_sha256(
            args.preprocessor_intake_bundle
        ),
    )
    model = build_faceid_model(
        backbone, contract, architecture=args.architecture
    ).to(device)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    model.quality_head.load_state_dict(
        checkpoint["quality_head_state_dict"], strict=True
    )
    gallery_result = extract_face_embeddings(model, gallery_loader, device)
    query_result = extract_face_embeddings(model, query_loader, device)
    metrics = evaluate_face_retrieval(
        query_embeddings=query_result["embeddings"],
        gallery_embeddings=gallery_result["embeddings"],
        query_identity_ids=query_result["identity_ids"],
        gallery_identity_ids=gallery_result["identity_ids"],
        query_template_ids=query_result["template_ids"],
        gallery_template_ids=gallery_result["template_ids"],
    )
    paired_baseline = None
    if "baseline_embeddings" in gallery_result and "baseline_embeddings" in query_result:
        paired_baseline = paired_face_retrieval_comparison(
            baseline_query_embeddings=query_result["baseline_embeddings"],
            baseline_gallery_embeddings=gallery_result["baseline_embeddings"],
            candidate_query_embeddings=query_result["embeddings"],
            candidate_gallery_embeddings=gallery_result["embeddings"],
            query_identity_ids=query_result["identity_ids"],
            gallery_identity_ids=gallery_result["identity_ids"],
        )
    report = {
        "schema_version": "cvi.faceid_evaluation.v2",
        "architecture": args.architecture,
        "interpretation": (
            "closed-set diagnostic on a manifest-distinct, training-identity-disjoint "
            f"{args.expected_split_role!r} split"
        ),
        "independence": independence,
        "source_samples": len(manifest["source_sample_ids"]),
        "face_samples": len(selected),
        "sample_coverage": len(selected) / len(manifest["source_sample_ids"]),
        "gallery_identities": len(gallery),
        "queries": len(queries),
        "Rank-1": metrics["Rank-1"],
        "Rank-5": metrics["Rank-5"],
        "MRR": metrics["MRR"],
        "paired_frozen_baseline": paired_baseline,
        "metric_note": (
            "One relevant gallery identity per query; MRR is reported instead of "
            "aliasing AP/INP."
        ),
        "provenance": {
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "training_split_sha256": checkpoint["training_split_sha256"],
            "training_roi_manifest_sha256": checkpoint["training_roi_manifest_sha256"],
            "evaluation_roi_manifest_sha256": independence[
                "evaluation_roi_manifest_sha256"
            ],
            "dino_local_artifact_contract_sha256": checkpoint[
                "dino_local_artifact_contract_sha256"
            ],
            "faceid_contract_sha256": checkpoint["faceid_contract_sha256"],
            **_git_provenance(repository),
            "weight_intake_bundle_sha256": checkpoint["weight_intake_bundle_sha256"],
            "preprocessor_intake_bundle_sha256": checkpoint[
                "preprocessor_intake_bundle_sha256"
            ],
            "dependency_lock_sha256": file_sha256(repository / "uv.lock"),
            "device": args.device,
            "precision": "float16 autocast" if device.type == "cuda" else "float32",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_private_json_bundle(((args.output, report),))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
