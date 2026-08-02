"""Train landmark-aware FaceID from a localization ROI manifest."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def _git_provenance(repository: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True, cwd=repository
    ).strip()
    status = subprocess.check_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        text=True,
        cwd=repository,
    )
    return {
        "code_commit": commit,
        "worktree_dirty": bool(status.strip()),
        "worktree_status_basis": (
            "git status --porcelain=v1 --untracked-files=normal; includes staged, "
            "unstaged, and untracked path status, not untracked file contents"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--architecture",
        choices=("regional_v4", "cls_residual_v5", "aligned_cls_residual_v5"),
        default="regional_v4",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Execution device; CUDA is never selected as an implicit CPU fallback.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from identity_methods.face.checkpoint import (
        build_checkpoint_bindings,
        build_faceid_source_contract,
        content_sha256,
        file_sha256,
        normalize_dino_local_artifact_contract,
    )
    from identity_methods.face.dataset import RoiFaceReIDDataset
    from experiments.face_evaluation import (
        evaluate_face_retrieval,
        extract_face_embeddings,
    )
    from identity_methods.face.losses import FaceIDObjective, FaceResidualObjective
    from identity_methods.face.sampler import FaceReIDSampler
    from identity_methods.face.trainer import (
        build_faceid_model,
        build_faceid_optimizer,
        load_receipt_bound_frozen_dino,
        train_faceid_epoch,
    )
    from localization.roi_manifest import read_roi_manifest

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu explicitly"
        )
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("FaceID CUDA training requires bfloat16 autocast support")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    class PhotometricAugment:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            gain = 0.85 + 0.30 * torch.rand(())
            bias = -0.05 + 0.10 * torch.rand(())
            noise = torch.randn_like(tensor) * (0.01 * torch.rand(()))
            return (tensor * gain + bias + noise).clamp(0.0, 1.0)

    manifest = read_roi_manifest(args.roi_manifest)
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
    records = tuple(selected.values())
    identities = sorted({record["registered_identity_id"] for record in records})
    np.random.RandomState(args.seed).shuffle(identities)
    split = int(0.8 * len(identities))
    train_candidates = set(identities[:split])
    counts = Counter(record["registered_identity_id"] for record in records)
    train_ids = {identity for identity, count in counts.items() if count >= 4}
    train_ids &= train_candidates
    dev_ids = {identity for identity in identities[split:] if counts[identity] >= 2}
    train_records = tuple(
        record for record in records if record["registered_identity_id"] in train_ids
    )
    dev_records = tuple(
        record for record in records if record["registered_identity_id"] in dev_ids
    )
    train_index = {identity: index for index, identity in enumerate(sorted(train_ids))}
    dev_index = {identity: index for index, identity in enumerate(sorted(dev_ids))}
    crop_root = args.roi_manifest.parent
    train_dataset = RoiFaceReIDDataset(
        crop_root,
        train_records,
        train_index,
        augment=PhotometricAugment(),
        align=args.architecture == "aligned_cls_residual_v5",
        paired_augment=args.architecture != "regional_v4",
    )
    dev_dataset = RoiFaceReIDDataset(
        crop_root,
        dev_records,
        dev_index,
        align=args.architecture == "aligned_cls_residual_v5",
    )
    sampler = FaceReIDSampler(
        [record["registered_identity_id"] for record in train_records],
        [record["capture_group_id"] or "unknown" for record in train_records],
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    backbone, contract = load_receipt_bound_frozen_dino(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
    )
    model = build_faceid_model(
        backbone, contract, architecture=args.architecture
    ).to(device)
    objective_type = (
        FaceResidualObjective
        if args.architecture != "regional_v4"
        else FaceIDObjective
    )
    objective = objective_type(model.output_dim, len(train_index)).to(device)
    optimizer = build_faceid_optimizer(model, objective)
    for group in optimizer.param_groups:
        group["lr"] = 3e-4
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_mrr = -1.0
    split_payload = {
        "seed": args.seed,
        "train_identities": sorted(train_ids),
        "dev_identities": sorted(dev_ids),
        "train_samples": sorted(record["sample_id"] for record in train_records),
        "dev_samples": sorted(record["sample_id"] for record in dev_records),
    }
    repository = Path(__file__).resolve().parents[1]
    dino_artifact_contract = normalize_dino_local_artifact_contract(
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
    faceid_contract = build_faceid_source_contract(
        repository, architecture=args.architecture
    )
    checkpoint_bindings = build_checkpoint_bindings(
        dino_local_artifact_contract=dino_artifact_contract,
        weight_intake_bundle_sha256=file_sha256(args.weight_intake_bundle),
        preprocessor_intake_bundle_sha256=file_sha256(args.preprocessor_intake_bundle),
        faceid_contract=faceid_contract,
        training_roi_manifest_sha256=content_sha256(manifest),
        training_identity_ids=sorted(train_ids),
    )
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        train_metrics = train_faceid_epoch(
            model,
            objective,
            train_loader,
            optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )
        dev = extract_face_embeddings(model, dev_loader, device)
        metrics = evaluate_face_retrieval(
            query_embeddings=dev["embeddings"],
            gallery_embeddings=dev["embeddings"],
            query_identity_ids=dev["identity_ids"],
            gallery_identity_ids=dev["identity_ids"],
            query_template_ids=dev["template_ids"],
            gallery_template_ids=dev["template_ids"],
        )
        row = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "Rank-1": metrics["Rank-1"],
            "Rank-5": metrics["Rank-5"],
            "MRR": metrics["MRR"],
        }
        history.append(row)
        print(json.dumps(row))
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            torch.save(
                {
                    **checkpoint_bindings,
                    "epoch": epoch + 1,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "quality_head_state_dict": model.quality_head.state_dict(),
                    "objective_state_dict": objective.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "identity_to_index": train_index,
                    "training_split_sha256": content_sha256(split_payload),
                    "MRR": best_mrr,
                    "Rank-1": metrics["Rank-1"],
                },
                args.output_dir / "best.pt",
            )
    summary = {
        "schema_version": "cvi.faceid_training_summary.v2",
        "architecture": args.architecture,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_images": len(train_records),
        "train_identities": len(train_ids),
        "dev_images": len(dev_records),
        "dev_identities": len(dev_ids),
        "best_MRR": best_mrr,
        "history": history,
        "provenance": {
            **_git_provenance(repository),
            "roi_manifest_sha256": checkpoint_bindings["training_roi_manifest_sha256"],
            "split_sha256": content_sha256(split_payload),
            "dino_local_artifact_contract_sha256": checkpoint_bindings[
                "dino_local_artifact_contract_sha256"
            ],
            "faceid_contract_sha256": checkpoint_bindings["faceid_contract_sha256"],
            "weight_intake_bundle_sha256": checkpoint_bindings[
                "weight_intake_bundle_sha256"
            ],
            "preprocessor_intake_bundle_sha256": checkpoint_bindings[
                "preprocessor_intake_bundle_sha256"
            ],
            "dependency_lock_sha256": file_sha256(repository / "uv.lock"),
            "device": args.device,
            "precision": "bfloat16 autocast" if device.type == "cuda" else "float32",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
