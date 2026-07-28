"""Evaluate A0/N0/NT/N3/F0 on capture-disjoint NoseID oracle DEV folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cvi.nose_id.checkpoint import load_training_checkpoint
from cvi.nose_id.config import NoseIDConfig
from cvi.nose_id.dataset import NoseIDDataset, load_identity_split, load_noseid_manifest
from cvi.nose_id.evaluation import (
    evaluate_dev_folds,
    extract_oracle_representations,
)
from cvi.nose_id.protocol import build_dev_n3_folds
from cvi.nose_id.trainer import build_noseid_model, load_receipt_bound_frozen_dino
from cvi.protected_io import write_private_json_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-role", choices=("DEV",), default="DEV")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.checkpoint, map_location="cpu")
    NoseIDConfig.from_dict(checkpoint["noseid_config"])
    rows = load_noseid_manifest(args.manifest)
    identity_split = load_identity_split(args.split_file)
    dev_rows = tuple(row for row in rows if row.split_role == args.split_role)
    mapping = {
        identity: index
        for index, identity in enumerate(
            sorted({row.registered_dog_id for row in dev_rows})
        )
    }
    dataset = NoseIDDataset(
        args.data_root,
        dev_rows,
        mapping,
        identity_split=identity_split,
        split_role="DEV",
    )
    folds = build_dev_n3_folds(dev_rows, seed=args.seed)
    backbone, contract = load_receipt_bound_frozen_dino(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
    )
    observed_contract = {
        "model_sha256": contract.model_sha256,
        "preprocessor_sha256": contract.preprocessor_sha256,
        "weight_receipt_sha256": contract.weight_receipt_sha256,
        "preprocessor_receipt_sha256": contract.preprocessor_receipt_sha256,
    }
    if checkpoint["dino_contract"] != observed_contract:
        raise ValueError("checkpoint DINO contract differs from local artifact")
    model = build_noseid_model(backbone, contract)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    representations = extract_oracle_representations(
        model,
        dataset,
        preprocessor=contract.preprocessor,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    report = {
        "schema_version": "cvi.noseid.oracle_evaluation.v1",
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_best_metric": checkpoint["best_metric"],
        "seed": args.seed,
        "protocol": "CAPTURE_DISJOINT_DEV_N3",
        **evaluate_dev_folds(representations, dataset, folds),
    }
    write_private_json_bundle(((args.output, report),))
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
