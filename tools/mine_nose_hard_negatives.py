"""Mine exact NoseID-v1 prototype hard neighbors from an admitted checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cvi.nose_id.checkpoint import load_training_checkpoint
from cvi.nose_id.dataset import NoseIDDataset, load_identity_split, load_noseid_manifest
from cvi.nose_id.evaluation import extract_oracle_representations
from cvi.nose_id.hard_negative import mine_hard_neighbors, select_session_balanced_indices
from cvi.nose_id.trainer import build_noseid_model, load_receipt_bound_frozen_dino
from cvi.protected_io import write_private_json_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-role", choices=("TRAIN",), default="TRAIN")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--samples-per-identity", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples_per_identity <= 0 or args.top_k <= 0 or args.batch_size <= 0:
        raise ValueError("sample, neighbor, and batch counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.checkpoint, map_location="cpu")
    rows = load_noseid_manifest(args.manifest)
    identity_split = load_identity_split(args.split_file)
    train_rows = tuple(row for row in rows if row.split_role == "TRAIN")
    mapping = {
        identity: index
        for index, identity in enumerate(
            sorted({row.registered_dog_id for row in train_rows})
        )
    }
    dataset = NoseIDDataset(
        args.data_root,
        train_rows,
        mapping,
        identity_split=identity_split,
        split_role="TRAIN",
    )
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
        include_appearance=False,
    )
    selected = select_session_balanced_indices(
        [row.registered_dog_id for row in train_rows],
        [row.session_id for row in train_rows],
        maximum_per_identity=args.samples_per_identity,
    )
    neighbors = mine_hard_neighbors(
        representations["N3"][selected],
        np.asarray([train_rows[index].registered_dog_id for index in selected]),
        top_k=args.top_k,
    )
    report = {
        "schema_version": "cvi.noseid.hard_neighbors.v1",
        "checkpoint_epoch": checkpoint["epoch"],
        "samples_per_identity": args.samples_per_identity,
        "top_k": args.top_k,
        "neighbors": {
            identity: list(values) for identity, values in sorted(neighbors.items())
        },
    }
    write_private_json_bundle(((args.output, report),))
    print(json.dumps({"identities": len(neighbors), "top_k": args.top_k}))


if __name__ == "__main__":
    main()
