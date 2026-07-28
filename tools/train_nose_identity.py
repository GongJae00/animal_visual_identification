"""Train the frozen-DINO NoseID-v1 oracle Stage-C experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cvi.nose_id.checkpoint import (
    replace_training_checkpoint,
    save_training_checkpoint,
)
from cvi.nose_id.config import NoseIDConfig, NoseIDTrainConfig
from cvi.nose_id.dataset import (
    NoseIDDataset,
    load_identity_split,
    load_noseid_manifest,
)
from cvi.nose_id.evaluation import (
    evaluate_dev_folds,
    extract_oracle_representations,
)
from cvi.nose_id.hard_negative import (
    mine_hard_neighbors,
    select_session_balanced_indices,
)
from cvi.nose_id.losses import NoseIDObjective
from cvi.nose_id.protocol import build_dev_n3_folds
from cvi.nose_id.sampler import CrossSessionPKBatchSampler
from cvi.nose_id.trainer import (
    build_frozen_stage_optimizer,
    build_noseid_model,
    build_stage_c_scheduler,
    load_receipt_bound_frozen_dino,
    train_frozen_stage_epoch,
)
from cvi.protected_io import write_private_json_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--micro-batch-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--mine-hard-negatives-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _dino_fields(contract) -> dict[str, str]:
    return {
        "model_sha256": contract.model_sha256,
        "preprocessor_sha256": contract.preprocessor_sha256,
        "weight_receipt_sha256": contract.weight_receipt_sha256,
        "preprocessor_receipt_sha256": contract.preprocessor_receipt_sha256,
    }


def _checkpoint_arguments(
    *,
    model,
    objective,
    optimizer,
    scheduler,
    scaler,
    identity_to_index,
    noseid_config,
    train_config,
    best_map,
    dino_contract,
    epoch,
    global_step,
) -> dict:
    return {
        "model": model,
        "objective": objective,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "identity_to_index": identity_to_index,
        "noseid_config": noseid_config,
        "train_config": train_config,
        "best_dev_n3_map": best_map,
        "dino_contract": dino_contract,
        "epoch": epoch,
        "global_step": global_step,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.epochs <= 0 or args.num_workers < 0:
        raise ValueError("epochs must be positive and num_workers non-negative")
    if args.eval_every <= 0 or args.mine_hard_negatives_every <= 0:
        raise ValueError("evaluation and mining intervals must be positive")
    if os.path.lexists(args.output_dir):
        raise FileExistsError("NoseID output directory must not exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows = load_noseid_manifest(args.manifest)
    identity_split = load_identity_split(args.split_file)
    train_rows = tuple(row for row in rows if row.split_role == "TRAIN")
    dev_rows = tuple(row for row in rows if row.split_role == "DEV")
    if not train_rows or not dev_rows:
        raise ValueError("NoseID training requires TRAIN and DEV rows")
    identity_to_index = {
        identity: index
        for index, identity in enumerate(
            sorted({row.registered_dog_id for row in train_rows})
        )
    }
    dev_identity_to_index = {
        identity: index
        for index, identity in enumerate(
            sorted({row.registered_dog_id for row in dev_rows})
        )
    }
    train_dataset = NoseIDDataset(
        args.data_root,
        train_rows,
        identity_to_index,
        identity_split=identity_split,
        split_role="TRAIN",
    )
    dev_dataset = NoseIDDataset(
        args.data_root,
        dev_rows,
        dev_identity_to_index,
        identity_split=identity_split,
        split_role="DEV",
    )
    sampler = CrossSessionPKBatchSampler(
        [row.registered_dog_id for row in train_rows],
        [row.session_id for row in train_rows],
        seed=args.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    folds = build_dev_n3_folds(dev_rows, seed=args.seed)

    backbone, contract = load_receipt_bound_frozen_dino(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
    )
    model = build_noseid_model(backbone, contract).to(device)
    objective = NoseIDObjective(512, len(identity_to_index)).to(device)
    optimizer = build_frozen_stage_optimizer(model, objective)
    scheduler = build_stage_c_scheduler(
        optimizer, steps_per_epoch=len(loader), epochs=args.epochs
    )
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not use_bfloat16
    )
    noseid_config = NoseIDConfig().to_dict()
    train_config = NoseIDTrainConfig(
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=64 // args.micro_batch_size,
        frozen_epochs=args.epochs,
        seed=args.seed,
    )
    dino_fields = _dino_fields(contract)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    checkpoint_dir.mkdir()
    metrics_dir.mkdir()
    global_step = 0
    best_map = 0.0
    best_key = (-1.0, -1.0, -1.0, float("-inf"))
    best_epoch = 0
    history: list[dict] = []
    first_quality_auxiliary: float | None = None
    gate_frozen = False
    started = time.time()
    for epoch_index in range(args.epochs):
        epoch_number = epoch_index + 1
        if gate_frozen and epoch_number == 4:
            for parameter in model.gate_head[-1].parameters():
                parameter.requires_grad = True
            gate_frozen = False
        sampler.set_epoch(epoch_index)
        train_metrics = train_frozen_stage_epoch(
            model,
            objective,
            loader,
            optimizer,
            device=device,
            epoch=epoch_index,
            scaler=scaler,
            scheduler=scheduler,
            micro_batch_size=args.micro_batch_size,
            use_bfloat16=use_bfloat16,
        )
        global_step += len(loader)
        if epoch_number % args.eval_every != 0:
            raise ValueError("Stage-C checkpoint selection requires evaluation every epoch")
        representations = extract_oracle_representations(
            model,
            dev_dataset,
            preprocessor=contract.preprocessor,
            device=device,
            batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
        )
        dev_report = evaluate_dev_folds(representations, dev_dataset, folds)
        aggregate = dev_report["aggregate"]
        events: list[str] = []
        if first_quality_auxiliary is None:
            first_quality_auxiliary = train_metrics["quality_auxiliary"]
        if (
            epoch_number == 5
            and train_metrics["quality_auxiliary"] >= first_quality_auxiliary
        ):
            objective.quality_auxiliary_weight.fill_(0.10)
            events.append("QUALITY_AUXILIARY_WEIGHT_INCREASED_TO_0.10")
        if epoch_number == 2:
            gates = representations["gates"]
            native = representations["native_short_side"]
            collapse = bool(
                np.any(np.median(gates, axis=0) > 0.90)
                or np.median(gates[:, 2]) > 0.70
                or (
                    np.any(native >= 224.0)
                    and np.median(gates[native >= 224.0, 1]) < 0.10
                )
            )
            if collapse:
                with torch.no_grad():
                    model.gate_head[-1].weight.zero_()
                    model.gate_head[-1].bias.copy_(
                        torch.log(
                            model.gate_head[-1].bias.new_tensor((0.45, 0.40, 0.15))
                        )
                    )
                for parameter in model.gate_head[-1].parameters():
                    parameter.requires_grad = False
                gate_frozen = True
                events.append("GATE_COLLAPSE_FIXED_WEIGHTS_THROUGH_EPOCH_3")
        selection_key = (
            aggregate["N3"]["mAP"],
            aggregate["N3"]["Rank-1"],
            aggregate["F0_FIXED"]["mAP"],
            -train_metrics["total"],
        )
        epoch_report = {
            "schema_version": "cvi.noseid.epoch_metrics.v1",
            "epoch": epoch_number,
            "global_step": global_step,
            "train": train_metrics,
            "development": dev_report,
            "events": events,
        }
        write_private_json_bundle(
            ((metrics_dir / f"epoch_{epoch_number:03d}.json", epoch_report),)
        )
        if selection_key > best_key:
            best_key = selection_key
            best_map = aggregate["N3"]["mAP"]
            best_epoch = epoch_number
            replace_training_checkpoint(
                checkpoint_dir / "best.pt",
                **_checkpoint_arguments(
                    model=model,
                    objective=objective,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    identity_to_index=identity_to_index,
                    noseid_config=noseid_config,
                    train_config=train_config,
                    best_map=best_map,
                    dino_contract=dino_fields,
                    epoch=epoch_number,
                    global_step=global_step,
                ),
            )
        replace_training_checkpoint(
            checkpoint_dir / "last.pt",
            **_checkpoint_arguments(
                model=model,
                objective=objective,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                identity_to_index=identity_to_index,
                noseid_config=noseid_config,
                train_config=train_config,
                best_map=best_map,
                dino_contract=dino_fields,
                epoch=epoch_number,
                global_step=global_step,
            ),
        )
        if epoch_number % 5 == 0:
            save_training_checkpoint(
                checkpoint_dir / f"epoch_{epoch_number:03d}.pt",
                **_checkpoint_arguments(
                    model=model,
                    objective=objective,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    identity_to_index=identity_to_index,
                    noseid_config=noseid_config,
                    train_config=train_config,
                    best_map=best_map,
                    dino_contract=dino_fields,
                    epoch=epoch_number,
                    global_step=global_step,
                ),
            )
        if (
            epoch_number >= 2
            and epoch_number % args.mine_hard_negatives_every == 0
        ):
            train_representations = extract_oracle_representations(
                model,
                train_dataset,
                preprocessor=contract.preprocessor,
                device=device,
                batch_size=args.micro_batch_size,
                num_workers=args.num_workers,
                include_appearance=False,
            )
            selected = select_session_balanced_indices(
                [row.registered_dog_id for row in train_rows],
                [row.session_id for row in train_rows],
                maximum_per_identity=8,
            )
            neighbors = mine_hard_neighbors(
                train_representations["N3"][selected],
                np.asarray([train_rows[index].registered_dog_id for index in selected]),
                top_k=50,
            )
            write_private_json_bundle(
                ((
                    args.output_dir / f"hard_neighbors_epoch_{epoch_number:03d}.json",
                    {
                        "schema_version": "cvi.noseid.hard_neighbors.v1",
                        "epoch": epoch_number,
                        "neighbors": {
                            identity: list(values)
                            for identity, values in sorted(neighbors.items())
                        },
                    },
                ),)
            )
            sampler.set_hard_neighbors(neighbors)
        history.append(
            {
                "epoch": epoch_number,
                "train_total": train_metrics["total"],
                "DEV_N3_mAP": aggregate["N3"]["mAP"],
                "DEV_N3_Rank-1": aggregate["N3"]["Rank-1"],
                "DEV_F0_FIXED_mAP": aggregate["F0_FIXED"]["mAP"],
                "events": events,
            }
        )
    summary = {
        "schema_version": "cvi.noseid.train_summary.v1",
        "best_epoch": best_epoch,
        "best_DEV_N3_mAP": best_map,
        "epochs": args.epochs,
        "global_step": global_step,
        "wall_seconds": time.time() - started,
        "device": str(device),
        "precision": "bf16" if use_bfloat16 else ("fp16" if device.type == "cuda" else "float32"),
        "history": history,
    }
    write_private_json_bundle(((args.output_dir / "train_summary.json", summary),))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
