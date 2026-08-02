"""Train and export a research-only MobileNetV4 dog nose localizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import random
import time
from hashlib import sha256


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap10k-zip", type=Path, required=True)
    parser.add_argument("--dogflw-zip", type=Path, required=True)
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        required=True,
        help="Local timm MobileNetV4 Conv Small model.safetensors path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if os.path.lexists(args.output_dir):
        raise FileExistsError("nose-localizer output directory must not exist")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence-threshold must be in [0, 1]")


def _canonical_sha256(payload: object) -> str:
    from hashlib import sha256

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)

    # Heavy model, data, and export dependencies stay below CLI parsing so
    # ``--help`` works in a base source checkout.
    import numpy as np
    import onnx
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from artifact_contracts.artifact_manifest import (
        ArtifactLicense,
        ImagePreprocessing,
        NoseDetectorManifest,
        UsageLane,
    )
    from localization.nose_region.localizer import (
        DOGFLW_DERIVATION,
        IMAGE_MEAN,
        IMAGE_STD,
        INPUT_SIZE,
        KEYPOINT_ORDER,
        MOBILENETV4_MODEL_NAME,
        NoseDetectorWrapper,
        ZipNoseKeypointDataset,
        file_sha256,
        keypoint_metrics,
        load_mobilenetv4_localizer,
        parse_ap10k_zip,
        parse_dogflw_zip,
        partial_keypoint_loss,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    source_hashes = {
        "ap10k_zip_sha256": file_sha256(args.ap10k_zip),
        "dogflw_zip_sha256": file_sha256(args.dogflw_zip),
        "backbone_safetensors_sha256": file_sha256(args.backbone_weights),
    }
    ap10k = parse_ap10k_zip(args.ap10k_zip)
    dogflw = parse_dogflw_zip(args.dogflw_zip)
    required = (
        ("AP-10K train", ap10k["train"]),
        ("AP-10K val", ap10k["val"]),
        ("AP-10K test", ap10k["test"]),
        ("DogFLW train", dogflw["train"]),
        ("DogFLW test", dogflw["test"]),
    )
    for label, records in required:
        if not records:
            raise ValueError(f"{label} publisher split is empty")

    dogflw_internal_dev = tuple(
        record
        for record in dogflw["train"]
        if sha256(record.sample_id.encode("utf-8")).digest()[0] < 26
    )
    dogflw_internal_train = tuple(
        record for record in dogflw["train"] if record not in dogflw_internal_dev
    )
    if not dogflw_internal_train or not dogflw_internal_dev:
        raise ValueError("DogFLW deterministic internal TRAIN/DEV split is empty")
    train_dataset = ConcatDataset(
        (
            ZipNoseKeypointDataset(ap10k["train"]),
            ZipNoseKeypointDataset(dogflw_internal_train),
        )
    )
    evaluation_datasets = {
        "ap10k_val": ZipNoseKeypointDataset(ap10k["val"]),
        "ap10k_test": ZipNoseKeypointDataset(ap10k["test"]),
        "dogflw_internal_dev": ZipNoseKeypointDataset(dogflw_internal_dev),
        "dogflw_test": ZipNoseKeypointDataset(dogflw["test"]),
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    evaluation_loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for name, dataset in evaluation_datasets.items()
    }

    model = load_mobilenetv4_localizer(args.backbone_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
    )

    def evaluate(loader: DataLoader) -> dict[str, object]:
        model.eval()
        predictions = []
        targets = []
        visibility = []
        normalizers = []
        with torch.inference_mode():
            for batch in loader:
                prediction = model(batch["image"].to(device, non_blocking=True))
                predictions.append(prediction.cpu())
                targets.append(batch["target"])
                visibility.append(batch["visibility"])
                normalizers.append(batch["normalizer"])
        return keypoint_metrics(
            torch.cat(predictions),
            torch.cat(targets),
            torch.cat(visibility),
            torch.cat(normalizers),
            confidence_threshold=args.confidence_threshold,
        )

    history: list[dict[str, object]] = []
    best_validation_nme = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.time()
    for epoch_index in range(args.epochs):
        model.train()
        sums = {"total": 0.0, "coordinate": 0.0, "confidence": 0.0}
        sample_count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            visibility = batch["visibility"].to(device, non_blocking=True)
            support = batch["support"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = partial_keypoint_loss(
                model(images), targets, visibility, support
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_size = int(images.shape[0])
            sample_count += batch_size
            for name in sums:
                sums[name] += float(losses[name].detach()) * batch_size
        scheduler.step()
        ap10k_validation = evaluate(evaluation_loaders["ap10k_val"])
        dogflw_validation = evaluate(evaluation_loaders["dogflw_internal_dev"])
        if ap10k_validation["NME"] is None or dogflw_validation["NME"] is None:
            raise RuntimeError("localizer validation NME has no eligible keypoints")
        validation_nme = (
            float(ap10k_validation["NME"]) + float(dogflw_validation["NME"])
        ) / 2.0
        epoch_number = epoch_index + 1
        history.append(
            {
                "epoch": epoch_number,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": {name: value / sample_count for name, value in sums.items()},
                "selection_mean_NME": validation_nme,
                "ap10k_val": ap10k_validation,
                "dogflw_internal_dev": dogflw_validation,
            }
        )
        if float(validation_nme) < best_validation_nme:
            best_validation_nme = float(validation_nme)
            best_epoch = epoch_number
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    final_metrics = {
        name: evaluate(loader) for name, loader in evaluation_loaders.items()
    }

    license_payload = {
        "license_id": "CC-BY-NC-4.0-derived",
        "usage_lane": "RESEARCH_ONLY",
        "reason": (
            "Derived from DogFLW CC-BY-NC-4.0 annotations; the combined model "
            "must remain non-commercial research-only."
        ),
    }
    split_counts = {
        "ap10k": {split: len(records) for split, records in ap10k.items()},
        "dogflw": {split: len(records) for split, records in dogflw.items()},
    }
    training_config = {
        "model_name": MOBILENETV4_MODEL_NAME,
        "input_size": INPUT_SIZE,
        "keypoint_order": list(KEYPOINT_ORDER),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "publisher_split_policy": {
            "training": ["ap10k/train", "dogflw/train deterministic 90%"],
            "selection": ["ap10k/val", "dogflw/train deterministic 10%"],
            "evaluation": ["ap10k/test", "dogflw/test"],
            "selection_rule": "lowest mean AP-10K/DogFLW development NME",
        },
    }
    bindings = {
        "schema_version": "cvi.nose_localizer.bindings.v1",
        "sources": source_hashes,
        "split_counts": split_counts,
        "training_config": training_config,
        "license": license_payload,
    }
    bindings["content_sha256"] = _canonical_sha256(bindings)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": "cvi.nose_localizer.checkpoint.v1",
            "bindings": bindings,
            "selected_epoch": best_epoch,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)

    model.to(torch.device("cpu")).eval()
    detector = NoseDetectorWrapper(model).eval()
    detector_path = args.output_dir / "detector.onnx"
    dummy = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            detector,
            (dummy,),
            detector_path,
            input_names=["images"],
            output_names=["detections"],
            opset_version=18,
            external_data=False,
            dynamo=False,
        )
    onnx_model = onnx.load(detector_path)
    onnx.checker.check_model(onnx_model)
    detector_sha256 = file_sha256(detector_path)
    manifest = NoseDetectorManifest(
        artifact_id=f"nose-localizer-detector-{detector_sha256[:16]}",
        artifact_sha256=detector_sha256,
        input_name="images",
        input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
        output_name="detections",
        output_shape=(1, 1, 5),
        license=ArtifactLicense(
            license_id="CC-BY-NC-4.0-derived",
            usage_lane=UsageLane.RESEARCH_ONLY,
        ),
        preprocessing=ImagePreprocessing(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bilinear",
            scale=1.0 / 255.0,
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
            clahe=None,
        ),
        confidence_threshold=args.confidence_threshold,
    )
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())

    summary = {
        "schema_version": "cvi.nose_localizer.training_summary.v1",
        "status": "RESEARCH_ONLY",
        "license": license_payload,
        "bindings": bindings,
        "checkpoint_sha256": checkpoint_sha256,
        "detector_onnx_sha256": detector_sha256,
        "manifest_sha256": file_sha256(manifest_path),
        "selected_epoch": best_epoch,
        "duration_seconds": time.time() - started,
        "history": history,
        "evaluation": final_metrics,
        "target_mapping": {
            "ap10k": {
                "left_eye_center": 0,
                "right_eye_center": 1,
                "nasal_inferior": "2 (publisher nose center/nose tip)",
            },
            "dogflw": {
                name: list(indices) for name, indices in DOGFLW_DERIVATION.items()
            },
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "LICENSE.txt").write_text(
        "RESEARCH_ONLY\nCC-BY-NC-4.0-derived\n\n"
        + license_payload["reason"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
