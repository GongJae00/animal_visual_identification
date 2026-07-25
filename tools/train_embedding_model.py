"""Train an ArcFace embedding model on oracle crops.

Reads a split registry binding to map samples to registered_dog_id labels,
loads oracle crops from the crop export directory, and runs supervised
ArcFace training.  The trained backbone can be exported to ONNX for
deployment.

Supported backbones (--backbone):
  dinov2-small   — DINOv2-small, 384-d (default)
  convnext-base  — ConvNeXt-Base, 768-d
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from cvi.protected_io import read_strict_json_object
from cvi.split_registry_binding import build_binding
from cvi.trainer import (
    ConvNeXtEmbedding,
    Dinov2Embedding,
    TrainConfig,
    train_model,
)

_BACKBONE_FACTORIES: dict[str, type[nn.Module]] = {
    "dinov2-small": Dinov2Embedding,
    "convnext-base": ConvNeXtEmbedding,
}


def _select_binding_records(
    binding_payload: dict,
    access_filter: str,
) -> list[dict]:
    return [
        b
        for b in binding_payload.get("bindings", [])
        if b.get("model_access") == access_filter
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--crop-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone", default="dinov2-small",
                        choices=list(_BACKBONE_FACTORIES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", default=None, action="store_true", dest="gc")
    parser.add_argument("--no-gradient-checkpointing", default=None, action="store_false", dest="gc")
    parser.add_argument("--compile", default=None, action="store_true", dest="compile")
    parser.add_argument("--no-compile", default=None, action="store_false", dest="compile")
    parser.add_argument("--preload", default=None, action="store_true", dest="preload")
    parser.add_argument("--no-preload", default=None, action="store_false", dest="preload")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(json.dumps({"event": "device", "device": str(device)}), flush=True)

    assignment = read_strict_json_object(args.assignment)
    binding = build_binding(assignment, args.registry_db)
    if not binding.is_valid:
        print(json.dumps({
            "event": "binding_invalid",
            "unregistered_count": len(binding.unregistered_tokens),
        }), flush=True)
        raise SystemExit(1)

    train_records = _select_binding_records(binding.to_dict(), "MODEL_TRAINING")
    val_records = _select_binding_records(binding.to_dict(), "MODEL_SELECTION")
    print(json.dumps({
        "event": "binding_stats",
        "train_identities": len(set(r["registered_dog_id"] for r in train_records)),
        "train_samples": sum(r["sample_count"] for r in train_records),
        "val_identities": len(set(r["registered_dog_id"] for r in val_records)),
        "val_samples": sum(r["sample_count"] for r in val_records),
    }), flush=True)

    config_dict = {}
    if args.config and args.config.exists():
        config_dict = json.loads(args.config.read_text())
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
    if args.lr is not None:
        config_dict["lr"] = args.lr
    if args.num_workers is not None:
        config_dict["num_workers"] = args.num_workers
    if args.gc is not None:
        config_dict["gradient_checkpointing"] = args.gc
    if args.compile is not None:
        config_dict["compile_model"] = args.compile
    if args.preload is not None:
        config_dict["preload_images"] = args.preload
    config_dict["checkpoint_dir"] = str(args.output_dir / "checkpoints")
    config = TrainConfig.from_dict(config_dict)

    backbone_factory = _BACKBONE_FACTORIES[args.backbone]
    t0 = time.time()
    summary = train_model(
        config=config,
        crop_root=args.crop_root,
        train_binding=train_records,
        val_binding=val_records,
        device=device,
        backbone_factory=backbone_factory,
    )
    elapsed = time.time() - t0

    summary["tool_provenance"] = {"tool": str(Path(__file__).resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )

    print(json.dumps({
        "status": "DONE",
        "elapsed_seconds": round(elapsed, 1),
        "best_checkpoint": summary.get("best_checkpoint"),
        "total_steps": summary["total_steps"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
