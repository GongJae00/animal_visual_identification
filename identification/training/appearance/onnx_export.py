"""Export a trained PyTorch checkpoint to ONNX.

Produces a single ONNX file suitable for OnnxExtractor.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch

from identification.training.appearance.trainer import (
    ArcFaceModel,
    ConvNeXtEmbedding,
    Dinov2Embedding,
    parse_training_checkpoint_config,
)


_BACKBONES = {
    "dinov2-small": Dinov2Embedding,
    "convnext-base": ConvNeXtEmbedding,
}


def reconstruct_model(
    payload: object, *, model_directory: Path | None = None
) -> ArcFaceModel:
    cfg = parse_training_checkpoint_config(payload)
    backbone_factory = _BACKBONES.get(cfg.model_name)
    if backbone_factory is None:
        raise RuntimeError(f"unsupported checkpoint backbone {cfg.model_name!r}")
    if model_directory is not None:
        backbone_factory = partial(
            backbone_factory, model_directory=model_directory
        )
    model = ArcFaceModel(cfg, backbone_factory=backbone_factory)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = reconstruct_model(payload, model_directory=args.model_dir)
    model.to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.export_to_onnx(args.output)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
