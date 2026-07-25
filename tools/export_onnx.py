"""Export a trained PyTorch checkpoint to ONNX.

Produces a single ONNX file suitable for OnnxExtractor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cvi.trainer import ArcFaceModel, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = TrainConfig(embedding_dim=args.embedding_dim, num_classes=0)
    model = ArcFaceModel(cfg)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.export_to_onnx(args.output)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
