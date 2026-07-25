"""Export pretrained PyTorch models to ONNX for evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def _export_dinov2_small(output_dir: Path) -> tuple[Path, dict]:
    """Export DINOv2-small ViT-S/14 → ONNX, return (model_path, backend_config)."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "facebook/dinov2-small", attn_implementation="sdpa"
    )
    model.eval()

    class Dinov2Wrapper(torch.nn.Module):
        def forward(self, x):
            out = model(x)
            return out.last_hidden_state[:, 0, :]

    wrapper = Dinov2Wrapper().eval()
    output_path = output_dir / "dinov2-small.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            input_names=["images"],
            output_names=["embedding"],
            dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=18,
        )
    _validate_onnx(output_path, 384)

    return output_path, _backend_config(384)


def _export_mobilenetv4(output_dir: Path) -> tuple[Path, dict]:
    """Export MobileNetV4-conv-small → ONNX, return (model_path, backend_config)."""
    import timm

    model = timm.create_model(
        "mobilenetv4_conv_small.e1200_r224_in1k", pretrained=True
    )
    model.eval()
    model.reset_classifier(0)

    class Mobilenetv4Wrapper(torch.nn.Module):
        def forward(self, x):
            out = model.forward_features(x)
            return out.mean(dim=[-2, -1])

    wrapper = Mobilenetv4Wrapper().eval()

    output_path = output_dir / "mobilenetv4-conv-small.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            input_names=["images"],
            output_names=["embedding"],
            dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=18,
        )
    _validate_onnx(output_path, 960)
    return output_path, _backend_config(960)


def _validate_onnx(path: Path, expected_dim: int) -> None:
    onnx.checker.check_model(str(path))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    assert inp.name == "images", f"input name: {inp.name}"
    assert out.name == "embedding", f"output name: {out.name}"
    assert out.shape[1] == expected_dim, f"output dim: {out.shape[1]} != {expected_dim}"
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = sess.run(["embedding"], {"images": dummy})[0]
    assert result.shape == (1, expected_dim), f"result shape: {result.shape}"
    assert np.all(np.isfinite(result)), "non-finite values in output"


def _backend_config(dim: int) -> dict:
    return {
        "schema_version": "cvi.onnx_backend_config.v1",
        "input_name": "images",
        "output_name": "embedding",
        "input_layout": "NCHW",
        "input_channels": 3,
        "input_height": 224,
        "input_width": 224,
        "vector_dimension": dim,
        "maximum_batch_size": 32,
    }


_MODEL_EXPORTERS = {
    "dinov2-small": _export_dinov2_small,
    "mobilenetv4-conv-small": _export_mobilenetv4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=list(_MODEL_EXPORTERS) + ["all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing model")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(_MODEL_EXPORTERS) if args.model == "all" else [args.model]

    for name in model_names:
        model_path = args.output_dir / f"{name.replace('_', '-')}.onnx"
        if model_path.exists() and not args.force:
            print(f"SKIP {name}: {model_path} exists (use --force to overwrite)")
            continue

        print(f"Exporting {name} ...", flush=True)
        exporter = _MODEL_EXPORTERS[name]
        model_path, config = exporter(args.output_dir)
        config_path = model_path.with_suffix(".onnx.config.json")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"  model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  config: {config_path}")


if __name__ == "__main__":
    main()
