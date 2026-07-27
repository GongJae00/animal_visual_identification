#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s <cpu|cuda>\n' "$0" >&2
}

if [[ $# -ne 1 ]] || [[ "$1" != "cpu" && "$1" != "cuda" ]]; then
    usage
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    printf 'error: uv is required: https://docs.astral.sh/uv/\n' >&2
    exit 1
fi

lane="$1"
printf 'Synchronizing the %s dependency lane...\n' "$lane"
uv sync \
    --locked \
    --extra "$lane" \
    --extra data \
    --extra models \
    --extra training \
    --group dev

LANE="$lane" uv run --no-sync python - <<'PY'
import importlib
import os

modules = (
    "cvi",
    "faiss",
    "jsonschema",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "pyarrow",
    "scipy",
    "tensorboard",
    "timm",
    "torch",
    "transformers",
)
for name in modules:
    importlib.import_module(name)
    print(f"[ok] import {name}")

import torch

lane = os.environ["LANE"]
print(f"[ok] selected dependency lane: {lane}")
if lane == "cuda" and not torch.cuda.is_available():
    raise SystemExit("error: cuda lane selected but torch.cuda.is_available() is false")

# Keep disabled research artifacts visible without presenting an install route.
print("[ok] SuperAnimal runtime: DISABLED")
PY
