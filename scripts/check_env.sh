#!/usr/bin/env bash
# CVI 설치 도우미 — CUDA + Python 환경 점검
# 실제 패키지 설치는 `uv sync --extra cuda` 로 진행합니다.
set -euo pipefail

echo "=== CVI: CUDA 환경 점검 ==="

# Python
PYTHON="$(command -v python3.12 || command -v python3)"
echo "  Python: $($PYTHON --version 2>&1)"

# CUDA toolkit
if command -v nvcc &>/dev/null; then
    echo "  nvcc: $(nvcc --version 2>&1 | grep release)"
else
    echo "  WARNING: nvcc 없음 — 학습은 안 되고 ONNX 추론만 가능"
fi

# NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU: $line"
    done
else
    echo "  WARNING: nvidia-smi 없음 — GPU 사용 불가"
fi

# uv
if ! command -v uv &>/dev/null; then
    echo "  uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "  > uv 설치 완료. 셸 재시작 후 다시 실행하세요."
    exit 0
fi
echo "  uv: $(uv --version 2>&1)"

echo ""
echo "  패키지 설치:"
echo "    uv sync --extra cuda --extra training"
echo ""
echo "  모델 다운로드:"
echo "    python tools/download_models.py --model miewid --hf-token \$HF_TOKEN"
