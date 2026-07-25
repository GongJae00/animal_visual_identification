#!/usr/bin/env bash
# CVI 환경 점검 — 데이터, 모델, 심링크, 의존성 전수 검증
set -euo pipefail

PASS=0; FAIL=0; WARN=0
pass() { PASS=$((PASS+1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }
warn() { WARN=$((WARN+1)); echo "  ⚠️  $1"; }

echo "=== CVI 환경 점검 ==="
echo ""

# ── 1. Python ──
echo "[1] Python"
PY=$(command -v python3 || true)
if [ -n "$PY" ]; then
    VER=$("$PY" --version 2>&1)
    pass "Python: $VER"
else
    fail "python3 not found"
fi

# ── 2. uv ──
echo ""
echo "[2] 패키지 매니저"
if command -v uv &>/dev/null; then
    pass "uv: $(uv --version 2>&1)"
    if [ -f pyproject.toml ]; then
        "$PY" -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null && pass "torch 설치됨" || fail "torch 미설치 → uv sync"
        "$PY" -c "import faiss; print(f'  faiss {faiss.__version__}')" 2>/dev/null && pass "faiss 설치됨" || fail "faiss 미설치 → uv sync"
        "$PY" -c "import onnxruntime; print(f'  onnxruntime {onnxruntime.__version__}')" 2>/dev/null && pass "onnxruntime 설치됨" || fail "onnxruntime 미설치 → uv sync"
    fi
else
    fail "uv not found → https://docs.astral.sh/uv/#installation"
fi

# ── 3. GPU ──
echo ""
echo "[3] GPU"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "  ✅ GPU: $line"
    done
    "$PY" -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
else
    warn "nvidia-smi 없음 — CPU 모드 (GPU 가속 불가)"
fi

# ── 4. 데이터 저장소 ──
echo ""
echo "[4] 데이터 저장소"

DATA_DIR=$("$PY" -c "from cvi.model_paths import DATA_DIR; print(DATA_DIR)" 2>/dev/null || echo "")
if [ -z "$DATA_DIR" ]; then
    fail "모듈 import 실패 — uv sync 또는 CVI_DATA_DIR 확인"
elif [ -L "$DATA_DIR" ]; then
    TARGET=$(readlink -f "$DATA_DIR")
    pass "심링크: $DATA_DIR → $TARGET"
elif [ -d "$DATA_DIR" ]; then
    pass "디렉토리: $DATA_DIR"
else
    fail "DATA_DIR 없음: $DATA_DIR"
    fail "→ CVI_DATA_DIR 환경변수 설정 또는 ~/cvi_data 심링크 생성"
fi

if [ -n "$DATA_DIR" ]; then
    "$PY" -c "
from cvi.model_paths import DATASETS_DIR, CHECKPOINTS_DIR, CACHE_DIR
import json
for name, p in [('datasets/', DATASETS_DIR), ('checkpoints/', CHECKPOINTS_DIR), ('cache/', CACHE_DIR)]:
    print(f'  {\"✅\" if p.exists() else \"❌\"} {name:14s}  {p}')
" 2>/dev/null
fi

# ── 5. 데이터셋 ──
echo ""
echo "[5] 데이터셋"
if [ -n "$DATA_DIR" ]; then
"$PY" -c "
from cvi.model_paths import SUPPORTED_DATASETS, dataset_path
for k, v in SUPPORTED_DATASETS.items():
    p = dataset_path(k)
    if not p.exists():
        print(f'  ❌ {k:14s}  디렉토리 없음 — 다운로드 필요')
        continue
    imgs = sum(1 for _ in p.rglob(\"*.jpg\"))
    url = v.get('url', '')
    if imgs > 0:
        print(f'  ✅ {k:14s}  {imgs:6d}장  {v[\"desc\"]}')
    elif url:
        print(f'  ⚠️  {k:14s}  (0장)  ZIP 압축해제 필요: cd {p} && unzip *.zip')
    else:
        print(f'  ⚠️  {k:14s}  (0장)  수동 준비 필요')
" 2>/dev/null
fi

# ── 6. ONNX 모델 ──
echo ""
echo "[6] 훈련된 ONNX 모델"
"$PY" -c "
from cvi.model_paths import DINOV2_SMALL_ONNX, MOBILENETV4_CONV_SMALL_ONNX
for name, p in [('dinov2-small', DINOV2_SMALL_ONNX), ('mobilenetv4-conv-small', MOBILENETV4_CONV_SMALL_ONNX)]:
    if p.exists():
        sz = p.stat().st_size / 2**20
        print(f'  ✅ {name:20s}  {sz:.0f} MB  {p}')
    else:
        print(f'  ❌ {name:20s}  미존재')
" 2>/dev/null || fail "model_paths import 실패 → uv sync 필요"

# ── 7. 캐시 레지스트리 ──
echo ""
echo "[7] 레지스트리"
"$PY" -c "
from cvi.model_paths import IDENTITY_REGISTRY_DB, BINDING_JSON
for name, p in [('identity_registry.db', IDENTITY_REGISTRY_DB), ('binding.json', BINDING_JSON)]:
    if p.exists():
        sz = p.stat().st_size / 1024
        print(f'  ✅ {name:20s}  {sz:.0f} KB')
    else:
        print(f'  ❌ {name:20s}  미존재')
" 2>/dev/null

# ── 8. 사전학습 백본 캐시 ──
echo ""
echo "[8] 사전학습 백본 캐시"
"$PY" -c "
import torch
from pathlib import Path

# torch.hub 캐시 확인 (DINOv2)
torch_cache = Path(torch.hub.get_dir()) / 'checkpoints' / 'dinov2_vits14_pretrain.pth'
if torch_cache.exists():
    sz = torch_cache.stat().st_size / 2**20
    print(f'  ✅ DINOv2-Small: 캐시됨 ({sz:.0f} MB, torch.hub)')
else:
    print(f'  ⚠️  DINOv2-Small: 미캐시 (torch.hub.load 시 자동 다운로드)')

# HF Hub 캐시 확인 (ConvNeXt)
from huggingface_hub import scan_cache_dir
try:
    hf_cache = scan_cache_dir()
    convnext_found = False
    dinov2_hf_found = False
    for repo in hf_cache.repos:
        if 'facebook/convnext-base' in repo.repo_id:
            convnext_found = True
        if 'facebook/dinov2-small' in repo.repo_id:
            dinov2_hf_found = True
    if convnext_found:
        print(f'  ✅ ConvNeXt-Base: 캐시됨 (HF Hub)')
    else:
        print(f'  ⚠️  ConvNeXt-Base: 미캐시 (transformers.from_pretrained 시 자동 다운로드)')
    if dinov2_hf_found:
        print(f'  ✅ DINOv2-Small: HF Hub 캐시됨')
except Exception:
    print(f'  ⚠️  HF Hub 캐시 확인 불가')
" 2>/dev/null || warn "백본 캐시 확인 중 오류"

# ── 9. 추가 모델 ──
echo ""
echo "[9] 추가 모델"
"$PY" -c "
from cvi.model_paths import MODELS_DIR, MIEWID_NOSE_ONNX_PATH, SUPERANIMAL_QUADRUPED_PATH, SUPERANIMAL_ONNX_PATH, DOGFLW_LANDMARK_PATH
import os
paths = [
    ('MiewID ONNX (비문)',      MIEWID_NOSE_ONNX_PATH,          '다운로드: uv run tools/download_models.py --model miewid'),
    ('SuperAnimal PT (랜드마크)', SUPERANIMAL_QUADRUPED_PATH,      '다운로드: uv run tools/download_models.py --model superanimal'),
    ('SuperAnimal ONNX',        SUPERANIMAL_ONNX_PATH,           'PT에서 변환 필요'),
    ('DogFLW TFLite',           DOGFLW_LANDMARK_PATH,            'HF 레포 미생성 — SuperAnimal로 대체'),
]
for name, p, note in paths:
    if p.exists():
        sz = p.stat().st_size / 2**20
        print(f'  ✅ {name:28s}  {sz:6.0f} MB')
    else:
        print(f'  ❌ {name:28s}  ({note})')
" 2>/dev/null

echo ""
echo "═══════════════════════════════════════"
echo "  ✅ $PASS pass  ⚠️  $WARN warn  ❌ $FAIL fail"
echo "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "  🔧 빠른 시작:"
    echo "    uv sync && uv run python tools/download_datasets.py && uv run python tools/download_models.py"
    exit "$FAIL"
fi
