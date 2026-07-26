# Canine Video Identity (CVI)

영상 기반 개체 식별 시스템을 목표로 하는 초기 연구·구현 저장소입니다.
현재 검증 가능한 recognizer 경로는 단일 appearance baseline 준비 단계이며,
비문·랜드마크·불확실성 채널은 학습 artifact와 성능 증거가 없어 기본 비활성화됩니다.

## 개요

```
현재: 이미지 crop → appearance embedding → exact cosine gallery search

목표: 영상 decode → 검출 → tracking → quality frame selection
      → 검증된 evidence → track aggregation → calibration/open-set → identity event
```

## 설치

### CUDA 환경 (개발/학습 서버)

```bash
git clone <repo-url>
cd canine_video_identity
uv sync --extra cuda --extra training
```

### CPU 환경 (Raspberry Pi / 엣지 디바이스)

```bash
git clone <repo-url>
cd canine_video_identity
uv sync --extra cpu
```

> `uv`가 설치되어 있지 않다면: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 모델 & 데이터 준비

```bash
uv run python tools/download_datasets.py    # 공개 데이터셋 다운로드
uv run python tools/download_models.py      # 추론 모델 다운로드
uv run bash scripts/check_env.sh            # 환경 전수 점검
```

## 빠른 시작

### 1. 개 검출

```python
from cvi.detection import DogDetector, DogDetectorConfig

detector = DogDetector(DogDetectorConfig(device="cuda:0"))
detections = detector.detect("dog_video.mp4")
```

### 2. 특징 추출

```python
from cvi.evidence import Dinov2WithUncertainty

appearance = Dinov2WithUncertainty()
embedding = appearance.extract(crop_image)  # 384-d L2 정규화 벡터
```

### 3. 개체 등록

```python
from cvi.deployment import CVIDeploymentCUDA

deploy = CVIDeploymentCUDA({
    "channels": {
        "appearance": {"type": "dinov2"},
    },
    "index_dir": "./cvi_index",
})
deploy.enroll(image, dog_id="뽀삐", breed="비글")
```

### 4. 개체 검색

```python
results = deploy.search(query_image, top_k=5)
for r in results:
    print(f"{r.registered_dog_id}: {r.similarity:.3f}")
    print(f"  증거: {r.evidence}")
```

## 디렉토리

| 디렉토리 | 설명 |
|----------|------|
| `src/cvi/` | 핵심 패키지 |
| `src/cvi/backbones/` | 백본 후보 (DINOv2, ConvNeXt; 가짜 TinyViT 비활성화) |
| `src/cvi/heads/` | 학습 헤드 (ArcFace, MagFace, Evidential) |
| `src/cvi/evidence/` | evidence 추출기와 fail-closed 모델 계약 |
| `src/cvi/fusion/` | 점수 융합 + 보정 + Open-Set 판정 |
| `src/cvi/index/` | FAISS 검색 인덱스 |
| `src/cvi/pipeline/` | 등록/검색 파이프라인 |
| `src/cvi/deployment/` | 배포 (CUDA/CPU 분기) |
| `tools/` | 학습/평가/다운로드 CLI 도구 |
| `tests/` | 단위·계약·CLI 회귀 테스트 |
| `models/` | 모델 가중치 (Git 미포함) |
| `data/` | 데이터셋 (Git 미포함) |
| `configs/` | 설정 파일 |
| `docs/` | 기술 문서 |

## 성능 지표

평가 프레임워크를 통한 채널별 독립 성능 분석:

```bash
uv run python tools/evaluate_multichannel.py \
    --evidence-config configs/evidence/multi.json \
    --query-pairs data/registry/val_pairs.json \
    --output report.json
```

산출 지표에는 verification, retrieval, calibration, open-set 지표가 포함됩니다.
OSCR은 아직 DEFERRED이며, 실제 negative trial 수가 지지하지 않는 FAR는
성능 주장에 사용하지 않습니다.

## 학술 인용

준비 중.

## 라이선스

저장소 코드 라이선스: UNVERIFIED (루트 LICENSE 미확정)
사전학습 모델은 code/weight/dataset 라이선스를 별도로 검증해야 합니다.
MiewID-msv3 code와 weight의 상업 이용·재배포 상태는 현재 UNVERIFIED입니다.
