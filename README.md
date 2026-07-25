# Canine Video Identity (CVI)

다중 증거(multi-evidence) 기반 개체 식별 시스템.
코 프린트(비문) + 얼굴 랜드마크 + 외형 특징을 조합하여 개별 개를 식별합니다.

## 개요

```
입력 영상 → [개 검출] → [3채널 특징 추출] → [Fusion] → [FAISS 검색] → 결과
                  │                  │                │            │
              YOLO                비문 2152d     concatenate   cosine sim
                                  랜드마크 256d   L2 정규화    + 증거분해
                                  외형 384d
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

### 모델 다운로드

```bash
python tools/download_models.py --model miewid --hf-token $HF_TOKEN
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
| `src/cvi/backbones/` | 백본 모델 (DINOv2, ConvNeXt, TinyViT) |
| `src/cvi/heads/` | 학습 헤드 (ArcFace, MagFace, Evidential) |
| `src/cvi/evidence/` | 증거 추출기 (비문, 랜드마크, 외형) |
| `src/cvi/fusion/` | 점수 융합 + 보정 + Open-Set 판정 |
| `src/cvi/index/` | FAISS 검색 인덱스 |
| `src/cvi/pipeline/` | 등록/검색 파이프라인 |
| `src/cvi/deployment/` | 배포 (CUDA/CPU 분기) |
| `tools/` | 학습/평가/다운로드 CLI 도구 |
| `tests/` | 단위 테스트 (77개) |
| `models/` | 모델 가중치 (Git 미포함) |
| `data/` | 데이터셋 (Git 미포함) |
| `configs/` | 설정 파일 |
| `docs/` | 기술 문서 |

## 성능 지표

평가 프레임워크를 통한 채널별 독립 성능 분석:

```bash
python tools/evaluate_multichannel.py \
    --evidence-config configs/evidence/multi.json \
    --query-pairs data/registry/val_pairs.json \
    --output report.json
```

산출 지표: AUC, EER, TAR@FAR, d' (분리도), mean positive/negative sim

## 학술 인용

준비 중.

## 라이선스

코드: MIT (또는 Apache 2.0, 결정 중)
사전학습 모델 가중치: 각 출처의 라이선스 따름 (MiewID, DINOv2, ConvNeXt 등)
