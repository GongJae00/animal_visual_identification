# CVI 프로젝트 가이드 (AI 에이전트 / 신규 개발자용)

## 프로젝트 개요

**Canine Video Identity** — 개체(개) 식별 시스템.
비문(코주름) + 얼굴 랜드마크 + 외형 특징을 조합하여 영상 속 개별 개를 식별함.

## 디렉토리 구조 (한눈에)

```
canine_video_identity/
├── pyproject.toml          # 설치: uv sync --extra cuda --extra training
├── README.md               # 프로젝트 개요 (한글)
├── AGENTS.md               # ← 이 파일
├── .gitignore
│
├── src/cvi/                # ← 핵심 패키지
│   ├── backbones/          #   백본 모델 (DINOv2, ConvNeXt, TinyViT)
│   ├── heads/              #   학습 헤드 (ArcFace, MagFace, Evidential)
│   ├── evidence/           #   증거 추출 (비문, 랜드마크, 외형, 화질)
│   ├── classifier/         #   품종 분류 (species→breed→color)
│   ├── fusion/             #   점수 융합+보정+OpenSet+Temporal
│   ├── index/              #   FAISS 인덱스 (계층형 품종 필터)
│   ├── pipeline/           #   등록/검색 파이프라인 통합
│   ├── deployment/         #   CUDA/CPU 배포 경로
│   ├── utils/              #   공통 유틸 (모델 경로, 메트릭)
│   ├── train/              #   학습 인프라 (설정, 데이터셋, 증강)
│   │
│   └── (75개 평면 모듈)    #   연구 인프라 — 변경 시 주의
│       ├── trainer.py      #     ArcFaceModel + 학습 루프
│       ├── detection.py    #     YOLO 개 검출
│       ├── gpu_index.py    #     FAISS GPU 인덱스
│       ├── identity_index.py
│       ├── search_engine.py
│       ├── post_search.py
│       ├── evidence_extractor.py
│       └── ... (계약, 검증, 평가, 데이터셋 등)
│
├── models/                 # ⚠️ Git 미포함 (생성 시 바이너리)
│   ├── backbones/          #   학습된 .pt 체크포인트
│   ├── onnx/               #   배포용 .onnx
│   ├── pretrained/         #   사전학습 + 다운로드 모델
│   └── checkpoints/        #   학습 중간 저장
│
├── data/                   # ⚠️ Git 미포함 (데이터 준비 가이드)
│   ├── processed/          #   전처리 크롭 (생성 필요)
│   ├── registry/           #   등록 정보 (생성 필요)
│   └── README.md           #   데이터 연결 가이드
│
├── tools/                  # CLI 도구 (64개)
│   ├── download_datasets.py    # 데이터셋 다운로드
│   ├── download_models.py      # 추론 모델 다운로드
│   ├── train_embedding_model.py # 백본 학습
│   ├── evaluate_multichannel.py # 평가
│   ├── export_onnx.py         # ONNX 변환
│   └── ...
│
├── scripts/
│   └── check_env.sh        #   ✅ 환경 전수 점검
│
├── tests/                  # 단위 테스트 (564개)
├── configs/                # 설정 파일 (6개 서브디렉토리)
├── docs/                   # 기술 문서
│   └── ARCHITECTURE.md     #   상세 아키텍처
│
└── ~/cvi_data/             # 외부 저장소 (심링크)
    ├── datasets/           #   원본 데이터셋 (4종)
    ├── checkpoints/        #   훈련된 ONNX 모델
    ├── cache/registries/   #   개체 등록 DB
    ├── receipts/           #   검증 영수증
    └── experiments/        #   평가 결과
```

## 불변 조건 (절대 깨면 안 됨)

1. `models/`, `data/` 디렉토리 내용은 Git에 포함하지 않음
2. 개체 ID(`registered_dog_id`), 시각 엔티티(`track`), 등록 ID(`identity`) 네임스페이스 분리
3. 미래 정보(future evidence)를 결정 시점 이전에 사용하지 않음
4. 임의 프레임 분할(random split) 사용하지 않음
5. CUDA/TensorRT 경로는 guarded optional — portable CPU fallback 필수
6. `research-implementation-check .` 통과해야 commit 가능

## 사용자/개발자 경계

```
┌─ 사용자 (배포/운영) ─────────────────────────────┐
│  from cvi import CVI, Match                       │
│                                                   │
│  cvi = CVI("configs/deployment/production.json")  │
│  cvi.enroll(image, "뽀삐", breed="비글")           │
│  results = cvi.search(image, top_k=5)             │
│  cvi.save()                                       │
└───────────────────────────────────────────────────┘

┌─ 연구자 (학습/평가) ─────────────────────────────┐
│  from cvi.training import TrainConfig             │
│  from cvi.backbones import get_backbone            │
│  from cvi.evidence import Dinov2WithUncertainty   │
│  from cvi.evaluation import evaluate_multichannel  │
└───────────────────────────────────────────────────┘

┌─ 내부 (계약/검증/데이터셋) ──────────────────────┐
│  from cvi.trainer import ArcFaceModel             │
│  from cvi.detection import DogDetector            │
│  from cvi.identity_index import IdentityIndex      │
│  ... (74개 평면 모듈, cvi/__init__.py로 re-export) │
└───────────────────────────────────────────────────┘
```

## 설치 및 실행

```bash
# 1. 환경 확인 (Python, GPU, 데이터, 모델 전수 점검)
uv run bash scripts/check_env.sh

# 2. 패키지 설치
uv sync --extra cuda --extra training

# 3. 데이터 연결 (예: R:\research-data\canine_video_identity_secure)
export CVI_DATA_DIR=/your/data/path
# 또는
ln -s /your/data/path ~/cvi_data
# 연결 확인
uv run bash scripts/check_env.sh

# 4. 데이터셋 + 모델 다운로드
uv run python tools/download_datasets.py
uv run python tools/download_models.py

# 5. 학습
uv run python tools/train_embedding_model.py --backbone dinov2-small \
    --crop-root ~/cvi_data/processed/oracle_crops/

# 6. 평가
uv run python tools/evaluate_multichannel.py \
    --evidence-config configs/evidence/multi.json \
    --query-pairs data/registry/val_pairs.json \
    --output report.json

# 7. 테스트
uv run python -m unittest discover -s tests -v
```

## 코드 변경 시 체크리스트

```bash
# 변경 전
research-implementation-check --pre .

# 변경 후
uv run python -m unittest discover -s tests -v
research-implementation-check .
```

## 주의사항

### `src/cvi/` 평면 모듈 (74개)

`trainer.py`, `detection.py`, `identity_index.py` 등은 기존 연구 인프라.
이 파일들은 다음의 이유로 평면 구조 유지:
- 수년간 쌓인 연구 계약/검증/평가 코드
- `from cvi.trainer import TrainConfig` 형태로 레거시 코드에서 직접 참조
- 새 기능은 하위 패키지(`evidence/`, `fusion/`, `heads/`, `backbones/`)로 추가

**절대**: 평면 모듈을 무작정 이동/삭제하지 말 것. `__init__.py`에서 re-export 필요.

### 새 모듈 추가 가이드

| 기능 | 추가 위치 |
|------|----------|
| 새 백본 (ViT, ResNet 등) | `src/cvi/backbones/` |
| 새 손실함수 (CosFace, AdaFace) | `src/cvi/heads/` |
| 새 증거 채널 (홍채, 귀 형태) | `src/cvi/evidence/` |
| 새 융합 전략 | `src/cvi/fusion/` |
| 새 배포 타겟 (Jetson, MPS) | `src/cvi/deployment/` |
| 연구 인프라 (계약, 검증) | `src/cvi/` 평면 |
