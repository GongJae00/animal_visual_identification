# 데이터 가이드

Git 저장소에는 데이터가 포함되지 않습니다. 아래 절차에 따라 외부 저장소의 데이터를 연결하거나 직접 다운로드합니다.

## 빠른 시작 (1분)

```bash
# 1. 환경 점검
uv run bash scripts/check_env.sh

# 2. 데이터 저장소 연결
export CVI_DATA_DIR=/your/data/path   # 실제 데이터가 있는 경로
# 또는
ln -s /your/data/path ~/cvi_data

# 3. 공개 데이터셋 + 모델 다운로드
uv run python tools/download_datasets.py
uv run python tools/download_models.py

# 4. 재확인
uv run bash scripts/check_env.sh
```

## 디렉토리 구조 (SSD 저장소 기준)

```
~/cvi_data  (또는 $CVI_DATA_DIR)
├── datasets/                        # 원본 데이터셋
│   ├── dogfacenet-224-zenodo-12578449-v1/   8,363장, 1,393마리
│   │   └── after_4_bis/{dog_id}/{img}.jpg
│   ├── mpdd-mendeley-v5j6m8dzhv-v1/        1,657장
│   │   └── MPDD/pytorch/{train,val,gallery,query}/*.jpg
│   ├── sibetan-official-2026-07-22/         1,755장, 224마리
│   │   └── Sibetan/{dog_id}/{img}.jpg
│   └── yt-bb-dog-outer-official-... /    ZIP (압축해제 필요)
│       └── YT-BB-dog/*.zip
│
├── checkpoints/                     # 훈련된 모델
│   └── deployment-eligible/onnx-models/
│       ├── dinov2-small.onnx               (1 MB + 86 MB data)
│       └── mobilenetv4-conv-small.onnx     (173 KB + 4.8 MB data)
│
├── cache/registries/                # 등록/매핑 정보
│   ├── identity_registry.db              1,393마리 등록
│   └── binding.json                      샘플↔개체 매핑
│
├── downloads/                       # 원본 ZIP 아카이브
├── experiments/                     # 평가 결과
├── receipts/                        # 검증 영수증
└── manifests/                       # 라이선스 명세
```

## 데이터 위치 설정 (3가지 방법)

```bash
# 방법 1: 환경변수 (권장)
export CVI_DATA_DIR=/mnt/ssd/canine_data

# 방법 2: 심볼릭 링크
ln -s /mnt/ssd/canine_data ~/cvi_data

# 방법 3: 영구 설정 (~/.zshrc)
echo 'export CVI_DATA_DIR=/mnt/ssd/canine_data' >> ~/.zshrc
```

우선순위: `CVI_DATA_DIR` 환경변수 > `~/cvi_data` 심링크

## 지원 데이터셋

| 이름 | 출처 | 이미지 | 개체 | 인증 | 상태 |
|------|------|--------|------|------|------|
| DogFaceNet | `dimidagd/DogFaceNet_224resize` (HF) | 8,363 | 1,393 | 공개 | ✅ |
| MPDD | Mendeley Data v5j6m8dzhv | 1,657 | - | 공개 | ✅ |
| SiBeTan | 직접 수집 (2026-07-22) | 1,755 | 224 | 제한 | ✅ |
| YT-BB-dog | Google Research | ~12,000 | ~25품종 | CC BY 4.0 | ⚠️ ZIP 압축해제 필요 |

## 다운로드 명령

```bash
# 전체 데이터셋 확인
uv run bash scripts/check_env.sh

# DogFaceNet 다운로드
uv run python tools/download_datasets.py --dataset dogfacenet

# YT-BB-dog 압축 해제
cd $(python -c "from cvi.model_paths import dataset_path; print(dataset_path('yt-bb-dog'))")/YT-BB-dog
unzip -q "YT-BB-Dog*.zip"
```

## 훈련 데이터 준비 절차

```bash
# 1. 개 검출 및 크롭 생성
uv run python tools/export_oracle_crops.py \
    --video-dir ~/cvi_data/datasets/ \
    --output ~/cvi_data/processed/oracle_crops/

# 2. 개체 등록
uv run python tools/build_identity_registry.py \
    --source ~/cvi_data/processed/oracle_crops/ \
    --output ~/cvi_data/processed/identities.json

# 3. 학습/평가 분할 생성
uv run python tools/bind_split_to_registry.py \
    --crop-root ~/cvi_data/processed/oracle_crops/ \
    --output ~/cvi_data/processed/split_binding.json

# 4. 백본 학습
uv run python tools/train_embedding_model.py \
    --backbone dinov2-small \
    --crop-root ~/cvi_data/processed/oracle_crops/ \
    --output-dir models/backbones/
```

## Python에서 경로 조회

```python
from cvi.model_paths import dataset_path, DATA_DIR, DATASETS_DIR

print(f"데이터 루트: {DATA_DIR}")
print(f"데이터셋 루트: {DATASETS_DIR}")

# 개별 데이터셋
path = dataset_path("dogfacenet")
# → ~/cvi_data/datasets/dogfacenet-224-zenodo-12578449-v1/
```

## 주의사항

- `data/` 디렉토리는 `.gitignore`에 의해 Git에서 제외됩니다
- 모든 경로는 `CVI_DATA_DIR` 환경변수 또는 `~/cvi_data` 심링크로 커스터마이즈 가능
- `uv run bash scripts/check_env.sh` 로 현재 연결 상태를 항상 확인 가능
- 사전학습 백본 가중치는 첫 실행 시 torch.hub에서 자동 다운로드됨
