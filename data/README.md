# 데이터 가이드

이 디렉토리는 개체 식별 모델의 학습/평가에 필요한 모든 데이터를 보관합니다.
Git 저장소에는 데이터가 포함되지 않으며, 아래 절차에 따라 로컬에 준비합니다.

## 데이터 위치 설정

외부 저장소(예: `/mnt/r/research-data/`)에 데이터를 보관하려면 심볼릭 링크를 연결하세요:

```bash
ln -s /mnt/r/research-data ~/cvi_data
```

또는 환경변수로 직접 지정:

```bash
export CVI_DATA_DIR=/mnt/r/research-data
```

우선순위: `CVI_DATA_DIR` 환경변수 > `~/cvi_data` 심링크 > 프로젝트 내 `data/`

## 디렉토리 구조

```
data/
├── raw/              # 원본 데이터 (다운로드 그대로)
│   └── dogfacenet/   #   DogFaceNet (dog_id/이미지.jpg)
│
├── processed/        # 전처리 완료 데이터
│   ├── oracle_crops/ #   YOLO 검출 + 얼굴 정렬 완료된 개별 크롭
│   └── nose_crops/   #   코 영역 크롭
│
└── registry/         # 등록 정보
    ├── identities.json      #   개체 등록 메타데이터
    └── split_binding.json   #   학습/검증/평가 분할
```

## 다운로드 가능한 데이터셋

| 데이터셋 | 출처 | 이미지 수 | 개체 수 | 인증 |
|----------|------|----------|--------|------|
| **DogFaceNet** | HuggingFace `dimidagd/DogFaceNet_224resize` | 8,363 | 1,393 | 공개 |
| **DogFaceNet-large** | HuggingFace `dimidagd/DogFaceNet_large` | 26,000+ | 다양 | 공개 |

다운로드 명령:

```bash
# 전체 (현재 DogFaceNet만)
python tools/download_datasets.py

# 개별
python tools/download_datasets.py --dataset dogfacenet

# 상태 확인
python tools/download_datasets.py --list
```

## 학습 데이터 준비 절차

### 1. 개 검출 및 크롭 생성

```bash
# YOLO로 개 검출 → 얼굴 크롭 추출
python tools/export_oracle_crops.py \
    --video-dir data/raw/ \
    --output data/processed/oracle_crops/
```

### 2. 개체 등록

```bash
python tools/build_identity_registry.py \
    --source data/processed/oracle_crops/ \
    --output data/registry/identities.json
```

### 3. 학습/평가 분할 생성

```bash
python tools/bind_split_to_registry.py \
    --crop-root data/processed/oracle_crops/ \
    --output data/registry/split_binding.json
```

### 4. 백본 학습

```bash
python tools/train_embedding_model.py \
    --backbone dinov2-small \
    --crop-root data/processed/oracle_crops/ \
    --output-dir models/backbones/
```

## 주의사항

- `data/` 디렉토리는 `.gitignore`에 의해 Git에서 제외됩니다
- 원본 데이터는 저작권 문제로 별도 저장소에 보관합니다
- 학습용 크롭은 `models/`의 ONNX 백본으로 임베딩 추출 후 FAISS 인덱스에 등록합니다
