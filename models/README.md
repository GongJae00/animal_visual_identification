# 모델 가이드

이 디렉토리는 학습된 모델 가중치, ONNX 변환본, 사전학습 백본을 보관합니다.
가중치는 Git에 포함되지 않으며, 아래 절차로 별도 획득합니다.

## 디렉토리 구조

```
models/
├── backbones/          # 학습 완료된 PyTorch 체크포인트 (.pt)
│   ├── dinov2_small_arcface.pt
│   ├── convnext_base_arcface.pt
│   └── tinyvit_nose_magface.pt
│
├── onnx/               # ONNX 변환본 (추론 배포용)
│   ├── dinov2_visual.onnx        #   외형 채널 (384-d)
│   ├── convnext_texture.onnx     #   질감 채널 (768-d)
│   ├── tinyvit_nose.onnx         #   비문 채널 (512-d)
│   └── miewid.onnx               #   MiewID 코 프린트 (2152-d)
│
├── pretrained/         # 사전학습 모델 (HuggingFace 등 다운로드)
│   ├── dogflw_landmark.tflite    #   DogFLW 랜드마크 검출기
│   └── superanimal_quadruped.pth #   SuperAnimal 사족동물 검출기
│
└── checkpoints/        # 학습 중간 저장본
```

## 사전학습 모델 다운로드

```bash
# 사용 가능한 모델 확인
python tools/download_models.py --list

# 개별 다운로드
python tools/download_models.py --model superanimal    # ✅ 공개 (다운로드 완료)
python tools/download_models.py --model miewid         # ⚠️ HF 토큰 필요
python tools/download_models.py --model dogflw-landmark # ❌ 레포 미생성
```

### MiewID 토큰 발급 방법

MiewID는 HuggingFace gated repo라 인증이 필요합니다:

1. https://huggingface.co/james-burgess/miewid 접속 → "Access repository" → 라이선스 동의
2. https://huggingface.co/settings/tokens → "Create new token" → Fine-grained, `james-burgess/miewid` 선택, Read 권한
3. 환경변수 설정: `export HF_TOKEN=hf_xxxxxxxxxxxxxxxx`
4. 다운로드: `python tools/download_models.py --model miewid`

## 백본 사전학습 가중치

학습에 사용되는 백본은 자동으로 HuggingFace에서 다운로드됩니다:

| 백본 | 소스 | 파라미터 | 상태 |
|------|------|---------|------|
| DINOv2-Small | `facebookresearch/dinov2` (torch.hub) | 22.1M | ✅ 캐시 완료 |
| ConvNeXt-Base | `facebook/convnext-base-224` (transformers) | 87.6M | ✅ 캐시 완료 |
| TinyViT | 커스텀 CNN (사전학습 불필요) | ~0.5M | 학습 시 생성 |

## 모델 명세

| 백본 | 용도 | 입력 | 출력 | 크기 |
|------|------|------|------|------|
| DINOv2-Small | 외형 | 224×224 | 384-d, L2=1 | 84 MB (ONNX) |
| ConvNeXt-Base | 질감 | 224×224 | 768-d, L2=1 | 340 MB (ONNX) |
| TinyViT | 비문 | 224×224 | 512-d, L2=1 | 24 MB (ONNX) |
| MiewID | 코 프린트 | 160×160 | 2152-d, L2=1 | 28 MB (ONNX) |

## 배포 정책

- **오픈소스**: 학습 코드 + 모델 구조 + ONNX 변환 스크립트 공개
- **가중치**: HuggingFace에 선택적 배포 (라이선스 확인 후)
- **임베딩 벡터**: 개별 등록된 개체의 임베딩은 공개하지 않음
