# 설정 파일 가이드

기능별로 그룹화된 설정 파일입니다. 각 하위 디렉토리는 다음을 담당합니다:

```
configs/
├── models/             # 모델 학습/추론 설정
│   ├── backbones/      #   백본 아키텍처 설정
│   └── heads/          #   손실함수 파라미터
│
├── pipeline/           # 파이프라인 설정
│   ├── evidence/       #   증거 채널 구성 + 화질 임계값
│   ├── fusion/         #   융합 가중치 + Open-Set 문턱값
│   └── detection/      #   YOLO 검출 파라미터
│
├── deployment/         # 배포 환경별 설정
│   ├── onnx_cpu_backend.example.json
│   ├── onnx_cuda_backend.example.json
│   ├── embedding_production_policy.example.json
│   └── ...
│
├── research/           # 연구 인프라 설정
│   ├── contracts/      #   계약/정책 (pairing, split, scoring)
│   ├── benchmarks/     #   벤치마크 (batch invariance, embedding cache)
│   └── processing/     #   처리 (control transform, capacity)
│
├── data/               # 데이터 처리 설정
│   └── crop_export_policy.example.json
│
├── pdq/                # PDQ 해시 설정
└── pretrained-weights/ # 사전학습 가중치 intake 설정
```

## 사용법

```bash
# 증거 채널 설정 예시
cp configs/pipeline/evidence/evidence_coverage.example.json configs/pipeline/evidence/mine.json
# → "channels" 항목에 원하는 증거 채널 추가

# 학습 설정 예시
# → configs/models/backbones/ 에 백본별 JSON 추가

# 배포 설정
cp configs/deployment/onnx_cuda_backend.example.json configs/deployment/production.json
```
