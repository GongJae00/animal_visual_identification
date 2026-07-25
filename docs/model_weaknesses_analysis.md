# Backbone 모델 알고리즘 취약점 분석 — Dog Identity Verification 관점

> 작성일: 2026-07-24
> 범위: CVI 프로젝트 G0/G1 단계. 각 채널의 실제 알고리즘적 결함을 구체적으로 지적.
> 일반론적 비판이 아닌, 논문/코드/리더보드/데이터셋 메타데이터에서 확인된 취약점만 기록.

---

## 1. DogFaceNet (Visual Face Recognition)

**구조**: FaceNet-style ResNet backbone + Triplet Loss, L2-normalized embedding → cosine similarity.
**데이터**: DogFaceNet_l