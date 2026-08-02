# 연구 진행 요약

이 문서는 기존 receipt-bound report와 저장소 문서에 기록된 trend만 짧게 정리합니다. 모든 수치는 research-only diagnostic이며 cross-session biometric validation, open-set 보증, 또는 deployment 성능이 아닙니다.

## 확인된 Trend

- Same-track unified A/F/N diagnostic의 identity-bound EVAL 105개체에서 frozen Appearance A0가 Rank-1 94.29%로 가장 강했습니다. DEV에서 선택한 A/F/N weight는 0.75/0.25/0.00이었고, A+F+N은 A0 Rank-1을 개선하지 못했습니다. MRR 변화도 -0.53%p였으며 bootstrap interval이 0을 포함했습니다.
- Face F5는 training identity와 분리된 publisher test에서 frozen F0보다 높은 Rank-1을 기록했습니다. 문서화된 비교는 37.96%에서 40.39%이며 identity-bootstrap 95% interval은 +2.00%p에서 +2.88%p입니다. A+F complementary value는 별도 calibration gate가 남아 있습니다.
- Appearance A4 residual candidate는 SiBeTan diagnostic에서 개선 trend를 보였지만 DogFace K1 regression이 관찰되어 multi-domain promotion gate를 통과하지 못했습니다. Frozen A0가 reference로 유지됩니다.
- Nose raw+mask+restoration score fusion은 same-track diagnostic에서 raw Nose K5 대비 Rank-1 +1.24%p였지만 95% interval이 -1.86%p에서 +4.97%p로 0을 포함했습니다.
- Source-coordinate head crop으로 동일 frozen localizer를 실행한 실험은 SiBeTan의 explicit Nose/muzzle availability를 6/1,755에서 901/1,755로 늘렸습니다. 이는 identity 성능 향상이나 fine Nose-print visibility를 의미하지 않습니다.

## 남은 핵심 검증

Cross-session/camera cohort, target association, manual Nose-region annotation subset, fixed-panel A/F/N complementary value, independent unknown-dog open-set evaluation이 아직 필요합니다. 상세 admission 조건은 [Roadmap](ROADMAP.md), 현재 software 범위는 [Architecture](ARCHITECTURE.md)를 따릅니다.

## 구조 Renewal 검증

2026-08-03 구조 renewal 후 동일한 11,009개 YT FIT crop과 receipt-bound A4 설정으로 15 epoch를 재실행했습니다. 모든 epoch loss와 DEV metric, epoch 10 선택 결과가 renewal 전 실행과 일치했습니다. 외부 DogFaceNet/MPDD/SiBeTan 8개 protocol 보고서도 checkpoint hash를 제외한 모든 값이 동일했습니다. 총 실행 시간은 동일 장비에서 1,448.9초에서 1,144.4초로 줄었지만, 반복 측정이 아닌 단일 paired run이므로 일반적인 성능 향상 수치로 해석하지 않습니다.
