# 연구 진행 요약

이 문서는 기존 receipt-bound report와 저장소 문서에 기록된 trend만 짧게 정리합니다. 모든 수치는 research-only diagnostic이며 cross-session biometric validation, open-set 보증, 또는 deployment 성능이 아닙니다.

## 확인된 Trend

- Same-track unified A/F/N diagnostic의 identity-bound EVAL 105개체에서 frozen Appearance A0가 Rank-1 94.29%로 가장 강했습니다. DEV에서 선택한 A/F/N weight는 0.75/0.25/0.00이었고, A+F+N은 A0 Rank-1을 개선하지 못했습니다. MRR 변화도 -0.53%p였으며 bootstrap interval이 0을 포함했습니다.
- Face F5는 training identity와 분리된 publisher test에서 frozen F0보다 높은 Rank-1을 기록했습니다. 문서화된 비교는 37.96%에서 40.39%이며 identity-bootstrap 95% interval은 +2.00%p에서 +2.88%p입니다. A+F complementary value는 별도 calibration gate가 남아 있습니다.
- Appearance A4 residual candidate는 SiBeTan diagnostic에서 개선 trend를 보였지만 DogFace K1 regression이 관찰되어 multi-domain promotion gate를 통과하지 못했습니다. Frozen A0가 reference로 유지됩니다.
- Nose raw+mask+restoration score fusion은 same-track diagnostic에서 raw Nose K5 대비 Rank-1 +1.24%p였지만 95% interval이 -1.86%p에서 +4.97%p로 0을 포함했습니다.
- Source-coordinate head crop으로 동일 frozen localizer를 실행한 실험은 SiBeTan의 explicit Nose/muzzle availability를 6/1,755에서 901/1,755로 늘렸습니다. 이는 identity 성능 향상이나 fine Nose-print visibility를 의미하지 않습니다.

## Full128 Successor 판정

Governance-v2 successor 비교는 score-bearing bytes를 사용하지 않은 identity/dependency role allocation과 고정 K1/K3/K5 panel을 사용했습니다. Seed index 0, 1, 2의 DEV Rank-1 point selection은 모두 B5-SPATIAL을 선택했지만, precommitted B5-SPATIAL minus B3 paired whole-identity 95% interval의 lower bound가 각각 -2.60%p, -2.60%p, 0.00%p였습니다. 세 seed 모두 strict positive lower-bound 조건을 충족해야 하는 promotion gate의 terminal 판정은 `NO_GO`입니다. 따라서 single-report DEV selection receipt는 point selection 기록일 뿐 scientific promotion이 아니며 B3를 안정적 research candidate로 유지합니다.

Seed별 B5-SPATIAL DEV Rank-1은 77.92%, 77.92%, 81.82%였고 B3 대비 paired 차이는 +1.30%p, +1.30%p, +2.60%p였습니다. 이 값은 동일한 governed retrospective panel의 descriptive result입니다. CAL은 reporting-only이고 exposed cohort는 retrospective diagnostic이며 independent final evaluation이 아닙니다.

실제 seed-20260811 B3와 B5-SPATIAL checkpoint를 다시 실행한 representation trace는 224x224 입력, 16x16 patch grid, 384D patch token, 128D L2 output을 artifact와 cache digest에 bind합니다. 복원된 embedding은 evaluation cache vector와 byte-for-byte 일치했고 pair score는 cache vector의 exact cosine으로 확인했습니다. B5 trace는 실행된 spatial-scorer logits와 normalized pooling weights를 포함하지만 transformer attention Q/K와 별도 patch correspondence는 제공하지 않습니다.

## Parser v6와 Full128 materialization

2026-08-13 dog-only parser policy v6과 dataset별 단일-dog route policy v3를
49,253개 canonical observation에 적용했습니다. 36,195개가 crop을 보유했고
13,058개가 terminal이었습니다. v5 대비 terminal에서 crop으로 복구된 observation은
518개였고 반대 방향 regression은 없었습니다. 기존 성공 35,677개 중 35,622개 crop은
byte-identical이었고 55개는 dog-only inference로 변경되었습니다. 이 비교는 parser policy,
runtime source closure, route-plan schema가 함께 달라지는 retrospective comparison이며 identity
성능 결과가 아닙니다.

Oxford dog 2,481장 평가에서 2,440장이 검출되었고 refined macro/micro IoU는
각각 0.9706/0.9666이었습니다. 고정 512장 batch benchmark에서는 batch 8/16이 batch 4보다
일부 빠르지만 semantic prediction과 terminal decision이 동일하지 않아 batch 4를 유지했습니다.
외부 evidence digest는 `CHANGELOG.md`의 Parser policy v6 항목에 기록합니다.

## N4 Metric Adapter 판정

Publisher test identity를 hash로 고정 분할한 DEV 75개체와 EVAL 188개체는 F5와 N3의 training/model-selection identity와 겹치지 않습니다. EVAL의 frozen A0, F5, N3 Rank-1은 각각 95.21%, 89.36%, 68.62%였고 A0+F5와 A0+F5+N3는 A0를 개선하지 못했습니다.

Frozen N3 vector에 최대 scale 0.1의 residual metric adapter만 적용한 N4 후보는 lineage DEV에서 선택되었습니다. Publisher EVAL에서 N3 대비 Rank-1은 68.62%에서 76.60%로 증가했고 identity-bootstrap 95% interval은 +3.19%p에서 +13.30%p였습니다. MRR은 +6.16%p, Rank-5는 +1.60%p였고, Rank-5 interval은 0을 포함했습니다. Rank-1 rescue/break는 20/5였습니다. Intra-identity cosine diameter 중앙값은 0.9623에서 0.8298로 감소했고 normalized prototype stability 중앙값은 0.6019에서 0.6669로 증가했습니다. 이 panel은 same-video-track earliest/latest 비교이므로 cross-session 또는 biometric validation이 아닙니다.

동일 checkpoint를 기존 SiBeTan K1/K3/K5 panel에 적용할 때 panel membership, evidence availability, quality, frozen YT fusion weight를 변경하지 않고 N3 vector만 치환했습니다. N3 단독 평가는 fixed gallery evidence가 불완전해 fail-closed로 abstain했습니다. N3 weight가 0.05인 A0+F0+N3의 128개 paired query에서 K1, K3, K5 Rank-1 변화는 모두 0이었습니다. K1 Rank-5는 -0.78%p였고 reciprocal-rank 변화도 domain-general improvement를 지지하지 않았습니다. 따라서 N4는 research candidate로 유지하며 canonical branch로 승격하지 않습니다.

현재 N3와 N4는 physical nose-ridge topology가 아니라 weak Nose/muzzle appearance embedding입니다. Anatomical alignment와 physical topology 검증은 완료된 결과가 아니라 남은 admission requirement입니다.

### Evidence artifacts

아래 canonical report digest는 Git 밖 experiment storage의 report 내부 `report_sha256`입니다. 각 report가 해당 실행의 data, model, preprocessing, split, source-closure와 적용 가능한 실행 설정을 bind합니다.

- `fixed-a0-f5-n3-evaluation-final-v4.json`: `89de7113efea9050bba114859448cb8109cdf58f6926407f9648ceb25e5833e1`, `workflows/evaluate_fixed_multievidence.py`
- `fixed-a0-f5-n3-topology-audit-final-v4.json`: `c5ef81f8d4b1601aeade7f054fbb1cb385356a02a16873e535651ea460695eaa`, `workflows/audit_identity_topology.py`
- `n4-metric-adapter-evaluation-final-v5.json`: `a0cb0880670edaf8fe5aa9ab98a526acbba6f5afe75bf8f074d6caf7e7c7041a`, `workflows/evaluate_n4_metric_adapter.py`
- `n4-sibetan-metric-adapter-evaluation-final-v4.json`: `59cd6c750d5807fac990dab58e104c69dd5710b1478467ba8635126d25fea6c1`, `workflows/evaluate_sibetan_multievidence.py`
- `full128-successor-final-evaluation-seed20260811/public-report.json`: `c09c622c00e8ba5ee889234ec94262ba3ec3364f5c0982084c6c135a34de4d2b`, `workflows/evaluate_full128_successors.py`
- `full128-successor-terminal-decision-three-seed-v2/terminal-decision.json`: decision digest `2eb46ff40839e9f927956e2923c89abacf6fbe66d6c0a3aadb43eda83d8ad63a`, `workflows/decide_full128_successor_multiseed.py`
- `representation-analysis-public-seed20260811-v2.json`: public-safe trace-summary digest `b08eabe77a00b58cf128f2cd9979b63284a5ec9e2f5188fc5d5a2f7f3a683a51`, `workflows/generate_full128_representation_traces.py` and `workflows/analyze_full128_successors.py`

## 남은 핵심 검증

Cross-session/camera cohort, target association, manual Nose-region annotation subset, multi-domain N4 transfer, fixed-panel A/F/N complementary value, independent unknown-dog open-set evaluation이 아직 필요합니다. Full128 successor에는 independent final cohort, mask/background perturbation artifact, target-device latency와 memory/power measurement가 없습니다. 상세 admission 조건은 [Roadmap](ROADMAP.md), 현재 software 범위는 [Architecture](ARCHITECTURE.md)를 따릅니다.
