# Ablation Results

Receipt-bound research diagnostics only. Not cross-session biometric validation,
open-set assurance, or deployment performance. Frozen functional path:
parser policy v6 + Appearance A0 + cosine retrieval. Ablation sets live in [archive/](../archive/README.md)
(`full128`, `appearance_face_nose`, `nose_metric`, `nose`, `face`).

## Comparable transfer protocol

Frozen before measuring. Train: `yt-bb-dog` official train IDs. Eval: Sibetan
held-out, identity-disjoint. Crops: parser policy v6, byte-bound when scored.
Gallery and query lists are hashed. Backbone is the only comparison variable.
Metrics: Rank-1 / Rank-5 / mAP. Split seed `0`. CLI:
`evaluation.commands.evaluate comparable-transfer`. Not a BIFOR sequence-mean
claim and not biometric validation.

## Current path

| Stage | Frozen choice | Why |
|---|---|---|
| Parsing | dog-only policy v6, route-plan v3, batch 4 | v5 cat+dog and extra-candidate rejects dropped usable single-dog crops. Batch 8/16 changed semantics. |
| Identification | Appearance A0 (DINOv2-small, receipt-bound) | Strongest closed-set diagnostic. Face/Nose fusion did not beat A0. |
| 등록 / 검색 | Gallery K/V, available-intersection weighted cosine | Not attention. Identity = max template. |
| Evaluation | identity-disjoint protocols | Random frame splits are forbidden. |

## Appearance / Face / Nose

| Setting | Compared | Result | Decision |
|---|---|---|---|
| Same-track A/F/N, EVAL 105 IDs | A0 vs DEV-fit A+F+N (0.75/0.25/0.00) | A0 Rank-1 **94.29%**. A+F+N did not improve Rank-1. MRR −0.53%p, CI includes 0. | Keep A0 |
| Face publisher test, train IDs held out | F0 vs F5 | Rank-1 37.96% → **40.39%**. ID-bootstrap +2.00 to +2.88%p. | F5 research candidate. A+F fusion still ungated. |
| Appearance residual A4 | A0 vs A4 | SiBeTan improved; DogFace K1 regressed. | No promotion. Keep A0 |
| Nose raw vs raw+mask+restoration, same-track | K5 raw vs fused | Rank-1 +1.24%p. 95% CI −1.86 to +4.97%p includes 0. | No promotion |
| Nose/muzzle availability, same frozen localizer | box crop vs source-coordinate head crop | Explicit availability 6/1,755 → 901/1,755 | Coverage only. Not identity gain. |
| Fixed panel A0 / F5 / N3, EVAL 188 IDs | singles vs A0+F5 vs A0+F5+N3 | Rank-1 A0 **95.21%**, F5 89.36%, N3 68.62%. Fusions did not beat A0. | Keep A0 |

## Full128 successors

Governed K1/K3/K5 panel. Promotion required a strict positive paired lower bound on all three seeds.

| Seed | DEV pick | B5-SPATIAL Rank-1 | vs B3 | Paired 95% lower | Gate |
|---|---|---|---|---|---|
| 0 | B5-SPATIAL | 77.92% | +1.30%p | −2.60%p | fail |
| 1 | B5-SPATIAL | 77.92% | +1.30%p | −2.60%p | fail |
| 2 | B5-SPATIAL | 81.82% | +2.60%p | 0.00%p | fail |

Terminal decision **NO_GO**. B3 remains the research candidate. Not connected to `IdentityEngine`.

## N4 residual adapter

| Setting | Compared | Result | Decision |
|---|---|---|---|
| Publisher EVAL 188 IDs, same-track | frozen N3 vs N4 | Rank-1 68.62% → **76.60%**. CI +3.19 to +13.30%p. MRR +6.16%p. Rank-5 CI includes 0. | Same-track only |
| SiBeTan K1/K3/K5, N3→N4 swap | A0+F0+N3 (w_N=0.05), 128 queries | Rank-1 change 0 / 0 / 0. K1 Rank-5 −0.78%p | No domain-general gain. Not promoted. |

N3/N4 are weak muzzle appearance vectors, not nose-print topology.

## Parser v6 materialization

| Item | Value |
|---|---|
| Policy | dog-only v6 + per-dataset route v3 |
| Observations | 49,253 canonical; 36,195 crops; 13,058 terminal |
| vs v5 | 518 terminal→crop recoveries; 0 reverse; 35,622/35,677 crops byte-identical |
| Oxford dog | 2,481 images, 2,440 detected; refined IoU 0.9706 / 0.9666 |
| Batch | 4 kept. 8/16 faster but not prediction-identical |

Not an identity metric.

## Evidence

Git-external `report_sha256` values. Re-run CLIs live under the matching set.

| Report | digest | Set / CLI |
|---|---|---|
| fixed A0/F5/N3 eval | `89de7113…833e1` | `archive/appearance_face_nose/commands/evaluate_fixed_multievidence.py` |
| topology audit | `c5ef81f8…695eaa` | `archive/shared_helpers/commands/audit_identity_topology.py` |
| N4 publisher eval | `a0cb0880…7e7041a` | `archive/nose_metric/commands/evaluate_n4_metric_adapter.py` |
| N4 SiBeTan swap | `59cd6c75…25fea6c1` | `archive/appearance_face_nose/commands/evaluate_sibetan_multievidence.py` |
| Full128 public report | `c09c622c…4de4d2b` | `archive/full128/commands/evaluate_full128_successors.py` |
| three-seed decision | `2eb46ff4…d8ad63a` | `archive/full128/commands/decide_full128_successor_multiseed.py` |
| representation traces | `b08eabe7…a683a51` | `archive/full128/commands/generate_full128_representation_traces.py` |

Still open: cross-session cohort, target association, manual Nose subset, independent unknown-dog open-set. Gates: [Roadmap](ROADMAP.md). Software scope: [Architecture](ARCHITECTURE.md).
