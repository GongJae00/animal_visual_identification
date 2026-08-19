# Appearance / Face / Nose fusion

Closed-set fusion of frozen Appearance (A0), Face (F5), and weak Nose (N3)
on identity-disjoint panels. Same-track unless noted.

| Setting | Compared | Result | Decision |
|---|---|---|---|
| EVAL 105 IDs | A0 vs DEV-fit A+F+N (0.75/0.25/0.00) | A0 Rank-1 **94.29%**. Fusion did not beat Rank-1. MRR −0.53%p, CI includes 0. | Keep A0 |
| EVAL 188 IDs, fixed panel | A0 / F5 / N3 / A0+F5 / A0+F5+N3 | Rank-1 A0 **95.21%**, F5 89.36%, N3 68.62%. Fusions did not beat A0. | Keep A0 |
| A4 residual | A0 vs A4 | SiBeTan up; DogFace K1 down | No promotion |

```bash
uv run python legacy/version/afn/workflows/audit_sibetan_diagnostics.py --help
uv run python legacy/version/afn/workflows/build_fixed_multievidence_panel.py --help
uv run python legacy/version/afn/workflows/evaluate_fixed_multievidence.py --help
uv run python legacy/version/afn/workflows/evaluate_sibetan_multievidence.py --help
uv run python legacy/version/afn/workflows/evaluate_yt_unified_multievidence.py --help
```
