# N4 residual adapter

Residual metric adapter (max scale 0.1) on frozen N3 vectors. N3/N4 are
weak muzzle appearance embeddings, not nose-print topology.

| Setting | Compared | Result | Decision |
|---|---|---|---|
| Publisher EVAL 188 IDs, same-track | N3 vs N4 | Rank-1 68.62% → **76.60%**. CI +3.19 to +13.30%p. Rank-5 CI includes 0. | Same-track only |
| SiBeTan K1/K3/K5, N3→N4 swap | A0+F0+N3 (w_N=0.05), 128 queries | Rank-1 Δ 0 / 0 / 0. K1 Rank-5 −0.78%p | No domain-general gain |

**Decision: not promoted.**

```bash
uv run python archive/nose_metric/commands/evaluate_n4_metric_adapter.py --help
uv run python archive/nose_metric/commands/evaluate_n4_robust_nose.py --help
uv run python archive/nose_metric/commands/materialize_n4_embedding_cache.py --help
uv run python archive/nose_metric/commands/train_n4_metric_adapter.py --help
```
