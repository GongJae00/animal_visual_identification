# Ablation archive

Completed comparison sets. Not the public runtime. Numbers are
research-only diagnostics. Index: [docs/RESEARCH_PROGRESS.md](../../docs/RESEARCH_PROGRESS.md).

| Set | Compared | Decision | Doc |
|---|---|---|---|
| [full128/](full128/README.md) | B0–B5 successors vs B3 | NO_GO. Keep B3 | one page |
| [afn/](afn/README.md) | A0 vs Face vs Nose vs fusions | Keep A0 | one page |
| [n4/](n4/README.md) | N3 vs N4 residual adapter | Not promoted | one page |
| [nose/](nose/README.md) | architecture, texture, restoration, fusion | No promotion | one page |
| [face/](face/README.md) | F0 vs F5, ROI face ReID | F5 candidate only | one page |

`common/` is shared helpers and example configs, not an ablation.

```bash
uv run python legacy/version/<set>/workflows/<command>.py --help
```
