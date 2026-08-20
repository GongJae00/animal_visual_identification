# Full128 B0–B5

Appearance 128-D family. B0/B1/B2 trained; B3 held as stable candidate;
B4/B5 (channel / spatial) compared on a governed K1/K3/K5 panel.

Promotion required a strict positive paired 95% lower bound on **all three
seeds**. DEV point selection picked B5-SPATIAL every seed. The interval gate
failed.

| Seed | DEV pick | B5-SPATIAL Rank-1 | vs B3 | Paired 95% lower | Gate |
|---|---|---|---|---|---|
| 0 | B5-SPATIAL | 77.92% | +1.30%p | −2.60%p | fail |
| 1 | B5-SPATIAL | 77.92% | +1.30%p | −2.60%p | fail |
| 2 | B5-SPATIAL | 81.82% | +2.60%p | 0.00%p | fail |

**Decision: NO_GO.** B3 stays the research candidate. Not connected to
`IdentityEngine`.

Training / scoring code for the live family remains under
`archive/full128/learning/` and `archive/full128/evaluation/`.
This folder holds the comparison CLIs.

```bash
uv run python archive/full128/commands/build_full128_experiment_inventory.py --help
uv run python archive/full128/commands/build_full128_route_plan.py --help
uv run python archive/full128/commands/decide_full128_successor_multiseed.py --help
uv run python archive/full128/commands/evaluate_full128_family.py --help
uv run python archive/full128/commands/evaluate_full128_successors.py --help
uv run python archive/full128/commands/generate_full128_representation_traces.py --help
uv run python archive/full128/commands/materialize_full128_route_plan.py --help
uv run python archive/full128/commands/render_full128_visual_audit.py --help
uv run python archive/full128/commands/run_full128_successors.py --help
uv run python archive/full128/commands/run_full128_training.py --help
```
