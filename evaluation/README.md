# Evaluation

In: identity-disjoint splits, pair lists, embeddings, protected receipts.

Out: verification, search metrics, calibration, robustness, localization, and
protected-evaluation reports. Algorithm packages do not import this package.

`splits/` owns identity-disjoint partitions and exposure. `search_metrics/`
is closed-set cosine retrieval. `verification/` owns pair curves and operating
thresholds. `parsing_protocol` catalogs parsing stage × data × metric ×
extraction (no measured values). `commands/` is the thin CLI.

Commands: `uv run python -m evaluation.commands.evaluate --help`

Parsing catalog (JSON, no measured values, not a figure):

```bash
uv run python -m evaluation.commands.evaluate parsing-protocol --output /tmp/parsing_protocol.json
uv run python -m evaluation.commands.evaluate optimization-protocol --output /tmp/optimization_protocol.json
```
