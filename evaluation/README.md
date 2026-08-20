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
uv run python -m evaluation.commands.evaluate comparable-transfer --help
```

`admit-models` writes capability-owned parser/DINOv2 bundles from local
checkpoints. `smoke` is synthetic. `run` materializes parser v6 crops (source
RGB if no single dog), embeds on CUDA, and scores the frozen gallery/query.

Comparable transfer (BIFOR-adjacent, not a BIFOR sequence-mean claim): train on official `yt-bb-dog` train IDs, evaluate identity-disjoint Sibetan with a seed-0 gallery/query freeze, parser policy v6 crops, Rank-1 / Rank-5 / mAP. Backbone is the only comparison variable.
