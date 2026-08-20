# Evaluation

In: identity-disjoint splits, pair lists, embeddings, protected receipts.

Out: verification, search metrics, calibration, robustness, localization, and
protected-evaluation reports. Algorithm packages do not import this package.

`splits/` owns identity-disjoint partitions and exposure. `search_metrics/`
is closed-set cosine retrieval. `verification/` owns pair curves and operating
thresholds. `commands/` is the thin CLI.

Commands: `uv run python -m evaluation.commands.evaluate --help`
