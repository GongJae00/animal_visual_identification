# Visualization

Observer only. Stages do not import this package. Callers import `export/`,
never `training/`. Missing activations are labeled absent; heatmaps are not
invented.

In: export traces (boxes, masks, crops, channel vectors, optional activations)
and paper figure-data bundles.

Out: gitignored `Visualization/vis/00_parsing/` … `05_search/` with inner
`00_`/`01_` substages. Paper `FIGURE_REGISTRY` 00–17 is a different sequence
and writes `Visualization/paper/` when requested.

Commands: `uv run python -m visualization.commands.render --help`
