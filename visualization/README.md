# Visualization

Observer only. Stages do not import this package. Callers import `export/`,
never `training/`. Missing activations are labeled absent; heatmaps are not
invented.

In: export traces (boxes, masks, crops, channel vectors, optional activations)
and paper figure-data bundles.

Out: gitignored `Visualization/vis/00_parsing/` … `05_search/` with inner
`00_`/`01_` substages. No title or caption is drawn on any raster; the
filename is the label. Identification and representation write PCA 2D/3D
scatters, same/different cosine, PCA cumulative variance, per-dimension
contribution, and per-channel cosine gap from embedding traces.
Parsing writes JSON (and a source image if supplied), not catalog plates.
An optimization protocol trace dumps `optimization.json` next to each
`00_parsing` … `05_search` stage; runtime rows stay on the protocol document.
Rasters are PNG only. Paper `FIGURE_REGISTRY` 00–17 is a different sequence
and writes `Visualization/paper/` when requested.

Commands: `uv run python -m visualization.commands.render --help`
