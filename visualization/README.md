# Visualization

Observer only. Stages do not import this package. Callers import `export/`,
never `training/`. Missing activations are labeled absent; heatmaps are not
invented.

In: export traces (boxes, masks, crops, channel vectors, optional activations)
and paper figure-data bundles.

Out: gitignored `Visualization/vis/00_parsing/` … `05_search/` with inner
`00_`/`01_` substages. No title or caption is drawn on any visual; the
filename is the label. Parsing detection writes one six-row, three-column
PDF per dataset as
`00_parsing/00_detection/detection_box_<dataset>.pdf`; the same directory also
contains `detection_metrics.pdf`, whose counts aggregate all parser inputs in
the execution trace. Each detection PDF contains three detected and three
undetected inputs when available. Detected samples additionally produce
`00_parsing/01_segmentation/segmentation_<dataset>.pdf` with the three highest
and three lowest parser-quality detected samples, each showing the segment input,
mask boundary overlay, and parser background-treated RGB. That directory also
contains `segmentation_metrics.pdf` with parser quality-state counts and mean
refinement diagnostics. The active appearance vector is observed under
`01_representation/01_channels/`, which writes `embedding_heatmap.pdf`,
`pca_variance.pdf`, `pca_components.pdf`, and `pca_identity.pdf`. The heatmap
keeps every embedding dimension in its original 0-based order, labels the
origin and dimensions with the highest mean absolute values, and groups rows by
parser detection status (`detected_samples` above `undetected_samples`).
The PCA component plate keeps every original dimension while labeling the largest
PC loadings, and the PCA sample plate uses the same parser-status groups. Identification
has no observer plate until additional per-channel outputs are available.
An optimization protocol trace dumps `optimization.json` next to each
`00_parsing` … `05_search` stage; runtime rows stay on the protocol document.
Paper `FIGURE_REGISTRY` 00–17 is a different sequence and writes
`Visualization/paper/` when requested.

The parsing plates are owned by `visualization/parsing/detection.py` and
`segmentation.py`; channel plates are owned by
`visualization/representation/channels.py`. Trace metadata carries the active
dataset labels and active parser/backbone identifiers, so changing a run does not
require changing the renderer. A comparable-transfer run passes `--clean` to
the first stage render, removing the previous `vis/` tree before writing the
new trace set. Direct stage rendering can use the same option:
`uv run python -m visualization.commands.render --help`.
